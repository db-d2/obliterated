"""A reader-side edit hidden among decoys, made on disk, and run from the matrices that were made."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from obliterated import detect, record, stream
from obliterated.edits import (
    _gain_on_disk,
    blind_readers_on_disk,
    readers_from,
    temporarily_blinded,
)
from obliterated.refusal import last_logits

from .conftest import make_lm
from .test_edits import PROMPTS, READABLE, rand_dir, readers_and_head, uneven_gains
from .test_reader_positives import saved
from .test_readers import BASE_REV, pair, snapshot

GAIN = 1.5


def effective(folder, base_folder=None) -> dict[str, torch.Tensor]:
    """Every reading matrix in ``folder`` composed with its norm, in float64: what acts on the stream."""
    got = saved(folder)
    gemma = "gemma" in (folder / "config.json").read_text()
    return {name: got[name].double() * _gain_on_disk(got[norm], gemma, 1e-6).double()
            for name, norm in detect.reader_layout(detect._tensor_index(base_folder or folder))}


@pytest.fixture
def made(tokenizer, tmp_path):
    base, _ = pair("llama_untied", tokenizer)
    folder = snapshot(tmp_path / "base", base, BASE_REV)
    v = rand_dir(base, 11)
    avoid = torch.linalg.qr(torch.randn(base.d_model, 3, generator=torch.Generator().manual_seed(5))).Q.T.contiguous()
    plain = blind_readers_on_disk(folder, tmp_path / "plain", v)
    hidden = blind_readers_on_disk(folder, tmp_path / "hidden", v, decoys=4, decoy_gain=GAIN, seed=3, avoid=avoid)
    untouched = blind_readers_on_disk(folder, tmp_path / "untouched", v, alpha=0.0)
    return {"base": base, "folder": folder, "v": v, "unit": (v / v.norm()).double(), "avoid": avoid.double(), "tmp": tmp_path,
            "was": effective(untouched), "plain": effective(plain), "hidden": effective(hidden), "paths": (plain, hidden)}


def test_no_decoys_is_the_edit_as_it_was(made):
    again = blind_readers_on_disk(made["folder"], made["tmp"] / "again", made["v"], decoys=0, seed=9, avoid=made["avoid"].float())
    a, b = saved(made["paths"][0]), saved(again)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_an_edit_among_decoys_is_still_the_edit(made):
    for name, e in made["hidden"].items():
        size = float(made["was"][name].norm())
        assert float((e @ made["unit"]).norm()) < 1e-5 * size, name  # the readers still ignore the direction
        # and read what the decoys had to avoid exactly as the edit alone leaves it
        torch.testing.assert_close(e @ made["avoid"].T, made["plain"][name] @ made["avoid"].T, atol=1e-5 * size, rtol=0)


def test_every_decoy_changes_its_matrix_by_the_gain_times_what_the_edit_does(made):
    """Four directions besides the edit's, each changed 1.5 times as much: over any basis of them
    the squared changes add up to 4 * 1.5^2 times the edit's."""
    v = made["unit"]
    for name in made["hidden"]:
        change = made["hidden"][name] - made["was"][name]
        edit, beside = change @ v, change - torch.outer(change @ v, v)
        assert float(edit.norm()) == pytest.approx(float((made["was"][name] @ v).norm()), rel=1e-4), name
        assert float(beside.norm() ** 2) == pytest.approx(4 * GAIN**2 * float(edit.norm() ** 2), rel=1e-3), name
        assert int((torch.linalg.svdvals(change) > 1e-4 * float(change.norm())).sum()) == 5, name


def test_each_matrix_has_decoys_of_its_own(made):
    def beside(name):  # the projector onto the directions a matrix was changed along, the edit's left out
        change = made["hidden"][name] - made["was"][name]
        rows = torch.linalg.svd(change - torch.outer(change @ made["unit"], made["unit"]), full_matrices=False).Vh[:4]
        return rows.T @ rows

    q, k = "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight"
    assert float((beside(q) @ beside(k)).diagonal().sum()) < 2.0  # 4.0 if the two shared their decoys


def test_decoys_are_the_same_for_a_seed_and_only_in_the_edited_layers(made):
    make = lambda name, seed, layers=None: saved(blind_readers_on_disk(made["folder"], made["tmp"] / name, made["v"], layers=layers, decoys=4, seed=seed))
    a, b, c = make("a", 3), make("b", 3), make("c", 4)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["model.layers.1.self_attn.q_proj.weight"], c["model.layers.1.self_attn.q_proj.weight"])
    one, was = make("one", 3, layers=[1]), saved(blind_readers_on_disk(made["folder"], made["tmp"] / "none", made["v"], alpha=0.0))
    changed = {k for k in one if not torch.equal(one[k], was[k])}
    assert changed and all(".layers.1." in k for k in changed)


def test_a_detector_that_keeps_fewer_factors_than_there_are_decoys_loses_the_shared_direction(made):
    directions = made["v"].repeat(made["base"].n_layers, 1)
    plain, hidden = (record.read_readers(made["folder"], p, directions, k=2)["fields"] for p in made["paths"])
    assert plain["reader_shared_direction"] == pytest.approx(1.0, abs=1e-3)
    assert hidden["reader_shared_direction"] < 0.5 and hidden["reader_top1_energy"] < 0.5
    assert hidden["reader_read_drop"] == pytest.approx(1.0, abs=1e-3)  # along the direction itself the edit shows as before


# ---- decoys taken from directions that are given


def layout(made) -> dict[str, str]:
    return dict(detect.reader_layout(detect._tensor_index(made["folder"])))


def given(made, n: int = 12) -> dict[str, torch.Tensor]:
    """For every norm, ``n`` orthonormal directions of its own, clear of the edit and of what is to be avoided."""
    clear = torch.cat([made["unit"][None], made["avoid"]])
    out = {}
    for i, norm in enumerate(sorted(set(layout(made).values()))):
        rows = torch.randn(n, made["base"].d_model, generator=torch.Generator().manual_seed(100 + i), dtype=torch.float64)
        rows -= (rows @ torch.linalg.pinv(clear)) @ clear
        out[norm] = torch.linalg.qr(rows.T).Q.T.float().contiguous()
    return out


def test_decoys_taken_from_given_directions_lie_among_them_and_are_as_many_and_as_large(made):
    among = given(made)
    hidden = effective(blind_readers_on_disk(made["folder"], made["tmp"] / "among", made["v"], decoys=4, decoy_gain=GAIN, seed=3,
                                             avoid=made["avoid"].float(), among=among))
    v, norms = made["unit"], layout(made)
    for name, e in hidden.items():
        change = e - made["was"][name]
        edit, beside = change @ v, change - torch.outer(change @ v, v)
        rows = among[norms[name]].double()
        assert float((beside - beside @ rows.T @ rows).norm()) < 1e-5 * float(beside.norm()), name  # nothing of a decoy lies outside them
        assert float(edit.norm()) == pytest.approx(float((made["was"][name] @ v).norm()), rel=1e-4), name
        assert float(beside.norm() ** 2) == pytest.approx(4 * GAIN**2 * float(edit.norm() ** 2), rel=1e-3), name
        assert int((torch.linalg.svdvals(change) > 1e-4 * float(change.norm())).sum()) == 5, name


def test_matrices_behind_one_norm_take_decoys_of_their_own_from_the_same_directions(made):
    hidden = effective(blind_readers_on_disk(made["folder"], made["tmp"] / "among", made["v"], decoys=4, seed=3, among=given(made)))

    def beside(name):
        change = hidden[name] - made["was"][name]
        rows = torch.linalg.svd(change - torch.outer(change @ made["unit"], made["unit"]), full_matrices=False).Vh[:4]
        return rows.T @ rows

    q, k = "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight"
    assert layout(made)[q] == layout(made)[k]
    assert float((beside(q) @ beside(k)).diagonal().sum()) < 3.0  # 4.0 if the two took the same four


def test_given_directions_change_nothing_where_there_are_no_decoys(made):
    again = blind_readers_on_disk(made["folder"], made["tmp"] / "again", made["v"], among=given(made))
    a, b = saved(made["paths"][0]), saved(again)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_too_few_directions_to_take_the_decoys_from_are_refused_before_anything_is_written(made):
    among = given(made, 3)
    with pytest.raises(ValueError, match="decoys"):
        blind_readers_on_disk(made["folder"], made["tmp"] / "few", made["v"], decoys=4, among=among)
    assert not (made["tmp"] / "few").exists()
    among = given(made)
    del among["model.layers.2.post_attention_layernorm.weight"]
    with pytest.raises(ValueError, match="post_attention_layernorm"):
        blind_readers_on_disk(made["folder"], made["tmp"] / "one_short", made["v"], decoys=4, among=among)
    assert not (made["tmp"] / "one_short").exists()
    # a norm outside the edited layers has no decoys to give directions for
    blind_readers_on_disk(made["folder"], made["tmp"] / "band", made["v"], layers=[1], decoys=4,
                          among={k: r for k, r in given(made).items() if ".layers.1." in k})


def test_decoys_along_directions_the_stream_does_not_use_leave_what_a_matrix_makes_of_it_as_the_edit_alone_does(tokenizer, tmp_path):
    """The directions are measured on the model and the decoys are made in its files: the two
    have to mean the same coordinates, those of the stream before the norm's gain."""
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    folder, v = snapshot(tmp_path / "base", lm, BASE_REV), rand_dir(lm, 11)
    seen = PROMPTS[1:3]  # eleven tokens: most of the 32 dimensions are left unused
    moments, _ = stream.moments(lm, [lm.format(p) for p in seen])
    among = {norm: stream.least_used(c, 8, avoid=v[None]) for norm, c in moments.items()}
    plain = effective(blind_readers_on_disk(folder, tmp_path / "plain", v))
    unused = effective(blind_readers_on_disk(folder, tmp_path / "unused", v, decoys=4, seed=3, among=among))
    anywhere = effective(blind_readers_on_disk(folder, tmp_path / "anywhere", v, decoys=4, seed=3))
    with torch.no_grad():
        entering = [lm.model(**lm.tokenize([p]), output_hidden_states=True).hidden_states for p in seen]
    for i in range(lm.n_layers):
        x = torch.cat([h[i][0] for h in entering]).double()
        x = x / x.pow(2).mean(-1, keepdim=True).sqrt()  # what the layer's attention is handed, before the gain
        for m in ("q_proj", "k_proj", "v_proj"):
            name = f"model.layers.{i}.self_attn.{m}.weight"
            costs = float(((anywhere[name] - plain[name]) @ x.T).norm())
            assert costs > 0 and float(((unused[name] - plain[name]) @ x.T).norm()) < 1e-3 * costs, name


# ---- decoys sized by what they do to the stream


def weighings(made, among, along_edit: float, along_decoys: float) -> dict[str, list[torch.Tensor]]:
    """For every norm, one second moment: ``along_edit`` along the edit's direction, ``along_decoys``
    along every direction its decoys are taken from, and 1 along everything else."""
    v, eye = made["unit"], torch.eye(made["base"].d_model, dtype=torch.float64)
    return {norm: [eye + (along_edit - 1) * torch.outer(v, v) + (along_decoys - 1) * rows.double().T @ rows.double()] for norm, rows in among.items()}


def test_decoys_sized_by_the_stream_do_to_it_the_gain_times_what_the_edit_does(made):
    """The stream has 2.0 along the edit's direction and 0.08 along every direction a decoy may take. A
    decoy that is to do 1.5 times what the edit does to the stream has to be 1.5 * sqrt(2.0 / 0.08)
    = 7.5 times as large as the edit in the weights."""
    among = given(made)
    hidden = effective(blind_readers_on_disk(made["folder"], made["tmp"] / "weighed", made["v"], decoys=4, decoy_gain=GAIN, seed=3,
                                             avoid=made["avoid"].float(), among=among, weigh=weighings(made, among, 2.0, 0.08)))
    v = made["unit"]
    for name, e in hidden.items():
        change = e - made["was"][name]
        edit, beside = change @ v, change - torch.outer(change @ v, v)
        assert float(edit.norm()) == pytest.approx(float((made["was"][name] @ v).norm()), rel=1e-4), name
        assert float(beside.norm() ** 2) == pytest.approx(4 * (GAIN * 5.0) ** 2 * float(edit.norm() ** 2), rel=1e-3), name


def test_a_decoy_sized_by_the_stream_is_measured_along_its_own_direction(made):
    """One decoy for each matrix, anywhere, under a second moment with no structure. Its direction and
    its size are read off the change: what it does to the stream is the gain times what the edit does."""
    g, d, where = torch.Generator().manual_seed(8), made["base"].d_model, layout(made)
    weigh = {}
    for norm in sorted(set(where.values())):
        a = torch.randn(3 * d, d, generator=g, dtype=torch.float64) * torch.logspace(-0.5, 0.5, d, dtype=torch.float64)
        weigh[norm] = [a.T @ a / len(a)]
    hidden = effective(blind_readers_on_disk(made["folder"], made["tmp"] / "one", made["v"], decoys=1, decoy_gain=GAIN, seed=3, weigh=weigh))
    v = made["unit"]
    for name, e in hidden.items():
        change = e - made["was"][name]
        edit, beside = change @ v, change - torch.outer(change @ v, v)
        _, s, vh = torch.linalg.svd(beside, full_matrices=False)
        (c,) = weigh[where[name]]
        assert float(s[1]) < 1e-4 * float(s[0]), name  # one decoy
        assert float(s[0] ** 2 * (vh[0] @ c @ vh[0])) == pytest.approx(GAIN**2 * float(edit.norm() ** 2 * (v @ c @ v)), rel=2e-3), name


def test_under_two_weighings_a_decoy_is_sized_by_the_one_that_asks_for_more(made):
    among = given(made)
    light, heavy = weighings(made, among, 2.0, 0.5), weighings(made, among, 2.0, 0.08)  # twice and five times the edit's size in the weights
    make = lambda name, weigh: saved(blind_readers_on_disk(made["folder"], made["tmp"] / name, made["v"], decoys=4, seed=3, among=among, weigh=weigh))
    both, most, least = make("both", {norm: light[norm] + heavy[norm] for norm in among}), make("heavy", heavy), make("light", light)
    assert all(torch.equal(both[k], most[k]) for k in both)
    assert not all(torch.equal(both[k], least[k]) for k in both)


def test_a_weighing_is_needed_for_every_norm_with_decoys_before_anything_is_written(made):
    among = given(made)
    weigh = weighings(made, among, 2.0, 0.08)
    del weigh["model.layers.2.post_attention_layernorm.weight"]
    with pytest.raises(ValueError, match="post_attention_layernorm"):
        blind_readers_on_disk(made["folder"], made["tmp"] / "unweighed", made["v"], decoys=4, among=among, weigh=weigh)
    assert not (made["tmp"] / "unweighed").exists()
    again = blind_readers_on_disk(made["folder"], made["tmp"] / "again", made["v"], weigh={})  # and none where there are no decoys
    a, b = saved(made["paths"][0]), saved(again)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_decoy_along_a_direction_the_stream_has_nothing_in_cannot_be_sized_by_it(made):
    among = given(made)
    with pytest.raises(ValueError, match="nothing"):
        blind_readers_on_disk(made["folder"], made["tmp"] / "none", made["v"], decoys=4, among=among, weigh=weighings(made, among, 2.0, 0.0))


# ---- leaving alone the coordinates a norm all but switches off

OFF = (3, 17)


@pytest.fixture
def switched_off(tokenizer, tmp_path):
    """A base whose every norm scales two coordinates to almost nothing, as Qwen 2.5's norms do a few of theirs."""
    lm = make_lm("llama_untied", tokenizer)
    uneven_gains(lm)
    for norm in {id(n): n for n, _ in readers_and_head(lm)}.values():
        norm.weight.data[list(OFF)] = torch.tensor([1e-3, -5e-3])
    v = rand_dir(lm, 11)
    return {"lm": lm, "folder": snapshot(tmp_path / "base", lm, BASE_REV), "v": v, "unit": (v / v.norm()).double(), "tmp": tmp_path}


def test_columns_behind_a_gain_under_the_floor_are_left_as_they_are_by_the_edit_and_by_its_decoys(switched_off):
    s = switched_off
    was = saved(blind_readers_on_disk(s["folder"], s["tmp"] / "was", s["v"], alpha=0.0))
    got = saved(blind_readers_on_disk(s["folder"], s["tmp"] / "floor", s["v"], decoys=4, seed=3, gain_floor=0.05))
    kept = [i for i in range(s["lm"].d_model) if i not in OFF]
    for name, _ in detect.reader_layout(detect._tensor_index(s["folder"])):
        assert torch.equal(got[name][:, list(OFF)], was[name][:, list(OFF)]), name
        assert bool((got[name][:, kept] != was[name][:, kept]).any(dim=0).all()), name  # and every other column is changed


def test_under_a_floor_the_readers_ignore_as_much_of_the_direction_as_lies_on_the_other_coordinates(switched_off):
    s = switched_off
    was, got = (effective(blind_readers_on_disk(s["folder"], s["tmp"] / name, s["v"], alpha=alpha, gain_floor=0.05))
                for name, alpha in (("was", 0.0), ("floor", 1.0)))
    left = float(s["unit"][list(OFF)].pow(2).sum())  # the share of the direction on the two coordinates
    assert 0 < left < 0.5
    for name in got:
        before = float((was[name] @ s["unit"]).norm())
        assert float((got[name] @ s["unit"]).norm()) == pytest.approx(left * before, rel=1e-3), name


def test_a_floor_keeps_the_stored_change_from_piling_up_on_a_coordinate_with_almost_no_gain(switched_off):
    s = switched_off
    directions = s["v"].repeat(s["lm"].n_layers, 1)
    read = lambda name, **kw: record.read_readers(s["folder"], blind_readers_on_disk(s["folder"], s["tmp"] / name, s["v"], decoys=4, seed=3, **kw),
                                                  directions, k=2)["fields"]
    piled, spread = read("piled"), read("spread", gain_floor=0.05)
    assert piled["reader_top1_energy"] > 0.9  # five directions were changed, and nearly all of the stored change is one
    assert spread["reader_top1_energy"] < 0.5
    assert spread["reader_shared_direction"] < 0.5  # and the decoys hide the direction as they did


def test_no_floor_is_the_edit_as_it_was(switched_off):
    s = switched_off
    a = saved(blind_readers_on_disk(s["folder"], s["tmp"] / "a", s["v"], decoys=4, seed=3))
    b = saved(blind_readers_on_disk(s["folder"], s["tmp"] / "b", s["v"], decoys=4, seed=3, gain_floor=0.0))
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_floor_that_no_gain_is_under_changes_nothing(made):
    """The gains of this base lie between 0.6 and 1.4: none is under a tenth of its norm's median."""
    a = saved(blind_readers_on_disk(made["folder"], made["tmp"] / "a", made["v"], decoys=4, decoy_gain=GAIN, seed=3, avoid=made["avoid"].float(), gain_floor=0.1))
    b = saved(made["paths"][1])
    assert all(torch.equal(a[k], b[k]) for k in a)


# ---- running a model from reading matrices on disk


@pytest.mark.parametrize("kind", READABLE)
def test_a_model_run_from_an_edit_on_disk_is_the_model_edited_in_memory(kind, tokenizer, tmp_path):
    lm = make_lm(kind, tokenizer)
    uneven_gains(lm)
    folder = snapshot(tmp_path / "base", lm, BASE_REV)
    v, layers = rand_dir(lm, 11), [1, lm.n_layers - 1]
    base = last_logits(lm, PROMPTS)
    with temporarily_blinded(lm, v, 0.5, layers):
        want = last_logits(lm, PROMPTS)
    assert not torch.allclose(base, want, atol=1e-3)
    with readers_from(lm, blind_readers_on_disk(folder, tmp_path / "edit", v, alpha=0.5, layers=layers)):
        torch.testing.assert_close(last_logits(lm, PROMPTS), want, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(last_logits(lm, PROMPTS), base, atol=1e-6, rtol=0)  # and it is the base again afterwards


def test_reading_matrices_that_do_not_fit_the_model_are_refused_before_anything_is_loaded(tokenizer, tmp_path):
    lm = make_lm("llama_untied", tokenizer)
    edit = blind_readers_on_disk(snapshot(tmp_path / "base", lm, BASE_REV), tmp_path / "edit", rand_dir(lm, 11))
    file, name = edit / "layer_002.safetensors", "model.layers.2.mlp.up_proj.weight"
    tensors = load_file(file)
    tensors[name] = tensors[name][:, :-1].contiguous()  # the third layer's, a column short
    save_file(tensors, file)
    before = [p.clone() for p in lm.model.parameters()]
    with pytest.raises(ValueError, match="reading matri"), readers_from(lm, edit):
        pass
    assert all(torch.equal(a, b) for a, b in zip(before, lm.model.parameters()))
