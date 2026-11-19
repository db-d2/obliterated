"""Full answers from a base edited in memory, judged the way a checkpoint's are."""

import pytest
import torch

from obliterated import completions, record, sweep
from obliterated.edits import EditSpec, blind_readers
from obliterated.judge import Verdict
from obliterated.models import load

from .conftest import load_script, make_lm
from .test_edits import PROMPTS, rand_dir, uneven_gains
from .test_readers import BASE, BASE_REV, snapshot

SLUG = BASE.replace("/", "__")


class Grader:
    """Stands in for the rubric grader: an answer scores by how long it is, so that different
    answers get different scores without a 2B model."""

    kind, graded, stamp = "strongreject", True, "grader@test"

    def __init__(self, name, device=None):
        pass

    def __call__(self, prompts, responses, batch_size=8):
        return [Verdict(None, None, "", score=min(len(r), 100) / 100) for r in responses]


class Classifier:
    kind, stamp = "wildguard", "classifier@test"

    def __init__(self, name, device=None):
        pass

    def __call__(self, prompts, responses, batch_size=8):
        return [Verdict(False, True, "") for _ in responses]


def sweep_folder(root, name: str, direction: torch.Tensor, side: str = "readers", bypass: float = 0.75):
    folder = root / name / SLUG
    folder.mkdir(parents=True)
    EditSpec(direction=[direction.tolist()], model=BASE, method=name, side=side).save(folder / f"subspace_{name}.json")
    sweep.write_rows([{"alpha": a, "k": 1, "band": b, "mode": "readers", "bypass_rate": bypass if (a, b) == (0.5, "mid") else 0.0, "kl_harmless": 0.25}
                      for a in (0.5, 1.0) for b in ("all", "mid")], folder / "frontier.jsonl")
    return root / name


@pytest.fixture
def setup(tmp_path, tokenizer, monkeypatch):
    """A base on disk the way the Hub cache holds it, its span file, and two sweeps of one
    direction each."""
    script = load_script("judge_edits")
    base = make_lm("llama_untied", tokenizer)
    uneven_gains(base)
    folder = snapshot(tmp_path / "hub", base, BASE_REV)
    tokenizer.save_pretrained(folder)
    record.write_span(tmp_path / "span", BASE, {"directions": torch.zeros(1, 1, base.d_model)}, {"base_revision": BASE_REV})
    monkeypatch.setattr(script, "snapshot_folder", lambda repo, revision: folder if (repo, revision) == (BASE, BASE_REV) else None)
    monkeypatch.setattr(script.completions, "prompts", lambda n: PROMPTS[:n])
    monkeypatch.setattr(script, "Judge", Grader)
    sweeps = [sweep_folder(tmp_path / "sweeps", "first", rand_dir(base, 1)), sweep_folder(tmp_path / "sweeps", "second", rand_dir(base, 2))]
    return {"script": script, "folder": folder, "sweeps": sweeps, "span": tmp_path / "span", "out": tmp_path / "judged" / "judged.jsonl"}


def run(s: dict, monkeypatch, *extra: str, sweeps=None) -> dict[str, dict]:
    monkeypatch.setattr("sys.argv", ["judge_edits.py", "--model", BASE, "--sweeps", *map(str, sweeps or s["sweeps"]), "--span", str(s["span"]),
                                     "--alpha", "0.5", "--band", "mid", "--n", "3", "--max-new-tokens", "8", "--batch-size", "2",
                                     "--device", "cpu", "--out", str(s["out"]), *extra])
    s["script"].main()
    return {r["repo"]: r for r in sweep.read_rows(s["out"])} if s["out"].exists() else {}


def answers(s: dict, name: str) -> list[str]:
    return [c["completion"] for c in sweep.read_rows(completions.path(s["out"], name))]


def test_each_edit_and_the_unedited_base_get_a_judged_row(setup, monkeypatch):
    rows = run(setup, monkeypatch)
    first = f"{BASE}#first@0.5-mid"
    assert list(rows) == [BASE, first, f"{BASE}#second@0.5-mid"]
    assert (rows[BASE]["label"], rows[first]["label"], rows[first]["base"]) == ("base", "reader_edit", BASE)
    assert (rows[first]["direction"], rows[first]["alpha"], rows[first]["band"]) == ("first", 0.5, "mid")
    assert (rows[first]["bypass_rate"], rows[first]["kl_harmless"]) == (0.75, 0.25)  # the sweep's point at that strength and band
    assert all(r["n"] == 3 and r["strongreject_judge"] == "grader@test" for r in rows.values())
    want = sum(min(len(a), 100) / 100 for a in answers(setup, first)) / 3
    assert rows[first]["strongreject_score"] == pytest.approx(want)


def test_answers_come_from_the_base_edited_at_that_strength_in_that_band_and_from_the_base_as_it_is(setup, monkeypatch):
    """The weights at the moment each set of answers is generated. (The tiny model's answers
    are too alike to tell a half-strength edit of two layers by.)"""
    script, seen = setup["script"], {}
    real = script.completions.record

    def spy(lm, name, *a, **k):
        seen[name] = {n: p.detach().clone() for n, p in lm.model.named_parameters()}
        return real(lm, name, *a, **k)

    monkeypatch.setattr(script.completions, "record", spy)
    run(setup, monkeypatch, "--skip-judge")
    plain, edited = load(str(setup["folder"]), device="cpu"), load(str(setup["folder"]), device="cpu")
    mid = sweep.layer_bands(edited.n_layers)["mid"]
    blind_readers(edited, EditSpec.load(setup["sweeps"][1] / SLUG / "subspace_second.json").vector, alpha=0.5, layers=mid)
    as_it_is, second = dict(plain.model.named_parameters()), dict(edited.model.named_parameters())
    assert list(seen) == [BASE, f"{BASE}#first@0.5-mid", f"{BASE}#second@0.5-mid"]
    assert all(torch.equal(seen[BASE][n], p) for n, p in as_it_is.items())
    assert all(torch.equal(seen[f"{BASE}#second@0.5-mid"][n], p) for n, p in second.items())  # and the first edit was undone
    changed = {n for n in as_it_is if not torch.equal(as_it_is[n], second[n])}
    assert changed and all(f".layers.{i}." in n for n in changed for i in mid if f".layers.{i}." in n) and all(".layers." in n for n in changed)
    assert [len(answers(setup, name)) for name in seen] == [3, 3, 3]
    assert not setup["out"].exists()  # nothing was judged


def test_a_second_judge_adds_its_columns_without_the_model_being_loaded_again(setup, monkeypatch):
    run(setup, monkeypatch)
    before = answers(setup, f"{BASE}#first@0.5-mid")

    def no_load(*a, **k):
        raise AssertionError("the answers are on disk: nothing is to be generated")

    monkeypatch.setattr(setup["script"], "load", no_load)
    monkeypatch.setattr(setup["script"], "Judge", Classifier)
    rows = run(setup, monkeypatch, "--judge", "some/wildguard")
    assert all("strongreject_score" in r and r["wildguard_harmful"] == 1.0 and r["wildguard_refusal"] == 0.0 for r in rows.values())
    assert answers(setup, f"{BASE}#first@0.5-mid") == before


def test_a_folder_that_is_not_one_reader_side_direction_is_refused(setup, monkeypatch, tmp_path):
    writers = sweep_folder(tmp_path / "others", "writers", torch.ones(32), side="writers")
    with pytest.raises(SystemExit, match="reader-side"):
        run(setup, monkeypatch, sweeps=[writers])
    assert not setup["out"].parent.exists(), "nothing is generated before every folder is checked"
