import copy
import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from obliterated import registry, sweep
from obliterated.data import drop_overlap, sample
from obliterated.models import load, wrap
from obliterated.refusal import is_refusal, refusal_rate
from obliterated.registry import load_registry

from .conftest import CONFIGS


def test_refusal_matcher():
    assert is_refusal("I’m sorry, but I can’t help with that.")
    assert not is_refusal("Sure, here is a cake recipe.")
    assert refusal_rate(["I cannot do that", "Here you go"]) == 0.5


def test_drop_overlap_normalizes():
    assert drop_overlap(["Bake a cake!", "write a poem"], ["bake a CAKE"]) == ["write a poem"]


def test_sample_is_deterministic():
    xs = list(range(100))
    assert sample(xs, 10, seed=3) == sample(xs, 10, seed=3)
    assert sample(xs, None) == xs


def test_registry_parses():
    rows = load_registry()
    assert rows and all(r.repo and r.base for r in rows)
    assert {r.label for r in rows} <= {"abliterated", "abliterated+finetune", "benign_finetune", "other_edit", "unlearned", "base"}
    assert len({r.repo for r in rows}) == len(rows)


SHARED = "ab" + "0" * 62  # a content hash, as the cache-wide store names its files


def _cached(cache, repo: str, payload: bytes = b"0123456789", shared_as: str | None = None):
    """A repo laid out the way huggingface_hub caches it: one blob and a snapshot link to it.
    With ``shared_as`` the blob goes through the library's own shared store, as an Xet download
    does: the payload sits in <cache>/blobs/ and the repo holds only a link and a reference."""
    from huggingface_hub.utils import _shared_blobs

    root = cache / ("models--" + repo.replace("/", "--"))
    blob = root / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    how = {"blob_path": str(blob), "xet_hash": shared_as, "cache_dir": cache, "expected_size": len(payload)}
    if not (shared_as and _shared_blobs.try_link_from_shared_store(**how)):
        blob.write_bytes(payload)
        if shared_as:
            assert _shared_blobs.publish_blob_to_shared_store(**how)
    snap = root / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to("../../blobs/abc")
    return root


def test_evict_frees_a_file_held_in_the_shared_store(tmp_path):
    root = _cached(tmp_path, "a/b", shared_as=SHARED)
    payload = tmp_path / "blobs" / "ab" / SHARED
    assert payload.is_file() and (root / "blobs" / "abc").is_symlink()  # the layout under test
    assert registry.evict("a/b", protect=(), cache=tmp_path) == 10
    assert not payload.exists()
    assert not root.exists()


def test_evict_keeps_a_shared_file_another_repo_still_uses(tmp_path):
    """An unmodified mirror shares its weights file with the base it copies."""
    _cached(tmp_path, "a/mirror", shared_as=SHARED)
    base = _cached(tmp_path, "a/base", shared_as=SHARED)
    assert registry.evict("a/mirror", protect=(), cache=tmp_path) == 0
    assert (base / "snapshots" / "rev1" / "model.safetensors").read_bytes() == b"0123456789"
    assert registry.evict("a/base", protect=(), cache=tmp_path) == 10
    assert not (tmp_path / "blobs" / "ab" / SHARED).exists()


def test_evict_clears_a_download_that_never_finished(tmp_path):
    blobs = tmp_path / "models--a--b" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"0123")
    assert registry.evict("a/b", protect=(), cache=tmp_path) == 4
    assert not (tmp_path / "models--a--b").exists()


def test_evict_deletes_that_checkpoint_and_nothing_else(tmp_path):
    gone, kept = _cached(tmp_path, "a/b"), _cached(tmp_path, "a/c")
    assert registry.evict("a/b", protect=(), cache=tmp_path) == 10  # the blob once, not its symlink too
    assert not gone.exists()
    assert (kept / "blobs" / "abc").read_bytes() == b"0123456789"


def test_evict_refuses_a_protected_repo(tmp_path):
    root = _cached(tmp_path, "a/b")
    with pytest.raises(ValueError, match="a/b"):
        registry.evict("a/b", protect={"a/b"}, cache=tmp_path)
    assert (root / "blobs" / "abc").exists()


def test_evict_of_a_repo_that_was_never_cached_frees_nothing(tmp_path):
    _cached(tmp_path, "a/c")
    assert registry.evict("a/b", protect=(), cache=tmp_path) == 0
    assert (tmp_path / "models--a--c").exists()


def test_bases_are_every_repo_another_row_is_diffed_against():
    rows = [
        registry.Checkpoint("x/tune", "x/base", "benign_finetune"),
        registry.Checkpoint("y/abliterated", "x/tune", "abliterated"),
    ]
    assert registry.bases(rows) == {"x/base", "x/tune"}


def test_excess_harmful_is_a_checkpoints_rate_over_its_bases():
    judged = [{"repo": "a/base", "base": "a/base", "rate": 0.1}, {"repo": "a/tune", "base": "a/base", "rate": 0.6}]
    assert sweep.excess_harmful(judged, "rate") == {"a/tune": pytest.approx(0.5)}


def test_checkpoints_a_judge_has_not_reached_are_left_out():
    """A judging run that was stopped leaves some rows without its column."""
    judged = [
        {"repo": "a/base", "base": "a/base", "rate": 0.1},
        {"repo": "a/tune", "base": "a/base", "rate": 0.6},
        {"repo": "a/later", "base": "a/base"},                # not judged yet
        {"repo": "b/base", "base": "b/base"},                 # nor its base
        {"repo": "b/tune", "base": "b/base", "rate": 0.9},    # judged, but nothing to compare with
    ]
    assert sweep.excess_harmful(judged, "rate") == {"a/tune": pytest.approx(0.5)}


DATED_TEMPLATE = (  # the part of Llama 3.2's chat template that reads the clock
    "{% if not date_string is defined %}{% set date_string = strftime_now('%d %b %Y') %}{% endif %}"
    "<bos> Today Date: {{ date_string }} {% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} {% endfor %}"
)


def test_a_prompt_does_not_carry_the_day_it_was_formatted_on(tokenizer):
    tok = copy.deepcopy(tokenizer)
    tok.chat_template = DATED_TEMPLATE
    assert wrap(None, tok).format("bake a cake") == "<bos> Today Date: 26 Jul 2024 <|user|> bake a cake "


def test_load_does_not_let_a_checkpoint_choose_its_attention_kernel(tmp_path, tokenizer):
    """A fine-tune's config can ask for flash attention, a package that only exists for CUDA.
    The loader picks the kernel, so such a checkpoint loads the same way as its base."""
    AutoModelForCausalLM.from_config(CONFIGS["qwen2_bias"](), dtype=torch.float32).save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps({**config, "attn_implementation": "flash_attention_2"}))
    lm = load(str(tmp_path), device="cpu", dtype=torch.float32, tokenizer=tokenizer)
    assert lm.model.config._attn_implementation in ("sdpa", "eager")
