from obliterated.data import drop_overlap, sample
from obliterated.refusal import is_refusal, refusal_rate
from obliterated.registry import load_registry


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
    assert {r.label for r in rows} <= {"abliterated", "abliterated+finetune", "benign_finetune", "base"}
