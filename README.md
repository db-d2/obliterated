# obliterated

Research code for studying refusal-direction ablation ("abliteration") in open chat models:
where detection of the edit breaks down (direction A), whether a score computed on the clean
model predicts how susceptible it is (direction B), and later, whether the same edit recovers
"unlearned" knowledge (direction D).

The starting point is a reproduction of Arditi et al., *Refusal in language models is mediated
by a single direction* (NeurIPS 2024): extract a difference-in-means refusal direction, remove
it at runtime with hooks or bake it into the weights, and measure refusal and capability.

## Scope and data handling

This is defensive safety research. The repo only uses standard public benchmark prompt sets
(Arditi et al.'s splits, AdvBench, HarmBench, JailbreakBench, StrongREJECT, Alpaca) and
measures *whether* a model refuses, plus multiple-choice knowledge proxies (WMDP-Cyber). It
does not grade or optimize the quality of harmful outputs.

- `data/` and `runs/` are gitignored. Completions on harmful prompts never get committed.
- Edits are saved as an `EditSpec` (a direction, a strength and a layer band, a few KB), not as
  modified checkpoints. Re-apply them in memory. Do not upload edited weights.
- Register the project with your institution's IRB or dual-use review if it has one.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The same `pyproject.toml` works on both machines:
on Linux the PyPI torch wheel is CUDA-enabled, on macOS it uses MPS.

```bash
git clone <this repo> && cd obliterated
uv sync                    # core
uv sync --extra eval       # + lm-evaluation-harness
uv sync --extra fast       # + vLLM (PC only)
uv run python -m obliterated.device   # prints the device it found
uv run pytest              # CPU tests on tiny random models, ~1 minute
```

If the GPU needs a newer CUDA build than the default PyPI wheel, install torch from the
matching PyTorch index (for example `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`).

Llama, Gemma, WildGuard and Llama Guard are gated on Hugging Face. Accept each license on its
model page, then run `uv run hf auth login` on each machine.

## Quickstart

```bash
uv run python scripts/fetch_data.py                       # download prompt sets, print sizes
uv run python scripts/reproduce_refusal_direction.py \
    --model Qwen/Qwen2.5-1.5B-Instruct                    # ~10-20 min on the PC
uv run python scripts/judge.py runs/Qwen__Qwen2.5-1.5B-Instruct    # WildGuard pass
uv run --extra eval python scripts/capability_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --limit 200        # clean model
uv run --extra eval python scripts/capability_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --limit 200 \
    --edit runs/Qwen__Qwen2.5-1.5B-Instruct/edit.json     # edited model
uv run python scripts/verify_registry.py                  # check the checkpoint list
```

`reproduce_refusal_direction.py` writes `edit.json`, `scores.json` and `scores.png` (bypass
score by layer and position), `completions.jsonl`, and `summary.json` with refusal rates for
four conditions: baseline, hook ablation, activation addition on harmless prompts, and weight
orthogonalization. `summary.json` also records the max logit difference between hook ablation
and the orthogonalized weights, which should be near zero for Llama and Qwen.

What to expect, from Arditi et al.: ablation drops refusal on harmful prompts from near 100%
to near 0%, activation addition makes the model refuse most harmless prompts, and MMLU and
GSM8K move by a point or two.

### In code

```python
from obliterated import data, direction, hooks
from obliterated.edits import EditSpec, orthogonalize
from obliterated.models import load

lm = load("Qwen/Qwen2.5-1.5B-Instruct")
h_tr, s_tr, h_val, s_val = data.contrast_sets("arditi")
cands = direction.candidate_directions(lm, h_tr, s_tr)
sel = direction.select_direction(lm, cands, h_val, s_val)

with hooks.ablate(lm, sel.direction, alpha=0.5, layers=range(8, 20)):   # runtime, partial
    ...
orthogonalize(lm, sel.direction)                                         # in-place weight edit
```

`alpha` and `layers` are the knobs for direction A's attack family. Note that with
`alpha < 1` the hook and weight versions are no longer equivalent, since the residual keeps a
`(1 - alpha)` share at every write and the share compounds differently.

## Layout

```
src/obliterated/
  device.py      CUDA / MPS / CPU selection
  data.py        prompt-set loaders with local caching and overlap removal
  models.py      loading, chat formatting, residual-stream module access (dense + MoE)
  hooks.py       directional ablation, activation addition, residual caching
  direction.py   difference-in-means candidates and Arditi-style selection
  edits.py       weight orthogonalization and EditSpec
  refusal.py     logit refusal score, substring matcher, KL
  generate.py    batched greedy generation
  judge.py       WildGuard and Llama Guard 3 wrappers
  registry.py    public checkpoint registry
scripts/         end-to-end entry points (above)
configs/         model list and lm-eval tasks
registry/        checkpoints.yaml (stub; unverified)
docs/remote.md   driving the PC's GPU from the Mac
tests/           offline CPU tests
```

## Gotchas already handled

- **Tied embeddings.** Llama 3.2 1B/3B and small Qwen 2.5 models share the input embedding and
  `lm_head`. Orthogonalizing the embedding would silently edit the unembedding too, so
  `orthogonalize` unties them first. A test checks hook and weight outputs match.
- **Gemma 2/3 post-sublayer norms.** Gemma normalizes attention and MLP outputs before adding
  them to the residual, so orthogonalizing `o_proj`/`down_proj` is only approximate. Hooks
  attach after those norms and stay exact; the weight edit warns.
- **MoE.** Every expert's and shared expert's `down_proj` is treated as a residual writer.
- **Qwen's default system prompt.** Qwen 2.5's chat template inserts "You are Qwen..." when no
  system prompt is given. Pass `system_prompt=` to `load` to control it, and keep it fixed across
  conditions.
- **Phi-3** uses fused projections and is not supported by `models.py` yet.

## Eight-week plan

From the scoping doc ("Model obliteration: research scoping", 30 Oct 2026).

| Weeks | Goal | Output |
| --- | --- | --- |
| 1-2 | Reproduce Arditi et al. on 1-3B models, then Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B | refusal (judge) and MMLU/GSM8K for baseline, ablation, orthogonalization |
| 2-3 | Attack family: strength alpha, number of directions k, layer band; hook and weight versions. Fill the registry with 20-50 public abliterated checkpoints plus benign fine-tunes | parameterized edits; verified registry |
| 3-5 | Detector battery: weight-diff spectral energy, activation gap along reference directions, logprob test. Plot attack success against AUROC for each setting | the detection-limit frontier (direction A's headline) |
| 4-6 | Susceptibility predictors on clean models: stable rank, cone dimension, probe margin; check they predict the measured drop | direction B's empirical result |
| 6-8 | Detector-aware attacker and the conditional bound under linear readout | evasion boundary; theory section |

Target: SoLaR, ATTRIB or MechInterp workshop at NeurIPS/ICLR for the frontier result. Direction
D (refusal ablation on RMU / Deep Ignorance unlearned checkpoints, scored with WMDP-Cyber)
follows once A and B are running.

## Hardware

- **PC** (24 GB NVIDIA GPU, 64 GB RAM): all GPU work. bf16 fits up to about 9B; Moonlight-16B-A3B
  needs 8-bit.
- **Mac** (M5, 64 GB): editing, analysis of saved activations, plots. It can run the 1-3B models
  and the judge on MPS, and Moonlight in bf16, but keep the main pipeline on CUDA.

See [docs/remote.md](docs/remote.md) for running everything on the PC from the Mac.
