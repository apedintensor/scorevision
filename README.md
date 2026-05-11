# scorevision

Mining framework for Bittensor **SN44 (Score Vision / TurboVision)** — focused on the **private-track Football Event Detection** skill (20% subnet emission).

Decoupled scaffold: each component (model zoo, data, eval, training, serving) is independent, so swapping a backbone is one line and porting between a 4060 dev box and a Lium / RunPod GPU box is a config flip.

## Project layout

```
.
├── eval/         Local scoring harness — bit-equal to the official validator
├── zoo/          Pretrained video model adapters (VideoMAEv2-*, InternVideo2-*)
├── benchmark/    Zero-shot model comparison runner
├── data/         Dataset adapters (synthetic, SoccerNet, Score samples)
├── train/        Fine-tuning scripts (Lium / RunPod target)
├── serve/        FastAPI miner endpoint + GHCR Dockerfile (private-track deploy)
├── scripts/      One-shot helpers (download models, setup wallet, refresh intel)
├── ops/          Cloud-GPU runbooks
└── tests/        Sanity tests (eval parity, model loading, smoke)
```

## Quickstart

```bash
# 0. Clone + create env
git clone https://github.com/apedintensor/scorevision.git
cd scorevision
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. Clone the read-only reference repo (validator + miner source of truth)
bash scripts/clone_turbovision.sh

# 2. Verify local eval scorer matches the official validator implementation
python -m eval.verify_against_official

# 3. Smoke-test the model zoo (downloads VideoMAE-v1 K400, runs on synthetic clips)
python -m benchmark.run_zero_shot videomae_v1_k400 --max-videos 2
```

## Scoring (private track — what we optimize)

Source of truth: `turbovision/scorevision/validator/central/private_track/scoring.py`.

Faithfully reimplemented in [`eval/scoring.py`](eval/scoring.py) (verified against the upstream impl bit-for-bit on 5000 random trials).

```
score = clamp(0, 1, ( Σ weight·time_decay over matched preds
                    − Σ weight over unmatched preds ) / Σ weight over GT)
```

Key insight: **false positives cost full action weight** (no decay). Breakeven precision is 0.50 when time-decay averages 1.0; higher for high-weight actions (goal=10.9, foul=7.7, save=7.3).

See [`eval/actions.py`](eval/actions.py) for the 15-class action table with weights, tolerances, and `min_score` parameters.

## Model zoo

| Model | HF repo | Params | 4060 (8GB) | Lium H100 |
|-------|---------|-------:|:----------:|:---------:|
| VideoMAE-v1 K400 | `MCG-NJU/videomae-base-finetuned-kinetics` | 86M | ✅ | ✅ |
| VideoMAEv2-Base | `OpenGVLab/VideoMAEv2-Base` | 86M | ✅ | ✅ |
| VideoMAEv2-Large | `OpenGVLab/VideoMAEv2-Large` | 305M | ✅ bf16 | ✅ |
| VideoMAEv2-Huge | `OpenGVLab/VideoMAEv2-Huge` | 633M | ❌ | ✅ |
| VideoMAEv2-giant | `OpenGVLab/VideoMAEv2-giant` | 1B | ❌ | ✅ |
| InternVideo2-Stage2_6B | `OpenGVLab/InternVideo2-Stage2_6B` | 6B | ❌ | ✅ |

Add a new model by adding one entry in `zoo/<adapter>.py` — see `zoo/base.py` for the protocol.

## Status

- ✅ Local eval harness (parity-tested)
- ✅ Scaffold + zoo + benchmark runner
- ✅ VideoMAE-v1 / VideoMAEv2 adapters
- 📋 Real football data adapter (SoccerNet / Score samples)
- 📋 Fine-tuning loop
- 📋 Serving (FastAPI + Dockerfile + sv deploy-pt-miner integration)

## References

- TurboVision repo (validator + miner source): https://github.com/score-technologies/turbovision
- Subnet console: https://console.scorevision.io/
- Manifest CDN: https://turbo.scoredata.me/
- Taostats: https://taostats.io/subnets/44
