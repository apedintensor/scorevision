# SN44 Mining — Project Notes

## Current state (2026-05-11)

- ✅ Old `score-vision` artifacts deleted; clean working dir
- ✅ Active `turbovision` repo cloned at `e:\sn44\turbovision\` (latest 2026-05-08, read-only reference)
- ✅ Local eval scoring harness at `eval/` — bit-equal to official (5000/5000 random trials)
- ✅ 30-day Discord intel report at `discord-intel/2026-05-11.md`
- ✅ Live leaderboard snapshot at `discord-intel/2026-05-11-leaderboard.md`
- ✅ Current manifest at `manifest-2026-05-01.json`
- ✅ Project scaffold laid out — see directory tree below
- 📋 Next: implement zoo/ model adapters + benchmark/ runner

## Directory layout

```
e:/sn44/
├── eval/                  ✅ DONE   Local scoring harness (parity-tested)
├── discord-intel/         ✅ DONE   Intel reports + leaderboard
├── turbovision/           ✅ DONE   Official repo (read-only reference)
├── manifest-2026-05-01.json
├── NOTES.md
│
├── data/                  📋 STARTED  synthetic.py + label_map.py landed; soccernet.py / score_samples.py TODO
├── zoo/                   📋 TODO    Model adapters (VideoMAEv2-*, InternVideo2-*)
├── benchmark/             📋 TODO    Zero-shot comparison runner
├── train/                 📋 TODO    Fine-tuning (Lium target)
├── serve/                 📋 TODO    FastAPI miner endpoint + Dockerfile
├── scripts/               📋 TODO    Download / setup helpers
├── ops/                   📋 TODO    Lium runbooks
└── tests/                 📋 TODO    Eval parity + smoke tests
```

## Strategic bet

**Private track Football Event Detection (20% subnet emission).**

- Current leader 5DyenUmATc… scores **38.8% / target 85%** — 46-point headroom
- Leader earned **$42,432** in ~4 weeks since element launch
- Top 4 cluster (38.8 / 36.8 / 30.4 / 31.7%), rank 5 drops to 5.4% — beatable wall
- Public elements are crowded near targets and capped at 30MB model size

## Architecture summary

| Track | Where model runs | Code visibility | Deploy CLI |
|-------|------------------|-----------------|------------|
| OS / Public | Chutes.ai (TEE pro_6000, ~$1.80/hr) | Public HF repo | `sv deploy-os-miner` |
| **Private** (our target) | **Your GPU + public IP** | Private GHCR Docker | `sv deploy-pt-miner` |

Private track validator flow:
1. Validator POSTs video URL to `your-ip:8000/challenge`
2. Your container downloads, runs predictor, returns `list[FramePrediction]`
3. Validator scores via `_legacy_score_predictions` (reproduced in `eval/scoring.py`)
4. Anti-copy + tiebreak: scores too similar to leader on ≥5 challenges → leader wins (earlier commit block)
5. Spot-check: Score pulls your Docker image + reruns on their hardware, compares to your live response

## Scoring math (private track, see `eval/scoring.py`)

```
score = clamp(0, 1, (sum(weight * time_decay for matched) - sum(weight for unmatched_preds)) / sum(weight for gt))
```

Critical insight: **false positives cost full action weight**, no decay. Breakeven precision is 0.50 at avg_decay=1.0. Higher-weight actions (goal=10.9, foul=7.7, save=7.3) require more conservative thresholds.

## Phase plan

| Phase | Where | Status |
|-------|-------|--------|
| 1. Project scaffold + eval harness | local 4060 | ✅ done |
| 2. Model zoo + zero-shot benchmark | local 4060 (Base/Large) | 🔄 next |
| 3. Training pipeline scaffold | local 4060 (smoke test only) | 📋 |
| 4. Real fine-tuning | Lium H100 / RunPod | 📋 |
| 5. Production deploy | Cloud GPU box + public IP | 📋 |

## Model zoo plan (zoo/)

| Model | HF repo | Params | 4060 (8GB) | Lium H100 |
|-------|---------|-------:|:----------:|:---------:|
| VideoMAEv2-Base | `OpenGVLab/VideoMAEv2-Base` | 86M | ✅ | ✅ |
| VideoMAEv2-Large | `OpenGVLab/VideoMAEv2-Large` | 305M | ✅ bf16 | ✅ |
| VideoMAEv2-Huge | `OpenGVLab/VideoMAEv2-Huge` | 633M | ❌ | ✅ |
| VideoMAEv2-giant ⭐ | `OpenGVLab/VideoMAEv2-giant` | 1B | ❌ | ✅ |
| InternVideo2-Stage2_6B 🏆 | `OpenGVLab/InternVideo2-Stage2_6B` | 6B | ❌ | ✅ |
| InternVideo2.5-Chat-8B (VLM) | `OpenGVLab/InternVideo2_5_Chat_8B` | 8B | ❌ | ✅ |
| InternVideo2-CLIP-S | `OpenGVLab/InternVideo2_CLIP_S` | small | ✅ | ✅ |

⭐ = engineering sweet spot (per user's reference table)
🏆 = SOTA candidate

## References

- Turbovision repo: https://github.com/score-technologies/turbovision (read-only reference at `turbovision/`)
- Console: https://console.scorevision.io/
- Manifest CDN: https://turbo.scoredata.me/
- Live miner reputation: https://console.scorevision.io/miners
- Score Discord channel: `1271486854830755981` in Bittensor server
