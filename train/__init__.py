"""Training / finetuning scripts.

Designed to run on Lium / RunPod GPU boxes. Local 4060 smoke-tests use
`config/4060_smoke.yaml`; real runs use `lium_h100.yaml` or `lium_8xh100.yaml`.

Pulls model from `zoo`, data from `data`, eval metric from `eval.score`.
"""
