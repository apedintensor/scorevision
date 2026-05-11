"""Production miner serving layer.

`predictor.py` wraps a zoo model + trained head behind a single
`predict_actions(video_path) -> list[FramePrediction]` call.
`main.py` mounts that behind FastAPI `/challenge` (matching turbovision's
private-track miner contract).

Build into the GHCR Docker image referenced by `sv deploy-pt-miner`.
"""
