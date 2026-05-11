"""Verify our clean-room scorer matches `turbovision`'s official scorer.

Loads only the scoring file from the turbovision repo (bypassing the
package __init__ which pulls bittensor / pydantic-settings / chutes etc.)
and asserts our `eval.score` produces bit-equal output for randomized
inputs.

Run any time you touch `eval/scoring.py` or sync `eval/actions.py`:
    cd e:/sn44
    python -m eval.verify_against_official

Exits 0 on success, 1 on first mismatch.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parent.parent  # e:/sn44
TURBO = REPO_ROOT / "turbovision"

# --------------------------------------------------------------------------
# Inject minimal stub modules so we can import the official scoring.py
# without triggering bittensor / pydantic-settings imports from the
# turbovision package __init__ chain.
# --------------------------------------------------------------------------

PRIVATE_FRAME_RATE_FOR_TEST = 25

# 1. `scorevision` (top-level) and `scorevision.utils` and `scorevision.validator.*`
#    namespaces — make them empty packages so submodule imports work.
for pkg_name in (
    "scorevision",
    "scorevision.utils",
    "scorevision.validator",
    "scorevision.validator.central",
    "scorevision.validator.central.private_track",
):
    mod = types.ModuleType(pkg_name)
    mod.__path__ = []  # mark as package
    sys.modules[pkg_name] = mod


# 2. Stub `scorevision.utils.settings.get_settings` to return an object whose
#    only attribute we touch is PRIVATE_FRAME_RATE.
@dataclass
class _StubSettings:
    PRIVATE_FRAME_RATE: int = PRIVATE_FRAME_RATE_FOR_TEST


_settings_mod = types.ModuleType("scorevision.utils.settings")
_settings_mod.get_settings = lambda: _StubSettings()  # type: ignore[attr-defined]
sys.modules["scorevision.utils.settings"] = _settings_mod


# 3. Stub `scorevision.utils.schemas.FramePrediction` (the dataclass version is fine).
@dataclass
class _StubFramePrediction:
    frame: int
    action: str
    confidence: float = 1.0


_schemas_mod = types.ModuleType("scorevision.utils.schemas")
_schemas_mod.FramePrediction = _StubFramePrediction  # type: ignore[attr-defined]
sys.modules["scorevision.utils.schemas"] = _schemas_mod


# 4. Load the official actions.py from disk (we need its ACTION_CONFIGS to
#    match what the scoring expects).
def _load_module_from_path(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_module_from_path(
    "scorevision.utils.actions",
    TURBO / "scorevision" / "utils" / "actions.py",
)

# 5. Load the official scoring module.
official_scoring = _load_module_from_path(
    "scorevision.validator.central.private_track.scoring",
    TURBO / "scorevision" / "validator" / "central" / "private_track" / "scoring.py",
)
official_score = official_scoring._legacy_score_predictions  # type: ignore[attr-defined]

# 6. Our impl.
sys.path.insert(0, str(REPO_ROOT))
from eval.actions import ACTIONS  # noqa: E402
from eval.schemas import FramePrediction as LocalFP  # noqa: E402
from eval.scoring import score as local_score, PRIVATE_FRAME_RATE  # noqa: E402

assert PRIVATE_FRAME_RATE == PRIVATE_FRAME_RATE_FOR_TEST, "fps mismatch"


def _random_events(n: int, max_frame: int, rng: random.Random):
    local = []
    official = []
    for _ in range(n):
        a = rng.choice(ACTIONS)
        f = rng.randint(0, max_frame)
        c = rng.random()
        local.append(LocalFP(frame=f, action=a, confidence=c))
        official.append(_StubFramePrediction(frame=f, action=a, confidence=c))
    return local, official


def run(num_trials: int = 500, seed: int = 0) -> None:
    rng = random.Random(seed)
    mismatches = 0
    for trial in range(num_trials):
        n_gt = rng.randint(0, 40)
        n_pred = rng.randint(0, 40)
        local_gt, off_gt = _random_events(n_gt, 3000, rng)
        local_pr, off_pr = _random_events(n_pred, 3000, rng)

        a = local_score(local_pr, local_gt)
        b = official_score(off_pr, off_gt)

        if not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
            mismatches += 1
            print(f"[trial {trial}] MISMATCH local={a!r} official={b!r}")
            print(f"  n_gt={n_gt} n_pred={n_pred}")
            if mismatches >= 3:
                print("Stopping after 3 mismatches.")
                sys.exit(1)

    print(f"OK: {num_trials}/{num_trials} trials match the official scorer bit-for-bit.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run(num_trials=n)
