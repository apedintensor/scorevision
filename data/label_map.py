"""SoccerNet-v2 action labels -> Score's 15-class action set.

SoccerNet-v2 has 17 categories; Score's private track has 15. Mapping
covers ~6 categories cleanly. The rest of Score's classes
(pass / pass_received / take_on / recovery / aerial_duel / save / tackle /
interception / block) are NOT directly available in SoccerNet-v2 — they
will need:
  - SoccerNet-v3 / SN-GSR (richer event labels), OR
  - Custom annotation, OR
  - Pseudo-labelling via a strong existing model + manual verification

For pipeline smoke testing, the coarse mapping below is enough to verify
the data->model->eval flow.
"""

from __future__ import annotations

# SoccerNet-v2 label string -> Score action key (None = drop, no mapping yet).
SOCCERNET_TO_SCORE: dict[str, str | None] = {
    "Ball out of play": "ball_out_of_play",
    "Throw-in": "ball_out_of_play",
    "Foul": "foul",
    "Indirect free-kick": None,
    "Clearance": "clearance",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Corner": "ball_out_of_play",
    "Substitution": "substitution",
    "Kick-off": None,
    "Yellow card": None,
    "Offside": None,
    "Direct free-kick": None,
    "Yellow->red card": None,
    "Goal": "goal",
    "Penalty": "shot",
    "Red card": None,
}
