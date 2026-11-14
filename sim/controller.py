"""Hand-written baseline: pure-pursuit path follower + proportional heading/stop logic, using the SAME noisy,
delayed observation vector the PPO policy gets (no privileged state).

Path (in the goal frame, x along the rows): current position -> pre-entry point 1.4 m before the row start in the
headland, on the lane line -> lane entry -> station. Pure pursuit with a 0.6 m look-ahead; spin in place when the
look-ahead point is more than ~50 deg off the nose. At the station: turn to the row heading, stop, and on DEPLOY
command the boom out.
"""
from __future__ import annotations

import math

import numpy as np


def _closest_ahead(path, p, look):
    """Point `look` metres beyond the projection of p on the polyline."""
    best, bi, bt = 1e9, 0, 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a; L2 = float(ab @ ab) + 1e-9
        t = float(np.clip((p - a) @ ab / L2, 0, 1))
        d = float(np.linalg.norm(a + t * ab - p))
        if d < best:
            best, bi, bt = d, i, t
    rem = look
    i, t = bi, bt
    while i < len(path) - 1:
        a, b = path[i], path[i + 1]
        seg = float(np.linalg.norm(b - a))
        left = (1 - t) * seg
        if rem <= left:
            return a + (t + rem / max(seg, 1e-9)) * (b - a), best
        rem -= left; i += 1; t = 0.0
    return path[-1], best


class PController:
    def __init__(self, look=0.6, k_w=3.0, v_max=0.85, spin=0.9):
        self.look, self.k_w, self.v_max, self.spin = look, k_w, v_max, spin
        self.reset()

    def reset(self):
        self.path = None
        self.turn_sign = 1.0
        self.docked = False

    def __call__(self, obs):
        dxw, dyw = obs[0] * 4, obs[1] * 4               # goal - rover (world == lane frame)
        herr = math.atan2(obs[5], obs[6])
        yaw = -herr                                       # station heading is 0
        goal_along = obs[7] * 4                           # station x - row start x
        deploy = obs[20] > 0
        p = np.array([-dxw, -dyw])                        # rover position in the goal frame
        dist = math.hypot(dxw, dyw)
        if self.path is None:
            pts = [p]
            if p[0] > -goal_along - 0.3 and abs(p[1]) > 0.15:   # started inside the rows region: back out first
                pts.append(np.array([-goal_along - 1.0, p[1]]))
            pts += [np.array([-goal_along - 1.4, 0.0]), np.array([-goal_along - 0.4, 0.0]), np.array([0.0, 0.0])]
            self.path = pts
        boom = -1.0
        if self.docked and dist > 0.35:
            self.docked = False
        if dist < 0.16:
            self.docked = True
        if not self.docked:
            tgt, cte = _closest_ahead(self.path, p, self.look)
            ang = math.atan2(tgt[1] - p[1], tgt[0] - p[0])
            e = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(e) > 2.6:
                e = abs(e) * self.turn_sign
            self.turn_sign = 1.0 if e >= 0 else -1.0
            if abs(e) > self.spin:
                v = 0.0
            else:
                v = self.v_max * math.cos(e)
                if dist < 0.8:
                    v = min(v, 0.25 + 0.5 * dist)
            w = float(np.clip(self.k_w * e, -1.2, 1.2))
        else:
            v = 0.0
            w = float(np.sign(herr) * np.clip(abs(self.k_w * herr), 0.9, 1.6)) if abs(herr) > 0.06 else 0.0
            if abs(herr) < 0.15 and deploy:
                boom = 1.0
        left = float(np.clip(v - 0.5 * w, -1, 1))
        right = float(np.clip(v + 0.5 * w, -1, 1))
        return np.array([left, right, boom], np.float32)
