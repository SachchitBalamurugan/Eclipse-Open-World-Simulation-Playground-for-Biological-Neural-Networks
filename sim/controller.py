"""Hand-written baseline: waypoint + proportional heading controller using the SAME noisy, delayed observation
the PPO policy gets (no privileged state).

Plan: if not yet in the target lane, go to a lane-entry waypoint in the headland, then drive along the lane to
the station, turn to the row heading, stop, and (on DEPLOY) command the boom out.
"""
from __future__ import annotations

import math

import numpy as np

from .build_model import ROW_X0


class PController:
    def __init__(self, k_v=1.2, k_w=1.6, v_max=0.8):
        self.k_v, self.k_w, self.v_max = k_v, k_w, v_max

    def reset(self):
        pass

    def __call__(self, obs):
        dxw, dyw = obs[0] * 4, obs[1] * 4               # goal - rover, world/lane frame
        herr = math.atan2(obs[5], obs[6])
        yaw = -herr                                       # goal yaw is 0
        goal_along = obs[7] * 4                           # goal_x - ROW_X0
        deploy = obs[20] > 0
        x_rel = goal_along - dxw                          # rover x - ROW_X0
        dist = math.hypot(dxw, dyw)
        # waypoint selection: stay in the headland until lined up with the lane, then drive down the lane
        lane_tol = 0.12 if x_rel > -0.3 else 0.2
        if abs(dyw) > lane_tol:
            wx, wy = (-0.9 if x_rel > -0.7 else -0.6) - x_rel, dyw if x_rel <= -0.7 else 0.0
        else:
            wx, wy = dxw, dyw
        boom = -1.0
        if dist > 0.15:
            ang = math.atan2(wy, wx)
            e = (ang - yaw + math.pi) % (2 * math.pi) - math.pi
            d = math.hypot(wx, wy)
            v = 0.0 if abs(e) > 0.6 else min(self.v_max, self.k_v * d) * math.cos(e)
            if dist < 0.6:  # slow, careful approach
                v = min(v, 0.35)
            w = self.k_w * e
        else:
            v = 0.0
            w = self.k_w * 1.5 * herr if abs(herr) > 0.06 else 0.0
            if abs(herr) < 0.15 and deploy:
                boom = 1.0
        left = float(np.clip(v - w * 0.5, -1, 1))
        right = float(np.clip(v + w * 0.5, -1, 1))
        return np.array([left, right, boom], np.float32)
