"""bnn_platform: one CLI and one registry for training robots with a simulated BNN controller.

The "BNN" here is the simulated Izhikevich culture in neurorover/culture/ (no living tissue).
Robots register themselves with register_robot(); see bnn_platform/robots/template.py.
"""
from .registry import BNNAdapter, RobotSpec, get_robot, list_robots, register_robot, run_trial

__all__ = ["BNNAdapter", "RobotSpec", "get_robot", "list_robots", "register_robot", "run_trial"]
