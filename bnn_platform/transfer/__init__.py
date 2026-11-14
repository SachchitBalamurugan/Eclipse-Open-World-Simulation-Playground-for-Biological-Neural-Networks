"""Sim-to-real transfer: package a culture-trained robot controller as a closed-loop protocol for a
living neuronal culture on a microelectrode array (MEA), and run that protocol.

What transfers: the *protocol* -- electrode roles, observation->stimulation encoding, spike->action
decoding (as an initial guess that is refit during calibration), the paired-stimulation training
schedule, timing, and safety limits. What does NOT transfer: the simulated synaptic weights. A
living culture has its own connectivity; it must be calibrated and trained in the dish.

    python -m bnn_platform.transfer targets
    python -m bnn_platform.transfer export --run user_runs/<id> [--target generic|cl1|mcs|maxwell]
    python -m bnn_platform.transfer validate --bundle <dir>
    python -m bnn_platform.transfer run --bundle <dir> --backend mock [--episodes 3]

See docs/SIM_TO_REAL.md.
"""
SCHEMA_VERSION = "bnn-transfer/1.0"
BIOLOGY_OF_SOURCE = "simulated"
