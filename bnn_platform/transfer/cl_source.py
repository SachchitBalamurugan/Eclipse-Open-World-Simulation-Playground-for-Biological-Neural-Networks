"""cl-sdk simulator data source that serves an exported (trained) culture, so CL1Backend can be
exercised end-to-end through the real Cortical Labs `cl` API without hardware. Simulation only:
on a CL1 / Cortical Cloud nothing is registered and the living culture answers instead."""
from __future__ import annotations

import pathlib
import pickle
import sys

from neurorover.culture.izh_source import METADATA, CultureSource

FACTORY = "bnn_platform.transfer.cl_source:create_source"
ROOT = pathlib.Path(__file__).resolve().parents[2]


class PickledCultureSource(CultureSource):
    def __init__(self, culture_pkl: str | None = None, rule: str = "frozen", seed: int = 0, **kw):
        super().__init__(settle_ms=0, seed=seed, rule="frozen")
        self._pkl = culture_pkl
        self._rule = rule

    def open(self) -> None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        if self._pkl:
            with open(self._pkl, "rb") as fh:
                obj = pickle.load(fh)
            self.culture = obj.culture if hasattr(obj, "culture") else obj
            self.culture.rule = self._rule
        else:
            super().open()
        self.culture.t_ms = 0
        self._ms = 0
        self._stims.clear()


def create_source(**config) -> PickledCultureSource:
    return PickledCultureSource(**config)


def register(**config) -> None:
    import cl
    cl.sim.set_simulator_data_source(FACTORY, config=config, metadata=METADATA)
