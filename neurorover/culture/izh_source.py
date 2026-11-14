"""Cortical Labs SDK simulator data source backed by the Izhikevich culture model.

Registers the in-silico culture (neurorover.culture.core.Culture) as the CL SDK
simulator's pull source, so code written against the real `cl` API
(`cl.open()`, `neurons.loop()`, `neurons.stim()`, `tick.analysis.spikes`) drives
the model. On a CL1 / Cortical Cloud the same closed-loop code runs unchanged;
it simply does not register this source.

Timing: the culture advances in 1 ms steps = 25 frames at 25 kHz. Millisecond k
covers frames [25k, 25k+25). A step is run only once a read() has covered the
whole millisecond, so every stim committed with a timestamp inside that
millisecond has already been delivered via on_stims() (the producer delivers
stims up to the end of the batch before calling read()). Spikes from step k are
stamped at frame 25k+24 (the last frame of that ms). Stims that arrive late
(timestamp in an already-simulated ms) are applied in the next step.

Usage:
    import cl
    from neurorover.culture.izh_source import register
    register(seed=0, rule="frozen")
    with cl.open() as neurons: ...
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from cl.sim import (
    DataSourceBatch,
    DataSourceSpike,
    DataSourceStim,
    SimulatorDataSource,
    SimulatorDataSourceMetadata,
)

from .core import N_CH, Culture

FPS = 25_000
FRAMES_PER_MS = FPS // 1000  # 25
SETTLE_MS = 1000

# Shared zero waveform + explicit mean: skips a per-spike np.zeros/np.mean in the
# SDK producer (the dominant cost at ~5k spikes/s). Spike waveforms are not modelled.
# SpikeRecord is the producer's internal record type; returning it directly hits
# the producer's no-conversion fast path. Falls back to the public DataSourceSpike.
try:
    from cl._sim._data_buffer import SPIKE_SAMPLES_TOTAL, SpikeRecord as _Spike
except Exception:  # pragma: no cover
    SPIKE_SAMPLES_TOTAL, _Spike = 75, DataSourceSpike
_ZERO_SAMPLES = np.zeros(SPIKE_SAMPLES_TOTAL, np.float32)

FACTORY = "neurorover.culture.izh_source:create_source"
METADATA = SimulatorDataSourceMetadata(
    channel_count=N_CH,
    frames_per_second=FPS,
    duration_frames=None,
    seekable=False,
    realtime_only=False,
    supports_accelerated=True,
)


class CultureSource(SimulatorDataSource):
    def __init__(self, settle_ms: int = SETTLE_MS, emit_frames: bool = False, **culture_kw):
        self._culture_kw = culture_kw
        self._settle_ms = int(settle_ms)
        self._emit_frames = bool(emit_frames)
        self.culture: Culture | None = None
        self._ms = 0                 # next millisecond to simulate
        self._stims: list = []       # heap of (timestamp, seq, channel, current_uA)
        self._seq = itertools.count()
        self.n_stims = 0
        self.n_spikes = 0

    @property
    def metadata(self) -> SimulatorDataSourceMetadata:
        return METADATA

    def open(self) -> None:
        # Built here (not __init__) because the constructor also runs in the parent.
        self.culture = Culture(**self._culture_kw)
        if self._settle_ms:
            self.culture.run(self._settle_ms)  # same settle as bench.run_job
        self.culture.t_ms = 0
        self._ms = 0
        self._stims.clear()

    def close(self) -> None:
        self.culture = None

    def on_stim(self, stim: DataSourceStim) -> None:
        cur = max((abs(c) for c in stim.phase_currents_uA), default=1.0)
        heapq.heappush(self._stims, (int(stim.timestamp), next(self._seq), int(stim.channel), float(cur)))
        self.n_stims += 1

    def on_stims(self, stims) -> None:
        for s in stims:
            self.on_stim(s)

    def read(self, from_timestamp: int, frame_count: int) -> DataSourceBatch:
        c = self.culture
        to_ts = from_timestamp + frame_count
        # Sequential producer; if it ever jumps forward, skip ahead without simulating.
        if self._ms * FRAMES_PER_MS < from_timestamp - FRAMES_PER_MS:
            self._ms = from_timestamp // FRAMES_PER_MS
        spikes: list[DataSourceSpike] = []
        ch_of = c.ch
        zs = _ZERO_SAMPLES
        while (self._ms + 1) * FRAMES_PER_MS <= to_ts:
            ms_end = (self._ms + 1) * FRAMES_PER_MS
            while self._stims and self._stims[0][0] < ms_end:
                _, _, chn, cur = heapq.heappop(self._stims)
                c.stim(chn, cur)
            fired = c.step()
            if fired.size:
                ts = ms_end - 1
                if ts >= from_timestamp:
                    for chn in ch_of[fired]:
                        spikes.append(_Spike(ts, int(chn), 0.0, zs) if _Spike is not DataSourceSpike
                                  else DataSourceSpike(ts, int(chn), zs, 0.0))
            self._ms += 1
        self.n_spikes += len(spikes)
        frames = np.zeros((frame_count, N_CH), np.int16) if self._emit_frames else None
        return DataSourceBatch(frames=frames, spikes=spikes)


def create_source(**config) -> CultureSource:
    return CultureSource(**config)


def register(**config) -> None:
    """Register the culture as the CL SDK simulator source (sim only)."""
    import cl
    cl.sim.set_simulator_data_source(FACTORY, config=config, metadata=METADATA)
