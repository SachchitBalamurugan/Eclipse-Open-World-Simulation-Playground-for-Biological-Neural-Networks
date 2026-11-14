"""MEA targets: physical layout, unavailable electrodes, stimulation units and hardware limits.

The simulated culture is an 8 x 8 grid of 64 electrodes. Sim channel ch sits at grid (gx, gy) =
(ch % 8, ch // 8). Every target maps that grid onto its own electrodes; roles that land on an
electrode the hardware cannot use (ground / reference / absent corner) are moved to the nearest
free usable electrode ("remap"), and that is recorded in electrode_map.csv.

Numbers marked VERIFY are taken from vendor documentation / literature as best known and MUST be
checked against the lab's own hardware manual and ethics/biosafety approvals before use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

GRID = 8


@dataclass
class Waveform:
    mode: str                 # "current" | "voltage"
    unit: str                 # "uA" | "mV"
    amplitude: float          # per-phase amplitude for encoder (sensory) pulses
    teach_amplitude: float    # per-phase amplitude for paired-stimulation (teach) pulses
    polarity: str             # "cathodic-first" | "anodic-first"
    phase_width_us: int
    interphase_us: int
    note: str = ""


@dataclass
class Target:
    id: str
    name: str
    vendor: str
    status: str               # "tested-mock" | "sdk-untested" | "stub-needs-vendor-sdk"
    n_electrodes: int
    pitch_um: float
    electrode_diameter_um: float
    layout: str
    waveform: Waveform
    limits: dict              # hardware/SDK limits (hard)
    unavailable: dict = field(default_factory=dict)   # sim-grid channel -> reason
    max_stim_electrodes: int | None = None
    format_notes: list = field(default_factory=list)
    sdk: str = ""
    region_block: int = 0      # maxwell: HD electrodes per region side

    # ---- mapping (sim grid -> target electrode) ----
    def electrode_id(self, ch: int):
        gx, gy = ch % GRID, ch // GRID
        if self.id == "mcs":
            return int(f"{gx + 1}{gy + 1}")       # MCS label "<column><row>", 1-based
        return ch                                  # generic / cl1 / maxwell-region: same index

    def label(self, ch: int) -> str:
        gx, gy = ch % GRID, ch // GRID
        if self.id == "mcs":
            return f"{gx + 1}{gy + 1}"
        if self.id == "maxwell":
            return f"R{gy}{gx}"
        if self.id == "cl1":
            return f"ch{ch}"
        return f"E{ch:02d}"

    def xy_um(self, ch: int):
        gx, gy = ch % GRID, ch // GRID
        return gx * self.pitch_um, gy * self.pitch_um

    def maxwell_block(self, ch: int):
        """MaxOne: 220 x 120 electrodes, 17.5 um pitch, index = row * 220 + col. Each sim channel is a
        region of block x block HD electrodes; region pitch 12 electrodes (210 um), grid centred."""
        gx, gy = ch % GRID, ch // GRID
        b, rp = self.region_block, 12
        col0 = (220 - (7 * rp + b)) // 2 + gx * rp
        row0 = (120 - (7 * rp + b)) // 2 + gy * rp
        rec = [(row0 + r) * 220 + (col0 + c) for r in range(b) for c in range(b)]
        stim = (row0 + b // 2) * 220 + (col0 + b // 2)
        return rec, stim

    def as_dict(self) -> dict:
        return dict(id=self.id, name=self.name, vendor=self.vendor, status=self.status,
                    n_electrodes=self.n_electrodes, pitch_um=self.pitch_um,
                    electrode_diameter_um=self.electrode_diameter_um, layout=self.layout,
                    max_stim_electrodes=self.max_stim_electrodes, limits=self.limits, sdk=self.sdk,
                    unavailable={str(k): v for k, v in self.unavailable.items()})


_CURRENT_DEFAULT = Waveform("current", "uA", 1.0, 1.0, "cathodic-first", 160, 0,
                            "CL-style 160 us/phase, 1 uA (same as neurorover/run_cl.py). VERIFY.")

TARGETS: dict[str, Target] = {
    "generic": Target(
        id="generic", name="Generic 8x8 MEA (hardware-agnostic)", vendor="any", status="tested-mock",
        n_electrodes=64, pitch_um=200.0, electrode_diameter_um=30.0,
        layout="8x8 grid, electrode id = sim channel, x = id % 8, y = id // 8",
        waveform=_CURRENT_DEFAULT,
        limits=dict(max_amplitude=3.0, amplitude_unit="uA", max_charge_per_phase_nC=1.0,
                    phase_width_step_us=20, max_burst_hz=200, min_lead_time_us=80),
        sdk="none -- implement MEABackend for your rig (bnn_platform/transfer/runtime.py)",
        format_notes=["Electrode ids are the simulator's channel numbers; map them to your rig's channel "
                      "numbering in your MEABackend."]),
    "cl1": Target(
        id="cl1", name="Cortical Labs CL1 / Cortical Cloud (cl API)", vendor="Cortical Labs",
        status="sdk-untested", n_electrodes=64, pitch_um=200.0, electrode_diameter_um=30.0,
        layout="64 channels (cl.ChannelSet indices 0-63); common MEA layout is column-major "
               "(channel = column*8 + row), i.e. the transpose of the sim grid -- distances are preserved",
        waveform=_CURRENT_DEFAULT,
        limits=dict(max_amplitude=3.0, amplitude_unit="uA", max_total_charge_nC=3.0,
                    max_charge_per_phase_nC=1.0, phase_width_step_us=20, max_burst_hz=200,
                    min_lead_time_us=80, source="cl-sdk 1.0.0 StimDesign/BurstDesign constraints"),
        unavailable={0: "ground", 7: "ground", 56: "ground", 63: "ground", 4: "reference"},
        sdk="cl (Cortical Labs SDK; `pip install cl-sdk` provides the simulator, hardware via CL1/Cloud)",
        format_notes=["Ground channels 0, 7, 56, 63 and reference channel 4 taken from "
                      "cl.analysis common MEA layout (cl-sdk 1.0.0). VERIFY on your device.",
                      "stim_schedule maps 1:1 to cl.StimDesign(phase_us, -I, phase_us, +I) one-shot "
                      "StimPlans scheduled on the 25 kHz device clock."]),
    "mcs": Target(
        id="mcs", name="Multi Channel Systems MEA2100 + 60MEA200/30iR", vendor="Multi Channel Systems",
        status="stub-needs-vendor-sdk", n_electrodes=60, pitch_um=200.0, electrode_diameter_um=30.0,
        layout="8x8 grid without corners (60 electrodes), label '<column><row>' 1-based (12..87); "
               "electrode 15 is the internal reference on 60MEA200/30iR",
        waveform=Waveform("voltage", "mV", 500.0, 500.0, "anodic-first", 400, 0,
                          "Voltage-controlled positive-first biphasic, 400 us/phase, 500 mV: inside the "
                          "range reported effective by Wagenaar et al. 2004 (J Neurosci Methods) for "
                          "TiN 60MEA. VERIFY with your STG and electrode type."),
        limits=dict(max_amplitude=900.0, amplitude_unit="mV", max_charge_per_phase_nC=None,
                    phase_width_step_us=20, max_burst_hz=200, min_lead_time_us=0),
        unavailable={0: "absent (corner 11)", 7: "absent (corner 81)", 56: "absent (corner 18)",
                     63: "absent (corner 88)", 32: "internal reference (15)"},
        sdk="McsUsbNet (.NET; Python via pythonnet) for MEA2100 data + STG stimulus download; "
            "real-time closed loop through the MEA2100 DSP / Multi Channel Experimenter feedback",
        format_notes=["electrode_id is the integer of the MCS label (e.g. 12 = column 1, row 2).",
                      "Hardware channel index differs per headstage/software; map labels -> HW "
                      "channels with your MC_Experimenter channel layout file."]),
    "maxwell": Target(
        id="maxwell", name="MaxWell Biosystems MaxOne / MaxTwo HD-MEA", vendor="MaxWell Biosystems",
        status="stub-needs-vendor-sdk", n_electrodes=26400, pitch_um=210.0, electrode_diameter_um=9.3,
        layout="220 x 120 electrodes, 17.5 um pitch (index = row*220 + col). Each sim channel = one "
               "region of 4x4 HD electrodes (16 recording electrodes, 1 stimulation electrode), 8x8 "
               "regions at 210 um pitch centred on the array -> 1024 routed recording electrodes",
        waveform=Waveform("voltage", "mV", 75.0, 150.0, "anodic-first", 200, 0,
                          "75 mV sensory / 150 mV teach follow the DishBrain MaxOne protocol "
                          "(Kagan et al. 2022, Neuron) as best known; phase width is a placeholder. "
                          "MaxLab amplitudes are DAC-code based. VERIFY."),
        limits=dict(max_amplitude=300.0, amplitude_unit="mV", max_charge_per_phase_nC=None,
                    phase_width_step_us=50, max_burst_hz=200, max_routed_electrodes=1024,
                    max_stimulation_units=32, min_lead_time_us=0),
        max_stim_electrodes=32, region_block=4,
        sdk="maxlab Python API (MaxLab Live) for configuration/routing/stimulation sequences; "
            "low-latency closed loop via the MaxLab Live real-time (C++) spike stream",
        format_notes=["electrode_id is the region index (sim channel); hw_record_electrodes / "
                      "hw_stim_electrode are MaxOne electrode indices.",
                      "Each stimulation region needs its own stimulation unit (32 per chip): stimulated "
                      "roles must be <= 32.",
                      "Spikes from the 16 electrodes of a region are summed; neighbouring HD electrodes "
                      "see the same neuron, so deduplicate (e.g. spike sorting or max-electrode) first."]),
}
TARGETS["cl1"].max_stim_electrodes = None
TARGETS["generic"].max_stim_electrodes = None
TARGETS["mcs"].max_stim_electrodes = None


def get_target(tid: str) -> Target:
    if tid not in TARGETS:
        raise KeyError(f"unknown target {tid!r}; one of {', '.join(TARGETS)}")
    return TARGETS[tid]


def remap_roles(roles: dict, target: Target) -> tuple[dict, list]:
    """roles: trained sim channel -> role dict. Returns (physical sim channel -> role dict with
    'trained_sim_channel', list of change records). Roles on unavailable electrodes move to the
    nearest free usable electrode (ties: same row first, then lower channel)."""
    used = set(roles)
    out, changes = {}, []
    for ch in sorted(roles):
        if ch not in target.unavailable:
            out[ch] = dict(roles[ch], trained_sim_channel=ch)
    for ch in sorted(roles):
        if ch in target.unavailable:
            gx, gy = ch % GRID, ch // GRID
            free = [c for c in range(GRID * GRID)
                    if c not in target.unavailable and c not in out and c not in used]
            if not free:
                raise ValueError("no free electrode left to remap onto")
            best = min(free, key=lambda c: ((c % GRID - gx) ** 2 + (c // GRID - gy) ** 2,
                                            c // GRID != gy, c))
            out[best] = dict(roles[ch], trained_sim_channel=ch)
            changes.append(dict(role=roles[ch]["role"], name=roles[ch].get("name"), from_sim_channel=ch,
                                to_sim_channel=best, reason=target.unavailable[ch],
                                from_electrode=target.electrode_id(ch), to_electrode=target.electrode_id(best)))
    return out, changes
