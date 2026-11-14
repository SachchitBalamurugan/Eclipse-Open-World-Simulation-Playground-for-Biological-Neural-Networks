"""Physical parameters of the frost-guard rover, one number per line with its source.

Sources (see sim/README.md for URLs):
  BOM   = hardware/BOM.md in this repo (real-parts bill of materials; its (est.) marks carried over)
  POL   = Pololu #4753 product page (50:1 37Dx70L 12 V gearmotor, 64 CPR encoder)
  ACT   = Actuonix L16 datasheet (L16-100-35-12-P), numbers as listed in the BOM
  SHT   = Sensirion SHT45 datasheet (+-0.1 C typ)
  UBX   = u-blox NEO-M9N product summary (~1.5 m CEP standalone)
  DUCK  = Open Duck Playground (apirrone/Open_Duck_Playground) randomize.py / joystick.py
  EST   = our own engineering estimate, not from a datasheet
"""
import math

# ------------------------------------------------------------------ drive train
WHEEL_RADIUS = 0.060            # m      BOM #8 Dagu Wild Thumper 120 x 60 mm wheel
WHEEL_WIDTH = 0.060             # m      BOM #8
WHEEL_MASS = 0.20               # kg     BOM #8 (est.)
MOTOR_MASS = 0.22               # kg     BOM #5 (est.)
GEAR_RATIO = 50                 # -      POL 50:1
STALL_TORQUE = 2.06             # N*m    POL 21 kg*cm @ 12 V (Pololu: extrapolated value)
CONT_TORQUE = 0.98              # N*m    POL recommended continuous load limit 10 kg*cm
NO_LOAD_RPM = 200.0             # rpm    POL @ 12 V
NO_LOAD_SPEED = NO_LOAD_RPM / 60 * 2 * math.pi   # rad/s at the wheel = 20.9
STALL_CURRENT = 5.5             # A      POL @ 12 V (extrapolated)
NO_LOAD_CURRENT = 0.20          # A      POL 200 mA (BOM lists ~0.15 A est.)
ENCODER_CPR_OUT = 64 * GEAR_RATIO   # counts/rev at output = 3200   POL
DRIVER_EFF = 0.95               # -      EST Cytron MDD10A MOSFET H-bridge loss (BOM #9)
TOP_SPEED = NO_LOAD_SPEED * WHEEL_RADIUS         # 1.26 m/s, BOM derived

# ------------------------------------------------------------------ power
BATT_VOLTAGE = 12.8             # V      BOM #12 Bioenno BLF-1220A LiFePO4
BATT_CAPACITY_AH = 20.0         # Ah     BOM #12
IDLE_CURRENT = 0.6              # A      BOM derived (Pi 5 + sensors @ 12.8 V, est.)

# ------------------------------------------------------------------ chassis geometry
DECK_L, DECK_W, DECK_T = 0.60, 0.40, 0.003   # m  BOM #18 3 mm aluminium 600 x 400
WHEELBASE = 0.44                # m      EST (wheel centres inside 600 mm rails)
TRACK = 0.40 + WHEEL_WIDTH      # m      EST wheel centres just outside the 400 mm deck
RIDE_Z = 0.035                  # m      EST deck height above the wheel axle
MAST_H = 0.80                   # m      BOM #17 800 mm 2020 mast

# ------------------------------------------------------------------ masses (BOM mass column)
M_DECK = 1.9                    # kg     BOM #18 (est.)
M_RAILS = 1.2                   # kg     BOM #17 1.8 kg set, minus the mast share (EST split)
M_MAST = 0.6                    # kg     BOM #17 share of 800 mm extrusion (EST split)
M_BATTERY = 2.5                 # kg     BOM #12 (est.)
M_ENCLOSURE = 0.5 + 0.05 + 0.08 + 0.3   # kg BOM #16 enclosure + #1 Pi 5 + #9 2x driver + #21 wiring
M_SHIELD = 0.6 + 0.01           # kg     BOM #3 Davis 7714 shield (est.) + #2 SHT45
M_GNSS = 0.01 + 0.17            # kg     BOM #14 NEO-M9N + #15 ANN-MB antenna (est.)
M_ACTUATOR = 0.06               # kg     BOM #10 L16 (est.)
M_COVER_CLOTH = 0.8             # kg     BOM #19 Agribon AG-30 roll (est.)
M_TUBE_ARM = 0.4                # kg     BOM #20 PVC roll tube + 2020 arm (est.)

# ------------------------------------------------------------------ cover deploy (actuator + boom)
ACT_STROKE = 0.100              # m      ACT L16-100: 100 mm stroke
ACT_SPEED = 0.032               # m/s    ACT 35:1 gearing, no-load 32 mm/s
ACT_FORCE = 50.0                # N      ACT 35:1 peak lift force
ACT_CURRENT = 0.65              # A      EST L16 12 V draw near peak force (datasheet stall ~650 mA, not re-verified)
BOOM_LEN = 0.45                 # m      EST 2020 arm length (BOM #20 says 500 mm tube)
LEVER = ACT_STROKE / (math.pi / 2)   # m  EST crank radius so the 100 mm stroke swings the boom 90 deg = 0.064
BOOM_MAX_TORQUE = ACT_FORCE * LEVER  # N*m 3.2, what the actuator can hold
BOOM_SPRING = 3.0               # N*m/rad EST counterbalance torsion spring NOT IN BOM (see README: without it
                                #         the 50 N L16 cannot lift the loaded boom past ~35 deg in sim)
N_PANELS = 4                    # -      EST cloth discretisation (hinged panel chain)
PANEL_LEN = 0.20                # m      EST lateral length per panel (0.8 m drape)
PANEL_W = 0.60                  # m      EST cloth width along the row

# ------------------------------------------------------------------ sensors
SHT45_NOISE = 0.1               # degC   SHT
GNSS_CEP_STANDALONE = 1.5       # m      UBX NEO-M9N; too coarse for 0.45 m lanes -> sim assumes RTK/row
GNSS_NOISE = (0.02, 0.08)       # m sigma EST RTK-grade (e.g. u-blox ZED-F9P, not in BOM) per-episode range
GYRO_NOISE = 0.1                # rad/s  DUCK noise scale gyro=0.1
ACCEL_NOISE = 0.05              # (g-normalised) DUCK accelerometer=0.05
HEADING_NOISE = 0.05            # rad    EST magnetometer/dual-antenna heading
ENC_NOISE = 0.3                 # rad/s  EST encoder quantisation + noise at 20 Hz

# ------------------------------------------------------------------ control / sim timing
SIM_DT = 0.005                  # s      EST (DUCK uses 0.002 for a biped; wheels are stiffer-tolerant)
CTRL_DT = 0.05                  # s      EST 20 Hz control on the Pi 5 (DUCK uses 50 Hz)
FILTER_CUTOFF_HZ = 4.0          # Hz     EST first-order low-pass on actions (DUCK LowPassActionFilter idea)
EPISODE_S = 25.0                # s      task time limit

# ------------------------------------------------------------------ domain randomisation (per episode)
DR = dict(
    friction=(0.5, 1.0),        # DUCK floor friction U(0.5, 1.0)
    mass_scale=(0.9, 1.1),      # DUCK body mass *U(0.9, 1.1)
    payload_delta=(-0.5, 0.5),  # kg EST battery/payload mass change (DUCK torso +-0.1 kg, scaled up)
    com_shift=(-0.03, 0.03),    # m EST payload CoM shift (DUCK dpos U(-0.05, 0.05))
    motor_strength=(0.85, 1.1), # DUCK actuator gain *U(0.9, 1.1), widened for battery sag (EST)
    noise_level=(0.5, 1.5),     # EST multiplier on all sensor noise
    obs_delay=(0, 2),           # control steps; DUCK imu_max_delay=3 at 50 Hz (~60 ms) -> 0-100 ms here
    act_delay=(0, 1),           # control steps; DUCK action_max_delay=3
    terrain=(0.0, 0.03),        # m EST bump amplitude
    wind=(0.0, 1.5),            # N EST gust force per unrolled cover panel (~4-5 m/s gust on 0.12 m2; DUCK pushes 0.1-1.0)
)
