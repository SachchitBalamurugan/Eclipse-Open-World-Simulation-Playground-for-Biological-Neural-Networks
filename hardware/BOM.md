# NeuroRover frost-guard rover - bill of materials

A small skid-steer field rover that drives to a crop row at dusk, reads air and leaf temperature, and (if the
culture votes DEPLOY) unrolls a frost cloth over a hoop frame. All parts are currently sold off-the-shelf items.
Prices are approximate single-unit USD retail (late 2026); **(est.)** marks numbers that are estimates rather than
datasheet or list values. Masses marked (est.) are not from a datasheet.

The in-app 3D rover (`app/parts.js`) mirrors this table: same parts, same dimensions, same physics numbers.

| # | Part | Model / part no. | Key specs | Qty | Unit $ | Line $ | Mass |
|---|------|------------------|-----------|----:|-------:|-------:|-----:|
| 1 | Controller | Raspberry Pi 5, 8 GB | BCM2712 4x Cortex-A76 2.4 GHz, Wi-Fi 5, 5 V / 5 A USB-C | 1 | 80 | 80 | 0.05 kg |
| 2 | Air temp / RH sensor | Sensirion SHT45 (Adafruit breakout #5665) | ±0.1 °C typ accuracy, ±1.0 %RH, I2C | 1 | 12.50 | 12.50 | <0.01 kg |
| 3 | Radiation shield for SHT45 | Davis Instruments 7714 | passive multi-plate shield, mast mount | 1 | 90 (est.) | 90 | 0.6 kg (est.) |
| 4 | Soil / leaf probe | DS18B20 waterproof probe (Adafruit #381) | ±0.5 °C (-10..+85 °C), 1-Wire, stainless tip | 1 | 9.95 | 9.95 | 0.03 kg |
| 5 | Drive motors | Pololu #4753, 50:1 37Dx70L mm 12 V, 64 CPR encoder | 200 rpm no-load, 21 kg·cm (2.06 N·m) stall (extrapolated), 5.5 A stall @ 12 V | 4 | 57 (est.) | 228 | 0.22 kg (est.) |
| 6 | Motor brackets | Pololu 37D bracket pair #1084 | stamped steel L-bracket | 2 | 8 | 16 | - |
| 7 | Wheel hubs | Pololu universal aluminium hub, 6 mm shaft #1083 (pair) | M3 tapped | 2 | 9 | 18 | - |
| 8 | Wheels | Dagu Wild Thumper wheel 120 x 60 mm | rubber off-road tyre, r = 0.060 m | 4 | 12 (est.) | 48 | 0.20 kg (est.) |
| 9 | Motor drivers | Cytron MDD10A dual-channel | 5-30 V, 10 A continuous / 30 A peak per channel, PWM+DIR | 2 | 22 (est.) | 44 | 0.04 kg |
| 10 | Cover deploy actuator | Actuonix L16-100-35-12-P | 100 mm stroke, 32 mm/s no-load, 50 N lift, potentiometer feedback | 1 | 100 | 100 | 0.06 kg (est.) |
| 11 | ADC for actuator pot | Adafruit ADS1115 #1085 | 16-bit, 4 ch, I2C | 1 | 14.95 | 14.95 | - |
| 12 | Battery | Bioenno Power BLF-1220A LiFePO4 | 12.8 V, 20 Ah (256 Wh), built-in BMS | 1 | 190 (est.) | 190 | 2.5 kg (est.) |
| 13 | 5 V supply for Pi | Pololu D36V50F5 step-down | 5 V, 5.5 A, 6-50 V in | 1 | 25 (est.) | 25 | - |
| 14 | GNSS receiver | SparkFun NEO-M9N breakout (GPS-17285) | u-blox M9, 4 constellations, ~1.5 m CEP | 1 | 75 (est.) | 75 | 0.01 kg |
| 15 | GNSS antenna | u-blox ANN-MB-00 multiband patch (SparkFun GPS-15192) | magnetic puck, 5 m cable | 1 | 50 (est.) | 50 | 0.17 kg (est.) |
| 16 | Electronics enclosure | Hammond 1554 series polycarbonate, ~200 x 120 x 75 mm | IP66 | 1 | 30 (est.) | 30 | 0.5 kg (est.) |
| 17 | Chassis rails | 2020 aluminium T-slot extrusion (e.g. Misumi HFS5-2020) | 2 x 600 mm + 3 x 400 mm + 800 mm mast | 1 set | 45 (est.) | 45 | 1.8 kg (est.) |
| 18 | Deck plate | 3 mm aluminium sheet 600 x 400 mm | | 1 | 30 (est.) | 30 | 1.9 kg (est.) |
| 19 | Frost cover | Agribon AG-30 row cover, 10 ft x 50 ft roll | ~0.9 oz/yd², ~70 % light transmission, a few °F of frost protection | 1 | 45 (est.) | 45 | 0.8 kg (est.) |
| 20 | Cover roll tube + arm | 50 mm PVC tube, 500 mm, + 20x20 extrusion arm | | 1 | 15 (est.) | 15 | 0.4 kg (est.) |
| 21 | Power wiring | 15 A ATC fuse holder, rocker switch, XT60, 14 AWG | | 1 set | 20 (est.) | 20 | 0.3 kg (est.) |

**Totals (est.):** 21 line items, 31 individual parts, about **USD 1,190**, rover mass about **10.8 kg** (all estimates;
mass dominated by battery, chassis and deck).

## Numbers the simulator uses (derived, est.)

| Quantity | Value | Source |
|---|---|---|
| Rover mass | 10.8 kg (est.) | sum of the mass column |
| Wheel radius | 0.060 m | Wild Thumper 120 mm wheel |
| Stall torque per motor | 2.06 N·m | Pololu #4753 (21 kg·cm, extrapolated) |
| No-load speed | 200 rpm -> top speed 200/60 · 2π · 0.06 = **1.26 m/s** | Pololu #4753 |
| Stall / free current per motor | 5.5 A / ~0.15 A (est.) | Pololu #4753 |
| Max drive force | 4 · 2.06 / 0.06 = 137 N, traction-limited to μ·m·g ≈ 95 N (μ = 0.9, est.) | derived |
| Idle electrical load | ~0.6 A @ 12.8 V (Pi 5 + sensors, est.) | derived |
| Battery capacity | 20 Ah | Bioenno BLF-1220A |
| Actuator speed / stroke | 32 mm/s, 100 mm | Actuonix L16-100-35 |
| Sensor noise model | reading = true air temp + N(0, 0.1 °C) | SHT45 typ accuracy |

## Interface to the culture

The neurons do **not** ride on the rover. The multi-electrode array (MEA) and the living culture would sit in a
Cortical Labs CL1 unit, or be rented through the Cortical Cloud; the rover is a networked client:

1. The Pi 5 samples SHT45 (air) and DS18B20 (leaf/soil) once per minute and reads GNSS position.
2. It sends the sensor vector to the culture service over Wi-Fi or an LTE hotspot (HTTPS / WebSocket).
3. The service encodes it as sensory stimulation on the MEA, reads the DEPLOY / HOLD motor-region spike counts,
   and returns the decision (and gives feedback stimulation the next morning, when the true outcome is known).
4. On DEPLOY the rover drives to the row and extends the actuator to unroll the cover; on HOLD it stays docked.
5. If the network is down the rover falls back to a fixed threshold rule (deploy if air < +1 °C), so the crop is
   never left unprotected because of the culture link.

Everything in this repository today is a **simulated culture model - no living neurons**.
