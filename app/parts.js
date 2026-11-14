// Parts table mirroring hardware/BOM.md. Dimensions in metres (scene units = metres). (est.) = estimate.
export const PARTS = {
  pi:      { name: 'Raspberry Pi 5 (8 GB)', spec: 'BCM2712 4x A76 2.4 GHz · Wi-Fi 5 · 5 V/5 A', qty: 1, usd: 80, kg: 0.05 },
  sht45:   { name: 'Sensirion SHT45 (Adafruit #5665)', spec: '±0.1 °C typ · ±1.0 %RH · I2C', qty: 1, usd: 12.5, kg: 0.005 },
  shield:  { name: 'Davis 7714 radiation shield', spec: 'passive multi-plate, houses SHT45', qty: 1, usd: 90, kg: 0.6, est: true },
  ds18b20: { name: 'DS18B20 waterproof probe (Adafruit #381)', spec: '±0.5 °C · 1-Wire · leaf/soil temp', qty: 1, usd: 9.95, kg: 0.03 },
  motor:   { name: 'Pololu #4753 50:1 37D gearmotor', spec: '12 V · 200 rpm · 2.06 N·m stall · 5.5 A stall · 64 CPR enc.', qty: 4, usd: 57, kg: 0.22, est: true },
  wheel:   { name: 'Dagu Wild Thumper wheel 120x60 mm', spec: 'r = 0.060 m · on Pololu #1083 hub', qty: 4, usd: 12, kg: 0.2, est: true },
  driver:  { name: 'Cytron MDD10A dual motor driver', spec: '5-30 V · 10 A cont / 30 A peak per ch', qty: 2, usd: 22, kg: 0.04, est: true },
  actuator:{ name: 'Actuonix L16-100-35-12-P', spec: '100 mm stroke · 32 mm/s · 50 N · pot feedback', qty: 1, usd: 100, kg: 0.06 },
  battery: { name: 'Bioenno BLF-1220A LiFePO4', spec: '12.8 V · 20 Ah · 256 Wh · BMS', qty: 1, usd: 190, kg: 2.5, est: true },
  buck:    { name: 'Pololu D36V50F5 step-down', spec: '5 V · 5.5 A for Pi 5', qty: 1, usd: 25, kg: 0.01, est: true },
  gnss:    { name: 'u-blox ANN-MB-00 + NEO-M9N', spec: 'multiband patch puck · M9 receiver ~1.5 m CEP', qty: 1, usd: 125, kg: 0.18, est: true },
  box:     { name: 'Hammond 1554 IP66 enclosure', spec: '~200x120x75 mm · Pi 5, 2x MDD10A, ADS1115, buck', qty: 1, usd: 30, kg: 0.5, est: true },
  rail:    { name: '2020 Al T-slot extrusion', spec: '2x 600 + 3x 400 mm frame · 800 mm mast', qty: 1, usd: 45, kg: 1.8, est: true },
  deck:    { name: '3 mm aluminium deck plate', spec: '600 x 400 mm', qty: 1, usd: 30, kg: 1.9, est: true },
  cover:   { name: 'Agribon AG-30 frost cloth', spec: '10x50 ft roll · ~70 % light transmission', qty: 1, usd: 45, kg: 0.8, est: true },
  roll:    { name: 'Cover roll tube + arm', spec: '50 mm PVC x 500 mm on 2020 arm', qty: 1, usd: 15, kg: 0.4, est: true },
  wiring:  { name: 'Fuse, switch, XT60, 14 AWG', spec: '15 A ATC fuse', qty: 1, usd: 20, kg: 0.3, est: true },
  ads1115: { name: 'Adafruit ADS1115 ADC', spec: '16-bit · reads actuator pot', qty: 1, usd: 14.95, kg: 0.005 },
  hubs:    { name: 'Pololu #1083 hubs + #1084 brackets', spec: '6 mm shaft hubs, 37D L-brackets', qty: 1, usd: 34, kg: 0.1, est: true },
};
// Physics numbers derived from the BOM (see hardware/BOM.md).
export const PHYS = {
  mass: 10.8,            // kg (est., sum of BOM masses)
  wheelR: 0.060,         // m
  stallTorque: 2.06,     // N·m per motor (Pololu 21 kg·cm, extrapolated)
  noLoadRpm: 200,
  stallA: 5.5, freeA: 0.15, // A per motor (free current est.)
  idleA: 0.6,            // A at 12.8 V, Pi + sensors (est.)
  batteryAh: 20,
  mu: 0.9,               // tyre-soil friction (est.)
  actSpeed: 0.032, actStroke: 0.100, // m/s, m
  shtSigma: 0.1,         // °C, SHT45 typ accuracy used as 1-sigma noise
};
PHYS.vmax = PHYS.noLoadRpm / 60 * 2 * Math.PI * PHYS.wheelR; // 1.257 m/s
export const bomTotals = () => {
  let usd = 0, n = 0, kg = 0;
  for (const p of Object.values(PARTS)) { usd += p.usd * p.qty; n += p.qty; kg += p.kg * p.qty; }
  return { usd, n, kg, lines: Object.keys(PARTS).length };
};
