# neurorover/envdata: real-data night conditions for the frost-guard rover

Realistic, variable environment conditions (temperature, dew point, humidity, wind and gusts, cloud,
rain, soil moisture and soil temperature) for the 35 NOAA stations of the hackathon freeze data, built
from real archives, plus the physical effects the robot sim needs (ground friction, wind load on the
frost cover, radiative-cooling cue, noisy sensor readings).

| File | What it does |
| --- | --- |
| `fetch.py` | Downloads Open-Meteo hourly archive data per station x frost season (resumable, rate-limit aware), caches raw JSON in `cache/raw/` (gitignored), builds the nightly table `cache/nightly.parquet` and the join report `results/join.json` |
| `generator.py` | `EnvGenerator` (fit on train years 2000-2014 only), `physics()`, `sensors()`, `SoilProbeModel` |
| `validate.py` | Generated vs held-out real nights 2015-2019 -> `results/validation.json`, `results/*.png` |
| `frost_risk.py` | Does real weather context improve freeze prediction? -> `results/frost_risk.json` |
| `stations.csv` | Names and coordinates of the 35 stations (NOAA GHCN-Daily station list) |

```
.venv/Scripts/python -m neurorover.envdata.fetch fetch --max-minutes 45   # re-run to fill missing seasons
.venv/Scripts/python -m neurorover.envdata.fetch aggregate
.venv/Scripts/python -m neurorover.envdata.validate
.venv/Scripts/python -m neurorover.envdata.frost_risk
.venv/Scripts/python -m neurorover.envdata.generator                      # demo night
```

## How sim/ should use it

```python
import numpy as np
from neurorover.envdata import EnvGenerator     # lazy: fits itself (train years only) on first call

gen = EnvGenerator()
rng = np.random.default_rng(0)

night = gen.sample_night("USW00093193", "2016-12-05", rng)        # date: ISO str, datetime.date or day-of-year int
season = gen.sample_season("USW00093193", "2016-12-01", 30, rng)  # 30 consecutive nights with persistence
night = gen.sample_night(None, 20, rng, tmin=-1.5)                 # random station, doy 20, forced station tmin

# sim/night_loop.py hook (already written by the sim agent) calls gen.night(station, date):
c = gen.night("USW00093193", "2016-12-05")
dr_override = {"friction": c["friction"],                       # ground_friction_mu
               "wind": 0.5 * 1.2 * 1.2 * c["wind_ms"] ** 2 * 0.12,  # or directly c["panel_force_gust_N"]
               "noise": c["noise_level"]}
```

Keys of a night dict (all floats unless noted):

| Key | Meaning |
| --- | --- |
| `air_tmin_c`, `frost` | Station-scale (NOAA) night minimum air temperature, the frost ground truth (`frost = air_tmin_c < 0`). Drawn from the train-only `neurorover.world.model.WorldModel` (VAR(1)) unless `tmin=` is passed |
| `reanalysis_tmin_night_c`, `reanalysis_tmax_calday_c` | Archive (reanalysis-grid) min 18:00-09:00 and calendar-day max, shifted with the analog |
| `tdmin_night_c`, `rh_mean_night_pct` | Night minimum dew point; mean RH 18:00-09:00 (RH from T and dew point, Magnus) |
| `wind_mean_kmh`, `wind_max_kmh`, `gust_max_kmh` | 10 m wind mean/max and max gust over the night |
| `cloud_mean_pct`, `precip_24h_mm` | Mean total cloud cover over the night; rain in the 24 h ending 09:00 |
| `soil_moist_06_m3m3`, `soil_temp_06_c` | ERA5-Land soil moisture and temperature, 0-7 cm, at 06:00 local |
| `ground_friction_mu`, `soil_relative_saturation` | Ground friction coefficient for the given `soil_type` (see assumptions) |
| `wind_mean_at_cover_ms`, `gust_at_cover_ms`, `air_density_kgm3` | Wind reduced to cover height 0.5 m |
| `cover_force_mean_N`, `cover_force_gust_N` | Drag on a deployed cover of `cover_area_m2` (default 2 m2) |
| `panel_force_gust_N` | Same for the sim's 0.12 m2 cover panel (`sim/params.py` DR wind) |
| `sky_emissivity`, `net_longwave_loss_wm2`, `leaf_minus_air_c`, `leaf_tmin_c` | Radiative cooling cue |
| `sensors` | `sht45_air_c`, `sht45_rh_pct`, `ds18b20_leaf_c`, `soil_probe_vwc`: noisy readings at the coldest hour |
| `analog` | The real (station, date) night the non-temperature variables were resampled from |

## Data sources and licences

* **Open-Meteo Historical Weather API**, https://open-meteo.com/en/docs/historical-weather-api, endpoint
  `https://archive-api.open-meteo.com/v1/archive` (ERA5 / ERA5-Land reanalysis, Copernicus).
  Licence CC BY 4.0. Attribution: *"Weather data by Open-Meteo.com" (https://open-meteo.com/), licensed
  under CC BY 4.0; contains modified Copernicus Climate Change Service information (ERA5, ERA5-Land).*
  Only the public station coordinates are sent. Requests: one per station x season (Sep 22 - Apr 30),
  8 hourly variables, 1 s between calls, retry with backoff, stop on the hourly/daily quota.
* **Freeze-preparation data** (hackathon resource, NOAA GHCN-Daily derived; read in place, never copied):
  `resources/challenges/freeze-preparation/09-freeze/data.json.gz`. Supplies station IDs, coordinates,
  dates, the lagged station tmin/tmax features and the freeze target.
* **KU Leuven DRIP soil-moisture data** (hackathon resource, read in place): Hendrickx et al. (2025),
  *In situ soil moisture measurement data for sensor calibration and pooled measurement error modeling*,
  KU Leuven RDR, https://doi.org/10.48804/M6WKTN, CC BY 4.0. Used for the soil-probe error model.
* **NOAA GHCN-Daily station list** (public domain, US Government):
  https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt -> `stations.csv`.
  All 35 stations are in California's Central Valley / Bay Area / Central Coast.
* Other challenge folders checked: 07-wheat (wheat-head images + RGB features, no weather / soil / dates
  usable for conditions), flusight, foodborne, trial, counterscreen, solubility, ired, sequencing QC:
  nothing environmental. **Not used.**

## Method

**Nightly aggregation** (`fetch.py`): night of target date D = 18:00 on D-1 to 09:00 on D, local time.
**Generator** (`generator.py`), fitted on 2000-2014 only:

1. Station tmin from the train-only WorldModel rollout (or passed in).
2. Analog resampling: a real train-year night of the same station within +-21 days of the day-of-year,
   weighted by a Gaussian kernel (1.5 C) on |station tmin of the analog - tmin|. Station tmin of each
   archive night comes from the NOAA lag features (`world.model.reconstruct`), so the dependence of wind,
   cloud, dew point, rain and soil state on how cold the night really was is the real one.
3. Temperature-like variables of the analog shifted by the tmin difference (soil temperature by half).
4. `sample_season`: with p = 0.8 the next night's analog is the real night after the previous analog (if
   it still matches the tmin within 3 C); otherwise the kernel also keeps soil moisture (0.015 m3/m3, but
   increases allowed after rain) and cloud (25 %) close to the previous night.
5. Stations without train data (4 cases-only stations) borrow the nearest fitted station.

**Physics (sources and assumptions)**

* Ground friction: `mu = mu_dry(soil) - drop(soil) * S^2`, S = theta / porosity (porosity by texture from
  Rawls, Brakensiek & Saxton 1982). `mu_dry` 0.55 (sand) - 0.75 (loam) and the wet drop 0.05 (sand) -
  0.40 (loam), with a cap of 0.30 on frozen wet ground, are **ASSUMPTIONS**, chosen to sit in the ranges
  of tyre-soil traction coefficients reported for dry vs wet agricultural soils (Wong, *Theory of Ground
  Vehicles*, 4th ed., 2008; ASABE D497) and to stay within the sim's existing DR friction range (0.5-1.0
  with frozen ground below it). Soil type codes are the Belgian texture classes used in DRIP
  (Z sand, S loamy sand, P light sandy loam, L sandy loam, A loam); default `soil_type="S"`.
* Wind on the cover: `F = 0.5 rho Cd A u^2`, Cd = 1.2 (flat sheet normal to flow, Hoerner 1965,
  conservative), rho from temperature and station elevation, u reduced from 10 m to 0.5 m with a log
  profile, z0 = 0.03 m (open farmland). Area: `hardware/BOM.md` lists only the Agribon roll (10 ft x
  50 ft), not a deployed area, so `cover_area_m2` defaults to an **assumed 2 m2**; `panel_force_gust_N`
  uses the sim's 0.12 m2 panel.
* Radiative cue: clear-sky emissivity from dew point (Berdahl & Martin 1984) with the cloud correction
  `(1 + 0.0224N - 0.0035N^2 + 0.00028N^3)`, N in tenths; net longwave loss `sigma T^4 (1 - eps)`;
  leaf-air offset from a simple leaf energy balance (Campbell & Norman 1998, 5 cm leaf).
* Sensors: SHT45 air = truth + per-device bias N(0, 0.05) + N(0, 0.1) C (datasheet +-0.1 C typ),
  RH + N(0, 1 %RH); DS18B20 leaf = truth + per-device bias U(-0.4, 0.4) + N(0, 0.1) C, 0.0625 C steps
  (datasheet +-0.5 C max); soil probe from the DRIP-fitted error model (below). `sample_season` keeps the
  per-device biases fixed across the season.

## Results (run of 2026-11-14)

**Coverage.** 430 of 835 needed station x season windows fetched (Open-Meteo quota is 5000 calls/hour,
a full pull costs ~11.6k): all 185 validation-year windows (seasons 2014-2019) and 245 train-year
windows (seasons 1999-2006 and 2013). 94,721 station-nights, 31 stations, `cache/nightly.parquet` 1.9 MB
(raw cache 24 MB, gitignored). Not yet fetched: train seasons 2007-2012 and the 2020-2025 cases years
(incl. the 4 cases-only stations). Re-run `fetch` later (resumable), then `aggregate`, `validate`, `frost_risk`.

**Join validation** (`results/join.json`, 82,852 labelled freeze rows joined):

| Check | Value |
| --- | --- |
| Agreement archive night tmin < 0 C vs freeze target | 0.962 (base rate 4.6 %; archive says frost on 1.6 %) |
| Freeze recall / precision of archive tmin < 0 C | 0.26 / 0.74 |
| Per-station agreement min / median / max | 0.897 / 0.966 / 1.000 |
| Pearson r, archive calendar-day tmin vs station tmin lag3 / lag4 / lag7 | 0.846 / 0.847 / 0.850 |
| Mean bias archive - station tmin | +0.95 C (reanalysis grid is warmer: station frost pockets are missed) |

So the reanalysis is a good proxy for weather *context* but misses most station frosts; the frost ground
truth in the generator is therefore the station-scale tmin (`air_tmin_c`), never the reanalysis one.

**Generator vs held-out real nights 2015-2019** (`results/validation.json`, 12,000 nights, generated
tmin from the WorldModel, not the real one):

| Variable | KS | real mean | gen mean |
| --- | --- | --- | --- |
| station tmin (C) | 0.126 | 7.14 | 6.34 |
| dew point min (C) | 0.035 | 4.44 | 4.64 |
| RH mean (%) | 0.064 | 77.3 | 79.6 |
| wind max (km/h) | 0.187 | 12.5 | 10.4 |
| gust max (km/h) | 0.064 | 26.6 | 24.6 |
| wind mean (km/h) | 0.129 | 7.97 | 6.84 |
| cloud mean (%) | 0.022 | 49.6 | 50.0 |
| precip 24 h (mm) | 0.022 | 2.58 | 2.28 |
| soil moisture 06:00 | 0.052 | 0.266 | 0.279 |
| soil temp 06:00 (C) | 0.140 | 9.66 | 8.74 |

| Spearman, within station | real | generated |
| --- | --- | --- |
| tmin ~ cloud (cloudy nights milder) | 0.289 | 0.151 |
| tmin ~ wind (windy nights milder) | 0.103 | 0.096 |
| tmin ~ dew point | 0.692 | 0.727 |
| soil moisture ~ precip | 0.559 | 0.558 |
| cloud ~ precip | 0.435 | 0.425 |
| tmin ~ soil temp | 0.873 | 0.857 |

Freeze frequency: overall real 4.4 % vs generated 5.7 % (the train years were colder, and the WorldModel
reproduces train climate); per station r = 0.94, MAE 1.6 points (`results/freeze_frequency.png`).
Lag-1 autocorrelation in 30-night seasons (real / gen): tmin 0.63 / 0.60, soil moisture 0.69 / 0.80,
cloud 0.24 / 0.44 (generated cloud spells are too persistent). Physics on generated nights (p5 / median /
p95): friction 0.52 / 0.58 / 0.64 (loamy sand), gust force on a 2 m2 cover 3.9 / 12 / 65 N, leaf - air
-1.8 / -1.2 / -0.6 C. Plots: `results/marginals.png`, `results/joint.png`, `results/freeze_frequency.png`.

**Soil-probe error model** (DRIP train split, per soil type): sensor = a * true + b + device offset + noise,
a = 0.56-0.82, b = -0.005-0.062, device offset sd 0-0.025, noise sd 0.020-0.029 m3/m3; residual sd
0.030 (train) / 0.038 (validation devices). The capacitance probes under-read by ~0.04-0.06 m3/m3.

**Frost risk: does real weather context help?** (`results/frost_risk.json`; `bench.fit_logistic`,
lambda = 1, fit on 50,730 train rows with covariates, AUC on all 32,122 validation rows)

| Features | Validation AUC |
| --- | --- |
| 14 encoded freeze features | 0.898 |
| + Open-Meteo covariates from the input window (nights D-3, D-4, mean D-9..D-3; no leakage) | **0.908** |
| + only dew point, cloud, wind, RH at D-3 | 0.905 |
| ORACLE (LEAKAGE, upper bound): + target-night covariates | 0.985 |
| ORACLE (LEAKAGE): archive target-night tmin alone | 0.963 |

Real lagged covariates add about +0.01 AUC; knowing the target night's weather (a forecast) would add far more.

## Limitations

* Partial fetch (see coverage): the generator has 52,168 complete analog nights over 31 stations, with a
  median of 9 contributing calendar years per station after its 2000-2014 date filter. Train frost-season
  windows for 2007-2012 and case years 2020-2025 were not fetched. Winds in the available analog pool are
  a little weaker than 2015-2019 (KS 0.13-0.19).
* Reanalysis (ERA5 ~25 km / ERA5-Land ~9 km) vs point station: archive tmin is ~1 C warmer and misses
  three-quarters of station frosts; soil moisture is the grid-cell model value, not field truth.
* The tmin ~ cloud dependence is weaker than real (0.15 vs 0.29), because the analog kernel (1.5 C) and the
  temperature shift blur it; narrow `BANDWIDTH` if the sim needs a stronger signal.
* tmin persistence and freeze rate come from the existing WorldModel (train climate, colder than 2015-19).
* Friction coefficients, the 2 m2 cover area, the noise_level multiplier and the leaf energy balance are
  labelled assumptions, not measurements. DRIP probes are Belgian field sensors used as a stand-in for the
  rover's probe; the "true" moisture fed to them is ERA5-Land.
* Hourly detail is aggregated away; the night dict gives night extremes/means, not an hourly trajectory.
