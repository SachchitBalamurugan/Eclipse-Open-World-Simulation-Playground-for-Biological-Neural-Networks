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

RESULTS_PLACEHOLDER
