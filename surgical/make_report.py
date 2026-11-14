"""Build results/learning_curve.png and print the eval table (markdown) from results/*.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
ev = json.loads((OUT / "eval.json").read_text())["results"]
rows = ["| controller | success (1 cm, 0.5 s hold) | final error median / mean (mm) | reached 1 cm | median time to 1 cm (s) | joint-limit steps | collision contacts (episodes) |",
        "|---|---|---|---|---|---|---|"]
for k in ["ik_dls", "ppo", "culture_teach", "culture_frozen", "random"]:
    if k not in ev:
        continue
    r = ev[k]
    tt = f"{r['time_to_reach_s_median']:.2f}" if r["time_to_reach_s_median"] is not None else "-"
    rows.append(f"| {k} | {100*r['success_rate']:.0f}% | {r['final_err_mm_median']:.1f} / {r['final_err_mm_mean']:.1f} | "
                f"{100*r['reached_1cm_frac']:.0f}% | {tt} | {r['joint_limit_steps_total']} | "
                f"{r['collision_contacts_total']} ({r['episodes_with_collision']}) |")
print("\n".join(rows))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for mode, col in (("teach", "#2a7de1"), ("frozen", "#999999")):
    f = OUT / f"curve_{mode}.json"
    if not f.exists():
        continue
    c = json.loads(f.read_text())["curve"]
    x = [r["teaching_steps"] for r in c]
    if mode == "frozen":
        tc = json.loads((OUT / "curve_teach.json").read_text())["curve"] if (OUT / "curve_teach.json").exists() else c
        x = [r["teaching_steps"] for r in tc]
        c = [c[0]] * len(x)
    ax[0].plot(x, [r["median_err_mm"] for r in c], "o-", color=col, label="culture_" + mode)
    ax[1].plot(x, [100 * r["success"] for r in c], "o-", color=col, label="culture_" + mode)
ax[0].set_xlabel("teaching steps"); ax[0].set_ylabel("median final error (mm)"); ax[0].set_yscale("log")
ax[1].set_xlabel("teaching steps"); ax[1].set_ylabel("success (%)"); ax[0].legend()
for a in ax:
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "learning_curve.png", dpi=120)
