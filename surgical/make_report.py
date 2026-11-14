"""Build results/learning_curve.png and print the eval table (markdown) from results/*.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"
SWEEP_LRS = (("0p02", 0.02), ("0p05", 0.05), ("0p1", 0.1), ("0p2", 0.2))


def plot_vm_lr_sweep():
    """Plot results/vm_lr_sweep/learning_curves.png and print the per-lr final table (4 Freestyle VM runs)."""
    d = OUT / "vm_lr_sweep"
    runs = [(lr, json.loads((d / f"curve_teach_lr{tag}.json").read_text())) for tag, lr in SWEEP_LRS
            if (d / f"curve_teach_lr{tag}.json").exists()]
    if not runs:
        return
    curves = [json.dumps(r["curve"], sort_keys=True) for _, r in runs]
    assert len(set(curves)) == len(curves), "duplicate sweep curves: files are copies, not distinct runs"
    for lr, r in runs:
        assert abs(r["lr"] - lr) < 1e-9 and r["episodes"] == 60, (lr, r["lr"], r["episodes"])
    cols = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for (lr, r), col in zip(runs, cols):
        c = r["curve"]
        ax.plot([p["episode"] for p in c], [p["median_err_mm"] for p in c], "o-", color=col, lw=2,
                label=f"lr = {lr:g} (final {c[-1]['median_err_mm']:.1f} mm)")
    ax.axhline(10, color="#444", ls="--", lw=1)
    ax.text(1, 11.5, "1 cm success threshold", fontsize=8, color="#444")
    ax.set_yscale("log")
    ax.set_xlabel("teaching episode (80 paired-teaching steps each)")
    ax.set_ylabel("median final error, 6 held-out targets (mm)")
    ax.set_title("Culture teaching, plasticity-lr sweep on 4 Freestyle VMs\n"
                 "0% success for every lr after 60 episodes (50-target final eval)", fontsize=10)
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(d / "learning_curves.png", dpi=130)
    plt.close(fig)
    rows = ["| plasticity lr | final success (50 targets) | final median error (mm) | final mean error (mm) |"
            " final 6-target median (mm) |", "|---:|---:|---:|---:|---:|"]
    for lr, r in runs:
        f = r["final_eval"]
        rows.append(f"| {lr:g} | {100*f['success_rate']:.0f}% | {f['median_err_mm']:.1f} | {f['mean_err_mm']:.1f} | "
                    f"{r['curve'][-1]['median_err_mm']:.1f} |")
    print("\n".join(rows))


if __name__ == "__main__" and "--sweep-only" in __import__("sys").argv:
    plot_vm_lr_sweep()
    raise SystemExit(0)
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
plot_vm_lr_sweep()
