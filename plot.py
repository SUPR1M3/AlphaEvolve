import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = 2.635
SEED_SCORE = 1.30
GROUPS = [("insp", "Step 5: islands + inspirations", "tab:green"),
          ("islands", "Step 4: islands", "tab:blue"),
          ("", "Step 3: bare loop", "tab:gray")]


def load(run_dir):
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        return [json.loads(line) for line in f]


def group(name):
    return next((label, color) for key, label, color in GROUPS if key in name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories (default: runs/*)")
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    dirs = sorted(d for d in (args.runs or glob.glob("runs/*"))
                  if os.path.exists(os.path.join(d, "log.jsonl")) and "test" not in d)
    runs = {os.path.basename(d.rstrip("/")): load(d) for d in dirs}
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    styles = ["-", "--", ":", "-."]
    per_group = {}
    for name, recs in runs.items():
        label, color = group(name)
        ls = styles[per_group.get(label, 0) % len(styles)]
        per_group[label] = per_group.get(label, 0) + 1
        best = [SEED_SCORE] + [r["best_score"] for r in recs]
        samples = [0] + [r["n_samples"] for r in recs]
        minutes = [0] + [r["wall_time"] / 60 for r in recs]
        kw = dict(color=color, linestyle=ls, drawstyle="steps-post", linewidth=1.8, label=f"{name}  ({best[-1]:.3f})")
        axes[0].plot(samples, best, **kw)
        axes[1].plot(minutes, best, **kw)
    for ax, xlabel in zip(axes, ["LLM samples", "wall-clock time (minutes)"]):
        ax.axhline(TARGET, color="black", linewidth=1, linestyle="--")
        ax.text(ax.get_xlim()[1], TARGET, " AlphaEvolve 2.635", va="bottom", ha="right", fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("best sum of radii so far")
    axes[0].set_ylim(1.2, 2.7)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Best score so far (Qwen3-8B, thinking off, 26 circles)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "best_score.png"), dpi=130)


    fig, ax = plt.subplots(figsize=(13, 0.55 * len(runs) + 1.5))
    names = list(runs)
    for y, name in enumerate(names):
        _, color = group(name)
        pts = [(r["iter"], r["score"]) for r in runs[name] if r.get("valid")]
        n = len(runs[name])
        ax.scatter([s for _, s in pts], [y] * len(pts), color=color, alpha=0.5, s=30)
        ax.text(2.66, y, f"valid {len(pts)}/{n}, distinct {len({round(s, 4) for _, s in pts})}",
                va="center", fontsize=8)
    ax.axvline(TARGET, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("score of each valid program")
    ax.set_title("Every valid program each run produced (overlapping dots = the same idea found repeatedly)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "attempts.png"), dpi=130)
    print(f"wrote {args.out}/best_score.png and {args.out}/attempts.png ({len(runs)} runs)")


if __name__ == "__main__":
    main()
