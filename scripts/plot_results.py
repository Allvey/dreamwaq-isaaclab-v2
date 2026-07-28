#!/usr/bin/env python3
"""Make report figures from rsl_rl TensorBoard logs.

Produces:
  1. reward_comparison.png  -- mean reward vs iteration (DreamWaQ vs baseline)
  2. cenet_losses.png       -- CENet vel / recon / KL losses vs iteration

rsl_rl logs reward under a tag like "Train/mean_reward" and the CENet losses
(returned by DreamWaQPPO.update()) under "Loss/cenet_vel_mse", etc. Tag prefixes
vary by rsl_rl version, so this script DISCOVERS tags by keyword instead of
hard-coding them, and prints what it found.

Run with the Isaac Lab python (has tensorboard + matplotlib):
  ~/IsaacLab/isaaclab.sh -p plot_results.py \
      --logdir ~/IsaacLab/logs/rsl_rl \
      --dreamwaq go2_dreamwaq_rough \
      --baseline unitree_go2_rough \
      --out ./figures

If --dreamwaq / --baseline name an experiment FOLDER, the latest run inside it
is used. You can also pass a full run path directly.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def _load_scalars(run_dir: str) -> dict[str, tuple[list[int], list[float]]]:
    """Return {tag: (steps, values)} for all scalar tags under run_dir.

    Tries tbparse first (fast, simple); falls back to tensorboard's
    EventAccumulator. Scans nested event files so passing an experiment folder
    or a specific run folder both work.
    """
    # --- attempt tbparse ---
    try:
        from tbparse import SummaryReader  # type: ignore

        reader = SummaryReader(run_dir, pivot=False, extra_columns=set())
        df = reader.scalars
        out: dict[str, tuple[list[int], list[float]]] = {}
        for tag in sorted(df["tag"].unique()):
            sub = df[df["tag"] == tag].sort_values("step")
            out[tag] = (sub["step"].tolist(), sub["value"].tolist())
        if out:
            return out
    except Exception:
        pass

    # --- fallback: EventAccumulator ---
    from tensorboard.backend.event_processing import event_accumulator as ea  # type: ignore

    event_files = glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        # maybe run_dir itself is an experiment folder; take the newest run
        raise FileNotFoundError(f"No TensorBoard event files under {run_dir}")

    merged: dict[str, dict[int, float]] = {}
    for ef in event_files:
        acc = ea.EventAccumulator(ef, size_guidance={ea.SCALARS: 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            d = merged.setdefault(tag, {})
            for e in acc.Scalars(tag):
                d[e.step] = e.value
    return {tag: (sorted(d), [d[s] for s in sorted(d)]) for tag, d in merged.items()}


def _resolve_run(logdir: str, name_or_path: str) -> str:
    """Resolve an experiment name OR path to the run dir that actually has events.

    Robust to stray empty subfolders (e.g. an 'outputs/' dir with no events):
    we locate the newest events.out.tfevents.* under the candidate and return
    the directory that contains it.
    """
    cand = name_or_path if os.path.isabs(name_or_path) else os.path.join(logdir, name_or_path)
    if not os.path.isdir(cand):
        raise FileNotFoundError(
            f"Not found: {cand}\n"
            f"  Check --logdir. rsl_rl writes to <cwd-when-you-launched-train>/logs/rsl_rl,\n"
            f"  so if you launched training from your home dir the logs are in ~/logs/rsl_rl,\n"
            f"  NOT ~/IsaacLab/logs/rsl_rl."
        )
    events = glob.glob(os.path.join(cand, "**", "events.out.tfevents.*"), recursive=True)
    if not events:
        subdirs = sorted(os.path.basename(d) for d in glob.glob(os.path.join(cand, "*")) if os.path.isdir(d))
        raise FileNotFoundError(
            f"No TensorBoard event files anywhere under {cand}\n"
            f"  Subfolders present: {subdirs}\n"
            f"  Point --logdir at the directory that actually holds the run "
            f"(the one containing model_*.pt and events.out.tfevents.*)."
        )
    run_dir = os.path.dirname(max(events, key=os.path.getmtime))
    if os.path.realpath(run_dir) != os.path.realpath(cand):
        print(f"  [{name_or_path}] using run: {os.path.relpath(run_dir, cand)}")
    return run_dir


def _find_tag(tags: list[str], *keywords: str) -> str | None:
    """First tag containing all keywords (case-insensitive); prefers shorter tags."""
    matches = [t for t in tags if all(k.lower() in t.lower() for k in keywords)]
    return min(matches, key=len) if matches else None


def _comparison_figure(plt, dw, bl, keywords, ylabel, title, filename, out,
                       dw_label="DreamWaQ (asymmetric + CENet)",
                       bl_label="Baseline (symmetric, no CENet)"):
    """One DreamWaQ-vs-baseline line plot, tag found by keyword."""
    tag_dw = _find_tag(list(dw), *keywords)
    if tag_dw is None:
        print(f"  WARNING: no tag matching {keywords} in DreamWaQ logs; skipping {filename}")
        return
    plt.figure(figsize=(7, 4.2))
    xs, ys = dw[tag_dw]
    plt.plot(xs, ys, label=dw_label, linewidth=2)
    if bl is not None:
        tag_bl = _find_tag(list(bl), *keywords)
        if tag_bl is not None:
            xb, yb = bl[tag_bl]
            plt.plot(xb, yb, label=bl_label, linewidth=2, linestyle="--")
    plt.xlabel("Training iteration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  wrote {path}  (tag: {tag_dw})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=os.path.expanduser("~/logs/rsl_rl"),
                    help="root of rsl_rl logs (where experiment folders live)")
    ap.add_argument("--dreamwaq", default="go2_dreamwaq_rough",
                    help="DreamWaQ experiment name or run path")
    ap.add_argument("--baseline", default=None,
                    help="baseline experiment name or run path (optional)")
    ap.add_argument("--out", default="./figures", help="output directory for PNGs")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)

    print("Loading DreamWaQ logs...")
    dw_run = _resolve_run(args.logdir, args.dreamwaq)
    dw = _load_scalars(dw_run)
    print(f"  found {len(dw)} scalar tags")

    bl = None
    if args.baseline:
        print("Loading baseline logs...")
        bl_run = _resolve_run(args.logdir, args.baseline)
        bl = _load_scalars(bl_run)
        print(f"  found {len(bl)} scalar tags")

    # ---- TensorBoard comparison figures (DreamWaQ vs baseline) ----
    _comparison_figure(plt, dw, bl, ("reward",), "Mean episodic reward",
                       "Training Reward: Baseline vs DreamWaQ",
                       "reward_comparison.png", args.out)
    _comparison_figure(plt, dw, bl, ("episode", "length"), "Steps",
                       "Mean Episode Length",
                       "episode_length.png", args.out)
    _comparison_figure(plt, dw, bl, ("error", "vel"), "error_vel_xy (m/s)",
                       "Linear-Velocity Tracking Error",
                       "tracking_error.png", args.out)
    _comparison_figure(plt, dw, bl, ("terrain",), "Terrain level",
                       "Terrain Curriculum Level Reached",
                       "terrain_level.png", args.out)

    # ---- Figure 2: CENet losses ----
    vel_tag = _find_tag(list(dw), "cenet", "vel")
    rec_tag = _find_tag(list(dw), "cenet", "recon")
    kl_tag = _find_tag(list(dw), "cenet", "kl")
    cenet_tags = [(t, lbl) for t, lbl in
                  [(vel_tag, "velocity MSE"), (rec_tag, "reconstruction MSE"), (kl_tag, "KL")]
                  if t is not None]
    if not cenet_tags:
        print("WARNING: no CENet tags found. Available tags:")
        for t in sorted(dw):
            print("   ", t)
    else:
        fig, axes = plt.subplots(1, len(cenet_tags), figsize=(4.5 * len(cenet_tags), 3.8))
        if len(cenet_tags) == 1:
            axes = [axes]
        for ax, (tag, lbl) in zip(axes, cenet_tags):
            xs, ys = dw[tag]
            ax.plot(xs, ys, linewidth=2)
            ax.set_title(f"CENet {lbl}")
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
        fig.suptitle("CENet hybrid-loss components")
        fig.tight_layout()
        f2 = os.path.join(args.out, "cenet_losses.png")
        fig.savefig(f2, dpi=150)
        print(f"  wrote {f2}  (tags: {[t for t, _ in cenet_tags]})")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
