"""Plot a Tinker training run from its metrics.jsonl: training label NLL by step, and dev AUROC.

usage: python3 scripts/plot_training.py [--run runs/v1] [--out tinker-v1-training.png] [--window 15]

Reads <run>/metrics.jsonl (one row per step from train_tinker.py, plus a final dev-only row at
step == total steps), <run>/config.json (price per million tokens) and <run>/checkpoints.jsonl
(the Tinker run id). Everything plotted and every number in the footnote comes from those files,
except the dev-scoring cost, which is not logged there (DEV_COST, from docs/RUNBOOK.md §1).

Top panel: `label_nll` (NLL of the label token on each training batch of 64, before that step's
update), each step as a faint dot, with a centred rolling median as the line. Bottom panel:
`dev/auroc` on the dev rows at each eval step (pre-step weights, so step 0 is the base model and
the last point is after the final update). Two panels, not two y-axes on one plot.

Needs matplotlib, which is not a project dependency: run it with any python3 that has it.
The output is an opaque RGB PNG laid out at 1100 x 940 px and saved at 2x (2200 x 1880), so the
text stays readable when DEV shrinks the image to its ~800 px column.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Dev scoring cost for runs/v1: about $0.15 for 6 dev evaluations (200,676 forward tokens),
# docs/RUNBOOK.md §1. metrics.jsonl does not log the dev forward tokens.
DEV_COST = {"runs/v1": 0.15}
DURATION = {"runs/v1": "9 min 24 s"}  # docs/RUNBOOK.md §1, wall time incl. dev evals + checkpoints

TITLE = "SecondRead v1 on Tinker: Qwen3.5-4B, MLP-only LoRA r16, 1 epoch"

# Chart tokens (light). Categorical slot 1 (blue) and slot 2 (orange); validated as a pair.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOSS = "#2a78d6"
AUROC = "#eb6834"

DPI = 100  # 1 layout px = 1/100 in; saved at 2x
PT = 72 / DPI  # px -> pt


def px(v: float) -> float:
    """Layout pixels to points (line widths, font sizes)."""
    return v * PT


def dot(diameter_px: float) -> float:
    """Scatter marker size (points squared) for a marker this many layout pixels across."""
    return px(diameter_px) ** 2


def load(run: Path) -> tuple[list[dict], dict, list[dict]]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").open() if line.strip()]
    cfg = json.loads((run / "config.json").read_text())
    ckpts = [json.loads(line) for line in (run / "checkpoints.jsonl").open() if line.strip()]
    return rows, cfg, ckpts


def rolling_median(y: list[float], window: int) -> list[float]:
    """Centred rolling median; the window shrinks at both ends instead of padding.

    A median rather than a mean, so the one step-1 spike (label NLL 3.42 at LR 2e-4, see
    docs/RUNBOOK.md §1) does not drag the first `window // 2` points of the line above the data.
    """
    h = window // 2
    return [statistics.median(y[max(0, i - h): i + h + 1]) for i in range(len(y))]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(px(1))
    ax.grid(axis="y", color=GRID, linewidth=px(1), linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelcolor=INK_2, labelsize=px(17), length=0, pad=px(10))
    ax.tick_params(axis="x", length=px(6), width=px(1), color=AXIS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/v1")
    ap.add_argument("--out", default="tinker-v1-training.png")
    ap.add_argument("--window", type=int, default=15, help="rolling-median window in steps")
    ap.add_argument("--ymax", type=float, default=1.6, help="top of the label NLL axis; higher points are marked off scale")
    a = ap.parse_args()

    run = Path(a.run)
    rows, cfg, ckpts = load(run)
    train = sorted((r for r in rows if "label_nll" in r), key=lambda r: r["step"])
    dev = sorted((r for r in rows if "dev/auroc" in r), key=lambda r: r["step"])
    steps = [r["step"] for r in train]
    nll = [r["label_nll"] for r in train]
    smooth = rolling_median(nll, a.window)
    dev_x = [r["step"] for r in dev]
    dev_y = [r["dev/auroc"] for r in dev]
    n_steps = len(train)
    total_steps = max(dev_x[-1], steps[-1] + 1)
    tokens = sum(r["num_tokens"] for r in train)
    rows_seen = sum(r["num_sequences"] for r in train)
    data = Path(cfg.get("data") or "")
    n_data = sum(1 for line in data.open() if line.strip()) if data.is_file() else rows_seen
    batch = cfg.get("batch_size")
    cost = tokens / 1e6 * cfg["price"]
    final = next((c for c in ckpts if c.get("name") == "final"), ckpts[-1])
    run_id = final["state_path"].removeprefix("tinker://").split("/")[0]  # <uuid>:train:0
    short_id = run_id.split("-")[0] + "…" + run_id[run_id.index(":"):]
    key = run.as_posix().rstrip("/")
    if final.get("elapsed_tokens") not in (None, tokens):
        raise SystemExit(f"token mismatch: metrics {tokens} vs checkpoint {final['elapsed_tokens']}")
    if final.get("step") not in (None, total_steps):
        raise SystemExit(f"step mismatch: metrics {total_steps} vs checkpoint {final['step']}")

    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "text.parse_math": False,  # "$1.07" is text, not mathtext
    })
    W, H = 1100, 940  # layout pixels
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=SURFACE)
    L, R = 72, 74  # left / right margins (the right one holds the end labels)

    def axes(top: float, height: float):
        return fig.add_axes(((L) / W, 1 - (top + height) / H, (W - L - R) / W, height / H))

    def text(x: float, top: float, s: str, size: float, color: str, **kw):
        return fig.text(x / W, 1 - top / H, s, ha="left", va="top", fontsize=px(size), color=color, **kw)

    text(L, 30, TITLE, 30, INK)
    left = n_data - rows_seen  # the partial last batch is not a step
    text(L, 76, f"{n_data:,} training rows: {n_steps} steps of {batch}" + (f", {left} left over" if left else "") + ".  "
                f"Dev AUROC {dev_y[0]:.3f} → {dev_y[-1]:.3f}.", 19, INK_2)

    # Top: training label NLL. -------------------------------------------------------------
    text(L, 136, "Training label NLL", 22, INK)
    text(L, 166, f"Cross-entropy on the Major/OK label token, per batch of {batch}.  "
                 f"Dots: each step.  Line: {a.window}-step rolling median.", 17, INK_2)
    ax1 = axes(212, 318)
    style_axes(ax1)
    ymax = a.ymax
    inside = [(x, y) for x, y in zip(steps, nll) if y <= ymax]
    off = [(x, y) for x, y in zip(steps, nll) if y > ymax]
    ax1.scatter([x for x, _ in inside], [y for _, y in inside], s=dot(6.5),
                color=LOSS, alpha=0.30, linewidths=0, zorder=2)
    for x, y in off:
        ax1.scatter([x], [ymax], marker="^", s=dot(11), color=LOSS, alpha=0.6, linewidths=0, zorder=3, clip_on=False)
        ax1.annotate(f"step {x}: {y:.2f}, off scale", (x, ymax), xytext=(px(12), 0), textcoords="offset points",
                     ha="left", va="center", fontsize=px(15), color=MUTED,
                     bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))
    ax1.plot(steps, smooth, color=LOSS, linewidth=px(2.5), solid_capstyle="round",
             solid_joinstyle="round", zorder=4)
    ax1.scatter([steps[-1]], [smooth[-1]], s=dot(11), color=LOSS, edgecolors=SURFACE, linewidths=px(2), zorder=5)
    ax1.annotate(f"{smooth[-1]:.2f}", (steps[-1], smooth[-1]), xytext=(px(12), 0), textcoords="offset points",
                 ha="left", va="center", fontsize=px(17), color=INK)
    ax1.set_ylim(0, ymax)
    ax1.set_yticks([0, 0.4, 0.8, 1.2, 1.6])
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.1f}"))

    # Bottom: dev AUROC. --------------------------------------------------------------------
    dev_n = cfg.get("dev_n")
    dev_split = Path(cfg.get("dev_split") or "dev").name.split(".")[0]
    text(L, 586, "Dev AUROC", 22, INK)
    text(L, 616, f"{dev_n} {dev_split} dev rows, exact p(Major) from Tinker.  Scored before the first step, "
                 f"every {cfg.get('eval_every')} steps and after the last one.", 17, INK_2)
    ax2 = axes(660, 150)
    style_axes(ax2)
    ax2.plot(dev_x, dev_y, color=AUROC, linewidth=px(2.5), solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax2.scatter(dev_x, dev_y, s=dot(12), color=AUROC, edgecolors=SURFACE, linewidths=px(2), zorder=4)
    ax2.annotate(f"{dev_y[0]:.3f}", (dev_x[0], dev_y[0]), xytext=(px(12), px(-14)), textcoords="offset points",
                 ha="left", va="center", fontsize=px(17), color=INK)
    ax2.annotate("base model", (dev_x[0], dev_y[0]), xytext=(px(64), px(-14)), textcoords="offset points",
                 ha="left", va="center", fontsize=px(15), color=MUTED)
    ax2.annotate(f"{dev_y[-1]:.3f}", (dev_x[-1], dev_y[-1]), xytext=(px(12), 0), textcoords="offset points",
                 ha="left", va="center", fontsize=px(17), color=INK)
    ax2.set_ylim(0.5, 0.8)
    ax2.set_yticks([0.5, 0.6, 0.7, 0.8])
    ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax2.annotate("0.5 = chance", (1.0, 0.5), xycoords=ax2.get_yaxis_transform(), xytext=(0, px(6)),
                 textcoords="offset points", ha="right", va="bottom", fontsize=px(15), color=MUTED)

    ticks = sorted(set(list(range(0, total_steps, 50)) + [total_steps]))
    for ax in (ax1, ax2):
        ax.set_xticks(ticks)
        ax.set_xlim(-3, total_steps + 3)
    ax2.set_xlabel("Training step", fontsize=px(16), color=INK_2, labelpad=px(8))

    # Footnote. -----------------------------------------------------------------------------
    dev_cost = DEV_COST.get(key)
    cost_s = f"${cost:.2f} training" + (f" + ≈${dev_cost:.2f} dev scoring" if dev_cost is not None else "")
    dur = DURATION.get(key)
    lr = f"{cfg.get('lr'):.0e}".replace("e-0", "e-")
    line1 = (f"Tinker run {short_id}  ·  {n_steps} steps  ·  {tokens:,} tokens  ·  {cost_s}"
             + (f"  ·  {dur}" if dur else ""))
    line2 = (f"LR {lr}, {cfg.get('lr_schedule')} decay, LoRA rank {cfg.get('rank')}, seed {cfg.get('seed')}.  "
             f"Plotted from {key}/metrics.jsonl.")
    text(L, 884, line1, 16, INK_2)
    text(L, 910, line2, 16, MUTED)

    fig.savefig(a.out, dpi=DPI * 2, facecolor=SURFACE)
    from PIL import Image  # ships with matplotlib; drop the (fully opaque) alpha channel
    Image.open(a.out).convert("RGB").save(a.out, optimize=True)
    print(f"wrote {a.out}: {n_steps} steps, {tokens:,} tokens, ${cost:.2f}, dev AUROC "
          + ", ".join(f"{x}:{y:.3f}" for x, y in zip(dev_x, dev_y))
          + f"; label NLL rolling median {smooth[0]:.3f} -> {smooth[-1]:.3f}")


if __name__ == "__main__":
    main()
