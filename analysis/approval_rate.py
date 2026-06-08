"""Main-text approval-rate figure (all non-abliterated models).

A graphical view of the per-round approval rate. This is a graph-format
complement to the three-data-point Round-1 / Round-mid / Round-late
columns in Table 2 of §4.1.

Reuses every helper from ``eval/approval_line_chart.py``: same per-round
yes-vote computation, same per-model sort, same legend block (icons in
the legend handles via ``AnnotationBbox``), same x-axis cut at the last
round where ≥ 10 % of games reach. The only addition is a shaded 95 %
Wilson CI band per line so the reader can see per-round uncertainty.

Output: ``analysis/plots/approval_rate.pdf`` and
``analysis/results/approval_rate.json``.

Run from the repo root:

    python -m analysis.approval_rate                     # all non-abliterated
    python -m analysis.approval_rate --include-abliterated

LaTeX rendering: the script will probe for ``pdflatex`` and, if absent
from ``$PATH``, prepend the cluster's TeX Live 2019 install
(``/sw/tools/texlive/2019/skl/bin/x86_64-linux``). If that path is also
missing, it falls back to ``use_latex=False``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import plot_config  # noqa: E402
from plot_config import (  # noqa: E402
    FIG_WIDTH, MIN_GAMES,
    setup_plot_style,
    extract_model_name,
    get_model_color, get_markerdata_for_model,
    load_summary_file,
    collect_model_keys,
    compute_win_rate,
    sort_models_by_winrate,
)

from analysis.common import ROOT, ALICE_ID  # noqa: E402

# ----- LaTeX rendering -------------------------------------------------------
# Probe for a working pdflatex; cluster has TeX Live 2019 in a non-standard
# path (no `module load texlive`) so prepend it to PATH ourselves if needed.
_TEXLIVE_PATH = "/sw/tools/texlive/2019/skl/bin/x86_64-linux"
if shutil.which("pdflatex") is None and Path(_TEXLIVE_PATH, "pdflatex").exists():
    os.environ["PATH"] = _TEXLIVE_PATH + os.pathsep + os.environ.get("PATH", "")
_USE_LATEX = shutil.which("pdflatex") is not None
setup_plot_style(use_latex=_USE_LATEX)

MIN_GAMES_AT_ROUND = 10  # drop rounds where fewer than this many games reach


def wilson_ci(yes: int, total: int) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if total == 0:
        return float("nan"), float("nan")
    p = yes / total
    z = 1.959963984540054  # 97.5th percentile of N(0,1)
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    halfwidth = (z / denom) * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return max(0.0, center - halfwidth), min(1.0, center + halfwidth)


# ------------------------------------------------------------------
# Per-round approval rate (cf. eval/approval_line_chart.py
# compute_per_round_rates, plus Wilson CIs).
# ------------------------------------------------------------------

def compute_per_round_rates(folder: Path) -> dict | None:
    """Return ``{round: {'yes', 'total', 'rate', 'lo', 'hi'}}`` or None.

    Round numbers are 1-based. Rounds where fewer than ``max(MIN_GAMES_AT_ROUND,
    10 % of games)`` reach are dropped, matching the cutoff used in
    ``eval/approval_line_chart.py``.
    """
    json_files = list(folder.glob("*_summary.json"))
    if len(json_files) < MIN_GAMES:
        return None

    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"yes": 0, "total": 0})
    n_games = 0

    for fpath in json_files:
        summary = load_summary_file(fpath)
        if summary is None:
            continue
        gs = summary.get("gameSetting")
        if gs is not None and gs.get("avalonSH") is not None:
            continue
        logs = summary.get("logs") or []
        if not logs:
            continue
        n_games += 1
        for round_idx, log in enumerate(logs):
            votes = log.get("votes")
            if not votes or not isinstance(votes, list):
                continue
            if len(votes) <= ALICE_ID or votes[ALICE_ID] is None:
                continue
            rnd = round_idx + 1
            counts[rnd]["total"] += 1
            if bool(votes[ALICE_ID]):
                counts[rnd]["yes"] += 1

    if n_games < MIN_GAMES:
        return None

    cutoff = max(MIN_GAMES_AT_ROUND, math.ceil(0.10 * n_games))
    result: dict[int, dict] = {}
    for rnd in sorted(counts.keys()):
        total = counts[rnd]["total"]
        if total < cutoff:
            break
        yes = counts[rnd]["yes"]
        lo, hi = wilson_ci(yes, total)
        result[rnd] = dict(yes=yes, total=total, rate=yes / total, lo=lo, hi=hi)
    return result or None


# ------------------------------------------------------------------
# Plotting (legend block lifted verbatim from eval/approval_line_chart.py)
# ------------------------------------------------------------------

def _fmt_pct(value: float) -> str:
    return fr"{int(value)}\%" if _USE_LATEX else f"{int(value)}%"


def plot_approval_lines(round_data: dict, baseline_names: set, out_pdf: Path):
    ordered_models = list(round_data.keys())
    if not ordered_models:
        print("No data to plot.")
        return

    # Hard x-axis cap regardless of per-model run-out
    X_MAX = 9.5
    max_round = max(
        (max(rd.keys()) for rd in round_data.values() if rd),
        default=10,
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.3))
    lines = []

    for model in ordered_models:
        rd = round_data[model]
        rounds = sorted(rd.keys())
        rates = [rd[r]["rate"] * 100 for r in rounds]
        lo = [rd[r]["lo"] * 100 for r in rounds]
        hi = [rd[r]["hi"] * 100 for r in rounds]
        m, ms = get_markerdata_for_model(model)
        color = get_model_color(model)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.18, linewidth=0)
        (line,) = ax.plot(
            rounds, rates,
            marker=m, color=color,
            linewidth=2, markersize=ms, label=model,
            markeredgecolor="white", markeredgewidth=1,
        )
        lines.append((model, line))

    ax.set_xlabel("")
    ax.set_ylabel(r"Approval rate (\%)" if _USE_LATEX else "Approval rate (%)")
    ax.grid(True, alpha=0.4)
    ax.set_ylim(None, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: _fmt_pct(y)))
    ax.set_xticks(range(1, int(X_MAX) + 1))
    ax.set_xlim(0.85, X_MAX)

    ax.annotate("Round", xy=(1, 0), xycoords=("data", "axes fraction"),
                xytext=(-15, -7), textcoords="offset points",
                ha="right", va="top", fontsize=plt.rcParams["axes.labelsize"])

    legend = ax.legend(
        framealpha=0,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        handlelength=0,
        handletextpad=1.9,
        ncol=1,
    )

    # Add model icons to legend (same block as eval/approval_line_chart.py)
    for model, handle in zip([m for m, _ in lines], legend.legend_handles):
        imagebox = plot_config.get_model_imagebox(model)
        if imagebox is not None:
            imagebox.set_zoom(imagebox.get_zoom() * 0.8)
            ab = AnnotationBbox(
                imagebox, (0.5, 0.5), xybox=(10, 0),
                xycoords=handle, boxcoords="offset points",
                frameon=False, box_alignment=(0.5, 0.5), zorder=10,
            )
            fig.add_artist(ab)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Approval-rate per-round line chart with Wilson 95% bands."
    )
    parser.add_argument("--include-abliterated", action="store_true",
                        help="Include abliterated model variants.")
    args = parser.parse_args()

    keys = collect_model_keys(
        include_abliterated=args.include_abliterated,
        include_baselines=False,
    )

    round_data: dict[str, dict[int, dict]] = {}
    win_rates: dict[str, float] = {}

    for key in keys:
        folder = ROOT / key
        if not folder.is_dir():
            continue
        name = extract_model_name(key)
        rd = compute_per_round_rates(folder)
        if rd is None:
            continue
        wr = compute_win_rate(folder)
        if wr is None:
            continue
        round_data[name] = rd
        win_rates[name] = wr
        max_r = max(rd.keys())
        print(f"{name:30s}  rounds 1-{max_r}  WR={wr:.0f}%")

    if not round_data:
        print("No data found. Run the simulator first.", file=sys.stderr)
        sys.exit(1)

    ordered, baseline_names = sort_models_by_winrate(round_data, win_rates)

    rounds_all = sorted({r for d in ordered.values() for r in d})
    header_rounds = rounds_all[: min(len(rounds_all), 10)]
    summary_rows = {}
    print("\nApproval rates by round:")
    print("Model".ljust(30) + "".join(f"  R{r:<5d}" for r in header_rounds))
    for name, rd in ordered.items():
        row_cells = [f"{100 * rd[r]['rate']:5.1f}" if r in rd else "  -  "
                     for r in header_rounds]
        print(name.ljust(30) + "".join(f"  {c} " for c in row_cells))
        first = min(rd)
        last = max(rd)
        summary_rows[name] = dict(
            round_first=first, round_last=last,
            rate_first=rd[first]["rate"] * 100,
            rate_last=rd[last]["rate"] * 100,
            delta_pp=(rd[last]["rate"] - rd[first]["rate"]) * 100,
            first_ci=[rd[first]["lo"] * 100, rd[first]["hi"] * 100],
            last_ci=[rd[last]["lo"] * 100, rd[last]["hi"] * 100],
        )

    out_pdf = ROOT / "analysis" / "plots" / "approval_rate.pdf"
    plot_approval_lines(ordered, baseline_names, out_pdf)

    out_json = ROOT / "analysis" / "results" / "approval_rate.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(dict(
            models=list(ordered.keys()),
            win_rates_pct={n: 100 * win_rates[n] for n in ordered},
            round_data=ordered,
            summary=summary_rows,
            use_latex=_USE_LATEX,
        ), f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
