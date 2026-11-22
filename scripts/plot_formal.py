"""Render report charts from the locked formal experiment results."""

import argparse
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from scripts.evaluate_formal import MODELS, read_csv, read_json
from scripts.prepare_data import AGE_GROUPS


LABELS = {"cnn": "Basic CNN", "frozen": "Frozen ResNet18", "finetune": "Fine-tuned ResNet18"}
COLORS = {"cnn": "#3c6b93", "frozen": "#25796f", "finetune": "#b45d30"}


def new_figure(width, height):
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def style(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def draw_history(experiment, output):
    figure = new_figure(11, 4.2)
    axes = figure.subplots(1, 2)
    warmup = read_csv(experiment / "warmup/history.csv")
    histories = {name: read_csv(experiment / name / "history.csv") for name in MODELS}
    for axis, column, title, ylabel in zip(
        axes,
        ("train_mse", "validation_mae"),
        ("Training loss", "Validation error"),
        ("MSE (years squared)", "MAE (years)"),
        strict=True,
    ):
        axis.plot([int(row["epoch"]) for row in warmup], [float(row[column]) for row in warmup],
                  color="#686868", linestyle="--", marker="o", label="Shared ResNet18 warmup")
        for name, rows in histories.items():
            offset = 0 if name == "cnn" else len(warmup)
            axis.plot([int(row["epoch"]) + offset for row in rows], [float(row[column]) for row in rows],
                      color=COLORS[name], marker="o", markersize=3, label=LABELS[name])
        axis.set(title=title, xlabel="Epoch including warmup", ylabel=ylabel, xticks=range(1, 11))
        style(axis)
    axes[1].legend(fontsize=8, loc="upper right", frameon=False)
    figure.savefig(output / "training_curves.png", dpi=160)


def draw_overall(evaluation, output):
    results = {row["strategy"]: row for row in read_csv(evaluation / "metrics.csv")}
    figure = new_figure(8.9, 4.2)
    axes = figure.subplots(1, 2, sharey=True)
    limit = max(float(row[metric]) for row in results.values() for metric in ("mae", "rmse"))
    for axis, metric in zip(axes, ("mae", "rmse"), strict=True):
        bars = axis.bar([LABELS[name] for name in MODELS],
                        [float(results[name][metric]) for name in MODELS],
                        color=[COLORS[name] for name in MODELS], width=0.66)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set(title=f"Test {metric.upper()}", ylabel="Error (years)", ylim=(0, limit * 1.17))
        axis.tick_params(axis="x", labelsize=8)
        style(axis)
    figure.savefig(output / "overall_errors.png", dpi=160)


def draw_scatter(evaluation, output):
    predictions = {
        name: [(int(row["age"]), float(row["prediction"]))
               for row in read_csv(evaluation / f"predictions_{name}.csv")]
        for name in MODELS
    }
    points = [value for pairs in predictions.values() for pair in pairs for value in pair]
    lower, upper = min(0, min(points)) - 2, max(points) + 2
    figure = new_figure(11.7, 4.4)
    axes = figure.subplots(1, 3, sharex=True, sharey=True)
    for axis, name in zip(axes, MODELS, strict=True):
        labels, estimates = zip(*predictions[name])
        axis.plot([lower, upper], [lower, upper], color="#606060", linestyle="--", linewidth=1)
        axis.scatter(labels, estimates, s=7, color=COLORS[name], alpha=0.24, linewidths=0)
        axis.set(title=LABELS[name], xlabel="Labeled age (years)",
                 xlim=(lower, upper), ylim=(lower, upper), aspect="equal")
        style(axis)
    axes[0].set_ylabel("Predicted age (years)")
    figure.savefig(output / "age_scatter.png", dpi=160)


def draw_groups(evaluation, output):
    groups = {(row["strategy"], row["age_group"]): row
              for row in read_csv(evaluation / "age_group_metrics.csv")}
    bands = [band for _, band in AGE_GROUPS]
    figure = new_figure(10, 4.7)
    axis = figure.subplots()
    offsets = (-0.24, 0, 0.24)
    for offset, name in zip(offsets, MODELS, strict=True):
        values = [float(groups[name, band]["mae"]) for band in bands]
        axis.bar([index + offset for index in range(len(bands))], values, width=0.22,
                 color=COLORS[name], label=LABELS[name])
    axis.set_xticks(range(len(bands)), [f"{band}\nn={groups['cnn', band]['samples']}" for band in bands])
    axis.set(title="Test MAE by labeled age group", xlabel="Age group (years)", ylabel="MAE (years)")
    axis.legend(frameon=False)
    style(axis)
    figure.savefig(output / "age_group_errors.png", dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.experiment / "test"
    read_json(evaluation / "evaluation.json")
    output = args.experiment / "figures"
    output.mkdir(exist_ok=True)
    draw_history(args.experiment, output)
    draw_overall(evaluation, output)
    draw_scatter(evaluation, output)
    draw_groups(evaluation, output)


if __name__ == "__main__":
    main()
