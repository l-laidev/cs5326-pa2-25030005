from argparse import ArgumentParser
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_sft(metrics, moving_window=250):
    order = ["baseline", "lr", "warmup", "norm", "final-run"]

    expToLatex = {
        "higher-warmup": r"w=5\%",
        "lower-warmup": r"w=1\%",
        "higher-lr": r"\alpha=10^{-3}",
        "lower-lr": r"\alpha=5\times10^{-6}",
        "higher-norm": "c=10.0",
        "lower-norm": "c=0.01",
    }
    
    f, ax = plt.subplots(1,2,figsize=(8,4))
    x_range = None
    for metric in order:
        for prefix in ["lower", "higher"] if metric not in ["baseline", "final-run"] else [""]:
            if metric != "final-run":
                exp = f"exp-{prefix}-{metric}" if metric != "baseline" else f"exp-{metric}"
            else:
                exp = metric
            
            if exp not in metrics: continue
        
            if x_range is None:
                x_range = len(metrics[exp]["train"])
            for phase in metrics[exp]:
                values = metrics[exp][phase]
                
                if phase == "train":
                    values = np.lib.stride_tricks.sliding_window_view(values, window_shape=moving_window).mean(axis=-1)

                step = x_range/len(values)
                start = step-1
                label = exp.replace("exp-", "").replace("-", " ").title().replace("Lr", "LR")
                if "baseline" not in exp and "final" not in exp:
                    latex = expToLatex[exp.replace("exp-", "")]
                    label = f"{label} (${latex}$)"
                
                ax_idx = 0 if phase == "train" else 1
                ax[ax_idx].plot(np.arange(start, x_range, step), values, label=label, marker="o" if phase != "train" else '')
                ax[ax_idx].grid()

                title = phase.title() + " Loss"
                if phase == "train":
                    title = f"{title}\n(moving average $w={moving_window}$)"
                
                ax[ax_idx].set_ylabel("Cross-Entropy")
                ax[ax_idx].set_xlabel("Training Step #")
                ax[ax_idx].set_title(title)
    
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1]._legend = None
    f.legend(handles, labels, loc="lower center", ncols=3)
    if len(metrics) > 1:
        f.tight_layout(rect=[0, 0.21, 1, 1])
    else:
        f.tight_layout(rect=[0, 0.1, 1, 1])
    
    return f

def plot_dpo(metrics, moving_window=250):
    f, ax = plt.subplots(2,3,figsize=(10,4))
    x_range = None
    for exp in metrics:
        if x_range is None:
            x_range = len(metrics[exp]["train_loss"])
        for metric_phase in metrics[exp]:
            values = metrics[exp][metric_phase]
            
            if "train" in metric_phase:
                values = np.lib.stride_tricks.sliding_window_view(values, window_shape=moving_window).mean(axis=-1)

            step = x_range/len(values)
            start = step-1
            
            ax_idx = 0 if "train" in metric_phase else 1
            metric_idx = 0 if "loss" in metric_phase else 1 if "acc" in metric_phase else 2

            ax[ax_idx][metric_idx].plot(np.arange(start, x_range, step), values, marker="o" if "train" not in metric_phase else '')
            ax[ax_idx][metric_idx].grid()

            axis_label = metric_phase.replace("_", " ").replace("rw", "Reward").replace("acc", "Accuracy").title()
            title = axis_label
            if "train" in metric_phase:
                title = f"{title}\n(moving average $w={moving_window}$)"
            
            # ax[ax_idx][metric_idx].set_ylabel(axis_label.replace("train", "".replace("val", "")))
            ax[ax_idx][metric_idx].set_xlabel("Training Step #")
            ax[ax_idx][metric_idx].set_title(title)

            max_, min_ = np.max(values), np.min(values)
            ax[ax_idx][metric_idx].set_yticks(np.arange(min_, max_, (max_-min_)/4))

            if "val" in metric_phase:
                best_idx = np.argmin(values) if "loss" in metric_phase else np.argmax(values)
                best_val = min_ if "loss" in metric_phase else max_
                ax[ax_idx][metric_idx].axvline(step*(best_idx+1), linestyle="-", color="purple", label=f"Best={best_val:.4f}")
                ax[ax_idx][metric_idx].legend(loc="center right")
    
    f.tight_layout()
    
    return f

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--final-only", type=bool, default=False)
    args = parser.parse_args()
    args.base = Path(args.base)

    metrics_paths = [args.base / f for f in os.listdir(args.base)]
    metrics = {}
    for p in metrics_paths:
        name = p.name.strip()
        if args.final_only:
            if "final" not in name: continue
        else:
            if "final" in name: continue
        
        with open(p / "metrics.json", "r") as f:
            metrics[name] =  json.loads(f.read())

    metrics = dict(sorted(metrics.items(), key=lambda x: x[0].split("-")[-1]))

    save_name = args.base.name
    save_suffix = 'final' if args.final_only else 'exp'
    if "sft" == args.base.name:
        f = plot_sft(metrics, moving_window=5_000 if args.final_only else 250)
        f.savefig(f"{save_name}-{save_suffix}.pdf")
    else:
        # only final-run is done for DPO
        for exp in metrics:
            f = plot_dpo({exp: metrics[exp]}, moving_window=5_000 if args.final_only else 250)
            dpo_save_name = exp.split("_")[0] + "_" + save_name
            f.savefig(f"{dpo_save_name}-{save_suffix}.pdf")
