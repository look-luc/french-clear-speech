from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
def _build_df(json_path: Path):
    train_loss = {}
    grad_norm = {}
    eval_loss = {}
    eval_cer = {}
    eval_wer = {}

    temp_df = pd.read_json(json_path, typ="series")

    if "log_history" in temp_df and isinstance(temp_df["log_history"], list):
        log_entries = temp_df["log_history"]
    else:
        log_entries = []

    for element in log_entries:
        if not isinstance(element, dict):
            continue

        step = element.get("step")
        if step is None:
            continue

        if "loss" in element:
            train_loss[step] = element["loss"]
        if "grad_norm" in element:
            grad_norm[step] = element["grad_norm"]
        if "eval_loss" in element:
            eval_loss[step] = element["eval_loss"]
        if "eval_CER" in element:
            eval_cer[step] = element["eval_CER"]
        if "eval_WER" in element:
            eval_wer[step] = element["eval_WER"]

    return train_loss, grad_norm, eval_loss, eval_cer, eval_wer

def _make_graph(metric: dict[str, float], metric_name: str, color: str, output_dir: Path):
    if not metric:
        print(f"Skipping graph for {metric_name}: No log data found.")
        return

    x = torch.tensor([float(k) for k in metric.keys()]).numpy()
    y = torch.tensor([float(v) for v in metric.values()]).numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', color=color, label=metric_name.capitalize())

    ax.set_title(f'{metric_name.capitalize()} Over Steps')
    ax.set_xlabel('Steps')
    ax.set_ylabel(metric_name.capitalize())

    ax.legend()
    ax.grid(True)

    file_name = metric_name.replace(" ", "_")
    save_path = output_dir / f"{file_name}.png"
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Made graph of {metric_name} as {file_name}.png at {output_dir}")

def metrics_graph(path_to_results: str = str(BASE_DIR / "results" / "log_1000_2")):
    results_dir = Path(path_to_results)

    if results_dir.name == "graphs":
        graphs_dir = results_dir
        search_dir = results_dir.parent
    else:
        graphs_dir = results_dir / "graphs"
        search_dir = results_dir

    graphs_dir.mkdir(parents=True, exist_ok=True)

    json_files = [p for p in search_dir.rglob("*.json") if "graphs" not in p.parts]
    if not json_files:
        raise FileNotFoundError(f"No .json log files found under directory: {search_dir}")

    result_json_path = json_files[0]

    metrics = _build_df(result_json_path)
    metric_names = [
        "training loss", "gradient normalization", "evaluation loss",
        "BLEU score", "CER score", "WER score"
    ]
    colors = ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#CC79A7']

    for metric, name, color in zip(metrics, metric_names, colors):
        _make_graph(metric, name, color, graphs_dir)

    print("Finished graphing metrics.")
