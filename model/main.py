import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch
from dotenv import load_dotenv

from .get_data.get_data import get_audio_files
from .graph_metric.graph import metrics_graph
from .transcription_model.base_model import French_Speech_text_base
from .transcription_model.model import French_Speech_text
from .transcription_model.model_experiment import (
    French_Clear_Speech_Model as experiment_model,
)
from .transcription_model.model_experiment import Qwen_clear_speech_model as qwen

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

script_dir = Path(__file__).resolve().parent
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _normalize_base_name(audio_name: str) -> str:
  base = audio_name.replace("tache01", "").replace("tache02", "")
  for token in ("Cl", "CS", "cl", "cs", "CL", "Cs"):
    base = base.replace(token, "")
  base = base.rstrip("_")
  return base if base else audio_name


def _load_single_audio_16k(item: tuple[str, str]) -> tuple[str, np.ndarray]:
  audio_name, audio_path = item
  array, sr = sf.read(audio_path, always_2d=False)
  if array.ndim > 1:
    array = np.mean(array, axis=-1)
  if sr != 16000:
    array = scipy.signal.resample_poly(array, 16000, sr)
  return audio_name, np.asarray(array, dtype=np.float32)


def _preload_audio_16k(
    audio_files: dict[str, str], max_workers: int = 8
) -> dict[str, np.ndarray]:
  items = list(audio_files.items())
  preloaded: dict[str, np.ndarray] = {}
  with ThreadPoolExecutor(max_workers=max_workers) as pool:
    for audio_name, audio_array in pool.map(_load_single_audio_16k, items):
      preloaded[audio_name] = audio_array
  return preloaded


def _save_figure_safely(fig: plt.Figure, target_path: Path, dpi: int = 300) -> None:
  target_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_fd, tmp_name = tempfile.mkstemp(
      prefix=f".{target_path.stem}_",
      suffix=".tmp.png",
      dir=str(target_path.parent),
  )
  os.close(tmp_fd)
  tmp_path = Path(tmp_name)
  try:
    fig.canvas.draw()
    fig.savefig(tmp_path, format="png", dpi=dpi)
    if not tmp_path.exists() or tmp_path.stat().st_size < len(PNG_MAGIC):
      raise RuntimeError(f"Saved figure at {tmp_path} is empty or truncated.")
    with open(tmp_path, "rb") as f:
      header = f.read(len(PNG_MAGIC))
    if header != PNG_MAGIC:
      raise RuntimeError(
          f"Saved file at {tmp_path} does not have a valid PNG header: {header!r}"
      )
    os.replace(tmp_path, target_path)
    print(f"[Plot OK] Wrote valid PNG to {target_path} ({target_path.stat().st_size} bytes)")
  except Exception as exc:
    print(f"[Plot ERROR] Failed to save {target_path}: {type(exc).__name__}: {exc}")
    raise
  finally:
    plt.close(fig)
    if tmp_path.exists():
      tmp_path.unlink(missing_ok=True)


def _plot_bar_chart(
    condition_keys: list,
    noise_names: list[str],
    avg_confidences: list[float],
    output_path: Path,
) -> None:
  fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
  x_positions = np.arange(len(condition_keys))
  bar_width = 0.5

  ax.bar(
      x_positions,
      avg_confidences,
      width=bar_width,
      color="skyblue",
      edgecolor="grey",
  )

  ax.set_xlabel("Condition / Noise Type", fontweight="bold", fontsize=12)
  ax.set_ylabel("Average Confidence", fontweight="bold", fontsize=12)
  ax.set_title(
      "Model Confidence Across Conditions (Lecture Audio)",
      fontweight="bold",
      fontsize=14,
  )
  ax.set_xticks(x_positions)
  ax.set_xticklabels(noise_names, rotation=60, ha="right", fontsize=10)

  _save_figure_safely(fig, output_path, dpi=300)


def _plot_scatter_chart(
    condition_keys: list,
    noise_names: list[str],
    outputs: dict,
    preloaded_audio: dict[str, np.ndarray],
    output_path: Path,
) -> None:
  rng = np.random.default_rng(42)
  fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
  x_positions = np.arange(len(condition_keys))
  markers = ["o", "^", "s", "D", "v", "P", "X", "*"]

  audio_file_list = list(preloaded_audio.keys())
  base_names = [_normalize_base_name(audio_name) for audio_name in audio_file_list]
  unique_base_names = sorted(set(base_names))
  cmap = plt.get_cmap(
      "tab10", len(unique_base_names) if len(unique_base_names) > 0 else 1
  )
  base_color_map = {base: cmap(i) for i, base in enumerate(unique_base_names)}

  plotted_points = 0
  for i, cond in enumerate(condition_keys):
    category_x = x_positions[i]

    for audio_name in preloaded_audio.keys():
      has_cl = any(sub in audio_name for sub in ["tach01", "tache01"])
      has_cs = any(sub in audio_name for sub in ["tach02", "tache02"])

      if not has_cl and not has_cs:
        continue

      scores = outputs[cond][audio_name]
      num_samples = len(scores)
      if num_samples == 0:
        continue

      jitter = rng.uniform(low=-0.15, high=0.15, size=num_samples)
      jittered_x = category_x + jitter

      marker = "o" if has_cl else "^" if has_cs else markers[-1]
      base = _normalize_base_name(audio_name)
      file_color = base_color_map.get(base, cmap(0))

      ax.scatter(
          jittered_x,
          scores,
          color=file_color,
          alpha=0.7,
          edgecolors="black",
          linewidths=1,
          marker=marker,
          label=audio_name if i == 0 else "",
      )
      plotted_points += num_samples

  print(f"[Plot Info] Scatter plotted {plotted_points} points across {len(condition_keys)} conditions.")

  ax.set_xlabel("Condition / Noise Type", fontweight="bold", fontsize=12)
  ax.set_ylabel("Confidence", fontweight="bold", fontsize=12)
  ax.set_title(
      "Individual Confidences Across Conditions (Lecture Audio)",
      fontweight="bold",
      fontsize=14,
  )
  ax.set_xticks(x_positions)
  ax.set_xticklabels(noise_names, rotation=60, ha="right", fontsize=10)

  handles, labels = ax.get_legend_handles_labels()
  by_label = {}
  for handle, label in zip(handles, labels):
    if label and label not in by_label:
      by_label[label] = handle

  if by_label:
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Audio File",
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

  _save_figure_safely(fig, output_path, dpi=300)


def run_model(what_model: str, noise_type, snr_db, batch_size: int = 64):
  if what_model == "experiment":
    run_start = time.perf_counter()
    outputs = {}
    model = experiment_model(use_compile=False)
    audio_files = get_audio_files()
    preloaded_audio = _preload_audio_16k(audio_files, max_workers=8)
    audio_items = list(preloaded_audio.items())

    noise_types = [
        "broadcast",
    ]
    condition_keys = noise_types
    csv_path = script_dir / "output.csv"

    with torch.inference_mode():
      for idx, cond in enumerate(condition_keys):
        cond_start = time.perf_counter()
        is_snr = isinstance(cond, (int, float))
        print("\n")
        print("|" + "=" * (len(str(cond)) + len("condition: ")) + "|")
        print(f"\n|condition: {cond}|\n")
        print("|" + "=" * (len(str(cond)) + len("condition: ")) + "|")

        confidence_scores = []
        outputs[cond] = {
            "avg confidence": 0.0,
        }
        for audio_name in preloaded_audio.keys():
          outputs[cond][audio_name] = []

        cond_records = []
        for start_idx in range(0, len(audio_items), batch_size):
          batch = audio_items[start_idx : start_idx + batch_size]
          batch_names = [name for name, _ in batch]
          batch_arrays = [arr for _, arr in batch]

          if is_snr:
            _, batch_confidences = model.transcribe(
                audio_array=batch_arrays,
                noise_type=None,
                snr_db=cond,
                sampling_rate=16000,
                max_new_tokens=128,
                return_transcriptions=False,
                num_beams=1,
            )
          else:
            _, batch_confidences = model.transcribe(
                audio_array=batch_arrays,
                noise_type=cond,
                sampling_rate=16000,
                max_new_tokens=128,
                return_transcriptions=False,
                num_beams=1,
            )

          for audio_name, score in zip(batch_names, batch_confidences):
            outputs[cond][audio_name].append(score)
            confidence_scores.append(score)
            cond_records.append({
                "audio_file": audio_name,
                "condition": str(cond),
                "confidence": score,
            })

        outputs[cond]["avg confidence"] = (
            float(np.mean(confidence_scores)) if confidence_scores else 0.0
        )

        cond_df = pd.DataFrame(cond_records)
        write_header = not csv_path.exists()
        cond_df.to_csv(csv_path, mode="a", header=write_header, index=False)

        cond_elapsed = time.perf_counter() - cond_start
        total_elapsed = time.perf_counter() - run_start
        remaining_conds = len(condition_keys) - (idx + 1)
        eta_seconds = cond_elapsed * remaining_conds
        print(
            f"[Timing] cond={cond} took {cond_elapsed:.1f}s | "
            f"elapsed={total_elapsed/60:.1f}m | est_remaining={eta_seconds/60:.1f}m"
        )

    noise_names = [str(k) for k in condition_keys]
    avg_confidences = [outputs[cond]["avg confidence"] for cond in condition_keys]

    _plot_bar_chart(
        condition_keys=condition_keys,
        noise_names=noise_names,
        avg_confidences=avg_confidences,
        output_path=script_dir / "avg_conf_bar_snr.png",
    )
    _plot_scatter_chart(
        condition_keys=condition_keys,
        noise_names=noise_names,
        outputs=outputs,
        preloaded_audio=preloaded_audio,
        output_path=script_dir / "individual_conf_scatter_snr.png",
    )

  elif what_model == "base":
    french_speech_transcription = French_Speech_text_base()
    output = ""
    try:
      output = french_speech_transcription.predict()
      save_status = f"Successfully trained model setup: {french_speech_transcription.model_id}"
    except Exception as e:
      save_status = f"ERROR with model: {str(e)}"
    output_dir = Path("./model/french_speech_transcription_base_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
      file.write(f"{save_status}\n{output}")
  elif what_model == "train":
    french_speech_transcription = French_Speech_text()
    try:
      french_speech_transcription.train()
      save_status = f"Successfully trained model setup: {french_speech_transcription.model_id}"
    except Exception as e:
      save_status = f"ERROR with model: {str(e)}"

    output_dir = Path("./model/french_speech_transcription_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
      file.write(save_status)
  elif what_model == "graph":
    metrics_graph()
  else:
    raise ValueError(
        f"{what_model.capitalize()} is not one of the options. Only pick one of"
        " these: train, transcribe, or graph."
    )
