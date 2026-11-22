import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def run_model(what_model: str, noise_type, snr_db, batch_size: int = 32):
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
        csv_path = Path("output.csv")

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
                        cond_records.append(
                            {
                                "audio_file": audio_name,
                                "condition": str(cond),
                                "confidence": score,
                            }
                        )

                outputs[cond]["avg confidence"] = (
                    float(np.mean(confidence_scores)) if confidence_scores else 0.0
                )

                # Checkpoint each condition immediately to CSV
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

        # Bar chart generation
        fig, ax = plt.subplots(figsize=(14, 8))
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
            "Model Confidence Across Conditions (Lecture Audio)", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(noise_names, rotation=60, ha="right", fontsize=10)

        plt.tight_layout()
        fig.savefig("avg_conf_bar_snr.png", dpi=300, bbox_inches="tight")

        # Scatter plot generation with different shapes per audio file and color-coded by file start
        fig, ax = plt.subplots(figsize=(14, 8))
        markers = ["o", "^", "s", "D", "v", "P", "X", "*"]

        audio_file_list = list(preloaded_audio.keys())
        base_names = []
        for audio_name in audio_file_list:
            base = audio_name.replace("tache01", "").replace("tache02", "")
            base = base.replace("tache01", "").replace("tache02", "").rstrip("_")
            if not base:
                base = audio_name
            base_names.append(base)

        unique_base_names = sorted(list(set(base_names)))
        colors = plt.get_cmap(
            "tab10", len(unique_base_names) if len(unique_base_names) > 0 else 1
        )
        base_color_map = {base: colors(i) for i, base in enumerate(unique_base_names)}

        for i, cond in enumerate(condition_keys):
            category_x = x_positions[i]

            for audio_idx, audio_name in enumerate(preloaded_audio.keys()):
                has_cl = any(sub in audio_name for sub in ["tach01", "tache01"])
                has_cs = any(sub in audio_name for sub in ["tach02", "tache02"])

                if not has_cl and not has_cs:
                    continue

                scores = outputs[cond][audio_name]
                num_samples = len(scores)

                if num_samples == 0:
                    continue

                jitter = np.random.uniform(low=-0.15, high=0.15, size=num_samples)
                jittered_x = category_x + jitter

                marker = "o" if has_cl else "^" if has_cs else markers[-1]

                base = (
                    audio_name.replace("tache01", "")
                    .replace("tache01", "")
                    .replace("tache01", "")
                    .replace("tache01", "")
                    .replace("tache01", "")
                    .replace("tache01", "")
                )
                base = (
                    base.replace("Cl", "")
                    .replace("CS", "")
                    .replace("cl", "")
                    .replace("cs", "")
                    .replace("CL", "")
                    .replace("Cs", "")
                    .rstrip("_")
                )
                if not base:
                    base = audio_name
                file_color = base_color_map[base]

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

        ax.set_xlabel("Condition / Noise Type", fontweight="bold", fontsize=12)
        ax.set_ylabel("Confidence", fontweight="bold", fontsize=12)
        ax.set_title(
            "Individual Confidences Across Conditions (Lecture Audio)", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(noise_names, rotation=60, ha="right", fontsize=10)

        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for handle, label in zip(handles, labels):
            if label and label not in by_label and (
                "Cl" in label or "CS" in label or not ("Cl" in label or "CS" in label)
            ):
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

        plt.tight_layout()
        fig.savefig("individual_conf_scatter_snr.png", dpi=300, bbox_inches="tight")

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
            f"{what_model.capitalize()} is not one of the options. Only pick one of these: train, transcribe, or graph."
        )
