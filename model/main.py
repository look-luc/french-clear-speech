import os
from functools import partial
from pathlib import Path
from typing import Any, cast

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from model.get_data.get_data import get_data

from .graph_metric.graph import metrics_graph
from .transcription_model.base_model import French_Speech_text_base, simple_progress
from .transcription_model.model import French_Speech_text
from .transcription_model.model_experiment import (
    French_Clear_Speech_Model as experiment_model,
)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

script_dir = Path(__file__).resolve().parent

def run_model(what_model: str, noise_type, snr_db):
    if what_model == "experiment":
        outputs = {}
        model = experiment_model()

        # Define the directory containing your local lecture audio files
        lecture_dir = script_dir.parent / "data" / "lecture"
        audio_extensions = [".wav", ".mp3", ".m4a", ".flac"]

        audio_files = {}
        if lecture_dir.exists():
            for audio_path in lecture_dir.iterdir():
                if audio_path.suffix.lower() in audio_extensions:
                    audio_name = audio_path.stem
                    audio_files[audio_name] = audio_path

        if not audio_files:
            print(f"Warning: No audio files found in {lecture_dir}")

        noise_types = [
            "base",
            "crystal_void",
            "absolute_acoustic",
            "studio_silence",
            "silver_spectrum",
            "broadcast_beam",
            "clear_horizon",
            "vintage_magnetic",
            "magnetic",
            "fog_on_the_wire",
            "relay",
            "shortwave_relay",
            "whisper_in_the_rain",
            "broadcast",
            "high_end_studio",
            "studio",
            "quiet_home",
            "library",
            "mild_office",
            "moderate_cafe",
            "severe_street",
            "extreme_cocktail"
        ]
        snr_targets = [-2.5, -7.5]
        condition_keys = list(snr_targets) + noise_types
        with torch.inference_mode():
            for cond in condition_keys:
                is_snr = isinstance(cond, (int, float))
                print("\n")
                print("|" + "=" * (len(str(cond)) + len("condition: ")) + "|")
                print(f"\n|condition: {cond}|\n")
                print("|" + "=" * (len(str(cond)) + len("condition: ")) + "|")

                confidence_scores = []
                outputs[cond] = {
                    "avg confidence": 0.0,
                }
                for audio_name in audio_files.keys():
                    outputs[cond][audio_name] = []

                for audio_name, audio_path in audio_files.items():
                    audio_array, sr = librosa.load(audio_path, sr=None, mono=True)

                    if sr != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    if is_snr:
                        _, batch_confidences = model.transcribe(
                            audio_array=audio_array,
                            noise_type=None,
                            snr_db=cond,
                            sampling_rate=sr,
                            max_new_tokens=128,
                            return_transcriptions=False,
                        )
                    else:
                        _, batch_confidences = model.transcribe(
                            audio_array=audio_array,
                            noise_type=cond,
                            sampling_rate=sr,
                            max_new_tokens=128,
                            return_transcriptions=False,
                        )

                    outputs[cond][audio_name].extend(batch_confidences)
                    confidence_scores.extend(batch_confidences)

                outputs[cond]["avg confidence"] = (
                    np.mean(confidence_scores) if confidence_scores else 0.0
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

        # Scatter plot generation with different shapes per audio file
        fig, ax = plt.subplots(figsize=(14, 8))
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

        for i, cond in enumerate(condition_keys):
            category_x = x_positions[i]

            for audio_idx, audio_name in enumerate(audio_files.keys()):
                has_cl = any(part == "CL" for part in audio_name.split("_"))
                has_cs = any(part == "CS" for part in audio_name.split("_"))

                if not has_cl and not has_cs:
                    continue

                scores = outputs[cond][audio_name]
                num_samples = len(scores)

                if num_samples == 0:
                    continue

                jitter = np.random.uniform(low=-0.15, high=0.15, size=num_samples)
                jittered_x = category_x + jitter
                marker = 's' if has_cl else '^' if has_cs else markers[-1]

                ax.scatter(
                    jittered_x,
                    scores,
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
            if label and label not in by_label and ("CL" in label or "CS" in label):
                by_label[label] = handle

        if by_label:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                title="Audio File",
                loc="upper right",
                frameon=True,
                fontsize=8
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
