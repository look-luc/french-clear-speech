import os
from functools import partial
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import concatenate_datasets
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

        # Define your specific audio files directly
        audio_files = [
            script_dir.parent.parent / "french-clear-speech" / "data" / "DATA_AnnaL_tache01_11025_1.wav",
            script_dir.parent.parent / "french-clear-speech" / "data" / "DATA_AnnaL_tache02_11025_128.wav",
        ]

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

        with torch.inference_mode():
            for n_type in snr_targets:
                print("\n")
                print("|" + "=" * (len(str(n_type)) + len("snr gap: ")) + "|")
                print(f"\n|snr gap: {n_type}|\n")
                print("|" + "=" * (len(str(n_type)) + len("snr gap: ")) + "|")

                confidence_scores = []
                outputs[n_type] = {
                    "confidence_list": [],
                    "avg confidence": 0.0,
                }

                for audio in audio_files:
                    audio_array, sr = librosa.load(audio, sr=None, mono=True)
                    if sr != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    _, batch_confidences = model.transcribe(
                        audio_array=audio_array,
                        noise_type=None,
                        snr_db=n_type,
                        sampling_rate=sr,
                        max_new_tokens=128,
                        return_transcriptions=False,
                    )
                    outputs[n_type]["confidence_list"].extend(batch_confidences)
                    confidence_scores.extend(batch_confidences)

                outputs[n_type]["avg confidence"] = (
                    np.mean(confidence_scores) if confidence_scores else 0.0
                )

            for types in noise_types:
                print("\n")
                print("|" + "=" * (len(str(types)) + len("noise type: ")) + "|")
                print(f"\n|noise type: {types}|\n")
                print("|" + "=" * (len(str(types)) + len("noise type: ")) + "|")

                confidence_scores = []
                outputs[types] = {
                    "confidence_list": [],
                    "avg confidence": 0.0,
                }

                for audio in audio_files:
                    audio_array, sr = librosa.load(audio, sr=None, mono=True)

                    # Resample to 16000 Hz here as well
                    if sr != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                        sr = 16000

                    _, batch_confidences = model.transcribe(
                        audio_array=audio_array,
                        noise_type=types,
                        sampling_rate=sr,
                        max_new_tokens=128,
                        return_transcriptions=False,
                    )
                    outputs[types]["confidence_list"].extend(batch_confidences)
                    confidence_scores.extend(batch_confidences)

                outputs[types]["avg confidence"] = (
                    np.mean(confidence_scores) if confidence_scores else 0.0
                )

            # 2. Evaluate noise types on the specific audio files
            for types in noise_types:
                print("\n")
                print("|" + "=" * (len(str(types)) + len("noise type: ")) + "|")
                print(f"\n|noise type: {types}|\n")
                print("|" + "=" * (len(str(types)) + len("noise type: ")) + "|")

                confidence_scores = []
                outputs[types] = {
                    "confidence_list": [],
                    "avg confidence": 0.0,
                }

                for audio in audio_files:
                    audio_array, sr = librosa.load(audio, sr=None, mono=True)
                    _, batch_confidences = model.transcribe(
                        audio_array=audio_array,
                        noise_type=types,
                        sampling_rate=sr,
                        max_new_tokens=128,
                        return_transcriptions=False,
                    )
                    outputs[types]["confidence_list"].extend(batch_confidences)
                    confidence_scores.extend(batch_confidences)

                outputs[types]["avg confidence"] = (
                    np.mean(confidence_scores) if confidence_scores else 0.0
                )

        noise_names = list(outputs.keys())
        avg_confidences = [outputs[n]["avg confidence"] for n in noise_names]

        # Bar chart generation
        fig, ax = plt.subplots(figsize=(12, 8))
        x_positions = np.arange(len(noise_names))
        bar_width = 0.5

        ax.bar(
            x_positions,
            avg_confidences,
            width=bar_width,
            color="skyblue",
            edgecolor="grey",
        )

        ax.set_xlabel("Noise Type", fontweight="bold", fontsize=12)
        ax.set_ylabel("Average Confidence", fontweight="bold", fontsize=12)
        ax.set_title(
            "Model Confidence Across Noise Types (Custom Audio)", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(noise_names, rotation=45, ha="right")

        plt.tight_layout()
        fig.savefig("avg_conf_bar_snr.png", dpi=300, bbox_inches="tight")

        # Scatter plot generation
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, noise in enumerate(noise_names):
            scores = outputs[noise]["confidence_list"]
            num_samples = len(scores)

            if num_samples == 0:
                continue

            category_x = x_positions[i]
            jitter = np.random.uniform(low=-0.15, high=0.15, size=num_samples)
            jittered_x = category_x + jitter

            ax.scatter(
                jittered_x,
                scores,
                alpha=0.7,
                edgecolors="black",
                linewidths=1,
                label=noise,
            )

        ax.set_xlabel("Noise Type", fontweight="bold", fontsize=12)
        ax.set_ylabel("Confidence", fontweight="bold", fontsize=12)
        ax.set_title(
            "Individual Confidences Across Noise Types (Custom Audio)", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(noise_names, rotation=45, ha="right")

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
