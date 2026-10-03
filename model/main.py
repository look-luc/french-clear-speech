import os
from pathlib import Path

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

def run_model(what_model: str, noise_type, snr_db):
    if what_model == "experiment":
        outputs = {}
        model = experiment_model()

        def collate_fn(batch):
            input_list = [item["input_features"] for item in batch]
            label_list = [item["labels"] for item in batch]

            padded_inputs = model.processor.feature_extractor.pad(
                [{"input_features": f} for f in input_list],
                return_tensors="pt",
            )

            padded_labels = model.processor.tokenizer.pad(
                [{"input_ids": label} for label in label_list],
                return_tensors="pt",
            )

            labels_tensor = padded_labels["input_ids"].masked_fill(
                padded_labels.attention_mask.ne(1), -100
            )

            if all(labels_tensor[:, 0] == model.processor.tokenizer.bos_token_id):
                labels_tensor = labels_tensor[:, 1:]

            return {
                "input_features": padded_inputs["input_features"],
                "labels": labels_tensor,
            }

        train_dataset, test_dataset = get_data(
            model.processor, model.feature_extractor
        )
        combined_data = concatenate_datasets([train_dataset, test_dataset]).with_format(
            "torch"
        )
        dataloader = DataLoader(
            combined_data,
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        noise_types = [
            "base",
            "crystal_void",
            "clear_horizon",
            "shortwave_relay",
            "quiet_home",
            "extreme_cocktail",
        ]

        with torch.inference_mode():
            for n_type in noise_types:
                print("\n")
                print("|" + "=" * (len(n_type) + len("noise type: ")) + "|")
                print(f"\n|noise type: {n_type}|\n")
                print("|" + "=" * (len(n_type) + len("noise type: ")) + "|")

                confidence_scores = []
                outputs[n_type] = {
                    "confidence_list": [],
                    "avg confidence": 0.0,
                }

                for batch in simple_progress(dataloader, desc="testing noise"):
                    transcriptions, batch_confidences = model.transcribe(
                        audio_array=batch["input_features"],
                        noise_type=n_type,
                        snr_db=snr_db,
                    )

                    outputs[n_type]["confidence_list"].extend(batch_confidences)
                    confidence_scores.extend(batch_confidences)

                outputs[n_type]["avg confidence"] = (
                    np.mean(confidence_scores) if confidence_scores else 0.0
                )

        noise_names = list(outputs.keys())
        avg_confidences = [outputs[n]["avg confidence"] for n in noise_names]

        # Bar chart
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
            "Model Confidence Across Noise Types", fontweight="bold", fontsize=14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(noise_names, rotation=45, ha="right")

        plt.tight_layout()
        fig.savefig("avg_conf_bar_snr.png", dpi=300, bbox_inches="tight")

        # Scatter plot
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
            "Individual Confidences Across Noise Types", fontweight="bold", fontsize=14
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
