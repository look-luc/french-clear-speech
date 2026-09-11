import os
from pathlib import Path

import numpy as np
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

def run_model(what_model:str, noise_type, cutoff_freq, snr_db):
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
        train_dataset, test_dataset = get_data(model.processor, model.feature_extractor)
        combined_data = concatenate_datasets([train_dataset, test_dataset]).with_format("torch")
        dataloader = DataLoader(
            combined_data,
            batch_size=64,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        noise_types = ["studio","mild_office","moderate_cafe","severe_street","extreme_cocktail"]
        for noise_type in noise_types:
            for batch in simple_progress(dataloader, desc="testing noise"):
                output, convidence = model.transcribe(batch["input_features"], noise_type, cutoff_freq, snr_db)
                outputs[noise_type] = {"transcription": []}
                outputs[noise_type] = {"confidence": []}
                outputs[noise_type]["transcription"].append(output)
                outputs[noise_type]["confidence"].append(convidence)
            outputs[noise_type]["avg confidence"] = np.mean(np.array(outputs[noise_type]["confidence"]))

        print(outputs)
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
        raise ValueError(f"{what_model.capitalize()} is not one of the options. Only pick one of these: train, transcribe, or graph.")
