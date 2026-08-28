import sys
from pathlib import Path

import evaluate
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]

if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from model.get_data.get_data import get_data

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def simple_progress(iterable, desc: str = "Evaluating Base Model"):
    total = len(iterable)
    for i, item in enumerate(iterable):
        print(
            f"\r{desc}: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)",
            end="",
            flush=True,
        )
        yield item


class French_Speech_text_base:
    def __init__(
        self,
        model_id: str = "bofenghuang/whisper-medium-french",
        level_tweak: float = 0.0,
        device: str = "",
    ) -> None:
        if device == "" or device is None:
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model_id = model_id
        (
            self.processor,
            self.model,
            self.train_split,
            self.test_split,
        ) = self._setup()

        self.normalizer = self.processor.tokenizer.basic_normalize
        self.level_tweak = level_tweak

        self.eval_dataset = concatenate_datasets(
            [self.train_split, self.test_split]
        )

        self.tach_1 = self.eval_dataset.filter(lambda text: "_tache01_" in text["text"])
        self.tach_2 = self.eval_dataset.filter(lambda text: "_tache02_" in text["text"])

        self.dataloader_tach1 = DataLoader(
            self.tach_1,
            batch_size=64,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.dataloader_tach2 = DataLoader(
            self.tach_2,
            batch_size=64,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def _setup(self):
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, use_safetensors=True
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(
            self.model_id, language="french", task="transcribe"
        )

        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="french", task="transcribe"
        )
        model.generation_config.forced_decoder_ids = forced_decoder_ids

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, use_safetensors=True
        )

        train_dataset, test_dataset = get_data(processor, feature_extractor)
        model.generation_config.max_length = None

        return processor, model, train_dataset, test_dataset

    def collate_fn(self, batch):
        input_list = [item["input_features"] for item in batch]
        label_list = [item["labels"] for item in batch]

        padded_inputs = self.processor.feature_extractor.pad(
            [{"input_features": f} for f in input_list],
            return_tensors="pt",
        )

        padded_labels = self.processor.tokenizer.pad(
            [{"input_ids": label} for label in label_list],
            return_tensors="pt",
        )

        labels_tensor = padded_labels["input_ids"].masked_fill(
            padded_labels.attention_mask.ne(1), -100
        )

        if all(labels_tensor[:, 0] == self.processor.tokenizer.bos_token_id):
            labels_tensor = labels_tensor[:, 1:]

        return {
            "input_features": padded_inputs["input_features"],
            "labels": labels_tensor,
        }

    def predict(self, max_samples: int | None = None):
        dataset_tache1 = self.tach_1
        dataset_tache2 = self.tach_2
        if max_samples is not None:
            dataset_tache2 = dataset_tache2.select(range(min(max_samples, len(dataset_tache2))))
            dataset_tache1 = dataset_tache1.select(range(min(max_samples, len(dataset_tache1))))
            dataloader1 = DataLoader(
                dataset_tache1,
                batch_size=64,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            dataloader2 = DataLoader(
                dataset_tache2,
                batch_size=64,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        else:
            dataloader1 = self.dataloader_tach1
            dataloader2 = self.dataloader_tach2

        predictions = []
        references = []

        cer_score_tache1 = 0
        wer_score_tache1 = 0
        cer_score_tache2 = 0
        wer_score_tache2 = 0
        for batch in simple_progress(dataloader1, desc="Evaluating Base Model"):
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=input_features, max_new_tokens=225
                )

            pred_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            pred_texts = [self.normalizer(p) for p in pred_texts]

            valid_labels = labels.masked_fill(
                labels == -100, self.processor.tokenizer.pad_token_id
            )
            ref_texts = self.processor.batch_decode(
                valid_labels, skip_special_tokens=True
            )

            ref_texts = [self.normalizer(r) for r in ref_texts]

            predictions.extend([p.strip() for p in pred_texts])
            references.extend([r.strip() for r in ref_texts])

            print()
            cer_score_tache1 = cer_metric.compute(
                predictions=predictions, references=references
            )
            wer_score_tache1 = wer_metric.compute(
                predictions=predictions, references=references
            )

        for batch in simple_progress(dataloader2, desc="Evaluating Base Model"):
            input_features = batch["input_features"].to(self.device)
            labels = batch["labels"]

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=input_features, max_new_tokens=225
                )

            pred_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            pred_texts = [self.normalizer(p) for p in pred_texts]

            valid_labels = labels.masked_fill(
                labels == -100, self.processor.tokenizer.pad_token_id
            )
            ref_texts = self.processor.batch_decode(
                valid_labels, skip_special_tokens=True
            )

            ref_texts = [self.normalizer(r) for r in ref_texts]

            predictions.extend([p.strip() for p in pred_texts])
            references.extend([r.strip() for r in ref_texts])

            print()
            cer_score_tache2 = cer_metric.compute(
                predictions=predictions, references=references
            )
            wer_score_tache2 = wer_metric.compute(
                predictions=predictions, references=references
            )

        cer_score = {"tache 1": cer_score_tache1, "tache 2": cer_score_tache2}
        wer_score = {"tache 1": wer_score_tache1, "tache 2": wer_score_tache2}
        return {"CER": cer_score, "WER": wer_score}
