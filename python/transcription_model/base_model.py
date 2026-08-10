import sys
from pathlib import Path

import evaluate
import torch
from datasets import concatenate_datasets
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoProcessor

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]

if str(python_dir) not in sys.path:
    sys.path.append(str(python_dir))

from python.get_data.get_data import get_data

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
        self.level_tweak = level_tweak

        self.eval_dataset = concatenate_datasets(
            [self.train_split, self.test_split]
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

        return processor, model, train_dataset, test_dataset

    def predict(self, max_samples: int | None = None):
        predictions = []
        references = []

        dataset = self.eval_dataset
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        for sample in simple_progress(dataset, desc="Evaluating Base Model"):
            input_features = torch.tensor(sample["input_features"], dtype=torch.float32)
            if input_features.ndim == 2:
                input_features = input_features.unsqueeze(0)
            input_features = input_features.to(self.device)

            reference_text = self.processor.tokenizer.decode(
                sample["labels"], skip_special_tokens=True
            )

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_features=input_features, max_new_tokens=225
                )

            pred_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            predictions.append(pred_text.strip())
            references.append(reference_text.strip())

        print()
        cer_score = cer_metric.compute(
            predictions=predictions, references=references
        )
        wer_score = wer_metric.compute(
            predictions=predictions, references=references
        )

        return {"CER": cer_score, "WER": wer_score}
