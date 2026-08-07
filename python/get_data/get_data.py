import os
import re
from pathlib import Path
from typing import cast

import soundfile as sf
from datasets import (
    Audio,
    load_dataset,
)
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/french_cleer_speech"
):
    local_praat_dir = Path(__file__).resolve().parents[2] / "praat" / "data"

    try:
        repo_dir = Path(
            snapshot_download(
                repo_id,
                repo_type="dataset",
                max_workers=2,
                token=os.getenv("HF_TOKEN")
            )
        )
    except (HfHubHTTPError, Exception) as e:
        print(f"Warning: Hub download rate limited or failed ({e}). Falling back to local data.")
        repo_dir = local_praat_dir

    ds_train = load_dataset(repo_id, split="train")
    ds_test = load_dataset(repo_id, split="test")

    ds_train = ds_train.map(lambda x: {"audio": str(repo_dir / x["file_name"])})
    ds_test = ds_test.map(lambda x: {"audio": str(repo_dir / x["file_name"])})

    ds_train = ds_train.filter(lambda x: Path(x["audio"]).exists())
    ds_test = ds_test.filter(lambda x: Path(x["audio"]).exists())

    # Disable automatic decoding to avoid torchcodec dependency
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    def prepare_dataset(batch):
        audio_arrays = []
        for path in batch["audio"]:
            audio_path = path["path"] if isinstance(path, dict) else path
            array, _ = sf.read(audio_path)
            audio_arrays.append(array)

        input_features = feature_extractor(
            audio_arrays,
            sampling_rate=16000
        ).input_features

        cleaned_texts = []
        pattern = r"\bl'\s+"
        for text in batch["text"]:
            text = text.replace("sp ", "").replace(" sp", "").replace("{ns}", "")
            text = re.sub(pattern, "l'", text, flags=re.IGNORECASE)
            cleaned_texts.append(text)

        labels = processor.tokenizer(cleaned_texts).input_ids

        return {
            "input_features": input_features,
            "labels": labels
        }

    processed_dataset_train = ds_train.map(
        prepare_dataset,
        batched=True,
        batch_size=16,
        remove_columns=cast(list[str], ds_train.column_names)
    )
    processed_dataset_test = ds_test.map(
        prepare_dataset,
        batched=True,
        batch_size=16,
        remove_columns=cast(list[str], ds_test.column_names)
    )

    return processed_dataset_train, processed_dataset_test
