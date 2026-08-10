import os
import re
from pathlib import Path
from typing import cast

import numpy as np
import scipy.signal
import soundfile as sf
from datasets import (
    Audio,
    concatenate_datasets,
    load_dataset,
)
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/french_cleer_speech",
    include_common_voice: bool = True,
    common_voice_repo: str = "mozilla-foundation/common_voice_17_0",
):
    local_praat_dir = Path(__file__).resolve().parents[2] / "praat" / "data"

    # --- 1. Load Local Dataset ---
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

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000, decode=False))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    ds_train = ds_train.select_columns(["audio", "text"])
    ds_test = ds_test.select_columns(["audio", "text"])

    cv_train = load_dataset("fsicoli/common_voice_17_0", "fr", split="train")
    cv_test = load_dataset("fsicoli/common_voice_17_0", "fr", split="test")

    cv_train = cv_train.rename_column("sentence", "text")
    cv_test = cv_test.rename_column("sentence", "text")

    cv_train = cv_train.cast_column("audio", Audio(sampling_rate=16000))
    cv_test = cv_test.cast_column("audio", Audio(sampling_rate=16000))

    cv_train = cv_train.select_columns(["audio", "text"])
    cv_test = cv_test.select_columns(["audio", "text"])

    ds_train = concatenate_datasets([ds_train, cv_train])
    ds_test = concatenate_datasets([ds_test, cv_test])

    def prepare_dataset(batch):
        audio_arrays = []
        for item in batch["audio"]:
            if isinstance(item, dict) and "array" in item and item["array"] is not None:
                array = item["array"]
                orig_sr = item.get("sampling_rate", 16000)
            else:
                audio_path = item["path"] if isinstance(item, dict) else item
                array, orig_sr = sf.read(audio_path)

            if array.ndim > 1:
                array = array.mean(axis=-1)

            if orig_sr != 16000:
                array = scipy.signal.resample_poly(array, 16000, orig_sr)

            audio_arrays.append(array.astype(np.float32))

        input_features = feature_extractor(
            audio_arrays,
            sampling_rate=16000
        ).input_features

        cleaned_texts = []
        pattern_tags = r"\{.*?\}|\bsp\b"
        pattern_l = r"\bl'\s+"
        pattern_spaces = r"\s+"

        for text in batch["text"]:
            text = text or ""
            text = re.sub(pattern_tags, "", text)
            text = re.sub(pattern_l, "l'", text, flags=re.IGNORECASE)
            text = re.sub(pattern_spaces, " ", text).strip()
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
