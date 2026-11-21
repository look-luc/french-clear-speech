import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")


def resolve_repo_dir(repo_id: str = "lookitsluc1/french_cleer_speech") -> Path:
    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            max_workers=2,
            token=hf_token,
        )
    )


def get_audio_files(
    repo_id: str = "lookitsluc1/french_cleer_speech",
    split: str = "train",
) -> dict[str, str]:
    ds = load_dataset(repo_id, split=split, token=hf_token)
    repo_dir = resolve_repo_dir(repo_id=repo_id)

    audio_files: dict[str, str] = {}
    for row in ds:
        raw_name = str(row.get("file_name", "")).strip()
        if not raw_name:
            continue

        candidate = repo_dir / raw_name
        if candidate.exists():
            audio_key = Path(raw_name).stem
            audio_files[audio_key] = str(candidate)

    if not audio_files:
        raise FileNotFoundError(
            f"No existing audio files found in snapshot {repo_dir} for {repo_id} (split={split}). "
            f"Available columns: {ds.column_names}"
        )

    return audio_files


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/french_cleer_speech",
):
    repo_dir = resolve_repo_dir(repo_id=repo_id)
    csv_path = repo_dir / "metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {repo_dir}")

    df = pd.read_csv(csv_path)
    ds_train = Dataset.from_pandas(df)
    ds_test = None

    ds_train = ds_train.map(lambda x: {"audio_path": str(repo_dir / x["file_name"])})
    ds_train = ds_train.filter(lambda x: Path(x["audio_path"]).exists())
    ds_train = ds_train.select_columns(["audio_path", "file_name"])

    def prepare_dataset(batch):
        paths = []
        audio_arrays = []
        for audio_path in batch["audio_path"]:
            paths.append(audio_path)
            array, orig_sr = sf.read(audio_path)

            if array.ndim > 1:
                array = array.mean(axis=-1)

            if orig_sr != 16000:
                array = scipy.signal.resample_poly(array, 16000, orig_sr)

            audio_arrays.append(array.astype(np.float32))

        input_features = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
        ).input_features

        return {
            "file_name": paths,
            "input_features": input_features,
        }

    processed_dataset_train = ds_train.map(
        prepare_dataset,
        batched=True,
        batch_size=128,
        remove_columns=cast(list[str], ds_train.column_names),
    )

    processed_dataset_test = None
    if ds_test is not None:
        processed_dataset_test = ds_test.map(
            prepare_dataset,
            batched=True,
            batch_size=128,
            remove_columns=cast(list[str], ds_test.column_names),
        )

    return processed_dataset_train, processed_dataset_test
