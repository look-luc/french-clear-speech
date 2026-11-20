import os
import tempfile
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
    repo_dir = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            max_workers=2,
            token=hf_token,
        )
    )
    return repo_dir


def _extract_audio_field(row: dict) -> dict | None:
    if isinstance(row.get("audio"), dict):
        return row["audio"]
    for value in row.values():
        if isinstance(value, dict) and ("array" in value or "path" in value):
            return value
    return None


def get_audio_files(
    repo_id: str = "lookitsluc1/french_cleer_speech",
    split: str = "train",
) -> dict[str, str]:
    ds = load_dataset(repo_id, split=split, token=hf_token)
    cache_dir = Path(tempfile.gettempdir()) / f"hf_audio_{repo_id.replace('/', '_')}_{split}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_files: dict[str, str] = {}
    for idx, row in enumerate(ds):
        raw_name = str(row.get("file_name") or row.get("id") or f"sample_{idx}").strip()
        audio_key = Path(raw_name).stem or f"sample_{idx}"

        audio_info = _extract_audio_field(row)
        if audio_info is None:
            continue

        candidate_path = audio_info.get("path")
        if candidate_path and Path(candidate_path).exists():
            audio_files[audio_key] = str(candidate_path)
            continue

        array = audio_info.get("array")
        sr = audio_info.get("sampling_rate", 16000)
        if array is not None:
            out_wav = cache_dir / f"{audio_key}.wav"
            sf.write(out_wav, np.asarray(array, dtype=np.float32), int(sr))
            audio_files[audio_key] = str(out_wav)

    if not audio_files:
        raise RuntimeError(
            f"Could not extract any audio files from {repo_id} (split={split}). "
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
