import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/tache_data",
):
    local_praat_dir = Path(__file__).resolve().parents[2] / "praat" / "data"

    try:
        repo_dir = Path(
            snapshot_download(
                repo_id,
                repo_type="dataset",
                max_workers=2,
                token=hf_token
            )
        )
    except (HfHubHTTPError, Exception) as e:
        print(f"Warning: Hub download rate limited or failed ({e}). Falling back to local data.")
        repo_dir = local_praat_dir

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
            sampling_rate=16000
        ).input_features

        return {
            "file_name": paths,
            "input_features": input_features,
        }

    processed_dataset_train = ds_train.map(
        prepare_dataset,
        batched=True,
        batch_size=128,
        remove_columns=cast(list[str], ds_train.column_names)
    )

    processed_dataset_test = None
    if ds_test is not None:
        processed_dataset_test = ds_test.map(
            prepare_dataset,
            batched=True,
            batch_size=128,
            remove_columns=cast(list[str], ds_test.column_names)
        )

    return processed_dataset_train, processed_dataset_test
