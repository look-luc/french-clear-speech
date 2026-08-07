from pathlib import Path
from typing import cast

from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/french_cleer_speech"
):
    repo_dir = Path(snapshot_download("lookitsluc1/french_cleer_speech", repo_type="dataset"))

    ds_train = load_dataset("lookitsluc1/french_cleer_speech", split="train")
    ds_test = load_dataset("lookitsluc1/french_cleer_speech", split="test")

    ds_train = ds_train.map(lambda x: {"audio": str(repo_dir / x["file_name"])})
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))

    ds_test = ds_train.map(lambda x: {"audio": str(repo_dir / x["file_name"])})
    ds_test = ds_test.cast_column("file_name", Audio(sampling_rate=16000))

    ds_train = ds_train.filter(lambda example: example["file_name"] is not None and example["file_name"]["array"] is not None)
    ds_test = ds_test.filter(lambda example: example["file_name"] is not None and example["file_name"]["array"] is not None)

    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processed_dataset_train = ds_train.map(
        prepare_dataset,
        remove_columns=cast(list[str],ds_train.column_names,)
    )
    processed_dataset_test = ds_test.map(
        prepare_dataset,
        remove_columns=cast(list[str],ds_test.column_names,)
    )

    return processed_dataset_train, processed_dataset_test
