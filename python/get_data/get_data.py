from datasets import Audio, load_dataset


def get_data(
    processor,
    feature_extractor,
    repo_id: str = "lookitsluc1/french_cleer_speech"
):
    ds_train = load_dataset(repo_id, split="train")
    ds_test = load_dataset(repo_id, split="test")

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))

    ds_train = ds_train.filter(lambda example: example["audio"] is not None and example["audio"]["array"] is not None)
    ds_test = ds_test.filter(lambda example: example["audio"] is not None and example["audio"]["array"] is not None)

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
        remove_columns=ds_train.column_names,
        num_proc=4
    )
    processed_dataset_test = ds_test.map(
        prepare_dataset,
        remove_columns=ds_test.column_names,
        num_proc=4
    )

    return processed_dataset_train, processed_dataset_test
