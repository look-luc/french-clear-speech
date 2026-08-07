from datasets import Audio, load_dataset


def get_data(
    processor,
    feature_extractor,
    repo_id: str="lookitsluc1/french_cleer_speech"
):
    ds_train = load_dataset(repo_id, split="train", streaming=True)
    ds_test = load_dataset(repo_id, split="test", streaming=True)

    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_test = ds_test.cast_column("audio", Audio(sampling_rate=16000))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # Extract features
        batch["input_features"] = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # Tokenize text transcript
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processed_dataset_train = ds_train.map(
        prepare_dataset,
        remove_columns=ds_train.column_names,
    )
    processed_dataset_test = ds_test.map(
        prepare_dataset,
        remove_columns=ds_test.column_names,
    )

    return processed_dataset_train, processed_dataset_test
