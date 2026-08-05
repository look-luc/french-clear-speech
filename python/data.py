from pathlib import Path

import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent

def get_train_test_dataloaders(
    processor,
    feature_extractor,
    data_collate,
    audio_dir=f"{BASE_DIR}/praat/data",
    txt_dir=f"{BASE_DIR}/praat/data",
    test_size=0.2
):
    audio_path, txt_path = Path(audio_dir), Path(txt_dir)
    all_pairs = []
    for wav_file in sorted(audio_path.glob("*.wav")):
        txt_file = txt_path / f"{wav_file.stem}.txt"
        if txt_file.exists():
            all_pairs.append((wav_file, txt_file))

    train_pairs, test_pairs = train_test_split(all_pairs, test_size=test_size, random_state=42)

    train_dataset = Data(processor, feature_extractor, train_pairs)
    test_dataset = Data(processor, feature_extractor, test_pairs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: data_collate(b, processor, feature_extractor)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: data_collate(b, processor, feature_extractor)
    )

    return train_loader, test_loader

class Data(Dataset):
    def __init__(self, processor, feature_extractor, data_pairs):
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.data_pairs = data_pairs

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx: int) -> dict:
        wav_path, txt_path = self.data_pairs[idx]

        waveform, sample_rate = torchaudio.load(str(wav_path))

        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform_1d = waveform.squeeze(0)

        outputs = self.feature_extractor(waveform_1d, sampling_rate=16000, return_tensors="pt")
        extracted_features = outputs.input_features[0]

        transcript = txt_path.read_text(encoding="utf-8").strip()
        label_ids = self.processor.tokenizer(transcript).input_ids

        return {"input_features": extracted_features, "labels": label_ids}
