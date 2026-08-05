from pathlib import Path

import torchaudio
import torchaudio.transforms as T

BASE_DIR = Path(__file__).resolve().parent.parent
class Data:
    def __init__(
        self,
        processor,
        feature_extractor,
        audio_path:str=f"{BASE_DIR}/praat/data",
        txt_path:str=f"{BASE_DIR}/praat/data",
    ) -> None:
        self.processor = processor
        self.feature_extractor = feature_extractor
        self.audio_path = Path(audio_path)
        self.txt_path = Path(txt_path)

        self.data_pairs = []
        for wav_file in sorted(self.audio_path.glob("*.wav")):
            txt_file = self.txt_path / f"{wav_file.stem}.txt"
            if txt_file.exists():
                self.data_pairs.append((wav_file, txt_file))

    def __len__(self) -> int:
            return len(self.data_pairs)

    def get_data(self, idx:int):
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
