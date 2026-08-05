from pathlib import Path

import numpy as np
import torch
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
        self.audio_path = audio_path
        self.txt_path = txt_path

    def get_data(self):
        wav_files = [p for p in Path(self.audio_path).iterdir() if p.is_file() and str(p)[-4:]==".wav"]
        txt_files = [p for p in Path(self.audio_path).iterdir() if p.is_file() and str(p)[-4:]==".txt"]

        labels = []
        audio_files = []
        for wav, txt in zip(wav_files, txt_files):
            waveform, sample_rate = torchaudio.load(str(wav))

            if sample_rate != 16000:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            transcript = np.loadtxt(txt, delimiter=" ")
            transcript_tensor = torch.from_numpy(transcript)

            waveform_1d = waveform.squeeze(0)

            outputs = self.feature_extractor(waveform_1d, sampling_rate=16000, return_tensors="pt")
            extracted_features = outputs.input_features[0]

            label_ids = self.processor.tokenizer(transcript_tensor).input_ids

            audio_files.append(extracted_features)
            labels.append(label_ids)

        return {"input_features": audio_files, "labels": labels}
