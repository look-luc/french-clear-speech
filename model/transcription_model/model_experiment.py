import os
import re
from pathlib import Path

import evaluate
import numpy as np
import scipy.signal as signal
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]
script_path = Path(__file__).resolve().parent

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

class French_Clear_Speech_Model:
    def __init__(
        self,
        model_id: str = "bofenghuang/whisper-medium-french",
        path_to_model: str = f"{script_path}/whisper-french-experiment",
        is_fine_tuned: bool = False,
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_to_model = path_to_model

        self.processor, self.feature_extractor, self.model = self._setup(
            is_fine_tuned=is_fine_tuned
        )

    def _setup(self, is_fine_tuned: bool = False):
        processor = AutoProcessor.from_pretrained(
            self.model_id, language="french", task="transcribe", token=hf_token
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, token=hf_token)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, use_safetensors=True, token=hf_token
        ).to(self.device)

        model.generation_config.language = None
        model.generation_config.task = None
        model.generation_config.use_timestamps = False

        if is_fine_tuned:
            peft_model = PeftModel.from_pretrained(model, self.path_to_model)
            peft_model = peft_model.merge_and_unload()
            peft_model.enable_input_require_grads()
            return processor, feature_extractor, peft_model
        else:
            return processor, feature_extractor, model

    def _apply_acoustic_degradation(
        self,
        audio_array: torch.Tensor,
        noise_type: str | None,
        cutoff_freq: int | None,
        snr_db: int | None,
        sample_rate: int | float = 16000.0,
    ):
        degraded_audio = audio_array.detach().cpu().numpy()

        def _simulate_noise_env(noise_type: str):
            snr_db, cutoff_freq = None, None
            match noise_type.lower():
                case "base":
                    snr_db = None
                    cutoff_freq = None
                case "crystal_void":
                    snr_db = 100
                    cutoff_freq = 22050
                case "absolute_acoustic":
                    snr_db = 95
                    cutoff_freq = 20500
                case "studio_silence":
                    snr_db = 90
                    cutoff_freq = 18500
                case "silver_spectrum":
                    snr_db = 85
                    cutoff_freq = 16500
                case "broadcast_beam":
                    snr_db = 80
                    cutoff_freq = 14200
                case "clear_horizon":
                    snr_db = 75
                    cutoff_freq = 12000
                case "VINTAGE_MAGNETIC":
                    snr_db = 70
                    cutoff_freq = 10000
                case "MAGNETIC":
                    snr_db = 65
                    cutoff_freq = 8500
                case "fog_on_the_wire":
                    snr_db = 60
                    cutoff_freq = 7000
                case "relay":
                    snr_db = 55
                    cutoff_freq = 5800
                case "shortwave_relay":
                    snr_db = 50
                    cutoff_freq = 4500
                case "whisper_in_the_rain":
                    snr_db = 45
                    cutoff_freq = 3500
                case "broadcast":
                    snr_db = 45
                    cutoff_freq = None
                case "high_end_studio":
                    snr_db = 35
                    cutoff_freq = 7000
                case "studio":
                    snr_db = 30
                    cutoff_freq = 6000
                case "quiet_home":
                    snr_db = 25
                    cutoff_freq = 5500
                case "library":
                    snr_db = 20
                    cutoff_freq = 4500
                case "mild_office":
                    snr_db = 15
                    cutoff_freq = 3400
                case "moderate_cafe":
                    snr_db = 10
                    cutoff_freq = 1500
                case "cafe":
                    snr_db = 5
                case "severe_street":
                    snr_db = 0
                    cutoff_freq = 800
                case "extreme_cocktail":
                    snr_db = -5
                    cutoff_freq = 500
            return snr_db, cutoff_freq

        if noise_type is not None:
            snr_db, cutoff_freq = _simulate_noise_env(noise_type)

        if cutoff_freq is not None:
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_freq / nyquist

            if 0 < normal_cutoff < 1.0:
                sos = signal.butter(
                    N=5, Wn=normal_cutoff, btype="low", analog=False, output="sos"
                )
                degraded_audio = signal.sosfilt(sos, degraded_audio, axis=-1)

        if snr_db is None:
            return degraded_audio
        elif snr_db is not None:
            signal_power = np.mean(np.square(degraded_audio))

            if signal_power > 0:
                noise_power = signal_power / (10 ** (snr_db / 10))
                # Match full multi-dimensional array shape (64, 80, 3000)
                noise = np.random.normal(
                    loc=0.0,
                    scale=np.sqrt(noise_power),
                    size=degraded_audio.shape,
                )
                degraded_audio += noise

            return degraded_audio

    def transcribe(
        self,
        audio_array: torch.Tensor,
        noise_type: str="studio",
        sampling_rate: int = 16000,
        cutoff_freq: int | None = 1500,
        snr_db: int | None = 10,
        temp: float = 0,
    ):
        processed_audio = self._apply_acoustic_degradation(
            audio_array, noise_type, cutoff_freq, snr_db, sampling_rate
        )
        if isinstance(processed_audio, tuple):
            processed_audio = processed_audio[0] if isinstance(processed_audio[0], np.ndarray) else processed_audio[1]
        else:
            processed_audio = processed_audio

        processed_audio = np.asarray(processed_audio, dtype=np.float64)

        if processed_audio.ndim == 1:
            return processed_audio

        processed_audio = np.mean(processed_audio, axis=-1)

        input_features = self.processor(
            processed_audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        output_ids = self.model.generate(
            input_features=input_features,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=temp,
        )

        transcription_list = self.processor.batch_decode(
            output_ids.sequences,
            skip_special_tokens=True,
        )

        transcription_text = transcription_list[0] if transcription_list else ""
        clean_transcription = re.sub(r"<\|.*?\|>|\[.*?\]", "", transcription_text).replace("fr","").strip()
        if hasattr(self.processor, "normalizer") and self.processor.normalizer is not None:
            clean_transcription = self.processor.normalizer(clean_transcription)

        confidence = self.model.compute_transition_scores(
            output_ids.sequences,
            output_ids.scores,
            normalize_logits=True,
        )
        non_pad_mask = (confidence != float('-inf')) and (~torch.isnan(confidence))
        valid_scores = confidence[non_pad_mask]

        if valid_scores.numel() > 0:
            avg_log_prob = torch.mean(valid_scores)
            conf_score = torch.exp(avg_log_prob).item
        else:
            conf_score = 0.0

        return clean_transcription, conf_score
