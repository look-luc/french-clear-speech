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

    def _get_noise_params(self, noise_type: str | None) -> tuple[float | None, int | None]: # returns snr_db and cutoff_freq in that order
        if noise_type is None:
            return None, None

        match noise_type.lower():
            case "base":
                return None, None
            case "crystal_void":
                return 25, 8000
            case "absolute_acoustic":
                return 20, 7000
            case "studio_silence":
                return 18, 6000
            case "silver_spectrum":
                return 15, 5200
            case "broadcast_beam":
                return 12, 4500
            case "clear_horizon":
                return 10, 3800
            case "vintage_magnetic":
                return 8, 3200
            case "magnetic":
                return 6, 2800
            case "fog_on_the_wire":
                return 4, 2400
            case "relay":
                return 2, 2000
            case "shortwave_relay":
                return 50, 4500
            case "whisper_in_the_rain":
                return 0, 1200
            case "broadcast":
                return -4, 1000
            case "high_end_studio":
                return -6, 850
            case "studio":
                return -8, 700
            case "quiet_home":
                return -10, 600
            case "library":
                return -12, 500
            case "mild_office":
                return -14, 500
            case "moderate_cafe":
                return -16, 400
            case "severe_street":
                return -18, 350
            case "extreme_cocktail":
                return -20, 300
            case _:
                return None, None

    def _apply_acoustic_degradation(
        self,
        input_features: torch.Tensor,
        noise_type: str | None = None,
        cutoff_freq: int | None = None,
        snr_db: float | None = None,
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """Calculates and applies lowpass frequency masking and noise injection on log-mel tensors."""
        if noise_type is not None:
            env_snr, env_cutoff = self._get_noise_params(noise_type)
            snr_db = snr_db if snr_db is not None else env_snr
            cutoff_freq = cutoff_freq if cutoff_freq is not None else env_cutoff

        degraded = input_features.clone()

        if cutoff_freq is not None:
            cutoff_bin = int((cutoff_freq / (sampling_rate / 2.0)) * 80)
            cutoff_bin = max(1, min(80, cutoff_bin))
            degraded[:, cutoff_bin:, :] = -80.0  # Log-mel floor

        if snr_db is not None:
            noise_std = 10.0 ** (-snr_db / 20.0)
            noise = torch.randn_like(degraded) * noise_std
            degraded = degraded + noise

        return degraded

    def transcribe(
        self,
        audio_array: torch.Tensor | np.ndarray,
        noise_type: str = "studio",
        sampling_rate: int = 16000,
        cutoff_freq: int | None = None,
        snr_db: int | None = None,
        temp: float = 0.0,
    ):
        input_features = torch.as_tensor(audio_array, device=self.device).float()

        degraded_features = self._apply_acoustic_degradation(
            input_features=input_features,
            noise_type=noise_type,
            cutoff_freq=cutoff_freq,
            snr_db=snr_db,
            sampling_rate=sampling_rate,
        )

        output_ids = self.model.generate(
            input_features=degraded_features,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=(temp > 0),
            temperature=temp,
        )

        transcription_list = self.processor.batch_decode(
            output_ids.sequences,
            skip_special_tokens=True,
        )

        clean_transcriptions = [
            re.sub(r"<\|.*?\|>|\[.*?\]", "", t).replace("fr", "").strip()
            for t in transcription_list
        ]

        transition_scores = self.model.compute_transition_scores(
            output_ids.sequences,
            output_ids.scores,
            normalize_logits=True,
        )

        conf_scores = []
        for seq_scores in transition_scores:
            valid_mask = (seq_scores != float("-inf")) & (~torch.isnan(seq_scores))
            valid_tok_scores = seq_scores[valid_mask]

            if valid_tok_scores.numel() > 0:
                sample_conf = torch.exp(torch.mean(valid_tok_scores)).item()
            else:
                sample_conf = 0.0
            conf_scores.append(sample_conf)

        return clean_transcriptions, conf_scores
