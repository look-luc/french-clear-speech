import os
import re
from pathlib import Path

import evaluate
import numpy as np
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
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.path_to_model = path_to_model

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")

        self.processor, self.feature_extractor, self.model = self._setup(
            is_fine_tuned=is_fine_tuned
        )

    def _setup(self, is_fine_tuned: bool = False):
        processor = AutoProcessor.from_pretrained(
            self.model_id, language="french", task="transcribe", token=hf_token
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, token=hf_token
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=hf_token,
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

    def _get_noise_params(
        self, noise_type: str | None
    ) -> float | int | None:
        if noise_type is None:
            return None

        match noise_type.lower():
            case "base":
                return None
            case "crystal_void":
                return 50
            case "absolute_acoustic":
                return 45
            case "studio_silence":
                return 40
            case "silver_spectrum":
                return 35
            case "broadcast_beam":
                return 30
            case "clear_horizon":
                return 25
            case "vintage_magnetic":
                return 20
            case "magnetic":
                return 15
            case "fog_on_the_wire":
                return 10
            case "relay":
                return 5
            case "shortwave_relay":
                return 0
            case "whisper_in_the_rain":
                return -5
            case "broadcast":
                return -10
            case "high_end_studio":
                return -15
            case "studio":
                return -20
            case "quiet_home":
                return -25
            case "library":
                return -30
            case "mild_office":
                return -35
            case "moderate_cafe":
                return -40
            case "severe_street":
                return -45
            case "extreme_cocktail":
                return -50
            case _:
                return None

    def _apply_acoustic_degradation(
        self,
        input_features: torch.Tensor,
        noise_type: str | None = None,
        snr_db: float | None = None,
        sampling_rate: int = 16000,
    ) -> torch.Tensor:
        """Calculates and applies lowpass frequency masking and noise injection on log-mel tensors."""
        if noise_type is not None:
            env_snr = self._get_noise_params(noise_type)
            snr_db = snr_db if snr_db is not None else env_snr

        degraded = input_features.clone()

        if snr_db is not None:
            noise_std = 10.0 ** (-snr_db / 20.0)
            noise = torch.randn_like(degraded) * noise_std
            degraded = degraded + noise

        return degraded

    @torch.inference_mode()
    def transcribe(
        self,
        audio_array: torch.Tensor | np.ndarray,
        noise_type: str|None = "studio",
        snr_db: int | float | None = None,
        sampling_rate: int = 16000,
        temp: float = 0.0,
        max_new_tokens: int = 128,
        return_transcriptions: bool = False,
    ):
        input_features = torch.as_tensor(
            audio_array, device=self.device, dtype=self.dtype
        )

        degraded_features = self._apply_acoustic_degradation(
            input_features=input_features,
            noise_type=noise_type,
            snr_db=snr_db,
            sampling_rate=sampling_rate,
        )

        output_ids = self.model.generate(
            input_features=degraded_features,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=(temp > 0),
            temperature=temp,
        )

        clean_transcriptions = []
        if return_transcriptions:
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
