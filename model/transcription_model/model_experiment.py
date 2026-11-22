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
    AutoModelForMultimodalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

script_path = Path(__file__).resolve().parent

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


class Qwen_clear_speech_model:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=self.dtype,
        )

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")

    def _get_noise_params(self, noise_type: str | None) -> float | int | None:
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
        noise_type: str | None = "studio",
        snr_db: int | float | None = None,
        sampling_rate: int = 16000,
        temp: float = 0.0,
        max_new_tokens: int = 128,
        return_transcriptions: bool = True,
    ):
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.detach().cpu().numpy()

        if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
            if audio_array.shape[0] < audio_array.shape[1]:
                audio_array = np.mean(audio_array, axis=0)
            else:
                audio_array = np.mean(audio_array, axis=1)

        inputs = self.processor.apply_transcription_request(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_features = inputs["input_features"].to(device=self.device, dtype=self.dtype)
        degraded_features = self._apply_acoustic_degradation(
            input_features=input_features,
            noise_type=noise_type,
            snr_db=snr_db,
            sampling_rate=sampling_rate,
        )

        inputs = inputs.to(self.device)
        inputs["input_features"] = degraded_features

        use_beam_search = temp == 0.0

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=5 if use_beam_search else 1,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=(temp > 0),
            temperature=temp,
        )

        generated_ids = output_ids.sequences[:, inputs["input_ids"].shape[1] :]
        if return_transcriptions:
            transcription = self.processor.decode(
                generated_ids, return_format="transcription_only"
            )[0]
        else:
            transcription = self.processor.decode(generated_ids, return_format="parsed")[0]

        if use_beam_search and getattr(output_ids, "sequences_scores", None) is not None:
            conf_scores = torch.exp(output_ids.sequences_scores).tolist()
        else:
            conf_scores = (
                [1.0] * len(transcription) if isinstance(transcription, list) else [1.0]
            )
        return transcription, conf_scores


class French_Clear_Speech_Model:
    def __init__(
        self,
        is_fine_tuned: bool = False,
        use_compile: bool = False,
    ) -> None:
        self.model_id = "bofenghuang/whisper-medium-french"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.path_to_model = f"{script_path}/whisper-french-experiment"

        if self.device == "cuda":
            torch.set_float32_matmul_precision("high")

        self.processor, self.feature_extractor, self.model = self._setup(
            is_fine_tuned=is_fine_tuned
        )

        if use_compile and self.device == "cuda":
            self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead")

    def _setup(self, is_fine_tuned: bool = False):
        processor = AutoProcessor.from_pretrained(
            self.model_id, language="french", task="transcribe", token=hf_token
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, token=hf_token
        )

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            use_safetensors=True,
            token=hf_token,
        ).to(self.device)

        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="fr", task="transcribe"
        )
        model.generation_config.use_timestamps = False
        model.generation_config.no_timestamps_token_id = (
            processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        )
        model.generation_config.max_length = None

        if is_fine_tuned:
            peft_model = PeftModel.from_pretrained(model, self.path_to_model)
            peft_model = peft_model.merge_and_unload()
            peft_model.enable_input_require_grads()
            return processor, feature_extractor, peft_model
        else:
            return processor, feature_extractor, model

    def _get_noise_params(self, noise_type: str | None) -> float | int | None:
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

    def _compute_batch_confidence(
        self, output_ids, batch_size: int, num_beams: int
    ) -> list[float]:
        if num_beams > 1 and getattr(output_ids, "sequences_scores", None) is not None:
            return torch.exp(output_ids.sequences_scores).float().cpu().tolist()

        if getattr(output_ids, "scores", None):
            transition_scores = self.model.compute_transition_scores(
                output_ids.sequences,
                output_ids.scores,
                normalize_logits=True,
            )
            pad_id = self.processor.tokenizer.pad_token_id
            eos_id = self.processor.tokenizer.eos_token_id

            conf_scores: list[float] = []
            for row_idx in range(transition_scores.shape[0]):
                row_scores = transition_scores[row_idx]
                # Exclude padded or infinite positions
                valid_mask = torch.isfinite(row_scores)
                if pad_id is not None and output_ids.sequences.shape[1] > 1:
                    gen_tokens = output_ids.sequences[row_idx, 1:]
                    if gen_tokens.shape[0] == row_scores.shape[0]:
                        valid_mask = valid_mask & (gen_tokens != pad_id)
                        if eos_id is not None:
                            valid_mask = valid_mask & (gen_tokens != eos_id)

                if valid_mask.any():
                    mean_log_prob = row_scores[valid_mask].mean()
                    conf_scores.append(float(torch.exp(mean_log_prob).item()))
                else:
                    conf_scores.append(0.0)
            return conf_scores

        return [1.0] * batch_size

    @torch.inference_mode()
    def transcribe(
        self,
        audio_array: torch.Tensor | np.ndarray | list[np.ndarray],
        noise_type: str | None = "studio",
        snr_db: int | float | None = None,
        sampling_rate: int = 16000,
        temp: float = 0.0,
        max_new_tokens: int = 128,
        return_transcriptions: bool = True,
        num_beams: int = 1,
    ):
        is_batched_list = isinstance(audio_array, list)
        if is_batched_list:
            audio_inputs = []
            for arr in audio_array:
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                if isinstance(arr, np.ndarray) and arr.ndim > 1:
                    arr = np.mean(arr, axis=0 if arr.shape[0] < arr.shape[1] else 1)
                audio_inputs.append(np.asarray(arr, dtype=np.float32))
            batch_size = len(audio_inputs)
        else:
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.detach().cpu().numpy()
            if isinstance(audio_array, np.ndarray) and audio_array.ndim > 1:
                audio_array = np.mean(
                    audio_array,
                    axis=0 if audio_array.shape[0] < audio_array.shape[1] else 1,
                )
            audio_inputs = np.asarray(audio_array, dtype=np.float32)
            batch_size = 1

        inputs = self.processor(
            audio_inputs,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding=is_batched_list,
        )

        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)
        attention_mask = (
            inputs.attention_mask.to(device=self.device)
            if "attention_mask" in inputs
            else None
        )

        degraded_features = self._apply_acoustic_degradation(
            input_features=input_features,
            noise_type=noise_type,
            snr_db=snr_db,
            sampling_rate=sampling_rate,
        )

        effective_beams = num_beams if temp == 0.0 else 1
        output_ids = self.model.generate(
            input_features=degraded_features,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=effective_beams,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=(temp > 0),
            temperature=temp,
        )

        clean_transcriptions: list[str] = []
        if return_transcriptions:
            transcription_list = self.processor.batch_decode(
                output_ids.sequences,
                skip_special_tokens=True,
            )
            clean_transcriptions = [
                re.sub(r"<\|.*?\|>|\[.*?\]", "", t).replace("fr", "").strip()
                for t in transcription_list
            ]

        conf_scores = self._compute_batch_confidence(
            output_ids=output_ids,
            batch_size=batch_size,
            num_beams=effective_beams,
        )
        return clean_transcriptions, conf_scores
