import os
import re
from pathlib import Path

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
        noise_type:str,
        cutoff_freq: int | None,
        snr_db: int | None,
        sample_rate: int=16000,
    ):
        degraded_audio = audio_array.detach().cpu().numpy()

        def _simulate_noise_env(noise_type:str):
            snr_db, cutoff_freq = 0, 0
            if noise_type.lower() == "studio":
                snr_db = 30
                cutoff_freq = None
            elif noise_type.lower() == "mild_office":
                snr_db = 15
                cutoff_freq = 3400
            elif noise_type.lower() == "moderate_cafe":
                snr_db = 10
                cutoff_freq = 1500
            elif noise_type.lower() == "severe_street":
                snr_db = 0
                cutoff_freq = 800
            elif noise_type.lower() == "extreme_cocktail":
                snr_db = -5
                cutoff_freq = 500
            return snr_db, cutoff_freq

        if cutoff_freq is None and snr_db is None:
            snr_db, cutoff_freq = _simulate_noise_env(noise_type)
        if cutoff_freq is not None:
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_freq / nyquist

            sos = signal.butter(
                N=5, Wn=normal_cutoff, btype="low", analog=False, output="sos"
            )
            degraded_audio = signal.sosfilt(sos, degraded_audio)

        if snr_db is not None:
            signal_power = np.mean(np.square(degraded_audio))

            if signal_power > 0:
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.random.normal(
                    loc=0.0,
                    scale=np.sqrt(noise_power),
                    size=len(degraded_audio),
                )
                degraded_audio = degraded_audio + noise

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

        input_features = self.processor(
            processed_audio, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(self.device)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="french", task="transcribe"
        )

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

        confidence = self.model.compute_transition_scores(
            output_ids.sequences,
            output_ids.scores,
            normalize_logits=True,
        )
        avg_log_prob = torch.mean(confidence)
        conf_score = torch.exp(avg_log_prob)

        transcription_text = transcription_list[0] if transcription_list else ""
        clean_transcription = re.sub(r"<\|.*?\|>|\[.*?\]", "", transcription_text).replace("fr","").strip()

        confidence_val = conf_score.item()

        return clean_transcription, confidence_val
