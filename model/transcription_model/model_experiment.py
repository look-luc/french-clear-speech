from pathlib import Path

import numpy as np
import scipy.signal as signal
import torch
from peft import PeftModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

python_dir = Path(__file__).resolve().parents[1]
root_dir = Path(__file__).resolve().parents[2]

script_path = root_dir = Path(__file__).resolve().parent
class French_Clear_Speech_Model:
    def __init__(
        self,
        model_id: str = "bofenghuang/whisper-medium-french",
        path_to_model:str=f"{script_path}/whisper-french-experiment"
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path_to_model = path_to_model
        self.processor, self.feature_extractor, self.model = self._setup()

    def _setup(self, is_fine_tuned:bool=False):
        processor = AutoProcessor.from_pretrained(
            self.model_id, language="french", task="transcribe"
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, use_safetensors=True
        ).to(self.device)

        model.generation_config.forced_decoder_ids = (
            processor.get_decoder_prompt_ids(language="french", task="transcribe")
        )
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
        sample_rate: int,
        cutoff_freq: int|None,
        snr_db: int|None,
    ):
        degraded_audio = audio_array.clone().numpy()

        if cutoff_freq is not None:
            nyquist = 0.5 * sample_rate
            normal_cutoff = cutoff_freq / nyquist

            # output="sos" returns a single matrix array, resolving tuple unpacking type stubs
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
        sampling_rate: int = 16000,
        cutoff_freq: int | None = 1500,
        snr_db: int | None = 10,
        temp: float = 0.5,
    ):
        processed_audio = self._apply_acoustic_degradation(
            audio_array, sampling_rate, cutoff_freq, snr_db
        )

        input_features = self.processor(
            processed_audio, sampling_rate=sampling_rate
        ).input_features

        input_features = input_features
        # input_features = input_features.to(self.device)

        output_ids = self.model.generate(
            input_features,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=True,
            temperature=temp,
        )

        transcription = self.processor.batch_decode(
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

        transcription_text = transcription[0] if transcription else ""
        confidence_val = conf_score.item()

        return transcription_text, confidence_val
