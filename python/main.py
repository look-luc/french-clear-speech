import os
from pathlib import Path

from dotenv import load_dotenv

from .graph_metric.graph import metrics_graph
from .transcription_model.base_model import French_Speech_text_base
from .transcription_model.model import French_Speech_text

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def run_model(what_model:str):
    if what_model == "base":
        french_speech_transcription = French_Speech_text_base()
        output = ""
        try:
            output = french_speech_transcription.predict()
            save_status = f"Successfully trained model setup: {french_speech_transcription.model_id}"
        except Exception as e:
            save_status = f"ERROR with model: {str(e)}"
        output_dir += Path("./model/french_speech_transcription_base_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
            file.write(f"{save_status}\n{output}")
    elif what_model == "train":
        french_speech_transcription = French_Speech_text()
        try:
            french_speech_transcription.train()
            save_status = f"Successfully trained model setup: {french_speech_transcription.model_id}"
        except Exception as e:
            save_status = f"ERROR with model: {str(e)}"

        output_dir = Path("./model/french_speech_transcription_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
            file.write(save_status)
    elif what_model == "transcribe":
        pass
    elif what_model == "graph":
        metrics_graph()
    else:
        raise ValueError(f"{what_model.capitalize()} is not one of the options. Only pick one of these: train, transcribe, or graph.")
