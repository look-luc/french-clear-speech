import os

from dotenv import load_dotenv
from graph_metric.graph import metrics_graph
from transcription_model.model import French_Speech_text

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def run_model(what_model:str):
    french_speech_transcription = French_Speech_text()
    if what_model == "train":
        french_sppech_transcription.train()
