import torch


class French_Speech_text:
    def __init__(
        self,
        model_id:str="bofenghuang/whisper-medium-french",
        device:str="cuda:0" if torch.cuda.is_available() else "mps:1" if  torch.backends.mps.is_available() else "cpu:3"
    ) -> None:
        self.device = torch.device(device)
        self.model_id = model_id
