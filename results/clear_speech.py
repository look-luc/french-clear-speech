import torch
from peft import PeftModel
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

base_model_id = "bofenghuang/whisper-medium-french"
adapter_path = "./whisper-french-experiment"
output_dir = "./whisper_merged"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
processor = AutoProcessor.from_pretrained("bofenghuang/whisper-medium-french", language="french", task="transcribe")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

merged_model = peft_model.merge_and_unload()

merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
processor.save_pretrained(output_dir)

merged_model.push_to_hub("lookitsluc1/clear-whisper-medium-french")
tokenizer.push_to_hub("lookitsluc1/clear-whisper-medium-french")
processor.push_to_hub("lookitsluc1/clear-whisper-medium-french")

feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_id)

feature_extractor.sampling_rate = 16000
feature_extractor.return_attention_mask = True

feature_extractor.save_pretrained(output_dir)
feature_extractor.push_to_hub("lookitsluc1/clear-whisper-medium-french")
