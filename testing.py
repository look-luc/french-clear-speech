import soundfile as sf
from datasets import load_dataset
from huggingface_hub import snapshot_download

ds = load_dataset("lookitsluc1/french_cleer_speech", split="train")

if "duration" in ds.column_names:
    total_seconds = sum(ds["duration"])
else:
    repo_dir = snapshot_download("lookitsluc1/french_cleer_speech", repo_type="dataset")

    total_seconds = sum(
        sf.info(f"{repo_dir}/{x['file_name']}").duration
        for x in ds
    )

print(f"Total Train Duration: {total_seconds / 3600:.2f} hours ({total_seconds:.2f} seconds)")
