import csv
from pathlib import Path

import soundfile as sf


def create_metadata_csv(audio_dir: str | Path, output_csv_path: str | Path = None):
    audio_dir = Path(audio_dir).resolve()
    if output_csv_path is None:
        output_csv_path = audio_dir / "metadata.csv"
    else:
        output_csv_path = Path(output_csv_path).resolve()

    valid_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    rows = []

    for file_path in sorted(audio_dir.rglob("*")):
        if file_path.suffix.lower() in valid_extensions and file_path.is_file():
            # Hugging Face expects relative paths from the metadata.csv location
            relative_path = file_path.relative_to(audio_dir).as_posix()

            try:
                info = sf.info(file_path)
                duration = round(info.duration, 2)
            except Exception:
                duration = None

            rows.append({
                "file_name": relative_path,
                "duration": duration,
                "text": ""  # Optional placeholder if text transcripts are added later
            })

    fieldnames = ["file_name", "duration", "text"]
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {output_csv_path} with {len(rows)} entries.")

if __name__ == "__main__":
    # Replace with the local path to your audio dataset folder
    create_metadata_csv("/Users/lucdenardi/Desktop/data/lecture")
