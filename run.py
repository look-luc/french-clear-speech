import argparse
import gc

import torch

from model.main import run_model


def parse_multiple_types(value):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Run transcription model pipeline")
    parser.add_argument("--model_type", type=str, default="train")
    parser.add_argument("--noise_type", type=str, default="studio")
    parser.add_argument("--cutoff_freq", type=parse_multiple_types, default=None)
    parser.add_argument("--snr_db", type=parse_multiple_types, default=None)
    args = parser.parse_args()

    print(f"running {args.model_type}, with {args.noise_type}, {args.cutoff_freq}, and {args.snr_db}\n\n")
    run_model(args.model_type, args.noise_type, args.cutoff_freq, args.snr_db)
