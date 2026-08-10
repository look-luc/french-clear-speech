import argparse
import gc

import torch

from python.main import run_model

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Run transcription model pipeline")
    parser.add_argument("-o", "--override", type=str, help="Model operation mode (train, transcribe, graph)")
    args = parser.parse_args()

    print(f"running {args.override}\n\n")
    run_model(args.override)
