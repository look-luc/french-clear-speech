import argparse
import gc

import torch

from python.main import run_model

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is NOT available. PyTorch will run on CPU!")

    parser = argparse.ArgumentParser(description="Run transcription model pipeline")
    parser.add_argument("-o", "--override", type=str, help="Model operation mode (train, transcribe, graph)")
    args = parser.parse_args()

    print(f"running {args.override}\n\n")
    run_model(args.override)
