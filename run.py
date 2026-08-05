import argparse
import gc

import torch

from python.main import run_model

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="all arguments to see what type of model will run")

    parser.add_argument("-o", "--override", type=str, help="Insert what model type want to use (train, transcribe, or graph)")

    args = parser.parse_args()

    print(f"running {args.override}\n\n")
    run_model(args.override)
