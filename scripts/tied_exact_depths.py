#!/usr/bin/env python3
"""Weight-tied ConvNeXt at the exact refinement depths 9R of Table 1 (full validation).

Table 1 used the nearest depths already in ``outputs/sharedConvnextAblation`` (32, 96, 128,
1024) for R = 4, 10, 32, 128. This evaluates D = 36, 90, 288, 1152 with Euler step 9/D
(T = 9) using the same checkpoint, transforms and evaluation loop, and writes to a new
directory so the original sweep is untouched.

    python scripts/tied_exact_depths.py --batch-size 128 --num-workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from shared_convnext_ablation import (  # noqa: E402
    CHECKPOINT, TRAIN_HORIZON, AblationRunner, append_result, cfg, config_name,
    load_completed_names, load_shared_convnext, row_to_record,
)

OUTPUT_DIR = ROOT / "outputs" / "tied_exact_depths"
DEPTHS = (36, 90, 288, 1152)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "results.csv"
    done = load_completed_names(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_shared_convnext(CHECKPOINT).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
    model.eval()
    n = model.stage3_length
    runner = AblationRunner(model, device=device, batch_size=args.batch_size, max_batches=None,
                            amp=False, n_blocks=n, num_workers=args.num_workers)
    for depth in DEPTHS:
        es = TRAIN_HORIZON / depth
        name = config_name(depth, es, "RK1")
        if name in done:
            print(f"{name} done")
            continue
        print(name, flush=True)
        row = runner.evaluate_configuration(cfg([n, n + 1], [0] * depth, euler_step=es))
        append_result(csv_path, row_to_record(name, row))


if __name__ == "__main__":
    main()
