#!/usr/bin/env python3
"""Compute SARM RA-BC weights for the card memory dataset.

Thin wrapper around lerobot's built-in compute_rabc_weights with
project-specific defaults.  Outputs a parquet file that train_contextvla.py
loads via RABCWeights to weight each training sample.

The underlying weight formula (from the SARM paper, Eq. 8-9) uses a
*progress delta*:
    delta = progress[t + chunk_size] - progress[t]
Samples that precede real forward progress get weight 1, samples that
precede no progress (or backward motion) get weight 0.

Usage (RunPod, after SARM training completes):

    python scripts/compute_sarm_rewards.py \\
        --checkpoint_path outputs/train/sarm_card_memory/checkpoints/last/pretrained_model \\
        --output_dir     outputs/rewards/sarm_card_memory

    # Faster: compute every 5 frames, interpolate the rest (~5x speedup)
    python scripts/compute_sarm_rewards.py \\
        --checkpoint_path outputs/train/sarm_card_memory/checkpoints/last/pretrained_model \\
        --output_dir     outputs/rewards/sarm_card_memory \\
        --stride 5

Then train ContextVLA with RA-BC:

    python scripts/train_contextvla.py \\
        --dataset_repo_id "omnaathg/so101_card_memory" \\
        --output_repo_id  "omnaathg/contextvla_card_memory_v3" \\
        --reward_path     "outputs/rewards/sarm_card_memory/rewards.parquet" \\
        --steps 30000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute SARM RA-BC weights for ContextVLA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint_path",
        required=True,
        help=(
            "Path to the pretrained_model dir inside the SARM checkpoint, e.g. "
            "outputs/train/sarm_card_memory/checkpoints/last/pretrained_model"
        ),
    )
    p.add_argument(
        "--dataset_repo_id",
        default="omnaathg/so101_card_memory",
        help="HuggingFace dataset repo ID used for SARM training",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/rewards/sarm_card_memory",
        help="Directory to write rewards.parquet",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Device for CLIP encoding and SARM inference",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Compute rewards every N frames and interpolate the rest "
            "(default 1 = every frame; stride=5 gives ~5x speedup with minor accuracy loss)"
        ),
    )
    p.add_argument(
        "--num_visualizations",
        type=int,
        default=3,
        help="Number of episodes to visualise after computation (0 to skip)",
    )
    p.add_argument(
        "--viz_dir",
        default="outputs/rewards/sarm_card_memory/viz",
        help="Directory to write episode visualisation plots",
    )
    p.add_argument(
        "--no_push_to_hub",
        action="store_true",
        help="Skip uploading rewards.parquet to the HuggingFace dataset repo",
    )
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rewards.parquet"

    logger.info(f"SARM checkpoint : {args.checkpoint_path}")
    logger.info(f"Dataset         : {args.dataset_repo_id}")
    logger.info(f"Output          : {output_path}")
    logger.info(f"Stride          : {args.stride}")

    from lerobot.rewards.sarm.compute_rabc_weights import compute_sarm_progress

    compute_sarm_progress(
        dataset_repo_id=args.dataset_repo_id,
        reward_model_path=args.checkpoint_path,
        output_path=str(output_path),
        head_mode="sparse",          # single_stage mode only has a sparse head
        device=args.device,
        num_visualizations=args.num_visualizations,
        output_dir=args.viz_dir,
        stride=args.stride,
    )

    logger.info(f"Rewards saved to {output_path}")

    if not args.no_push_to_hub:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            logger.info(f"Uploading rewards.parquet to {args.dataset_repo_id} ...")
            api.upload_file(
                path_or_fileobj=str(output_path),
                path_in_repo="rewards.parquet",
                repo_id=args.dataset_repo_id,
                repo_type="dataset",
            )
            logger.info("Upload complete.")
        except Exception as e:
            logger.warning(f"Hub upload failed (skipping): {e}")

    logger.info("\nNext step — train ContextVLA with RA-BC:")
    logger.info(
        f"  python scripts/train_contextvla.py "
        f'--reward_path "{output_path}" '
        f"--dataset_repo_id {args.dataset_repo_id} "
        f"--output_repo_id omnaathg/contextvla_card_memory_v3 "
        f"--steps 30000"
    )


if __name__ == "__main__":
    main()
