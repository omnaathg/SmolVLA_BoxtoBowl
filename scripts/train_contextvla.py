#!/usr/bin/env python3
"""Train ContextVLAPolicy end-to-end on a language-conditioned card memory dataset.

Usage (RunPod A100):
    python scripts/train_contextvla.py \
        --dataset_repo_id omnaathg/so101_card_memory \
        --output_repo_id  omnaathg/contextvla_card_memory \
        --output_dir      outputs/contextvla_card_memory \
        --steps 30000 \
        --batch_size 32 \
        --device cuda

Optional — warm-start action expert from pretrained SmolVLA base:
    --base_model lerobot/smolvla_base

VRAM notes (A100 80GB, freeze_vision_encoder=True, n_obs_steps=11):
  batch=16, bfloat16 → ~40-50 GB  (fits A100 80GB)
  batch=8            → ~20-25 GB  (fits A100 40GB)
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Import lerobot constants and dataset ────────────────────────────────────
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.types import FeatureType, PolicyFeature

# ── Import our policy ───────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from contextvla import ContextVLAConfig, ContextVLAPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo_id",  required=True)
    p.add_argument("--output_repo_id",   required=True,
                   help="HuggingFace repo to push trained policy to")
    p.add_argument("--output_dir",       default="outputs/contextvla")
    p.add_argument("--base_model",       default=None,
                   help="Optional: load action-expert weights from this SmolVLA checkpoint")
    p.add_argument("--steps",            type=int,   default=30_000)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--warmup_steps",     type=int,   default=1_000)
    p.add_argument("--weight_decay",     type=float, default=1e-10)
    p.add_argument("--grad_clip",        type=float, default=10.0)
    p.add_argument("--n_obs_steps",      type=int,   default=11)
    p.add_argument("--temporal_stride",  type=int,   default=30)
    p.add_argument("--fps",              type=int,   default=30)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--save_every",       type=int,   default=5_000)
    p.add_argument("--log_every",        type=int,   default=100)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--num_workers",      type=int,   default=4)
    return p.parse_args()


def build_config(args) -> ContextVLAConfig:
    return ContextVLAConfig(
        n_obs_steps=args.n_obs_steps,
        temporal_stride=args.temporal_stride,
        use_compression=False,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        load_vlm_weights=True,         # loads SmolVLM2-500M backbone from HuggingFace
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.tripod_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 512, 512)
            ),
            "observation.images.gripper_cam": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 512, 512)
            ),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))
        },
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        optimizer_grad_clip_norm=args.grad_clip,
        scheduler_warmup_steps=args.warmup_steps,
        scheduler_decay_steps=args.steps,
    )


def build_delta_timestamps(config: ContextVLAConfig, fps: int) -> dict:
    obs_offsets = config.observation_delta_indices  # e.g. [-210, -180, ..., 0]
    img_ts = [i / fps for i in obs_offsets]         # [-7.0, -6.0, ..., 0.0]
    action_ts = [i / fps for i in range(config.chunk_size)]
    return {
        "observation.images.tripod_cam":  img_ts,
        "observation.images.gripper_cam": img_ts,
        OBS_STATE:                        [0.0],
        ACTION:                           action_ts,
    }


def pretokenize_tasks(dataset: LeRobotDataset, processor, max_length: int, device) -> dict:
    """Pre-tokenize every unique task string in the dataset."""
    task_tokens = {}
    for idx, task_str in dataset.meta.tasks.items():
        text = task_str if task_str.endswith("\n") else task_str + "\n"
        enc = processor.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        task_tokens[int(idx)] = {
            "input_ids":      enc["input_ids"].squeeze(0).to(device),
            "attention_mask": enc["attention_mask"].squeeze(0).bool().to(device),
        }
    logger.info(f"Pre-tokenized {len(task_tokens)} unique tasks")
    return task_tokens


def add_language_tokens(batch: dict, task_tokens: dict, device) -> dict:
    """Insert OBS_LANGUAGE_TOKENS / OBS_LANGUAGE_ATTENTION_MASK into batch."""
    task_indices = batch.get("task_index")
    if task_indices is None:
        raise KeyError(
            "'task_index' not found in batch — ensure the dataset has per-episode tasks."
        )
    ids    = torch.stack([task_tokens[int(i)]["input_ids"]      for i in task_indices])
    masks  = torch.stack([task_tokens[int(i)]["attention_mask"] for i in task_indices])
    batch[OBS_LANGUAGE_TOKENS]          = ids.to(device)
    batch[OBS_LANGUAGE_ATTENTION_MASK]  = masks.to(device)
    return batch


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # ── Build config & delta_timestamps ─────────────────────────────────
    config = build_config(args)
    delta_ts = build_delta_timestamps(config, args.fps)
    logger.info(f"Temporal offsets: {config.observation_delta_indices}")

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_ts)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    data_iter = iter(dataloader)
    logger.info(f"Dataset: {len(dataset)} samples")

    # ── Policy ───────────────────────────────────────────────────────────
    logger.info("Building ContextVLAPolicy...")
    policy = ContextVLAPolicy(config)
    policy.to(device)

    if args.base_model:
        logger.info(f"Loading action-expert weights from {args.base_model}...")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as BasePolicy
        base = BasePolicy.from_pretrained(args.base_model)
        missing, unexpected = policy.model.load_state_dict(
            base.model.state_dict(), strict=False
        )
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        del base
        logger.info("Base weights loaded.")

    total_params  = sum(p.numel() for p in policy.parameters())
    train_params  = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Parameters: {total_params:,} total, {train_params:,} trainable")

    # ── Tokenizer for language conditioning ─────────────────────────────
    processor = AutoProcessor.from_pretrained(config.vlm_model_name)
    task_tokens = pretokenize_tasks(dataset, processor, config.tokenizer_max_length, device)

    # ── Optimizer + LR schedule ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(config.optimizer_betas[0], config.optimizer_betas[1]),
        eps=config.optimizer_eps,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        t = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        # Cosine decay to scheduler_decay_lr / lr
        min_lr_ratio = config.scheduler_decay_lr / args.lr
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
            1.0 + torch.cos(torch.tensor(t * 3.14159)).item()
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ────────────────────────────────────────────────────
    logger.info(f"Training for {args.steps} steps, batch={args.batch_size}...")
    policy.train()
    loss_history = []
    best_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        batch = add_language_tokens(batch, task_tokens, device)

        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            logger.info(
                f"Step {step}/{args.steps} | loss={loss_val:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{step/elapsed:.1f} it/s"
            )

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = output_dir / f"checkpoint_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                },
                ckpt_path,
            )
            logger.info(f"Saved: {ckpt_path}")
            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(policy.state_dict(), output_dir / "best_model.pt")
                logger.info(f"New best loss: {best_loss:.4f}")

    # ── Push to HuggingFace Hub ──────────────────────────────────────────
    logger.info(f"Pushing policy to {args.output_repo_id}...")
    policy.push_to_hub(args.output_repo_id)
    logger.info("Done.")

    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f)

    total_time = time.time() - t0
    logger.info(
        f"Training complete in {total_time/60:.1f} min. Best loss: {best_loss:.4f}"
    )


if __name__ == "__main__":
    main()
