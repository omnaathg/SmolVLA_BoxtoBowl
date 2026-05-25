"""ContextVLA: temporal multi-frame extension of SmolVLA for non-Markovian tasks.

Key changes vs SmolVLA:
  1. ContextVLAFlowMatching.embed_prefix  — encodes T frames per camera instead of 1.
  2. ContextVLAPolicy.prepare_images      — keeps (B, T, C, H, W) instead of taking [:, -1].
  3. ContextVLAPolicy.reset / select_action — maintains a rolling temporal frame buffer
                                             at inference time.
"""

import math
from collections import deque

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    pad_tensor,
    resize_with_pad,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.utils import populate_queues

from .configuration_contextvla import ContextVLAConfig


class ContextVLAFlowMatching(VLAFlowMatching):
    """VLAFlowMatching with multi-frame prefix embedding.

    The only change is embed_prefix: instead of encoding one frame per camera,
    we encode T frames per camera and concatenate their patch embeddings in time order.
    """

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build prefix embeddings from T frames per camera.

        Args:
            images:     list of (B, T, C, H, W) tensors, one per camera.
                        Falls back to base behaviour when ndim==4 (T=1 already squeezed).
            img_masks:  list of (B,) bool tensors, one per camera (True = real frame).
            lang_tokens: (B, L) long.
            lang_masks:  (B, L) bool.
            state:       (B, state_dim) float.

        Returns:
            embs:       (B, seq_len, D) — concatenated prefix embeddings.
            pad_masks:  (B, seq_len) bool — True for real tokens.
            att_masks:  (B, seq_len) bool — 0 = bidirectional, 1 = causal barrier.
        """
        # Graceful fallback: if images are already single-frame (B, C, H, W), use base.
        if images[0].ndim == 4:
            return super().embed_prefix(images, img_masks, lang_tokens, lang_masks, state)

        embs = []
        pad_masks_list = []
        att_mask_ints = []

        bsize = images[0].shape[0]
        device = images[0].device

        for img, img_mask in zip(images, img_masks):
            # img: (B, T, C, H, W)
            B, T, C, H, W = img.shape

            # Encode all T frames through the frozen SigLIP vision encoder.
            flat = img.reshape(B * T, C, H, W)                    # (B*T, C, H, W)
            img_emb = self.vlm_with_expert.embed_image(flat)       # (B*T, patches, D)
            patches = img_emb.shape[1]
            D = img_emb.shape[2]

            img_emb = img_emb.reshape(B, T * patches, D)           # (B, T*patches, D)
            img_emb = img_emb * math.sqrt(D)                       # same scaling as base

            # img_mask is (B,) — expand to cover all T*patches positions.
            expanded_mask = img_mask[:, None].expand(B, T * patches)   # (B, T*patches)

            embs.append(img_emb)
            pad_masks_list.append(expanded_mask)
            att_mask_ints += [0] * (T * patches)  # images attend bidirectionally

        # Language tokens — same as base.
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks_list.append(lang_masks)
        att_mask_ints += [0] * lang_emb.shape[1]  # language attends bidirectionally

        # State token — same as base; causal barrier prevents images/lang attending to state.
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        embs.append(state_emb)
        pad_masks_list.append(state_mask)
        att_mask_ints += [1] * states_seq_len  # state = causal barrier

        embs = torch.cat(embs, dim=1)                               # (B, seq_len, D)
        pad_masks = torch.cat(pad_masks_list, dim=1)               # (B, seq_len)
        att_masks = torch.tensor(att_mask_ints, dtype=torch.bool, device=device)
        att_masks = att_masks[None, :].expand(bsize, -1)           # (B, seq_len)

        # Pad to prefix_length if a fixed length is requested (default prefix_length=-1, skip).
        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        return embs, pad_masks, att_masks


class ContextVLAPolicy(SmolVLAPolicy):
    """SmolVLAPolicy with a T-frame temporal context window.

    Training:  batch["observation.images.*"] has shape (B, T, C, H, W) from the dataset
               (delta_timestamps provides T frames per camera per sample).

    Inference: a rolling deque buffers the last n_obs_steps single frames from the robot
               and stacks them into (B=1, T, C, H, W) before each forward pass.
    """

    config_class = ContextVLAConfig
    name = "contextvla"

    def __init__(self, config: ContextVLAConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Replace base VLAFlowMatching with our multi-frame version.
        # (super().__init__ creates a VLAFlowMatching first; we discard it here.)
        self.model = ContextVLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self._obs_queues: dict[str, deque] = {}
        self.reset()

    # ── Observation preparation ──────────────────────────────────────────

    def prepare_images(self, batch: dict) -> tuple[list[Tensor], list[Tensor]]:
        """Keep full (B, T, C, H, W) temporal dimension instead of taking [:, -1].

        For missing cameras the base fallback (empty -1 tensor) is used.
        """
        images = []
        img_masks = []

        present_keys = [k for k in self.config.image_features if k in batch]
        missing_keys = [k for k in self.config.image_features if k not in batch]

        if not present_keys:
            raise ValueError(
                f"All image features missing from batch. "
                f"batch keys={list(batch.keys())}, "
                f"image_features={list(self.config.image_features)}"
            )

        last_img = None
        last_mask = None

        for key in present_keys:
            img = batch[key]  # (B, T, C, H, W) — do NOT collapse T

            if self.config.resize_imgs_with_padding is not None:
                if img.ndim == 5:
                    B, T = img.shape[:2]
                    flat = img.reshape(B * T, *img.shape[2:])
                    flat = resize_with_pad(flat, *self.config.resize_imgs_with_padding, pad_value=0)
                    img = flat.reshape(B, T, *flat.shape[1:])
                else:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0  # [0, 1] → [-1, 1] for SigLIP

            bsize = img.shape[0]
            device = img.device

            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)  # (B,)

            images.append(img)
            img_masks.append(mask)
            last_img, last_mask = img, mask

        # Handle explicitly declared empty cameras (same as base).
        for i in range(len(missing_keys)):
            if i >= self.config.empty_cameras:
                break
            if last_img is not None:
                # Use the last real frame's last timestep as a -1-filled placeholder.
                ref = last_img[:, -1] if last_img.ndim == 5 else last_img
                images.append(torch.ones_like(ref) * -1)
                img_masks.append(torch.zeros_like(last_mask))

        return images, img_masks

    # ── Temporal buffering for inference ────────────────────────────────

    def reset(self):
        """Reset action queue and per-camera temporal frame buffers."""
        super().reset()  # sets self._queues = {ACTION: deque(maxlen=n_action_steps)}
        self._obs_queues = {
            key: deque(maxlen=self.config.n_obs_steps)
            for key in self.config.image_features
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise=None, **kwargs) -> Tensor:
        """Select one action, maintaining a T-frame rolling observation buffer.

        Single frames arrive from the robot each step; we accumulate them into a
        (B=1, T, C, H, W) tensor before calling the model.  The first T-1 steps are
        padded by repeating the earliest available frame (consistent with training
        behaviour where the dataset pads episode-start frames).
        """
        self.eval()
        batch = self._prepare_batch(batch)

        # Append current single-frame images to each camera's rolling buffer.
        for key in self.config.image_features:
            if key not in batch:
                continue
            frame = batch[key]
            if frame.ndim == 5:
                frame = frame[:, -1]  # collapse any leftover T dim → (B, C, H, W)
            self._obs_queues[key].append(frame)

            # Build (B, T, C, H, W), padding with the oldest frame if buffer not full yet.
            frames = list(self._obs_queues[key])
            while len(frames) < self.config.n_obs_steps:
                frames.insert(0, frames[0])
            batch[key] = torch.stack(frames, dim=1)  # (B, T, C, H, W)

        # Populate action queue (note: image keys are NOT in self._queues, so they
        # aren't touched by populate_queues — only non-image observations would be).
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise, **kwargs)
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, :, :original_action_dim]
            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()
