# RLT Stage 2: Integration Learnings

## What Worked

Stage 2 is now running: the robot executes the VLA policy smoothly during warmup episodes, intervention detection works via the leader arm, and the RL training loop collects transitions into the replay buffer.

## Critical Issues Encountered & Fixes

### 1. State Vector Order (THE BIGGEST BUG)

**Problem:** The observation state was built using `sorted()` on joint names, producing alphabetical order:
```
elbow_flex, gripper, shoulder_lift, shoulder_pan, wrist_flex, wrist_roll
```
But the model was trained with motor definition order:
```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```
The normalizer applies per-element mean/std, so wrong order = every joint gets the wrong normalization = garbage model input = erratic robot motion.

**Fix:** Build state vector explicitly using `JOINT_NAMES` order:
```python
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
state_names = [f"{j}.pos" for j in JOINT_NAMES]
observation_frame["observation.state"] = np.array(
    [obs.get(name, 0.0) for name in state_names], dtype=np.float32
)
```

### 2. Action Chunking — Must Use Action Queue

**Problem:** Calling `vla_model.sample_actions()` every step (30Hz) and only using `action[0]` each time. This generates a brand new 50-step trajectory every 33ms and always jumps to the start — causing jerky motion.

**Fix:** Use `predict_action()` from `lerobot.utils.control_utils` — the exact same function `lerobot-record` uses. It calls `policy.select_action()` which manages an internal action queue (deque of 50 actions). The model is only queried when the queue is empty.

### 3. Do NOT Call `.float()` on SmolVLA

**Problem:** Calling `smolvla_policy.to(device).float()` converts the entire model to float32. `lerobot-record` does NOT do this. It changes the model's numerical behavior and produces different (wrong) outputs.

**Fix:** Just `smolvla_policy.to(device)` — no `.float()`.

### 4. SmolVLA Features Not Populated by `from_pretrained`

**Problem:** `SmolVLAPolicy.from_pretrained()` does not populate `input_features`, `output_features`, or `image_features` from the HF config. Methods like `prepare_images()` fail with "All image features are missing."

**Fix:** Manually set features before loading:
```python
from lerobot.configs.types import FeatureType, PolicyFeature
smolvla_config.input_features = {
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
}
smolvla_config.output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
}
```

### 5. Preprocessor/Postprocessor Must Be Loaded from Pretrained Checkpoint

**Problem:** The postprocessor contains normalization stats (mean/std from the training dataset) needed to unnormalize actions back to robot-scale degrees. Without it, raw model outputs are in normalized space and meaningless to the robot.

**Fix:** Load from the HF repo:
```python
from lerobot.processor import PolicyProcessorPipeline
smolvla_preprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=smolvla_path,
    config_filename="policy_preprocessor.json",
    overrides={"device_processor": {"device": device}, "rename_observations_processor": {"rename_map": rename_map}},
    ...
)
smolvla_postprocessor = PolicyProcessorPipeline.from_pretrained(
    pretrained_model_name_or_path=smolvla_path,
    config_filename="policy_postprocessor.json",
    ...
)
```

### 6. Camera Rename Map Required

**Problem:** Robot cameras are named `webcam` and `arm_cam`, but the trained policy expects `camera1` and `camera2`.

**Fix:**
```python
rename_map = {
    "observation.images.webcam": "observation.images.camera1",
    "observation.images.arm_cam": "observation.images.camera2",
}
```

### 7. Camera Resolution — Use Native, Not Model Size

**Problem:** Setting camera config to 224x224 or 512x512 fails because the webcam doesn't support those resolutions natively.

**Fix:** Capture at 640x480 (native). The policy's `prepare_images()` handles resizing to 512x512 with padding internally.

### 8. Episode Length — 50 Steps Is Not Enough

**Problem:** Default `steps_per_episode=50` gives ~1.7 seconds at 30Hz. Pick-and-place needs 30-50 seconds.

**Fix:** Set `--steps_per_episode 1500` (50 seconds at 30Hz).

### 9. `torch.load` Needs `NormalizationMode` Allowlisted

**Problem:** Stage 1 checkpoint contains `NormalizationMode` enum which `weights_only=True` blocks.

**Fix:**
```python
from lerobot.configs.types import NormalizationMode
torch.serialization.add_safe_globals([NormalizationMode])
```

### 10. LeRobot Module Paths

The correct import paths (not `so101_follower`):
```python
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
```

Camera config uses `index_or_path` (not `camera_index`). Robot/leader IDs must match existing calibration files (`my_follower`, `my_leader`).

## Golden Rule

**To match lerobot-record inference exactly, use `predict_action()` from `lerobot.utils.control_utils`.** Do not try to manually replicate the pipeline — there are too many subtle details (state order, normalization stats, action chunking, image preprocessing) that must match exactly.

## Key Files

| File | Purpose |
|------|---------|
| `run_rlt_stage2.sh` | Launch script with all config flags |
| `scripts/train_rlt_stage2.py` | Stage 2 training code |
| `scripts/run_rlt_inference.py` | Deployment inference (needs same fixes) |
| `run_inference.sh` | Working lerobot-record inference (reference) |
| `checkpoints/rlt_stage1/best_checkpoint.pt` | Frozen RLT encoder |
| `checkpoints/rlt_stage2/` | Stage 2 actor/critic checkpoints |
