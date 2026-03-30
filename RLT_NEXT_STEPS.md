# RLT: Current Status (2026-03-25)

## What's Done
- **Stage 1**: RLT encoder trained (best loss 0.707). Checkpoint: `checkpoints/rlt_stage1/best_checkpoint.pt`
- **Stage 2**: 20 episodes collected (5 warmup + 15 training), 630 RL updates. Checkpoints in `checkpoints/rlt_stage2/`
- **Inference script**: Working with spacebar VLA↔Actor switching
- **Full pipeline verified**: VLA inference matches lerobot-record exactly, z_rl computation works on CPU, actor outputs reach robot

## What's NOT Working Yet
- **Actor is untrained** — 630 updates with exploding critic (loss ~4 trillion). Outputs garbage when switched to.
- **Need 100+ episodes** for meaningful actor learning (paper trains for hours)

## Before Next Training Session

### Fix critic instability (do this first)
In `scripts/train_rlt_stage2.py`, add gradient clipping to critic and actor updates. Find the RL update section and add after each `.backward()`:
```python
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
```
Also consider lowering learning rates:
```
--actor_lr 1e-4 --critic_lr 1e-4
```
(currently 3e-4 each)

### Run more episodes
```bash
./run_rlt_stage2.sh
```
Target: 50-100 total training episodes. Watch for:
- Critic loss should decrease over episodes (currently exploding)
- Q mean should stabilize to reasonable values (0-10 range)
- More intervention episodes = better actor learning

### Test actor
```bash
./run_rlt_inference.sh
```
- Starts in VLA mode (robot moves normally)
- Press SPACE to switch to RL actor
- Press SPACE again to go back to VLA
- Press Ctrl+C to stop

## Architecture Summary (matches RLT paper)

### Training (Stage 2)
```
Each episode:
  1. Robot runs VLA (via predict_action, same as lerobot-record)
  2. Human grabs leader arm to intervene when needed (5s release timeout)
  3. Right arrow = end episode, Escape = stop training
  4. After episode: batch compute z_rl on CPU (~1s per snapshot, every 50 steps)
  5. Build transitions → replay buffer
  6. Run actor-critic RL updates
```

### Inference (paper's strategy)
```
VLA handles easy parts (approach, positioning)
  ↓ press SPACE
RL Actor takes full control for critical phase
  - Computes z_rl from frozen VLA embeddings (CPU)
  - Actor outputs action chunk conditioned on (z_rl, state, ref_actions)
  - Actions sent to robot sequentially from chunk
  ↓ press SPACE
Back to VLA
```

### Key difference from blending approach
The paper does NOT blend VLA + actor. The actor **replaces** the VLA during the critical phase. The actor is conditioned on VLA reference actions as input (for BC regularization), but its output is the sole action sent to the robot.

## Commands
```bash
# Stage 2 training
./run_rlt_stage2.sh

# Inference with VLA↔Actor switching
./run_rlt_inference.sh

# Test intervention detection only (no VLA needed)
source activate.sh && python scripts/test_intervention.py

# Pure VLA inference (baseline, same as yesterday)
./run_inference.sh
```

## Key Files
| File | Purpose |
|------|---------|
| `run_rlt_stage2.sh` | Stage 2 training launch |
| `run_rlt_inference.sh` | Inference with actor switching |
| `run_inference.sh` | Pure VLA baseline (lerobot-record) |
| `scripts/train_rlt_stage2.py` | Stage 2 training code |
| `scripts/run_rlt_inference.py` | Inference with VLA↔Actor |
| `scripts/test_intervention.py` | Standalone intervention tester |
| `RLT_STAGE2_LEARNINGS.md` | All integration bugs and fixes |
| `checkpoints/rlt_stage1/best_checkpoint.pt` | Frozen RLT encoder |
| `checkpoints/rlt_stage2/` | Actor/critic checkpoints |

## Training Log (20 episodes)
- Episodes 1-5: warmup, reward=0 (no RL)
- Episodes 6-20: all reward=1.0 (success via intervention or manual label)
- Interventions in ~10/15 training episodes (400-900 steps each)
- 326 transitions in buffer (94 from human)
- 630 RL updates, critic loss ~4 trillion (unstable — needs gradient clipping)
