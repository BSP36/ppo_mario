# Proximal Policy Optimization on Super Mario Bros

This repository provides an implementation of **Proximal Policy Optimization (PPO)** applied to the classic *Super Mario Bros* environments.
It allows you to train and evaluate reinforcement learning agents that learn to play Mario using PyTorch.

See [docs/PPO.md](./docs/PPO.md) for a brief introduction to Proximal Policy Optimization.

---

## Getting Started

### Step 1. Set up the environment

We recommend using a virtual environment (e.g., `venv`):

```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

If you plan to use CUDA for acceleration, please ensure that you install the appropriate PyTorch version for your GPU and CUDA toolkit.

---

### Step 2. Train on Stage 1-1

To begin training an agent on World 1-1:

```bash
python train.py --name test1-1 --world 1 --stage 1 --device cuda:0 --version 0 --frame_size 64
```

* The `--version` flag specifies the environment mode (see [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) for details).
* Training logs are written to TensorBoard. You can monitor progress with:

```bash
tensorboard --logdir ./experiments/test1-1/runs
```

---

### Step 3. Run Inference

After training, you can test your trained agent:

```bash
python test.py --name test1-1 --ckpt best_model
```

A display window will open, allowing you to watch your agent play Mario in real time.

---
