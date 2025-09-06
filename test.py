import os
import time
import argparse
import torch
import torch.nn.functional as F

from env_mario.env import MarioEnvironment
from networks.model import ActorCritic
from config.args import load_config

def parse_args_test():
    """
    Parse command-line arguments and load experiment configuration.

    Returns:
        argparse.Namespace: Parsed arguments with experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Test a trained Mario agent.")
    parser.add_argument("--name", type=str, required=True, help="Experiment name.")
    parser.add_argument("--ckpt", type=str, default="best_model", help="Checkpoint filename (without extension).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for action selection.")
    args_test = parser.parse_args()

    args = load_config(os.path.join("./experiments", args_test.name, "config.yaml"))
    args.output_root = os.path.join("./experiments", args_test.name)
    args.video_path = os.path.join(args.output_root, f"play{args.world}-{args.stage}.mp4")
    args.ckpt = os.path.join(args.output_root, "checkpoints", f"{args_test.ckpt}.pth")
    args.temperature = args_test.temperature
    return args

def test(args, device="cpu"):
    """
    Run the trained Mario agent in the environment and render gameplay.

    Args:
        args (argparse.Namespace): Experiment configuration and arguments.
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    env = MarioEnvironment(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        num_colors=args.num_colors,
        frame_size=args.frame_size,
        num_skip=args.num_skip,
        version=args.version,
        output_path=args.video_path
    )

    # Load model
    model = ActorCritic(
        in_channels=env.num_states,
        num_actions=env.num_actions,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        num_repeat=args.num_repeat,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    # Inference loop
    state = torch.from_numpy(env.reset()).to(device)
    while True:
        logits, _ = model(state)
        dist = torch.distributions.Categorical(probs=F.softmax(logits / args.temperature, dim=-1))
        action = dist.sample().item()
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.env.render()
        time.sleep(0.02)

        if info.get("flag_get", False):
            break

if __name__ == "__main__":
    args = parse_args_test()
    test(args)
