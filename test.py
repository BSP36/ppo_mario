import os
import time
import argparse
import torch
import torch.nn.functional as F
from torchinfo import summary

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
    args.output = os.path.join("./experiments", args_test.name)
    args.output_path = os.path.join(args.output, f"play{args.world}-{args.stage}.mp4")
    args.ckpt = os.path.join(args.output, "checkpoints", f"{args_test.ckpt}.pth")
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
        checkpoint["world"],
        checkpoint["stage"],
        checkpoint["action_type"],
        output_path=args.output_path
    )

    state_dim = env.num_states
    image_size = env.width

    # Load model
    model = ActorCritic(state_dim, env.num_actions, image_size, args.num_repeat, args.base_channels)
    summary(model, input_size=(1, state_dim, image_size, image_size))
    model.load_state_dict(checkpoint["actor_state_dict"])
    model.eval().to(device)

    # Inference loop
    state = torch.from_numpy(env.reset()).to(device)
    while True:
        logits = model(state)
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
