import os
import yaml
import argparse


def parse_args():
    """
    Parse command-line arguments to configure PPO training for Super Mario Bros.

    Returns:
        argparse.Namespace: Parsed arguments with configuration values.
    """
    parser = argparse.ArgumentParser(
        description="PPO configuration for Super Mario Bros reinforcement learning."
    )

    # Device configuration
    parser.add_argument('--device', type=str, default="cpu",
                        help='Device to use for training (e.g., cpu, cuda:0, mps).')

    # PPO and RL hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor for rewards.')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='Lambda parameter for Generalized Advantage Estimation (GAE).')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Entropy coefficient for exploration.')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Clipping parameter for PPO surrogate objective.')
    
    #  Model parameters
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Number of base channels in ConvNeXt blocks.')
    parser.add_argument('--num_stages', type=int, default=3,
                        help='Number of stages in the network.')
    parser.add_argument('--num_repeat', type=int, default=2,
                        help='Number of times to repeat ConvNeXt blocks.')

    # Training parameters
    parser.add_argument('--num_envs', type=int, default=8,
                        help='Number of parallel environment instances.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of epochs per update.')
    parser.add_argument('--num_local_steps', type=int, default=128,
                        help='Number of steps to run per environment per update.')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Number of steps between saving checkpoints.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate for the actor-critic optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) coefficient.')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='Path to pre-trained model (if any).')

    # Output and environment settings
    parser.add_argument('--name', type=str, default="test1-1",
                        help='Experiment name for saving outputs.')
    parser.add_argument('--num_ppo_updates', type=int, default=5000,
                        help='Total number of PPO updates to perform.')

    # Super Mario Bros environment configuration
    parser.add_argument('--version', type=int, default=3,
                        help='Environment version (0, 1, 2, or 3).')
    parser.add_argument('--frame_size', type=int, default=32,
                        help='Size (height and width) to which each frame is resized.')
    parser.add_argument('--num_colors', type=int, default=3,
                        help='Number of color channels for observations (1 for grayscale, 3 for RGB).')
    parser.add_argument('--num_skip', type=int, default=4,
                        help='Number of frames to skip (repeat the same action).')
    parser.add_argument('--world', type=int, default=1,
                        help='World number in Super Mario Bros.')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage number in Super Mario Bros.')
    parser.add_argument('--action_type', type=str, default="simple",
                        help='Action set type (e.g., simple, complex).')

    args = parser.parse_args()

    # Set up output directories
    args.output_root  = os.path.join("./experiments", args.name)
    os.makedirs(os.path.join(args.output_root, "checkpoints"), exist_ok=True)

    # Sanity checks
    assert args.version in [0, 1, 2, 3], "Version must be 0, 1, 2, or 3."
    assert args.action_type in ["right", "simple", "complex"], "Invalid action type."
    assert args.num_colors in [1, 3], "num_colors must be 1 (grayscale) or 3 (RGB)."

    # Sanity check: batch size must divide local steps
    assert (args.num_local_steps * args.num_envs) % args.batch_size == 0, \
        "Batch size must divide (num_local_steps * num_envs)."

    # Save args as YAML
    with open(os.path.join(args.output_root, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    return args


def load_config(config_path):
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        argparse.Namespace: Configuration parameters as a namespace.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)