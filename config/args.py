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

    # Learning rates
    parser.add_argument('--lr_actor', type=float, default=1e-4,
                        help='Learning rate for the actor network.')
    parser.add_argument('--lr_critic', type=float, default=5e-4,
                        help='Learning rate for the critic network.')

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
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs per update.')
    parser.add_argument('--num_local_steps', type=int, default=512,
                        help='Number of steps to run per environment per update.')
    parser.add_argument('--num_global_steps', type=int, default=int(5e6),
                        help='Total number of environment steps to train for.')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Number of steps between saving checkpoints.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization) coefficient.')

    # Output and environment settings
    parser.add_argument('--output', type=str, default="test1-1",
                        help='Name of the output directory for logs and checkpoints.')
    parser.add_argument('--num_episode', type=int, default=10000,
                        help='Number of episodes to run.')

    # Super Mario Bros environment configuration
    parser.add_argument('--world', type=int, default=1,
                        help='World number in Super Mario Bros.')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage number in Super Mario Bros.')
    parser.add_argument('--action_type', type=str, default="simple",
                        help='Action set type (e.g., simple, complex).')
    parser.add_argument('--pre_trained', type=str, default=None,
                        help='Path to pre-trained model (if any).')

    args = parser.parse_args()

    # Set up output directories
    output_root  = os.path.join("./experiments", args.output)
    os.makedirs(os.path.join(output_root, "checkpoints"), exist_ok=True)

    # Sanity check: batch size must divide local steps
    assert args.num_local_steps % args.batch_size == 0, \
        "num_local_steps must be divisible by batch_size."

    # Save args as YAML
    with open(os.path.join(output_root, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)
    
    args.output = output_root
    
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