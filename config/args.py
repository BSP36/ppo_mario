import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_actor', type=float, default=1e-4)
    parser.add_argument('--lr_critic', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument("--num_local_steps", type=int, default=512)
    parser.add_argument("--num_global_steps", type=int, default=int(5e6))
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    # parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros")
    parser.add_argument("--save_path", type=str, default="checkpoint")
    parser.add_argument("--num_episode", type=int, default=10000)
    # 
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--pre_trained", type=str, default="")
    args = parser.parse_args()

    assert args.num_local_steps % args.batch_size == 0
    os.makedirs(args.save_path, exist_ok=True)
    
    return args