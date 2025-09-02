import argparse
import torch
from env_mario.env import create_train_env
from ppo_mario.networks.model import ActorCritic
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import torch.nn.functional as F
import time
from torchinfo import summary


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Contra Nes""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument("--saved_path", type=str, default="checkpoint")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--name", type=str, default="best_model")
    parser.add_argument("--T", type=float, default=0.01)
    args = parser.parse_args()
    return args


def test(args, device="cpu"):
    torch.manual_seed(123)
    if args.action_type == "right":
        actions = RIGHT_ONLY
    elif args.action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(
        args.world, args.stage, actions,
        f"{args.output_path}/video_{args.world}_{args.stage}.mp4"
        )
    state_dim = env.observation_space.shape[0]
    image_size = env.observation_space.shape[1]
    model = ActorCritic(
        state_dim,
        len(actions),
        image_size=image_size, base_channels=32, num_repeat=1)
    summary(model, input_size=(1, state_dim, image_size, image_size))
    pth_path = f"{args.saved_path}/{args.name}.pth"
    checkpoint = torch.load(pth_path, map_location="cpu")
    print(f'episode: {checkpoint["episode"]}')
    model.load_state_dict(checkpoint["actor_state_dict"])
    model.eval().to(device)
    state = torch.from_numpy(env.reset()).to(device)
    while True:
        logits = model(state)
        # policy = F.softmax(logits, dim=1)
        # action = torch.argmax(policy).item()
        dist = torch.distributions.Categorical(probs=F.softmax(logits / args.T, dim=-1))
        action = dist.sample().item()
        # print(action.shape)
        state, reward, done, info = env.step(action)
        # print(reward)
        
        state = torch.from_numpy(state)
        env.render()
        time.sleep(0.02)
        # time.sleep(0.5)
        if info["flag_get"]:
            print("World {} stage {} completed".format(args.world, args.stage))
            break


if __name__ == "__main__":
    args = get_args()
    test(args)
