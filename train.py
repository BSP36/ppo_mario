import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchinfo import summary

from networks.model import ActorCritic, clip_loss


@torch.no_grad()
def play(env, actor: nn.Module, device="cuda"):
    """
    Plays one complete episode using the given actor (policy network) and returns the total reward.

    Args:
        env: The environment to interact with.
        actor (nn.Module): The policy network.
        device (str, optional): Device to run the actor on. Defaults to "cuda".

    Returns:
        float: Total reward accumulated during the episode.
    """
    actor.eval()
    actor = actor.to(device)

    state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)
    total_reward = 0.0
    done = False

    while not done:
        dist = torch.distributions.Categorical(logits=actor(state))
        action = dist.sample()
        state, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state).float().to(device) # s_{t+1}
        total_reward += reward

    return total_reward


@torch.no_grad()
def rollout(
    env,
    actor: nn.Module,
    critic: nn.Module,
    state: torch.Tensor,
    num_local_steps: int,
    gamma: float,
    gae_lambda: float
):
    """
    Collects a trajectory of experience from the environment using the current policy (actor)
    and value function (critic), and computes Generalized Advantage Estimation (GAE).

    Args:
        env: The environment to interact with.
        actor (nn.Module): The policy network.
        critic (nn.Module): The value network.
        state (torch.Tensor): Initial state.
        num_local_steps (int): Number of steps to collect in the rollout.
        gamma (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for GAE.

    Returns:
        Tuple containing:
            - advantages (Tensor): Estimated advantages for each step.
            - log_policies (Tensor): Log probabilities of actions taken.
            - states (Tensor): States encountered during the rollout.
            - value_states (Tensor): Estimated state values.
            - actions (Tensor): Actions taken during the rollout.
    """
    device = state.device
    actor.eval()
    critic.eval()
    state = state.to(device)

    log_policies, actions, values, states, rewards, dones = [], [], [], [], [], []
    for _ in range(num_local_steps):
        dist = torch.distributions.Categorical(logits=actor(state))
        action = dist.sample()

        states.append(state.cpu())
        actions.append(action.cpu())
        log_policies.append(dist.log_prob(action).cpu())
        values.append(critic(state).cpu())

        # Step environment
        state_np, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state_np).float().to(device)

        rewards.append(reward)
        dones.append(done)

        if done:
            state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)

    v_next = critic(state).cpu()

    # Compute GAE advantages
    advantages = []
    gae = 0.0
    for v, r, done in list(zip(values, rewards, dones))[::-1]:
        delta = r + gamma * (1 - done) * v_next - v
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        advantages.insert(0, gae)
        v_next = v

    # Stack results
    advantages = torch.cat(advantages, dim=0).float()
    log_policies = torch.stack(log_policies, dim=0).float()
    rewards = torch.tensor(rewards)[:, None].float()
    states = torch.cat(states, dim=0).float()
    value_states = advantages + torch.cat(values, dim=0).float()
    actions = torch.stack(actions, dim=0)

    return advantages, log_policies, states, value_states, actions, state


def train(env, args, device="cuda"):
    """
    Trains the Actor-Critic model using Proximal Policy Optimization (PPO).

    Args:
        env: The environment to interact with.
        args: Arguments containing hyperparameters and settings.
        device (str, optional): Device to run the training on. Defaults to "cuda".
    """
    # Define model dimensions
    state_dim = env.num_states
    image_size = env.width
    n_actions = env.num_actions

    # Initialize actor and critic networks
    actor = ActorCritic(state_dim, n_actions, image_size=image_size, base_channels=32, num_repeat=1)
    critic = ActorCritic(state_dim, 1, image_size=image_size, base_channels=32, num_repeat=1)
    summary(actor, input_size=(1, state_dim, image_size, image_size))
    # summary(critic, input_size=(1, state_dim, image_size, image_size))

    # Load pre-trained weights if provided
    if args.pre_trained != "":
        checkpoint = torch.load(args.pre_trained, map_location="cpu")
        actor.load_state_dict(checkpoint["actor_state_dict"])
        critic.load_state_dict(checkpoint["critic_state_dict"])
        start_episode = checkpoint["episode"]
    else:
        start_episode = 0

    actor = actor.to(device)
    critic = critic.to(device)
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Initialize environment state
    state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)

    with tqdm(range(0, args.num_episode)) as pbar:
        actor_loss, critic_loss = 0.0, 0.0
        for episode in pbar:
            # Collect rollout/trajectory
            advantages, log_policies_old, states, value_states, actions, state = rollout(
                env, actor, critic, state, args.num_local_steps, args.gamma, args.gae_lambda
            )
            # state = states[-1][None, :].to(device)
            # print(state.shape, state.device)
            # exit()

            # PPO update for several epochs
            actor.train()
            critic.train()
            for _ in range(args.num_epochs):
                indices = torch.randperm(args.num_local_steps)
                for b in range(args.num_local_steps // args.batch_size):
                    batch_idx = indices[b * args.batch_size:(b + 1) * args.batch_size]

                    s = states[batch_idx].to(device)
                    dist = torch.distributions.Categorical(logits=actor(s))
                    log_policy = dist.log_prob(actions[batch_idx].squeeze(-1).to(device)).unsqueeze(-1)

                    # PPO clipped surrogate loss for actor
                    actor_loss = clip_loss(
                        log_policies_old[batch_idx].to(device),
                        log_policy,
                        advantages[batch_idx].to(device),
                        args.epsilon
                    )
                    # Encourage exploration via entropy bonus
                    actor_loss = actor_loss - args.beta * dist.entropy().mean()
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    # Critic loss (value function regression)
                    critic_loss = F.mse_loss(critic(s), value_states[batch_idx].to(device))
                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()

            # Evaluate policy after update
            tot_r = play(env, actor, device=device)

            pbar.set_description(
                f"[EP {start_episode+episode+1}, ALoss {actor_loss:.2f}, CLoss {critic_loss:.2f}, Reward {tot_r:.2f}]"
            )

            # Save model checkpoint periodically
            if (episode + 1) % args.save_interval == 0:
                torch.save({
                    'episode': start_episode + episode + 1,
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'world': args.world,
                    'stage': args.stage,
                    'action_type': args.action_type,
                }, os.path.join(args.save_path, f'{start_episode + episode + 1}.pth'))

if __name__ == "__main__":
    from config.args import parse_args
    from env_mario.env import MarioEnvironment

    args = parse_args()
    env = MarioEnvironment(args.world, args.stage, args.action_type)
    train(env, args, device="mps")

    