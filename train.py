import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from networks.model import ActorCritic, clip_loss
from utils.lr_scheduler import make_linear_decay_scheduler

@torch.no_grad()
def evaluate_policy(env, model: nn.Module, gamma: float, device="cuda"):
    """
    Runs a single episode in the environment using the current policy in evaluation mode.
    At each step, the agent selects the greedy (highest probability) action.
    Returns both the total reward and the discounted reward accumulated over the episode.

    Args:
        env: The environment to interact with.
        model (nn.Module): The policy network.
        gamma (float): Discount factor for computing discounted reward.
        device (str, optional): Device to run the actor on. Defaults to "cuda".

    Returns:
        Tuple[float, float]:
            - total_reward: Sum of all rewards collected in the episode.
            - discounted_reward: Sum of discounted rewards using gamma.
    """
    model.eval()
    model = model.to(device)

    state = torch.from_numpy(env.reset()).float().to(device)
    total_reward = 0.0
    discounted_reward = 0.0
    discount = 1.0
    done = False

    while not done:
        logits, _ = model(state)
        action = torch.argmax(logits, dim=-1)  # Select greedy action
        state_np, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state_np).float().to(device)
        total_reward += reward
        discounted_reward += discount * reward
        discount *= gamma

    return total_reward, discounted_reward


@torch.no_grad()
def rollout(
    env,
    model: nn.Module,
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
        model (nn.Module): The policy and value network.
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
    model.eval()
    state = state.to(device)

    log_policies, actions, values, states, rewards, dones = [], [], [], [], [], []
    for _ in range(num_local_steps):
        logits, value = model(state) # (N, A), (N, 1)
        # 
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample() # (N,)

        states.append(state.cpu()) # s_t
        actions.append(action.cpu()[:, None]) # a_t ~ \pi(\cdot|s_t)
        log_policies.append(dist.log_prob(action).cpu()[:, None]) # \log \pi(a_t|s_t)
        values.append(value.cpu()) # V(s_t)

        # Step environment
        actions_np = action.cpu().numpy().astype(np.int64)
        state, reward, done, info = env.step(actions_np)
        
        rewards.append(torch.from_numpy(reward).float()[:, None]) # (N, 1)
        dones.append(torch.from_numpy(done).long()[:, None]) # (N, 1)
        state = torch.from_numpy(state).float().to(device) # (N, C, H, W)

    _, v_next = model(state)
    v_next = v_next.cpu()

    # Compute GAE advantages
    advantages = []
    gae = 0.0
    for v, r, done in list(zip(values, rewards, dones))[::-1]:
        delta = r + gamma * (1 - done) * v_next - v
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        advantages.insert(0, gae)
        v_next = v

    # Stack results
    advantages = torch.cat(advantages, dim=0).float() # [T, 1]
    log_policies = torch.cat(log_policies, dim=0).float() # [T, 1]
    rewards = torch.cat(rewards, dim=0).float() # [T, 1]
    states = torch.cat(states, dim=0).float() # [T, C, H, W]
    value_states = advantages + torch.cat(values, dim=0).float() # [T, 1]
    actions = torch.cat(actions, dim=0) # [T, 1]
    
    # normalize advantages for the policy loss only
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, log_policies, states, value_states, actions, state


def train(train_env, val_env, args, device="cuda"):
    """
    Trains the Actor-Critic model using Proximal Policy Optimization (PPO).

    Args:
        train_env: The training environment.
        val_env: The validation environment.
        args: Arguments containing hyperparameters and settings.
        device (str, optional): Device to run the training on. Defaults to "cuda".
    """
    # Define model dimensions
    state_dim = val_env.num_states
    image_size = args.frame_size
    n_actions = val_env.num_actions

    # Initialize actor and critic networks
    model = ActorCritic(
        in_channels=state_dim,
        num_actions=n_actions,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        num_repeat=args.num_repeat,
    )
    summary(model, input_size=(1, state_dim, image_size, image_size))

    # Load pre-trained weights if provided
    if args.pre_trained is not None:
        checkpoint = torch.load(args.pre_trained, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        start_update = checkpoint["update"]
    else:
        start_update = 0

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = make_linear_decay_scheduler(
        optimizer,
        total_updates=args.num_ppo_updates,
        warmup_ratio=0.01,
        min_lr_ratio=0.1
    )
    
    # Summary writer
    writer = SummaryWriter(os.path.join(args.output_root, "runs"))

    # Initialize environment stateprint
    state = torch.from_numpy(train_env.reset()).float().to(device) # [N, C, H, W]
    
    best_reward = -float('inf')
    for update in range(args.num_ppo_updates):
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], update)
        # Collect rollout/trajectory
        advantages, log_policies_old, states, value_states, actions, state = rollout(
            train_env, model, state, args.num_local_steps, args.gamma, args.gae_lambda
        )

        # PPO update for several epochs
        model.train()
        actor_loss_tot, entropy_loss_tot, critic_loss_tot = 0.0, 0.0, 0.0
        for _ in range(args.num_epochs):
            num_samples = advantages.shape[0]
            indices = torch.randperm(num_samples)
            for b in range(num_samples // args.batch_size):
                batch_idx = indices[b * args.batch_size:(b + 1) * args.batch_size]

                s = states[batch_idx].to(device)
                logits, value = model(s)
                dist = torch.distributions.Categorical(logits=logits)
                log_policy = dist.log_prob(actions[batch_idx][:, 0].to(device))[:, None]

                # PPO clipped surrogate loss for actor
                actor_loss = clip_loss(
                    log_policies_old[batch_idx].to(device),
                    log_policy,
                    advantages[batch_idx].to(device),
                    args.epsilon
                )
                # Encourage exploration via entropy bonus
                entropy_loss = dist.entropy().mean()
                actor_loss = actor_loss - args.beta * entropy_loss
                # Critic loss (value function regression)
                critic_loss = F.mse_loss(value, value_states[batch_idx].to(device))

                loss = actor_loss + 0.5 * critic_loss
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                actor_loss_tot += actor_loss.item()
                entropy_loss_tot += entropy_loss.item()
                critic_loss_tot += critic_loss.item()

        scheduler.step()

        # Evaluate policy after update
        total_reward, discounted_reward = evaluate_policy(val_env, model, args.gamma, device=device)

        # Output
        norm = args.num_epochs * (num_samples // args.batch_size)
        print(
            f"[Update {start_update + update + 1}/{args.num_ppo_updates}] "
            f"Actor Loss: {actor_loss_tot / norm:.4f} | "
            f"Critic Loss: {critic_loss_tot / norm:.4f} | "
            f"Entropy: {entropy_loss_tot / norm:.4f} | "
            f"Total Reward: {total_reward:.2f}"
        )
        writer.add_scalar('Loss/Actor', actor_loss_tot / norm, update)
        writer.add_scalar('Loss/Clip', (actor_loss_tot + args.beta * entropy_loss_tot) / norm, update)
        writer.add_scalar('Loss/Entropy', entropy_loss_tot / norm, update)
        writer.add_scalar('Loss/Critic', critic_loss_tot /norm, update)
        writer.add_scalar('Reward/Total', total_reward, update)
        writer.add_scalar('Reward/Discounted', discounted_reward, update)

        # Save model checkpoint periodically
        if (update + 1) % args.save_interval == 0:
            torch.save({
                'update': start_update + update + 1,
                'state_dict': model.state_dict(),
                'world': args.world,
                'stage': args.stage,
                'action_type': args.action_type,
            }, os.path.join(args.output_root, "checkpoints", f'{start_update + update + 1}.pth'))
        
        if best_reward < total_reward:
            best_reward = total_reward
            torch.save({
                'update': start_update + update + 1,
                'state_dict': model.state_dict(),
                'world': args.world,
                'stage': args.stage,
                'action_type': args.action_type,
            }, os.path.join(args.output_root, "checkpoints", f'best_model.pth'))
    
    writer.close()

if __name__ == "__main__":
    from config.args import parse_args
    from env_mario.env import MarioEnvironment
    from env_mario.vec_env import build_vec_env

    args = parse_args()
    train_env = build_vec_env(
        num_envs=args.num_envs,
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        num_colors=args.num_colors,
        frame_size=args.frame_size,
        num_skip=args.num_skip,
        version=args.version,
        output_path=None,
        base_seed=0
    )
    val_env = MarioEnvironment(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        num_colors=args.num_colors,
        frame_size=args.frame_size,
        num_skip=args.num_skip,
        version=args.version,
        output_path=None,
    )
    train(train_env, val_env, args, device=args.device)

    