import torch
import torch.nn as nn

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