import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from module.model import ActorCritic, policy_loss
from torchinfo import summary


@torch.no_grad()
def play(env, actor, device="mps"):
    actor = actor.eval().to(device)

    state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)
    total_reward = 0.0
    cond = True
    while cond:
        probs = actor(state)
        dist = torch.distributions.Categorical(probs=F.softmax(probs, dim=-1))
        action = dist.sample()
        state, reward, done, _ = env.step(action.item())
        state = torch.from_numpy(state).float().to(device) # s_{t+1}

        total_reward += reward
        cond = not done
    
    return total_reward


def train(env, args, device="mps"):
    torch.manual_seed(1)

    state_dim = env.num_states
    image_size = env.width
    n_actions = env.num_actions
    actor = ActorCritic(state_dim, n_actions, image_size=image_size, base_channels=32, num_repeat=1)
    critic = ActorCritic(state_dim, 1, image_size=image_size, base_channels=32, num_repeat=1)
    summary(actor, input_size=(1, state_dim, image_size, image_size))
    # summary(critic, input_size=(1, state_dim, image_size, image_size))
    # exit()
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
    # initialization
    state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)
    with tqdm(range(0, args.num_episode)) as pbar:
        actor_loss, critic_loss = 0.0, 0.0
        for episode in pbar:
            # Rollout
            old_log_policies, actions, values, states, rewards, dones = [], [], [], [], [], []
            with torch.no_grad():
                for _ in range(args.num_local_steps):
                    states.append(state) # s_t

                    probs = actor(state)
                    dist = torch.distributions.Categorical(probs=F.softmax(probs, dim=-1))
                    action = dist.sample()

                    actions.append(action.cpu()) # a_t
                    old_log_policies.append(dist.log_prob(action).cpu()) # log \pi(a_t|s_t)
                    values.append(critic(state).cpu()) # V(s_t)

                    # step with environment
                    state, reward, done, _ = env.step(action.item())
                    state = torch.from_numpy(state).float().to(device) # s_{t+1}

                    rewards.append(reward)
                    dones.append(done)

                    if done:
                        state = torch.from_numpy(env.reset()[0]).float()[None, :].to(device)
                        done = False

                v_next = critic(state).cpu()

                # advantage (GAE, generalized advantage estimation)
                advantages = []
                gae = 0.0
                for v, r, done in list(zip(values, rewards, dones))[::-1]:
                    delta = r + args.gamma * (1 - done) * v_next - v
                    gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                    advantages.insert(0, gae)
                    v_next = v

                # Registration
                advantages = torch.cat(advantages, dim=0).float()
                old_log_policies = torch.stack(old_log_policies, dim=0).float()
                rewards = torch.tensor(rewards)[:, None].float()
                states = torch.cat(states, dim=0).float()
                value_states = advantages + torch.cat(values, dim=0).float()
                actions = torch.stack(actions, dim=0)

            # Train
            for _ in range(args.num_epochs):
                indices = torch.randperm(args.num_local_steps)
                for b in range(args.num_local_steps // args.batch_size):
                    batch_idx = indices[b * args.batch_size:(b + 1) * args.batch_size]

                    s = states[batch_idx].to(device)
                    probs = actor(s)
                    dist = torch.distributions.Categorical(probs=F.softmax(probs, dim=-1))
                    log_policy = dist.log_prob(actions[batch_idx].squeeze(-1).to(device)).unsqueeze(-1)

                    actor_loss = policy_loss(
                        old_log_policies[batch_idx].to(device),
                        log_policy,
                        advantages[batch_idx].to(device),
                        args.epsilon
                    )
                    actor_loss = actor_loss - args.beta * torch.mean(dist.entropy())
                    optimizer_actor.zero_grad()
                    actor_loss.backward()
                    optimizer_actor.step()

                    # ctiric
                    critic_loss = F.mse_loss(critic(s), value_states[batch_idx].to(device))

                    optimizer_critic.zero_grad()
                    critic_loss.backward()
                    optimizer_critic.step()
            
            tot_r = play(env, actor)
            
            pbar.set_description(
                f"[EP {start_episode+episode+1}, ALoss {actor_loss:.2}, CLoss {critic_loss:.2}, Reward {tot_r:.2}]")
            if (episode + 1) % args.save_interval == 0:
                torch.save({
                    'episode': start_episode+episode+1,
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'world': args.world,
                    'stage': args.stage,
                    'action_type': args.action_type,
                    }, os.path.join(args.save_path, f'{start_episode+episode+1}.pth')
                )
            print("")

if __name__ == "__main__":
    from argument import get_args
    from module.env import MarioEnvironment

    args = get_args()
    env = MarioEnvironment(args.world, args.stage, args.action_type)
    train(env, args)

    