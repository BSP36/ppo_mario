# Proximal Policy Optimization
> ⚠️ Note: This document contains several mathematical equations.  
> GitHub’s Markdown renderer may not display them correctly.  
> For proper formatting, please open this file in a local Markdown viewer (e.g., VS Code) or a LaTeX-compatible editor.

- PPO: https://arxiv.org/pdf/1707.06347
- TRPO: https://arxiv.org/pdf/1502.05477
- A2C: https://arxiv.org/pdf/1602.01783
- GAE: https://arxiv.org/pdf/1506.02438

## 1. Introduction
### 1-1. Markov Decision Process (MDP)
---

Let's begin by introducing the **Markov Decision Process (MDP)**, which serves as the foundation of reinforcement learning.

An MDP is defined by the tuple $(S, A, P, R)$, where:

- **State space $S$**: The set of all possible states that the environment can occupy.
- **Action space $A$**: The set of all possible actions the agent can take.
- **Transition probability $P(s_{t+1} \mid s_t, a_t)$**: The probability that the environment transitions to state $s_{t+1} \in S$ at time step $t+1$, given that it was in state $s_t \in S$ and the agent took action $a_t \in A$ at time $t$.
- **Reward function $R(s_t, a_t, s_{t+1})$**: The immediate reward received after transitioning from state $s_t$ to $s_{t+1}$ as a result of taking action $a_t$.

By definition, an MDP satisfies the Markov property: the probability of transitioning to the next state depends only on the current state and action, not on the sequence of previous states and actions.

### 1-2. Policy
---

A **policy** $\pi_{\theta}(a_t|s_t)$ is a function, parameterized by $\theta$, that defines the agent's behavior by specifying the probability of taking action $a_t$ given state $s_t$. Once the policy and the initial state distribution $\rho_0(s_0)$ are specified, the evolution of the system can be described as:

$$
\begin{equation}
P_{\pi_{\theta}, T}(s_0, a_0, s_1, a_1, \ldots, s_{T}) =
\rho_0(s_0)\prod_{t=0}^{T-1}
P(s_{t+1}|s_{t}, a_{t})\,\pi_{\theta}(a_t|s_t).
\end{equation}
$$

In reinforcement learning, the objective is to optimize the policy to maximize the expected cumulative reward:

$$
\begin{gather}
\theta^*_t := 
\arg\max_{\theta} \eta_t(\pi_{\theta}),
\\
\eta_t(\pi_{\theta}) := 
\mathbb{E}_{P_{\pi_{\theta}, \infty}}\left[
    G_t
\right],
\\
G_t := \sum_{k=0}^{\infty}\gamma^{k}R(s_{t+k}, a_{t+k}, s_{t+k+1}),

\end{gather}
$$

where $0 \leq \gamma \leq 1$ is the discount factor. The discount factor determines the importance of future rewards, with smaller values making the agent prioritize immediate rewards.

### 1-3. Useful Functions
---

#### 1-3-1. State-Action Value Function ($Q$)

There are several strategies to solve the maximization problem in reinforcement learning. One standard approach is to use the **state-action value function**:

$$
Q_{\pi}(s, a) := \mathbb{E}_{P_{\pi, \infty}}\left[ G_t \mid s_t = s, a_t = a \right],
$$

where $Q_{\pi}(s, a)$ represents the expected return when starting from state $s$, taking action $a$, and thereafter following policy $\pi$.
Note that $Q_{\pi}(s, a)$ is independent of the time step, as the Markov property ensures that the future is conditionally independent of the past given the present state and action.

The state-action value function satisfies the **Bellman equation**:

$$
\begin{align}
Q_{\pi}(s, a)
& =
\mathbb{E}_{P_{\pi, \infty}}
\left[
    R(s_t, a_t, s_{t+1}) + \gamma G_{t+1}
     |
    s_{t} = s, a_{t} = a
\right]
\\
\\
&= \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q_{\pi}(s', a') \right],
\end{align}
$$

where $s'$ denotes the next state, and $a'$ denotes the next action.

#### 1-3-2. State Value Function ($V$)
---

A related function is the **state value function**:

$$
V_{\pi}(s) := \mathbb{E}_{a \sim \pi(\cdot|s)}\left[ Q_{\pi}(s, a) \right] = \sum_{a \in A} \pi(a|s) Q_{\pi}(s, a).
$$

The state value function also satisfies a Bellman equation:

$$
V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V_{\pi}(s') \right].
$$

The policy optimization objective can be rewritten in terms of the state value function as:

$$
\begin{gather}
\eta_t(\pi)
=
\sum_{s_t}P(s_t|\pi) V_{\pi}(s_t),
\\
P(s_t|\pi)
:=
\sum_{s_0, a_0, ..., a_{t-1}}P_{\pi, t}(s_0, a_0, ..., a_{t-1}, s_t).
\end{gather}
$$

#### 1-3-3. Advantage Function

While both $Q$ and $V$ functions can be noisy in practice, their difference often yields a more stable and numerically tractable quantity. This motivates the definition of the **advantage function**:

$$
\begin{align}
A_{\pi}(s, a) & := Q_{\pi}(s, a) - V_{\pi}(s) \\
& =
\sum_{s'}P(s'|s, a)\left\{
R(s, a, s') + \gamma V_{\pi}(s') - V_{\pi}(s)
\right\}
\end{align}
$$

which measures how much better (or worse) taking action $a$ in state $s$ is compared to the average action under policy $\pi$.

## 2. Surrogate Objectives

### 2-1. Trust Region Policy Optimization (TRPO)
---
#### 2-1-1. Formulation
The original objective $\eta$ is generally difficult to optimize directly. To obtain a more tractable form, we make use of a key relationship between objectives corresponding to different policies:
$$
\begin{align}
\eta_0(\pi') - \eta_0(\pi)
& =
-\mathbb{E}_{s_0\sim \rho_{0}}
\left[
    V_{\pi}(s_0)
\right]
+
\mathbb{E}_{P_{\pi', \infty}}
\left[
    \sum_{k=0}^{\infty}\gamma^k R(s_{k}, a_{k}, s_{k+1})
\right]
\\
& =
\mathbb{E}_{P_{\pi', \infty}}
\left[
    - V_{\pi}(s_0)
    +
    \sum_{k=0}^{\infty}\gamma^k R(s_{k}, a_{k}, s_{k+1})
\right]
\\
& =
\mathbb{E}_{P_{\pi', \infty}}
\left[
    \sum_{k=0}^{\infty}\gamma^k\left\{
         R(s_{k}, a_{k}, s_{k+1})
        + \gamma V_{\pi}(s_{k+1}) - V_{\pi}(s_k)
    \right\}
\right]
\end{align}
$$
Define
$$
\begin{align}
A_k & := 
\mathbb{E}_{P_{\pi', \infty}}
\left[
R(s_{k}, a_{k}, s_{k+1})
        + \gamma V_{\pi}(s_{k+1}) - V_{\pi}(s_k)
\right]
\\
& =
\sum_{s_k, a_k, s_{k+1}}
P(s_{k+1}|s_k, a_k)\pi'(a_k|s_k)
P(s_k|\pi')
\left[
R(s_{k}, a_{k}, s_{k+1})
        + \gamma V_{\pi}(s_{k+1}) - V_{\pi}(s_k)
\right]
\\
& =
\sum_{s_k, a_k}
\pi'(a_k|s_k)
P(s_k|\pi')
A_{\pi}(a_k, s_k)
\end{align}
$$
Thus, 
$$
\begin{align}
\eta_0(\pi') - \eta_0(\pi)
& =
\sum_{k=0}^{\infty}\gamma^k
\sum_{s_k, a_k}
\pi'(a_k|s_k)
P(s_k|\pi')
A_{\pi}(a_k, s_k)
\\
& =
\sum_{s}
\left(
\sum_{k=0}^{\infty}\gamma^k
P(s_k|\pi')
\right)
\sum_{a}
\pi'(a|s)
A_{\pi}(a, s)
\\
& =:
\sum_{s, a}\rho_{\pi'}(s)
\pi'(a|s)
A_{\pi}(a, s),
\end{align}
$$

This result shows that any policy update $\pi \to \pi'$ for which
$\sum_{s, a}\rho_{\pi'}(s)\pi'(a|s)A_{\pi}(a, s) > 0$
will improve the expected return. Hence, the next parameter vector $\theta'$ should be chosen to maximize this term:

$$
\theta \to \theta' = \arg\max_{\theta'}\sum_{s, a}\rho_{\theta'}(s)
\pi_{\theta'}(a|s)
A_{\pi_{\theta}}(a, s) 
$$

However, the dependence of $\rho_{\pi'}(s)$ on $\theta'$ makes this optimization intractable.
To address this, we adopt a **local approximation** by assuming $\rho_{\pi'}(s) \approx \rho_{\pi}(s)$.
Under this assumption, the surrogate objective becomes

$$
\begin{align}
L_{TRPO}(\theta)
& :=
\sum_{s, a}\rho_{\theta_{\mathrm{old}}}(s)
\pi_{\theta}(a|s)
A_{\pi_{\theta_{\mathrm{old}}}}(a, s) 
\\
& =
\mathbb{E}_{a, s \sim \rho_{\pi_{\mathrm{old}}}\pi_{\theta_{\mathrm{old}}}}
\left[
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_{\mathrm{old}}}(a|s)}
A_{\pi_{\theta_{\mathrm{old}}}}(a, s) 
\right].
\end{align}
$$

Note that this surrogate objective is only valid if
$\pi_{\theta}(a|s) / \pi_{\theta_{\mathrm{old}}}(a|s) \approx 1$.
Therefore, the policy update must be constrained to stay within a trust region.
In TRPO, this is enforced by introducing the Lagrange multiplier on the KL divergence $D_{KL}(\pi_{\theta_{\mathrm{old}}}|\pi_{\theta})$.

#### 2-1-2. Calculation
For numerical implementation, we first specify the procedure for collecting training data.
The current policy $\pi_{\mathrm{old}}$ is fixed, and Markov chains of the form
${(s_0, a_0), (s_1, a_1), \dots}$ are generated as trajectories for training.

The TRPO objective can then be written as


$$
\begin{align}
L_{TRPO}(\theta)
 & =
\sum_{t=0}^{\infty}
\sum_{a_t, s_t}
P(s_t, a_t|\pi_{\theta_{\mathrm{old}}})
\left[
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
\gamma^t A_{\pi_{\theta_{\mathrm{old}}}}(a_t, s_t) 
\right]
\\
& =
\mathbb{E}_{t\sim [0, \infty], \, (s_t, a_t) \sim P_{\theta}(s_t, a_t|\pi_{\theta_{\mathrm{old}}})}
\left[
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
\gamma^t A_{\pi_{\theta_{\mathrm{old}}}}(a_t, s_t) 
\right].
\\
& =:
\mathbb{E}_{t}
\left[
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
\hat{A}_t
\right].
\end{align}
$$
In practice, these expectations are approximated by Monte Carlo sampling.
Specifically, one should sample $(s_t, a_t)$ such that $t$ is chosen uniformly at random, 
and the state–action pair at timestep $t$ is selected independently of other timesteps.
This condition is satisfied by drawing $(s, a)$ uniformly at random from the set of collected Markov chains.
(The treatment of $\hat{A}_t$ will be discussed shortly.)


Following Schulman et al. (2015, 2017), the surrogate objective in TRPO and PPO omits explicit $\gamma^t$ weighting and uses uniform timestep sampling as a practical variance-reduction technique (see, e.g., [arXiv:2306.13284](https://arxiv.org/pdf/2306.13284)).
Adopting this convention, we will simply set
$\hat{A}_t = A_{\pi_{\theta_{\mathrm{old}}}}$.



### 2-2. Proximal Policy Optimization (PPO)
---
The TRPO framework provides a principled way to optimize policies while ensuring stability.
However, in practice, controlling the Lagrange multiplier for the KL-divergence constraint is challenging and computationally expensive.

To address this, OpenAI proposed Proximal Policy Optimization (PPO) as a simpler and more efficient alternative.
Instead of enforcing the trust region via a constrained optimization problem, PPO directly clips the policy ratio, yielding the surrogate objective:

$$
\begin{gather}
L_{\mathrm{CLIP}}(\theta)
:=
\mathbb{E}_{t}\left[
    \min(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)
\right],
\\
r_t := \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}
\end{gather}
$$

Here, $r_t(\theta)$ represents the probability ratio between the new and old policies.
The clipping operation explicitly restricts this ratio to remain within the interval $[1-\epsilon, 1+\epsilon]$, preventing excessively large policy updates while still encouraging improvement.


## 3. Actor-Critic scheme
TBA
<!-- ### 3-1. Critic
Both PPO and TRPO schemes require the advantage function of $A_{\pi_{\mathrm{old}}}$.
However, each advantage function demands to calculate $E_{P^{\theta}_{\infty}}[\cdot]$, and this calculation is extreamly time-consuming. Thus, let us estimate a state-value function with neural network:
$$
v_{\phi}(s) \sim V_{\pi_{}}(s),
$$
Then $V_{\phi}$ is refered to as **critic**, and we must train this function with a loss function
$$
L_{VF}(\phi) = \mathbb{E}_{t}\left[\left|v_{\phi}(s_t) - V_{\pi_{\mathrm{old}}}(s_t)\right|^2\right]
$$
And the GT is evaluated from Markov chanin by
$$

$$


### 3-2. Temporal differences -->
