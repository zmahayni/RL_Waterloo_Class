import gymnasium as gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch, utils.common
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Deep Q Learning
# Slide 14
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-spring20/slides/cs885-lecture4b.pdf

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
MINIBATCH_SIZE = 10     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 5        # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 300          # Total number of episodes to learn over
TEST_EPISODES = 1       # Test episodes after every train episode
HIDDEN = 512            # Hidden nodes
TARGET_UPDATE_FREQ = 10 # Target network update frequency
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.01      # At the end, keep epsilon at this value

# Global variables
EPSILON = STARTING_EPSILON
Q = None

# Create environment / buffer / networks / optimizer
def greedy_policy(env, obs):
    global Q
    if isinstance(obs, tuple):  # (obs, info) from Gymnasium
        obs = obs[0]
    obs = np.asarray(obs, dtype=np.float32)
    obs = t.f(obs).view(-1, OBS_N)
    return torch.argmax(Q(obs)).item()

def create_everything(seed):
    utils.seed.seed(seed)

    # Modern Gym API: CartPole-v1, seed via reset + action space
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    test_env = gym.make("CartPole-v1")
    test_env.reset(seed=10+seed)
    test_env.action_space.seed(10+seed)

    buf = utils.buffers.ReplayBuffer(BUFSIZE)

    Q = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)

    Qt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)

    OPT = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT

# Update a target network using a source network
def update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)

# Epsilon-greedy policy
def policy(env, obs):
    global EPSILON, Q

    # Normalize obs to a flat float array (handle (obs, info) or dict if ever present)
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict):
        if 'obs' in obs:
            obs = obs['obs']
        elif 'state' in obs:
            obs = obs['state']
        else:
            obs = np.asarray(list(obs.values()), dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)

    obs = t.f(obs).view(-1, OBS_N)  # to torch tensor on device

    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        qvalues = Q(obs)
        action = torch.argmax(qvalues).item()

    # Anneal epsilon
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    return action

# One training step block over a minibatch
def update_networks(epi, buf, Q, Qt, OPT):
    if len(buf.buf) < MINIBATCH_SIZE:
        return 0.0  # skip until enough data
    # Sample a minibatch (s, a, r, s', d)
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)

    # Q(s,a) for chosen actions
    qvalues = Q(S).gather(1, A.view(-1, 1)).squeeze()

    # max_a' Qt(s',a')
    q2values = torch.max(Qt(S2), dim=1).values

    # Targets: r + gamma * max_a' Qt(s', a') * (1 - done)
    targets = R + GAMMA * q2values * (1 - D)

    # MSE loss
    loss = torch.nn.MSELoss()(targets.detach(), qvalues)

    OPT.zero_grad()
    loss.backward()
    OPT.step()

    # Periodic hard update of target net
    if epi % TARGET_UPDATE_FREQ == 0:
        update(Qt, Q)

    return loss.item()

# Train for one seed, return rolling-25 test reward curve
def train(seed):
    global EPSILON, Q
    print(f"Seed={seed}")

    env, test_env, buf, Q, Qt, OPT = create_everything(seed)
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []

    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:
        # Collect one episode into replay buffer
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)

        # Train after some warmup episodes
        if epi >= TRAIN_AFTER_EPISODES:
            for _ in range(TRAIN_EPOCHS):
                update_networks(epi, buf, Q, Qt, OPT)

        # Evaluate
        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for _ in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, greedy_policy, render=False)
            Rews.append(sum(R))
        testRs.append(sum(Rews)/TEST_EPISODES)


        # Rolling-25 average curve
        last25testRs.append(sum(testRs[-25:]) / len(testRs[-25:]))
        pbar.set_description(f"R25({last25testRs[-1]:.1f})")

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean ± std band
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    xs = range(len(mean))
    plt.plot(xs, mean, color=color, label=label)
    plt.fill_between(xs, np.maximum(mean-std, 0), np.minimum(mean+std, 200), color=color, alpha=0.3)

if __name__ == "__main__":
    # =========================
    # EXPERIMENT 1: Target net update frequency
    # =========================
    target_freqs = [1, 10, 50, 100]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    plt.figure(figsize=(8,5))
    for i in range(len(target_freqs)):
        freq = target_freqs[i]
        color = colors[i]

        TARGET_UPDATE_FREQ = freq

        curves = []
        for j in range(len(SEEDS)):
            seed = SEEDS[j]
            curves.append(train(seed))

        plot_arrays(curves, color, label=f"target_update_every={freq}")

    plt.title("CartPole DQN: rolling-25 avg reward vs episodes (target update freq)")
    plt.xlabel("Episode")
    plt.ylabel("Avg reward (last 25 episodes)")
    plt.ylim(0, 200)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # =========================
    # EXPERIMENT 2: Mini-batch size
    # =========================
    TARGET_UPDATE_FREQ = 10
    batch_sizes = [1, 10, 50, 100]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    plt.figure(figsize=(8,5))
    for i in range(len(batch_sizes)):
        bs = batch_sizes[i]
        color = colors[i]

        MINIBATCH_SIZE = bs

        curves = []
        for j in range(len(SEEDS)):
            seed = SEEDS[j]
            curves.append(train(seed))

        plot_arrays(curves, color, label=f"batch_size={bs}")

    plt.title("CartPole DQN: rolling-25 avg reward vs episodes (batch size)")
    plt.xlabel("Episode")
    plt.ylabel("Avg reward (last 25 episodes)")
    plt.ylim(0, 200)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
