import gymnasium as gym
import numpy as np
import utils2.envs, utils2.seed, utils2.buffers, utils2.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

SEEDS = [1, 2, 3, 4, 5]
t = utils2.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4  # State space size
ACT_N = 2  # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000  # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1  # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64  # How many examples to sample per train step
GAMMA = 0.99  # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4  # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10  # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25  # Train for these many epochs every time
BUFSIZE = 10000  # Replay buffer size
EPISODES = 500  # Total number of episodes to learn over
TEST_EPISODES = 10  # Test episodes
HIDDEN = 512  # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10  # Target network update frequency

# Suggested constants
ATOMS = 51  # Number of atoms for distributional network
ZRANGE = [0, 200]  # Range for Z projection

# Global variables
EPSILON = STARTING_EPSILON
Z = None


# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils2.seed.seed(seed)
    env = utils2.envs.TimeLimit(utils2.envs.NoisyCartPole(), 500)
    test_env = utils2.envs.TimeLimit(utils2.envs.NoisyCartPole(), 500)
    buf = utils2.buffers.ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N * ATOMS),
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N * ATOMS),
    ).to(DEVICE)
    OPT = torch.optim.Adam(Z.parameters(), lr=LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT


# Create epsilon-greedy policy
def policy(env, obs):
    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)

    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        with torch.no_grad():
            z_dist = Z(obs).view(-1, ACT_N, ATOMS)
            z_dist = torch.softmax(z_dist, dim=2)
            support = torch.linspace(ZRANGE[0], ZRANGE[1], ATOMS).to(DEVICE)
            q_values = torch.sum(z_dist * support, dim=2)
            action = torch.argmax(q_values).item()

    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))

    return action


# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)

    support = torch.linspace(ZRANGE[0], ZRANGE[1], ATOMS).to(DEVICE)
    delta_z = (ZRANGE[1] - ZRANGE[0]) / (ATOMS - 1)

    with torch.no_grad():
        z_dist_next = Zt(S2).view(-1, ACT_N, ATOMS)
        z_dist_next = torch.softmax(z_dist_next, dim=2)
        q_next = torch.sum(z_dist_next * support, dim=2)
        best_actions = torch.argmax(q_next, dim=1)
        z_dist_next_best = z_dist_next[range(MINIBATCH_SIZE), best_actions]

        projected_atoms = R.unsqueeze(1) + GAMMA * support.unsqueeze(0) * (
            1 - D.unsqueeze(1)
        )
        projected_atoms = torch.clamp(projected_atoms, ZRANGE[0], ZRANGE[1])

        b_float = (projected_atoms - ZRANGE[0]) / delta_z
        b_lower = torch.floor(b_float).long().clamp(0, ATOMS - 1)
        b_upper = torch.ceil(b_float).long().clamp(0, ATOMS - 1)

        weight_upper = b_float - b_lower.float()
        weight_lower = 1 - weight_upper

        m = torch.zeros(MINIBATCH_SIZE, ATOMS).to(DEVICE)
        m.scatter_add_(1, b_lower, weight_lower * z_dist_next_best)
        m.scatter_add_(1, b_upper, weight_upper * z_dist_next_best)

    z_dist = Z(S).view(-1, ACT_N, ATOMS)
    z_dist = torch.softmax(z_dist, dim=2)
    z_dist_a = z_dist[range(MINIBATCH_SIZE), A.squeeze()]

    log_preds = torch.log(z_dist_a + 1e-8)
    loss = -(m.detach() * log_preds).sum(dim=1).mean()

    OPT.zero_grad()
    loss.backward()
    OPT.step()

    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Zt.load_state_dict(Z.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):
    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:
        S, A, R = utils2.envs.play_episode_rb(env, policy, buf)

        if epi >= TRAIN_AFTER_EPISODES:
            for tri in range(TRAIN_EPOCHS):
                update_networks(epi, buf, Z, Zt, OPT)

        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils2.envs.play_episode(test_env, policy, render=False)
            Rews += [sum(R)]
        testRs += [sum(Rews) / TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:]) / len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs


# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(
        range(len(mean)),
        np.maximum(mean - std, 0),
        np.minimum(mean + std, 500),
        color=color,
        alpha=0.3,
    )


if __name__ == "__main__":
    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, "b", "c51")
    plt.legend(loc="best")
    plt.show()
