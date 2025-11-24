import gymnasium as gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 2
ACT_N = 2
STARTING_EPSILON = 1.0
STEPS_MAX = 10000
EPSILON_END = 0.1
MINIBATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-4
TRAIN_AFTER_EPISODES = 10
TRAIN_EPOCHS = 25
BUFSIZE = 10000
EPISODES = 2000
TEST_EPISODES = 10
HIDDEN = 512
TARGET_NETWORK_UPDATE_FREQ = 10
LSTM_HIDDEN = 128

EPSILON = STARTING_EPSILON
Q = None
hidden_state = None


class DRQN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_N, HIDDEN)
        self.lstm = torch.nn.LSTM(HIDDEN, LSTM_HIDDEN, batch_first=True)
        self.fc2 = torch.nn.Linear(LSTM_HIDDEN, ACT_N)

    def forward(self, x, hidden):
        x = torch.relu(self.fc1(x))
        if hidden is None:
            lstm_out, hidden = self.lstm(x.unsqueeze(1))
        else:
            lstm_out, hidden = self.lstm(x.unsqueeze(1), hidden)
        q_values = self.fc2(lstm_out.squeeze(1))
        return q_values, hidden

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, LSTM_HIDDEN).to(DEVICE),
            torch.zeros(1, batch_size, LSTM_HIDDEN).to(DEVICE),
        )


# Create environment
# Create replay buffer
# Create network for Q(s, a)
# Create target network
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    test_env = utils.envs.TimeLimit(utils.envs.PartiallyObservableCartPole(), 200)
    buf = utils.buffers.ReplayBuffer(BUFSIZE, recurrent=True)
    Q = DRQN().to(DEVICE)
    Qt = DRQN().to(DEVICE)
    OPT = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
    return env, test_env, buf, Q, Qt, OPT


# Create epsilon-greedy policy
# TODO: Adjust this policy to handle hidden states?
def policy(env, obs):
    global EPSILON, EPSILON_END, STEPS_MAX, Q, hidden_state
    obs = t.f(obs).view(-1, OBS_N)

    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        with torch.no_grad():
            qvalues, hidden_state = Q(obs, hidden_state)
            action = torch.argmax(qvalues).item()

    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))

    return action


# Update networks
def update_networks(epi, buf, Q, Qt, OPT):
    S, A, R, S2, D = buf.sample(MINIBATCH_SIZE, t)

    h_q = Q.init_hidden(1)
    h_qt = Qt.init_hidden(1)

    qvalues_list = []
    for i in range(MINIBATCH_SIZE):
        q_out, h_q = Q(S[i : i + 1], h_q)
        qvalues_list.append(q_out)
    qvalues = torch.cat(qvalues_list, dim=0)
    qvalues = qvalues.gather(1, A.view(-1, 1)).squeeze()

    h_qt = Qt.init_hidden(1)
    q2values_list = []
    for i in range(MINIBATCH_SIZE):
        q2_out, h_qt = Qt(S2[i : i + 1], h_qt)
        q2values_list.append(q2_out)
    q2values = torch.cat(q2values_list, dim=0)
    q2values = torch.max(q2values, dim=1).values

    targets = R + GAMMA * q2values * (1 - D)

    loss = torch.nn.MSELoss()(targets.detach(), qvalues)

    OPT.zero_grad()
    loss.backward()
    OPT.step()

    if epi % TARGET_NETWORK_UPDATE_FREQ == 0:
        Qt.load_state_dict(Q.state_dict())

    return loss.item()


# Play episodes
# Training function
def train(seed):
    global EPSILON, Q, hidden_state
    print("Seed=%d" % seed)

    env, test_env, buf, Q, Qt, OPT = create_everything(seed)
    EPSILON = STARTING_EPSILON
    hidden_state = Q.init_hidden(1)

    testRs = []
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:
        hidden_state = Q.init_hidden(1)
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)

        if epi >= TRAIN_AFTER_EPISODES:
            for tri in range(TRAIN_EPOCHS):
                update_networks(epi, buf, Q, Qt, OPT)

        Rews = []
        for epj in range(TEST_EPISODES):
            hidden_state = Q.init_hidden(1)
            S, A, R = utils.envs.play_episode(test_env, policy, render=False)
            Rews += [sum(R)]
        testRs += [sum(Rews) / TEST_EPISODES]

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
        np.minimum(mean + std, 200),
        color=color,
        alpha=0.3,
    )


if __name__ == "__main__":
    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, "b", "drqn")
    plt.legend(loc="best")
    plt.show()
