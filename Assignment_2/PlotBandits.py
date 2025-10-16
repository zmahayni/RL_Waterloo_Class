import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Assignment_1 import MDP
import RL2


def sampleBernoulli(mean):
    return 1 if np.random.rand(1) < mean else 0


def run_epsilon_greedy_rewards(mdp, n_iters, n_trials):
    nA = mdp.nActions
    rewards = np.zeros((n_trials, n_iters))
    for trial in range(n_trials):
        counts = np.zeros(nA, dtype=int)
        sums = np.zeros(nA, dtype=float)
        for t in range(1, n_iters + 1):
            eps = 1.0 / t
            if np.random.rand() < eps:
                a = np.random.randint(nA)
            else:
                means = np.divide(sums, np.maximum(1, counts))
                max_val = np.max(means)
                candidates = np.where(means == max_val)[0]
                a = int(np.random.choice(candidates))
            r = sampleBernoulli(mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r
            rewards[trial, t-1] = r
    return rewards


def run_thompson_rewards(mdp, n_iters, n_trials, k=1):
    nA = mdp.nActions
    rewards = np.zeros((n_trials, n_iters))
    for trial in range(n_trials):
        alpha = np.ones(nA)
        beta = np.ones(nA)
        for t in range(n_iters):
            samples = np.zeros(nA)
            for a in range(nA):
                draws = np.random.beta(alpha[a], beta[a], size=int(k))
                samples[a] = np.mean(draws)
            a_sel = int(np.argmax(samples))
            r = sampleBernoulli(mdp.R[a_sel, 0])
            if r >= 1:
                alpha[a_sel] += 1
            else:
                beta[a_sel] += 1
            rewards[trial, t] = r
    return rewards


def run_ucb_rewards(mdp, n_iters, n_trials):
    nA = mdp.nActions
    rewards = np.zeros((n_trials, n_iters))
    for trial in range(n_trials):
        counts = np.zeros(nA, dtype=int)
        sums = np.zeros(nA, dtype=float)
        t = 0
        for a in range(nA):
            if t >= n_iters:
                break
            r = sampleBernoulli(mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r
            rewards[trial, t] = r
            t += 1
        while t < n_iters:
            means = np.divide(sums, np.maximum(1, counts))
            ucb = means + np.sqrt(2.0 * np.log(max(1, t)) / np.maximum(1, counts))
            a = int(np.argmax(ucb))
            r = sampleBernoulli(mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r
            rewards[trial, t] = r
            t += 1
    return rewards


def main():
    T = np.array([[[1]], [[1]], [[1]]])
    R = np.array([[0.3], [0.5], [0.7]])
    discount = 0.999
    mdp = MDP.MDP(T, R, discount)

    n_iters = 200
    n_trials = 1000

    eg_rewards = run_epsilon_greedy_rewards(mdp, n_iters, n_trials).mean(axis=0)
    ts_rewards = run_thompson_rewards(mdp, n_iters, n_trials, k=1).mean(axis=0)
    ucb_rewards = run_ucb_rewards(mdp, n_iters, n_trials).mean(axis=0)

    x = np.arange(n_iters)
    plt.figure(figsize=(8,5))
    plt.plot(x, ucb_rewards, label='UCB')
    plt.plot(x, eg_rewards, label='Epsilon-greedy (epsilon=1/t)')
    plt.plot(x, ts_rewards, label='Thompson (k=1, Beta(1,1))')
    plt.xlabel('Iteration #')
    plt.ylabel('Average reward at iteration')
    plt.title('Bandit Algorithms (averaged over {} trials)'.format(n_trials))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'bandits.png'), dpi=150)


if __name__ == '__main__':
    main()
