import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Assignment_1 import MDP
from Assignment_2 import RL2


def build_maze_mdp():
    # Copy of TestRL2Maze.py construction (without prints)
    T = np.zeros([4,17,17])
    a = 0.8; b = 0.1
    # up (0)
    T[0,0,0]=a+b; T[0,0,1]=b
    T[0,1,0]=b; T[0,1,1]=a; T[0,1,2]=b
    T[0,2,1]=b; T[0,2,2]=a; T[0,2,3]=b
    T[0,3,2]=b; T[0,3,3]=a+b
    T[0,4,4]=b; T[0,4,0]=a; T[0,4,5]=b
    T[0,5,4]=b; T[0,5,1]=a; T[0,5,6]=b
    T[0,6,5]=b; T[0,6,2]=a; T[0,6,7]=b
    T[0,7,6]=b; T[0,7,3]=a; T[0,7,7]=b
    T[0,8,8]=b; T[0,8,4]=a; T[0,8,9]=b
    T[0,9,8]=b; T[0,9,5]=a; T[0,9,10]=b
    T[0,10,9]=b; T[0,10,6]=a; T[0,10,11]=b
    T[0,11,10]=b; T[0,11,7]=a; T[0,11,11]=b
    T[0,12,12]=b; T[0,12,8]=a; T[0,12,13]=b
    T[0,13,12]=b; T[0,13,9]=a; T[0,13,14]=b
    T[0,14,16]=1
    T[0,15,11]=a; T[0,15,14]=b; T[0,15,15]=b
    T[0,16,16]=1
    # down (1)
    T[1,0,0]=b; T[1,0,4]=a; T[1,0,1]=b
    T[1,1,0]=b; T[1,1,5]=a; T[1,1,2]=b
    T[1,2,1]=b; T[1,2,6]=a; T[1,2,3]=b
    T[1,3,2]=b; T[1,3,7]=a; T[1,3,3]=b
    T[1,4,4]=b; T[1,4,8]=a; T[1,4,5]=b
    T[1,5,4]=b; T[1,5,9]=a; T[1,5,6]=b
    T[1,6,5]=b; T[1,6,10]=a; T[1,6,7]=b
    T[1,7,6]=b; T[1,7,11]=a; T[1,7,7]=b
    T[1,8,8]=b; T[1,8,12]=a; T[1,8,9]=b
    T[1,9,8]=b; T[1,9,13]=a; T[1,9,10]=b
    T[1,10,9]=b; T[1,10,14]=a; T[1,10,11]=b
    T[1,11,10]=b; T[1,11,15]=a; T[1,11,11]=b
    T[1,12,12]=a+b; T[1,12,13]=b
    T[1,13,12]=b; T[1,13,13]=a; T[1,13,14]=b
    T[1,14,16]=1
    T[1,15,14]=b; T[1,15,15]=a+b
    T[1,16,16]=1
    # left (2)
    T[2,0,0]=a+b; T[2,0,4]=b
    T[2,1,1]=b; T[2,1,0]=a; T[2,1,5]=b
    T[2,2,2]=b; T[2,2,1]=a; T[2,2,6]=b
    T[2,3,3]=b; T[2,3,2]=a; T[2,3,7]=b
    T[2,4,0]=b; T[2,4,4]=a; T[2,4,8]=b
    T[2,5,1]=b; T[2,5,4]=a; T[2,5,9]=b
    T[2,6,2]=b; T[2,6,5]=a; T[2,6,10]=b
    T[2,7,3]=b; T[2,7,6]=a; T[2,7,11]=b
    T[2,8,4]=b; T[2,8,8]=a; T[2,8,12]=b
    T[2,9,5]=b; T[2,9,8]=a; T[2,9,13]=b
    T[2,10,6]=b; T[2,10,9]=a; T[2,10,14]=b
    T[2,11,7]=b; T[2,11,10]=a; T[2,11,15]=b
    T[2,12,8]=b; T[2,12,12]=a+b
    T[2,13,9]=b; T[2,13,12]=a; T[2,13,13]=b
    T[2,14,16]=1
    T[2,15,11]=a; T[2,15,14]=b; T[2,15,15]=b
    T[2,16,16]=1
    # right (3)
    T[3,0,0]=b; T[3,0,1]=a; T[3,0,4]=b
    T[3,1,1]=b; T[3,1,2]=a; T[3,1,5]=b
    T[3,2,2]=b; T[3,2,3]=a; T[3,2,6]=b
    T[3,3,3]=a+b; T[3,3,7]=b
    T[3,4,0]=b; T[3,4,5]=a; T[3,4,8]=b
    T[3,5,1]=b; T[3,5,6]=a; T[3,5,9]=b
    T[3,6,2]=b; T[3,6,7]=a; T[3,6,10]=b
    T[3,7,3]=b; T[3,7,7]=a; T[3,7,11]=b
    T[3,8,4]=b; T[3,8,9]=a; T[3,8,12]=b
    T[3,9,5]=b; T[3,9,10]=a; T[3,9,13]=b
    T[3,10,6]=b; T[3,10,11]=a; T[3,10,14]=b
    T[3,11,7]=b; T[3,11,11]=a; T[3,11,15]=b
    T[3,12,8]=b; T[3,12,13]=a; T[3,12,12]=b
    T[3,13,9]=b; T[3,13,14]=a; T[3,13,13]=b
    T[3,14,16]=1
    T[3,15,11]=b; T[3,15,15]=a+b
    T[3,16,16]=1

    R = -1 * np.ones([4,17])
    R[:,14] = 100
    R[:,9] = -70
    R[:,16] = 0
    discount = 0.95
    return MDP.MDP(T, R, discount)


def simulate_model_based_returns(n_episodes=200, n_steps=100, epsilon=0.05, trials=100, plan_interval=10):
    mdp = build_maze_mdp()
    s0 = 0
    defaultT = np.ones([mdp.nActions, mdp.nStates, mdp.nStates]) / mdp.nStates
    initialR = np.zeros([mdp.nActions, mdp.nStates])
    gamma = mdp.discount

    returns = np.zeros((trials, n_episodes))

    for tr in range(trials):
        rl = RL2.RL2(mdp, np.random.normal)
        # Internal model
        nS, nA = mdp.nStates, mdp.nActions
        Ns = np.zeros((nA, nS), dtype=int)
        Nss = np.zeros((nA, nS, nS), dtype=int)
        Rhat = initialR.copy().astype(float)

        def build_est():
            Test = np.zeros((nA, nS, nS))
            Rest = np.zeros((nA, nS))
            for a in range(nA):
                for s in range(nS):
                    if Ns[a, s] > 0:
                        Test[a, s] = Nss[a, s] / max(1, Ns[a, s])
                        Rest[a, s] = Rhat[a, s]
                    else:
                        Test[a, s] = defaultT[a, s]
                        Rest[a, s] = initialR[a, s]
            return Test, Rest

        # Initial plan
        Test, Rest = build_est()
        est_mdp = MDP.MDP(Test, Rest, mdp.discount)
        V, _, _ = est_mdp.valueIteration(initialV=np.zeros(nS), nIterations=200, tolerance=1e-4)
        policy = est_mdp.extractPolicy(V)

        for ep in range(n_episodes):
            s = s0
            G = 0.0
            gpow = 1.0
            for t in range(n_steps):
                if np.random.rand() < epsilon:
                    a = np.random.randint(nA)
                else:
                    a = int(policy[s])
                r, sp = rl.sampleRewardAndNextState(s, a)
                G += gpow * r
                gpow *= gamma
                Ns[a, s] += 1
                Nss[a, s, sp] += 1
                Rhat[a, s] += (r - Rhat[a, s]) / Ns[a, s]
                s = sp
            # Re-plan only every few episodes for speed
            if (ep + 1) % plan_interval == 0:
                Test, Rest = build_est()
                est_mdp = MDP.MDP(Test, Rest, mdp.discount)
                V, _, _ = est_mdp.valueIteration(initialV=V, nIterations=100, tolerance=1e-3)
                policy = est_mdp.extractPolicy(V)
            returns[tr, ep] = G
    return returns


def simulate_qlearning_returns(n_episodes=200, n_steps=100, epsilon=0.05, trials=100):
    mdp = build_maze_mdp()
    s0 = 0
    gamma = mdp.discount
    returns = np.zeros((trials, n_episodes))

    for tr in range(trials):
        rl = RL2.RL2(mdp, np.random.normal)
        nS, nA = mdp.nStates, mdp.nActions
        Q = np.zeros((nA, nS))
        Nsa = np.zeros((nA, nS), dtype=int)
        for ep in range(n_episodes):
            s = s0
            G = 0.0
            gpow = 1.0
            for t in range(n_steps):
                if np.random.rand() < epsilon:
                    a = np.random.randint(nA)
                else:
                    q_s = Q[:, s]
                    max_q = np.max(q_s)
                    cands = np.where(q_s == max_q)[0]
                    a = int(np.random.choice(cands))
                r, sp = rl.sampleRewardAndNextState(s, a)
                Nsa[a, s] += 1
                alpha = 1.0 / Nsa[a, s]
                td_target = r + gamma * np.max(Q[:, sp])
                Q[a, s] += alpha * (td_target - Q[a, s])
                G += gpow * r
                gpow *= gamma
                s = sp
            returns[tr, ep] = G
    return returns


def main():
    n_episodes = 200
    n_steps = 100
    trials = 100
    epsilon = 0.05
    plan_interval = 10  # plan every 10 episodes for speed

    mb_returns = simulate_model_based_returns(n_episodes, n_steps, epsilon, trials, plan_interval).mean(axis=0)
    ql_returns = simulate_qlearning_returns(n_episodes, n_steps, epsilon, trials).mean(axis=0)

    x = np.arange(n_episodes)
    plt.figure(figsize=(8,5))
    plt.plot(x, mb_returns, label='Model-based RL (eps=0.05)')
    plt.plot(x, ql_returns, label='Q-learning (eps=0.05)')
    plt.xlabel('Episode #')
    plt.ylabel('Avg cumulative discounted reward per episode')
    plt.title('Maze: Model-based RL vs Q-learning (avg over {} trials)'.format(trials))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'maze_rl.png'), dpi=150)


if __name__ == '__main__':
    main()
