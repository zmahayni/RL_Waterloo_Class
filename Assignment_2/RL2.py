import numpy as np
from Assignment_1 import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''
        nS = self.mdp.nStates
        nA = self.mdp.nActions

        # Empirical model estimates
        Ns = np.zeros((nA, nS), dtype=int)            # N(s,a)
        Nss = np.zeros((nA, nS, nS), dtype=int)       # N(s,a,s')
        Rhat = initialR.copy().astype(float)          # mean reward estimate per (a,s)

        # Helper to build estimated MDP from counts and defaults
        def build_estimated_mdp():
            Test = np.zeros((nA, nS, nS))
            Rest = np.zeros((nA, nS))
            for a in range(nA):
                for s in range(nS):
                    if Ns[a, s] > 0:
                        Test[a, s, :] = Nss[a, s, :] / max(1, Ns[a, s])
                        Rest[a, s] = Rhat[a, s]
                    else:
                        Test[a, s, :] = defaultT[a, s, :]
                        Rest[a, s] = initialR[a, s]
            return Test, Rest

        # Initialize policy/value using defaults
        Test, Rest = build_estimated_mdp()
        est_mdp = MDP.MDP(Test, Rest, self.mdp.discount)
        V, _, _ = est_mdp.valueIteration(initialV=np.zeros(nS), nIterations=1000, tolerance=1e-6)
        policy = est_mdp.extractPolicy(V)

        # Run episodes and update model online
        for _ in range(nEpisodes):
            s = s0
            for _t in range(nSteps):
                # epsilon-greedy action w.r.t current policy
                if np.random.rand() < epsilon:
                    a = np.random.randint(nA)
                else:
                    a = int(policy[s])

                r, sp = self.sampleRewardAndNextState(s, a)

                # Update counts and reward mean for (a,s)
                Ns[a, s] += 1
                Nss[a, s, sp] += 1
                # incremental mean
                Rhat[a, s] += (r - Rhat[a, s]) / Ns[a, s]

                # Rebuild estimated MDP and re-plan
                Test, Rest = build_estimated_mdp()
                est_mdp = MDP.MDP(Test, Rest, self.mdp.discount)
                V, _, _ = est_mdp.valueIteration(initialV=V, nIterations=100, tolerance=1e-4)
                policy = est_mdp.extractPolicy(V)

                s = sp

        return [V,policy]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor).  Use epsilon = 1 / # of iterations.

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        nA = self.mdp.nActions
        # For bandit, single state assumed
        counts = np.zeros(nA, dtype=int)
        sums = np.zeros(nA, dtype=float)

        for t in range(1, nIterations + 1):
            eps = 1.0 / t
            if np.random.rand() < eps:
                a = np.random.randint(nA)
            else:
                # choose best empirical mean; tie-break uniformly
                means = np.divide(sums, np.maximum(1, counts))
                max_val = np.max(means)
                candidates = np.where(means == max_val)[0]
                a = int(np.random.choice(candidates))

            r = self.sampleReward(self.mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r

        empiricalMeans = np.divide(sums, np.maximum(1, counts))
        return empiricalMeans

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        nA = self.mdp.nActions
        alpha = prior[:, 0].astype(float).copy()
        beta = prior[:, 1].astype(float).copy()
        counts = np.zeros(nA, dtype=int)
        sums = np.zeros(nA, dtype=float)

        for _t in range(nIterations):
            # sample k means for each arm and average them
            samples = np.zeros(nA)
            for a in range(nA):
                draws = np.random.beta(alpha[a], beta[a], size=int(k))
                samples[a] = np.mean(draws)
            a_sel = int(np.argmax(samples))

            r = self.sampleReward(self.mdp.R[a_sel, 0])
            counts[a_sel] += 1
            sums[a_sel] += r

            # Beta-Bernoulli update
            if r >= 1:  # for Bernoulli rewards in {0,1}
                alpha[a_sel] += 1
            else:
                beta[a_sel] += 1

        empiricalMeans = np.divide(sums, np.maximum(1, counts))
        return empiricalMeans

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''
        nA = self.mdp.nActions
        counts = np.zeros(nA, dtype=int)
        sums = np.zeros(nA, dtype=float)

        t = 0
        # initialize by pulling each arm once (if possible)
        for a in range(nA):
            if t >= nIterations:
                break
            r = self.sampleReward(self.mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r
            t += 1

        while t < nIterations:
            means = np.divide(sums, np.maximum(1, counts))
            ucb = means + np.sqrt(2.0 * np.log(max(1, t)) / np.maximum(1, counts))
            a = int(np.argmax(ucb))
            r = self.sampleReward(self.mdp.R[a, 0])
            counts[a] += 1
            sums[a] += r
            t += 1

        empiricalMeans = np.divide(sums, np.maximum(1, counts))
        return empiricalMeans

    def qLearning(self, s0, initialQ, nEpisodes, nSteps, epsilon=0.05):
        '''Q-learning with epsilon-greedy exploration and alpha = 1/N(s,a)'''
        nS = self.mdp.nStates
        nA = self.mdp.nActions
        Q = initialQ.copy().astype(float)
        Nsa = np.zeros((nA, nS), dtype=int)
        gamma = self.mdp.discount

        for _ in range(nEpisodes):
            s = s0
            for _t in range(nSteps):
                if np.random.rand() < epsilon:
                    a = np.random.randint(nA)
                else:
                    # greedy w.r.t Q
                    q_s = Q[:, s]
                    max_q = np.max(q_s)
                    candidates = np.where(q_s == max_q)[0]
                    a = int(np.random.choice(candidates))

                r, sp = self.sampleRewardAndNextState(s, a)
                Nsa[a, s] += 1
                alpha = 1.0 / Nsa[a, s]
                td_target = r + gamma * np.max(Q[:, sp])
                Q[a, s] += alpha * (td_target - Q[a, s])
                s = sp

        # derive greedy policy from Q
        policy = np.argmax(Q, axis=0).astype(int)
        return [Q, policy]