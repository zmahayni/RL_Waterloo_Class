import numpy as np
import Assignment_1.MDP as MDP

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
        '''Model-based RL with epsilon-greedy exploration; updates policy via DP each step.

        Inputs:
        s0 -- initial state
        defaultT -- default transition when (s,a) not visited (|A|x|S|x|S|)
        initialR -- initial reward estimate (|A|x|S|)
        nEpisodes -- # episodes
        nSteps -- # steps per episode
        epsilon -- exploration prob

        Outputs:
        V -- final value function
        policy -- final policy
        '''

        nA = self.mdp.nActions
        nS = self.mdp.nStates
        gamma = self.mdp.discount

        T_counts = np.zeros((nA, nS, nS))
        SA_counts = np.zeros((nA, nS))
        R_sums = np.zeros((nA, nS))

        T_hat = defaultT.astype(float).copy()
        R_hat = initialR.astype(float).copy()

        V_hat = np.zeros(nS)
        policy_hat = np.zeros(nS, dtype=int)

        for _ in range(nEpisodes):
            s = s0
            for _ in range(nSteps):
                # Plan on current model
                model = MDP.MDP(T_hat, R_hat, gamma)
                V_hat, _, _ = model.valueIteration(initialV=V_hat, nIterations=100, tolerance=1e-6)
                policy_hat = model.extractPolicy(V_hat)

                # Epsilon-greedy action
                if np.random.rand() < epsilon:
                    a = np.random.randint(nA)
                else:
                    a = int(policy_hat[s])

                # Interact with true env
                r, s_next = self.sampleRewardAndNextState(s, a)

                # Update counts and estimates
                SA_counts[a, s] += 1.0
                R_sums[a, s] += r
                T_counts[a, s, s_next] += 1.0

                # Update R_hat
                R_hat[a, s] = R_sums[a, s] / SA_counts[a, s]

                # Update T_hat for (a,s)
                T_hat[a, s, :] = defaultT[a, s, :]  # fallback
                if SA_counts[a, s] > 0:
                    T_hat[a, s, :] = T_counts[a, s, :] / SA_counts[a, s]

                s = s_next

        # Final planning
        model = MDP.MDP(T_hat, R_hat, gamma)
        V_hat, _, _ = model.valueIteration(initialV=V_hat, nIterations=1000, tolerance=1e-8)
        policy_hat = model.extractPolicy(V_hat)

        return [V_hat, policy_hat]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon-greedy for bandits. Use epsilon_t = 1 / t.

        Inputs:
        nIterations -- # pulls

        Outputs:
        empiricalMeans -- mean reward per arm (|A|)
        '''

        nA = self.mdp.nActions
        state = 0  # single-state bandit
        counts = np.zeros(nA)
        sums = np.zeros(nA)
        means = np.zeros(nA)

        for t in range(1, nIterations + 1):
            eps = 1.0 / t
            if np.random.rand() < eps:
                a = np.random.randint(nA)
            else:
                # Break ties randomly
                best = np.where(means == means.max())[0]
                a = int(np.random.choice(best))
            r = self.sampleReward(self.mdp.R[a, state])
            counts[a] += 1.0
            sums[a] += r
            means[a] = sums[a] / counts[a]
        return means

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling for Bernoulli bandits.

        Inputs:
        prior -- Beta prior per arm (|A|x2): [alpha, beta]
        nIterations -- # pulls
        k -- # samples per arm (average these k samples)

        Outputs:
        empiricalMeans -- mean reward per arm (|A|)
        '''

        nA = self.mdp.nActions
        state = 0
        alpha_beta = prior.astype(float).copy()  # shape (nA,2)
        counts = np.zeros(nA)
        sums = np.zeros(nA)
        means = np.zeros(nA)

        for _ in range(nIterations):
            samples = np.zeros(nA)
            for a in range(nA):
                draws = np.random.beta(alpha_beta[a,0], alpha_beta[a,1], size=int(k))
                samples[a] = draws.mean()
            # Choose arm
            best = np.where(samples == samples.max())[0]
            a = int(np.random.choice(best))
            # Observe reward
            r = self.sampleReward(self.mdp.R[a, state])
            # Update posterior
            alpha_beta[a,0] += r
            alpha_beta[a,1] += (1 - r)
            # Track empirical means
            counts[a] += 1.0
            sums[a] += r
            means[a] = sums[a] / counts[a]        
        return means

    def UCBbandit(self,nIterations):
        '''UCB1 for bandits.

        Inputs:
        nIterations -- # pulls

        Outputs: 
        empiricalMeans -- mean reward per arm (|A|)
        '''

        nA = self.mdp.nActions
        state = 0
        counts = np.zeros(nA)
        sums = np.zeros(nA)
        means = np.zeros(nA)

        t = 0
        # Initialize by pulling each arm once (if possible)
        for a in range(nA):
            if t >= nIterations:
                break
            r = self.sampleReward(self.mdp.R[a, state])
            counts[a] += 1.0
            sums[a] += r
            means[a] = sums[a] / counts[a]
            t += 1

        while t < nIterations:
            ucb = np.zeros(nA)
            total = counts.sum() + 1e-12
            for a in range(nA):
                if counts[a] == 0:
                    ucb[a] = np.inf
                else:
                    bonus = np.sqrt(2.0 * np.log(total) / counts[a])
                    ucb[a] = means[a] + bonus
            best = np.where(ucb == np.max(ucb))[0]
            a = int(np.random.choice(best))
            r = self.sampleReward(self.mdp.R[a, state])
            counts[a] += 1.0
            sums[a] += r
            means[a] = sums[a] / counts[a]
            t += 1

        return means