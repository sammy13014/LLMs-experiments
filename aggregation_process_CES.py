from Likelihoods import *
import numpy as np


def bayes_plus_CES_aggregate(N_run, Lx, Ly, rho, beliefs_init):
    beliefs = [beliefs_init]
    for i in range(N_run):
        mu = beliefs[-1]                     
        agg = np.zeros((2, K))
        for agent, other in [(0, 1), (1, 0)]:
            if rho == 0:
                agg[agent] = np.sqrt(mu[agent] * mu[other])         
            else:
                agg[agent] = (0.5*mu[agent]**rho + 0.5*mu[other]**rho) ** (1/rho)
            agg[agent] /= agg[agent].sum()

        obs_x, obs_y = sample_true_state(Lx, Ly)
        post = np.zeros((2, K))
        for k in range(K):
            post[0, k] = agg[0, k] * Lx[k][ord(obs_x) - ord('A')]
            post[1, k] = agg[1, k] * Ly[k][ord(obs_y) - ord('A')]
        post[0] /= post[0].sum()
        post[1] /= post[1].sum()

        beliefs.append(post)
    return beliefs

def plot_bayes_plus_CES_aggregate(N_run, Lx, Ly, rhos, beliefs_init):
    plt.figure(figsize=(12, 8))
    for rho in rhos:
        beliefs = bayes_plus_CES_aggregate(N_run, Lx, Ly, rho, beliefs_init)
        beliefs = np.array(beliefs)
        plt.plot(beliefs[:, 0, 0], label=f'Agent 1, rho={rho}')
        plt.plot(beliefs[:, 1, 0], label=f'Agent 2, rho={rho}', linestyle='--')
    plt.xlabel('Time step')
    plt.ylabel('Belief in state A')
    plt.title('Bayesian + CES Aggregation of Beliefs')
    plt.legend()
    plt.grid()
    plt.show()

def conververgence_rate(beliefs):
    t=np.arange(1, len(beliefs))
    rate_iter=np.clip(beliefs[1:],1e-300,None)
    return -np.log(rate_iter)/t[:,None,None]

def plot_bayes_plus_CES_convergence_rate(N_run, Lx, Ly, rhos, beliefs_init):
    false_states = [s for s in STATES if s != TRUE]

    all_rates = {}
    for rho in rhos:
        beliefs = np.array(bayes_plus_CES_aggregate(N_run, Lx, Ly, rho, beliefs_init))
        all_rates[rho] = conververgence_rate(beliefs)

    for agent in range(2):
        for state_k in false_states:
            plt.figure(figsize=(12, 8))
            for rho in rhos:
                plt.plot(all_rates[rho][:, agent, state_k], label=f"rho={rho}")
            plt.xlabel("Time step")
            plt.ylabel(r"$-\frac{1}{t}\log \mu_t(\theta)$")
            plt.title(f"Agent {agent+1}, learning rate on state {LABELS[state_k]}")
            plt.legend(fontsize=8)
            plt.grid()
            plt.show()
            
