import os
import numpy as np
from mkv.value_iteration import value_iteration

def normalize_range(vals, left, right):
    """
    normalize to (left, right)
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return left + (vals - min_val) / (max_val - min_val) * (right - left)

def normalize(vals):
    """
    normalize to (0, 1)
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)

def compute_state_expectation(feat_map, mdp, trajs):
    """
    compute state visit expectation from the demostration trajs
    """
    state_exp = np.zeros([feat_map.shape[1]])
    for episode in trajs:
        for step in episode:
            state_idx = mdp.s2idx[step]
            state_exp += feat_map[state_idx, :]
    state_exp = state_exp / len(trajs)

    return state_exp

def compute_state_visit_freq(feat_map, mdp, trajs, policy, deterministic):
    """
    compute the expected state visit frequency under policy
    inputs:
        feat_map: feature for each state, here use identity matrix
        mdp: ice hockey mdp, fom mdp.hockeymdp.HockeyMdp
        trajs: demonstrations extracted from play by play data. output of "extract_demonstrations(file)" function
        policy: output of mkv.value_iteration.value_iteration
    return:
        p: Nx1 vector, state visit frequency
    """
    N_STATES = feat_map.shape[1]
    t_list = [len(episode) for episode in trajs]
    T = max(t_list)
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([N_STATES, T])

    for episode in trajs:
        begin_idx = mdp.s2idx[ episode[0] ]
        mu[begin_idx, 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)

    for s in mdp.s:
        for t in range(T-1):
            if deterministic:
                mu[mdp.s2idx[s], t+1] = sum(
                    [  mu[mdp.s2idx[pre_s], t] * mdp.get_trans_prob(pre_s, policy[pre_s], s) for pre_s in mdp.pre_s[s]  ]
                )
            else:
                mu[mdp.s2idx[s], t+1] = sum(
                    [
                        sum(
                            [  mu[mdp.s2idx[pre_s], t] * mdp.get_trans_prob(pre_s, a, s) * policy[pre_s][a] for a in mdp.s_a[pre_s] ]
                        )
                        for pre_s in mdp.pre_s[s]
                    ]
                )
    p = np.sum(mu, 1)
    return p

def maxent_irl(feat_map, mdp, gamma, trajs, theta, rbg, lr, deterministic=False):
    """
    Maximum Entropy Inverse RL

    inputs:
        feat_map: feature for each state, here use identity matrix
        mdp: ice hockey mdp, fom mkv.markovGame.MarkovGame
        gamma: discount factor
        trajs: demonstrations extracted from play by play data. output of "extract_demonstrations(file)" function
        theta: parameters of reward function
        lr: learning rate
    return:
        theta_new
        reward_new
    """

    # calculate state expectation from demo
    state_exp = compute_state_expectation(feat_map, mdp, trajs)
    # compute reward
    reward = np.dot(feat_map, theta)
    # compute policy
    _, policy = value_iteration(mdp, reward, gamma, error=0.01, deterministic=deterministic)
    # compute state visit frequency
    svf = compute_state_visit_freq(feat_map, mdp, trajs, policy, deterministic=deterministic)
    # compute panelty
    gp = 2 * np.exp(-1 * np.power(reward - rbg, 2) / 2)
    gp = gp * (reward - rbg)
    # compute grad
    grad = state_exp - feat_map.T.dot(svf) - gp
    #print("sum grad: ", sum(grad))
    # update params
    theta_new = theta + lr * grad
    # compute new reward
    reward_new = np.dot(feat_map, theta_new)

    return theta_new, normalize_range(reward_new, 0, 2)

