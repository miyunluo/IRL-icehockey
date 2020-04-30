import numpy as np

def value_iteration(mdp, reward, gamma, error, deterministic=True):
    """
    do value iteration for mdp using dynamic programming

    args:
        mdp   : treat Markov Game as a partial mdp
        reward: list, reward[i] is the reward for i-th state
        gamma : discount factor
        error : threshold for a stop
    return
        values: {state: value}
        policy: {state: action} or {state: {action and prob}} 
    """

    values = {}
    policy = {}
    for s in mdp.s:
        values[s] = 0

    while True:
        values_tmp = values.copy()
        for s in mdp.s:
            if s in mdp.end_s:
                values[s] = reward[mdp.s2idx[s]]
                continue
            
            values[s] = max(
                [
                    sum(
                        [mdp.get_trans_prob(s, a, nxs)*(reward[mdp.s2idx[s]]+gamma*values_tmp[nxs]) for nxs in mdp.get_nxs(s, a)]
                    )
                    for a in mdp.get_act(s)
                ]
            )
        if max([abs(values[s]-values_tmp[s]) for s in mdp.s]) < error:
            break

    if deterministic:
        # generate deterministic policy
        for s in mdp.s:
            if s in mdp.end_s:
                continue

            act_idx = np.argmax(
                [
                    sum(
                        [mdp.get_trans_prob(s, a, nxs)*(reward[mdp.s2idx[s]]+gamma*values[nxs]) for nxs in mdp.get_nxs(s, a)]
                    )
                    for a in mdp.get_act(s)
                ]
            )
            policy[s] = mdp.get_act(s)[act_idx]

    else:
        # generate stochastic policy
        for s in mdp.s:
            if s in mdp.end_s:
                continue

            v_s = np.array(
                [
                    sum(
                        [mdp.get_trans_prob(s, a, nxs)*(reward[mdp.s2idx[s]]+gamma*values[nxs]) for nxs in mdp.get_nxs(s, a)]
                    )
                    for a in mdp.get_act(s)
                ]
            )
            prob = v_s / np.sum(v_s)
            policy[st] = { mdp.get_act(s)[i] : prob[i] for i in range(len(prob)) }

    return values, policy