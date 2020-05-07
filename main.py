import os
import pickle
import numpy as np
from mkv.markovGame import MarkovGame
from irl.maxent_irl import maxent_irl
from utils.extract import extract_demonstrations

def run(csv_dir, mg, HorA, save_dir, deter):
    # build feature matrix
    N_STATES = len(mg.s)
    feat_map = np.eye(N_STATES)
    # domain konwledge reward
    rbg = [1. for s in mg.s]
    if HorA == 'Home':
        rbg[mg.s2idx[mg.end_s[0]]] = 2.
        rbg[mg.s2idx[mg.end_s[1]]] = 0.
    if HorA == 'Away':
        rbg[mg.s2idx[mg.end_s[0]]] = 0.
        rbg[mg.s2idx[mg.end_s[1]]] = 2.

    theta = rbg.copy()
    gamma = 0.9
    lr = 0.001
    n_iters = 5
    if not os.path.exists(save_dir+'/'+HorA):
        os.mkdir(save_dir+'/'+HorA)
    # train
    file_all = os.listdir(csv_dir)
    for iter in range(n_iters):
        for i in range(len(file_all)):
            print("#### Game ", str(i+1), " out of ", str(len(file_all)), " | iter ", str(iter+1), " ####")
            trajs = extract_demonstrations(csv_dir, file_all[i])
            if trajs == []:
                continue
            theta, reward = maxent_irl(feat_map, mg, gamma, trajs, theta, rbg, lr, deter)
            # save
            save_theta = save_dir + '/' + HorA + '/' + 'iter_' + str(iter) + '_aft_game_' + str(i) + '_theta'
            save_reward = save_dir + '/' + HorA + '/'+ 'iter_' + str(iter) + '_aft_game_' + str(i) + '_reward'
            with open(save_theta, 'wb') as f:
                pickle.dump(theta, f)
            with open(save_reward, 'wb') as f:
                pickle.dump(reward, f)
        
if __name__ == '__main__':
    csv_dir = '/home/yudong/Documents/Slgq/data'
    save_dir = '/home/yudong/Documents/Slgq/save_reward'
    mg = MarkovGame(csv_dir)
    run(csv_dir, mg, 'Home', save_dir, True)
    run(csv_dir, mg, 'Away', save_dir, True)