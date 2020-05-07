import csv
from mkv.markovGame import location
from mkv.markovGame import acts

def get_events(csv_dir, f):
    events = []
    with open(csv_dir+'/'+f, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            act      = row['act']
            goalDiff = row['goalDiff']
            manPower = row['manPower']
            period   = row['period']
            xCoord   = float(row['xCoord'])
            yCoord   = float(row['yCoord'])
            HorA     = row['H/A']
            loc      = location(xCoord, yCoord)

            s = goalDiff + ',' + manPower + ',' + period + ',' + loc + ',' + HorA
            a = str(acts.index(act))
            events.append((s,a))
    return events

def curr_s_a(events, idx):
    """
    Get current state(goalDiff, manPower, period, loc, HorA) and action(act)
    """
    (s, a) = events[idx]
    return s, a

def next_s(events, idx):
    """
    For convenience, not add an end state.
    Thus the range to use this function is from 0 to idx-2
    """
    (s,a) = events[idx]
    if a == str(acts.index('goal')): # score a goal
        h_w = s[-1]
        return '*,*,*,*,'+h_w

    # next state idx+1
    (nx_s, nx_a) = events[idx+1]
    return nx_s

def extract_demonstrations(csv_dir, f):
    """
    extract demostrations from play by play data
    goal is the end signal of an episode
    """
    events = get_events(csv_dir, f)
    trajs = []
    episode = []
    for idx in range(len(events)):
        s, a = curr_s_a(events, idx)
        episode.append(s)
        if a == str(acts.index('goal')):
            nx_s = next_s(events, idx)
            episode.append(nx_s)
            trajs.append(episode)

            episode = []
        else:
            continue
    if episode != []:
        trajs.append(episode)

    trajs_select = [episode for episode in trajs if len(episode)>150]
    # it is optinal if you want to make trajs shorter
    trajs_clip = [episode[-150:] for episode in trajs_select]

    return trajs_clip

def test_extract_demonstrations():
    d = {}
    csv_dir = '/home/yudong/Documents/Slgq/data'
    file_all = os.listdir(csv_dir)
    for file in file_all:
        trajs = extract_demonstrations(csv_dir, file)
        episode_1 = [episode[0] for episode in  trajs]
        for e in episode_1:
            goal_diff, _,_,_,_ = e.split(',')
            if goal_diff in d:
                d[goal_diff] += 1
            else:
                d[goal_diff] = 1
    print(d)

if __name__ == '__main__':
    import os
    test_extract_demonstrations()