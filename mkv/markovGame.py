import os
import csv

def check_csv_seq(csv_dir, f):
    gt_list = []
    with open(csv_dir+'/'+f, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            gameTime = row['gameTime']
            gt_list.append(gameTime)
    inc = all(float(x)<=float(y) for x, y in zip(gt_list, gt_list[1:]))
    if not inc:
        raise Exception('This csv file is not in gameTime increasing order:', f)

def location(x, y):
    """
    Divide rink into six block
    """
    ###########
    # 1 |2| 3 #
    #---|+|---#
    # 4 |5| 6 #
    ###########
    if x < -25 and y > 0:
        return str(1)
    elif x >= -25 and x < 25 and y > 0:
        return str(2)
    elif x >= 25 and y > 0:
        return str(3)
    elif x < -25 and y <= 0:
        return str(4)
    elif x >= -25 and x < 25 and y <= 0:
        return str(5)
    elif x >= 25 and y <= 0:
        return str(6)

acts = ['assist', 'block', 'carry', 'check', 'controlledbreakout','controlledentryagainst',
                    'dumpin','dumpinagainst','dumpout','faceoff','goal','icing','lpr','offside','pass',
                    'pass1timer','penalty','pressure','pscarry','pscheck','pslpr','pspuckprotection',
                    'puckprotection','reception','receptionprevention','shot','shot1timer',
                    'socarry','socheck','sogoal','solpr','sopuckprotection','soshot']

class MarkovGame(object):

    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.acts = acts
        self.trans = self._build_transition(csv_dir)
        self._decomposition()

    def _build_transition(self, csv_dir):
        print('###### Building Markov Game Model from data ######')
        """
        trans is a dict(), key is the current state, value is a dict(),
              where key is action + next_state, value is the occurrence number in our data
        """
        trans = {}

        def insert2dict(s, a, nx_s):
            if s in trans:
                to_dict = trans[s]
                key = a + '+' + nx_s
                if key in to_dict:
                    to_dict[key] = to_dict[key] + 1
                else:
                    to_dict[key] = 1
            else:
                to_dict = {}
                key = a + '+' + nx_s
                to_dict[key] = 1
                trans[s] = to_dict

        for f in os.listdir(csv_dir):
            # first check if data in gameTime increasing order
            check_csv_seq(csv_dir, f) 

            pre_s, pre_a = '', ''
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

                    #print(type(act), type(goalDiff), type(manPower), type(period), type(HorA))
                    s = goalDiff + ',' + manPower + ',' + period + ',' + loc + ',' + HorA
                    a = str(self.acts.index(act))
                    
                    if pre_s == '' and pre_a == '':
                        pre_s = s
                        pre_a = a
                    elif pre_a == str(self.acts.index('goal')):
                        """
                        Add goal state after 'goal' action for convenience
                        """
                        if pre_s[-1] == 'H':
                            insert2dict(pre_s, pre_a, '*,*,*,*,H')
                        if pre_s[-1] == 'A':
                            insert2dict(pre_s, pre_a, '*,*,*,*,A')
                        pre_s = s
                        pre_a = a
                    else:
                        insert2dict(pre_s, pre_a, s)
                        pre_s = s
                        pre_a = a
        return trans

    def _decomposition(self):
        print('######              decomposing             ######')
        pre_s        = {} # a dict: {state : [list of previous state]}
        s_a          = {} # a dict: {state : [list of actions]}
        s_a_freq     = {} # a ditt: {state,action : frequency}
        s_a_nxs      = {} # a dict: {state,action : [list of next state]}
        s_a_nxs_freq = {} # a dict: {state,action,nx state : frequency}

        for s in self.trans.keys():
            to_dict = self.trans[s]
            to_keys = to_dict.keys()
            for to_key in to_keys:
                a, nxs = to_key.split('+')
                num    = to_dict[to_key]

                # update pre_s
                if nxs in pre_s:
                    if s not in pre_s[nxs]:
                        pre_s[nxs].append(s)
                else:
                    pre_s[nxs] = [s]

                # update s_a
                if s in s_a:
                    action_list = s_a[s]
                    if a not in action_list:
                        s_a[s].append(a)
                else:
                    s_a[s] = [a]

                # update s_a_freq
                s_and_a = s + '+' + a
                if s_and_a in s_a_freq:
                    s_a_freq[s_and_a] += num
                else:
                    s_a_freq[s_and_a]  = num

                # update s_a_nxs
                s_and_a = s + '+' + a
                if s_and_a in s_a_nxs:
                    next_state_list = s_a_nxs[s_and_a]
                    if nxs not in next_state_list:
                        s_a_nxs[s_and_a].append(nxs)
                else:
                    s_a_nxs[s_and_a] = [nxs]

                # update s_a_nxs_freq
                s_and_a_and_nxs = s + '+' + a + '+' + nxs
                if s_and_a_and_nxs in s_a_nxs_freq:
                    s_a_nxs_freq[s_and_a_and_nxs] += num
                else:
                    s_a_nxs_freq[s_and_a_and_nxs]  = num

        tmp = [s for s in self.trans.keys()]
        tmp.append('*,*,*,*,H')
        tmp.append('*,*,*,*,A')

        self.s            = tmp
        self.end_s        = ['*,*,*,*,H','*,*,*,*,A']
        self.s2idx        = {tmp[i]:i for i in range(len(tmp))}
        self.pre_s        = pre_s
        self.s_a          = s_a
        self.s_a_freq     = s_a_freq
        self.s_a_nxs      = s_a_nxs
        self.s_a_nxs_freq = s_a_nxs_freq  

    def _get_nxs_and_prob(self, s, a):
        """
        get all the (next state, probablity) pair
        if taking action (a) at state (s)
        """
        k1 = '%s+%s'%(s, a)
        freq     = self.s_a_freq[k1]
        nxs_list = self.s_a_nxs[k1]
        nxs_and_prob = []

        for nxs in nxs_list:
            k2 = '%s+%s+%s'%(s, a, nxs)
            this_freq = self.s_a_nxs_freq[k2]
            this_prob = float(this_freq) / float(freq)
            nxs_and_prob.append([nxs, this_prob])
        
        return nxs_and_prob

    def get_trans_prob(self, s, a, nx_s):
        """
        The transition probability of landing at next state
        when taking action (a) at state (s)
        """
        if s == '*,*,*,*,H' or s == '*,*,*,*,A':
            return 0
        if a not in self.s_a[s]:
            return 0

        nxs_and_prob = self._get_nxs_and_prob(s, a)
        for nxs, prob in nxs_and_prob:
            if nxs == nx_s:
                return prob

        return 0

    def get_act(self, s):
        return self.s_a[s]
    
    def get_nxs(self, s, a):
        key = '%s+%s'%(s, a)
        return self.s_a_nxs[key]
                
if __name__ == '__main__':
    check_csv_seq('/home/yudong/Documents/Slgq/data','5665.csv')
    mg = MarkovGame('/home/yudong/Documents/Slgq/data')