"""
This script converts results of task 3 into our evaluation format.
"""
import operator
import numpy as np

# conversion #1: GatorSense
teams = ['GatorSense', 'BRG', 'StanfordCCB', 'FEM']
for t in teams:
    path_to_sub = '../income/%s/species_id_subm_ori.csv' % t
    with open(path_to_sub, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        lines = [l.split(',') for l in lines]
        target_id = list(set([l[1] for l in lines]))
        out = []
        out.append('CLS,'+','.join(target_id))
        id2index = {tid:k for k,tid in enumerate(target_id)}
        lines = sorted(lines, key=operator.itemgetter(0))
        cur_id = lines[0][0]
        vec = np.zeros((len(target_id)))
        for crownid, cls, prob in lines:
            if crownid != cur_id:
                out.append(cur_id + ',' + ','.join(['%.6f' % p for p in vec]))
                vec = np.zeros((len(target_id)))
                cur_id = crownid
            idx = id2index[cls]
            prob = float(prob)
            vec[idx] = prob
    with open('../income/%s/species_id_subm.csv' % t, 'wb+') as fp:
        fp.write('\n'.join(out))


# conversion #1: Conor
teams = ['Conor']
for t in teams:
    path_to_sub = '../income/%s/species_id_subm_ori.csv' % t
    with open(path_to_sub, 'r') as fp:
        lines = fp.read().strip().split('\n')
        lines = [l.split(',') for l in lines]
        target_id = list(set([l[1] for l in lines]))
        out = []
        out.append('CLS,'+','.join(target_id))
        id2index = {tid:k for k,tid in enumerate(target_id)}
        lines = sorted(lines, key=operator.itemgetter(0))
        cur_id = lines[0][0]
        vec = np.zeros((len(target_id)))
        for crownid, cls, prob in lines:
            if crownid != cur_id:
                out.append(cur_id + ',' + ','.join(['%.6f' % p for p in vec]))
                vec = np.zeros((len(target_id)))
                cur_id = crownid
            idx = id2index[cls]
            prob = float(prob)
            vec[idx] = prob
    with open('../income/%s/species_id_subm.csv' % t, 'wb+') as fp:
        fp.write('\n'.join(out))



