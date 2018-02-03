"""
This script converts results of task 3 into our evaluation format.
"""
import operator
import numpy as np

# conversion #1: Conor
teams = ['FEM']
for t in teams:
    path_to_sub = '../income/%s/itc_ground_relation_test.csv' % t
    with open(path_to_sub, 'r') as fp:
        lines = fp.read().strip().split('\n')[1:]
        lines = [l.split(',') for l in lines]
        target_id = list(set([l[0] for l in lines]))
        out = []
        out.append('ALIGN,'+','.join(target_id))
        id2index = {tid:k for k,tid in enumerate(target_id)}
        for crownid, stemid in lines:
            vec = np.zeros((len(target_id)))
            idx = id2index[crownid]
            vec[idx] = 1.0
            out.append(stemid + ',' + ','.join(['%.6f' % p for p in vec]))
    with open('../income/%s/alignment_out.csv' % t, 'wb+') as fp:
        fp.write('\n'.join(out))

