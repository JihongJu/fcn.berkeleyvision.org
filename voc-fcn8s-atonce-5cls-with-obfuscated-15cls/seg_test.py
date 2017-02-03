import caffe
import score

import numpy as np
import os
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')

# solverstates
snapshot_prefix = 'snapshot'
#solverstates = [os.path.join(snapshot_prefix, ss)
#        for ss in os.listdir(snapshot_prefix)
#        if ss.endswith('.solverstate')]
#solverstates.sort()
solverstates = ['train_iter_4000.solverstate',
                'train_iter_8000.solverstate',
                'train_iter_12000.solverstate',
                'train_iter_16000.solverstate',
                'train_iter_20000.solverstate',
                'train_iter_24000.solverstate',
                'train_iter_28000.solverstate',
                'train_iter_32000.solverstate',
                'train_iter_36000.solverstate',
                'train_iter_40000.solverstate',
                'train_iter_44000.solverstate',
                'train_iter_48000.solverstate',
                'train_iter_52000.solverstate',
                'train_iter_56000.solverstate',
                'train_iter_60000.solverstate',
                'train_iter_64000.solverstate',
                'train_iter_68000.solverstate',
                'train_iter_72000.solverstate',
                'train_iter_76000.solverstate',
                'train_iter_80000.solverstate',
                'train_iter_84000.solverstate',
                'train_iter_88000.solverstate',
                'train_iter_92000.solverstate',
                'train_iter_96000.solverstate',
                'train_iter_100000.solverstate',
                'train_iter_104000.solverstate',
                'train_iter_108000.solverstate',
                'train_iter_112000.solverstate',
                'train_iter_116000.solverstate',
                'train_iter_120000.solverstate',
                ]
solverstates = [os.path.join(snapshot_prefix, s)
        for s in solverstates]

# scoring
#train = np.loadtxt('../data/pascal-obfuscated/VOC2011/ImageSets/Segmentation/train.txt', dtype=str)
val = np.loadtxt('../data/pascal-obfuscated/seg11valid.txt', dtype=str)

for ss in solverstates:
    solver.restore(ss)
    #score.seg_trains(solver, False, train, layer='score')
    score.seg_tests(solver, False, val, layer='score')
