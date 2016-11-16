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
snapshot_prefix = 'obfuscated_training_with_vgg16_snapshot'
solverstates = [os.path.join(snapshot_prefix, ss)
        for ss in os.listdir(snapshot_prefix)
        if ss.endswith('.solverstate')]

# scoring
train = np.loadtxt('../data/pascal-obfuscated/VOC2011/ImageSets/Segmentation/train.txt', dtype=str)
val = np.loadtxt('../data/pascal-obfuscated/seg11valid.txt', dtype=str)

for ss in solverstates:
    solver.restore(ss)
    score.seg_tests(solver, False, train, layer='score')
    score.seg_tests(solver, False, val, layer='score')
