import caffe
import surgery, score

import numpy as np
import os
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

solverstate = 'snapshot/train_iter_20000.solverstate'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.restore(solverstate)

# scoring
val = np.loadtxt('../data/pascal-obfuscated/seg11valid.txt', dtype=str)

for _ in range(75):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
