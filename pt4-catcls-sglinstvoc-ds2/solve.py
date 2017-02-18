import caffe
import surgery, score

import numpy as np
import os
import sys

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/pascal-subsampl/VOC2011/ImageSets/Segmentation/seg11valid_pt4_sglinst.txt', dtype=str)

for _ in range(75):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
