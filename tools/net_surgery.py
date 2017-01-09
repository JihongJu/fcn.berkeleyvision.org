import caffe
import surgery
import os
import argparse

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sp', help='source prototxt')
parser.add_argument('--tp', help='target prototxt')
parser.add_argument('--sc', help='sorce caffemodel')
parser.add_argument('--tc', help='target caffemodel')
args = parser.parse_args()
source_prototxt = args['--sp']
target_prototxt = args['--tp']
source_caffemodel = args['--sc']
target_caffemodel = args['--tc']

# Load the original network and extract the fully connected layers' parameters.
base_net = caffe.Net(source_prototxt,
                source_caffemodel,
                caffe.TEST)
print "Base net: {}".format(base_net.params)

# n_cl = 21
new_net = caffe.Net(target_prototxt,
                caffe.TEST)
print "New net: {}".format(new_net.params)

# transplant
surgery.transplant(new_net, base_net)
new_net.save(target_caffemodel)
