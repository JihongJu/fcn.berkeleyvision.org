import numpy as np
from PIL import Image
from voc_helper import VOC

import caffe

# init voc helper
helper = VOC('../data/pascal-obfuscated/VOC2011')

# load net
net = caffe.Net('deploy.prototxt', 'fcn8s-heavy-pascal.caffemodel', caffe.TEST)

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
images = ['2007_003742', '2007_000033', '2007_000061',
        '2007_000129', '2007_000323', '2007_000332']
for f in images:
    im = Image.open('../data/pascal-obfuscated/VOC2011/JPEGImages/{}.jpg'.format(f))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)
    # save the prediction
    #out = (out - np.min(out)) * 255 / np.max(out)
    #result = Image.fromarray(out.astype(np.uint8))
    result = helper.transfer(out.astype(np.uint8))
    result.save('{}_out.png'.format(f))
    # image
    im.save('{}.png'.format(f))
