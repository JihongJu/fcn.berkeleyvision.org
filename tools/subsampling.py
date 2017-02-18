import os
import numpy as np
from PIL import Image

def subsample_image(srcdir, tgtdir, rate=2):
    """
    Subsample image
    """
    for f in os.listdir(srcdir):
        print("Subsampling {}...").format(f)
        im = Image.open(os.path.join(srcdir, f))
        in_ = np.array(im)
        out_ = in_[::rate,::rate,:]
        im = Image.fromarray(out_)
        im.save(os.path.join(tgtdir, f))

def subsample_label(srcdir, tgtdir, rate=2):
    """
    Subsample image
    """
    for f in os.listdir(srcdir)[:]:
        print("Subsampling {}...").format(f)
        im = Image.open(os.path.join(srcdir, f))
        p = im.getpalette()
        in_ = np.array(im)
        out_ = in_[::rate,::rate]
        im = Image.fromarray(out_.astype(np.uint8))
        im.convert("P")
        im.putpalette(p)
        im.save(os.path.join(tgtdir, f))

srcimg_dir = '../data/pascal/VOC2011/JPEGImages/'
tgtimg_dir = '../data/pascal-subsampl/VOC2011/JPEGImages/'
subsample_image(srcimg_dir, tgtimg_dir, rate=4)

srclabel_dir = '../data/pascal/VOC2011/SegmentationBinary/'
tgtlabel_dir = '../data/pascal-subsampl/VOC2011/SegmentationBinary/'
subsample_label(srclabel_dir, tgtlabel_dir, rate=4)

srclabel_dir = '../data/pascal/VOC2011/SegmentationClass/'
tgtlabel_dir = '../data/pascal-subsampl/VOC2011/SegmentationClass/'
subsample_label(srclabel_dir, tgtlabel_dir, rate=4)

srclabel_dir = '../data/pascal/VOC2011/SegmentationCategory/'
tgtlabel_dir = '../data/pascal-subsampl/VOC2011/SegmentationCategory/'
subsample_label(srclabel_dir, tgtlabel_dir, rate=4)
