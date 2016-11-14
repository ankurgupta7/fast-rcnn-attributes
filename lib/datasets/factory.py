# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}


from datasets.attributes import attributes
import numpy as np

# Setup face detector
attributes_devkit_path = '/home/agupta82/fast-rcnn/data/attributes'
for split in ['train', 'test']:
    name = '{}_{}'.format('attributes', split)
    __sets[name] = (lambda split=split: attributes(split, attributes_devkit_path))    



def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
