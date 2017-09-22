import numpy as np
from PIL import Image
import os, sys
import tensorflow as tf

out_size = 256
dataset_dir = "subflower2daisy"
path = './datasets/{}/'.format(dataset_dir + '/trainB')
# path = "./horse2zebra/trainB/"
dirs = os.listdir(path)
for img_name in dirs:
    img = Image.open(path + img_name)
    width, height = img.size

    if height > width:
        crop_size = width
    else:
        crop_size = height
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2

    crop_img = img.crop((left, top, right, bottom))
    out_img = crop_img.resize((out_size, out_size), Image.ANTIALIAS)
    out_img.save(path + img_name)
    out_width, out_height = out_img.size
    print("width %d" % (out_width))
    print("height %d" % out_height)
