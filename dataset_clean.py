import numpy as np
from PIL import Image
import os, sys
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

dataset_dir = "sunflower2daisy"
path = './datasets/{}/'.format(dataset_dir + '/trainB')

dirs = os.listdir(path)

img_format = ['.jpg', '.jpeg', '.png']
for img_name in dirs:
    # select the file format
    if os.path.splitext(img_name)[1] not in img_format:
        os.remove(path + img_name)
        print path + img_name
    else:
        try:
            fp = open(path + img_name, 'rb')
            img = np.array(Image.open(fp))
            imglen = len(img.shape)

            if imglen == 2:
                # delete the gray file
                os.remove(path + img_name)
                print path + img_name

        except:
            # delete the file that can not open
            fp.close()
            os.remove(path + img_name)
        else:
            continue


