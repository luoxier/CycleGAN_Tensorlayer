import os
import pprint
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from random import shuffle
# from model_upsampling import *
from model_deconv import *

import argparse
from collections import namedtuple

pp = pprint.PrettyPrinter()
flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [100]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("weight_decay", 1e-5, "Weight decay for l2 loss")
flags.DEFINE_float("pool_size", 50, 'size of image buffer that stores previously generated images, default: 50')
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 1, "The number of batch images [1] if we use InstanceNormLayer !")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [256]")
flags.DEFINE_integer("gf_dim", 32, "Size of generator filters in first layer")
flags.DEFINE_integer("df_dim", 64, "Size of discriminator filters in first layer")
# flags.DEFINE_integer("class_embedding_size", 5, "Size of class embedding")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 200, "The interval of saveing checkpoints. [200]")
flags.DEFINE_string("dataset_dir", "horse2zebra", "The name of dataset [horse2zebra, apple2orange, sunflower2daisy and etc]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("direction", "forward", "The direction of generator [forward, backward]")
flags.DEFINE_string("test_dir", "./test", "The direction of test")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
# flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

sess = tf.Session()

ni = int(np.sqrt(FLAGS.batch_size))

def train_cyclegan():
    lamda = 10
    # num_fake = 0

    ni = int(np.sqrt(FLAGS.batch_size))
    h, w = 256, 256

    ## data augmentation
    def prepro(x):
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        x = tl.prepro.rotation(x, rg=16, is_random=True, fill_mode='nearest')
        x = tl.prepro.imresize(x, size=[int(h * 1.2), int(w * 1.2)], interp='bicubic', mode=None)
        x = tl.prepro.crop(x, wrg=h, hrg=w, is_random=True)
        x = x / (255. / 2.)
        x = x - 1.
        return x

    def rescale(x):
        x = x / (255. / 2.)
        x = x - 1.
        return x

    real_A = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='real_A')
    real_B = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='real_B')
    fake_A_pool = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='fake_A')
    fake_B_pool = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='fake_B')

    gen_B, gen_B_out = cyclegan_generator_resnet(real_A, 9, is_train=True, reuse=False, name='gen_A2B')
    gen_A, gen_A_out = cyclegan_generator_resnet(real_B, 9, is_train=True, reuse=False, name='gen_B2A')
    cyc_B, cyc_B_out = cyclegan_generator_resnet(gen_A_out, 9, is_train=True, reuse=True, name='gen_A2B')
    cyc_A, cyc_A_out = cyclegan_generator_resnet(gen_B_out, 9, is_train=True, reuse=True, name='gen_B2A')

    d_real_A, d_real_A_logits = cyclegan_discriminator_patch(real_A, is_train=True, reuse=False, name='dis_A')  # dx
    d_real_B, d_real_B_logits = cyclegan_discriminator_patch(real_B, is_train=True, reuse=False, name='dis_B')  # dy
    d_fake_A, d_fake_A_logits = cyclegan_discriminator_patch(gen_A_out, is_train=True, reuse=True, name='dis_A')  # d_fy
    d_fake_B, d_fake_B_logits = cyclegan_discriminator_patch(gen_B_out, is_train=True, reuse=True, name='dis_B')  # d_gx

    d_A_pool, d_A_pool_logits = cyclegan_discriminator_patch(fake_A_pool, is_train=True, reuse=True, name='dis_A')  # d_fakex
    d_B_pool, d_B_pool_logits = cyclegan_discriminator_patch(fake_B_pool, is_train=True, reuse=True, name='dis_B')  # d_fakey

        ## test inference
        # gen_B_test, gen_B_test_logits = cyclegan_generator_resnet(real_A, 9, is_train=False, reuse=True, name='gen_A2B')
        # gen_A_test, gen_A_test_logits = cyclegan_generator_resnet(real_B, 9, is_train=False, reuse=True, name='gen_B2A')

    ## calculate cycle loss
    cyc_loss = tf.reduce_mean(tf.abs(cyc_A_out - real_A)) + tf.reduce_mean(tf.abs(cyc_B_out - real_B))
        # cyc_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(cyc_A - real_A), [1, 2, 3])) + tf.reduce_mean(
        #     tf.reduce_mean(tf.abs(cyc_B - real_B), [1, 2, 3]))

    ## calculate adversial loss
    g_loss_A2B = tf.reduce_mean(tf.squared_difference(d_fake_B_logits, tf.ones_like(d_fake_B_logits)), name='g_loss_b')
        # g_loss_A2B = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(d_fake_B, tf.ones_like(d_fake_B)), [1, 2, 3]), name='g_loss_b')

    g_loss_B2A = tf.reduce_mean(tf.squared_difference(d_fake_A_logits, tf.ones_like(d_fake_A_logits)),name='g_loss_a')
        # g_loss_B2A = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(d_fake_A, tf.ones_like(d_fake_A)), [1, 2, 3]), name='g_loss_a')

    ## calculate totalloss of generator
    g_a2b_loss = lamda * cyc_loss + g_loss_A2B  # forward
    g_b2a_loss = lamda * cyc_loss + g_loss_B2A  # backward

    ## calculate discriminator loss
    d_a_loss = (tf.reduce_mean(tf.squared_difference(d_real_A_logits, tf.ones_like(d_real_A_logits))) + tf.reduce_mean(tf.square(d_fake_A_logits))) / 2.0
        # d_a_loss = (tf.reduce_mean(
        #     tf.reduce_mean(tf.squared_difference(d_real_A, tf.ones_like(d_real_A)), [1, 2, 3])) + tf.reduce_mean(
        #     tf.reduce_mean(tf.square(d_fake_A), [1, 2, 3]))) / 2.0
    d_b_loss = (tf.reduce_mean(tf.squared_difference(d_real_B_logits, tf.ones_like(d_real_B_logits))) + tf.reduce_mean(tf.square(d_fake_B_logits))) / 2.0
        # d_b_loss = (tf.reduce_mean(
        #     tf.reduce_mean(tf.squared_difference(d_real_B, tf.ones_like(d_real_B)), [1, 2, 3])) + tf.reduce_mean(
        #     tf.reduce_mean(tf.square(d_fake_B), [1, 2, 3]))) / 2.0

        # t_vars = tf.trainable_variables()

    g_A2B_vars = tl.layers.get_variables_with_name('gen_A2B', True, True)
    g_B2A_vars = tl.layers.get_variables_with_name('gen_B2A', True, True)
    d_A_vars = tl.layers.get_variables_with_name('dis_A', True, True)
    d_B_vars = tl.layers.get_variables_with_name('dis_B', True, True)

    # with tf.device('/gpu:0'):
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(FLAGS.learning_rate, trainable=False)
    g_a2b_optim = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(g_a2b_loss, var_list=g_A2B_vars)
    g_b2a_optim = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(g_b2a_loss, var_list=g_B2A_vars)
    d_a_optim = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(d_a_loss, var_list=d_A_vars)
    d_b_optim = tf.train.AdamOptimizer(lr_v, beta1=FLAGS.beta1).minimize(d_b_loss, var_list=d_B_vars)

    ## init params
    tl.layers.initialize_global_variables(sess)

    net_g_A2B_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g_A2B.npz'.format(FLAGS.dataset_dir))
    net_g_B2A_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g_B2A.npz'.format(FLAGS.dataset_dir))
    net_d_A_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d_A.npz'.format(FLAGS.dataset_dir))
    net_d_B_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_d_B.npz'.format(FLAGS.dataset_dir))

    tl.files.load_and_assign_npz(sess=sess, name=net_g_A2B_name, network=gen_B)
    tl.files.load_and_assign_npz(sess=sess, name=net_g_B2A_name, network=gen_A)
    tl.files.load_and_assign_npz(sess=sess, name=net_d_A_name, network=d_fake_A)
    tl.files.load_and_assign_npz(sess=sess, name=net_d_B_name, network=d_fake_B)

    ##========================= TRAIN MODELS ================================##
    iter_counter = 1
    start_time = time.time()

    dataA, dataB, im_test_A, im_test_B = tl.files.load_cyclegan_dataset(filename=FLAGS.dataset_dir, path='datasets')

    sample_A = np.asarray(dataA[0: 16])
    sample_B = np.asarray(dataB[0: 16])
    sample_A = tl.prepro.threading_data(sample_A, fn=rescale)
    sample_B = tl.prepro.threading_data(sample_B, fn=rescale)

    tl.vis.save_images(sample_A, [4, 4], './{}/sample_A.jpg'.format(FLAGS.sample_dir))
    tl.vis.save_images(sample_B, [4, 4], './{}/sample_B.jpg'.format(FLAGS.sample_dir))

    shuffle(dataA)
    shuffle(dataB)

    for epoch in range(FLAGS.epoch):
        ## change learning rate
        if epoch >= 100:
            new_lr = FLAGS.learning_rate - FLAGS.learning_rate * (epoch - 100) / 100
            sess.run(tf.assign(lr_v, new_lr))
            print("New learning rate %f" % new_lr)

        batch_idxs = min(min(len(dataA), len(dataB)), FLAGS.train_size) // FLAGS.batch_size
        for idx in range(0, batch_idxs):
            batch_imgA = tl.prepro.threading_data(dataA[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size], fn=prepro)
            batch_imgB = tl.prepro.threading_data(dataB[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size], fn=prepro)

            gen_A_temp_out, gen_B_temp_out = sess.run([gen_A_out, gen_B_out],
                                feed_dict={real_A: batch_imgA, real_B: batch_imgB})

            ## update forward network
            _, errGA2B = sess.run([g_a2b_optim, g_a2b_loss],
                                feed_dict={real_A: batch_imgA, real_B: batch_imgB})
            ## update DB network
            _, errDB = sess.run([d_b_optim, d_b_loss],
                                feed_dict={real_A: batch_imgA, real_B: batch_imgB, fake_B_pool: gen_B_temp_out})
            ## update (backword) network
            _, errGB2A = sess.run([g_b2a_optim, g_b2a_loss],
                                feed_dict={real_A: batch_imgA, real_B: batch_imgB})
            ## update DA network
            _, errDA = sess.run([d_a_optim, d_a_loss],
                                feed_dict={real_A: batch_imgA, real_B: batch_imgB, fake_A_pool: gen_A_temp_out})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_a_loss: %.8f, d_b_loss: %.8f, g_a2b_loss: %.8f, g_b2a_loss: %.8f" \
                % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time, errDA, errDB, errGA2B, errGB2A))

            iter_counter += 1
            # num_fake += 1

            if np.mod(iter_counter, 500) == 0:
                oA, oB = sess.run([gen_A_out, gen_B_out],
                                feed_dict={real_A: sample_A, real_B: sample_B})
                tl.vis.save_images(oA, [4, 4],
                                   './{}/B2A_{:02d}_{:04d}.jpg'.format(FLAGS.sample_dir, epoch, idx))
                print("save image gen_A, Epoch: %2d idx:%4d" % (epoch, idx))
                tl.vis.save_images(oB, [4, 4],
                                   './{}/A2B_{:02d}_{:04d}.jpg'.format(FLAGS.sample_dir, epoch, idx))
                print("save image gen_B, Epoch: %2d idx:%4d" % (epoch, idx))

        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(gen_B.all_params, name=net_g_A2B_name, sess=sess)
            tl.files.save_npz(gen_A.all_params, name=net_g_B2A_name, sess=sess)
            tl.files.save_npz(d_fake_A.all_params, name=net_d_A_name, sess=sess)
            tl.files.save_npz(d_fake_B.all_params, name=net_d_B_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")


def test_cyclegan():
    """Test cyclegan"""

    def pro(x):
        x = x / (255. / 2.)
        x = x - 1.
        return x

    test_A = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='test_x')
    test_B = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim],
                            name='test_y')
    # testB = cyclegan_generator_resnet(test_A, options, True, name="gen_forward")
    # testA = cyclegan_generator_resnet(test_B, options, True, name="gen_backward")
    test_gen_A2B, test_gen_A2B_logits = cyclegan_generator_resnet(test_A, 9, is_train=False, reuse=False, name='gen_A2B')
    test_gen_B2A, test_gen_B2A_logits = cyclegan_generator_resnet(test_B, 9, is_train=False, reuse=False, name='gen_B2A')

    out_var, in_var = (test_B, test_A) if FLAGS.direction == 'forward' else (test_A, test_B)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    net_g_A2B_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g_A2B.npz'.format(FLAGS.dataset_dir))
    net_g_B2A_name = os.path.join(FLAGS.checkpoint_dir, '{}_net_g_B2A.npz'.format(FLAGS.dataset_dir))
    tl.files.load_and_assign_npz(sess=sess, name=net_g_A2B_name, network=test_gen_A2B)
    tl.files.load_and_assign_npz(sess=sess, name=net_g_B2A_name, network=test_gen_B2A)

    dataA, dataB, im_test_A, im_test_B = tl.files.load_cyclegan_dataset(filename=FLAGS.dataset_dir, path='datasets')

    if FLAGS.direction == 'forward':
        sample_files = im_test_A
        net_g_logits = test_gen_A2B_logits

    elif FLAGS.direction == 'backward':
        sample_files = im_test_B
        net_g_logits = test_gen_B2A_logits
    else:
        raise Exception('--direction must be forward or backward')

    batch_idxs = (len(sample_files)) // FLAGS.batch_size
    for idx in range(0, batch_idxs):
        sample_image = threading_data(sample_files[idx * FLAGS.batch_size:(idx + 1) * FLAGS.batch_size],fn=pro)
        fake_img = sess.run(net_g_logits, feed_dict={in_var: sample_image})
        tl.vis.save_images(fake_img, [ni, ni], './{}/A2B_{}_{:04d}.jpg'.format(FLAGS.test_dir, FLAGS.direction, idx))


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', dest='phase', default='train', help='train, test')
    args = parser.parse_args()

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.test_dir):
        os.makedirs(FLAGS.test_dir)

    # if args.phase == 'train':
    #     train_cyclegan()
    # elif args.phase == 'test':
    #     test_cyclegan()
    train_cyclegan()
    # test_cyclegan()


if __name__ == '__main__':
    tf.app.run()
