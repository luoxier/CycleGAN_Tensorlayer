import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *


def cyclegan_generator_resnet(image, num=9, is_train=True, reuse=False, batch_size=None, name='generator'):
    b_init = tf.constant_initializer(value=0.0)
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        gf_dim = 32

        net_in = InputLayer(image, name='in')
        net_pad = PadLayer(net_in, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='inpad')

        net_c1 = Conv2d(net_pad, gf_dim, (7, 7), (1, 1), act=None,
                padding='VALID', W_init=w_init, name='c7s1-32')  # c7s1-32   shape:(1,256,256,32)
        net_c1 = InstanceNormLayer(net_c1, act=tf.nn.relu, name='ins1')

        net_c2 = Conv2d(net_c1, gf_dim * 2, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init,name='d64')  # d64   shape:(1,128,128,64)
        net_c2 = InstanceNormLayer(net_c2, act=tf.nn.relu, name='ins2')

        net_c3 = Conv2d(net_c2, gf_dim * 4, (3, 3), (2, 2), act=None,
                padding='SAME', W_init=w_init, name='d128')  # d128   shape(1,64,64,128)
        net_c3 = InstanceNormLayer(net_c3, act=tf.nn.relu, name='ins3')

        n = net_c3
        for i in range(num):
            n_pad = PadLayer(n, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='pad_%s' % i)
            nn = Conv2d(n_pad, gf_dim * 4, (3, 3), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=b_init, name='res/c1/%s' % i)
            nn = InstanceNormLayer(nn, act=tf.nn.relu, name='res/ins/%s' % i)

            nn = PadLayer(nn, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='pad2_%s' % i)
            nn = Conv2d(nn, gf_dim * 4, (3, 3), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=b_init, name='res/c2/%s' % i)
            nn = InstanceNormLayer(nn, name='res/ins2/%s' % i)

            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        net_r9 = n
        # net_d1 = DeConv2d(net_r9, gf_dim * 2, (3, 3), out_size=(128,128),
        #     strides=(2, 2), padding='SAME', batch_size=batch_size, act=None, name='u64')  # 

        size_d1 = net_r9.outputs.get_shape().as_list()
        net_up1 = UpSampling2dLayer(net_r9, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                   align_corners=False,name= 'up1/upsample2d')
        net_up1_pad = PadLayer(net_up1, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup1')
        net_d1 = Conv2d(net_up1_pad, gf_dim * 2, (3, 3), (1, 1), act=None,padding='VALID', W_init=w_init, name='u64')
        net_d1 = InstanceNormLayer(net_d1, act=tf.nn.relu, name='inso1')

        # net_d2 = DeConv2d(net_d1, gf_dim, (3, 3), out_size=(256,256),
        #     strides=(2, 2), padding='SAME',batch_size=batch_size, act=None, name='u32')  # u32
        size_d2 = net_d1.outputs.get_shape().as_list()
        net_up2 = UpSampling2dLayer(net_d1, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1, align_corners=False,
                              name='up2/upsample2d')
        net_up2_pad = PadLayer(net_up2, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT", name='padup2')
        net_d2 = Conv2d(net_up2_pad, gf_dim , (3, 3), (1, 1), act=None,padding='VALID', W_init=w_init, name='u32')
        net_d2 = InstanceNormLayer(net_d2, act=tf.nn.relu, name='inso2')

        net_d2_pad = PadLayer(net_d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT", name='pado')
        net_c4 = Conv2d(net_d2_pad, 3, (7, 7), (1, 1), act=tf.nn.tanh,
            padding='VALID', name=name+'c7s1-3')  # c7s1-3
    return net_c4, net_c4.outputs


def cyclegan_discriminator_patch(inputs, is_train=True, reuse=False, name='discriminator'):
    df_dim = 64  # Dimension of discrim filters in first conv layer. [64]
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        patch_inputs = tf.random_crop(inputs, [1, 70, 70, 3])
        net_in = InputLayer(patch_inputs, name='d/in')
        # 1st
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu,
                        padding='SAME', W_init=w_init, name='d/h0/conv2d')  # C64
        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h1/conv2d')
        net_h1 = InstanceNormLayer(net_h1, act=lrelu, name="d/h1/instance_norm")
        # 2nd
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h2/conv2d')
        net_h2 = InstanceNormLayer(net_h2, act=lrelu, name="d/h2/instance_norm")
        # 3rd
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None,
                        padding='SAME', W_init=w_init, name='d/h3/conv2d')
        net_h3 = InstanceNormLayer(net_h3, act=lrelu, name="d/h3/instance_norm")
        # output
        net_h4 = Conv2d(net_h3, 1, (4, 4), (1, 1), act=None,
                        padding='SAME', W_init=w_init, name='d/h4/conv2d')

        logits = net_h4.outputs

    return net_h4,logits
