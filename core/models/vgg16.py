# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

from hed.losses import sigmoid_cross_entropy_balanced
from hed.utils._io import IO
from hed.losses import vertices_and_edges_loss

class Vgg16():

    def __init__(self, cfgs, run='training'):

        self.cfgs = cfgs
        self.io = IO()

        # base_path = os.path.abspath(os.path.dirname(__file__))
        # weights_file = os.path.join(base_path, self.cfgs['model_weights_path'])
        #
        # self.data_dict = np.load(weights_file, encoding='latin1').item()
        # self.io.print_info("Model weights loaded from {}".format(self.cfgs['model_weights_path']))

        self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']])
        self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])
        self.coordinates = tf.placeholder(tf.float32, [None, self.cfg[run]['coords']])

        self.define_model()


    def define_model(self):

        """
        Load VGG params from disk without FC layers A
        Add branch layers (with deconv) after each CONV block
        """

        start_time = time.time()
        # Stage 1
        #conv1_1
        with tf.name_scope("conv1_1") as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)

        #conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)

        #side_1
        self.side_1 = self.side_layer(self.conv1_2, "side_1", 1)

        #pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

        #Stage 2
        #conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)

        #siede_2
        self.side_2 = self.side_layer(self.conv2_2, "side_2", 2)

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

        #Stage 3
        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)

        self.side_3 = self.side_layer(self.conv3_3, "side_3", 4)

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

        #Stage 4
        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)

        self.side_4 = self.side_layer(self.conv4_3, "side_4", 8)

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

        #Stage 5
        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)

        #side_5
        self.side_5 = self.side_layer(self.conv5_3, "side_5", 16)

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        self.io.print_info('Added CONV-BLOCK-5+SIDE-5')


        self.side_outputs = [self.side_1, self.side_2, self.side_3, self.side_4, self.side_5]

        w_shape = [1, 1, len(self.side_outputs), 1]
        self.fuse = self.conv_layer(tf.concat(self.side_outputs, axis=3),
                                    w_shape, name='fuse_1', use_bias=False,
                                    w_init=tf.constant_initializer(0.2))

        self.io.print_info('Added FUSE layer')

        # complete output maps from side layer and fuse layers
        self.outputs = self.side_outputs + [self.fuse]

        #self.data_dict = None


        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 1024],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)

        self.io.print_info('Added fc1 layer')

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([1024, 512],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)

        self.io.print_info('Added fc2 layer')

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([512, 8],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[8], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)

        self.io.print_info('Added coordinates output layer')

        self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))



    def conv_layer(self, x, W_shape, b_shape=None, name=None,
                   padding='SAME', use_bias=True, w_init=None, b_init=None):

        W = self.weight_variable(W_shape, w_init)
        tf.summary.histogram('weights_{}'.format(name), W)

        if use_bias:
            b = self.bias_variable([b_shape], b_init)
            tf.summary.histogram('biases_{}'.format(name), b)

        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

        return conv + b if use_bias else conv

    def deconv_layer(self, x, upscale, name, padding='SAME', w_init=None):

        x_shape = tf.shape(x)
        in_shape = x.shape.as_list()

        w_shape = [upscale * 2, upscale * 2, in_shape[-1], 1]
        strides = [1, upscale, upscale, 1]

        W = self.weight_variable(w_shape, w_init)
        tf.summary.histogram('weights_{}'.format(name), W)

        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[2]]) * tf.constant(strides, tf.int32)
        deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

        return deconv

    def side_layer(self, inputs, name, upscale):
        """
            https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
            1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
        """
        with tf.variable_scope(name):

            in_shape = inputs.shape.as_list()
            w_shape = [1, 1, in_shape[-1], 1]

            classifier = self.conv_layer(inputs, w_shape, b_shape=1,
                                         w_init=tf.constant_initializer(),
                                         b_init=tf.constant_initializer(),
                                         name=name + '_reduction')

            classifier = self.deconv_layer(classifier, upscale=upscale,
                                           name='{}_deconv_{}'.format(name, upscale),
                                           w_init=tf.truncated_normal_initializer(stddev=0.1))

            return classifier


    def weight_variable(self, shape, initial):

        init = initial(shape)
        return tf.Variable(init)

    def bias_variable(self, shape, initial):

        init = initial(shape)
        return tf.Variable(init)

    def setup_testing(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs for predictions
        """

        self.predictions = []

        for idx, b in enumerate(self.outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            self.predictions.append(output)

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """

        self.predictions = []
        self.coord_pred = []
        self.loss = 0

        self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            print("The {} layer's shape is:".format(idx), b.get_shape())
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        self.coord_pred.append(self.fc3l)

        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')
        coincide_cost = vertices_and_edges_loss(self.fuse, self.fc3l, self.coordinates)

        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)
        self.loss += coincide_cost

        pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)

        self.merged_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/val')

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variable
        saver = tf.train.Saver(var_list)
        saver.restore(sess, saver_path=path)
        print("model restored from %s" % path)
