# Adapted from : VGG 16 model : https://github.com/machrisaa/tensorflow-vgg
import time
import os
import inspect

import numpy as np
from termcolor import colored
import tensorflow as tf

from hed.losses import sigmoid_cross_entropy_balanced
from hed.utils._io import IO

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

slim = tf.contrib.slim

class MobileNet_v1():

    def __init__(self, cfgs, run='training',
                 min_depth=8,
                 depth_multiplier=1.0,
                 output_stride=None,
                 use_explicit_padding=False,
                 scope=None):

        self.min_depth = min_depth
        self.depth_multiplier = depth_multiplier
        self.output_stride = output_stride
        self.use_explicit_padding = use_explicit_padding
        self.scope = scope
        self.kernel_size = [3,3]

        self.cfgs = cfgs
        self.io = IO()

        self.images = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], self.cfgs[run]['n_channels']])
        self.edgemaps = tf.placeholder(tf.float32, [None, self.cfgs[run]['image_height'], self.cfgs[run]['image_width'], 1])
        self.coordinates = tf.placeholder(tf.float32, [None, 8])
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.define_model()


    def define_model(self):



        depth = lambda d: max(int(d * self.depth_multiplier), self.min_depth)

        # Used to find thinned depths for each layer.
        if self.depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')

        if self.output_stride is not None and self.output_stride not in [8, 16, 32]:
            raise ValueError('Only allowed output_stride values are 8, 16, 32.')

        padding = 'SAME'
        if self.use_explicit_padding:
            padding = 'VALID'

        with tf.variable_scope(self.scope, 'MobilenetV1', [self.images]):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
                # The current_stride variable keeps track of the output stride of the
                # activations, i.e., the running product of convolution strides up to the
                # current network layer. This allows us to invoke atrous convolution
                # whenever applying the next convolution would result in the activations
                # having output stride larger than the target output_stride.
                current_stride = 1

                # The atrous convolution rate parameter.
                rate = 1

                start_time = time.time()
                self.net = self.images
                with tf.variable_scope('Stage_1') as scope:
                    with tf.variable_scope('conv2d_0') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 2
                        else:
                            layer_stride = 2
                            layer_rate = 1
                            current_stride *= 2
                        if self.use_explicit_padding:
                            self.net = self._fixed_padding(self.net, self.kernel_size)
                        self.conv2d_0 = slim.conv2d(self.net, depth(32), self.kernel_size, stride=2, normalizer_fn=slim.batch_norm,
                                               scope='Conv2d_0')
                    self.side_out0 = self.side_layer(self.conv2d_0, name='side_out0', upscale=2)
                    self.io.print_info('Added CONV-BLOCK-0+SIDE-0')

                with tf.variable_scope('Stage_2') as scope:
                    with tf.variable_scope('conv2d_1') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_0 = self._fixed_padding(self.conv2d_0, self.kernel_size)
                        conv2d_1_dw = slim.separable_conv2d(self.conv2d_0, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_1_depthwise')
                        self.conv2d_1_pw = slim.conv2d(conv2d_1_dw, depth(64), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_1_pointwise')

                    with tf.variable_scope('conv2d_2') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 2
                        else:
                            layer_stride = 2
                            layer_rate = 1
                            current_stride *= 2
                        if self.use_explicit_padding:
                            self.conv2d_1_pw = self._fixed_padding(self.conv2d_1_pw, self.kernel_size)

                        conv2d_2_dw = slim.separable_conv2d(self.conv2d_1_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_2_depthwise')
                        self.conv2d_2_pw = slim.conv2d(conv2d_2_dw, depth(128), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_2_pointwise')
                    self.side_out1 = self.side_layer(self.conv2d_2_pw, name='side_out1', upscale=4)
                    self.io.print_info('Added CONV-BLOCK-1+SIDE-1')

                    with tf.variable_scope('Stage_3') as scope:
                        with tf.variable_scope('conv2d_3') as scope:
                            if self.output_stride is not None and current_stride == self.output_stride:
                                # If we have reached the target output_stride, then we need to employ
                                # atrous convolution with stride=1 and multiply the atrous rate by the
                                # current unit's stride for use in subsequent layers.
                                layer_stride = 1
                                layer_rate = rate
                                rate *= 1
                            else:
                                layer_stride = 1
                                layer_rate = 1
                                current_stride *= 1
                            if self.use_explicit_padding:
                                self.conv2d_2_pw = self._fixed_padding(self.conv2d_2_pw, self.kernel_size)
                            conv2d_3_dw = slim.separable_conv2d(self.conv2d_2_pw, None, self.kernel_size,
                                                                depth_multiplier=1,
                                                                stride=layer_stride,
                                                                rate=layer_rate,
                                                                normalizer_fn=slim.batch_norm,
                                                                scope='Conv2d_3_depthwise')
                            self.conv2d_3_pw = slim.conv2d(conv2d_3_dw, depth(128), [1, 1],
                                                      stride=1,
                                                      normalizer_fn=slim.batch_norm,
                                                      scope='Conv2d_3_pointwise')
                        with tf.variable_scope('conv2d_4') as scope:
                            if self.output_stride is not None and current_stride == self.output_stride:
                                # If we have reached the target output_stride, then we need to employ
                                # atrous convolution with stride=1 and multiply the atrous rate by the
                                # current unit's stride for use in subsequent layers.
                                layer_stride = 1
                                layer_rate = rate
                                rate *= 2
                            else:
                                layer_stride = 2
                                layer_rate = 1
                                current_stride *= 2
                            if self.use_explicit_padding:
                                self.conv2d_3_pw = self._fixed_padding(self.conv2d_3_pw, self.kernel_size)

                            conv2d_4_dw = slim.separable_conv2d(self.conv2d_3_pw, None, self.kernel_size,
                                                                depth_multiplier=1,
                                                                stride=layer_stride,
                                                                rate=layer_rate,
                                                                normalizer_fn=slim.batch_norm,
                                                                scope='Conv2d_4_depthwise')
                            self.conv2d_4_pw = slim.conv2d(conv2d_4_dw, depth(256), [1, 1],
                                                      stride=1,
                                                      normalizer_fn=slim.batch_norm,
                                                      scope='Conv2d_4_pointwise')
                        self.side_out2 = self.side_layer(self.conv2d_4_pw, name='side_out2', upscale=8)
                        self.io.print_info('Added CONV-BLOCK-2+SIDE-2')

                with tf.variable_scope('Stage_4') as scope:
                    with tf.variable_scope('conv2d_5') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_4_pw = self._fixed_padding(self.conv2d_4_pw, self.kernel_size)
                        conv2d_5_dw = slim.separable_conv2d(self.conv2d_4_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_5_depthwise')
                        self.conv2d_5_pw = slim.conv2d(conv2d_5_dw, depth(256), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_5_pointwise')
                    with tf.variable_scope('conv2d_6') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 2
                        else:
                            layer_stride = 2
                            layer_rate = 1
                            current_stride *= 2
                        if self.use_explicit_padding:
                            self.conv2d_5_pw = self._fixed_padding(self.conv2d_5_pw, self.kernel_size)

                        conv2d_6_dw = slim.separable_conv2d(self.conv2d_5_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_6_depthwise')
                        self.conv2d_6_pw = slim.conv2d(conv2d_6_dw, depth(512), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_6_pointwise')
                    self.side_out3 = self.side_layer(self.conv2d_6_pw, name='side_out3', upscale=16)
                    self.io.print_info('Added CONV-BLOCK-3+SIDE-3')

                with tf.variable_scope('Stage_5') as scope:
                    with tf.variable_scope('conv2d_7') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_6_pw = self._fixed_padding(self.conv2d_6_pw, self.kernel_size)
                        conv2d_7_dw = slim.separable_conv2d(self.conv2d_6_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_7_depthwise')
                        self.conv2d_7_pw = slim.conv2d(conv2d_7_dw, depth(512), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_7_pointwise')
                    with tf.variable_scope('conv2d_8') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_7_pw = self._fixed_padding(self.conv2d_7_pw, self.kernel_size)

                        conv2d_8_dw = slim.separable_conv2d(self.conv2d_7_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_8_depthwise')
                        self.conv2d_8_pw = slim.conv2d(conv2d_8_dw, depth(512), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_8_pointwise')
                    with tf.variable_scope('conv2d_9') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_8_pw = self._fixed_padding(self.conv2d_8_pw, self.kernel_size)
                        conv2d_9_dw = slim.separable_conv2d(self.conv2d_8_pw, None, self.kernel_size,
                                                            depth_multiplier=1,
                                                            stride=layer_stride,
                                                            rate=layer_rate,
                                                            normalizer_fn=slim.batch_norm,
                                                            scope='Conv2d_9_depthwise')
                        self.conv2d_9_pw = slim.conv2d(conv2d_9_dw, depth(512), [1, 1],
                                                  stride=1,
                                                  normalizer_fn=slim.batch_norm,
                                                  scope='Conv2d_9_pointwise')
                    with tf.variable_scope('conv2d_10') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_9_pw = self._fixed_padding(self.conv2d_9_pw, self.kernel_size)

                        conv2d_10_dw = slim.separable_conv2d(self.conv2d_9_pw, None, self.kernel_size,
                                                             depth_multiplier=1,
                                                             stride=layer_stride,
                                                             rate=layer_rate,
                                                             normalizer_fn=slim.batch_norm,
                                                             scope='Conv2d_10_depthwise')
                        self.conv2d_10_pw = slim.conv2d(conv2d_10_dw, depth(512), [1, 1],
                                                   stride=1,
                                                   normalizer_fn=slim.batch_norm,
                                                   scope='Conv2d_10_pointwise')

                    with tf.variable_scope('conv2d_11') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_10_pw = self._fixed_padding(self.conv2d_10_pw, self.kernel_size)

                        conv2d_11_dw = slim.separable_conv2d(self.conv2d_10_pw, None, self.kernel_size,
                                                             depth_multiplier=1,
                                                             stride=layer_stride,
                                                             rate=layer_rate,
                                                             normalizer_fn=slim.batch_norm,
                                                             scope='Conv2d_11_depthwise')
                        self.conv2d_11_pw = slim.conv2d(conv2d_11_dw, depth(512), [1, 1],
                                                   stride=1,
                                                   normalizer_fn=slim.batch_norm,
                                                   scope='Conv2d_11_pointwise')
                    self.side_out4 = self.side_layer(self.conv2d_11_pw, name='side_out4', upscale=16)
                    self.io.print_info('Added CONV-BLOCK-4+SIDE-4')

                with tf.variable_scope('Stage_6') as scope:
                    with tf.variable_scope('conv2d_12') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 2
                        else:
                            layer_stride = 2
                            layer_rate = 1
                            current_stride *= 2
                        if self.use_explicit_padding:
                            self.conv2d_11_pw = self._fixed_padding(self.conv2d_11_pw, self.kernel_size)
                        conv2d_12_dw = slim.separable_conv2d(self.conv2d_11_pw, None, self.kernel_size,
                                                             depth_multiplier=1,
                                                             stride=layer_stride,
                                                             rate=layer_rate,
                                                             normalizer_fn=slim.batch_norm,
                                                             scope='Conv2d_12_depthwise')
                        self.conv2d_12_pw = slim.conv2d(conv2d_12_dw, depth(1024), [1, 1],
                                                   stride=1,
                                                   normalizer_fn=slim.batch_norm,
                                                   scope='Conv2d_12_pointwise')
                    with tf.variable_scope('conv2d_13') as scope:
                        if self.output_stride is not None and current_stride == self.output_stride:
                            # If we have reached the target output_stride, then we need to employ
                            # atrous convolution with stride=1 and multiply the atrous rate by the
                            # current unit's stride for use in subsequent layers.
                            layer_stride = 1
                            layer_rate = rate
                            rate *= 1
                        else:
                            layer_stride = 1
                            layer_rate = 1
                            current_stride *= 1
                        if self.use_explicit_padding:
                            self.conv2d_12_pw = self._fixed_padding(self.conv2d_12_pw, self.kernel_size)

                        conv2d_13_dw = slim.separable_conv2d(self.conv2d_12_pw, None, self.kernel_size,
                                                             depth_multiplier=1,
                                                             stride=layer_stride,
                                                             rate=layer_rate,
                                                             normalizer_fn=slim.batch_norm,
                                                             scope='Conv2d_13_depthwise')
                        self.conv2d_13_pw = slim.conv2d(conv2d_13_dw, depth(1024), [1, 1],
                                                   stride=1,
                                                   normalizer_fn=slim.batch_norm,
                                                   scope='Conv2d_13_pointwise')
                    self.side_out5 = self.side_layer(self.conv2d_13_pw, name='side_out5', upscale=32)
                    self.io.print_info('Added CONV-BLOCK-5+SIDE-5')
                self.side_outputs = [self.side_out0, self.side_out1, self.side_out2, self.side_out3, self.side_out4, self.side_out5]
                w_shape = [1, 1, len(self.side_outputs), 1]
                self.fuse = self.conv_layer(tf.concat(self.side_outputs, axis=3),
                                            w_shape, name='fuse_1', use_bias=False,
                                            w_init=tf.constant_initializer(0.2))

                self.io.print_info('Added FUSE layer')
                self.outputs = self.side_outputs + [self.fuse]

                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                filters = [64, 128, 256, 128, 64, 32, 8]
                kernels = [7, 3, 3, 3, 3, 3, 3]
                strides = [2, 0, 2, 4, 4, 2,2]
                x = self._conv(self.fuse, kernels[0], filters[0], strides[0])
                #self.io.print_info('Added  here')
                x = self._bn(x)
                #self.io.print_info('Added  here1')
                x = self._relu(x)
                #self.io.print_info('Added  here2')
                x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
                self.io.print_info('Add ResNet_conv1 layer')
                self.rn1 = x
                # conv2_x
                x = self._residual_block(x, name='ResNet_conv2_1')
                x = self._residual_block(x, name='ResNet_conv2_2')
                self.rn2 = x

                self.io.print_info("Add ResNet_conv2 layer")

                # conv3_x
                x = self._residual_block_first(x, filters[2], strides[2], name='ResNet_conv3_1')
                x = self._residual_block(x, name='ResNet_conv3_2')
                self.rn3 = x
                self.io.print_info("Add ResNet_conv3 layer")
                # conv4_x
                x = self._residual_block_first(x, filters[3], strides[3], name='ResNet_conv4_1')
                x = self._residual_block(x, name='ResNet_conv4_2')
                self.io.print_info("Add ResNet_conv4 layer")
                self.rn4 = x
                # conv5_x
                x = self._residual_block_first(x, filters[4], strides[4], name='ResNet_conv5_1')
                x = self._residual_block(x, name='ResNet_conv5_2')
                self.io.print_info("Add ResNet_conv5 layer")
                self.rn5 = x
                # conv6_x
                x = self._residual_block_first(x, filters[5], strides[5], name='ResNet_conv6_1')
                x = self._residual_block(x, name='ResNet_conv6_2')
                self.io.print_info("Add ResNet_conv6 layer")
                self.rn6 = x
                # conv7_x
                x = self._residual_block_first(x, filters[6], strides[6], name='ResNet_conv7_1')
                x = self._residual_block(x, name='ResNet_conv7_2')
                self.io.print_info("Add ResNet_conv7 layer")
                self.regressor = tf.layers.flatten(x)

                self.io.print_info("Build model finished: {:.4f}s".format(time.time() - start_time))



    def conv_layer(self, x, W_shape, b_shape=None, name=None,
                   padding='SAME', use_bias=True, w_init=None, b_init=None):

        W = self.weight_variable(W_shape, w_init)
        #tf.summary.histogram('weights_{}'.format(name), W)

        if use_bias:
            b = self.bias_variable([b_shape], b_init)
            #tf.summary.histogram('biases_{}'.format(name), b)

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
        self.coords_prediction = self.fc3l

    def _fixed_padding(self, inputs, kernel_size, rate=1):
        """Pads the input along the spatial dimensions independently of input size.

        Pads the input such that if it was used in a convolution with 'VALID' padding,
        the output would have the same dimensions as if the unpadded input was used
        in a convolution with 'SAME' padding.

        Args:
          inputs: A tensor of size [batch, height_in, width_in, channels].
          kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
          rate: An integer, rate for atrous convolution.

        Returns:
          output: A tensor of size [batch, height_out, width_out, channels] with the
            input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                                 kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
        pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
        pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
        pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                        [pad_beg[1], pad_end[1]], [0, 0]])
        return padded_inputs

    def setup_training(self, session):

        """
            Apply sigmoid non-linearity to side layer ouputs + fuse layer outputs
            Compute total loss := side_layer_loss + fuse_layer_loss
            Compute predicted edge maps from fuse layer as pseudo performance metric to track
        """

        self.predictions = []
        # self.coord_pred = []
        self.loss = 0

        self.io.print_warning('Deep supervision application set to {}'.format(self.cfgs['deep_supervision']))

        for idx, b in enumerate(self.side_outputs):
            output = tf.nn.sigmoid(b, name='output_{}'.format(idx))
            #print("The {} layer's shape is:".format(idx), b.get_shape())
            cost = sigmoid_cross_entropy_balanced(b, self.edgemaps, name='cross_entropy{}'.format(idx))

            self.predictions.append(output)
            if self.cfgs['deep_supervision']:
                self.loss += (self.cfgs['loss_weights'] * cost)

        # self.coord_pred.append(self.fc3l)


        fuse_output = tf.nn.sigmoid(self.fuse, name='fuse')
        fuse_cost = sigmoid_cross_entropy_balanced(self.fuse, self.edgemaps, name='cross_entropy_fuse')
        # coincide_cost = vertices_and_edges_loss(self.fuse, self.fc3l, self.coordinates)

        coord_cost = tf.losses.huber_loss(self.coordinates, self.regressor)


        self.predictions.append(fuse_output)
        self.loss += (self.cfgs['loss_weights'] * fuse_cost)
        self.loss += coord_cost
        # self.loss += coincide_cost

        pred = tf.cast(tf.greater(fuse_output, 0.5), tf.int32, name='predictions')
        error = tf.cast(tf.not_equal(pred, tf.cast(self.edgemaps, tf.int32)), tf.float32)
        cd_error = tf.cast(tf.square(self.regressor - self.coordinates), tf.float32)
        self.cd_error = tf.reduce_mean(cd_error, name='regression_error')
        self.error = tf.reduce_mean(error, name='pixel_error')

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('error', self.error)
        tf.summary.scalar('cd_error', self.cd_error)

        self.merged_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/train', session.graph)
        self.val_writer = tf.summary.FileWriter(self.cfgs['save_dir'] + '/val')

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variable
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        print("model restored from %s" % path)

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _bn(self, x, name='bn'):
        return self.bn(x,self.is_train, name)

    def bn(self,x, is_train, name='bn'):
        moving_average_decay = 0.9
        # moving_average_decay = 0.99
        # self.io.print_info('Added  here')
        # moving_average_decay_init = 0.99
        with tf.variable_scope(name) as scope:
            decay = moving_average_decay

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            with tf.device('/CPU:0'):
                mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                                 initializer=tf.zeros_initializer(), trainable=False)
                sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer(), trainable=False)
                beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                                   initializer=tf.zeros_initializer())
                gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer())
            # BN when training
            update = 1.0 - decay
            # with tf.control_dependencies([tf.Print(decay, [decay])]):
            # update_mu = mu.assign_sub(update*(mu - batch_mean))
            update_mu = mu.assign_sub(update * (mu - batch_mean))
            update_sigma = sigma.assign_sub(update * (sigma - batch_var))
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

            mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                                lambda: (mu, sigma))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)

        return bn

    def _relu(self, x, leakness=0.0, name=None):
        if leakness > 0.0:
            name = 'lrelu' if name is None else name
            return tf.maximum(x, x * leakness, name='lrelu')
        else:
            name = 'relu' if name is None else name
            return tf.nn.relu(x, name='relu')

    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", name="conv"):
        in_shape = x.get_shape()
        with tf.variable_scope(name):
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                                     tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / filter_size / filter_size / out_channel)))
            conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], pad)
        return conv
