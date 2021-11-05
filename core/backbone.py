#! /usr/bin/env python
# coding=utf-8

import core.common as common
import tensorflow as tf

def encoder(input_data, trainable):
    """
    """
    with tf.variable_scope('mask_encoder'):
        input_data = common.convolutional(input_data, filters_shape=(3, 3,  1,  32), 
                                          trainable=trainable, name='conv0')
        
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 64,  64),
                                          trainable=trainable, name='conv1_1')

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv2', downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128,  128),
                                          trainable=trainable, name='conv2_1')

        mask_route_1 = common.convolutional(input_data, filters_shape=(3, 3, 128,  256),
                                          trainable=trainable, name='conv3', downsample=True)
        mask_route_1 = common.convolutional(mask_route_1, filters_shape=(3, 3, 256,  256),
                                          trainable=trainable, name='conv3_1')

        mask_route_2 = common.convolutional(mask_route_1, filters_shape=(3, 3,  256, 512),
                                          trainable=trainable, name='conv4', downsample=True)
        mask_route_2 = common.convolutional(mask_route_2, filters_shape=(3, 3,  512, 512),
                                          trainable=trainable, name='conv4_1')

        input_data = common.convolutional(mask_route_2, filters_shape=(3, 3,  512, 1024),
                                          trainable=trainable, name='conv5', downsample=True)
        input_data = common.convolutional(input_data, filters_shape=(3, 3,  1024, 1024),
                                          trainable=trainable, name='conv5_1')

    return mask_route_1, mask_route_2, input_data

def darknet53(input_data, trainable):

    # with tf.variable_scope('darknet', reuse=tf.AUTO_REUSE):
    with tf.compat.v1.variable_scope('darknet'):

        # print 'hello world1', input_data.get_shape()

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  3,  32), trainable=trainable, name='conv0')
        
        # print 'hello world2', input_data.get_shape()

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 32,  64),
                                          trainable=trainable, name='conv1', downsample=True)

        # # embed mask
        # mask_1 = common.downsample(input_mask, input_data)
        # input_data = tf.concat([input_data, mask_1], axis=-1)

        for i in range(1):
            input_data = common.residual_block(input_data,  64,  32, 64, trainable=trainable, name='residual%d' %(i+0))

        input_data = common.convolutional(input_data, filters_shape=(3, 3,  64, 128),
                                          trainable=trainable, name='conv4', downsample=True)

        # # embed mask
        # mask_2 = common.downsample(input_mask, input_data)
        # input_data = tf.concat([input_data, mask_2], axis=-1)

        for i in range(2):
            input_data = common.residual_block(input_data, 128,  64, 128, trainable=trainable, name='residual%d' %(i+1))

        input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                          trainable=trainable, name='conv9', downsample=True)

        # # embed mask
        # mask_3 = common.downsample(input_mask, input_data)
        # input_data = tf.concat([input_data, mask_3], axis=-1)

        for i in range(8):
            input_data = common.residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                          trainable=trainable, name='conv26', downsample=True)

        # # embed mask
        # mask_4 = common.downsample(input_mask, input_data)
        # input_data = tf.concat([input_data, mask_4], axis=-1)

        for i in range(8):
            input_data = common.residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                          trainable=trainable, name='conv43', downsample=True)

        # # embed mask
        # mask_5 = common.downsample(input_mask, input_data)
        # input_data = tf.concat([input_data, mask_5], axis=-1)

        for i in range(4):
            input_data = common.residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data