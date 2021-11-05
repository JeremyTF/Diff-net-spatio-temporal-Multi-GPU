#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg
# from core.convlstm import *

class LayernormConvLSTMCell():

    def __init__(self, input_dim, hidden_dim, kernel_size):
        self.activation_function = tf.nn.relu

        self.input_dim = input_dim # 32*16
        self.hidden_dim = hidden_dim # 32*16

        self.kernel_size = kernel_size # 3*3
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # 1, 1

        # self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
        #                       out_channels=4 * self.hidden_dim,
        #                       kernel_size=self.kernel_size,
        #                       padding=self.padding,
        #                       bias=False)  # [2*32*16, 4*32*16, 3*3, [1, 1]]

    def next(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined = tf.concat([input_tensor, h_cur], axis=-1)

        # combined_conv = self.conv(combined)
        in_channels = self.input_dim + self.hidden_dim
        out_channels= 4 * self.hidden_dim

        print(in_channels,  out_channels)

        combined_conv = common.convolutional(combined, (3, 3, in_channels,  out_channels), True, 'lstm_encoding')

        # cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, num_or_size_splits=4, axis=-1)
        print(cc_i.get_shape(), cc_f.get_shape(), cc_o.get_shape(), cc_g.get_shape(), combined_conv.get_shape())

        # b, c, h, w = h_cur.size()
        # b, c, h, w = tf.shape(h_cur)

        # i = torch.sigmoid(cc_i)
        # f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)

        print('bn before', cc_g.get_shape())
        # cc_g = torch.layer_norm(cc_g, [h, w])
        cc_g = tf.layers.batch_normalization(cc_g, beta_initializer=tf.zeros_initializer(),
                                        gamma_initializer=tf.ones_initializer(),
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(), training=True)
        print('bn after', cc_g.get_shape())
        g = self.activation_function(cc_g)

        print('activation_function', g.get_shape())

        print(f.get_shape(), c_cur.get_shape(), i.get_shape(), g.get_shape(), combined_conv.get_shape())
        c_next = f * c_cur + i * g
        # c_next = torch.layer_norm(c_next, [h, w])
        c_next = tf.layers.batch_normalization(c_next, beta_initializer=tf.zeros_initializer(),
                                        gamma_initializer=tf.ones_initializer(),
                                        moving_mean_initializer=tf.zeros_initializer(),
                                        moving_variance_initializer=tf.ones_initializer(), training=True)

        h_next = o * self.activation_function(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        # print('init_hidden')
        # print(batch_size, height, width, self.hidden_dim)
        # print(30*'-')
        # return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
        #         torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
        return (tf.zeros([batch_size, height, width, self.hidden_dim], dtype=tf.float32),
                tf.zeros([batch_size, height, width, self.hidden_dim], dtype=tf.float32))

hyper_channels = 32
class LSTMFusion():
    def __init__(self):
        input_size = 1536

        hidden_size = 1536

        self.lstm_cell = LayernormConvLSTMCell(input_dim=input_size,
                                                  hidden_dim=hidden_size,
                                                  kernel_size=(3, 3))

    def next(self, current_encoding, current_state, input_resolution):
        batch, channel, height, width = current_encoding.get_shape()
        # batch, channel, height, width = current_encoding.shape
        # print(batch, channel, height, width)

        if current_state is None:
            hidden_state, cell_state = self.lstm_cell.init_hidden(batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                  image_size=(input_resolution/32, input_resolution/32))
        else:
            hidden_state, cell_state = current_state

        next_hidden_state, next_cell_state = self.lstm_cell.next(input_tensor=current_encoding,
                                                            cur_state=[hidden_state, cell_state])

        return next_hidden_state, next_cell_state

class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, input_mask,trainable):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD
        self.lstm_fusion      = LSTMFusion()
        self.lstm_state_bottom = None

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox = self.__build_nework(input_data, input_mask,input_size=608)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def model_loss(self, input_data, input_mask, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox, input_size):

        total_loss = 0
        for idx in range(cfg.TRAIN.INPUT_SEQUENCES_SIZE):
            per_input_data = input_data[:, idx, :, :, :]
            per_input_mask = input_mask[:, idx, :, :, :]
            per_label_sbbox = label_sbbox[:, idx, :, :, :, :]
            per_label_mbbox = label_mbbox[:, idx, :, :, :, :]
            per_label_lbbox = label_lbbox[:, idx, :, :, :, :]
            per_true_sbbox = true_sbbox[:, idx, :, :]
            per_true_mbbox = true_mbbox[:, idx, :, :]
            per_true_lbbox = true_lbbox[:, idx, :, :]


            # print per_input_data.get_shape(), per_input_mask.get_shape()

            try:
                self.per_conv_lbbox, self.per_conv_mbbox, self.per_conv_sbbox = self.__build_nework(per_input_data, per_input_mask, input_size)
                # print 'per_conv_lbbox: ', self.per_conv_lbbox.get_shape(), self.per_conv_mbbox.get_shape(), self.per_conv_sbbox.get_shape()

            except:
                raise NotImplementedError("Can not build up yolov3 network!")

            with tf.variable_scope('pred_sbbox'):
                self.per_pred_sbbox = self.decode(self.per_conv_sbbox, self.anchors[0], self.strides[0])

            with tf.variable_scope('pred_mbbox'):
                self.per_pred_mbbox = self.decode(self.per_conv_mbbox, self.anchors[1], self.strides[1])

            with tf.variable_scope('pred_lbbox'):
                self.per_pred_lbbox = self.decode(self.per_conv_lbbox, self.anchors[2], self.strides[2])

            giou_loss, conf_loss, prob_loss = self.compute_loss(per_label_sbbox, per_label_mbbox, per_label_lbbox, 
                                                                per_true_sbbox, per_true_mbbox, per_true_lbbox)


            total_loss = total_loss + giou_loss + conf_loss + prob_loss

        return total_loss

    def __build_nework(self, input_data, input_mask, input_size):

        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        mask_route_1, mask_route_2, input_mask = backbone.encoder(input_mask, self.trainable)

        image_fp = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        mask_fp = common.convolutional(input_mask, (1, 1, 1024,  512), self.trainable, 'conv52_1')

        # fp diff unit
        be_corrected_fp = image_fp + mask_fp
        to_add = image_fp - mask_fp
        to_del = mask_fp - image_fp
        input_data = tf.concat([be_corrected_fp, to_add, to_del], axis=-1)

        # Conv_LSTM #
        print("before Conv_LSTM: ", input_data.get_shape())
        self.lstm_state_bottom = self.lstm_fusion.next(current_encoding=input_data,
                                            current_state=self.lstm_state_bottom, input_resolution=input_size)
        input_data = self.lstm_state_bottom[0]
        print("after Conv_LSTM: ", input_data.get_shape())
        # Conv_LSTM #

        input_data = common.convolutional(input_data, (3, 3,  1536, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        image_fp = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        mask_route_2 = common.convolutional(mask_route_2, (1, 1, 512,  256), self.trainable, 'conv58_1')
        # fp diff unit
        be_corrected_fp = image_fp + mask_route_2
        to_add = image_fp - mask_route_2
        to_del = mask_route_2 - image_fp
        input_data = tf.concat([be_corrected_fp, to_add, to_del], axis=-1)

        input_data = common.convolutional(input_data, (3, 3, 768, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        image_fp = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        mask_route_1 = common.convolutional(mask_route_1, (1, 1, 256, 128), self.trainable, 'conv64_1')
        # fp diff unit
        be_corrected_fp = image_fp + mask_route_1
        to_add = image_fp - mask_route_1
        to_del = mask_route_1 - image_fp
        input_data = tf.concat([be_corrected_fp, to_add, to_del], axis=-1)

        input_data = common.convolutional(input_data, (3, 3, 384, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        # print 'conv_lbbox.get_shape(): ', conv_lbbox.get_shape(), conv_mbbox.get_shape(), conv_sbbox.get_shape()

        return conv_lbbox, conv_mbbox, conv_sbbox

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        conf_focal = self.focal(respond_bbox, pred_conf)

        # conf_loss = conf_focal * (
        #         respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        #         +
        #         respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        # )

        # prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        conf_loss = conf_focal * (
                respond_bbox * (-respond_bbox * tf.math.log(tf.maximum(tf.sigmoid(conv_raw_conf), 1e-15))) \
                - (1 - respond_bbox) * tf.math.log(tf.maximum(1 - tf.sigmoid(conv_raw_conf), 1e-15)) \
                + respond_bgd * (-respond_bbox * tf.math.log(tf.maximum(tf.sigmoid(conv_raw_conf), 1e-15))) \
                - (1 - respond_bbox) * tf.math.log(tf.maximum(1 - tf.sigmoid(conv_raw_conf), 1e-15))
        )

        prob_loss = respond_bbox * (-label_prob * tf.math.log(tf.maximum(tf.sigmoid(conv_raw_prob), 1e-15)) - (1 - label_prob) * tf.math.log(tf.maximum(1 - tf.sigmoid(conv_raw_prob), 1e-15)))

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss


    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.per_conv_sbbox, self.per_pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.per_conv_mbbox, self.per_pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.per_conv_lbbox, self.per_pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss


