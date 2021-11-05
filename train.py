#! /usr/bin/env python
# coding=utf-8

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale    = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes             = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes         = len(self.classes)
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight      = cfg.TRAIN.INITIAL_WEIGHT
        self.time                = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./log"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)

        self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.9

        self.sess                = tf.compat.v1.Session(config=self.config)

        input_size = cfg.TRAIN.INPUT_SIZE

        with tf.name_scope('define_input'):
            self.input_data   = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, input_size, input_size, 3], name='input_data')
            self.input_mask   = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, input_size, input_size, 1], name='input_mask')
            self.label_sbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, input_size/8, input_size/8, 3, 8], name='label_sbbox')
            self.label_mbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, input_size/16, input_size/16, 3, 8], name='label_mbbox')
            self.label_lbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, input_size/32, input_size/32, 3, 8], name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='lbboxes')
            self.trainable     = tf.compat.v1.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = YOLOV3(self.trainable)
            # self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
            #                                         self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
            #                                         self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.model.model_loss(self.input_data, self.input_mask,
                                                self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes, input_size)
            self.net_var = tf.compat.v1.global_variables()

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
                                        dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                    (1 + tf.cos(
                                        (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
                # true_fn = lambda: self.learn_rate_init,
                # false_fn = lambda: self.learn_rate_end,

            )
            self.global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            self.moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, self.global_step_update]):
                    with tf.control_dependencies([self.moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)
            # second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4, 0.9, 0.9999).minimize(self.loss)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, self.global_step_update]):
                    with tf.control_dependencies([self.moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.compat.v1.train.Saver(self.net_var)
            self.saver  = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=10)

        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar("learn_rate",      self.learn_rate)
            # tf.compat.v1.summary.scalar("giou_loss",  self.giou_loss)
            # tf.compat.v1.summary.scalar("conf_loss",  self.conf_loss)
            # tf.compat.v1.summary.scalar("prob_loss",  self.prob_loss)
            tf.compat.v1.summary.scalar("total_loss", self.loss)

            logdir = "./log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.compat.v1.summary.merge_all()
            self.summary_writer  = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph)

    def train(self):
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train hdmap-diff from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            # if epoch <= self.first_stage_epochs:：挖潜
            #     train_op = self.train_op_with_frozen_variables
            # else:
            train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)#创建进度条
            # print(len(self.trainset))

            train_epoch_loss, test_epoch_loss = [], []
            # idx = 0
            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                                                self.input_data:   train_data[0],
                                                self.input_mask:   train_data[1],
                                                self.label_sbbox:  train_data[2],
                                                self.label_mbbox:  train_data[3],
                                                self.label_lbbox:  train_data[4],
                                                self.true_sbboxes: train_data[5],
                                                self.true_mbboxes: train_data[6],
                                                self.true_lbboxes: train_data[7],
                                                self.trainable:    True,
                })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" %train_step_loss)

            # for test_data in self.testset:
            #     test_step_loss = self.sess.run( self.loss, feed_dict={
            #                                     self.input_data:   test_data[0],
            #                                     self.input_mask:   test_data[1],
            #                                     self.label_sbbox:  test_data[2],
            #                                     self.label_mbbox:  test_data[3],
            #                                     self.label_lbbox:  test_data[4],
            #                                     self.true_sbboxes: test_data[5],
            #                                     self.true_mbboxes: test_data[6],
            #                                     self.true_lbboxes: test_data[7],
            #                                     self.trainable:    False,
            #     })

            #     test_epoch_loss.append(test_step_loss)

            train_epoch_loss = np.mean(train_epoch_loss)
            ckpt_file = "./ckpts/checkpointloss=%.4f.ckpt" % train_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f  Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':YoloTrain().train()




