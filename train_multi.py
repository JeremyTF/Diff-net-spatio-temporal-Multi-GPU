#! /usr/bin/env python
# coding=utf-8

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset1 import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg

PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

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
        self.num_gpus            = 1
        self.batch_size          = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_bbox_per_scale  = 150
        self.train_logdir        = "./convlstm/log"
        self.trainset            = Dataset('train')
        self.testset             = Dataset('test')
        self.steps_per_period    = len(self.trainset)

        self.config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.9

        self.sess                = tf.compat.v1.Session(config=self.config)

        self.input_size = cfg.TRAIN.INPUT_SIZE

        with tf.name_scope('define_input'):
            self.input_data   = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, self.input_size, self.input_size, 3], name='input_data')
            self.input_mask   = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, self.input_size, self.input_size, 1], name='input_mask')
            self.label_sbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, self.input_size/8, self.input_size/8, 3, 8], name='label_sbbox')
            self.label_mbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, self.input_size/16, self.input_size/16, 3, 8], name='label_mbbox')
            self.label_lbbox  = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, self.input_size/32, self.input_size/32, 3, 8], name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, shape=[cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.INPUT_SEQUENCES_SIZE, 150, 4], name='lbboxes')
            self.trainable     = tf.compat.v1.placeholder(dtype=tf.bool, name='training')



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




    def train(self):

        config = tf.ConfigProto(intra_op_parallelism_threads=8)
        sess = tf.Session(config=config)

        with tf.device('/cpu:0'):
            tower_grads = []
            for i in range(self.num_gpus):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    _input_data = self.input_data[i * self.batch_size: (i + 1) * self.batch_size]
                    _input_mask = self.input_mask[i * self.batch_size: (i + 1) * self.batch_size]
                    _label_sbbox = self.label_sbbox[i * self.batch_size: (i + 1) * self.batch_size]
                    _label_mbbox = self.label_mbbox[i * self.batch_size: (i + 1) * self.batch_size]
                    _label_lbbox = self.label_lbbox[i * self.batch_size: (i + 1) * self.batch_size]
                    _true_sbboxes = self.true_sbboxes[i * self.batch_size: (i + 1) * self.batch_size]
                    _true_mbboxes = self.true_mbboxes[i * self.batch_size: (i + 1) * self.batch_size]
                    _true_lbboxes = self.true_lbboxes[i * self.batch_size: (i + 1) * self.batch_size]

                    model = YOLOV3(self.trainable)
                    net_var_m = tf.global_variables()
                    loss_single_gpu = model.model_loss(
                        _input_data,_input_mask,
                        _label_sbbox, _label_mbbox, _label_lbbox,
                        _true_sbboxes, _true_mbboxes, _true_lbboxes, self.input_size)
                    # loss_single_gpu = giou_loss + conf_loss + prob_loss
                    # tf.get_variable_scope().reuse_variables()

                    second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate)
                    loss_compute = second_stage_optimizer.compute_gradients(loss_single_gpu)
                    tower_grads.append(loss_compute)

            tower_grads = average_gradients(tower_grads)
            loss_tower_grads_apply = second_stage_optimizer.apply_gradients(tower_grads)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([loss_tower_grads_apply, self.global_step_update]):
                    with tf.control_dependencies([self.moving_ave]):
                        train_op_with_all_variables = tf.no_op()

            with tf.name_scope('loader_and_saver'):
                loader = tf.train.Saver(net_var_m)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

            with tf.name_scope('summary'):
                tf.summary.scalar("learn_rate", self.learn_rate)
                # tf.summary.scalar("giou_loss", giou_loss)
                # tf.summary.scalar("conf_loss", conf_loss)
                # tf.summary.scalar("prob_loss", prob_loss)
                tf.summary.scalar("total_loss", loss_single_gpu)

                logdir = "./convlstm/log/"
                if os.path.exists(logdir): shutil.rmtree(logdir)
                os.mkdir(logdir)
                write_op = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(logdir, graph=sess.graph)

            sess.run(tf.global_variables_initializer())

            try:
                print('=> Restoring weights from: %s ... ' % self.initial_weight)
                loader.restore(sess, self.initial_weight)
            except:
                print('=> %s does not exist !!!' % self.initial_weight)
                print('=> Now it starts to train YOLOV3 from scratch ...')
                self.first_stage_epochs = 0

            coor = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coor)
            try:
                while not coor.should_stop():
                    # 1.
                    for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):

                        train_op = train_op_with_all_variables

                        pbar = tqdm(self.trainset)
                        train_epoch_loss, test_epoch_loss = [], []
                        for train_data in pbar:
                            _, summary, train_step_loss, global_step_val = sess.run(
                                [train_op, write_op, loss_single_gpu, self.global_step], feed_dict={
                                    self.input_data: train_data[0],
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
                            summary_writer.add_summary(summary, global_step_val)
                            pbar.set_description("train loss: %.2f" %train_step_loss)

                        train_epoch_loss = np.mean(train_epoch_loss)
                        ckpt_file = "./afs/diff_net/MC3D_CONVLSTM_test_loss=%.4f.ckpt" % train_epoch_loss
                        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        print("=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ..."
                              % (epoch, log_time, train_epoch_loss, ckpt_file))
                        saver.save(sess, ckpt_file, global_step=epoch)
            except tf.errors.OutOfRangeError:
                print('Done!kill all the threads......')
            finally:
                coor.request_stop()
                print('all the threads all asked to stop')
            coor.join(threads)
            print('killed all the threads')


if __name__ == '__main__':YoloTrain().train()




