#! /usr/bin/env python
# coding=utf-8

import os
import cv2
import cv2 as cv
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import numexpr as ne
from concurrent.futures import ThreadPoolExecutor

np.seterr(divide='ignore', invalid='ignore')

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        self.time_steps = cfg.TRAIN.INPUT_SEQUENCES_SIZE


    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            txt = [line.replace('New_yizhuang/YiZhuangDaLuWang/', '/media/jiangshengjie/Elements/New_yizhuang_MC3D/home/wangning/Desktop/data/New_yizhuang/YiZhuangDaLuWang/') for line
                   in txt]
            annotations = [line.strip().strip(',') for line in txt if len(line.strip().split(',')[2:]) != 0]
            # print annotations
        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        pool = ThreadPoolExecutor(max_workers=10)

        thread0 = pool.submit(self.get_data, 0, 1)
        thread1 = pool.submit(self.get_data, 1, 0)
        thread2 = pool.submit(self.get_data, 2, 0)
        thread3 = pool.submit(self.get_data, 3, 0)
        thread4 = pool.submit(self.get_data, 4, 0)
        thread5 = pool.submit(self.get_data, 5, 0)
        thread6 = pool.submit(self.get_data, 6, 0)
        thread7 = pool.submit(self.get_data, 7, 0)

        batch_image0, batch_mask0, batch_label_sbbox0, batch_label_mbbox0, batch_label_lbbox0, \
        batch_sbboxes0, batch_mbboxes0, batch_lbboxes0 = thread0.result()

        batch_image1, batch_mask1, batch_label_sbbox1, batch_label_mbbox1, batch_label_lbbox1, \
        batch_sbboxes1, batch_mbboxes1, batch_lbboxes1 = thread1.result()

        batch_image2, batch_mask2, batch_label_sbbox2, batch_label_mbbox2, batch_label_lbbox2, \
        batch_sbboxes2, batch_mbboxes2, batch_lbboxes2 = thread2.result()

        batch_image3, batch_mask3, batch_label_sbbox3, batch_label_mbbox3, batch_label_lbbox3, \
        batch_sbboxes3, batch_mbboxes3, batch_lbboxes3 = thread3.result()

        batch_image4, batch_mask4, batch_label_sbbox4, batch_label_mbbox4, batch_label_lbbox4, \
        batch_sbboxes4, batch_mbboxes4, batch_lbboxes4 = thread4.result()

        batch_image5, batch_mask5, batch_label_sbbox5, batch_label_mbbox5, batch_label_lbbox5, \
        batch_sbboxes5, batch_mbboxes5, batch_lbboxes5 = thread5.result()

        batch_image6, batch_mask6, batch_label_sbbox6, batch_label_mbbox6, batch_label_lbbox6, \
        batch_sbboxes6, batch_mbboxes6, batch_lbboxes6 = thread6.result()

        batch_image7, batch_mask7, batch_label_sbbox7, batch_label_mbbox7, batch_label_lbbox7, \
        batch_sbboxes7, batch_mbboxes7, batch_lbboxes7 = thread7.result()




        thread24 = pool.submit(self.add_numpy, batch_image0, batch_image1, batch_image2, batch_image3, batch_image4,
                               batch_image5, batch_image6,
                               batch_image7)

        thread25 = pool.submit(self.add_numpy, batch_label_sbbox0, batch_label_sbbox1, batch_label_sbbox2,
                               batch_label_sbbox3, batch_label_sbbox4, batch_label_sbbox5,
                               batch_label_sbbox6, batch_label_sbbox7
                               )

        thread26 = pool.submit(self.add_numpy, batch_label_mbbox0, batch_label_mbbox1, batch_label_mbbox2,
                               batch_label_mbbox3, batch_label_mbbox4, batch_label_mbbox5,
                               batch_label_mbbox6, batch_label_mbbox7
                               )

        thread27 = pool.submit(self.add_numpy, batch_label_lbbox0, batch_label_lbbox1, batch_label_lbbox2,
                               batch_label_lbbox3, batch_label_lbbox4, batch_label_lbbox5,
                               batch_label_lbbox6, batch_label_lbbox7
                               )

        thread28 = pool.submit(self.add_numpy, batch_sbboxes0, batch_sbboxes1, batch_sbboxes2, batch_sbboxes3,
                               batch_sbboxes4, batch_sbboxes5,
                               batch_sbboxes6, batch_sbboxes7 )

        thread29 = pool.submit(self.add_numpy, batch_mbboxes0, batch_mbboxes1, batch_mbboxes2, batch_mbboxes3,
                               batch_mbboxes4, batch_mbboxes5,
                               batch_mbboxes6, batch_mbboxes7 )

        thread30 = pool.submit(self.add_numpy, batch_lbboxes0, batch_lbboxes1, batch_lbboxes2, batch_lbboxes3,
                               batch_lbboxes4, batch_lbboxes5,
                               batch_lbboxes6, batch_lbboxes7 )

        thread31 = pool.submit(self.add_numpy, batch_mask0, batch_mask1, batch_mask2, batch_mask3, batch_mask4,
                               batch_mask5, batch_mask6, batch_mask7)

        # start threads to add numpy array

        batch_image = thread24.result()

        batch_mask = thread31.result()

        batch_label_sbbox = thread25.result()

        batch_label_mbbox = thread26.result()

        batch_label_lbbox = thread27.result()

        batch_sbboxes = thread28.result()

        batch_mbboxes = thread29.result()

        batch_lbboxes = thread30.result()

        pool.shutdown()
        return batch_image, batch_mask, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
               batch_sbboxes, batch_mbboxes, batch_lbboxes

    def add_numpy(self, a1, a2, a3, a4, a5, a6, a7, a8):
        return ne.evaluate("a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8")


    def get_data(self,num,batch_count):

        with tf.device('/cpu:0'):
            self.train_input_size = self.train_input_sizes
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.time_steps, self.train_input_size, self.train_input_size, 3))

            batch_mask = np.zeros((self.batch_size, self.time_steps, self.train_input_size, self.train_input_size, 1))

            batch_label_sbbox = np.zeros((self.batch_size, self.time_steps, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.time_steps, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.time_steps, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.time_steps, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.time_steps, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.time_steps, self.max_bbox_per_scale, 4))

            num = num

            random_v = {}
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]

                    elem = annotation.split(',')

                    random_v['random_flip'] = random.random()
                    random_v['random_crop'] = random.random()
                    random_v['random_translate'] = random.random()

                    previous_id_first = 0

                    previous_id = 0
                    for time_step in range(self.time_steps+1):

                        image = elem[previous_id + 0]
                        mask = elem[previous_id + 1]
                        # image = cv.imread(image)
                        # mask = cv.imread(mask)
                        # cv.namedWindow('image_source', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                        # cv2.imshow('image_source', image)
                        # cv.namedWindow('mask_source', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
                        # cv2.imshow('mask_source', mask/100)
                        # cv2.waitKey(1)
                        num_box = int(elem[previous_id + 2])
                        boxes = ','.join(elem[previous_id + 3: previous_id + 3 + int(num_box)])

                        # print (image, mask)

                        if previous_id>0:

                            sub_annotation = ','.join([image, mask, boxes])
                            print(sub_annotation)
                            # previous_id = previous_id + 3 + int(num_box)

                        # sub_annotation = ','.join([image, mask, boxes])
                        # try:
                            image, mask, bboxes = self.parse_annotation(sub_annotation, random_v)
                            # except:
                            #     continue
                            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                            batch_image[num, time_step-1, :, :, :] = image
                            batch_mask[num, time_step-1, :, :, :] = mask[:, :, np.newaxis]
                            batch_label_sbbox[num, time_step-1, :, :, :, :] = label_sbbox
                            batch_label_mbbox[num, time_step-1, :, :, :, :] = label_mbbox
                            batch_label_lbbox[num, time_step-1, :, :, :, :] = label_lbbox
                            batch_sbboxes[num, time_step-1, :, :] = sbboxes
                            batch_mbboxes[num, time_step-1, :, :] = mbboxes
                            batch_lbboxes[num, time_step-1, :, :] = lbboxes

                        previous_id = previous_id + 3 + int(num_box)
                    num += 2
                self.batch_count += batch_count
                return batch_image, batch_mask, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, mask, bboxes, random_v):

        if random_v['random_flip'] < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            mask = mask[:, ::-1]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, mask, bboxes

    def random_crop(self, image, mask, bboxes, random_v):

        if random_v['random_crop'] < 0.5:
            h, w, _ = image.shape
            # max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            #
            # max_l_trans = max_bbox[0]
            # max_u_trans = max_bbox[1]
            # max_r_trans = w - max_bbox[2]
            # max_d_trans = h - max_bbox[3]
            #
            # crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            # crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            # crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            # crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            crop_xmin, crop_ymin, crop_xmax, crop_ymax = random_v['random_crop_minxy_maxxy']
            print('crop_xmin, crop_ymin, crop_xmax, crop_ymax: ', crop_xmin, crop_ymin, crop_xmax, crop_ymax)

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            mask = mask[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            cv2.imshow('image_preporcess', image)
            cv2.imshow('mask_preprocess', mask / 100)
            # print 'mask_preprocess value: {}'.format(np.unique(mask))
            cv2.waitKey(0)



            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, mask, bboxes

    def random_translate(self, image, mask, bboxes, random_v):

        if random_v['random_translate'] < 0.5:
            h, w, _ = image.shape
            # max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            #
            # max_l_trans = max_bbox[0]
            # max_u_trans = max_bbox[1]
            # max_r_trans = w - max_bbox[2]
            # max_d_trans = h - max_bbox[3]
            #
            # tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            # ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
            tx, ty = random_v['random_translate_xy']
            # print('tx, ty: ', tx, ty)

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, mask, bboxes

    def parse_annotation(self, annotation, random_v):
        line = annotation.strip(',').split(',')
        image_path = line[0]
        mask_path = line[1]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        if not os.path.exists(mask_path):
            raise KeyError("%s does not exist ... " %mask_path)

        image = np.array(cv2.imread(image_path))
        mask = np.array(cv2.imread(mask_path, 0)) * 128.0

        # cv.namedWindow('image_source', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv2.imshow('image_source', image)
        # cv.namedWindow('mask_source', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv2.imshow('mask_source', mask)
        # cv2.waitKey(1)

        bboxes = np.array([list(map(int, box.split())) for box in line[2:]])

        # print 'source mask value: {}'.format(np.unique(mask))

        if self.data_aug:
            image, mask, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(mask), np.copy(bboxes), random_v)
            # image, mask, bboxes = self.random_crop(np.copy(image), np.copy(mask), np.copy(bboxes), random_v)
            # image, mask, bboxes = self.random_translate(np.copy(image), np.copy(mask), np.copy(bboxes), random_v)

        # cv2.imshow('image', image)
        # cv2.imshow('mask', mask)
        # cv.waitKey(0)
        # print 'mask value: {}'.format(np.unique(mask))

        image, mask, bboxes = utils.image_preporcess(np.copy(image), np.copy(mask), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        #
        # cv.namedWindow('image_preporcess', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv2.imshow('image_preporcess', image)
        # cv.namedWindow('mask_preprocess', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        # cv2.imshow('mask_preprocess', mask / 100)
        # print 'mask_preprocess value: {}'.format(np.unique(mask))
        # cv2.waitKey(0)

        return image, mask, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        # print [[self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale, 5 + self.num_classes] for i in range(3)]

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
        
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
            
            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)
                    yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)
                yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)

                # print 'label shape is: {}'.format(np.array(label).shape)
                # print best_detect, yind, xind, best_anchor
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




