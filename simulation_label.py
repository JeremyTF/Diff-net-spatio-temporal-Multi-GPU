#! /usr/bin/env python
# coding=utf-8
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
@file simulation_label.py
@author helei07(@baidu.com)
@date 2020/10/14
"""

import os
import random

import numpy as np
import cv2


def read_label(label_file):
    """
    """
    imageid_label = {}
    with open(label_file, 'r') as label:
        lines = label.readlines()
        for line in lines:
            line = line.strip('\n')
            line_elem = line.split(' ')
            image_id = line_elem[0]
            label = line_elem[1:]
            imageid_label[image_id] = label

    return imageid_label


def simu_less_label(label):
    """
    """
    label_num = len(label)
    id_ll = random.randint(0, label_num - 1)
    tmp = label[id_ll]
    res = ','.join(tmp.split(',')[0:4]) + ',2'
    label[id_ll] = res

    return 0


def show_image(mask, image_id):
    """
    """
    cv2.namedWindow('valid_mask', 1)
    cv2.imshow('valid_mask', mask * 255)
    cv2.waitKey(1)

    cv2.imwrite('masks_vis/' + image_id, \
        mask * 255) 

    return 0


def write_image(mask, image_id):
    """
    """
    cv2.imwrite('simu_del_center/' + image_id, \
        mask * 255) 
    
    return 0


def obtain_box_info(label, image_id):
    """
    """
    mask_size = 1080, 1920
    mask = np.ones(mask_size, dtype=np.uint8)
    mask[360:1080, 0:1920] = 0
    mask[0:300, 0:1920] = 0
    mask[0:1080, 0:300] = 0
    mask[0:1080, 1620:1920] = 0

    box_num = len(label)
    height = []
    width = []

    for box in label:
        box_elem = [float(x) for x in box.split(',')]
        width.append(abs(box_elem[2] - box_elem[0]))
        height.append(abs(box_elem[3] - box_elem[1]))
        center_x = abs(box_elem[2] + box_elem[0]) * 0.5
        center_y = abs(box_elem[3] + box_elem[1]) * 0.5
        x_min = int(center_x - 135)
        x_max = int(center_x + 135)
        y_min = max(int(center_y - 235), 0)
        y_max = min(int(center_y + 235), 1080)

        mask[y_min:y_max, x_min:x_max] = 0

    write_image(mask, image_id)

    candidate_center = []
    for i in range(1080):
        for j in range(1920):
            if mask[i, j] == 1:
                candidate_center.append([j, i])

    # print len(candidate_center)
    random_num = random.randint(0, len(candidate_center))
    candidate_center_res = candidate_center[random_num]

    candidate_width = np.int64(np.mean(width))
    candidate_height = np.int64(np.mean(height))
    # print candidate_height, candidate_width

    res_x_min = int(abs(candidate_center_res[0] - 0.5 * candidate_width))
    res_x_max = int(abs(candidate_center_res[0] + 0.5 * candidate_width))
    res_y_min = int(abs(candidate_center_res[1] - 0.5 * candidate_height))
    res_y_max = int(abs(candidate_center_res[1] + 0.5 * candidate_height))

    return [res_x_min, res_y_min, res_x_max, res_y_max]


def simu_more_label(label, image_id):
    """
    """
    # print label
    ml = obtain_box_info(label, image_id)
    annotation = ','.join([str(x) for x in ml]) + ',1'
    label.append(annotation)
    # print label
    # print 30 * '-'
    
    return 0


def gen_true_mask(image_id, simu_label):
    """
    """

    mask_size = 1080, 1920
    mask = np.zeros(mask_size, dtype=np.uint8)
    label_name = image_id.split('.')[0] + '.png'

    for box in simu_label.split(' '):
        elem = [int(float(x)) for x in box.split(',')]
        
        x_min = min(elem[0], elem[2])
        x_max = max(elem[0], elem[2])
        y_min = min(elem[1], elem[3])
        y_max = max(elem[1], elem[3])

        if (elem[4] == 0 or elem[4] == 1):
            mask[y_min:y_max, x_min:x_max] = 1
        
    cv2.imwrite(os.path.join('masks', label_name), mask)
    show_image(mask, image_id)

    return 0

    
def show_image_label(image_id, simu_label):
    image_path = os.path.join('images', image_id)
    image = cv2.imread(image_path, -1)

    label_char = {'0':'be_corrected', '1':'to_del', '2':'to_add'}
    label_color = {'0':(0, 0, 255), '1':(0, 255, 0), '2':(255, 0, 0)}

    for box in simu_label.split(' '):
        elem = [int(float(x)) for x in box.split(',')]
        
        label_id = str(elem[-1])
        cv2.rectangle(image, (int(elem[0]), int(elem[1])), \
            (int(elem[2]), int(elem[3])), label_color[label_id], 2)
        # print label_id, label_char[label_id], label_color[label_id]

        left = min(elem[0], elem[2])
        top = min(elem[1], elem[3])
        labelSize, baseLine = cv2.getTextSize(label_char[label_id], \
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, labelSize[1])

        cv2.rectangle(image, (left, int(top - round(1.5*labelSize[1]))), \
            (int(left + round(1.5*labelSize[0])), top + int(baseLine)), label_color[label_id], cv2.FILLED)

        cv2.putText(image, label_char[label_id], (left, top), \
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    cv2.namedWindow('label', 1)
    cv2.imshow('label', image)
    cv2.waitKey(1)
    cv2.imwrite('vis/' + image_id, image) 

    return 0


def copy_image():
    """
    """
    # copy image data to specific directory 
    source_image_dir = '/home/lei/end-to-end-hdmap-diff/data/download_tool/MKZ148_20200815150836/source_images'
    for image_id in imageid_label.keys():
        # label = imageid_label[image_id]
        source_image_path = os.path.join(source_image_dir, image_id)
        os.system('cp %s %s'%(source_image_path, 'images'))

    return 0

def write_data(label, file_path):
    """
    """
    base_dir = '/home/lei/end-to-end-hdmap-diff/data/codes'
    with open(file_path, 'w') as file:
        for line in label:
            elem = line.split(' ')
            image_id = os.path.join(base_dir, 'images', elem[0])
            mask_id = os.path.join(base_dir, 'masks', elem[0].split('.')[0] + '.png')
            label_info = ' '.join([image_id, mask_id, ' '.join(elem[1:])])
            print label_info
            file.write(label_info + "\n")

    return 0

def main():
    """
    """
    
    label_file = 'true.txt'

    imageid_label = read_label(label_file)
    # print imageid_label

    # simulation label
    # with open('label.txt', 'a') as simu:

    train_label = []
    test_label = []
    for idx, image_id in enumerate(imageid_label.keys()):

        # if idx > 3:
        #     break
        label = imageid_label[image_id]
        # print label
        simu_less_label(label)

        # print label
        simu_more_label(label, image_id)
        # print label
        simu_label = ' '.join([x for x in label])
        # print simu_label

        show_image_label(image_id, simu_label)

        if idx % 10:
            train_label.append(image_id + ' ' + simu_label)
        else:
            test_label.append(image_id + ' ' + simu_label)

        gen_true_mask(image_id, simu_label)

    write_data(train_label, 'train.txt')
    write_data(test_label, 'test.txt')

    return 0

if __name__ == '__main__':
    main()