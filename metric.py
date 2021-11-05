#! /usr/bin/env python
# coding=utf-8

import os
import glob

def obtain_label(file_dir, file_suffix):
    """
    """
    imageid_elem = {}
    target_file = os.path.join(file_dir, file_suffix)
    file_names = glob.glob(target_file)
    for per_file in file_names:
        read_data(imageid_elem, per_file)

    return imageid_elem


def read_data(imageid_elem, file_path):
    """
    """
    key = file_path.split('/')[-1]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n')
            elem = line.split(' ')
            if key not in imageid_elem.keys():
                imageid_elem[key] = set([])
            imageid_elem[key].add(line)

    return 0

def box_check(box):
    """
    """
    box = [float(x) for x in box]
    x_min = min(box[0], box[2])
    x_max = max(box[0], box[2])
    y_min = min(box[1], box[3])
    y_max = max(box[1], box[3])

    return [x_min, y_min, x_max, y_max]

def info_box(box):
    """
    """
    h = abs(box[3] - box[1])
    w = abs(box[2] - box[0])
    center_y = abs(box[3] + box[1])
    center_x = abs(box[2] + box[0])

    return [center_x, center_y, w, h]

def inter_area(box1_info, box2_info):
    """
    """
    inter_w = box1_info[2] + box2_info[2] - abs(box1_info[0] - box2_info[0])
    inter_h = box1_info[3] + box2_info[3] - abs(box1_info[1] - box2_info[1])

    if (inter_w < 0 and inter_h < 0):
        return None

    return inter_h * inter_w

def iou(box1, box2):
    """
    """
    # print box1, box2
    box1 = box_check(box1)
    box2 = box_check(box2)
    box1_info = info_box(box1)
    box2_info = info_box(box2)

    inter_area_v = inter_area(box1_info, box2_info)
    uion_area = box1_info[2] * box1_info[3] + box2_info[2] * box2_info[3]

    # iou = inter_area_v / uion_area

    if inter_area_v == None:
        return False, 0

    return True, inter_area_v / uion_area

def metric_precision(info1, info2):
    """
    """

    num_all = 0
    num_succ = 0

    num_corrected_all = 0
    num_corrected_succ = 0

    num_del_all = 0
    num_del_succ = 0

    num_add_all = 0
    num_add_succ = 0

    for key in info1.keys():
        num_all = num_all + len(info1[key])
        for elem in info1[key]:
            elem_class = elem.split(' ')[0]
            elem_box = elem.split(' ')[2:]

            if elem_class == 'be_corrected':
                num_corrected_all = num_corrected_all + 1
            elif elem_class == 'to_del':
                num_del_all = num_del_all + 1
            else:
                num_add_all = num_add_all + 1

            if key in info2.keys():
                for candidate_elem in info2[key]:
                    candidate_class = candidate_elem.split(' ')[0]
                    candidate_box = candidate_elem.split(' ')[1:]
                    if elem_class == candidate_class:
                        # print elem_box, candidate_box
                        bool_succ, _ = iou(elem_box, candidate_box)
                        if bool_succ:
                            num_succ = num_succ + 1

                            if elem_class == 'be_corrected':
                                num_corrected_succ = num_corrected_succ + 1
                            elif elem_class == 'to_del':
                                num_del_succ = num_del_succ + 1
                            else:
                                num_add_succ = num_add_succ + 1

                            break

    metric = 1.0 * num_succ / num_all

    print 'num_corrected_all: {}, num_corrected_succ: {}, corrected_precision: {}'.format(num_corrected_all, \
        num_corrected_succ, 1.0 * num_corrected_succ / num_corrected_all)

    print 'num_del_all: {}, num_del_succ: {}, del_precision: {}'.format(num_del_all, \
        num_del_succ, 1.0 * num_del_succ / num_del_all)

    print 'num_add_all: {}, num_add_succ: {}, add_precision: {}'.format(num_add_all, \
        num_add_succ, 1.0 * num_add_succ / num_add_all)

    return metric, num_succ, num_all

def metric_recall(info1, info2):
    """
    """

    num_all = 0
    num_succ = 0

    num_corrected_all = 0
    num_corrected_succ = 0

    num_del_all = 0
    num_del_succ = 0

    num_add_all = 0
    num_add_succ = 0

    for key in info1.keys():
        num_all = num_all + len(info1[key])
        for elem in info1[key]:
            elem_class = elem.split(' ')[0]
            elem_box = elem.split(' ')[1:]

            if elem_class == 'be_corrected':
                num_corrected_all = num_corrected_all + 1
            elif elem_class == 'to_del':
                num_del_all = num_del_all + 1
            else:
                num_add_all = num_add_all + 1

            if key in info2.keys():
                for candidate_elem in info2[key]:
                    candidate_class = candidate_elem.split(' ')[0]
                    candidate_box = candidate_elem.split(' ')[2:]
                    if elem_class == candidate_class:
                        # print elem_box, candidate_box
                        bool_succ, _ = iou(elem_box, candidate_box)
                        if bool_succ:
                            num_succ = num_succ + 1

                            if elem_class == 'be_corrected':
                                num_corrected_succ = num_corrected_succ + 1
                            elif elem_class == 'to_del':
                                num_del_succ = num_del_succ + 1
                            else:
                                num_add_succ = num_add_succ + 1

                            break

    metric = 1.0 * num_succ / num_all

    print 30 * '-'

    print 'num_corrected_gt: {}, num_corrected_succ: {}, corrected_recall: {}'.format(num_corrected_all, \
        num_corrected_succ, 1.0 * num_corrected_succ / num_corrected_all)

    print 'num_del_gt: {}, num_del_succ: {}, del_recall: {}'.format(num_del_all, \
        num_del_succ, 1.0 * num_del_succ / num_del_all)

    print 'num_add_gt: {}, num_add_succ: {}, add_recall: {}'.format(num_add_all, \
        num_add_succ, 1.0 * num_add_succ / num_add_all)

    print 30 * '-'

    return metric, num_succ, num_all

def precision_recall(gt_info, pre_info):
    """
    """
    precision, num_succ_p, num_all_p = metric_precision(pre_info, gt_info)
    recall, num_succ_r, num_all_r = metric_recall(gt_info, pre_info)

    print '#detection succ: {}, #detection all: {}, precision: {}'.format(num_succ_p, num_all_p, precision)
    print '#detection gt: {}, #ground-truth: {}, recall: {}'.format(num_succ_r, num_all_r, recall)

    return [precision, recall]

def main():
    """
    """
    gt_info = obtain_label('mAP/ground-truth', '*.txt')
    pre_info = obtain_label('mAP/predicted', '*.txt')

    precision, recall = precision_recall(gt_info, pre_info)
    # print 'precision: {} recall: {}'.format(precision, recall)

    return 0

if __name__ == '__main__':
    main()