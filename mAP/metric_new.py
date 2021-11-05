import glob
import os

class_dict = {'be_corrected':0,
              'to_add':1,
              'to_del':2}
# class_dict = {'traffic_light':0}

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

def obtain_label(file_dir, file_suffix):
    """
    """
    imageid_elem = {}
    target_file = os.path.join(file_dir, file_suffix)
    file_names = glob.glob(target_file)
    for per_file in file_names:
        read_data(imageid_elem, per_file)

    return imageid_elem

def str_parse_gt(gt_str_list):
    bbox_info_list = []
    for box_str in gt_str_list:
        box_elem_str_list = box_str.split(' ')
        box_elem_str_list = [elem for elem in box_elem_str_list if len(elem)>1]
        class_str = box_elem_str_list[0]
        bbox = [int(elem) for elem in box_elem_str_list[1:]]
        class_num = class_dict[class_str]
        tmp_box_info={'class':class_num,'box':bbox}
        bbox_info_list.append(tmp_box_info)

    return bbox_info_list

def str_parse_pred(pred_str_list):
    bbox_info_list = []
    for box_str in pred_str_list:
        box_elem_str_list = box_str.split(' ')
        box_elem_str_list = [elem for elem in box_elem_str_list if len(elem) > 1]
        class_str = box_elem_str_list[0]
        bbox = [int(elem) for elem in box_elem_str_list[2:]]
        if len(bbox)<4:
            return None
        class_num = class_dict[class_str]
        tmp_box_info = {'class': class_num, 'box': bbox}
        bbox_info_list.append(tmp_box_info)

    return bbox_info_list

def get_info_dict(gt_info, pre_info):
    metric_info_dict = {}
    for key in gt_info.keys():
        ground_truth = None
        pred = None
        if key in gt_info.keys():
            ground_truth = str_parse_gt(gt_info[key])


        if key  in pre_info.keys():
            pred = str_parse_pred(pre_info[key])

        if (ground_truth is not None) and (pred is not None):
            tmp_dict = {key:{'gt':ground_truth,
                             'pred':pred}}
            metric_info_dict.update(tmp_dict)
    return metric_info_dict


def compute_iou(box1, box2,expand=True):
    try:
        width = abs(box1[0] - box1[2])
        height = abs(box1[1] - box1[3])
        if expand == True:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0]- 1 * width, box1[1]- 1 * height, box1[2]+ 1 * width, box1[3] + 1 * height
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0]- 1 * width, box2[1]- 1 * height, box2[2]+1 * width, box2[3]+ 1 * height
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    except:
        pass

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)



    width = max(inter_rect_x2 - inter_rect_x1 + 1,0)
    high = max(inter_rect_y2 - inter_rect_y1 + 1,0)
    inter_area = width * high

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
#
# def compute_recal(metric_info_dict):
#     """
#     metric_info_dict:{'name.txt':{'gt':[],'pred':[]}}
#     """
#
#     pred_count = 0
#     correct_count= 0
#
#     be_corrected = 0
#     be_corrected_count = 0
#
#     to_add = 0
#     to_add_count=0
#
#     to_del = 0
#     to_del_count = 0
#     for key in metric_info_dict.keys():
#         pair_info = metric_info_dict[key]
#         groundtruth = pair_info['gt']
#         prediction = pair_info['pred']
#         for pred_box_dict in prediction:
#             pred_count+=1
#             pred_class = pred_box_dict['class']
#
#             if pred_class == 0:
#                 be_corrected += 1
#             if pred_class == 1:
#                 to_add += 1
#             if pred_class == 2:
#                 to_del += 1
#
#             for gt_box_dict in groundtruth:
#                 gt_class = gt_box_dict['class']
#                 if gt_class == pred_class:
#                     iou = compute_iou(gt_box_dict['box'],pred_box_dict['box'])
#                     if iou >0.01:
#                         correct_count+=1
#                         if gt_class==0:
#                             be_corrected_count+=1
#                             break
#                         if gt_class == 1:
#                             to_add_count += 1
#                             break
#                         if gt_class == 2:
#                             to_del_count+=1
#                             break
#
#
#     recall = correct_count / pred_count
#     recall_be_correct = be_corrected_count / be_corrected
#     recall_to_add = to_add_count / to_add
#     recall_to_del = to_del_count / to_del
#     print(recall,recall_be_correct,recall_to_add,recall_to_del,correct_count)

def compute_recall_precision(metric_info_dict):
    """
    metric_info_dict:{'name.txt':{'gt':[],'pred':[]}}
    """

    gt_count = 0
    correct_count= 0

    be_corrected = 0
    be_corrected_count = 0

    to_add = 0
    to_add_count=0

    to_del = 0
    to_del_count = 0
    for key in metric_info_dict.keys():
        pair_info = metric_info_dict[key]
        groundtruth = pair_info['gt']
        prediction = pair_info['pred']
        for gt_box_dict in groundtruth:
            gt_count+=1
            gt_class = gt_box_dict['class']

            if gt_class == 0:
                be_corrected += 1
            if gt_class == 1:
                to_add += 1
            if gt_class == 2:
                to_del += 1

            for pred_box_dict in prediction:
                pred_class = pred_box_dict['class']
                # if True:
                if gt_class == pred_class:
                    iou = compute_iou(gt_box_dict['box'],pred_box_dict['box'],expand=True)
                    if iou >0.3:

                        if gt_class==0:
                            be_corrected_count+=1
                            correct_count += 1
                            break
                        if gt_class == 1:
                            to_add_count += 1
                            correct_count += 1
                            break
                        if gt_class == 2:
                            to_del_count+=1
                            correct_count += 1
                            break


    recall            = correct_count/gt_count
    recall_be_correct =  be_corrected_count/be_corrected
    recall_to_add     = to_add_count/to_add
    recall_to_del     = to_del_count/to_del



    total_num = 0
    pred_be_corrected_num = 0
    pred_to_add_num = 0
    pred_to_del_num = 0
    for pred_key in metric_info_dict.keys():
        pred_info = metric_info_dict[pred_key]['pred']
        for bbox in pred_info:
            box_class = bbox['class']
            if box_class == 0:
                pred_be_corrected_num += 1
            if box_class == 1:
                pred_to_add_num += 1
            if box_class == 2:
                pred_to_del_num += 1

    total_num = pred_be_corrected_num + pred_to_add_num +pred_to_del_num

    precision             = correct_count/total_num
    precision_be_correct  =  be_corrected_count/pred_be_corrected_num
    precision_to_add      = to_add_count/pred_to_add_num
    precision_to_del      = to_del_count/pred_to_del_num
    print('total_recall    :{:.2f}%({}/{}), recall_be_correct    :{:.2f}%({}/{}), recall_to_add    :{:.2f}%({}/{}),   recall_to_del   :{:.2f}%({}/{})'.format(recall*100,  correct_count,gt_count,recall_be_correct*100,be_corrected_count,be_corrected, recall_to_add*100, to_add_count,to_add,recall_to_del*100,to_del_count,to_del))
    print('total_precision :{:.2f}%({}/{}), precision_be_correct :{:.2f}%({}/{}), precision_to_add :{:.2f}%({}/{}),  precision_to_del :{:.2f}%({}/{})'.format(precision*100,correct_count,total_num,precision_be_correct*100, be_corrected_count,pred_be_corrected_num,precision_to_add*100,to_add_count,pred_to_add_num,precision_to_del*100,to_del_count,pred_to_del_num))
    # print(
    #     'total_recall    :{:.2f}%({}/{}), recall_be_correct    :{:.2f}%({}/{})'.format(
    #         recall * 100, correct_count, gt_count, recall_be_correct * 100, be_corrected_count, be_corrected,
    #         ))
    # print(
    #     'total_precision :{:.2f}%({}/{}), precision_be_correct :{:.2f}%({}/{})'.format(
    #         precision * 100, correct_count, total_num, precision_be_correct * 100, be_corrected_count,
    #         pred_be_corrected_num, ))


def main():
    gt_info = obtain_label('./ground-truth', '*.txt')
    pre_info = obtain_label('./predicted', '*.txt')
    metric_info_dict = get_info_dict(gt_info,pre_info)
    precision= compute_recall_precision(metric_info_dict)
    print(0)
    # recall= compute_recal(metric_info_dict)



if __name__ == '__main__':
    main()
