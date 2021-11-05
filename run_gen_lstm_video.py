import argparse
import os
import numpy
from data_dreaming.code.save_file import load_info_from_json, store_info_to_json
# from data_dreaming.code.visualize import new_simulation_function

def generation():
    ## Read json file
    #读取video json数据。
    json_file_dict=load_info_from_json('/media/wangning/Elements/convlstm/generation/total_video.json')
    # old_file_dict=load_info_from_json('D:/端到端更图代码和文件/generation/train_version2.json')
    total_sequence_list = []
    # task_info = json_file_dict['MKZ118_20201215104203']
    # 遍历每个task_id
    for task_key in json_file_dict.keys():
        if task_key not in ['MKZ156_20201215112828', 'MKZ073_20201216225330', 'HQEV503_20201215091205']:
            task_info = json_file_dict[task_key]
            # 遍历每个video
            for video_id in task_info.keys():
                video_info = task_info[video_id]
                # 计算整个video中总帧数
                seq_num = len(video_info)
                start_id = 0
                end_id = seq_num-8
                for i in range(start_id,end_id):
                    sequence_list = video_info[i:i+4]
                    # 将整个序列分拆成8帧的短序列
                    total_sequence_list.append(sequence_list)

    # 存放txt文件路径
    txt_path = '/media/wangning/Elements/convlstm/lstm_train_s4.txt'
    with open(txt_path, 'w') as lstm_label:
        # 遍历每个短序列
        for video_seq in total_sequence_list:
            total_line_str = ''
            for image_info in video_seq:
                # 获得每帧的信息
                # image_path = image_info['img_path'].split('data')[1]
                # mask_path = image_info['msk_path'].split('data')[1]
                image_path = image_info['img_path']
                mask_path = image_info['msk_path']
                be_correct = image_info['be_correct']
                to_add = image_info['to_add']
                to_del = image_info['to_del']
                # 对图像中总的box数进行计算

                box_num = 0

                line_str = ' '

                if len(be_correct) > 0:
                    for box_str in be_correct:
                        if len(box_str) >len('None'):
                            box_str = box_str[1:-1].split()
                            local_str = ','.join([box_str[0],box_str[1],box_str[2],box_str[3],'0'])
                            line_str = line_str+ ' ' + local_str
                            box_num+=1

                if len(to_add) > len('None'):
                    box_str = to_add[1:-1].split()
                    to_add_str = ','.join([box_str[0], box_str[1], box_str[2], box_str[3], '1'])
                    line_str = line_str + ' ' + to_add_str
                    box_num += 1

                if len(to_del) > len('None'):
                    box_str_del = to_del[1:-1].split()
                    to_del_str_0 = ','.join([box_str_del[0], box_str_del[1], box_str_del[2], box_str_del[3], '2'])
                    line_str = line_str +' ' + to_del_str_0
                    box_num += 1

                line_str = str(box_num)+line_str+' '
                line_str = image_path+' '+' '+mask_path+' ' + line_str
                total_line_str = total_line_str+line_str
                # print('debug')


            lstm_label.write(total_line_str + '\n')


            # print('stop')
    # store_info_to_json(total_sequence_list, '/home/wangning/Desktop/data/yizhuang/generation/lstm_train_data_version_1.json')




    print('stop')


if __name__ == '__main__':

    generation()