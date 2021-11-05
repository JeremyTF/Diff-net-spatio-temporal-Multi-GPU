import cv2 as cv
import numpy as np

failure_path = './failurecase.txt'
label_path = '/media/jiangshengjie/Elements/convlstm_MC3D/MC3D_train_lstm.txt'
target_task_to_delete = ['MKZ077_20201020132603','MKZ077_20201118113046','MKZ098_20201130082203','MKZ101_20200826110403','MKZ101_20200907120555','MKZ101_20200916105901','MKZ101_20200916145951','MKZ101_20201229114610','MKZ103_20200630132915','MKZ103_20201023155433']
with open(label_path,'r') as f:
    txt = f.readlines()
    txt = [line.replace('New_yizhuang/YiZhuangDaLuWang/', '/media/jiangshengjie/Elements/New_yizhuang_MC3D/home/wangning/Desktop/data/New_yizhuang/YiZhuangDaLuWang/') for line
                in txt]
    annotations = [line.strip().strip(',') for line in txt]
    # annotations = [line.strip() for line in annotations if line.split('/')[6] not in target_task_to_delete]
    np.random.shuffle(annotations)

path_s7 = '/media/jiangshengjie/Elements/convlstm_MC3D/MC3D_lstm_s7.txt'
with open(path_s7,'w') as f:
    for line in annotations:

        elem = line.split(',')

        previous_id = 0
        for time_step in range(8):
            image = elem[previous_id + 0]
            mask = elem[previous_id + 1]
            path_mess = ','.join([image,mask])+','
            num_box = int(elem[previous_id + 2])
            boxes = ','.join(elem[previous_id + 3: previous_id + 3 + int(num_box)])

            if previous_id >0:

            # print (image, mask)
                sub_annotation = ','.join([image, mask, boxes])
                previous_id = previous_id + 3 + int(num_box)
                f.write(sub_annotation)
                f.write(',')
            previous_id = previous_id + 3 + int(num_box)
        f.write('\n')


            # line = sub_annotation.split(',')
            # path = str(line[0])
            # mask = str(line[1])
            # mask = cv.imread(mask)
            # mask = mask*100
            # cv.imshow('mask',mask)
            # cv.waitKey(0)
            # if path.split('/')[6]
            # bboxes= np.array([list(map(int, box.split())) for box in line[2:-1]])
            # img = cv.imread(path)
            # mask = cv.imread(mask)
            # bboxes_gt, classes_gt = box[:, :4], box[:, 4]
            # for box in bboxes:
            #     bbox_coor = box[:4]
            #     width = abs(bbox_coor[0] - bbox_coor[2])
            #     height = abs(bbox_coor[1] - bbox_coor[3])
            #     if width * height >= 60 and bbox_coor[2] >= 1080/3:
            #
            #         classes = box[5]
            #         xmin, ymin, xmax, ymax = list(map(int, bbox_coor))
            #         # box = line[1:-1]
            #         # box1 = box[0:1:1]
            #         cv.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),2)
            #         img = cv.putText(img,str(classes),(xmin,ymin),cv.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 3)
            #     else:
            #         continue
            # # cv.namedWindow('mask', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            # # cv.imshow('mask', mask*100)
            # # cv.waitKey(1)
            # cv.namedWindow('img', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            # cv.imshow('img',img)
            # cv.waitKey(1)
            # previous_id = previous_id + 3 + int(num_box)