import os
import shutil
import cv2


def check_the_simu_3d(root_data_path):
    folder_list = os.listdir(os.path.join(root_data_path,'YiZhuangDaLuWang'))
    with open('/home/wangning/Desktop/data/New_yizhuang/labels_3d.txt','r') as f:
        label_path_list=f.readlines()


    for path in label_path_list:
        absolute_path = os.path.join(root_data_path,path).split('\n')[0]
        with open(absolute_path, 'r') as f:
            images_info = f.readlines()
        for img_info in images_info[:-2]:
            image_info_list = img_info.split(',')
            img_path,msk_path = image_info_list[0],image_info_list[1]
            image_abs_path = os.path.join('/home/wangning/Desktop/data',img_path)
            img_data = cv2.imread(image_abs_path,1)

            for bbox_info in image_info_list[2:-1]:
                elem_list =[int(n) for n in bbox_info.split(' ')]

                bbox = [elem_list[0],elem_list[1],elem_list[2],elem_list[3]]
                class_id = elem_list[4]
                light_id = elem_list[5]
                color_name = (0,255,0)
                if class_id ==0:
                    color_name = (255,0,0)
                elif class_id==1:
                    color_name=(0,255,0)
                elif class_id ==2:
                    color_name=(0,0,255)


                cv2.rectangle(img_data,(bbox[0],bbox[1]),(bbox[2],bbox[3]),color_name,-1)
                cv2.putText(img_data, str(light_id), (bbox[2],bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_name, 2)

            cv2.imshow('image',img_data)
            cv2.waitKey(1)






if __name__ == '__main__':
    root_data_path = '/home/wangning/Desktop/data/New_yizhuang'
    check_the_simu_3d(root_data_path)