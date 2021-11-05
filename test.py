import cv2
import os

# path = '/home/wangning/Desktop/data/yizhuang/video_masks/MKZ044_20201215132049/66/'
#
# for filename in os.listdir(path):
#     print(filename)
#     img = cv2.imread(path+'/'+filename)
#     cv2.imshow(path+'/'+filename,img*255)
#     cv2.waitKey(0)

txtpath = '/media/wangning/Elements/convlstm/lstm_new_train_copy.txt'
count = 0
with open(txtpath,'r') as f:
    txt = f.readlines()
    for line in txt:
        count +=1
print(count)
