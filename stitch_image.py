import cv2
import numpy as np
import os

temp_dir = 'results_temp/'
list_path = '/home/hejiawen/pytorch/ZRRP/AI-OCT/data/cutted_data_path/val_images.txt'
save_dir = 'results_OCT_0607/'
name_list = list()

with open(list_path, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split('/')
        name = line[-2] + '_' + line[-1]
        name = name[:-4]
        name_list.append(name)

num = 0

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for name in name_list:
    img1 = cv2.imread(temp_dir + name + '_1.png')
    img2 = cv2.imread(temp_dir + name + '_2.png')
    img3 = cv2.imread(temp_dir + name + '_3.png')
    img4 = cv2.imread(temp_dir + name + '_4.png')
    image1 = np.concatenate([img1, img2], axis=1)
    image2 = np.concatenate([img3, img4], axis=1)
    image = np.concatenate([image1, image2], axis=0)
    cv2.imwrite(save_dir + name + '.png', image)
    num += 1
    if num % 100 == 0:
        print(num, 'OK!!!')

