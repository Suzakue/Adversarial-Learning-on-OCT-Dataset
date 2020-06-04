import os.path as osp
import cv2

# path = './dataset/VOC2012/SegmentationClassAug'
# image_list_path = './dataset/voc_list/train_aug.txt'
#
# image_list = [i_id.strip() for i_id in open(image_list_path)]
#
# labels = dict()
# for i in range(5):
#     label_file = osp.join(path, "%s.png" % image_list[i])
#     label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
#     print(label.shape)
#     for h in label:
#         for g in h:
#             labels[g] = None
#     print(labels.keys())

path = '/home/hejiawen/pytorch/ZRRP/AI-OCT/data/cutted_data/Edema_validationset/label_images' \
       '/PC011_MacularCube512x128_8-26-2013_17-4-6_OS_sn8547_cube_z_labelMark/53.bmp'
label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
labels = dict()
for h in label:
    for g in h:
        labels[g] = None

print(label.shape)
print(labels.keys())

