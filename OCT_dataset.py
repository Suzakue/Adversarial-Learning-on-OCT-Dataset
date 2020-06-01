import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image


LABEL_LIST_PATH = '/home/hejiawen/pytorch/ZRRP/AI-OCT/data/cutted_data_path/val_labels.txt'

class VOCDataSet(data.Dataset):
    def __init__(self, root, image_path, label_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.image_path = image_path
        self.label_path = label_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(image_path)]
        self.label_ids = [i_id.strip() for i_id in open(label_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for idx, name in enumerate(self.img_ids):
            img_file = osp.join(self.root, name)
            label_file = osp.join(self.root, self.label_ids[idx])
            # OCT dataset 切图
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 1})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 2})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 3})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 4})

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # OCT dataset 切图
        if datafiles['num'] == 1:
            image = image[0:256, 0:256]
            label = label[0:256, 0:256]
        elif datafiles['num'] == 2:
            image = image[0:256, 256:512]
            label = label[0:256, 256:512]
        elif datafiles['num'] == 3:
            image = image[256:512, 0:256]
            label = label[256:512, 0:256]
        elif datafiles['num'] == 4:
            image = image[256:512, 256:512]
            label = label[256:512, 256:512]
        else:
            print('datafile.num error!!!!!!!!')

        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name, datafiles['num']


class VOCGTDataSet(data.Dataset):
    def __init__(self, root, image_path, label_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.image_path = image_path
        self.label_path = label_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(image_path)]
        self.label_ids = [i_id.strip() for i_id in open(label_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for idx, name in enumerate(self.img_ids):
            img_file = osp.join(self.root, name)
            label_file = osp.join(self.root, self.label_ids[idx])

            # OCT dataset 切图
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 1})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 2})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 3})
            self.files.append({"img": img_file, "label": label_file, "name": name, "num": 4})

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # OCT dataset 切图
        if datafiles['num'] == 1:
            image = image[0:256, 0:256]
            label = label[0:256, 0:256]
        elif datafiles['num'] == 2:
            image = image[0:256, 256:512]
            label = label[0:256, 256:512]
        elif datafiles['num'] == 3:
            image = image[256:512, 0:256]
            label = label[256:512, 0:256]
        elif datafiles['num'] == 4:
            image = image[256:512, 256:512]
            label = label[256:512, 256:512]
        else:
            print('datafile.num error!!!!!!!!')

        size = image.shape
        name = datafiles["name"]

        attempt = 0
        while attempt < 10 :
            if self.scale:
                image, label = self.generate_scale_label(image, label)

            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                attempt += 1
                continue
            else:
                break

        if attempt == 10:
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(image[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        label = np.asarray(label[h_off: h_off+self.crop_h, w_off: w_off+self.crop_w], np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name, datafiles['num']


if __name__ == '__main__':
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
