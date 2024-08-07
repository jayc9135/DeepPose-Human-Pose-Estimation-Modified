# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:46:18 2020

@author: MrHossein
"""

from scipy.io import loadmat
from skimage import io

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import numpy as np
import imgaug.augmenters as iaa

import os
import warnings

warnings.filterwarnings('ignore')


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]


def update_joint_coordinate(joint_coord, orig_img, out_size, mode='resize'):
    # print(f"joint_coord shape: {joint_coord.shape}")  # Debugging print
    # print(f"orig_img shape: {orig_img.shape}")  # Debugging print
    #
    # if joint_coord.shape[1] != 14:
    #     raise ValueError(
    #         f"Expected joint_coord shape (2, 14), but got {joint_coord.shape}")

    # Og from here
    joints = np.zeros((2, 14))

    if mode == 'resize':
        joints[0] = joint_coord[0] * (out_size[0] / float(orig_img.shape[1]))
        joints[1] = joint_coord[1] * (out_size[1] / float(orig_img.shape[0]))

    return joints


def prepare_joints(dataset_path, train_data, valid_data, test_data, img_size):
    joint_path = os.path.join(dataset_path, 'joints.mat')
    # Read Joints Coordination
    joint_content = loadmat(joint_path)
    joints_coord = joint_content['joints']
    # print(">>>>>>>>>>>>", joints_coord.shape)
    # joints_coord = joints_coord.transpose(2, 0, 1)
    # print(">>>>>>>>>>>>", joints_coord.shape)
    # joints_coord = joints_coord[:, :2, :]
    # print(">>>>>>>>>>>>", joints_coord.shape)

    joints_coord = joints_coord.transpose(2, 1, 0)
    joints_coord = joints_coord[:, :2, :]
    # print(">>>>>>>>>>>>", joints_coord.shape)

    num_samples = joints_coord.shape[0]
    joints = torch.zeros((num_samples, 28))
    joints_2d = torch.zeros((num_samples, 2, 14))

    # # for Train Joints
    # for batch_idx, (images, label) in enumerate(train_data):
    #     for i in range(len(label[0])):
    #         img = io.imread(label[0][i])
    #         coord_index = int(label[0][i][-8:-4]) - 1
    #         scaled_joint = update_joint_coordinate(joints_coord[coord_index],
    #                                                img,
    #                                                img_size,
    #                                                mode='resize')
    #         joints_2d[coord_index] = torch.tensor(scaled_joint)
    #         scaled_joint = coordinate_normalize(scaled_joint)
    #         for j in range(14):
    #             joints[coord_index][2 * j] = scaled_joint[0][j]
    #             joints[coord_index][2 * j + 1] = scaled_joint[1][j]
    #
    # # for Valid Joints
    # for batch_idx, (images, label) in enumerate(valid_data):
    #     for i in range(len(label[0])):
    #         img = io.imread(label[0][i])
    #         scaled_joint = update_joint_coordinate(
    #             joints_coord[int(label[0][i][-8:-4]) - 1], img, img_size,
    #             mode='resize')
    #         joints_2d[int(label[0][i][-8:-4]) - 1] = torch.tensor(scaled_joint)
    #         scaled_joint = coordinate_normalize(scaled_joint)
    #         for j in range(14):
    #             joints[int(label[0][i][-8:-4]) - 1][2 * j] = scaled_joint[0][j]
    #             joints[int(label[0][i][-8:-4]) - 1][2 * j + 1] = \
    #             scaled_joint[1][j]
    #
    # # for Test Joints
    # for batch_idx, (images, label) in enumerate(test_data):
    #     for i in range(len(label[0])):
    #         img = io.imread(label[0][i])
    #         scaled_joint = update_joint_coordinate(
    #             joints_coord[int(label[0][i][-8:-4]) - 1], img, img_size,
    #             mode='resize')
    #         joints_2d[int(label[0][i][-8:-4]) - 1] = torch.tensor(scaled_joint)
    #         scaled_joint = coordinate_normalize(scaled_joint)
    #         for j in range(14):
    #             joints[int(label[0][i][-8:-4]) - 1][2 * j] = scaled_joint[0][j]
    #             joints[int(label[0][i][-8:-4]) - 1][2 * j + 1] = \
    #             scaled_joint[1][j]

    def process_data(data, true_joints, true_joints_2d):
        for batch_idx, (images, label) in enumerate(data):
            for i in range(len(label[0])):
                img = io.imread(label[0][i])
                coord_index = int(label[0][i][-8:-4]) - 1
                scaled_joint = update_joint_coordinate(
                    joints_coord[coord_index, :2, :], img, img_size,
                    mode='resize')
                true_joints_2d[coord_index] = torch.tensor(scaled_joint)
                scaled_joint = coordinate_normalize(scaled_joint)
                for j in range(14):
                    true_joints[coord_index][2 * j] = scaled_joint[0][j]
                    true_joints[coord_index][2 * j + 1] = scaled_joint[1][j]

    process_data(train_data, joints, joints_2d)
    process_data(valid_data, joints, joints_2d)
    process_data(test_data, joints, joints_2d)

    return joints, joints_2d


def coordinate_normalize(joints):
    Bh = 220
    Bw = 220
    joints[0] /= Bw
    joints[1] /= Bh
    joints -= 0.5
    return joints


def Inverse_coordinate_normalize(joints):
    joint_2d = torch.zeros((2, 14))
    for i in range(14):
        joint_2d[0][i] = joints[2 * i]
        joint_2d[1][i] = joints[2 * i + 1]

    Bh = 220
    Bw = 220
    joint_2d += 0.5
    joint_2d[0] *= Bw
    joint_2d[1] *= Bh

    joint = torch.zeros((1, 28))
    for j in range(14):
        joint[0][2 * j] = joint_2d[0][j]
        joint[0][2 * j + 1] = joint_2d[1][j]

    return joint


def prepare_data(args, dataset_path, train_ratio, valid_ratio, test_ratio,
                 image_size=220):
    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.452, 0.445, 0.379], [0.216, 0.201, 0.203])
    ])

    data = MyImageFolder(os.path.join(dataset_path, ''),
                         transform=data_transform)
    train_set, valid_set, test_set = random_split(data, [
        int(len(data) * train_ratio), int(len(data) * valid_ratio),
        int(len(data) * test_ratio)])
    train_data_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                                   **args)
    valid_data_loader = DataLoader(valid_set, batch_size=128, shuffle=True,
                                   **args)
    test_data_loader = DataLoader(test_set, batch_size=128, shuffle=True,
                                  **args)

    true_joints, true_joints_2d = prepare_joints(dataset_path,
                                                 train_data_loader,
                                                 valid_data_loader,
                                                 test_data_loader,
                                                 (image_size, image_size))

    return train_data_loader, valid_data_loader, test_data_loader, true_joints, true_joints_2d


def train_data_augmentation(train_data, true_joints, true_joints_2d,
                            batch_size):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        sometimes(iaa.Affine(rotate=(-25, 25))),
        sometimes(iaa.Affine(scale={"x": (0.7, 1.1), "y": (0.7, 1.1)})),
        sometimes(iaa.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})),
        sometimes(iaa.Affine(shear=(-5, 5)))
    ])

    true_joints_aug = torch.zeros((4000, 28))
    true_joints_2d_aug = torch.zeros((4000, 2, 14))

    true_joints_aug[0:2000] = true_joints
    true_joints_2d_aug[0:2000] = true_joints_2d

    train_data_aug = []
    train_label_aug = []

    # Save Original Image and Labels
    for batch_idx, (images, label) in enumerate(train_data):
        for i in range(len(label[0])):
            train_label_aug.append(int(label[0][i][-8:-4]) - 1)
            train_data_aug.append(images[0][i])

    # Save Augmentation Image and Labels
    start_index = 2000
    train_label = []
    end_index = 0
    index = 0
    data_len = len(train_label_aug)

    for batch_idx, (images, label) in enumerate(train_data):
        for i in range(len(label[0])):
            train_label.append(int(label[0][i][-8:-4]) - 1)
            train_label_aug.append((int(label[0][i][-8:-4]) - 1) + start_index)

        end_index += len(images[0])
        target = true_joints_2d[
            train_label[(batch_idx * batch_size):end_index]]

        # Start for Aug,entation
        batch_image = np.transpose(images[0].numpy(), (0, 2, 3, 1)).reshape(
            len(images[0]), 1, 220, 220, 3)
        joint = np.zeros((len(images[0]), 1, 14, 2))
        for i in range(len(images[0])):
            for j in range(14):
                joint[i][0][j][0] = target[i][0][j]
                joint[i][0][j][1] = target[i][1][j]

        image_aug = np.zeros((len(images[0]), 220, 220, 3))
        kps_aug = np.zeros((len(images[0]), 1, 14, 2))
        for i in range(len(batch_image)):
            image_aug[i], kps_aug[i] = seq(images=batch_image[i],
                                           keypoints=joint[i])

        # Save Image Augmentation
        for i in range(len(image_aug)):
            train_data_aug.append(torch.tensor(
                np.transpose(np.float32(image_aug[i]), (2, 0, 1))))

        # Save Joint Labels and Labels
        temp_joint = torch.zeros((2, 14))
        for i in range(len(kps_aug)):
            for j in range(14):
                temp_joint[0][j] = kps_aug[i][0][j][0]
                temp_joint[1][j] = kps_aug[i][0][j][1]

            true_joints_2d_aug[
                train_label_aug[data_len + i + index]] = temp_joint
            temp_2d = coordinate_normalize(temp_joint)
            for j in range(14):
                true_joints_aug[train_label_aug[data_len + i + index]][2 * j] = \
                temp_2d[0][j]
                true_joints_aug[train_label_aug[data_len + i + index]][
                    2 * j + 1] = temp_2d[1][j]

        index += len(images[0])

    return train_data_aug, train_label_aug, true_joints_aug, true_joints_2d_aug
