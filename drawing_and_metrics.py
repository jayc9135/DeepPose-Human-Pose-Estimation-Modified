# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:50:15 2020

@author: MrHossein
"""

import preparing_data
from torchvision import transforms
from PIL import ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def draw_lines(image, joints_coordinate, name, save=False):
    """
    0.  Right ankle
    1.  Right knee
    2.  Right hip
    3.  Left hip
    4.  Left knee
    5.  Left ankle
    6.  Right wrist
    7.  Right elbow
    8.  Right shoulder
    9.  Left shoulder
    10. Left elbow
    11. Left wrist
    12. Neck
    13. Head top
    """
    left_foot = [(joints_coordinate[0][5], joints_coordinate[1][5]), (joints_coordinate[0][4], joints_coordinate[1][4])]
    right_foot = [(joints_coordinate[0][0], joints_coordinate[1][0]),
                  (joints_coordinate[0][1], joints_coordinate[1][1])]
    left_hip = [(joints_coordinate[0][4], joints_coordinate[1][4]), (joints_coordinate[0][3], joints_coordinate[1][3])]
    right_hip = [(joints_coordinate[0][1], joints_coordinate[1][1]), (joints_coordinate[0][2], joints_coordinate[1][2])]
    left_hand = [(joints_coordinate[0][11], joints_coordinate[1][11]),
                 (joints_coordinate[0][10], joints_coordinate[1][10])]
    right_hand = [(joints_coordinate[0][6], joints_coordinate[1][6]),
                  (joints_coordinate[0][7], joints_coordinate[1][7])]
    left_arm = [(joints_coordinate[0][10], joints_coordinate[1][10]),
                (joints_coordinate[0][9], joints_coordinate[1][9])]
    right_arm = [(joints_coordinate[0][7], joints_coordinate[1][7]), (joints_coordinate[0][8], joints_coordinate[1][8])]
    body = [(joints_coordinate[0][12], joints_coordinate[1][12]), (
    (joints_coordinate[0][3] + joints_coordinate[0][2]) / 2, (joints_coordinate[1][3] + joints_coordinate[1][2]) / 2)]
    head = [(joints_coordinate[0][13], joints_coordinate[1][13]), (joints_coordinate[0][12], joints_coordinate[1][12])]

    d = ImageDraw.Draw(image)
    d.line(left_foot, fill='blue', width=2)
    d.line(right_foot, fill='blue', width=2)
    d.line(left_hip, fill='green', width=2)
    d.line(right_hip, fill='green', width=2)
    d.line(left_hand, fill='red', width=2)
    d.line(right_hand, fill='red', width=2)
    d.line(left_arm, fill='yellow', width=2)
    d.line(right_arm, fill='yellow', width=2)
    d.line(body, fill='brown', width=2)
    d.line(head, fill='pink', width=2)

    plt.imshow(image)
    if save:
        image.save(name)
    plt.show()


def PDJ_metric(predicted_joints, true_joints, limbs_name):
    """
    0.  Right ankle
    1.  Right knee
    2.  Right hip
    3.  Left hip
    4.  Left knee
    5.  Left ankle
    6.  Right wrist
    7.  Right elbow
    8.  Right shoulder
    9.  Left shoulder
    10. Left elbow
    11. Left wrist
    12. Neck
    13. Head top
    """
    # Calculate True Distance of each Limb
    body_distance = np.linalg.norm(true_joints[:, 2] - true_joints[:, 9])
    correct_parts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Claculate Distance between True Joints and Predicted Joints in each Limb
    for i in range(14):
        joint_distance = np.linalg.norm(true_joints[:, i] - predicted_joints[:, i])
        if joint_distance <= (0.2 * body_distance):
            correct_parts[i] = 1

    return correct_parts


def PCP_metric(predicted_joints, true_joints, limbs_name):
    """
    0.  Right ankle
    1.  Right knee
    2.  Right hip
    3.  Left hip
    4.  Left knee
    5.  Left ankle
    6.  Right wrist
    7.  Right elbow
    8.  Right shoulder
    9.  Left shoulder
    10. Left elbow
    11. Left wrist
    12. Neck
    13. Head top
    """
    # Calculate True Distance of each Limb
    true_limb_len = dict()
    true_limb_len[limbs_name[0]] = np.linalg.norm(true_joints[:, 0] - true_joints[:, 1])
    true_limb_len[limbs_name[1]] = np.linalg.norm(true_joints[:, 1] - true_joints[:, 2])
    true_limb_len[limbs_name[2]] = np.linalg.norm(true_joints[:, 3] - true_joints[:, 4])
    true_limb_len[limbs_name[3]] = np.linalg.norm(true_joints[:, 4] - true_joints[:, 5])
    true_limb_len[limbs_name[4]] = np.linalg.norm(true_joints[:, 6] - true_joints[:, 7])
    true_limb_len[limbs_name[5]] = np.linalg.norm(true_joints[:, 7] - true_joints[:, 8])
    true_limb_len[limbs_name[6]] = np.linalg.norm(true_joints[:, 9] - true_joints[:, 10])
    true_limb_len[limbs_name[7]] = np.linalg.norm(true_joints[:, 10] - true_joints[:, 11])

    correct_parts = [0, 0, 0, 0, 0, 0, 0, 0]
    # Claculate Distance between True Joints and Predicted Joints in each Limb
    for i in range(8):
        if i == 2 or i == 3:
            j = i + 1
        elif i == 4 or i == 5:
            j = i + 2
        elif i == 6 or i == 7:
            j = i + 3
        else:
            j = i

        joint_distance1 = np.linalg.norm(true_joints[:, j] - predicted_joints[:, j])
        joint_distance2 = np.linalg.norm(true_joints[:, j + 1] - predicted_joints[:, j + 1])
        if joint_distance1 <= (true_limb_len[limbs_name[i]] / 2) and joint_distance2 <= (
                true_limb_len[limbs_name[i]] / 2):
            correct_parts[i] = 1

    return correct_parts


def correct_percentage(image_label, predicted_joint, true_joint, names, metric='PCP'):
    if (metric == 'PCP'):
        total_correct_percentage = [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        total_correct_percentage = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    joint_temp = np.zeros((len(image_label), 28))
    index = 0
    for i in range(len(predicted_joint)):
        for j in range(len(predicted_joint[i])):
            joint_temp[index] = np.array(
                preparing_data.Inverse_coordinate_normalize(torch.tensor(predicted_joint[i][j])))
            index += 1

    for i in range(len(image_label)):
        orig_joint = true_joint[image_label[i]]

        pred_joint = torch.zeros((2, 14))
        temp = joint_temp[i]
        for i in range(14):
            pred_joint[0][i] = temp[2 * i]
            pred_joint[1][i] = temp[2 * i + 1]

        if (metric == 'PCP'):
            correct_part = PCP_metric(pred_joint, orig_joint, names)
            total_correct_percentage = np.array(total_correct_percentage) + np.array(correct_part)
        else:
            correct_part = PDJ_metric(pred_joint, orig_joint, names)
            total_correct_percentage = np.array(total_correct_percentage) + np.array(correct_part)

    return (np.array(total_correct_percentage) / len(image_label)) * 100.00


def draw_selected_image(image_index, predicted_joint, true_joints, image, label, batch_size, save=False,
                        name1='out_image1.jpg', name2='true_image1.jpg'):
    if image_index > 299:
        image_index = 299

    invers_normalize = transforms.Normalize([-0.452 / 0.216, -0.445 / 0.201, -0.379 / 0.203],
                                            [1 / 0.216, 1 / 0.201, 1 / 0.203])
    r = int(image_index / batch_size)
    index = int(image_index % batch_size)
    trans1 = transforms.ToPILImage()

    input1 = predicted_joint[r][index].cpu()
    input2 = invers_normalize(image[r][index].cpu())
    input3 = label[image_index]

    joint_temp = preparing_data.Inverse_coordinate_normalize(input1).reshape(28)
    pred_joint = torch.zeros((2, 14))
    for i in range(14):
        pred_joint[0][i] = joint_temp[2 * i]
        pred_joint[1][i] = joint_temp[2 * i + 1]

    orig_image = trans1(input2)
    draw_lines(orig_image, pred_joint, name1, save)

    true_joint = preparing_data.Inverse_coordinate_normalize(true_joints[input3]).reshape(28)
    true_joint_2d = torch.zeros((2, 14))
    for i in range(14):
        true_joint_2d[0][i] = true_joint[2 * i]
        true_joint_2d[1][i] = true_joint[2 * i + 1]

    orig_image = trans1(input2)
    draw_lines(orig_image, true_joint_2d, name2, save)
