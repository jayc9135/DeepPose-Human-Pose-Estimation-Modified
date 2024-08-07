# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:55:10 2020

@author: MrHossein
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def train_augment(device, model, train_data_aug, train_label_aug, true_joints, optimizer, loss, epoch, batch_size):
    image_out = []
    label_out = []
    output_log = []
    total_loss = 0

    rand_mask = np.random.permutation(len(train_data_aug))
    count = int(len(train_data_aug) / batch_size)

    temp_label = np.array(train_label_aug)
    temp_data = torch.zeros((len(train_data_aug), 3, 220, 220))
    for i in range(len(train_data_aug)):
        temp_data[i] = train_data_aug[i]

    print('Training Epoch: {}'.format(epoch))
    model.train()
    for i in range(count):
        print('.', end='')

        data = temp_data[rand_mask[i * batch_size:(i + 1) * batch_size]].to(device)
        image_out.append(data)

        target = true_joints[temp_label[rand_mask[i * batch_size:(i + 1) * batch_size]]].to(device)
        temp = temp_label[rand_mask[i * batch_size:(i + 1) * batch_size]]
        for j in range(len(temp)):
            label_out.append(temp[j])

        optimizer.zero_grad()
        output = model(data)
        output_log.append(output)
        loss_out = loss(output, target)
        loss_out.backward()
        optimizer.step()
        total_loss += loss_out.item()

    print('\tTrain Loss: {:.6f}'.format(total_loss / (i + 1)))
    return output_log, image_out, label_out, (total_loss / (i + 1))


def train(device, model, train_data, true_joints, optimizer, loss, epoch, batch_size):
    output_log = []
    train_label = []
    img = []
    total_loss = 0
    end_index = 0

    print('Training Epoch: {}'.format(epoch))
    model.train()
    for batch_idx, (images, label) in enumerate(train_data):
        print('.', end='')
        for i in range(len(label[0])):
            train_label.append(int(label[0][i][-8:-4]) - 1)

        data = images[0].to(device)
        img.append(data)
        end_index += len(images[0])
        target = true_joints[train_label[(batch_idx * batch_size):end_index]].to(device)

        optimizer.zero_grad()
        output = model(data)
        output_log.append(output)
        loss_out = loss(output, target)
        loss_out.backward()
        optimizer.step()
        total_loss += loss_out.item()

    print('\tTrain Loss: {:.6f}'.format(total_loss / (batch_idx + 1)))
    return output_log, img, train_label, (total_loss / (batch_idx + 1))


### Validation Step
def valid(device, model, valid_data, true_joints, loss, batch_size):
    output_log = []
    valid_label = []
    img = []
    total_loss = 0
    end_index = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(valid_data):
            for i in range(len(label[0])):
                valid_label.append(int(label[0][i][-8:-4]) - 1)

            data = images[0].to(device)
            img.append(data)
            end_index += len(images[0])
            target = true_joints[valid_label[(batch_idx * batch_size):end_index]].to(device)

            output = model(data)
            output_log.append(output)
            loss_out = loss(output, target)
            total_loss += loss_out.item()

    print('\t\tValid Loss: {:.6f}\n'.format(total_loss / (batch_idx + 1)))
    print('========================================================================================')
    return output_log, img, valid_label, (total_loss / (batch_idx + 1))


def test(device, model, test_data):
    output_log = []
    test_label = []
    img = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, label) in enumerate(test_data):
            for i in range(len(label[0])):
                test_label.append(int(label[0][i][-8:-4]) - 1)

            data = images[0].to(device)
            img.append(data)
            output = model(data)
            output_log.append(output)

    return output_log, img, test_label


def show_plot(train_loss, valid_loss, pcp_train_acc, pcp_valid_acc, limbs_name):
    # Plot For Loss
    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('Traning and Validation Loss')
    plt.plot(train_loss, color='blue', label='Train Loss')
    plt.plot(valid_loss, color='orange', label='Valid Loss')
    plt.legend()
    plt.savefig('TV_Loss.jpg', dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('PCP Metric Accuracy For Traning')
    for i in range(8):
        plt.plot(pcp_train_acc[:, i], label=limbs_name[i])
    plt.legend()
    plt.savefig('PCP_TR_Acc.jpg', dpi=200)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=100)
    plt.title('PCP Metric Accuracy For Validation')
    for i in range(8):
        plt.plot(pcp_valid_acc[:, i], label=limbs_name[i])
    plt.legend()
    plt.savefig('PCP_V_Acc.jpg', dpi=200)
    plt.show()


def print_function(train_acc, valid_acc, epochs, names, metric='PCP'):
    if metric == 'PCP':
        # For Traing Data
        print('In PCP Metric, The output training Accuracy for Each Limb In Epoch {}:'.format(epochs))
        for i in range(8):
            print('The Accuracy For {} is:\t{:.2f}%'.format(names[i], train_acc[i]))

        print('In PCP Metric, The output Validation Accuracy for Each Limbs In Epoch {}:'.format(epochs))
        for i in range(8):
            print('The Accuracy For {} is:\t{:.2f}%'.format(names[i], valid_acc[i]))
    else:
        print('\n')
        # For Traing Data
        print('In PDJ Metric, The output Training Accuracy for Each joint In Epoch {}:'.format(epochs))
        for i in range(14):
            print('The Accuracy For {} is:\t{:.2f}%'.format(names[i], train_acc[i]))

        print('In PDJ Metric, The output Validation Accuracy for Each joint In Epoch {}:'.format(epochs))
        for i in range(14):
            print('The Accuracy For {} is:\t{:.2f}%'.format(names[i], valid_acc[i]))
