import os

import necessary_function
import drawing_and_metrics
import preparing_data
from model import cnn_pose_detect
import time
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import numpy as np
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    limbs_name = ['Right Lower Leg', 'Right Upper Leg', 'Left Upper Leg', 'Left Lower Leg', 'Right Lower Arm',
                  'Right Upper Arm', 'Left Upper Arm', 'Left Lower Arm']
    joint_name = ['Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist',
                  'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck', 'Head Top']
    no_cuda = True
    augmentation = False
    epoches = 10
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Prepare Data
    dataset_path = './data'
    train_data, valid_data, test_data, true_joints, true_joints_2d = preparing_data.prepare_data(kwargs, dataset_path, 0.7, 0.15, 0.15, image_size=220)
    if augmentation:
        train_data_aug, train_label_aug, true_joints_aug, true_joints_2d_aug = preparing_data.train_data_augmentation(
            train_data, true_joints, true_joints_2d, 128)

    start_time = time.time()
    model = cnn_pose_detect.CNN_Pos_D().to(device)

    # Check if a saved checkpoint exists
    checkpoint_path = "CNN_Pos_Detection_final.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading model weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.000001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.97)
    loss = nn.MSELoss(reduction='sum')

    train_loss = np.zeros((1, epoches))
    valid_loss = np.zeros((1, epoches))
    pcp_train_accuracy = np.zeros((epoches, 8))
    pcp_valid_accuracy = np.zeros((epoches, 8))
    pdj_train_accuracy = np.zeros((epoches, 14))
    pdj_valid_accuracy = np.zeros((epoches, 14))

    # Training Process
    for i in range(epoches):
        if augmentation:
            train_output_joints, train_image, train_label, train_loss[0][i] = necessary_function.train_augment(device, model, train_data_aug, train_label_aug, true_joints_aug, optimizer, loss, i + 1, 128)
        else:
            train_output_joints, train_image, train_label, train_loss[0][i] = necessary_function.train(device, model, train_data, true_joints, optimizer, loss, i + 1, 128)
        valid_output_joints, valid_image, valid_label, valid_loss[0][i] = necessary_function.valid(device, model, valid_data, true_joints, loss, 128)
        scheduler.step()

        # Save model after each epoch
        if i % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{i+1}.pt")

        # Check PCP Accuracy
        if augmentation:
            pcp_train_accuracy[i] = drawing_and_metrics.correct_percentage(train_label, train_output_joints, true_joints_2d_aug, limbs_name, metric='PCP')
        else:
            pcp_train_accuracy[i] = drawing_and_metrics.correct_percentage(train_label, train_output_joints, true_joints_2d, limbs_name, metric='PCP')
        pcp_valid_accuracy[i] = drawing_and_metrics.correct_percentage(valid_label, valid_output_joints, true_joints_2d, limbs_name, metric='PCP')
        necessary_function.print_function(pcp_train_accuracy[i], pcp_valid_accuracy[i], i + 1, limbs_name, metric='PCP')

        # Check PDJ Accuracy
        if augmentation:
            pdj_train_accuracy[i] = drawing_and_metrics.correct_percentage(train_label, train_output_joints, true_joints_2d_aug, joint_name, metric='PDJ')
        else:
            pdj_train_accuracy[i] = drawing_and_metrics.correct_percentage(train_label, train_output_joints, true_joints_2d, joint_name, metric='PDJ')
        pdj_valid_accuracy[i] = drawing_and_metrics.correct_percentage(valid_label, valid_output_joints, true_joints_2d, joint_name, metric='PDJ')

        necessary_function.print_function(pdj_train_accuracy[i], pdj_valid_accuracy[i], i + 1, joint_name, metric='PDJ')

    # Show Graph Results
    necessary_function.show_plot(train_loss[0], valid_loss[0], pcp_train_accuracy, pcp_valid_accuracy, limbs_name)

    # Save the trained model
    torch.save(model.state_dict(), "CNN_Pos_Detection_final.pt")
    end_time = time.time()
    print(f'Total Time For Training : {(end_time - start_time) / 60:.2f} min')
