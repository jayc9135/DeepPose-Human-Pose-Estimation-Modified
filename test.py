import necessary_function
import drawing_and_metrics
import preparing_data
from model import cnn_pose_detect
import torch
import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    limbs_name = ['Right Lower Leg', 'Right Upper Leg', 'Left Upper Leg',
                  'Left Lower Leg', 'Right Lower Arm',
                  'Right Upper Arm', 'Left Upper Arm', 'Left Lower Arm']
    joint_name = ['Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip',
                  'Left Knee', 'Left Ankle', 'Right Wrist',
                  'Right Elbow', 'Right Shoulder', 'Left Shoulder',
                  'Left Elbow', 'Left Wrist', 'Neck', 'Head Top']
    no_cuda = True
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Prepare Data
    dataset_path = './data'
    _, _, test_data, _, true_joints_2d = preparing_data.prepare_data(kwargs,
                                                                     dataset_path,
                                                                     0.7, 0.15,
                                                                     0.15,
                                                                     image_size=220)

    # Load the model
    model = cnn_pose_detect.CNN_Pos_D().to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load('CNN_Pos_Detection_final.pt'))

    # Set the model to evaluation mode
    model.eval()

    # Test Process
    test_output_joints, test_image, test_label = necessary_function.test(
        device, model, test_data)
    pcp_test_acc = drawing_and_metrics.correct_percentage(test_label,
                                                          test_output_joints,
                                                          true_joints_2d,
                                                          limbs_name,
                                                          metric='PCP')
    pdj_test_acc = drawing_and_metrics.correct_percentage(test_label,
                                                          test_output_joints,
                                                          true_joints_2d,
                                                          joint_name,
                                                          metric='PDJ')

    # Calculate and display combined accuracy
    combined_pcp_accuracy = pcp_test_acc.mean()
    combined_pdj_accuracy = pdj_test_acc.mean()
    overall_combined_accuracy = (combined_pcp_accuracy + combined_pdj_accuracy) / 2

    print(f'Combined PCP Accuracy: {combined_pcp_accuracy:.2f}%')
    print(f'Combined PDJ Accuracy: {combined_pdj_accuracy:.2f}%')
    print(f'Overall Combined Accuracy: {overall_combined_accuracy:.2f}%')
