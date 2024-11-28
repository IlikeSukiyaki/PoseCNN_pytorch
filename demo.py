import os
import time

os.environ["TZ"] = "US/Eastern"
time.tzset()

import matplotlib.pyplot as plt
from pose_cnn import hello_pose_cnn
from p3_helper import hello_helper
from rob599 import reset_seed
from rob599.grad import rel_error
import torch
from rob599 import PROPSPoseDataset
from rob599 import reset_seed, visualize_dataset
import torchvision.models as models
from pose_cnn import FeatureExtraction
from rob599 import reset_seed
from pose_cnn import FeatureExtraction, SegmentationBranch


# Ensure helper functions run correctly
hello_pose_cnn()
hello_helper()

# Check last modification time of pose_cnn.py
pose_cnn_path = os.path.join("/home/yifeng/PycharmProjects/TestEnv/PoseCNN/pose_cnn.py")
pose_cnn_edit_time = time.ctime(os.path.getmtime(pose_cnn_path))
print("pose_cnn.py last edited on %s" % pose_cnn_edit_time)

# Set up matplotlib plotting parameters
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# Check for CUDA availability
if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -------------------------- Dataset Preparation --------------------------------
# NOTE: Set `download=True` for the first time to download the dataset.
# After downloading, set `download=False` for faster execution.

data_root = "/home/yifeng/PycharmProjects/TestEnv/PoseCNN"
# Prepare train dataset
train_dataset = PROPSPoseDataset(
    root=data_root,
    split="train",
    download=False  # Change to True for the first-time download
)

# Prepare validation dataset
val_dataset = PROPSPoseDataset(
    root=data_root,
    split="val",
    download=False
)

# Print dataset sizes
print(f"Dataset sizes: train ({len(train_dataset)}), val ({len(val_dataset)})")


reset_seed(0)

grid_vis = visualize_dataset(val_dataset,alpha = 0.25)
plt.axis('off')
plt.imshow(grid_vis)
plt.show()

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Based on PoseCNN section III.B, the output features should
# be 1/8 and 1/16 the input's spatial resolution with 512 channels
print('feature1 expected shape: (N, {}, {}, {})'.format(512, 480//8, 640//8))
print('feature2 expected shape: (N, {}, {}, {})'.format(512, 480//16, 640//16))
print()

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
feature_extractor = FeatureExtraction(pretrained_model=vgg16)

dummy_input = {'rgb': torch.zeros((2,3,480,640))}
feature1, feature2 = feature_extractor(dummy_input)

print('feature1 shape:', feature1.shape)
print('feature2 shape:', feature2.shape)

