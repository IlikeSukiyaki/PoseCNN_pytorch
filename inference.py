import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import multiprocessing
import rob599
from pose_cnn import PoseCNN, eval
from rob599 import PROPSPoseDataset
import os
import matplotlib.pyplot as plt
# Check for CUDA availability
if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
rob599.reset_seed(0)
NUM_CLASSES = 10
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count()

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


dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
posecnn_model = PoseCNN(pretrained_backbone = vgg16,
                models_pcd = torch.tensor(val_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                cam_intrinsic = val_dataset.cam_intrinsic).to(DEVICE)
posecnn_model.load_state_dict(torch.load(os.path.join("/home/yifeng/PycharmProjects/TestEnv/posecnn_model.pth")))

num_samples =5
for i in range(num_samples):
    out = eval(posecnn_model, dataloader, DEVICE)

    plt.axis('off')
    plt.imshow(out)
    plt.show()