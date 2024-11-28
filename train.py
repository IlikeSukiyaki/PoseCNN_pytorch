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
import time
from torch.utils.data import DataLoader
import torchvision.models as models
import multiprocessing

from rob599 import reset_seed
from pose_cnn import PoseCNN
from tqdm import tqdm

# Set a few constants related to data loading.
NUM_CLASSES = 10
BATCH_SIZE = 4
NUM_WORKERS = multiprocessing.cpu_count()

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

reset_seed(0)

dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
posecnn_model = PoseCNN(pretrained_backbone = vgg16,
                models_pcd = torch.tensor(train_dataset.models_pcd).to(DEVICE, dtype=torch.float32),
                cam_intrinsic = train_dataset.cam_intrinsic).to(DEVICE)
posecnn_model.train()

optimizer = torch.optim.Adam(posecnn_model.parameters(), lr=0.001,
                            betas=(0.9, 0.999))


loss_history = []
log_period = 5
_iter = 0


st_time = time.time()
for epoch in range(10):
    train_loss = []
    dataloader.dataset.dataset_type = 'train'
    for batch in dataloader:
        for item in batch:
            batch[item] = batch[item].to(DEVICE)
        loss_dict = posecnn_model(batch)
        optimizer.zero_grad()
        total_loss = 0
        for loss in loss_dict:
            total_loss += loss_dict[loss]
        total_loss.backward()
        optimizer.step()
        train_loss.append(total_loss.item())

        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in loss_dict.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())
        _iter += 1

    print('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                  ', ' + 'Epoch %02d' % epoch + ', ' + 'Training finished' + f' , with mean training loss {np.array(train_loss).mean()}'))

torch.save(posecnn_model.state_dict(), os.path.join("your path here", "posecnn_model.pth"))

plt.title("Training loss history")
plt.xlabel(f"Iteration (x {log_period})")
plt.ylabel("Loss")
plt.plot(loss_history)
plt.show()