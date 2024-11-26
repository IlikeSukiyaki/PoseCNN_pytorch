"""
Implements the PoseCNN network architecture in PyTorch.
"""
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torchvision.models as models
from torchvision.ops import RoIPool

import numpy as np
import random
import statistics
import time
from typing import Dict, List, Callable, Optional

from rob599 import quaternion_to_matrix
from p3_helper import HoughVoting, _LABEL2MASK_THRESHOL, loss_cross_entropy, loss_Rotation, IOUselection


def hello_pose_cnn():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from pose_cnn.py!")


class FeatureExtraction(nn.Module):
    """
    Feature Embedding Module for PoseCNN. Using pretrained VGG16 network as backbone.
    """    
    def __init__(self, pretrained_model):
        super(FeatureExtraction, self).__init__()
        embedding_layers = list(pretrained_model.features)[:30]
        ## Embedding Module from begining till the first output feature map
        self.embedding1 = nn.Sequential(*embedding_layers[:23])
        ## Embedding Module from the first output feature map till the second output feature map
        self.embedding2 = nn.Sequential(*embedding_layers[23:])

        for i in [0, 2, 5, 7, 10, 12, 14]:
            self.embedding1[i].weight.requires_grad = False
            self.embedding1[i].bias.requires_grad = False
    
    def forward(self, datadict):
        """
        feature1: [bs, 512, H/8, W/8]
        feature2: [bs, 512, H/16, W/16]
        """ 
        feature1 = self.embedding1(datadict['rgb'])
        feature2 = self.embedding2(feature1)
        return feature1, feature2

class SegmentationBranch(nn.Module):
    """
    Instance Segmentation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 64):
        super(SegmentationBranch, self).__init__()

        self.num_classes = num_classes
        
        self.conv1_feat1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        self.relu1_feat1 = nn.ReLU()
        self.conv1_feat2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        self.relu1_feat2 = nn.ReLU()


        self.upsample_f1tof2 = nn.Upsample(scale_factor=2, mode='bilinear')

        # Using nearest neighbor interpolation to upsample from feature level 2 to full size
        # Note: For 'nearest', the align_corners option is not applicable and thus not used
        self.upsample_f2tofullsize = nn.Upsample(scale_factor=8, mode='bilinear')
        # print("*********", )
        self.conv2 = nn.Conv2d(hidden_layer_dim, num_classes + 1, kernel_size=1)
        # self.relu2 = nn.ReLU()
        self.softmax_Seg = nn.Softmax(dim=1)

        # Initialize layers
        nn.init.kaiming_normal_(self.conv1_feat1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1_feat2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        nn.init.zeros_(self.conv1_feat1.bias)
        nn.init.zeros_(self.conv1_feat2.bias)
        nn.init.zeros_(self.conv2.bias)



    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            probability: Segmentation map of probability for each class at each pixel.
                probability size: (B,num_classes+1,H,W)
            segmentation: Segmentation map of class id's with highest prob at each pixel.
                segmentation size: (B,H,W)
            bbx: Bounding boxs detected from the segmentation. Can be extracted 
                from the predicted segmentation map using self.label2bbx(segmentation).
                bbx size: (N,6) with (batch_ids, x1, y1, x2, y2, cls)
        """
        probability = None
        segmentation = None
        bbx = None

        # Replace "pass" statement with your code
        feature1 = self.relu1_feat1(self.conv1_feat1(feature1))
        feature2 = self.relu1_feat2(self.conv1_feat2(feature2))

        feature2 = self.upsample_f1tof2(feature2)

        feature = torch.add(feature1, feature2)
        feature = self.upsample_f2tofullsize(feature)
        
        feature = self.conv2(feature)
        # feature = self.relu2(feature)
        probability = self.softmax_Seg(feature)

        segmentation = torch.argmax(probability, dim=1)
        bbx = self.label2bbx(segmentation)

        return probability, segmentation, bbx
    
    def label2bbx(self, label):
        bbx = []
        bs, H, W = label.shape
        device = label.device
        label_repeat = label.view(bs, 1, H, W).repeat(1, self.num_classes, 1, 1).to(device)
        label_target = torch.linspace(0, self.num_classes - 1, steps = self.num_classes).view(1, -1, 1, 1).repeat(bs, 1, H, W).to(device)
        mask = (label_repeat == label_target)
        for batch_id in range(mask.shape[0]):
            for cls_id in range(mask.shape[1]):
                if cls_id != 0: 
                    # cls_id == 0 is the background
                    y, x = torch.where(mask[batch_id, cls_id] != 0)
                    if y.numel() >= _LABEL2MASK_THRESHOL:
                        bbx.append([batch_id, torch.min(x).item(), torch.min(y).item(), 
                                    torch.max(x).item(), torch.max(y).item(), cls_id])
        bbx = torch.tensor(bbx).to(device)
        return bbx
        
        
class TranslationBranch(nn.Module):
    """
    3D Translation Estimation Module for PoseCNN. 
    """    
    def __init__(self, num_classes = 10, hidden_layer_dim = 128):
        super(TranslationBranch, self).__init__()

        # Replace "pass" statement with your code
        self.num_classes = num_classes
        
        self.conv1_feat1 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        self.relu1_feat1 = nn.ReLU()
        self.conv1_feat2 = nn.Conv2d(512, hidden_layer_dim, kernel_size=1)
        self.relu1_feat2 = nn.ReLU()

        self.upsample_f1tof2 = nn.Upsample(scale_factor=2)
        self.upsample_f2tofullsize = nn.Upsample(scale_factor=8)
        self.conv2 = nn.Conv2d(hidden_layer_dim, num_classes*3, kernel_size=1)

        # Initialize layers
        nn.init.kaiming_normal_(self.conv1_feat1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv1_feat2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

        nn.init.zeros_(self.conv1_feat1.bias)
        nn.init.zeros_(self.conv1_feat2.bias)
        nn.init.zeros_(self.conv2.bias)


    def forward(self, feature1, feature2):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
        Returns:
            translation: Map of object centroid predictions.
                translation size: (N,3*num_classes,H,W)
        """
        translation = None

        # Replace "pass" statement with your code
        feature1 = self.relu1_feat1(self.conv1_feat1(feature1))
        feature2 = self.relu1_feat2(self.conv1_feat2(feature2))
        feature2 = self.upsample_f1tof2(feature2)
        feature = torch.add(feature1, feature2)
        feature = self.upsample_f2tofullsize(feature)
      
        translation = self.conv2(feature)

        return translation

class RotationBranch(nn.Module):
    """
    3D Rotation Regression Module for PoseCNN. 
    """    
    def __init__(self, feature_dim = 512, roi_shape = 7, hidden_dim = 4096, num_classes = 10):
        super(RotationBranch, self).__init__()


        # Replace "pass" statement with your code
        self.roi_feat1 = RoIPool(output_size=roi_shape, spatial_scale=1/8)
        self.roi_feat2 = RoIPool(output_size=roi_shape, spatial_scale=1/16)
        self.fc1 = nn.Linear(feature_dim*roi_shape*roi_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4 * num_classes)



    def forward(self, feature1, feature2, bbx):
        """
        Args:
            feature1: Features from feature extraction backbone (B, 512, h, w)
            feature2: Features from feature extraction backbone (B, 512, h//2, w//2)
            bbx: Bounding boxes of regions of interst (N, 5) with (batch_ids, x1, y1, x2, y2)
        Returns:
            quaternion: Regressed components of a quaternion for each class at each ROI.
                quaternion size: (N,4*num_classes)
        """
        quaternion = None

        # Replace "pass" statement with your code
        feature1 = self.roi_feat1.forward(feature1, bbx)
        feature2 = self.roi_feat1.forward(feature2, bbx)

        feature = torch.add(feature1, feature2)
        feature = feature.view(feature.shape[0], -1)

        feature = self.fc1.forward(feature)
        feature = self.fc2.forward(feature)
        quaternion = self.fc3.forward(feature)


        return quaternion

class PoseCNN(nn.Module):
    """
    PoseCNN
    """
    def __init__(self, pretrained_backbone, models_pcd, cam_intrinsic):
        super(PoseCNN, self).__init__()

        self.iou_threshold = 0.7
        self.models_pcd = models_pcd
        self.cam_intrinsic = cam_intrinsic


        # Replace "pass" statement with your code
        self.feature_extraction = FeatureExtraction(pretrained_backbone)
        self.segmentation_branch = SegmentationBranch()
        self.translation_branch = TranslationBranch()
        self.rotation_branch = RotationBranch()



    def forward(self, input_dict):
        """
        input_dict = {
            'rgb',
            'depth',
            'objs_id',
            'mask',
            'bbx',
            'RTs'
        }
        """


        if self.training:
            loss_dict = {
                "loss_segmentation": 0,
                "loss_centermap": 0,
                "loss_R": 0
            }

            gt_bbx = self.getGTbbx(input_dict)

            # Important: the rotation loss should be calculated only for regions
            # of interest that match with a ground truth object instance.
            # Note that the helper function, IOUselection, may be used for 
            # identifying the predicted regions of interest with acceptable IOU 
            # with the ground truth bounding boxes.
            # If no ROIs result from the selection, don't compute the loss_R
            
            # Replace "pass" statement with your code
            # Feature extraction
            feature1, feature2 = self.feature_extraction.forward(input_dict)

            # loss_dict["loss_segmentation"] and calculated using the loss_cross_entropy(.) function.             
            probability, segmentation, bbx = self.segmentation_branch.forward(feature1, feature2)
            loss_dict["loss_segmentation"] = loss_cross_entropy(probability, input_dict["label"])

            # The training loss for translation should be stored in loss_dict["loss_centermap"] using the L1loss function. 
            translation = self.translation_branch.forward(feature1, feature2)
            loss_Translation = nn.L1Loss()
            # loss_dict["loss_centermap"] using the L1loss function. 
            loss_dict["loss_centermap"] = loss_Translation(translation, input_dict["centermaps"])

            # The training loss for rotation should be stored in loss_dict["loss_R"] using the given loss_Rotation function. 
            if bbx.numel() > 0:
                sel_bbx = IOUselection(bbx.to(feature1), gt_bbx, threshold=self.iou_threshold)

                if sel_bbx.numel() > 0:
                    quaternion = self.rotation_branch.forward(feature1, feature2, sel_bbx[:, :-1])
                    pred_Rs, label = self.estimateRotation(quaternion, sel_bbx)
                    gt_Rs = self.gtRotation(sel_bbx, input_dict)
                    loss_dict["loss_R"] = loss_Rotation(pred_Rs, gt_Rs, label, self.models_pcd)


            
            return loss_dict
        else:
            output_dict = None
            segmentation = None

            with torch.no_grad():

                # Replace "pass" statement with your code
                feature1, feature2 = self.feature_extraction.forward(input_dict)
                probability, segmentation, bbx = self.segmentation_branch.forward(feature1, feature2)
                translation = self.translation_branch.forward(feature1, feature2)
                pred_centers, depths = HoughVoting(segmentation, translation)

                quaternion = self.rotation_branch.forward(feature1, feature2, bbx[:, :-1].to(dtype=feature1.dtype))
                pred_Rs, _ = self.estimateRotation(quaternion, bbx)


                output_dict = self.generate_pose(pred_Rs, pred_centers, depths, bbx)


            return output_dict, segmentation
    
    def estimateTrans(self, translation_map, filter_bbx, pred_label):
        """
        translation_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        label: a tensor [batch_size, num_classes, height, width]
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            trans_map = translation_map[batch_id, (cls-1) * 3 : cls * 3, :]
            label = (pred_label[batch_id] == cls).detach()
            pred_T = trans_map[:, label].mean(dim=1)
            pred_Ts[idx] = pred_T
        return pred_Ts

    def gtTrans(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Ts = torch.zeros(N_filter_bbx, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Ts[idx] = input_dict['RTs'][batch_id][cls - 1][:3, [3]].T
        return gt_Ts 

    def getGTbbx(self, input_dict):
        """
            bbx is N*6 (batch_ids, x1, y1, x2, y2, cls)
        """
        gt_bbx = []
        objs_id = input_dict['objs_id']
        device = objs_id.device
        ## [x_min, y_min, width, height]
        bbxes = input_dict['bbx']
        for batch_id in range(bbxes.shape[0]):
            for idx, obj_id in enumerate(objs_id[batch_id]):
                if obj_id.item() != 0:
                    # the obj appears in this image
                    bbx = bbxes[batch_id][idx]
                    gt_bbx.append([batch_id, bbx[0].item(), bbx[1].item(),
                                  bbx[0].item() + bbx[2].item(), bbx[1].item() + bbx[3].item(), obj_id.item()])
        return torch.tensor(gt_bbx).to(device=device, dtype=torch.int16)
        
    def estimateRotation(self, quaternion_map, filter_bbx):
        """
        quaternion_map: a tensor [batch_size, num_classes * 3, height, width]
        filter_bbx: N_filter_bbx * 6 (batch_ids, x1, y1, x2, y2, cls)
        """
        N_filter_bbx = filter_bbx.shape[0]
        pred_Rs = torch.zeros(N_filter_bbx, 3, 3)
        label = []
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            quaternion = quaternion_map[idx, (cls-1) * 4 : cls * 4]
            quaternion = nn.functional.normalize(quaternion, dim=0)
            pred_Rs[idx] = quaternion_to_matrix(quaternion)
            label.append(cls)
        label = torch.tensor(label)
        return pred_Rs, label

    def gtRotation(self, filter_bbx, input_dict):
        N_filter_bbx = filter_bbx.shape[0]
        gt_Rs = torch.zeros(N_filter_bbx, 3, 3)
        for idx, bbx in enumerate(filter_bbx):
            batch_id = int(bbx[0].item())
            cls = int(bbx[5].item())
            gt_Rs[idx] = input_dict['RTs'][batch_id][cls - 1][:3, :3]
        return gt_Rs 

    def generate_pose(self, pred_Rs, pred_centers, pred_depths, bbxs):
        """
        pred_Rs: a tensor [pred_bbx_size, 3, 3]
        pred_centers: [batch_size, num_classes, 2]
        pred_depths: a tensor [batch_size, num_classes]
        bbx: a tensor [pred_bbx_size, 6]
        """        
        output_dict = {}
        for idx, bbx in enumerate(bbxs):
            bs, _, _, _, _, obj_id = bbx
            R = pred_Rs[idx].numpy()
            center = pred_centers[bs, obj_id - 1].numpy()
            depth = pred_depths[bs, obj_id - 1].numpy()
            if (center**2).sum().item() != 0:
                T = np.linalg.inv(self.cam_intrinsic) @ np.array([center[0], center[1], 1]) * depth
                T = T[:, np.newaxis]
                if bs.item() not in output_dict:
                    output_dict[bs.item()] = {}
                output_dict[bs.item()][obj_id.item()] = np.vstack((np.hstack((R, T)), np.array([[0, 0, 0, 1]])))
        return output_dict


def eval(model, dataloader, device, alpha = 0.35):
    import cv2
    model.eval()

    sample_idx = random.randint(0,len(dataloader.dataset)-1)
    ## image version vis
    rgb = torch.tensor(dataloader.dataset[sample_idx]['rgb'][None, :]).to(device)
    inputdict = {'rgb': rgb}
    pose_dict, label = model(inputdict)
    poselist = []
    rgb =  (rgb[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return dataloader.dataset.visualizer.vis_oneview(
        ipt_im = rgb, 
        obj_pose_dict = pose_dict[0],
        alpha = alpha
        )

