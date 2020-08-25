# from __future__ import print_function, division

# from torch.utils.data import Dataset, DataLoader
# import torchvision
# from torchvision import transforms, utils

# import os

# from skimage import io, transform
# import numpy as np
# import matplotlib as plt

# import argparse
# import os



# import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# import sklearn.utils
import torchvision
from torchvision.models import DenseNet, ResNet


# class Model(nn.Module):
    
#     def __init__(self, args):
#         super(Model, self).__init__()

#         self.backbone_model: DenseNet = torchvision.models.densenet121(pretrained=True)

#         self.fc1 = nn.Linear(in_features=self.backbone_model.classifier.in_features, out_features=args.classes_amount) # muzikas instrumentu klases

#     def forward(self, x):

#         out = self.backbone_model.features.forward(x)
#         out = F.adaptive_avg_pool2d(out, output_size=(1,1))
#         out = out.view(out.size(0), -1)
#         out = self.fc1.forward(out)

#         out = torch.softmax(out, dim=1)
        
#         return out


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        self.backbone_model: ResNet = torchvision.models.resnet18(pretrained=False)
        
                
        self.features = torch.nn.Sequential(
            self.backbone_model.conv1,
            self.backbone_model.bn1,
            self.backbone_model.relu,
            self.backbone_model.maxpool,


            self.backbone_model.layer1,
            nn.Dropout(0.8),
            self.backbone_model.layer2,
            nn.Dropout(0.8),
            self.backbone_model.layer3,
            nn.Dropout(0.8),
            self.backbone_model.layer4,
            nn.Dropout(0.8)

          
        )

        self.fc1 = nn.Linear(in_features=self.backbone_model.fc.in_features, out_features=args.classes_amount)
        
        self.lin_layer1 = nn.Sequential(
            nn.Linear(in_features=self.backbone_model.fc.in_features,out_features=args.classes_amount),
            nn.Dropout(0.8),
            nn.BatchNorm1d(num_features=args.classes_amount),
            nn.ReLU()

        )
        


    def forward(self, x):
        
        x = x.repeat(1,3,1,1)
        out = self.features.forward(x)
        out = F.adaptive_avg_pool2d(out, output_size=(1,1))
        out = out.view(out.size(0), -1)
        out = self.lin_layer1.forward(out)

        out = torch.softmax(out, dim=1)

        return out