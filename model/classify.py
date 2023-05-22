import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import os.path as osp
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from PIL import Image, ImageDraw, ImageFont


class PassportClassifier:
    def __init__(self, pth_device='cpu', model_path='passport_classifier_12.pth'):
        self.device = pth_device
        self.model = models.resnet101(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    @torch.no_grad()
    def classify(self, img):
        
        pil_img = Image.fromarray(img)
        image = pil_img.resize((224, 224))
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        # Make the prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities).item()

        if predicted_class == 1:
            return "passport"
        else:
            return "non-passport"
