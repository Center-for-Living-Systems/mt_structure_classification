import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, labeled=True):
        self.data = dataframe.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.labeled = labeled
        if labeled:
            self.label_to_index = {label: idx for idx, label in enumerate(sorted(self.data["label"].unique()))}
            self.data["label"] = self.data["label"].map(self.label_to_index)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.labeled:
            label = int(self.data.iloc[idx, 1])
            return image, label, img_name  # Return image path for saving later
        else:
            return image, img_name  # For unlabeled test data


# Function to initialize different models
def initialize_model(model_name, num_classes, device):
    if model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )
    elif model_name == "resnet":
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "convnext":
        model = models.convnext_tiny(pretrained=True)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Unsupported model type")
    return model.to(device)




# Function to load the best trained model
def load_best_model(model_name, model_path, num_classes, device):
    model = initialize_model(model_name, num_classes, device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Function to generate a CSV file from test images
def generate_test_csv(test_image_dir):
    image_files = [f for f in os.listdir(test_image_dir) if f.endswith(".png")]
    test_df = pd.DataFrame(image_files, columns=["filename"])
    return test_df


def basic_transform():
    # Define Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

# Function to classify a test dataset and save images in predicted label folders
def classify_and_save_images(model_name, test_image_dir, model_path, output_dir,label_to_index,result_csv,device):
    test_df = generate_test_csv(test_image_dir)
    test_dataset = CustomDataset(test_df, test_image_dir, basic_transform(),labeled=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = load_best_model(model_name, model_path, 5, device)
    
    os.makedirs(output_dir, exist_ok=True)

    results = []
    
    with torch.no_grad():
        for images, img_paths in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for img_path, pred in zip(img_paths, preds.cpu().numpy()):                
                pred_label = list(label_to_index.keys())[pred]
                label_folder = os.path.join(output_dir, pred_label)
                os.makedirs(label_folder, exist_ok=True)
                
                img_name = os.path.basename(img_path)
                img_save_path = os.path.join(label_folder, img_name)
                
                img = Image.open(img_path)
                img.save(img_save_path)
                
                results.append({"filename": img_name, "predicted_label": pred_label})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(result_csv, index=False)
    print(f"Results saved to: {result_csv}")
    
    print(f"Images saved in respective predicted label folders under: {output_dir}")
    return results_df