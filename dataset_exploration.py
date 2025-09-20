from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
### Divide training into train and validation set, first we obtain the indexes through stratified sampling then we move
###  the photos 
def split_train_val():
    train_val_dict = np.load('./labels/train_val_labels.npy', allow_pickle=True).item()
    all_idx = list(train_val_dict.keys())
    all_labels = list(train_val_dict.values())
    train_idx, val_idx = train_test_split(all_idx, test_size=0.11, random_state=42, stratify=all_labels)
    np.save("./labels/train_idx.npy", np.array(train_idx))
    np.save("./labels/val_idx.npy", val_idx)

def move_validation_into_folder():
    found = 0
    val_idx = np.load("./labels/val_idx.npy")
    train_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\train_set\\"
    val_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\val_set\\"
    for filename in os.listdir(train_path):
        idx_str = filename.replace("train_", "").replace(".jpg", "")
        if int(idx_str) in val_idx:
            found += 1
            os.rename(os.path.join(train_path, filename), os.path.join(val_path, f"val_{idx_str}.jpg"))
            if found %100 == 0:
                print(f"Moved {found} photos")


def get_train_val_label_dicts():
    train_val_dict = np.load('./labels/train_val_labels.npy', allow_pickle=True).item()

    train_idx = np.load("./labels/train_idx.npy")
    val_idx = np.load("./labels/val_idx.npy")

    train_labels = {idx: train_val_dict[idx] for idx in train_idx}
    val_labels = {idx: train_val_dict[idx] for idx in val_idx}

    np.save("./labels/train_dict.npy", train_labels, allow_pickle=True)
    np.save("./labels/val_dict.npy", val_labels, allow_pickle=True)

### Generate label numpy array from training annotations csv file
###  I have slightly modified the .csv file removing the 'train_', 'val_' and '.jpg' from each row in 
###  order to avoid doing it here in the code
def get_all_labels_from_annotations():
    train_csv_path ="C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\train_info.csv"
    test_csv_path ="C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\test_info.csv"

    train_val_labels = dict()
    test_labels = dict()
    train_csv_file = pd.read_csv(train_csv_path, names=['filename', 'label'])
    for _, row in train_csv_file.iterrows():
        file_idx = int(row['filename'])
        label = int(row['label'])
        
        train_val_labels[file_idx] = label
    np.save("./labels/train_val_labels.npy", train_val_labels, allow_pickle=True)
    test_csv_file = pd.read_csv(test_csv_path, names=['filename', 'label'])
    for _, row in test_csv_file.iterrows():
        file_idx = int(row['filename'])
        label = int(row['label'])
        
        test_labels[file_idx] = label
    np.save("./labels/test_labels.npy", test_labels, allow_pickle=True)




def check_file_sizes():
    unique_sizes = []
    folder_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\train_set\\"
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path)
            if 256 not in img.size:
                unique_sizes.append(img.size)
                print(img.size)


def change_filename():
    folder_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\test_set\\"
    for filename in os.listdir(folder_path):
        if filename.startswith("val_"):
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.replace("val_", "test_", 1)  # only replace first occurrence
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)

def count_labels():
    label_dict_path = './labels/train_dict.npy'
    labels_dict = np.load(label_dict_path, allow_pickle=True).item()
    all_labels = list(labels_dict.values())
    label_counts = Counter(all_labels)
    sorted_counts = dict(sorted(label_counts.items()))
    return sorted_counts

def compute_train_weights():
    label_dict_path = './labels/train_dict.npy'
    labels_dict = np.load(label_dict_path, allow_pickle=True).item()
    all_labels = list(labels_dict.values())
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / np.mean(weights)
    class_weights = np.zeros(len(unique_labels))
    for label, w in zip(unique_labels, weights):
        class_weights[label] = w
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device='cpu')
    np.save('./weigh_train.npy', weight_tensor.cpu().numpy())

if __name__ == '__main__':
    compute_train_weights()
    