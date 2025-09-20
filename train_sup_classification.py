import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

###LOCAL IMPORTS
from datasets.sup_classification_ds import SupervisedClassificationDS
from networks.sup_classification_network import SupClassificationNetwork

train_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\train_set\\"
val_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\val_set\\"
test_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\test_set\\"
train_dict_p = ".\\labels\\train_dict.npy"
val_dict_p = ".\\labels\\val_dict.npy"
test_dict_p = ".\\labels\\test_dict.npy"



def classification_pipeline():

    batch_size = 64
    device = 'cuda'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size = 3, sigma = (0.2, 1))], p = 0.5),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = SupervisedClassificationDS(train_path, train_dict_p, transform)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    val_ds = SupervisedClassificationDS(val_path, val_dict_p, transform)
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    test_ds = SupervisedClassificationDS(test_path, test_dict_p, transform)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    lr = 0.0001
    patience = 15
    delta = 0.005
    weights = torch.tensor(np.load('./weight_train.npy'), dtype=torch.float32).to(device)
    l_fun = nn.CrossEntropyLoss(weight=weights)

    epoch_train_losses = []
    epoch_train_accuracy = []
    epoch_train_recall = []
    epoch_train_f1score = []
    epoch_train_precision = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    epoch_val_recall = []
    epoch_val_f1score = []
    epoch_val_precision = []
    epoch_lrs = []
    best_loss = None

    net = SupClassificationNetwork()
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.08, 
        steps_per_epoch=len(train_dl),
        epochs=35,
        pct_start=0.1
    )
    accumulator_steps = 4
    for cur_epoch in range(35):
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        all_preds = []
        all_targets = []
        step = 0
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            inp = inp.to(device)
            gt = gt.to(device)
            out = net(inp)
            loss = l_fun(out, gt)/accumulator_steps
            
            loss.backward()
            if (step + 1) %accumulator_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            epoch_loss += loss.item() * accumulator_steps * inp.size(0)
            pred = torch.argmax(out, dim = 1)
            correct = (pred == gt).sum().item()
            epoch_correct += correct
            epoch_samples += inp.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(gt.cpu().numpy())
            step += 1
        epoch_loss = epoch_loss/epoch_samples
        epoch_accuracy = epoch_correct/epoch_samples
        epoch_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracy.append(epoch_accuracy)
        epoch_train_recall.append(epoch_recall)
        epoch_train_f1score.append(epoch_f1)
        epoch_train_precision.append(epoch_precision)
        current_lr = scheduler.get_last_lr()[0]
        epoch_lrs.append(current_lr)
        print(f"Epoch {cur_epoch}")
        print(f" Training - loss: {epoch_loss}, accuracy: {epoch_accuracy}, precision: {epoch_precision}, recall: {epoch_recall}, f1 score: {epoch_f1}")
        val_loss, val_acc, val_prec, val_recall, val_f1, _ = validation(val_dl, net, device, 'val') 
        epoch_val_losses.append(val_loss)
        epoch_val_accuracy.append(val_acc)
        epoch_val_recall.append(val_recall)
        epoch_val_f1score.append(val_f1)
        epoch_val_precision.append(val_prec)
        if best_loss == None or val_loss < best_loss - delta:
            best_loss = val_loss
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'learning_rates': np.array(epoch_lrs),
                # Accuracy
                'train_accuracy': np.array(epoch_train_accuracy),
                'val_accuracy': np.array(epoch_val_accuracy),
                
                # Loss
                'train_loss': np.array(epoch_train_losses),
                'val_loss': np.array(epoch_val_losses),

                # Precision
                'train_precision': np.array(epoch_train_precision),
                'val_precision': np.array(epoch_val_precision),

                # Recall
                'train_recall': np.array(epoch_train_recall),
                'val_recall': np.array(epoch_val_recall),

                # F1-score
                'train_f1': np.array(epoch_train_f1score),
                'val_f1': np.array(epoch_val_f1score)
            }, "./models/take2_best_sup_classification_model.pth")
        else:
            patience -=1
            print(f"Patience down to {patience}")
            if patience <= 0: 
                print("Early stopping triggered!")
                break
        
    print("Training Stopped, next metrics will be on the test set")
    test_loss, test_acc, test_prec, test_recall, test_f1, cm_test = validation(test_dl, net, device, 'test')
    torch.save({
        'epoch': cur_epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'learning_rates': np.array(epoch_lrs),
        # Training metrics
        'train_loss': np.array(epoch_train_losses),
        'train_acc': np.array(epoch_train_accuracy),
        'train_precision': np.array(epoch_train_precision),
        'train_recall': np.array(epoch_train_recall),
        'train_f1': np.array(epoch_train_f1score),

        # Validation metrics
        'val_loss': np.array(epoch_val_losses),
        'val_accuracy': np.array(epoch_val_accuracy),
        'val_precision': np.array(epoch_val_precision),
        'val_recall': np.array(epoch_val_recall),
        'val_f1': np.array(epoch_val_f1score),

        #Test metrics
        'test_loss': np.array(test_loss),
        'test_acc': np.array(test_acc),
        'test_precision': np.array(test_prec),
        'test_recall': np.array(test_recall),
        'test_f1': np.array(test_f1),
        'confusion_matrix_test' : np.array(cm_test)

    }, "./models/take2_last_sup_classification_model.pth")

def validation(dl, net, device, mode):
    loss_fun = nn.CrossEntropyLoss()
    net.eval()
    cm = None
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_preds = []
    all_targets = []
    for inp, gt in dl:
        inp = inp.to(device)
        gt = gt.to(device)
        with torch.no_grad():
            out = net(inp)
        loss = loss_fun(out, gt)
        #getting the total loss for the batch, used to get the total average after
        total_loss += loss.item() * inp.size(0)
        pred = torch.argmax(out, dim = 1)
        correct = (pred == gt).sum().item()
        total_correct += correct
        total_samples += inp.size(0) 
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(gt.cpu().numpy())
    
    avg_loss = total_loss/total_samples
    accuracy = total_correct/total_samples 
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    print(f" Validation - loss: {avg_loss:.4f}, acc: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, F1: {f1:.4f}")
    
    if mode == "test":
        cm = confusion_matrix(all_targets, all_preds)
    net.train()
    return avg_loss, accuracy, precision, recall, f1, cm


if __name__ == '__main__':
    classification_pipeline()
