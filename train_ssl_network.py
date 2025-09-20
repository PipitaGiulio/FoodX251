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
from torch.cuda.amp import autocast, GradScaler

###LOCAL IMPORT
from datasets.pretext_dataset import SelfSupervisedPretextDS
from networks.ssl_network import SSLNetwork
from datasets.sup_classification_ds import SupervisedClassificationDS

train_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\train_set\\"
val_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\val_set\\"
test_path = "C:\\Users\\giuli\\Desktop\\UNI\\1 anno\\Supervised Learning\\Project\\Dataset\\test_set\\"
train_dict_p = ".\\labels\\train_dict.npy"
val_dict_p = ".\\labels\\val_dict.npy"
test_dict_p = ".\\labels\\test_dict.npy"

def pretext_task():
    ### Increase batch size with Gradient Accumulation
    batch_size = 128
    device = 'cuda'
    transform = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.GaussianBlur(kernel_size = 15, sigma = (0.2, 2)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    train_ds = SelfSupervisedPretextDS(train_path, transform)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    net = SSLNetwork()
    net.to(device)
    accumulator_steps = 4
    target_lr = 0.3 * (batch_size * accumulator_steps / 256) 
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=target_lr,
        momentum=0.9,
        weight_decay=0.000001,
        nesterov=False  
    )
    warmup_epochs = 5
    
    total_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(total_epochs - warmup_epochs)
    )
    delta = 0.005
    best_loss = None
    train_losses = []
    lrs = []
    temperature = 0.2
    scaler = GradScaler()
    for cur_epoch in range(50):
        if cur_epoch < warmup_epochs:
            lr = target_lr * (cur_epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0.0
        epoch_samples = 0
        i = 0
        all_out1 = []
        all_out2 = []
        for inp1, inp2 in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            with autocast():
                inp1 = inp1.to(device)
                inp2 = inp2.to(device)
                out1 = net(inp1)
                out2 = net(inp2)
                #loss = xtentloss(out1, out2, temperature, device)/accumulator_steps
                all_out1.append(out1)
                all_out2.append(out2)
                
                epoch_samples += inp1.size(0)
                if (i+1)%accumulator_steps==0 or (i+1) == len(train_dl):
                    all_out1 = torch.cat(all_out1, dim=0)
                    all_out2 = torch.cat(all_out2, dim=0)
                    loss = xtentloss(all_out1, all_out2, temperature, device)
                    scaler.scale(loss).backward()
                    epoch_loss += loss.item() * all_out1.size(0) 
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    all_out1 = []
                    all_out2 = []
                    torch.cuda.empty_cache()
            i += 1


        epoch_loss = epoch_loss/epoch_samples
        lrs.append(lr)
        train_losses.append(epoch_loss)
        print(f"Epoch {cur_epoch}")
        print(f" Training - loss: {epoch_loss} LR: {lr:.6f}")

        if best_loss == None or epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            torch.save({
                'epoch': cur_epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'train_loss': np.array(train_losses),
            }, "./models/xtend_best_ssl_pretext_model.pth")
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'train_loss': np.array(train_losses),
    }, "./models/xtent_last_ssl_pretext_model.pth")

### Function to compute the xtent loss
def xtentloss(z1, z2, temperature, device):
    z = torch.cat([z1, z2], dim = 0)
    #It works as similarity matrix since both z1 and z2 are normalized in the network structure
    sim_matrix = torch.mm(z, z.t())
    batch_size = z1.size(0)

    #Remove Self Similarity
    diag_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix.masked_fill(mask = diag_mask, value=-1e4)
    #Tune by temperature
    sim_matrix /= temperature
    #for pseudo labels, for each image only one positive pair it is formed and due to the z1 and z2 structure:
    # for z1 we know it is at z1 index + 128 positions (while for z2 at z2 index - 128 positions)
    pseudo_labels = torch.cat([
        torch.arange(batch_size, batch_size * 2),
        torch.arange(0, batch_size)
    ], dim=0).to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(sim_matrix, pseudo_labels)

    return loss

###  Perform the test on the pretext trained model
def test_pretext():
    batch_size = 32
    device = 'cuda'
    net = SSLNetwork()  
    checkpoint = torch.load("./models/xtent_last_ssl_pretext_model.pth", map_location="cuda", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.fc = nn.Sequential(
        nn.Linear(256 * 9, 512),
        nn.BatchNorm1d(512),  
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 251)  
    )
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.4), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    test_ds = SupervisedClassificationDS(test_path, test_dict_p, transform)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    test_loss, test_acc, test_prec, test_recall, test_f1, cm_test = validation(test_dl, net, device, 'test')
    np.save('./confusion_matrix/cm_pretext_task.npy', cm_test)


def fine_tune_ssl_net():
    device = 'cuda'
    net = SSLNetwork()  
    checkpoint = torch.load("./models/xtent_last_ssl_pretext_model.pth", map_location="cuda", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.fc = nn.Sequential(
        nn.Linear(256 * 9, 512),
        nn.BatchNorm1d(512),  
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 251)  
    )
    set_requires_grad(net, False)
    set_requires_grad(net.fc, True)
    net.to(device)

    ###same as for supervised
    batch_size = 32
    device = 'cuda'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.4), 
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

    ###POSITIONAL WEIGHTS?
    lr = 0.01
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
    #optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum=0.9, weight_decay=0.0001)
    optimizer = torch.optim.SGD(net.fc.parameters(), lr = lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        
        factor=0.1,        
        patience=2,        
        min_lr=0.00001     
    )
    for cur_epoch in range(35):
        #These lines are used to progressingly unfreeze the convolutional blocks
        """ if (cur_epoch + 1) == 5:
            set_requires_grad(net.conv5, True)
        elif (cur_epoch + 1) == 10:
            set_requires_grad(net.conv4, True)
        elif (cur_epoch + 1) == 15:
            set_requires_grad(net.conv3, True) """
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_correct = 0
        all_preds = []
        all_targets = []
        for inp, gt in tqdm(train_dl, desc=f"Epoch {cur_epoch}"):
            inp = inp.to(device)
            gt = gt.to(device)
            out = net(inp)
            loss = l_fun(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inp.size(0)
            pred = torch.argmax(out, dim = 1)
            correct = (pred == gt).sum().item()
            epoch_correct += correct
            epoch_samples += inp.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(gt.cpu().numpy())
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

        scheduler.step(val_loss)
        
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
            }, "./models/xtent_fc_best_ssl_classification_model.pth")
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

        # Final test metrics
        'test_loss': np.array(test_loss),
        'test_acc': np.array(test_acc),
        'test_precision': np.array(test_prec),
        'test_recall': np.array(test_recall),
        'test_f1': np.array(test_f1),
        'confusion_matrix_test' : np.array(cm_test)

    }, "./models/xtent_fc_last_ssl_classification_model.pth")


### Function to define which layers have to be updated and which should be freezed
def set_requires_grad(layers, grad):
    for param in layers.parameters():
        param.requires_grad = grad


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
    #pretext_task()
    fine_tune_ssl_net()
