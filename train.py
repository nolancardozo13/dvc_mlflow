import os
import numpy as np
import pandas as pd
import argparse
import warnings
import mlflow
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
import dvc.api


class BlindnessDataset(Dataset):
    def __init__(self, data_path, num_classes = 5, phase = 'train'):
        scores_df =  pd.read_csv(data_path+'/train_val.csv')
        scores_df = scores_df[scores_df['split'] == phase].reset_index(drop = True)
        self.scores = scores_df['diagnosis']
        self.file_names = scores_df['id_code']
        self.num_classes = num_classes
        self.data_path = data_path
        self.phase = phase
          
    def _transform(self, image):
        transform = transforms.Compose([transforms.Resize((512, 512)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                       ])
        return transform(image)
    
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, index):
        file_name = self.file_names[index]
        # Get Images of the patient
        image = self._transform(Image.open(self.data_path+'/'+self.phase+'_images/'+file_name+'.png').convert('RGB'))
        label = self.scores[index]
        levels = [1]*label + [0]*(self.num_classes - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        return image, label, levels


class model_classifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(model_classifier, self).__init__()
        resnet = models.resnet18(pretrained = pretrained, progress = False)
        self.fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_in_features, (self.num_classes-1)*2)
        )

    def forward(self, inputs): # inputs.shape = samples x height x width x channels
        x = self.features(inputs)
        logits = self.classifier(x)
        logits = logits.view(-1, (self.num_classes-1), 2)
        probas = nn.functional.softmax(logits, dim=2)[:, :, 1]
        return logits, probas
    

def task_importance_weights(data_path):
    scores_df =  pd.read_csv(data_path+'/train_val.csv')
    label_array = torch.tensor(scores_df.diagnosis, dtype=torch.float)
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)
    m = torch.zeros(uniq.shape[0])
    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())
    imp = m/torch.max(m)
    imp = imp[0:int(args.num_classes)-1]
    return imp


def cost_fn(logits, levels, imp, reduction = "mean"):
    val = (-torch.sum((nn.functional.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + nn.functional.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    if reduction == "none":
        return val
    else:
        return torch.mean(val)


def train_model(model, dataloaders, optimizer, lr_scheduler, model_path, imp_weights, num_epochs=25):
    with mlflow.start_run() as mlrun:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count()>1:
            model = nn.DataParallel(model)

        if os.path.exists(model_path):
            mlflow.pytorch.load_model(model_path)
            
        imp_weights = imp_weights.to(device)
        model = model.to(device)

        best_kappa = 0.0
        for epoch in range(1, num_epochs+1):
            print("epoch: "+str(epoch))
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                mae = 0
                kappa = 0
                all_preds = []
                all_labels = []
                # Iterate over data.
                for (inputs, labels, levels) in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    levels = levels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # FORWARD AND BACK PROP
                        logits, probas = model(inputs)
                        loss = cost_fn(logits, levels, imp_weights)
                        # Get model predictions
                        pred = probas > 0.5
                        preds =  torch.sum(pred,dim = 1)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    mae += torch.sum(torch.abs(preds - labels))
                    all_preds.append(preds)
                    all_labels.append(labels)
                epoch_loss = running_loss / len(dataloaders[phase].sampler)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].sampler)
                epoch_mae = mae.double() / len(dataloaders[phase].sampler)
                all_labels = torch.cat(all_labels, 0)
                all_preds = torch.cat(all_preds, 0)
                kappa = cohen_kappa_score(all_labels.cpu().numpy(),all_preds.cpu().numpy(), weights = 'quadratic')
                if phase == 'val':
                    lr_scheduler.step(epoch_loss)
                    mlflow.log_metric("val_loss", round(epoch_loss,4), step = epoch)
                    mlflow.log_metric("val_accuracy", round(epoch_acc.cpu().numpy().item(),4), step = epoch)
                    mlflow.log_metric("val_mae", round(epoch_mae.cpu().numpy().item(), 4), step = epoch)
                    mlflow.log_metric("val_kappa", round(kappa, 4), step = epoch)
                else:
                    mlflow.log_metric("train_loss", round(epoch_loss, 4))
                    mlflow.log_metric("train_accuracy", round(epoch_acc.cpu().numpy().item(),4), step = epoch)
                    mlflow.log_metric("train_mae", round(epoch_mae.cpu().numpy().item(), 4), step = epoch)
                    mlflow.log_metric("train_kappa", round(kappa, 4), step = epoch)

                # save model
                if phase == 'val' and kappa > best_kappa:
                    best_kappa = kappa
                    scripted_model = torch.jit.script(model)
        mlflow.pytorch.save_model(scripted_model, model_path)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", default = 5, type = int, help = "The number of classes in the dataset")
    parser.add_argument("--batch_size", default = 16, type = int, help = "The batch size to be used when training")
    parser.add_argument("--epochs", default = 5, type = int , help = "The number of epochs to be used when training")
    parser.add_argument("--lr", default = 0.000001, type = float, help = "The learning rate to start with when training")
    parser.add_argument("--weight_decay", default = 0.01, type = float, help = "The weight decay to use for regularization")
    parser.add_argument("--pre_trained" , default = True, type = bool , help = "The number of epochs to be used when training")
    parser.add_argument("--data_path", type = str, help = "The path of the dataset in the repository")
    parser.add_argument("--model_path", default = "model", type = str, help = "The path to save the model")
    args = parser.parse_args()


    train_dataset = BlindnessDataset(data_path = args.data_path, num_classes = args.num_classes, phase = 'train')
    val_dataset = BlindnessDataset(data_path = args.data_path, num_classes = args.num_classes, phase = 'val')
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size)
    data_loaders = {'train': train_loader, 'val': val_loader}


    model = model_classifier(num_classes= args.num_classes, pretrained= args.pre_trained)

    imp = task_importance_weights(args.data_path)

    optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay = args.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,verbose = True)
    print('------start training------')
    train_model(model=model, dataloaders=data_loaders, optimizer = optimizer, lr_scheduler=lr_scheduler, 
        model_path = args.model_path, imp_weights = imp, num_epochs= args.epochs)



