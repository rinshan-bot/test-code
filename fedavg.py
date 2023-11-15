#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from google.colab import drive
# drive.mount('/content/gdrive')


# ## Device Loader

# In[2]:


import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms
from PIL import Image
from skimage.feature import local_binary_pattern
# from tqdm import tqdm
import matplotlib.pyplot as plt
# import cv2
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader(DataLoader):
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
          for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)
device = get_device()
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if device.type == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ## Data Processing

# In[3]:


class_to_idx = {"real": 0, "attack": 1}
idx_to_class = {0: "real", 1: "attack"}
class CustomImageFolderDataset(Dataset):
    def __init__(self, root_dir,mode,transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.data = self.load_data()
        self.transform = transform
    def load_data(self):
        root_dir_real = os.path.join(self.root_dir,self.mode,"real")
        root_dir_attack = os.path.join(self.root_dir,self.mode,"attack")
        data = []
        class_idx = class_to_idx['real']
        for img_file in os.listdir(root_dir_real):
            if img_file.endswith('.pgm'):
                img_path = os.path.join(root_dir_real, img_file)
                data.append((img_path, class_idx))
        class_idx = class_to_idx['attack']
        for img_file in os.listdir(root_dir_attack):
              if img_file.endswith('.pgm'):
                img_path = os.path.join(root_dir_attack, img_file)
                data.append((img_path, class_idx))
        return data

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

def imageDis(idx, data, labels):
    if idx == -1:
        img = data[idx].permute(1, 2, 0)# Rearrange dimensions (channels, height, width) to (height, width, channels)
        label = labels.item()
    else:
        img = data[idx].permute(1, 2, 0)  # Rearrange dimensions (channels, height, width) to (height, width, channels)
        label = labels[idx].item()
  # print(img_path)
    plt.title(f"Label - {idx_to_class.get(label)} - Shape - {img.shape}")
    plt.imshow(img)
    plt.show()


# In[7]:


root_dir = 'dataset/'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
batch_size = 128
train_dataset = CustomImageFolderDataset(root_dir,"train",transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomImageFolderDataset(root_dir,"validation",transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

total_train_size = len(train_dataset)
total_test_size = len(test_dataset)
print(total_train_size,total_test_size)


# In[8]:


# Split the dataset into multiple clients using DataLoader and Subset
def split_dataset(dataset, num_clients, batch_size=32):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_splits = []
    for _ in range(num_clients):
        client_data = DataLoader(Subset(dataset, np.random.choice(len(dataset), size=batch_size, replace=False)),
                                  batch_size=batch_size, shuffle=True)
        # client_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_splits.append(client_data)
    return data_splits


# ## Display Image

# In[9]:


# # # Example usage of the DataLoader
for batch_idx, (inputs, labels) in enumerate(train_loader):
    # print(input[0].shape)
    imageDis(0,inputs,labels)
    if batch_idx == 1:
        break


# ## Model

# In[10]:


from torch.nn import functional as F
import torchvision.models as models
import torch.nn as nn
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GlobalAveragePool2d(nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))

class FederatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.aux_logits = True
        resnet = nn.Sequential(*list(resnet.children())[:-2])
        resnet[0] = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = resnet
        self.global_avg_pool = GlobalAveragePool2d()
        self.fc1 = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)
        self.track_layers = {'resnet':resnet,'fc1': self.fc1, 'fc2': self.fc2}

    def forward(self, x):
        x = self.resnet(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_track_layers(self):
        return self.track_layers
    
    def apply_parameters(self, parameters_dict):
        with torch.no_grad():
            for layer_name in parameters_dict:
                if hasattr(self.track_layers[layer_name], 'weight'):
                    self.track_layers[layer_name].weight.data *= 0
                    self.track_layers[layer_name].weight.data += parameters_dict[layer_name]['weight']
                if hasattr(self.track_layers[layer_name], 'bias'):
                    self.track_layers[layer_name].bias.data *= 0
                    self.track_layers[layer_name].bias.data += parameters_dict[layer_name]['bias']

    def get_parameters(self):
        parameters_dict = dict()
        for layer_name in self.track_layers:
            if hasattr(self.track_layers[layer_name], 'weight') and hasattr(self.track_layers[layer_name], 'bias'):
                parameters_dict[layer_name] = {
                    'weight': self.track_layers[layer_name].weight.data.clone(),
                    'bias': self.track_layers[layer_name].bias.data.clone()
                }
        return parameters_dict

    def _process_batch(self, images,labels):
        # print(images)
        # print(images[0].shape)
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, labels)
        accuracy = self.batch_accuracy(outputs, labels)
        return (loss, accuracy)

    def fit(self, data_loader, epochs, lr, batch_size=128, opt=torch.optim.SGD):
        # dataloader = DeviceDataLoader(DataLoader(dataset, batch_size, shuffle=True), device)
        optimizer = torch.optim.Adam(self.parameters(), lr)
#         optimizer = torch.optim.Adam(self.parameters(),lr=lr,weight_decay=1e-4)
        # optimizer = opt(self.parameters(), lr)
        history = []
        for epoch in range(epochs):
            losses = []
            accs = []
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                loss, acc = self._process_batch(inputs,labels)
                # break
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss.detach()
                losses.append(loss)
                accs.append(acc)
            avg_loss = torch.stack(losses).mean().item()
            avg_acc = torch.stack(accs).mean().item()
            history.append((avg_loss, avg_acc))
        return history

    def batch_accuracy(self, outputs, labels):
        with torch.no_grad():
            _, predictions = torch.max(outputs, dim=1)
            return torch.tensor(torch.sum(predictions == labels).item() / len(predictions))

    def evaluate(self, data_loader, batch_size=128):
        losses = []
        accs = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                loss, acc = self._process_batch(inputs,labels)
                losses.append(loss)
                accs.append(acc)
        avg_loss = torch.stack(losses).mean().item()
        avg_acc = torch.stack(accs).mean().item()
        return (avg_loss, avg_acc)
# summary(to_device(FederatedNet(), device),(3,224,224))


# ## Federated Averaging

# In[ ]:


class Client:
    def __init__(self, client_id, dataset,learning_rate,epochs_per_client,batch_size):
        self.client_id = client_id
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epochs_per_client = epochs_per_client
        self.batch_size = batch_size

    def get_dataset_size(self):
        return len(self.dataset)

    def get_client_id(self):
        return self.client_id

    def train(self, parameters_dict):
        net = to_device(FederatedNet(), device)
        net.apply_parameters(parameters_dict)

        train_history = net.fit(self.dataset, self.epochs_per_client, self.learning_rate, self.batch_size)
        print('{}: Loss = {}, Accuracy = {}'.format(self.client_id, round(train_history[-1][0], 4), round(train_history[-1][1], 4)))
        return net.get_parameters()


# ## Initlize Model

# In[ ]:


num_clients = 7
rounds = 3
epochs_per_client = 5

learning_rate = 0.001
client_data_loaders = split_dataset(train_dataset, num_clients, batch_size=32)
clients = [Client('client_' + str(i), client_data_loaders[i],learning_rate,epochs_per_client,batch_size) for i in range(num_clients)]
global_net = to_device(FederatedNet(), device)


# ## Model Training

# In[ ]:


communication_times = []
history = []
for i in range(rounds):
    print('Start Round {} ...'.format(i + 1))
    curr_parameters = global_net.get_parameters()
    new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])
    for client in clients:
        start_time = time.time()
        client_parameters = client.train(curr_parameters)
        fraction = client.get_dataset_size() / total_train_size
        for layer_name in client_parameters:
            new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
            new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
    global_net.apply_parameters(new_parameters)

    train_loss, train_acc = global_net.evaluate(train_loader)
    test_loss, test_acc = global_net.evaluate(test_loader)
    print('After round {}, train_loss = {}, test_loss = {}, train_acc = {}, test_acc = {}\n'.format(i + 1, round(train_loss, 4),
            round(test_loss, 4),round(train_acc, 4), round(test_acc, 4)))
    end_time = time.time()
    comm_time = end_time - start_time
    communication_times.append(comm_time)
    history.append((train_loss, test_loss, train_acc, test_acc))


# ## Graphs

# In[ ]:


plt.plot([i + 1 for i in range(len(history))], [history[i][0] for i in range(len(history))], color='r', label='train loss')
plt.plot([i + 1 for i in range(len(history))], [history[i][1] for i in range(len(history))], color='b', label='test loss')
plt.legend()
plt.title('Training Loss')
plt.show()


# In[ ]:


plt.plot([i + 1 for i in range(len(history))], [history[i][2] for i in range(len(history))], color='r', label='train acc')
plt.plot([i + 1 for i in range(len(history))], [history[i][3] for i in range(len(history))], color='b', label='test acc')
plt.legend()
plt.title('Training Acc')
# plt.ylim(0.7,0.75)
plt.show()


