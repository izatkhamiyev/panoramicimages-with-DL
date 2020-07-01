import torch
import torch.nn as nn
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
  


mmmin = -200

dddiff = 400




class MyTrainData(torch.utils.data.dataset.Dataset):
       
    def __init__(self):
        self.length = 48000
            
    def __getitem__(self, index):
        
        tr = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

        img_left = np.load("filtered_img_left_mixed/{}.npz".format(index))['arr_0']
        img_left = Image.fromarray(img_left)
        tr_img_left = tr(img_left)
        
        img_right = np.load("filtered_img_right_mixed/{}.npz".format(index))['arr_0']
        img_right = Image.fromarray(img_right)
        tr_img_right = tr(img_right)
        
        stacked = torch.cat((tr_img_left, tr_img_right), dim = 0)
        
        corner_diff = (np.load('filtered_corn_diff_mixed/{}.npz'.format(index))['arr_0'] - mmmin) / dddiff
        dx_corner_diff = np.array([corner_diff[ev * 2] for ev in range(4)])
        corner_diff = torch.tensor(dx_corner_diff)
        
        return stacked, corner_diff
    
    def __len__(self):
        return self.length
    
class MyValidationData(torch.utils.data.dataset.Dataset):
       
    def __init__(self):
        self.length = 2000
            
    def __getitem__(self, index):
        index += 48000
        
        tr = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

        img_left = np.load("filtered_img_left_mixed/{}.npz".format(index))['arr_0']
        img_left = Image.fromarray(img_left)
        tr_img_left = tr(img_left)
        
        img_right = np.load("filtered_img_right_mixed/{}.npz".format(index))['arr_0']
        img_right = Image.fromarray(img_right)
        tr_img_right = tr(img_right)
        
        stacked = torch.cat((tr_img_left, tr_img_right), dim = 0)
        
        corner_diff = (np.load('filtered_corn_diff_mixed/{}.npz'.format(index))['arr_0'] - mmmin) / dddiff
        dx_corner_diff = np.array([corner_diff[ev * 2] for ev in range(4)])
        corner_diff = torch.tensor(dx_corner_diff)
        
        return stacked, corner_diff
    
    def __len__(self):
        return self.length
    
    
class MyTestData(torch.utils.data.dataset.Dataset):
    
    def __init__(self):
        self.length = 2000
            
    def __getitem__(self, index):
        
        index += 50000
        
        tr = torchvision.transforms.Compose([torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()])

        img_left = np.load("filtered_img_left_mixed/{}.npz".format(index))['arr_0']
        img_left = Image.fromarray(img_left)
        tr_img_left = tr(img_left)
        
        img_right = np.load("filtered_img_right_mixed/{}.npz".format(index))['arr_0']
        img_right = Image.fromarray(img_right)
        tr_img_right = tr(img_right)
        
        stacked = torch.cat((tr_img_left, tr_img_right), dim = 0)
        
        corner_diff = (np.load('filtered_corn_diff_mixed/{}.npz'.format(index))['arr_0'] - mmmin) / dddiff
        dx_corner_diff = np.array([corner_diff[ev * 2] for ev in range(4)])
        corner_diff = torch.tensor(dx_corner_diff)
        
        return stacked, corner_diff
    
    def __len__(self):
        return self.length
    
class DeepHomNet(nn.Module):

    def __init__(self):
        super(DeepHomNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 16 * 16)
        x = self.classifier(x)
        return x
    

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)





train_set = MyTrainData()
val_set = MyValidationData()

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = 64

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8),
    'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


model = DeepHomNet().to(device)
model.apply(init_weights)

#criterion = nn.MSELoss()
criterion = nn.L1Loss(reduction = 'mean')
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)
#optimizer_ft = optim.SGD(model.parameters(), lr=0.001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
scheduler = exp_lr_scheduler
optimizer = optimizer_ft

num_epochs=50


best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10

all_losses = []

cum_val_epoch_loss = 0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    trained_items = 0

    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            for param_group in optimizer.param_groups:
                print("LR", param_group['lr'])

            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        cum_val_epoch_loss = 0
        epoch_samples = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)         
            trained_items += inputs.size()[0]    

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                cum_val_epoch_loss += loss.sum().item()
                if (trained_items % (8 * batch_size) == 0):
                    print("loss at {} = {}".format(trained_items, loss.sum().item()))
#                    print("output = ")
                    print(outputs[0])
                    all_losses.append(loss.sum().item())
#                    print("input = ")
#                    print(inputs[0])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            epoch_samples += 1
        
        epoch_loss = cum_val_epoch_loss / epoch_samples

        # deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            print("saving best model")
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "mixed_deep_hom_net_wts_even")
            np.save("mixed_validation_corner_diff_output_even", outputs.cpu().detach().numpy())

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

test_loader = DataLoader(MyTestData(), batch_size=batch_size, shuffle=False, num_workers=0)

model_outputs = []
loss_count = 0
total_loss = 0
losses = []
for inputs, labels in test_loader:
    inputs = inputs.float().to(device)
    labels = labels.float().to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        model_outputs.append(outputs.cpu().detach().numpy())
        total_loss += loss.item()
        loss_count += 1
        losses.append(loss.item())

print("Test loss = {}".format(total_loss / loss_count))
np.save("mixed_deep_hom_net_test_losses_even", losses)
np.save("mixed_deep_hom_net_test_outputs_even", model_outputs)
np.save("mixed_deep_hom_net_training_loss_even", all_losses)