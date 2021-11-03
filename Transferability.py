import torch 
import os 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
from torch.optim import Adam
import torchaudio
import pandas as pd
import numpy as np
import time # import time 
import sys # Import System 
from torch import optim, cuda # import optimizer  and CUDA
import random
import torch.nn.functional as F
from FD_Custom_Dataloader import TSData_Train
from sklearn.metrics import confusion_matrix
Num_Class=10
learning_rate=0.0008
batch_size=128
SEED = 1234 # Initialize seed 
EPOCHS=100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor()])
train_dataset=TSData_Train(transform=train_transformations)
train_size = int(0.8 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Test_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
#valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader
class Model_M2(nn.Module):
    def __init__(self):
        super(Model_M2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4,stride=1,padding = 1)
        self.mp1 = nn.MaxPool2d(kernel_size=4,stride=2)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4,stride =1)
        self.mp2 = nn.MaxPool2d(kernel_size=4,stride=2)
        self.fc1= nn.Linear(1024,256)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp1(self.conv1(x)))
        x = F.relu(self.mp2(self.conv2(x)))
        x = x.view(in_size,-1)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
class Model_M1(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Model_M1, self).__init__()
        self.CNN1=nn.Conv2d(1,32,kernel_size=(9,9))
        self.MP1=nn.MaxPool2d(kernel_size=2)
        self.CNN2=nn.Conv2d(32,32,kernel_size=(9,9))
        self.MP2=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(128,64)
        self.fc2=nn.Linear(64,96)
        self.fc3=nn.Linear(96,Num_Class)
    def forward(self,TS):
        x1=self.CNN1(TS)
        x1=F.relu(x1)
        x2=self.MP1(x1)
        x3=self.CNN2(x2)
        x3=F.relu(x3)
        x4=self.MP2(x3)
        #print(x4.shape)
        #x5=x4.view(x4.size(0),-1)
        x5=torch.flatten(x4, start_dim=1)
        #print(x5.shape)
        x6=self.fc1(x5)
        x7=self.fc2(x6)
        x8=self.fc3(x7)
        #x8=F.softmax(x8)
        #print(x8)
        return x8
class Model_M3(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Model_M3, self).__init__()
        self.CNN1=nn.Conv2d(1,6,kernel_size=(5,5))
        self.MP1=nn.MaxPool2d(kernel_size=2)
        self.CNN2=nn.Conv2d(6,16,kernel_size=(5,5))
        self.MP2=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,Num_Class)
    def forward(self,TS):
        x1=self.CNN1(TS)
        x1=F.relu(x1)
        x2=self.MP1(x1)
        x3=self.CNN2(x2)
        x3=F.relu(x3)
        x4=self.MP2(x3)
        #print(x4.shape)
        #x5=x4.view(x4.size(0),-1)
        x5=torch.flatten(x4, start_dim=1)
        #print(x5.shape)
        x6=self.fc1(x5)
        x7=self.fc2(x6)
        x8=self.fc3(x7)
        #x8=F.softmax(x8)
        #print(x8)
        return x8        
#Sounce_Model=Model_M1()
#Sounce_Model=Model_M2()
Sounce_Model=Model_M3()
#Target_Model=Model_M1()
Target_Model=Model_M2()
#Target_Model=Model_M3()
Source_Model=Sounce_Model.to(device)
Target_Model=Target_Model.to(device)
#Fault_Model_optimizer = optim.Adam(Fault_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function 
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc  
#Source_Model1_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN1.pt') # Define Path to save the model 
#Source_Model2_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN2.pt') # Define Path to save the model 
Source_Model3_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN3s.pt') # Define Path to save the model 
#Target_Model1_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN1.pt') # Define Path to save the model 
Target_Model2_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN2.pt') # Define Path to save the model 
#Target_Model3_SAVE_PATH = os.path.join("/home/mani/Desktop/IJCNN 2022/Bear Fault Detection", 'Fault_CNN3s.pt') # Define Path to save the model 

#Source_Model.load_state_dict(torch.load(Source_Model1_SAVE_PATH)) 
#Source_Model.load_state_dict(torch.load(Source_Model2_SAVE_PATH))
Source_Model.load_state_dict(torch.load(Source_Model3_SAVE_PATH))
#Target_Model.load_state_dict(torch.load(Target_Model1_SAVE_PATH))
Target_Model.load_state_dict(torch.load(Target_Model2_SAVE_PATH))
#Target_Model.load_state_dict(torch.load(Target_Model3_SAVE_PATH))

import torchattacks
#attack=torchattacks.BIM(Source_Model, 32/255,1/255,0)
#attack=torchattacks.FGSM(Source_Model,eps=32/255)
attack=torchattacks.PGD(Source_Model,32/255,1/255)
#attack=torchattacks.CW(Source_Model,c=1, kappa=0, lr=0.01)
def evaluate1(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    ADV_Dist=0
    all_preds = torch.tensor([])
    all_preds=all_preds.to(device)
    model.eval() # call model object for evaluation 
    #with torch.no_grad(): # Without computation of gredient 
    for (x,y) in iterator:
        x=x.float()
        x=x.to(device) # Transfer data to device 
        y=y.to(device) # Transfer label  to device 
        y=y.long()
        count=count+1
        adv_images=attack(x,y)
        adv_images=adv_images.to(device)
        L2_Dist=torch.norm(torch.abs(adv_images-x))
        x=x.detach()
        Predicted_Label = model(adv_images) # Predict claa label
        preds = (nn.functional.softmax(model(adv_images),dim=1)).max(1,keepdim=True)[1]
        all_preds = torch.cat((all_preds, preds.float()),dim=0) 
        loss = criterion(Predicted_Label, y) # Compute Loss 
        acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
        #print("Validation Iteration Number=",count)
        epoch_loss += loss.item() # Compute Sum of  Loss 
        epoch_acc += acc.item() # Compute  Sum of Accuracy
        ADV_Dist=ADV_Dist+L2_Dist   
    return epoch_loss / len(iterator), epoch_acc / len(iterator),all_preds, ADV_Dist/len(iterator), all_preds 
def Class_Distribution(Class_Dist):
    arr=Class_Dist.detach().cpu().numpy()
    uniqueValues, occurCount = np.unique(arr, return_counts=True)
    occurCount=(occurCount/len(arr))*100
    print("Unique Classes=",uniqueValues)
    print("Class Distribution=",occurCount)
test_loss, test_acc,Class_Dist, ADV_Dist,Adv_test_preds = evaluate1(Target_Model, device, test_loader, criterion) # Compute Test Accuracy on Unseen Signals 
#Class_Distribution(Class_Dist)
test_acc=100-(test_acc*100)
print("|Test Loss=",test_loss,"Test Accuracy=",test_acc)

