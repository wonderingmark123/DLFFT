import numpy as np
from ast import Num
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from models import NNLiner
from tqdm import trange
class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self,fileFolder="..\\data\\GeneratorMatlab\\"
                    ,TotalNum = 1000):
                    
        if not os.path.isdir(fileFolder):
            raise IOError(' This filefolder doesn\'t exist {:} '.format(fileFolder))
        self.fileFolder = fileFolder
        self.decay = np.load(os.path.join(fileFolder,'DecayBox')).astype(np.float32)
        self.freq = np.load(os.path.join(fileFolder,'FrequencyBox')).astype(np.float32)
        # loading data
        self.len = TotalNum
        self.ImgName = os.path.join(self.fileFolder
                    ,'InputImage')
        self.images = []
        for index in range(TotalNum):
            x = np.load("".join([self.ImgName,str(index+1)]))
            self.images.append(np.abs(x).astype(np.float32))
        

    def __getitem__(self, index):
        # x = np.load("".join([self.ImgName,str(index+1)]))
        # x = np.abs(x)
        return  self.images[index], self.decay[index],self.freq[index]
    def imsize(self):
        return np.size(self.images[0])
    def __len__(self):
        return self.len
def SaveModel(model,FileFolder,TrainingLosses,epoch):
    if not os.path.isdir(FileFolder):
        os.mkdir(FileFolder)
    state = {
        'net':model.state_dict(), 
        'TrainingLosses': TrainingLosses,
        'epoch':epoch,
    }
    torch.save(state,os.path.join(FileFolder,'Modelpara.pth'))
def LoadModel(model,SaveModelFile):
    state  = torch.load(os.path.join(SaveModelFile,'Modelpara.pth'))
    model.load_state_dict(state['net'])
    epoch               = state['epoch']
    epochTrainingLoss   = state['TrainingLosses']
    return model,epoch,epochTrainingLoss
def Testing(
    batchSize   = 64,
    numWorkers  = 2,
    LinerPara   = [200,100,9],
    PinMemory   = True,
    Numtesting      = 200,
    learning_rate = 1e-4,
    decay       = 1e-6,
    ModelSaveFolder = "..\\data\\Models\\NNliner1000",
    DataFolder      = "..\\data\\GeneratorMatlabTest/"):
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ",device)
    Testset = DealDataset(DataFolder,TotalNum=Numtesting)
    Imsize   = Testset.imsize()
    LinerPara = np.append(Imsize , LinerPara)
    model = NNLiner(LinerPara=LinerPara)
    model = model.to(device)
    trainloader = DataLoader(Testset,batchSize,shuffle=True,pin_memory=PinMemory,num_workers=numWorkers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    model,epochNow,epochTrainingLoss = LoadModel(model,ModelSaveFolder)
    plt.plot(range(len(epochTrainingLoss)),epochTrainingLoss)
    plt.show()
    with torch.no_grad():
        model.zero_grad()
        test_loss  = []
        for batch, (Image, Decay, Freq) in enumerate(trainloader):
            # optimizer.zero_grad()
            model.zero_grad()
            Image, Decay,Freq = Image.to(device), Decay.to(device),Freq.to(device)
            
            FreqPred = model(Image)
            loss = F.mse_loss(FreqPred,Freq)
            print(FreqPred[0,:])
            print(Freq[0,:])
            test_loss.append(loss.item() )

def Training(
    batchSize   = 64,
    numWorkers  = 2,
    LinerPara   = [200,100,9],
    PinMemory   = True,
    epochs      = 100,
    learning_rate = 1e-4,
    decay       = 1e-6,
    totalNum    = 1000,
    ModelSaveFolder = "..\\data\\Models\\NNliner1000",
    DataFolder      = "..\\data\\GeneratorMatlab\\"):
    
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ",device)
    trainset = DealDataset(DataFolder,TotalNum=totalNum)
    Imsize   = trainset.imsize()
    LinerPara = np.append(Imsize , LinerPara)
    model = NNLiner(LinerPara=LinerPara)
    model = model.to(device)
    trainloader = DataLoader(trainset,batchSize,shuffle=True,pin_memory=PinMemory,num_workers=numWorkers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, factor=0.9)
    epoch_trainLoss = []
    maxLoss         = 99999
    for epoch in trange(0,epochs):
        # ----------------------------------------------------------
        #                   training process
        # ----------------------------------------------------------
        model.train()
        train_losses = []; 
        for batch, (Image, Decay, Freq) in enumerate(trainloader):
            # optimizer.zero_grad()
            model.zero_grad()
            Image, Decay,Freq = Image.to(device), Decay.to(device),Freq.to(device)
            
            FreqPred = model(Image)
            loss = F.mse_loss(FreqPred,Freq)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() )

        # print('Train: {}, Loss: {}'.format(epoch, np.mean(train_losses)))
        epoch_trainLoss.append(np.mean(train_losses))
        if epoch_trainLoss[-1]< maxLoss and epoch>80:
            SaveModel(model,ModelSaveFolder,epoch_trainLoss,epoch)
    return