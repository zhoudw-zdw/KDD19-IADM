import torch
import scipy.io as sio
from torch.utils.data import Dataset

class MNIST_dataset(Dataset):
    def __init__(self,  img_path,Train):
        self.data = sio.loadmat(img_path)
        if Train:
            self.x = self.data['trainx']
            self.y = self.data['trainy']
            print(img_path, 'Is Loaded as TRAIN File.')
        else:
            self.x = self.data['testx']
            self.y = self.data['testy']
            print(img_path,'Is Loaded as TEST File.')
    def __getitem__(self, index):
        image = torch.from_numpy(self.x[index]).float()
        label = torch.from_numpy(self.y[index]).long()
        return image,label
    def __len__(self):
        return len(self.y)

def Prepare_Concept_Drift_MNIST():
    batch_size=1
    Testset=[MNIST_dataset('./Incremental MNIST/ldx.mat',False),MNIST_dataset('./Incremental MNIST/lux.mat',False),
             MNIST_dataset('./Incremental MNIST/rdx.mat', False),MNIST_dataset('./Incremental MNIST/rux.mat',False)]
    train_loader = torch.utils.data.DataLoader(dataset=MNIST_dataset('./Incremental MNIST/trainingset.mat',True), batch_size=batch_size)
    testloader=[torch.utils.data.DataLoader(dataset=setitem, batch_size=batch_size) for setitem in Testset]
    return train_loader,testloader
