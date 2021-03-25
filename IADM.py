"""
Code of Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability
If there are any questions, please feel free to contact with the authors:
    Yang Yang(yangy@lamda.nju.edu.cn) and Da-Wei Zhou(zhoudw@lamda.nju.edu.cn).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Dataset
from sklearn.preprocessing import MinMaxScaler
from Draw import DrawIADM
import datetime
from copy import deepcopy
from Test import Test
import os


from argparse import ArgumentParser
parser = ArgumentParser('IADM PyTorch Implementation Demo')
parser.add_argument('--alp', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--drawstep', type=int, default=12000)
parser.add_argument('--lamda', type=int, default=1865)
parser.add_argument('--gpu', type=int, default=4)
args = parser.parse_args()
torch.cuda.set_device(args.gpu)
min_max_scaler = MinMaxScaler()

class IADM_NET(nn.Module):
    def __init__(self):
        super(IADM_NET, self).__init__()
        self.l1 = nn.Linear(784, 1000)
        self.l2 = nn.Linear(1000, 500)
        self.l3 = nn.Linear(500, 250)
        self.l4 = nn.Linear(250, 100)
        self.l5 = nn.Linear(100, 50)
        self.llist = [self.l1, self.l2, self.l3, self.l4, self.l5]

        self.f1 = nn.Linear(1000, 10)
        self.f2 = nn.Linear(500, 10)
        self.f3 = nn.Linear(250, 10)
        self.f4 = nn.Linear(100, 10)
        self.f5 = nn.Linear(50, 10)
        self.flist = [self.f1, self.f2, self.f3, self.f4, self.f5]

        self.AttentionNet=nn.Sequential(
            nn.Linear(10,50),
            nn.ReLU(),
            nn.Linear(50,50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
    def forward(self, x):
        x = x.view(-1, 784)
        temp=F.relu(self.llist[0](x))
        o1=self.flist[0](temp)
        alpha1=self.AttentionNet(o1)

        temp = F.relu(self.llist[1](temp))
        o2 = self.flist[1](temp)
        alpha2 = self.AttentionNet(o2)

        temp = F.relu(self.llist[2](temp))
        o3 = self.flist[2](temp)
        alpha3 = self.AttentionNet(o3)

        temp = F.relu(self.llist[3](temp))
        o4 = self.flist[3](temp)
        alpha4 = self.AttentionNet(o4)

        temp = F.relu(self.llist[4](temp))
        o5 = self.flist[4](temp)
        alpha5 = self.AttentionNet(o5)

        Att_weight=torch.Tensor([alpha1,alpha2,alpha3,alpha4,alpha5]).view(-1, 1)
        Att_weight = torch.Tensor(min_max_scaler.fit_transform(Att_weight)).cuda().view(-1)
        Att_weight=F.softmax(Att_weight,dim=0)
        outputs = [o1, o2, o3, o4, o5]
        Ensemble_prob=Att_weight[0] * outputs[0] + Att_weight[1] * outputs[1] + Att_weight[2] * outputs[2] + Att_weight[3] * outputs[3] + Att_weight[4] * outputs[4]
        return Ensemble_prob,Att_weight

#hyperparameters
LR=args.lr
drawstep=args.drawstep
alp=args.alp
lmd=args.lamda
consolidate_step=60000
lamda=torch.Tensor([lmd]).cuda()
Cuda=True

def Update(optimizer,loss,net,various_loss):
    optimizer.zero_grad()
    only_ewc_loss=various_loss.ewc_loss()
    total_loss = loss+lamda * only_ewc_loss
    total_loss.backward()
    optimizer.step()
    various_loss.updata_fisher(batch_size=1)
    return only_ewc_loss

def variable(t: torch.Tensor, cuda):
    t = Variable(t).cuda() if cuda else Variable(t)
    return t

class loss_calculate(object):
    def __init__(self, model: nn.Module, alpha, cuda):
        self.model = model
        self.means = {}
        self.fisher = {}
        self.dynamic_fisher = {}
        self.cuda = cuda
        self.alpha = alpha
        self.totalfisher={}
        self.avgfisher={}
        self.fisher0={}
        self.wfisher0={}
        self.fisher1={}
        self.wfisher1 = {}
        self.fisher2={}
        self.wfisher2 = {}
        self.fisher3={}
        self.wfisher3 = {}
        self.fisherlist=[self.fisher0,self.fisher1,self.fisher2,self.fisher3]
        self.weighted_fisher_list=[self.wfisher0, self.wfisher1, self.wfisher2, self.wfisher3]
        for n, p in self.model.named_parameters():
            self.means[n] = p.data.cuda()
            self.fisher[n] = torch.FloatTensor([0]).cuda()
            self.dynamic_fisher[n] = torch.FloatTensor([0]).cuda()
            self.totalfisher[n]=torch.FloatTensor([0]).cuda()
            self.fisher0[n] = torch.FloatTensor([0]).cuda()
            self.fisher1[n] = torch.FloatTensor([0]).cuda()
            self.fisher2[n] = torch.FloatTensor([0]).cuda()
            self.fisher3[n] = torch.FloatTensor([0]).cuda()
            self.wfisher0[n] = torch.FloatTensor([0]).cuda()
            self.wfisher1[n] = torch.FloatTensor([0]).cuda()
            self.wfisher2[n] = torch.FloatTensor([0]).cuda()
            self.wfisher3[n] = torch.FloatTensor([0]).cuda()
            self.avgfisher[n] = torch.FloatTensor([0]).cuda()

    def updata_fisher(self, batch_size):
        for n, p in self.model.named_parameters():
            if self.cuda:
                self.dynamic_fisher[n] = self.dynamic_fisher[n].cuda()
            self.dynamic_fisher[n] = (1 - self.alpha) * self.dynamic_fisher[n] + self.alpha * (p.grad.data ** 2) / batch_size

    def consolidate(self,episode):
        self.fisherlist[episode]=deepcopy(self.dynamic_fisher)
        self.weight_attention(episode)
        for n, p in self.model.named_parameters():
            self.totalfisher[n] = self.totalfisher[n] + self.weighted_fisher_list[episode][n]
            self.avgfisher[n]=self.totalfisher[n]/(episode+1)
            self.means[n] = deepcopy(p.data)
            self.dynamic_fisher[n] = torch.FloatTensor([0]).cuda()

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            fisher = variable(self.avgfisher[n], self.cuda)
            mean = variable(self.means[n], self.cuda)
            _loss = fisher * (p - mean) ** 2
            loss += _loss.sum()
        return loss

    def weight_attention(self,episode):
        for n, p in self.model.named_parameters():
            if 'l1' in n or 'f1' in n:
                self.weighted_fisher_list[episode][n]= 10 * weight[0] * self.fisherlist[episode][n]
            elif 'l2' in n or 'f2' in n:
                self.weighted_fisher_list[episode][n]= 10 * weight[1] * self.fisherlist[episode][n]
            elif 'l3' in n or 'f3' in n:
                self.weighted_fisher_list[episode][n]= 10 * weight[2] * self.fisherlist[episode][n]
            elif 'l4' in n or 'f4' in n:
                self.weighted_fisher_list[episode][n]= 10 * weight[3] * self.fisherlist[episode][n]
            elif 'l5' in n or 'f5' in n:
                self.weighted_fisher_list[episode][n]= 10 * weight[4] * self.fisherlist[episode][n]

def DrawTrainingLine():
    filedir = './Result/IADM/'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    strofparameters = '_alp_' + str(alp)  + 'lamda_' + str(lmd)
    picname = 'IADM_' + nowtime + strofparameters + '.png'
    DrawIADM(acclist,picname,filedir)

if __name__ == "__main__":
    print('At ',datetime.datetime.now().strftime("%m%d-%H%M%S")," .Start Training MNIST")
    IADM = IADM_NET().cuda()
    trainset, testsets = Dataset.Prepare_Concept_Drift_MNIST()
    criterion = nn.CrossEntropyLoss()
    optimizer=optim.SGD(IADM.parameters(), lr=LR)
    sum_loss = 0.0
    epoch=0
    acclist=None
    alphalist=[]
    various_loss = loss_calculate(IADM, alpha=alp, cuda=Cuda)
    for i, data in enumerate(trainset, 0):
        IADM.train()
        length = len(trainset)
        inputs, labels = data
        labels = variable(labels.view(-1),True)
        outputs,weight = IADM(variable(inputs, True))
        loss = criterion(outputs,labels)
        ewcloss=Update(optimizer, loss, IADM, various_loss)
        sum_loss += loss.item()
        if (i+1)%drawstep==0:
            print('[epoch:%d, iter:%d] Average Loss: %.03f   ' % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1)))
            print('Clf Loss: %.05f'%(loss.item()), '\nEWC Loss: %.05f'%(ewcloss.item()),'\nTotal Loss: %.05f'%((loss+lamda * ewcloss).item()))
            print('Attention Weight : ',weight.cpu())
            acclist=Test(testsets, acclist, IADM)
            print('--------------------------------------------------------------------------------')
        if (i+1)%consolidate_step==0:
            various_loss.consolidate(int((i+1)/60000)-1)
            alphalist.append(weight)
        if i>60000 and i%60000==30000:
            LR=LR*0.9
            optimizer=optim.SGD(IADM.parameters(), lr=LR)
    nowtime = datetime.datetime.now().strftime("%m%d-%H%M%S")
    print('At ', nowtime, " Training OVER.")
    DrawTrainingLine()