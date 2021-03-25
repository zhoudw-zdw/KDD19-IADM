import torch
import numpy as np
def Test(testsets,acclist,model):
    performance1 = TestWithSet(testsets[0],model)
    performance2 = TestWithSet(testsets[1],model)
    performance3 = TestWithSet(testsets[2],model)
    performance4 = TestWithSet(testsets[3],model)
    performancemean=(performance1+performance2+performance3+performance4)/4
    performancelist=np.vstack([performance1,performance2,performance3,performance4,performancemean])
    acclistthisturn=performancelist[:,0].reshape(-1,1)
    acclist=acclistthisturn if acclist is None else np.hstack([acclist,acclistthisturn])
    print('Accuracy on Dataset 1 is :  ', acclistthisturn[0].item())
    print('Accuracy on Dataset 2 is :  ', acclistthisturn[1].item())
    print('Accuracy on Dataset 3 is :  ', acclistthisturn[2].item())
    print('Accuracy on Dataset 4 is :  ', acclistthisturn[3].item())
    print('Mean Accuracy is :  ', acclistthisturn[4].item())
    return acclist


def TestWithSet(set,net):
    net.eval()
    predic=[]
    truelabel=[]
    for (images,labels) in set:
        images, labels = images.cuda(), labels.cuda()
        outputs,weight = net(images)
        _, predicted = torch.max(outputs.data, 1)
        predic+=predicted.detach().cpu().numpy().tolist()
        truelabel+=labels.detach().cpu().numpy().tolist()
    truelabel=np.array(truelabel).reshape(-1)
    performance=cal_performance(predic,truelabel)
    return performance



def cal_performance(y, true_label):
    acc = (y == true_label).sum() / len(y)
    return acc