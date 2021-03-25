import matplotlib.pyplot as plt
import numpy as np

def DrawIADM(acc,picname,filedir):
    acc1=acc[0]
    acc2=acc[1]
    acc3=acc[2]
    acc4=acc[3]
    plt.switch_backend('agg')
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    depth=len(acc1)
    x=np.linspace(1,depth,depth)
    acc1 = np.array(acc1)
    acc2 = np.array(acc2)
    acc3 = np.array(acc3)
    acc4 = np.array(acc4)
    plt.plot(x, acc1, 'r', label='$x_1$',linewidth=3)
    plt.plot(x, acc2, 'g', label='$x_2$',linewidth=3)
    plt.plot(x, acc3, 'b', label='$x_3$',linewidth=3)
    plt.plot(x, acc4, 'm', label='$x_4$',linewidth=3)
    plt.xlabel('Iteration',font)
    plt.ylim(0.4,1)
    plt.ylabel('Accuracy',font)
    plt.legend(prop=font,loc='lower right')
    plt.grid()
    plt.savefig(filedir + picname)


