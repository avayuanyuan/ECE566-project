import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt
def partition(m):
    # there are m+1 numbers [0,m] and want a size 2 multiset
    s = []
    for i in range(m+1):
        for j in range(i,m+1):
            s.append([i,j])
    return np.array(s)

def partition_single(m):
    s = []
    for i in range(m+1):
        s.append([i])
    return np.array(s)

def pmf(k,n,p): 
    a = factorial(n)
    b = factorial(k,exact=False)
    c = factorial(n-k,exact=False)
    A = a/b/c
    return A*np.power(p,k)*np.power((1-p),(n-k))

def FY(y):
    a = factorial(197,exact=False)
    b = factorial(125 -y,exact=False)
    c = factorial(y,exact=False)
    d = factorial(18,exact=False)
    e = factorial(20,exact=False)
    f = factorial(34,exact=False)
    A = 1./b/c*1.0e200
    return A*np.power(0.5,125-y)

def gradient(theta_t,theta,s, P_S,F_Y, get_loss = False):
    P_s = np.sum(P_S[s],axis=1)
    F_y = F_Y[s]
    term1_array = F_y*np.power(theta/4.,s)
    term1_up = np.sum(term1_array*s,axis=1)
    term1_dn = np.sum(term1_array,axis=1)
    term1 = 1./theta*term1_up/term1_dn
    term2 = 18.0/(theta-1.)
    term3 = 10.0/(theta-1.)
    term4 = 30.0/(theta)
    gradient =  np.sum(P_s*(term1+term2+term3+term4))
    if get_loss:
        log = np.sum(F_y * np.power(theta/4.,s)*((1-theta)/4.)**38 * (theta/4.)**34,axis=1)
        loss = np.sum(P_s*np.log(log))
        return gradient, loss/s.shape[0]
    return gradient


def numerical_gradient(theta_t,theta0,s,P_S,F_Y,get_loss=False):
    P_s = np.sum(P_S[s],axis=1)
    F_y = F_Y[s]
    theta = theta0-1.0e-10
    log_s = np.sum(F_y * np.power(theta/4.,s)*((1-theta)/4.)**38 * (theta/4.)**34,axis=1)
    loss_s = np.sum(P_s*np.log(log_s))
    theta = theta0+1.0e-10
    log_b = np.sum(F_y * np.power(theta/4.,s)*((1-theta)/4.)**38 * (theta/4.)**34,axis=1)
    loss_b = np.sum(P_s*np.log(log_b))
    grad = (loss_b-loss_s)/2.*1.0e10
    if get_loss:
        return grad,loss_s
    return grad


def train(epochs,theta_t,S,F_Y,batch_size=100):
    P_S = pmf(np.array(range(126)),125,theta_t/4./(0.5+theta_t/4.))
    shuffle_index = np.arange(len(S))
    stratify_index = [int(i*batch_size) for i in range(int(np.floor(len(S)/batch_size)) )]
    stratify_index.append(len(S))
    cycle = len(stratify_index)-1
    theta = theta_t
    loss_list = []
    loss_x = []
    lr = 0.1
    for epoch in range(epochs):
        np.random.shuffle(shuffle_index)
        shuffledS = S[shuffle_index,:]
        total_loss = 0.0
        #if epoch > 30:
        lr = lr/np.sqrt(epoch/10.0+1)
        for i in range(cycle):
            index = np.arange(stratify_index[i%cycle],stratify_index[i%cycle+1])
            s = shuffledS[index,:]
            g, loss = numerical_gradient(theta_t,theta,s,P_S,F_Y,True)
            theta = theta +lr/batch_size*g
            total_loss += loss
            #if i%10 ==0 :
            #    loss_list.append(total_loss/(i+1))
            #    loss_x.append(epoch+1.*i/cycle)
        loss_list.append(total_loss/len(S))
    #plt.plot(loss_list)
    #plt.plot(loss_x,loss_list)
    #plt.show()
    return theta,loss_list

def MEM(iterations,epochs):

    F_Y = FY(np.arange(126))
    S = partition(125)
    theta_list = [0.3]
    plt.figure()
    total_loss = np.zeros((iterations,epochs))
    for it in range(iterations):
        theta_t = theta_list[it]
        theta,loss_list = train(epochs,theta_t,S,F_Y)
        theta_list.append(theta)
        total_loss[it,:] = np.array(loss_list)
        #plt.plot(np.array(loss_list),label=str(it))
        #plt.yscale("log")
    total_loss =(np.max(total_loss)-total_loss)+1.0e-16
    for it in range(iterations):
        plt.plot(total_loss[it,:],label=str(it))
    plt.yscale("log")
    plt.legend(loc="best")
    plt.xlabel("iteration")
    plt.gca().invert_yaxis()
    plt.show()

    print "MEM", theta_list
    return theta_list

def EM(iterations,epochs):

    F_Y = FY(np.arange(126))
    S = partition_single(125)
    theta_list = [0.3]
    
    for it in range(iterations):
        theta_t = theta_list[it]
        theta, loss_list = train(epochs,theta_t,S,F_Y)
        theta_list.append(theta)
    print "EM",theta_list
    return theta_list
#FY(np.arange(126))
iterations=10
epochs=50
theta_list = MEM(iterations,epochs)
theta_list_EM = EM(iterations,epochs)
plt.plot(theta_list[1:],'o-',label="MEM")
plt.plot(theta_list_EM[1:],'o-',label="EM")
plt.legend(loc="best")
plt.xlabel("iteration")
plt.show()
