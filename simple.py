import numpy as np
from scipy.misc import factorial
import time
class Binomial:
    def __init__(self,n,p):
        self.n = n
        self.p = p
    def pmf(self,k):
        if (k>self.n):
            print ("error in factorial")
            exit()
        a,b,c =factorial(np.array([self.n, k, self.n-k]),exact=True) 
        A = a/b/c
        return A*self.p**(k)*(1-self.p)**(self.n-k)

    def cdf(self,k):
        pass
        return
    def __call__(self):
        # return a sample
        return

def pmf(k,n,p): 
    a,b,c =factorial(np.array([n, k, n-k]),exact=True) 
    A = a/b/c
    return A*p**k*(1-p)**(n-k)



def partition(N):
    s = []
    for i in range(N+1):
        for j in range(i,N+1):
            a = i
            b = j-i
            c = N-j
            s.append( [a,b,c] )
    return s

def MEM(N):
    # N = multiset size
    theta = 0.1
    theta_list = np.linspace(0.0,1,101)
    theta_list = theta_list[1:-1]
    set_partition = partition(N)
    result_list = [0.1]
    p0_ = pmf(0,2,theta_list)
    p1_ = pmf(1,2,theta_list)
    p2_ = pmf(2,2,theta_list)
    P_ = np.vstack([p0_,p1_,p2_])
    time_list = []
    set_partition = np.array(set_partition)
    for epoch in range(10):
        tik = time.time()
        bi = Binomial(2,theta)
        p0 = bi.pmf(0)
        p1 = bi.pmf(1)
        p2 = bi.pmf(2)
        PS_list = np.sum(set_partition*np.array([p0,p1,p2]),axis=1,keepdims=True)
        log_list = np.log(set_partition.dot(P_)*p1_)
        #for i, s in enumerate(set_partition):
        #    PS_list[i,0] = np.sum(np.array(s)*np.array([p0,p1,p2]))
        #    p0_ = pmf(0,2,theta_list)
        #    p1_ = pmf(1,2,theta_list)
        #    p2_ = pmf(2,2,theta_list)
        #    a, b,c = s
        #    log_list[i,:] =np.log( (a*p0_ +b *p1_ + c*p2_) *p1_ )
        Q_list = np.sum(PS_list * log_list,axis=0)
        theta = theta_list[np.argmax(Q_list)]
        tok =time.time()
        time_used = tok-tik
        time_list.append(time_used)
        result_list.append(theta)
    result_list = np.array(result_list)
    diff= result_list[1:] - result_list[:-1]
    index = np.where(diff ==0)[0][0] 
    return result_list,time_list,index

def EM():
    theta = 0.1
    theta_list = np.linspace(0.0,1,101)
    theta_list = theta_list[1:-1]
    set_partition = np.identity(3)
    result_list = [0.1]
    p0_ = pmf(0,2,theta_list)
    p1_ = pmf(1,2,theta_list)
    p2_ = pmf(2,2,theta_list)
    P_ = np.vstack([p0_,p1_,p2_])
    time_list = []
    for epoch in range(10):
        tik = time.time()
        bi = Binomial(2,theta)
        p0 = bi.pmf(0)
        p1 = bi.pmf(1)
        p2 = bi.pmf(2)
        PS_list = np.array([p0,p1,p2])
        log_list = np.log(P_*p1_)
        #Q_list = np.sum(PS_list * log_list,axis=0)
        Q_list = PS_list.dot(log_list)
        theta = theta_list[np.argmax(Q_list)]
        tok =time.time()
        time_used = tok-tik
        time_list.append(time_used)
        result_list.append(theta)
    result_list = np.array(result_list)
    diff= result_list[1:] - result_list[:-1]
    index = np.where(diff ==0)[0][0] 
    return result_list,time_list,index


rl1,tl1,id1= EM()
rl2,tl2,id2 = MEM(2)
rl3,tl3,id3 = MEM(3)
rl7,tl7,id7 =MEM(7)
rl10,tl10,id10 =MEM(10)

print id1,id2,id3,id7,id10
import matplotlib.pyplot as plt
plt.plot(rl1,'o-',label='EM')
plt.plot(rl2,'o-',label="MEM 2")
plt.plot(rl3,'o-',label="MEM 3")
plt.plot(rl7,'o-',label="MEM 7")
plt.plot(rl10,'o-',label="MEM 10")

plt.legend(loc="best")
plt.show()

plt.figure()
plt.plot(tl1,'o-',label='EM')
plt.plot(tl2,'o-',label="MEM 2")
plt.plot(tl3,'o-',label="MEM 3")
plt.plot(tl7,'o-',label="MEM 7")
plt.plot(tl10,'o-',label="MEM 10")
plt.legend(loc="best")
plt.title("time used in each iteration")
plt.figure()
kl = [1,2,3,7,10]
tl = np.mean(np.vstack([tl1,tl2,tl3,tl7,tl10]),axis=1)
plt.plot(kl,tl,'o-')
plt.ylabel("time")
plt.title("average time for each iteration")

tll = [0,0,0,0,0]
tll[0] = np.sum(tl1[:id1+1])
tll[1] = np.sum(tl2[:id2+1])
tll[2] = np.sum(tl3[:id3+1])
tll[3] = np.sum(tl7[:id7+1])
tll[4] = np.sum(tl10[:id10+1])
plt.figure()
#plt.plot(kl, tl*np.array([id1,id2,id3,id7,id10]),'o-')
plt.plot(kl, tll,'o-')
plt.title("total time until convergence")
plt.show()
