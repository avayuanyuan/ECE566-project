import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize,minimize_scalar
import matplotlib.pyplot as plt

def sampler(N,pi,mu1,sigma1,mu2,sigma2):
    u = np.random.uniform(0.0,1.0,N)
    u = u<pi
    g1 = np.random.normal(mu1,sigma1,N)
    g2 = np.random.normal(mu2,sigma2,N)
    sample = (1-u)*g1 + u*g2
    return sample



def MonteCarlo(sample_size,pi_t,mu1_t,sigma1_t,mu2_t,sigma2_t,sample):
    phi1 = norm.pdf(sample,mu1_t,np.sqrt(sigma1_t))
    phi2 = norm.pdf(sample,mu2_t,np.sqrt(sigma2_t))
    ruberic1 = 2.0*(1-pi_t)*phi1
    ruberic2 = (1-pi_t)*phi1 + pi_t*phi2
    ruberic3 = 2.0*pi_t*phi2

    ruberic0 = np.vstack([ruberic1,ruberic2,ruberic3]).T
    ruberic = ruberic0/np.sum(ruberic0,axis=1,keepdims=True)
    ruberic[:,1] = ruberic[:,1] + ruberic[:,0]
    ruberic[:,2] = ruberic[:,2] + ruberic[:,1]
    U = np.random.uniform(0.0,1.0,(len(sample),sample_size))
    S = np.ones((len(sample),sample_size))*2
    ruberic1 = np.reshape(ruberic[:,0],(ruberic.shape[0],1))
    ruberic2 = np.reshape(ruberic[:,1],(ruberic.shape[0],1))
    S[np.where(U < ruberic1)] = 0
    S[np.where( np.logical_and(U<ruberic2,U>=ruberic1))] = 1
    return S, ruberic0

def pdf(Y,mu,sigma):
    ans = norm.pdf(Y,mu,sigma)
    return ans

def CalQ(params,s,sample,ruberic0):
    pi,mu1,sigma1,mu2,sigma2 = params
    pdf1 =(1.0-pi)* pdf(sample,mu1,np.sqrt(sigma1))
    pdf2 = pi*pdf(sample,mu2,np.sqrt(sigma2))
    pdf1 = np.reshape(pdf1,(pdf1.shape[0],1))
    pdf2 = np.reshape(pdf2,(pdf2.shape[0],1))
    log = (2-s)*pdf1 + s*pdf2
    log = np.sum(np.log(log+1.0e-16),axis=0)
    answer = np.sum(log)
    return answer/s.shape[1]

def CalQ_pi(pi,params,s,sample,ruberic0):
    params_ = np.copy(params)
    params_[0] = pi
    return CalQ(params_,s, sample,ruberic0)
def CalQ_mu1(mu1,params,s,sample,ruberic0):
    params_ = np.copy(params)
    params_[1] = mu1
    return CalQ(params_,s,sample,ruberic0)

def CalQ_sigma1(sigma1,params,s,sample,ruberic0):
    params_ = np.copy(params)
    params_[2] = sigma1
    return CalQ(params_,s,sample,ruberic0)

def CalQ_mu2(mu2,params,s,sample,ruberic0):
    params_ = np.copy(params)
    params_[3] = mu2
    return CalQ(params_,s,sample,ruberic0)

def CalQ_sigma2(sigma2,params,s,sample,ruberic0):
    params_ = np.copy(params)
    params_[4] = sigma2
    return CalQ(params_,s,sample,ruberic0)


def numerical_gradient(params,S,sample,ruberic0):
    gradient = np.zeros(len(params))
    for i in range(len(params)):
        epsilon = np.zeros(len(params))
        epsilon[i]=1.0e-10
        params_m=params-epsilon
        params_p=params+epsilon
        diff = CalQ(params_p,S,sample,ruberic0)-CalQ(params_m,S,sample,ruberic0)
        gradient[i] = diff/(2.*1.0e-10)
    return gradient

# epochs = 400, epoch>250, epoch%70 , lr/2.0, lr= 0.01
# epochs = 500, epoch>250, epoch%70, lr/2.0, lr= 0.02
# epochs = 200, epoch>100, epoch%50, lr/2.0, lr= 0.01

def train(params_t,S,sample,ruberic0,epochs =500,batch_size = 2000):
    shuffle_index = np.arange(S.shape[1])
    stratify_index = [int(i*batch_size) for i in range(int(np.floor(S.shape[1]/batch_size)) )]
    stratify_index.append(S.shape[1])
    cycle = len(stratify_index)-1
    params = params_t
    loss_list = []
    lr = 0.1
    for epoch in range(epochs):
        np.random.shuffle(shuffle_index)
        shuffledS = S[:,shuffle_index]
        total_loss=0.0
        if epoch>=250 and epoch %70 ==0 :
            lr = lr/2.0
        #lr = lr/np.sqrt(epoch+1)
        for i in range(cycle):
            index = np.arange(stratify_index[i%cycle],stratify_index[i%cycle+1])
            s = shuffledS[:,index]
            g = numerical_gradient(params,s,sample,ruberic0)
            g = g/np.linalg.norm(g)
            loss = CalQ(params,s,sample,ruberic0)
            params = params+lr*g
            total_loss +=loss
        loss_list.append(total_loss/S.shape[1])
    print params[0],params[1],np.sqrt(params[2]),params[3],np.sqrt(params[4])
    return params,loss_list
    

def MEM(N):
    #pi,mu1,sigma1,mu2,sigma2 = [0.6,-5.,2.5,5.,2.5]
    #sample = sampler(N,pi,mu1,sigma1,mu2,sigma2)
    sample = [6.998,7.163,5.877,-7.577,-4.299,10.176,1.517,5.645,7.565,\
            2.866,-7.549,3.480,-3.098,5.228,11.371,-1.961,5.517,-4.569]
    sample = np.array(sample)

    iterations = 20
    sample_size = 2000
    params_t = [0.3,0.0,4.0,0.5,4.0]
    params_list = [params_t]
    not_converge = True
    it = 0
    plt.figure()
    while not_converge:
    #for it in range(iterations):
        if it > 3:
            smaple_size=5000
        pi_t,mu1_t,sigma1_t,mu2_t,sigma2_t = params_t
        S,ruberic0 = MonteCarlo(sample_size, pi_t,mu1_t, sigma1_t,mu2_t,sigma2_t, sample)
        S = S.astype(int)
        new_params,loss_list=train(params_t,S,sample,ruberic0)
        plt.plot(loss_list,label=str(it))
        #new_pi = minimize_scalar(CalQ_pi,bounds=(0.,1.),args=(params_t,S,sample,ruberic0))
        #new_pi = new_pi.x
        #new_mu1 = minimize_scalar(CalQ_mu1,args=(params_t,S,sample,ruberic0))
        #new_mu1 = new_mu1.x
        #new_sigma1 = minimize_scalar(CalQ_sigma1,bounds=(0.,100.),args=(params_t,S,sample,ruberic0))
        #new_sigma1 = new_sigma1.x
        #new_mu2 = minimize_scalar(CalQ_mu2,args=(params_t,S,sample,ruberic0))
        #new_mu2 = new_mu2.x
        #new_sigma2 = minimize_scalar(CalQ_sigma2,bounds=(0.,100.0),args=(params_t,S,sample,ruberic0))
        #new_sigma2 = new_sigma2.x
        #new_params = np.array([new_pi,new_mu1,new_sigma1,new_mu2,new_sigma2])

        #new_params=minimize(CalQ,params_t,args=(S,sample,ruberic0))
        #new_params =new_params.x
        #err = [new_params[0]-params_t[0],new_params[1]-params_t[1],\
        #        np.sqrt(new_params[2])-np.sqrt(params_t[2]),\
        #        new_params[3]-params_t[3],\
        #        np.sqrt(new_params[4])-np.sqrt(params_t[4])]
        err = [new_params[0]-params_t[0],new_params[1]-params_t[1],\
                new_params[2]-params_t[2],\
                new_params[3]-params_t[3],\
                new_params[4]-params_t[4]]

        err = np.max(np.abs(err))
        print it, err
        if err < 0.1 :
            not_converge=False
        params_t = np.copy(new_params)
        new_params[2] = np.sqrt(new_params[2])
        new_params[4] = np.sqrt(new_params[4])
        params_list.append(new_params)
        it = it+1
    plt.legend(loc="best")
    plt.show()
    return sample,np.array(params_list)



#pi,mu1,sigma1,mu2,sigma2 = [0.6,-5.,2.5,5.,2.5]
#sample = sampler(20,pi,mu1,sigma1,mu2,sigma2)
#S,ruberic0 = MonteCarlo(3, 0.3,0.0, 2,0.5,2, sample)
#print (sample)
#params = [0.9,-5.,2.5,5.,2.5]
#print S
#print CalQ(params,S,sample,ruberic0)/(S.shape[1])
#print numerical_gradient(params,S,sample,ruberic0)/2000

sample,params_list = MEM(20)
print (sample)
print (params_list)

    


