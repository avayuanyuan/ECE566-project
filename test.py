import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def sampler(N):
    x1 = np.random.uniform(3.0,4.5,N)
    y1 = np.random.uniform(4.,8.,N)
    x2 = np.random.uniform(3.5,8.5,N)
    y2 = np.random.uniform(2.0,5.0,N)
    U = np.random.uniform(0.0,1.0,N)

    X1 = np.vstack([x1,y1]).T
    X2 = np.vstack([x2,y2]).T

    U = np.reshape(U<0.5,(U.shape[0],1))

    return U*X1 +(1-U)*X2

def sampler_(N,pi,mu1,sigma1,mu2,sigma2):
    u = np.random.uniform(0.0,1.0,N)
    u = u<pi
    g1 = np.random.normal(mu1,sigma1,N)
    g2 = np.random.normal(mu2,sigma2,N)
    sample = (1-u)*g1 + u*g2
    add_sample = sample[np.random.randint(N)]
    sample = np.hstack( [sample, add_sample,add_sample,add_sample])
    np.random.shuffle(sample)
    return sample


def GETMax(params_t,sample):
    pi_t,mu1_t,sigma1_t,mu2_t,sigma2_t = params_t
    phi1 = norm.pdf(sample,mu1_t,np.sqrt(sigma1_t))
    phi2 = norm.pdf(sample,mu2_t,np.sqrt(sigma2_t))
    ruberic1 = (1.0-pi_t)*phi1
    ruberic2 = pi_t*phi2
    normalize = (ruberic1+ruberic2)
    ruberic1 = ruberic1/normalize
    ruberic2 = ruberic2/normalize
    new_mu1 = np.sum(ruberic1*sample) /np.sum(ruberic1)
    new_sigma1 = np.sum(ruberic1*(sample-mu1_t)**2)/np.sum(ruberic1)
    new_mu2 = np.sum(ruberic2*sample)/np.sum(ruberic2)
    new_sigma2 = np.sum(ruberic2*(sample-mu2_t)**2)/np.sum(ruberic2)
    new_pi = np.sum(ruberic2)/(ruberic2.shape[0])
    new_params = [new_pi,new_mu1,new_sigma1,new_mu2,new_sigma2]
    return np.array(new_params)


def EM(N):
    pi,mu1,sigma1,mu2,sigma2 = [0.6,-5,2.,5,2.]
    sample = sampler_(N,pi,mu1,sigma1,mu2,sigma2)
    #sample = [6.998,7.163,5.877,-7.577,-4.299,10.176,1.517,5.645,7.565,\
    #        2.866,-7.549,3.480,-3.098,5.228,11.371,-1.961,5.517,-4.569]
    sample = np.array(sample)
    iterations = 30
    sample_size= 10000
    params_t = [0.3,0.,4.,0.5,4.]
    params_list = [params_t]
    not_converge=True
    it = 0
    while not_converge:
    #for it in range(iterations):
        #pi_t,mu1_t,sigma1_t,mu2_t,sigma2_t = params_t
        #S = MonteCarlo(sample_size,pi_t,mu1_t,sigma1_t,mu2_t,sigma2_t,sample)
        #new_params = GETMax(params_t,S,sample)
        new_params = GETMax(params_t,sample)
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
        it +=1
        params_list.append(new_params)
    return np.array(params_list)

print EM(20)
