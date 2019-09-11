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
    norm1 = norm(mu1_t,sigma1_t)
    norm2 = norm(mu2_t,sigma2_t)
    phi1 = norm1.pdf(sample)
    phi2 = norm2.pdf(sample)
    ruberic1 = (1.0-pi_t)*phi1
    ruberic2 = pi_t*phi2
    normalize = (ruberic1+ruberic2)
    ruberic1 = ruberic1/normalize
    S = np.ones((len(sample),sample_size))
    U = np.random.uniform(0.,1.,(len(sample),sample_size))

    ruberic1 = np.reshape(ruberic1,(ruberic1.shape[0],1))
    S[np.where(U<ruberic1)] = 0
    del norm1, norm2
    return S.astype(int)

def pdf(Y,mu,sigma):
    norm1 = norm(mu,sigma)
    ans = norm1.pdf(Y)
    del norm1
    return ans


def CalQ(params,s,sample):
    pi,mu1,sigma1,mu2,sigma2 = params
    pdf1 = (1.0-pi) *pdf(sample,mu1,sigma1)
    pdf2 = pi*pdf(sample,mu2,sigma2)
    pdf1 = pdf1.reshape((pdf1.shape[0],1))
    pdf2 = pdf2.reshape((pdf2.shape[0],1))
    log = (1-s)*pdf1 + s*pdf2
    log = np.sum(np.log(log),axis=0)
    answer = np.sum(log)
    return answer

#def GETMax(params,s,sample):
#    def CalQ_pi(pi,mu1,sigma1,mu2,sigma2,s,sample):
#        params = np.array([pi,mu1,sigma1,mu2,sigma2])
#        return -CalQ(params,s,sample)
#    def CalQ_mu1(mu1,pi,sigma1,mu2,sigma2,s,sample):
#        params = np.array([pi,mu1,sigma1,mu2,sigma2])
#        return -CalQ(params,s,sample)
#    def CalQ_sigma1(sigma1,pi,mu1,mu2,sigma2,s,sample):
#        params = np.array([pi,mu1,sigma1,mu2,sigma2])
#        return -CalQ(params,s,sample)
#    def CalQ_mu2(mu2,pi,mu1,sigma1,sigma2,s,sample):
#        params = np.array([pi,mu1,sigma1,mu2,sigma2])
#        return -CalQ(params,s,sample)
#    def CalQ_sigma2(sigma2,pi,mu1,sigma1,mu2,s,sample):
#        params = np.array([pi,mu1,sigma1,mu2,sigma2])
#        return -CalQ(params,s,sample)
#    new_params = np.copy(params)
#    pi,mu1,sigma1,mu2,sigma2 = new_params
#    new_pi = minimize_scalar(CalQ_pi, bounds=(0.,1.0),args=(mu1,sigma1,mu2,sigma2,s,sample)).x
#    new_params[0] = new_pi
#    pi,mu1, sigma1,mu2,sigma2 = new_params
#    new_mu1 = minimize_scalar(CalQ_mu1,args=(pi,sigma1,mu2,sigma2,s,sample)).x
#    new_params[1] = new_mu1
#    pi,mu1, sigma1,mu2,sigma2 = new_params
#    new_mu2 = minimize_scalar(CalQ_mu2,args=(pi,mu1,sigma1,sigma2,s,sample)).x
#    new_params[3] = new_mu2
#    pi,mu1, sigma1,mu2,sigma2 = new_params
#    new_sigma1 = minimize_scalar(CalQ_sigma1, bounds=(0.,10.0),args=(pi,mu1,mu2,sigma2,s,sample)).x
#    new_params[2] = new_sigma1
#    pi,mu1, sigma1,mu2,sigma2 = new_params
#    new_sigma2 = minimize_scalar(CalQ_sigma2, bounds=(0.,10.0),args=(pi,mu1,sigma1,mu2,s,sample)).x
#    new_params[4] = new_sigma2
#    pi,mu1, sigma1,mu2,sigma2 = new_params
#
#    return np.array(new_params)

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
    #pi,mu1,sigma1,mu2,sigma2 = [0.6,-5,2.,5,2.]
    #sample = sampler(N,pi,mu1,sigma1,mu2,sigma2)
    sample = [6.998,7.163,5.877,-7.577,-4.299,10.176,1.517,5.645,7.565,\
            2.866,-7.549,3.480,-3.098,5.228,11.371,-1.961,5.517,-4.569]
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
