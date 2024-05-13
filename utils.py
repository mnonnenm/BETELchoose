import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.special import loggamma

from estimating_equations import EE_linear, EE_quadratic

def toy_setups(setup):

    assert setup in ['1D', '2D']
    if setup == '1D':
        M = 1
        EE = EE_linear
    elif setup == '2D':
        M = 2
        EE = EE_quadratic

    D = 1 # univariate observations x_j
    K = M # use as many estimating equations as there are model parameters

    g = EE(D,M,K)

    # manually set estimating equations to g(x,𝜃) = x - 𝜃 such that the model fits the mean 𝜃 = E[X]
    d = g.state_dict()
    if setup == '1D':
        d['A'] = torch.tensor([[1.0,-1.0]])
        d['c'] = torch.tensor([0.0])

    elif setup == '2D':
        d['A'] = torch.tensor([[1.0,-1.0, 0.0],
                               [0.0, 0.0,-1.0]])
        d['B'] = torch.tensor([[[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
                               [[1.0, 0.0, 0.0],
                                [0.0,-1.0, 0.0],
                                [0.0, 0.0, 0.0]]])
        d['c'] = torch.tensor([0.0, 0.0])    
    g.load_state_dict(d)

    return g

def toy_simulator(T, 𝜇, 𝜎=1.0, dist='gauss'):
    assert dist in ['gauss', 'gamma']
    N = np.size(𝜇)
    if dist == 'gauss':
        x = np.atleast_2d(𝜎).T * np.random.normal(size=(N,T)) + np.atleast_2d(𝜇).T
    elif dist == 'gamma':
        assert np.all(𝜇 >= 0.0)
        s = 𝜎**2 / 𝜇
        k = 𝜇 / s
        x = np.zeros((N,T))
        if N > 1:
            for i in range(N):
                x[i] = np.random.gamma(size=T, shape=k[i], scale=s[i])
        else:
                x = np.random.gamma(size=T, shape=k, scale=s).reshape(1,-1)
        
    return x 

def plot_res(g, simulator, log_pX𝜃, dtype, setup, dist, Ts=[10,100,1000], N=50, 𝜇=0.5, 𝜎=1.0, print_modes=False):
    if setup == '1D':
        𝜃 = torch.linspace(0.2, 1., N+2)[1:-1].unsqueeze(-1) # range of test values for 𝜃
        plt.figure(figsize=(16,5))
    elif setup == '2D':
        grids = np.meshgrid(np.linspace(0.1, 1., N+2)[1:-1], np.linspace(0.5, 2.0, N))
        𝜃 = torch.stack([torch.tensor(xx.flatten()) for xx in grids],axis=-1)
        plt.figure(figsize=(16,12))

    for i,T in enumerate(Ts):

        X = torch.tensor(simulator(T, 𝜇=𝜇, 𝜎=𝜎, dist=dist), dtype=dtype).unsqueeze(-1)
        if setup == '1D':
            X = X[0].unsqueeze(0).repeat(N,1,1) # fix one dataset
        elif setup == '2D':
            X = X[0].unsqueeze(0).repeat(𝜃.shape[0],1,1) # fix one dataset
        ll = log_pX𝜃(g, X, 𝜃)

        if setup == '1D':
            if dist == 'gauss':
                ll_true = (- 0.5 * T/𝜎**2 * (𝜃.squeeze(1) - X.mean(axis=(1,2)))**2).detach().numpy()
            elif dist == 'gamma':
                s = 𝜎**2 / 𝜃
                k = 𝜃 / s
                ll_true = ((k-1) * np.log(X.squeeze(-1)) -X.squeeze(-1)/s - k * np.log(s) - loggamma(s)).mean(axis=1)
        elif setup == '2D':
            if dist == 'gauss':
                ll_true = (- 0.5 * T * torch.log(𝜃[:,1]) - (0.5/(𝜃[:,1:]) * (𝜃[:,:1] - X.squeeze(-1))**2).sum(axis=-1)).detach().numpy().reshape(N,N)
            elif dist == 'gamma':
                raise NotImplemented()

        if setup == '1D':
            plt.subplot(1,len(Ts),i+1)
            plt.plot(𝜃.detach().numpy(), ll.detach().numpy() - ll.max().detach().numpy(), 
                     'x', color='orange', label='Empirical log-likelihood')
            plt.plot(𝜃.detach().numpy(),ll_true - ll_true.max(), 
                     '.', color='b', label='Gaussian log-likelihood') 
            plt.xlabel(r'mean parameter estimate $\theta = \hat{\mu}$')
            plt.ylabel('Log (empirical) likelihood')
        elif setup == '2D':
            plt.subplot(2,len(Ts),i+1)
            plt.imshow(ll.detach().numpy().reshape(N,N) - ll.max().detach().numpy())
            if i == 0:
                plt.ylabel('Empirical log-likelihood')
            else:
                plt.ylabel(r'variance parameter estimate $\theta = \hat{\sigma^2}$')
            plt.xlabel(r'mean parameter estimate $\theta = \hat{\mu}$')
            plt.xticks([0, N-1], [0.1, 1.0])
            plt.yticks([0, N-1], [0.5, 2.0])                
            plt.title("T = "+str(T))

            plt.subplot(2,len(Ts),len(Ts)+i+1)
            plt.imshow(ll_true - ll_true.max())
            if i == 0:
                plt.ylabel('Gaussian log-likelihood')
            else:
                plt.ylabel(r'variance parameter estimate $\theta = \hat{\sigma^2}$')
            plt.xlabel(r'mean parameter estimate $\theta = \hat{\mu}$')
            plt.xticks([0, N-1], [0.1, 1.0])
            plt.yticks([0, N-1], [0.5, 2.0])

        if print_modes:
            print(𝜃[np.argmax(ll_true)], 𝜃[np.argmax(ll.detach().numpy())])
        if i == 0:
            plt.legend()
        plt.title("T = "+str(T))