import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_res(g, simulator, log_pX𝜃, dtype, setup, dist, Ts=[10,100,1000], N=50, 𝜇=0.5, 𝜎=1.0, print_modes=False):
    if setup == '1D':
        𝜃 = torch.linspace(0.2, 1., N+2)[1:-1].unsqueeze(-1) # range of test values for 𝜃
    elif setup == '2D':
        grids = np.meshgrid(np.linspace(0.1, 1., N+2)[1:-1], np.linspace(0.5, 2.0, N))
        𝜃 = torch.stack([torch.tensor(xx.flatten()) for xx in grids],axis=-1)

    plt.figure(figsize=(16,5))
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
        elif setup == '2D':
            plt.subplot(2,len(Ts),i+1)
            plt.imshow(ll.detach().numpy().reshape(N,N) - ll.max().detach().numpy(), label='Empirical log-likelihood')
            plt.title("T = "+str(T))

            plt.subplot(2,len(Ts),len(Ts)+i+1)
            plt.imshow(ll_true - ll_true.max(), label='Gaussian log-likelihood')
        if print_modes:
            print(𝜃[np.argmax(ll_true)], 𝜃[np.argmax(ll.detach().numpy())])
        if i == 0:
            plt.legend()
        plt.title("T = "+str(T))