import numpy as np
import torch
import scipy.optimize as spo
from scipy.optimize import fsolve
from etel import log_w_opt, Φ_np

dtype = torch.float64
dtype_np = np.float64

def w_opt(𝜆, G):

    𝜂 = torch.bmm(G, 𝜆.unsqueeze(-1)).squeeze(-1)
    T = 𝜂.shape[-1]
    w_inv =  T * (1. + 𝜂)
    #if torch.any(w_inv < 1.): # check if 𝜆 in domain !
    #    print('warning: invalid w !')

    return 1. / w_inv

def F(𝜆, G):

    w = w_opt(𝜆, G)

    return torch.bmm(w.unsqueeze(-2), G).squeeze(-2)

def comp_dFd𝜆(𝜆, G):

    w = w_opt(𝜆, G)
    T = w.shape[-1]
    GTdiagw2 = (G  * (w**2).unsqueeze(-1)).transpose(-1,-2)
    dFd𝜆 = - T * torch.bmm(GTdiagw2, G)

    return dFd𝜆

def solve_𝜆_ETEL(G):
    # We need 𝜆'g(x[i],𝜃) >= 1/T -1, or else the weights w(𝜆,𝜃) can be negative or larger than 1 !
    # To avoid Lagrange multipliers on Lagrange multipliers 𝜆, we first solve for w and then for 𝜆.
    # We use the ETEL code for finding w from an exponential family guaranteed to contain w.
    with torch.no_grad():
        N, T, K = G.shape
        assert T >= K
        𝜆0 = np.zeros(K)
        𝜆_ETEL = np.zeros((N,K))
        𝜆 = torch.zeros((N,K))
        for i in range(N):
            def Φ_G(𝜆):
                return Φ_np(𝜆, G[i].numpy())
            𝜆_ETEL[i] = spo.minimize(Φ_G, 𝜆0)['x']
        w = torch.exp(log_w_opt(torch.tensor(𝜆_ETEL,dtype=dtype), G))
        idx_K = torch.argsort(w, axis=1)[:,-K:]
        for i in range(N):
            𝜆[i] = torch.linalg.solve(G[i,idx_K[i]], 1./(T*w[i,idx_K[i]]) - 1.)
    return 𝜆

def solve_𝜆(G):
    # this version ignores the issue that the Lagrange multipliers themselves have linear constraints !
    # We need 𝜆'g(x[i],𝜃) >= 1/T -1, or else the weights w(𝜆,𝜃) can be negative or larger than 1.
    # We avoid this by initializating with a very round-about call to similar ETEL code.
    with torch.no_grad():
        N, K = G.shape[0], G.shape[-1]
        𝜆0 = solve_𝜆_ETEL(G).detach().numpy()
        𝜆 = torch.zeros((N, K))
        for i in range(N):
            def F_G(𝜆):
                return F(torch.tensor(𝜆,dtype=dtype).unsqueeze(0), G[i].unsqueeze(0)).numpy().squeeze(0)
            def jacF_G(𝜆):
                J = comp_dFd𝜆(torch.tensor(𝜆,dtype=dtype).unsqueeze(0), G[i].unsqueeze(0))
                return J.numpy().squeeze(0)
            𝜆[i] = torch.tensor(fsolve(F_G, 𝜆0[i], fprime=jacF_G), dtype=dtype)

    #assert torch.all(1./w_opt(𝜆, G) >= 1.0) # assure 𝜆 is in valid domain: 0 <= w[t] <= 1 for all t
    return 𝜆

def log_pX𝜃(g, X, 𝜃, eps=1e-4):

    G = g(X, 𝜃)
    𝜆 = solve_𝜆(G)
    w = w_opt(𝜆, G)
    log_p = torch.log(w).sum(axis=-1)

    # check for estimating equation constraints to hold
    GTw = torch.bmm(w.unsqueeze(-2), G).transpose(-1,-2)
    idx_n_bad = torch.any(1./w_opt(𝜆, G) < 1.0, axis=-1) # invalid w resp. 𝜆
    log_p[idx_n_bad] = - torch.inf
    idx_n_bad = torch.abs(GTw.squeeze(-1)).mean(axis=-1) >= eps # moment failed
    log_p[idx_n_bad] = - torch.inf
    
    return log_p

def comp_dFd𝜙(𝜆, G, w, GTw, GTdiagw, dGd𝜙):

    pass

    return dFd𝜙

def grad_log_pX𝜃(g, X, 𝜃, eps=1e-4):

    pass

    return grad

def loss_InfoNCE(g, X, 𝜃):

    pass

    return (log_p - log_normalizers).sum(axis=0)

def grad_InfoNCE(g, X, 𝜃):

    pass

    return grads.sum(axis=0), losses.sum(axis=0)
