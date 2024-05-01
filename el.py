import numpy as np
import torch
import scipy.optimize as spo
from scipy.special import logsumexp

dtype = torch.float64
dtype_np = np.float64

def w_opt(𝜆, G):
    𝜂 = torch.bmm(G, 𝜆.unsqueeze(-1)).squeeze(-1)
    w_inv =  T * (1. + 𝜂)
    assert np.all(w_inv > 1.) # check if 𝜆 in domain !
    return 1. / w_inv

def F(𝜆, G):
    w = w_opt(𝜆, G)
    return torch.bmm(w.unsqueeze(-2), G).squeeze(-2)

def F_np(𝜆, G):
    w = w_opt(𝜆, G)
    return torch.bmm(w.unsqueeze(-2), G).squeeze(-2)

def comp_dFd𝜆(𝜆, G):

    w = w_opt(𝜆, G)
    T = w.shape[-1]
    GTdiagw2 = (G  * (w**2).unsqueeze(-1)).transpose(-1,-2)
    dFd𝜆 = - T * torch.bmm(GTdiagw2, G)

    return dFd𝜆

def solve_𝜆(G):
    with torch.no_grad():
        N, K = G.shape[0], G.shape[-1]
        𝜆0 = np.zeros((N, K))
        𝜆 = np.zeros_like(𝜆0)
        for i in range(N):
            def F_G(𝜆):
                return F_np(𝜆, G[i].numpy())
            def gradF_G(𝜆):
                return comp_dFd𝜆(𝜆, G[i].numpy())
            𝜆[i] = spo.newton(F_G, 𝜆0[i], fprime=gradF_G)['x']
    return torch.tensor(𝜆,dtype=dtype)

def log_pX𝜃(g, X, 𝜃, eps=1e-4):
    pass
    return log_p

def comp_dFd𝜆(𝜆, G, GTdiagw):

    pass

    return dFd𝜆

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
