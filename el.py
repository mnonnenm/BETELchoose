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

def comp_dFd𝜆(𝜆, G, GTdiagw2=None):

    w = w_opt(𝜆, G)
    T = w.shape[-1]
    GTdiagw2 = (G  * (w**2).unsqueeze(-1)).transpose(-1,-2) if GTdiagw2 is None else GTdiagw2
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

def comp_dFd𝜙(𝜆, w, GTdiagw2, dGd𝜙):

    dFd𝜙 = {}
    T = w.shape[-1]
    for (p, dGdp) in dGd𝜙.items():
        𝜆TdGdp = (dGdp.flatten(3) * 𝜆.unsqueeze(-2).unsqueeze(-1)).sum(axis=-2) # sum over K
        wTdGdp = (dGdp.flatten(3) * w.unsqueeze(-1).unsqueeze(-1)).sum(axis=1)  # sum over T
        dFd𝜙[p] = wTdGdp/T  - torch.bmm(GTdiagw2, 𝜆TdGdp)

    return dFd𝜙

def grad_log_pX𝜃(g, X, 𝜃, eps=1e-4):

    N,T = X.shape[:2]  # N x T x D
    G = g(X, 𝜃)        # N x T x K
    𝜆 = solve_𝜆(G)     # N     x K  
    w = w_opt(𝜆, G)    # N x T

    GTw = torch.bmm(w.unsqueeze(-2), G).transpose(-1,-2)
    idx_n_good = torch.abs(GTw.squeeze(-1)).mean(axis=-1) < eps
    if idx_n_good.sum() < N:
        print(f"warning, {N - idx_n_good.sum()} out of {N} datapoints have zero likelihood and no gradient")
        X, 𝜃, G, 𝜆, w, GTw = X[idx_n_good], 𝜃[idx_n_good], G[idx_n_good], 𝜆[idx_n_good], w[idx_n_good], GTw[idx_n_good]

    GTdiagw2 = (G  * (w**2).unsqueeze(-1)).transpose(-1,-2)
    dGd𝜙 = g.jacobian_pars(X, 𝜃)                    # struct of { par : N x T x K x dim(par) }

    # inverse function theorem 
    dFd𝜆 = comp_dFd𝜆(𝜆, G, GTdiagw2)         # N     x K x K
    dFd𝜙 = comp_dFd𝜙(𝜆, w, GTdiagw2, dGd𝜙)  # struct of { par : N x K x dim(par) }
    d𝜆d𝜙 = {}                                # struct of { par : N x K x dim(par) }
    for (p, dFdp) in dFd𝜙.items():
        d𝜆d𝜙[p] = - torch.linalg.solve(dFd𝜆, dFdp) # Inverse function theorem: d𝜆d𝜙 = inv(dFd𝜆) * dFd𝜙 

    # differentiating w* and g_𝜙(X,𝜃) wrt 𝜙        
    diff  = (1. - T * w).unsqueeze(-1)              # N x T x 1
    grad = {}                                       # struct of { par : N x T x dim(par) }
    for (p, dGdp) in dGd𝜙.items():
        grad[p] = torch.zeros((N, np.prod(dGdp.shape[3:]))) # N x dim(par)
        grad[p][idx_n_good] = torch.bmm(𝜆.unsqueeze(-2), (diff.unsqueeze(-1) * dGdp.flatten(3)).sum(axis=1)).squeeze(-2)
    for (p, d𝜆dp) in d𝜆d𝜙.items():
        grad[p][idx_n_good] = grad[p][idx_n_good] + torch.bmm((diff * G).sum(axis=-2).unsqueeze(-2), d𝜆dp).squeeze(-2)

    return grad

def loss_InfoNCE(g, X, 𝜃):

    pass

    return (log_p - log_normalizers).sum(axis=0)

def grad_InfoNCE(g, X, 𝜃):

    pass

    return grads.sum(axis=0), losses.sum(axis=0)
