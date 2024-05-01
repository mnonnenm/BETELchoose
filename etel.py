import numpy as np
import torch
import scipy.optimize as spo
from scipy.special import logsumexp

dtype = torch.float64
dtype_np = np.float64

def log_w_opt(𝜆, G):
    𝜂 = torch.bmm(G, 𝜆.unsqueeze(-1)).squeeze(-1)
    return 𝜂 - torch.logsumexp(𝜂,axis=-1).unsqueeze(-1)

def Φ(𝜆, G):
    𝜂 = torch.matmul(G, 𝜆.unsqueeze(-1)).squeeze(-1)
    return torch.logsumexp(𝜂, axis=-1)

def Φ_np(𝜆, G):
    𝜂 = G.dot(𝜆)    
    return logsumexp(𝜂, axis=-1)

def gradΦ(𝜆, G):
    w = torch.exp(log_w_opt(𝜆, G))
    return torch.bmm(w.unsqueeze(-2), G).squeeze(-2)

def solve_𝜆(G):
    with torch.no_grad():
        N, K = G.shape[0], G.shape[-1]
        𝜆0 = np.zeros((N, K))
        𝜆 = np.zeros_like(𝜆0)
        for i in range(N):
            def Φ_G(𝜆):
                return Φ_np(𝜆, G[i].numpy())
            𝜆[i] = spo.minimize(Φ_G, 𝜆0[i])['x']
    return torch.tensor(𝜆,dtype=dtype)

def log_pX𝜃(g, X, 𝜃, eps=1e-4):
    G = g(X, 𝜃)
    𝜆 = solve_𝜆(G)
    log_w = log_w_opt(𝜆, G)
    log_p = log_w.sum(axis=-1)

    # check for estimating equation constraints to hold
    GTw = torch.bmm(torch.exp(log_w).unsqueeze(-2), G).transpose(-1,-2)
    idx_n_bad = torch.abs(GTw.squeeze(-1)).mean(axis=-1) >= eps
    log_p[idx_n_bad] = - torch.inf # likelihood = 0 if origin not in convex hull !
    
    return log_p

def comp_dFd𝜆(𝜆, G, GTdiagw):

    F = gradΦ(𝜆, G) # F(𝜆) = dΦd𝜆
    dFd𝜆 = torch.bmm(GTdiagw, G) - torch.bmm(F.unsqueeze(-1), F.unsqueeze(-2))

    return dFd𝜆

def comp_dFd𝜙(𝜆, G, w, GTw, GTdiagw, dGd𝜙):

    M = torch.eye(𝜆.shape[-1]).unsqueeze(0) - torch.bmm(GTw, 𝜆.unsqueeze(-2))

    dFd𝜙 = {}
    for (p, dGdp) in dGd𝜙.items():
        dFdp = torch.bmm(M, (dGdp.flatten(3) * w.unsqueeze(-1).unsqueeze(-1)).sum(axis=1))
        dFd𝜙[p] = dFdp + torch.bmm(GTdiagw, (dGdp.flatten(3) * 𝜆.unsqueeze(-2).unsqueeze(-1)).sum(axis=-2))

    return dFd𝜙

def grad_log_pX𝜃(g, X, 𝜃, eps=1e-4):

    N,T = X.shape[:2]                               # N x T x D
    G = g(X, 𝜃)                                     # N x T x K
    𝜆 = solve_𝜆(G)                                  # N     x K  
    w = torch.exp(log_w_opt(𝜆, G))                  # N x T

    GTw = torch.bmm(w.unsqueeze(-2), G).transpose(-1,-2)
    idx_n_good = torch.abs(GTw.squeeze(-1)).mean(axis=-1) < eps
    if idx_n_good.sum() < N:
        print(f"warning, {N - idx_n_good.sum()} out of {N} datapoints have zero likelihood and no gradient")
        X, 𝜃, G, 𝜆, w, GTw = X[idx_n_good], 𝜃[idx_n_good], G[idx_n_good], 𝜆[idx_n_good], w[idx_n_good], GTw[idx_n_good]

    GTdiagw = (G  * w.unsqueeze(-1)).transpose(-1,-2)
    dGd𝜙=g.jacobian_pars(X, 𝜃)                      # struct of { par : N x T x K x dim(par) }

    # inverse function theorem 
    dFd𝜆 = comp_dFd𝜆(𝜆, G, GTdiagw)                 # N     x K x K
    dFd𝜙 = comp_dFd𝜙(𝜆, G, w, GTw, GTdiagw, dGd𝜙)  # struct of { par : N x K x dim(par) }
    d𝜆d𝜙 = {}                                       # struct of { par : N x K x dim(par) }
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

    N = X.shape[0]
    idx = torch.arange(N)
    idx𝜃, idxX = torch.repeat_interleave(idx, N), idx.repeat(N)
    log_p_all = log_pX𝜃(g, X[idxX], 𝜃[idx𝜃]).reshape(N,N)
    log_normalizers = torch.logsumexp(log_p_all, axis=-1)
    log_p = torch.diag(log_p_all)
    return (log_p - log_normalizers).sum(axis=0)

def grad_InfoNCE(g, X, 𝜃):

    N = X.shape[0]
    assert 𝜃.shape[0] == N # could untie this, but not necessary for basic usage

    idx = torch.arange(N)
    idx𝜃, idxX = torch.repeat_interleave(idx, N), idx.repeat(N)
    log_p_all = log_pX𝜃(g, X[idxX], 𝜃[idx𝜃]).reshape(N,N) # 𝜃s constant across rows, Xs constant across columns

    log_normalizers = torch.logsumexp(log_p_all, axis=-1)
    log_p = torch.diag(log_p_all)
    losses = log_p - log_normalizers

    v = (torch.eye(N) - torch.exp(log_p_all - log_normalizers.unsqueeze(-1))).unsqueeze(-1)  
    grad_log_p_all = grad_log_pX𝜃(g, X[idxX], 𝜃[idx𝜃]).reshape(N,N,-1)

    grads = (v * grad_log_p_all).sum(axis=-2)

    return grads.sum(axis=0), losses.sum(axis=0)
