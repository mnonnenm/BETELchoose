import numpy as np
import torch

class EE_linear(torch.nn.Module):
    def __init__(self, D, M, K):
        super().__init__()
        self.D = D # dimensionality of each data point x_i, i=1,...,T
        self.M = M # dimensionality of model parameter 𝜃
        self.K = K # number of estimating equations
        self.A = torch.nn.Parameter(torch.randn((K,D+M))/np.sqrt(D+M))
        self.c = torch.nn.Parameter(torch.randn((K)))
    def forward(self, x, 𝜃):
        # x.shape = N x T x D
        # 𝜃.shape = N x     M
        x_full = torch.cat((x,𝜃.unsqueeze(-2).expand(-1,x.shape[-2],-1)),dim=-1)
        # Ax.shape = N x T x K
        Ax = torch.matmul(self.A.unsqueeze(0).unsqueeze(0), x_full.unsqueeze(-1)).squeeze(-2)
        out = Ax  + self.c.unsqueeze(0).unsqueeze(0)
        return out

    def jacobian_pars(self, X, 𝜃):
        N,T,D = X.shape
        M = 𝜃.shape[-1]
        # X.shape     = N x T x D
        # 𝜃.shape     = N x     M
        X_full = torch.cat((X,𝜃.unsqueeze(-2).expand(-1,X.shape[-2],-1)),dim=-1)         # N x T x (D+M)
        eye = torch.eye(self.K).reshape(1,1,self.K,self.K).repeat(N,T,1,1).unsqueeze(-1) # N x T x K x K x 1
        dGdA = eye.repeat(1,1,1,1,D+M) * X_full.reshape(N,T,1,1,D+M)                     # N x T x K x K x D+M
        dGdc = eye.squeeze(-1)                                                           # N x T x K x K
        return {'A': dGdA, 'c' : dGdc}

class EE_quadratic(EE_linear):
    def __init__(self, D, M, K):
        super().__init__(D, M, K)
        self.B = torch.nn.Parameter(torch.randn((K,D+M,D+M))/np.sqrt(D+M))

    def forward(self, x, 𝜃):
        # x.shape = N x T x D
        # 𝜃.shape = N x     M
        x_full = torch.cat((x,𝜃.unsqueeze(-2).expand(-1,x.shape[-2],-1)),dim=-1)
        # Ax.shape = N x T x K
        Ax = torch.matmul(self.A.unsqueeze(0).unsqueeze(0), x_full.unsqueeze(-1))
        # Bx.shape = N x T x K x D
        Bx = torch.matmul(self.B.unsqueeze(0).unsqueeze(0), x_full.unsqueeze(-2).unsqueeze(-1))
        out = (Ax + torch.matmul(x_full.unsqueeze(-2).unsqueeze(-2), Bx).squeeze(-2)).squeeze(-1) + self.c.unsqueeze(0).unsqueeze(0)
        return out

    def jacobian_pars(self, X, 𝜃):
        N,T,D = X.shape
        M = 𝜃.shape[-1]
        # X.shape     = N x T x D
        # 𝜃.shape     = N x     M
        X_full = torch.cat((X,𝜃.unsqueeze(-2).expand(-1,X.shape[-2],-1)),dim=-1)         # N x T x (D+M)
        eye = torch.eye(self.K).reshape(1,1,self.K,self.K).repeat(N,T,1,1).unsqueeze(-1) # N x T x K x K x 1
        dGdA = eye.repeat(1,1,1,1,D+M) * X_full.reshape(N,T,1,1,D+M)                     # N x T x K x K x D+M
        XXT = torch.matmul(X_full.unsqueeze(-1), X_full.unsqueeze(-2))
        dGdB = eye.unsqueeze(-1).repeat(1,1,1,1,D+M,D+M) * XXT.unsqueeze(2).unsqueeze(2) # N x T x K x K x D+M x D+M
        dGdc = eye.squeeze(-1)                                                           # N x T x K x K
        return {'A': dGdA, 'B' : dGdB, 'c' : dGdc}
