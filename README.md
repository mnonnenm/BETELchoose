# BETELchoose
Learning estimating equations from data for Bayesian Exponentially Tilted Empirical Likelihood.

This is work in progress about learning empirical likelihood models (i.e. the estimating equations) from data.
Since merely evaluating empirical likelihoods involves solving a system of (nonlinear) equations, learning those equations requires the implicit function theorem to acquire gradients on model parameters.

Implements for Empirical Likelihoods and Exponentially Tilted Empirical Likelihoods, currently only for linear and quadratic system of estimating equations (the latter currently is bottlenecked by PyTorch, which does not make it easy to get Jacobians for torch.nn.Module() classes with respect to model parameters).

See the [development](https://github.com/mnonnenm/BETELchoose/blob/main/dev.ipynb) notebook for the current use cases.
