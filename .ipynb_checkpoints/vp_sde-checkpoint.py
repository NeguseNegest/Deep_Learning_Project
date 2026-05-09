import torch
import math

class VP_SDE:
    def __init__(self, params):
        self.beta_min = params['beta_min'] #b0 = 0.1 in experiment
        self.beta_max = params['beta_max'] #bN = 20 in experiment
        self.N = params['time_steps'] #N = 1000 in experiment
        self.dt = 1 / self.N
        self.sqrt_dt = math.torch(self.dt)
        r"""cool"""
    
    def sde(self, x, t):
        r""""Returns the drift and diffusion of VP SDE"""
        beta_t = self.beta_min + t * ( self.beta_max - self.beta_min)

        drift = - 0.5 * (beta_t) * x 
        diffusion = torch.sqrt(beta_t) * torch.randn_like(x)

        return drift, diffusion
    

    def perturbation_kernel(self, x0, t):
        r"""Returns the pertubation kernel p_{0t}(x(t) | x(0))"""
        args = -0.5 * t**2 * (self.beta_max - self.beta_min) - t * self.beta_min
        mean = torch.exp(0.5 * args) * x0
        var = 1 - torch.exp(args)
        return mean, var
    
    def sde_disc(self, x ,t):
        r"""In equation dX = f(t, X)dt + g(t, X)dW, then this produces \
        f(t,X)dt, g(t, X) \sqrt{dt}. Note that dW := N(0,I) * \sqrt{dt}!"""
        drift, diffusion = self.sde(x, t)
        return drift * self.dt + diffusion * self.sqrt_dt
    
    def reverse_sde(self, x, t, score):
        r"""Returns the drift and diffusion for the reverse SDE"""
        forward_drift, forward_diffusion = self.sde(x, t)
        score_val = score(x, t)
        reverse_drift = forward_drift - forward_diffusion[:, None, None, None]**2 * score_val
        reverse_diffusion = forward_diffusion
        return reverse_drift, reverse_diffusion
    
    def reverse_sde_disc(self, x, t):
        pass

    def normal_dist_sample(self, shape):
        return torch.randn_like(shape)


def score_function(model):
    def new_score(x, t):
        new_t = t * 999
        score = model(x, new_t)
        x0 = torch.zeros_like(x)
        _, std = VP_SDE.perturbation_kernel(x0, t)
        score /= std[:, None, None, None]
        return score
    return new_score


class EulerForward:
    def __init__(self):
        pass

    def new_sample(self, x, t, score_function, model_sde):
        r"""Euler forward for VP SDE"""
        backward_dt = - model_sde.dt
        drift, diff = model_sde.reverse_sde(x, t, score_function)
        z = torch.randn_like(x)
        x_new = x + drift * backward_dt + diff[:, None, None, None] * torch.sqrt(- backward_dt) * z
        return x_new
    

@torch.no_grad
def sample_image(sde, score_function):
    pass