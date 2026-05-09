import torch
import math

class VP_SDE:
    def __init__(self, params):
        self.beta_min = params['beta_min']
        self.beta_max = params['beta_max']
        self.N = params['time_steps']
        self.dt = 1 / self.N
        self.sqrt_dt = torch.sqrt(torch.tensor(self.dt))
        self.T = 1
    
    def sde(self, x, t):
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion
    

    def perturbation_kernel(self, x0, t):
        int_beta = (self.beta_min * t + 0.5 * t**2 * (self.beta_max - self.beta_min))
        int_beta = int_beta.view(-1, 1, 1, 1)

        mean = torch.exp(-0.5 * int_beta) * x0
        
        # FIX 1: variance -> std (CRITICAL)
        std = torch.sqrt(1.0 - torch.exp(-int_beta))
    
        return mean, std
    
    def sde_disc(self, x ,t):
        drift, diffusion = self.sde(x, t)
        return drift * self.dt + diffusion * self.sqrt_dt
    
    def reverse_sde(self, x, t, score):
        forward_drift, forward_diffusion = self.sde(x, t)
        score_val = score(x, t)

        reverse_drift = forward_drift - (forward_diffusion[:, None, None, None] ** 2) * score_val
        reverse_diffusion = forward_diffusion

        return reverse_drift, reverse_diffusion
    

    def reverse_sde_disc(self, x, t):
        pass

    def normal_dist_sample(self, x):
        return torch.randn_like(x)



def score_function(model, sde):
    def new_score(x, t):
        score = model(x, t)
        
        # FIX 2: correct std usage
        x0 = torch.zeros_like(x)
        _, std = sde.perturbation_kernel(x0, t)

        return -score / std
    return new_score



# Their loss function (corrected minimal)
def get_loss_fn(sde, eps=1e-5):
    def loss_fn(model, batch):
        score_fn = score_function(model, sde)

        t = torch.rand(batch.shape[0], device=batch.device) * (1 - eps) + eps
        z = torch.randn_like(batch)

        mean, std = sde.perturbation_kernel(batch, t)

        # FIX 3: correct noise scaling (std is now real std)
        perturbed_data = mean + std * z

        score = score_fn(perturbed_data, t)

        losses = torch.square(score * std[:, None, None, None] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

        return torch.mean(losses)
    
    return loss_fn



class EulerForward:
    def __init__(self):
        pass

    def new_sample(self, x, t, score_function, model_sde):
        backward_dt = -model_sde.dt

        drift, diff = model_sde.reverse_sde(x, t, score_function)
        z = torch.randn_like(x)

        x_new = x + drift * backward_dt + diff[:, None, None, None] * torch.sqrt(-backward_dt) * z
        return x_new
    

@torch.no_grad()
def get_samples(x, score, sde, examples, eul_for, eps=0.01):
    batch_size = x.shape[0]

    t = torch.ones((batch_size,), device=x.device)

    # FIX 4: prior sampling (already correct idea)
    x_old = sde.normal_dist_sample(x)

    N = sde.N
    t_i = torch.linspace(1, eps, N, device=x.device)

    lists_of_progress = []

    for i in range(N):
        x_new = eul_for.new_sample(x_old, t, score, sde)

        # FIX 5: REMOVE THIS (breaks diffusion physics)
        # x_new = x_new * t_i[i]

        x_old = x_new

        if i % max(1, (N // examples)) == 0:
            lists_of_progress.append(x_old)

    return x_new, lists_of_progress