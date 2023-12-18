import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from hw_diff.model.DDPM.utils import extract_into_tensor, get_named_beta_schedule

class Diffusion:
    def __init__(
        self,
        *,
        betas: torch.Tensor
    ):
        """
        Class that simulates Diffusion process. Does not store model or optimizer.
        """
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]), ], dim=0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * betas

        # log calculation clipped because posterior variance is 0.
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]], dim=0)
        )
        self.posterior_mean_coef1 = alphas.sqrt() * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = self.alphas_cumprod_prev.sqrt() * betas / (1 - self.alphas_cumprod)

    def q_mean_variance(self, x0, t):
        """
        Get mean and variance of distribution q(x_t | x_0). Use equation (1).
        """

        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        variance = extract_into_tensor(1 - self.alphas_cumprod, t, x0.shape)
        log_variance = variance.log()
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute mean and variance of diffusion posterior q(x_{t-1} | x_t, x_0).
        Use equation (2) and (3).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
            extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_start
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse data for a given number of diffusion steps.
        Sample from q(x_t | x_0).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        mean, variance, _ = self.q_mean_variance(x_start, t)
        return mean + variance.sqrt() * noise

    def p_mean_variance(self, model_output, x, t):
        """
        Apply model to get p(x_{t-1} | x_t). Use Equation (2) and plug in \hat{x}_0;
        """
        model_variance = torch.cat([self.posterior_variance[1:2], self.betas[1:]], dim=0)
        model_log_variance = torch.log(model_variance)
        model_variance = extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        model_mean = self.q_posterior_mean_variance(pred_xstart, x, t)[0]

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Get \hat{x0} from epsilon_{theta}. Use equation (4) to derive it.
        """

        return extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps

    def p_sample(self, model_output, x, t):
        """
        Sample from p(x_{t-1} | x_t).
        """
        out = self.p_mean_variance(model_output, x, t)
        noise = torch.randn_like(x)
        nonzero_mask = extract_into_tensor(t > 0, timesteps=None, broadcast_shape=noise.shape)
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample}

    def p_sample_loop(self, model, shape, device):
        """
        Samples a batch=shape[0] using diffusion model.
        """
        x = torch.randn(*shape, device=device)

        for i in tqdm(reversed(range(self.num_timesteps))):
            t = torch.tensor([i] * shape[0], device=x.device)
            with torch.no_grad():
                model_output = model(x, t)
                out = self.p_sample(
                    model_output,
                    x,
                    t
                )
                x = out["sample"]
        return x

    def __call__(self, model, x0):
        t = torch.randint(0, self.num_timesteps, size=(x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_output = model(x_t, t)
        return {
            "pred_noise": model_output,
            "gt_noise": noise
        }