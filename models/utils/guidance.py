from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import torch
from typing import Optional


from models.gcd_diff import ConstraintAnalyzer, ConstraintOutputs


class BaseGuidance:
    def __init__(self, config):
        self.config = config
        self.guidance_scale = config["guidance_scale"]

    def __call__(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ):
        raise NotImplementedError


class CFGuidance(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)

    def __call__(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ) -> torch.Tensor:
        device = input.device
        guidance_scale = self.guidance_scale
        uncond = torch.zeros_like(cond)
        # batched uncond/cond forward
        traj_in = torch.cat([input, input], dim=0)
        cond_in = torch.cat([uncond, cond], dim=0)
        t_in = t.to(device)
        model_out = model(traj_in, t_in, cond_in)
        out_uncond, out_cond = model_out.chunk(2, dim=0)
        pred_type = scheduler.config.prediction_type
        guided = out_cond
        if guidance_scale == 1.0:
            guided = out_cond
        elif guidance_scale == 0.0:
            guided = out_uncond
        elif pred_type in ("epsilon", "v_prediction"):
            guided = out_uncond + guidance_scale * (out_cond - out_uncond)
        x_prev = scheduler.step(guided, t, input).prev_sample
        return x_prev


class EnergyGuidance(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)
        self.eta = config["guidance_eta"]

    def __call__(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ) -> torch.Tensor:
        device = input.device
        eta = self.eta
        cond_zero = torch.zeros_like(cond)
        assert "analyzer" in kwargs
        analyzer: ConstraintAnalyzer = kwargs["analyzer"]
        B, N, D = input.shape
        device = input.device
        # TODO: better with dict
        # enable grads only here
        with torch.enable_grad():
            x_var = input.detach().clone().requires_grad_(True)
            # IMPORTANT: use slices of x_var so E depends on x_var
            states_dict = {
                "desired_pos": x_var[:, :, 2:4],
                "current_pos": x_var[:, :, 4:6],
            }
            actions_dict = {
                "vel": x_var[:, :, :2],
            }
            env_dict = {"goal": torch.zeros((2,), device=device, dtype=x_var.dtype)}

            out: ConstraintOutputs = analyzer(
                states=states_dict,
                actions=actions_dict,
                environment=env_dict,
                trajectory=x_var,
                mask=None,
            )

            E = out.energy #.mean()  # .mean()  # .mean()  # or sum()
            # print("x_var req:", x_var.requires_grad)
            # print("E req:", E.requires_grad, "grad_fn:", E.grad_fn)
            (grad_E,) = torch.autograd.grad(
                E,
                x_var,
                retain_graph=False,
                create_graph=False,
                grad_outputs=torch.ones_like(E),
            )
        grad_E[:, :, :2] = 0.0
        grad_E[:, :, 4:6] = 0.0
        denoised = model(input, t, cond_zero)
        # map energy grad into denoiser output space
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            # choose sigma_t (diffusers schedulers vary; DDIM uses alpha_cumprod commonly)
            # robust fallback: compute sigma from alpha_cumprod
            t_idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0].item()
            alpha = scheduler.alphas_cumprod[scheduler.timesteps[t_idx]].to(
                input.device
            )
            sigma_t = torch.sqrt(1.0 - alpha)

            x_prev = denoised - (eta * sigma_t) * grad_E.detach()

        elif pred_type == "v_prediction":
            # rough-but-works: treat like epsilon guidance after converting v->eps
            t_idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0].item()
            alpha = scheduler.alphas_cumprod[scheduler.timesteps[t_idx]].to(
                input.device
            )
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus = torch.sqrt(1.0 - alpha)

            # v -> eps: eps = sqrt(1-a)*x_t + sqrt(a)*v  (common diffusers convention)
            eps = sqrt_one_minus * input + sqrt_alpha * denoised

            eps = eps + (eta * sqrt_one_minus) * grad_E

            # eps -> v: v = (eps - sqrt(1-a)*input) / sqrt(a)
            x_prev = (eps - sqrt_one_minus * input) / (sqrt_alpha + 1e-8)

        else:
            raise TypeError(f"Unknown prediction type: {pred_type}")

        return x_prev


class EnergyGuidanceGrad(BaseGuidance):
    def __init__(self, config):
        super().__init__(config)
        self.eta = config["guidance_eta"]

    def __call__(
        self,
        input: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        scheduler: Optional[DDIMScheduler],
        model: Optional[torch.nn.Module],
        **kwargs,
    ) -> torch.Tensor:
        device = input.device
        eta = self.eta
        cond_zero = torch.zeros_like(cond)
        assert "analyzer" in kwargs
        analyzer: ConstraintAnalyzer = kwargs["analyzer"]
        B, N, D = input.shape
        device = input.device
        # TODO: better with dict
        x_var = input  # .detach().clone().requires_grad_(True)
        # IMPORTANT: use slices of x_var so E depends on x_var
        states_dict = {
            "desired_pos": x_var[:, :, 2:4],
            "current_pos": x_var[:, :, 4:],
        }
        actions_dict = {
            "vel": x_var[:, :, :2],
        }
        env_dict = {"goal": torch.zeros((2,), device=device, dtype=x_var.dtype)}

        out: ConstraintOutputs = analyzer(
            states=states_dict,
            actions=actions_dict,
            environment=env_dict,
            trajectory=x_var,
            mask=None,
        )
        grad_E = torch.zeros_like(input, device=device, dtype=input.dtype)
        grad_E_des_pos = out.energy
        grad_E[:, :, 2:4] = grad_E_des_pos[:, :, 2:]
        denoised = model(input, t, cond_zero)
        # map energy grad into denoiser output space
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            # choose sigma_t (diffusers schedulers vary; DDIM uses alpha_cumprod commonly)
            # robust fallback: compute sigma from alpha_cumprod
            t_idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0].item()
            alpha = scheduler.alphas_cumprod[scheduler.timesteps[t_idx]].to(
                input.device
            )
            sigma_t = torch.sqrt(1.0 - alpha)

            x_prev = denoised - (eta * sigma_t) * grad_E

        elif pred_type == "v_prediction":
            # rough-but-works: treat like epsilon guidance after converting v->eps
            t_idx = (scheduler.timesteps == t).nonzero(as_tuple=True)[0].item()
            alpha = scheduler.alphas_cumprod[scheduler.timesteps[t_idx]].to(
                input.device
            )
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_one_minus = torch.sqrt(1.0 - alpha)

            # v -> eps: eps = sqrt(1-a)*x_t + sqrt(a)*v  (common diffusers convention)
            eps = sqrt_one_minus * input + sqrt_alpha * denoised

            eps = eps + (eta * sqrt_one_minus) * grad_E

            # eps -> v: v = (eps - sqrt(1-a)*input) / sqrt(a)
            x_prev = (eps - sqrt_one_minus * input) / (sqrt_alpha + 1e-8)

        else:
            raise TypeError(f"Unknown prediction type: {pred_type}")

        return x_prev
