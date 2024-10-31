from typing import List

import torch
import torch.nn.functional as F
from torchdiffeq import odeint

from unet import Unet


class ConditionalOTFlowMatching:
  def __init__(
      self,
      unet: Unet,
      input_shape: List[int],
      sigma_min: float,
  ):
    self.v = unet
    self.input_shape = input_shape
    self.sigma_min = sigma_min

  def cond_psi(
      self,
      t: torch.Tensor,
      x_0: torch.Tensor,
      x_1: torch.Tensor
  ) -> torch.Tensor:
    b, *_ = x_0.shape
    return (
        (1 - (1 - self.sigma_min) * t.view((b, 1, 1, 1))) * x_0 +
        t.view((b, 1, 1, 1)) * x_1
    )

  def u(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    return x_1 - (1 - self.sigma_min) * x_0

  def training_step(self, x, c):
    # x: (b, c, h, w) [images]
    # c: (b, ) [labels]
    b, *_ = x.shape
    # Sample time
    t = torch.rand((b, ), device=x.device)
    # Set conditional flow endpoints
    x_0 = torch.randn_like(x)
    x_1 = x
    # Push x_0 through conditional flow
    psi = self.cond_psi(t, x_0, x_1)
    # Regress the ground truth conditional flow field
    v = self.v(psi, t, c)
    u = self.u(x_0, x_1)
    return F.mse_loss(v, u)

  @torch.inference_mode()
  def sample(
      self,
      c: torch.Tensor,
      w: float = 1.0,
      num_ode_steps: int = 100,
      return_trajectory: bool = False
  ):
    # c: (b, ) [labels]
    b = c.shape[0]
    # Sample some noise
    x_0 = torch.randn([b] + self.input_shape, device=c.device)
    # Solve the flow ODE system
    x_t = odeint(
        lambda t, x: (
            (1 - w) * self.v(x, t * torch.ones((b, ), device=x.device)) +
            w * self.v(x, t * torch.ones((b, ), device=x.device), c)
        ),
        x_0,
        torch.linspace(0, 1,  num_ode_steps, device=c.device),
        rtol=1e-5,
        atol=1e-5,
        method="dopri5"  # taken from the paper
    )
    if return_trajectory:
      return x_t
    else:
      # Grab the final state
      x_1 = x_t[-1, ...]
      return x_1
