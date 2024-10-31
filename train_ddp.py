import os
from typing import List

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchdiffeq import odeint
from torchvision import datasets, transforms
from tqdm import tqdm

from unet import Unet


class ConditionalOTFlowMatching:
    def __init__(self, unet: Unet, input_shape: List[int], sigma_min: float):
        self.v = unet
        self.input_shape = input_shape
        self.sigma_min = sigma_min

    def cond_psi(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        b, *_ = x_0.shape
        return (
            (1 - (1 - self.sigma_min) * t.view((b, 1, 1, 1))) * x_0 +
            t.view((b, 1, 1, 1)) * x_1
        )

    def u(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        return x_1 - (1 - self.sigma_min) * x_0

    def training_step(self, x, c):
        b, *_ = x.shape
        t = torch.rand((b,), device=x.device)
        x_0 = torch.randn_like(x)
        x_1 = x
        psi = self.cond_psi(t, x_0, x_1)
        v = self.v(psi, t, c)
        u = self.u(x_0, x_1)
        return F.mse_loss(v, u)

    @torch.inference_mode()
    def sample(self,
               c: torch.Tensor,
               w: float = 1.0,
               num_ode_steps: int = 100,
               return_trajectory: bool = False
               ):
        self.v.eval()
        b = c.shape[0]
        x_0 = torch.rand([b] + self.input_shape, device=c.device)
        x_t = odeint(
            lambda t, x: (
                (1 - w) * self.v(x, t * torch.ones((b,), device=x.device)) +
                w * self.v(x, t * torch.ones((b,), device=x.device), c)
            ),
            x_0,
            torch.linspace(0, 1, num_ode_steps, device=c.device),
            rtol=1e-5,
            atol=1e-5,
            method="dopri5"
        )
        x_1 = x_t[-1, ...]
        if return_trajectory:
            return x_1, x_t
        else:
            return x_1


def main(rank, world_size):
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size
    )
    device = torch.device(f"cuda:{rank}")

    num_classes = 10
    v = Unet(
        channels=64,
        channel_mults=[1, 2, 4],
        in_channels=1,
        num_classes=num_classes,
        class_conditional=True,
        class_dropout_prob=0.2,
    ).to(device)

    v = DDP(v, device_ids=[rank])

    cfm = ConditionalOTFlowMatching(
        unet=v,
        input_shape=[1, 28, 28],
        sigma_min=0.002,
    )

    batch_size = 32
    num_epochs = 64
    num_workers = 4
    prefetch_factor = 2
    lr = 3e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    optimizer = torch.optim.AdamW(v.parameters(), lr=lr)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_loader = tqdm(
            train_loader,
            desc=f'epoch [{epoch + 1}/{num_epochs}]',
            leave=False
        )
        for x, c in progress_loader:
            x = x.to(device)
            c = c.to(device)
            cfm.v.train()
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = cfm.training_step(x, c)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if rank == 0:
                progress_loader.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        if rank == 0:
            print(
                f"epoch [{epoch + 1}/{num_epochs}], avg loss: {avg_loss:.4f}"
            )
        if rank == 0:
            print("saving model...")
            torch.save(cfm.v.state_dict(), 'v.pth')
            torch.save(optimizer.state_dict(), 'adamw.pth')


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "22355"
    world_size = 2  # Number of GPUs
    spawn(main, args=(world_size,), nprocs=world_size, join=True)
