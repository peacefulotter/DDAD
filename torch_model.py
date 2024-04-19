import torch
import numpy as np
import torch.nn as nn

from ddad import DDADUnetSize
from unet import UNetModel
from reconstruction import Reconstruction


class DDADModel(nn.Module):
    def __init__(
        self,
        img_size=256,
        unet_size: DDADUnetSize = DDADUnetSize.M,
        in_channels: int = 1,
    ) -> None:
        super(DDADModel, self).__init__()
        self.unet = UNetModel(
            img_size=img_size, size=unet_size, in_channels=in_channels
        )
        # unet = unet.to(config.model.device) # TODO(anomalib): handled by anomalib?
        # unet.train() # TODO(anomalib): handled by anomalib?
        # unet = nn.DataParallel() # TODO(anomalib): handled by anomalib?

        self.reconstruction = Reconstruction(
            self.unet, self.config
        )  # TODO(config): replace
        self.transform = A.Compose([A.CenterCrop(224, 224)])  # TODO: remove this?

    # TODO(config): replace
    def _get_loss(self, x_0, t, config):
        x_0 = x_0.to(config.model.device)
        betas = np.linspace(
            config.model.beta_start,
            config.model.beta_end,
            config.model.trajectory_steps,
            dtype=np.float64,
        )
        b = torch.tensor(betas).type(torch.float).to(config.model.device)
        e = torch.randn_like(x_0, device=x_0.device)
        at = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)

        x = at.sqrt() * x_0 + (1 - at).sqrt() * e
        output = self.unet(x, t.float())
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    # TODO(config): replace
    def forward(self, batch: dict[torch.Tensor]) -> torch.Tensor | dict:
        t = torch.randint(
            0,
            config.model.trajectory_steps,
            (batch["image"][0].shape[0],),
            device=config.model.device,
        ).long()
        loss = self._get_loss(
            self.unet, batch["image"][0], t, config
        )  # TODO: why are they only taking index 0
        return loss
