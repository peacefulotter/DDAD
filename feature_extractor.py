import logging
import torch
import albumentations as A
import torchvision.transforms as T

from anomalib.models.components import AnomalyModule

from dataset import *
from dataset import *
from unet import *
from visualize import *
from resnet import *
from reconstruction import *
from light_model import DDAD

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

"""
train_dataset = Dataset_maker(
        root=config.data.data_dir,
        category=config.data.category,
        config=config,
        is_train=True,
    )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.DA_batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    """

# TODO(timm) use timm instead of manually creating the backbone feature extractor
feature_extractors = {
    "wide_resnet101_2": wide_resnet101_2,
    "wide_resnet50_2": wide_resnet50_2,
    "resnet50": resnet50,
}


class DomainAdapterModel(nn.Module):
    def __init__(self, backbone: str) -> None:
        super(DomainAdapter, self).__init__()
        fe_creator = wide_resnet101_2
        if backbone in feature_extractors:
            fe_creator = feature_extractors[backbone]
        else:
            logging.warning(
                "Feature extractor is not correctly selected, Default: wide_resnet101_2"
            )

        self.feature_extractor = fe_creator(pretrained=True)
        self.frozen_feature_extractor = fe_creator(pretrained=True)

        # .to(device), nn.DataParallel() # TODO(anomalib): handled by anomalib?

        self.frozen_feature_extractor.eval()

    # TODO(anomalib): handled by anomalib?
    def load_pretrained(self):
        checkpoint = torch.load(...)
        self.feature_extractor.load_state_dict(checkpoint)


class DomainAdapter(AnomalyModule):
    def __init__(self, ddad: DDAD, backbone: str = "wide_resnet101_2") -> None:
        super(DomainAdapter, self).__init__()
        self.ddad = ddad
        self.model = DomainAdapterModel(backbone)
        self.reconstruction = Reconstruction(ddad.model.unet)
        self.cos_loss = nn.CosineSimilarity()
        self.transform = A.Compose(
            [
                A.Lambda(lambda t: (t + 1) / 2),
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def configure_optimizer(self):
        optimizer = torch.optim.AdamW(
            self.model.feature_extractor.parameters(), lr=1e-4
        )
        return {"optimizer": optimizer}

    def _get_loss(self, reconst_fe, target_fe, reconst_frozen_fe, target_frozen_fe):

        loss1, loss2, loss3 = 0, 0, 0

        def _view(arr, i):
            return arr[i].view(arr[i].shape[0], -1)

        for i in range(len(reconst_fe)):
            reconst_view = _view(reconst_fe, i)
            target_view = _view(target_fe, i)
            reconst_frozen_view = _view(reconst_frozen_fe, i)
            targe_frozen_view = _view(target_frozen_fe, i)

            _loss1 = 1 - self.cos_loss(reconst_view, target_view)
            _loss2 = 1 - self.cos_loss(target_view, targe_frozen_view)
            _loss3 = 1 - self.cos_loss(reconst_view, reconst_frozen_view)

            loss1 += torch.mean(_loss1)
            loss2 += torch.mean(_loss2) * self.DLlambda  # TODO: what is this parameter
            loss3 += torch.mean(_loss3) * self.DLlambda  # TODO: what is this parameter

        loss = loss1 + loss2 + loss3
        return {"loss": loss}

    def forward(self, batch: dict[str, str | torch.Tensor]) -> torch.Tensor | dict:
        half_batch_size = batch["image"].shape[0] // 2
        target = batch["image"][:half_batch_size]
        input = batch["image"][half_batch_size:]

        x0 = self.reconstruction(input, target)
        # x0 = x0.to(device) # TODO(anomalib): hanlded by anomalib?
        x0 = self.transform(x0)
        target = self.transform(target)

        reconst_fe = self.model.feature_extractor(x0)
        target_fe = self.model.feature_extractor(target)

        with torch.no_grad():
            reconst_frozen_fe = self.model.frozen_feature_extractor(x0)
            target_frozen_fe = self.model.frozen_feature_extractor(target)

        loss = self._get_loss(
            reconst_fe, target_fe, reconst_frozen_fe, target_frozen_fe
        )
        return {"loss": loss}

    # TODO: not called train
    def train(self, fine_tune: bool):
        if not fine_tune:
            """
                checkpoint = torch.load(
                os.path.join(
                    os.path.join(os.getcwd(), config.model.checkpoint_dir),
                    config.data.category,
                    f"feat{config.model.DA_chp}",
                )
            )   
            """
            self.model.load_pretrained()
            return

        self.ddad.eval()
        self.model.feature_extractor.train()

        # TODO: anything below hasnt been modified

        for epoch in range(config.model.DA_epochs):
            for step, batch in enumerate(trainloader):
                half_batch_size = batch[0].shape[0] // 2
                target = batch[0][:half_batch_size].to(config.model.device)
                input = batch[0][half_batch_size:].to(config.model.device)

                x0 = reconstruction(input, target, config.model.w_DA)[-1].to(
                    config.model.device
                )
                x0 = transform(x0)
                target = transform(target)

                reconst_fe = feature_extractor(x0)
                target_fe = feature_extractor(target)

                target_frozen_fe = frozen_feature_extractor(target)
                reconst_frozen_fe = frozen_feature_extractor(x0)

                loss = loss_fucntion(
                    reconst_fe, target_fe, target_frozen_fe, reconst_frozen_fe, config
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1} | Loss: {loss.item()}")
            # if (epoch+1) % 5 == 0:
            torch.save(
                feature_extractor.state_dict(),
                os.path.join(
                    os.path.join(os.getcwd(), config.model.checkpoint_dir),
                    config.data.category,
                    f"feat{epoch+1}",
                ),
            )
