import torch
import albumentations as A

from anomalib.models.components import AnomalyModule

from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
from ddad import DDADUnetSize
from torch_model import DDADModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


class DDAD(AnomalyModule):
    def __init__(
        self,
        img_size: int = 256,
        unet_size: DDADUnetSize = DDADUnetSize.M,
        in_channels: int = 1,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
    ) -> None:
        super(DDAD, self).__init__()
        self.model = DDADModel(
            img_size=img_size, unet_size=unet_size, in_channels=in_channels
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def on_train_start(self) -> None:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.model.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {"optimizer": optimizer}

    def training_step(self, batch, *args, **kwargs):
        del args, kwargs  # These variables are not used
        loss = self.model(batch=batch)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    # TOOD: move this to some other function than __call__
    # This trains the domain adapter first (and reconstruction?)
    # the rest is inference
    def __call__(self) -> Any:
        domain_adaptater = DomainAdapter(backbone)

        feature_extractor = domain_adaptater.train(self.unet, fine_tune=False)
        feature_extractor.eval()

        # TODO: anything below hasnt been modified yet

        labels_list = []
        predictions = []
        anomaly_map_list = []
        gt_list = []
        reconstructed_list = []
        forward_list = []

        with torch.no_grad():
            for input, gt, labels in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)

                anomaly_map = self.transform(anomaly_map)
                gt = self.transform(gt)

                forward_list.append(input)
                anomaly_map_list.append(anomaly_map)

                gt_list.append(gt)
                reconstructed_list.append(x0)
                for pred, label in zip(anomaly_map, labels):
                    labels_list.append(0 if label == "good" else 1)
                    predictions.append(torch.max(pred).item())

        metric = Metric(
            labels_list, predictions, anomaly_map_list, gt_list, self.config
        )
        metric.optimal_threshold()
        if self.config.metrics.auroc:
            print(
                "AUROC: ({:.1f},{:.1f})".format(
                    metric.image_auroc() * 100, metric.pixel_auroc() * 100
                )
            )
        if self.config.metrics.pro:
            print("PRO: {:.1f}".format(metric.pixel_pro() * 100))
        if self.config.metrics.misclassifications:
            metric.miscalssified()
        reconstructed_list = torch.cat(reconstructed_list, dim=0)
        forward_list = torch.cat(forward_list, dim=0)
        anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        pred_mask = (anomaly_map_list > metric.threshold).float()
        gt_list = torch.cat(gt_list, dim=0)
        if not os.path.exists("results"):
            os.mkdir("results")
        if self.config.metrics.visualisation:
            visualize(
                forward_list,
                reconstructed_list,
                gt_list,
                pred_mask,
                anomaly_map_list,
                self.config.data.category,
            )
