"""Original RGB/alpha/BCE and robust physical NLL+MSE, computed in FP32."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def robust_gaussian_nll(mu, var, target):
    mu, var, target = mu.float(), var.float().clamp(min=1e-2), target.float()
    return (0.5*math.log(2*math.pi)+0.5*var.log()+0.5*(target-mu).square()/var).clamp(max=100.).mean()


class SupervisedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weights = {k: float(config.get(k, default)) for k, default in
                        (("l2_loss", 1.), ("opacity_loss", .1), ("physical_loss", .1),
                         ("E_loss", .01), ("nu_loss", .01), ("lpips_loss", 0.))}
        if min(self.weights.values()) < 0 or sum(self.weights.values()) == 0:
            raise ValueError("Loss weights must be nonnegative with at least one enabled")
        self.lpips = None
        if self.weights["lpips_loss"]:
            from .perceptual import OfflineLPIPS
            self.lpips = OfflineLPIPS(config.get("lpips_weights", "weights/lpips"))

    @property
    def image_enabled(self):
        return any(self.weights[k] for k in ("l2_loss", "opacity_loss", "lpips_loss"))

    def forward(self, prediction, rendered, batch):
        losses = {}
        w = self.weights
        # autocast-disabled math preserves both mean and variance gradients.
        with torch.autocast(device_type=prediction["phys_logits"].device.type, enabled=False):
            total = prediction["phys_logits"].new_zeros((), dtype=torch.float32)
            if self.image_enabled:
                rgba = batch["test_images"].float()
                if rgba.shape[2] != 4:
                    raise ValueError("Supervised target images require RGBA")
                target = rgba[:, :, :3]*rgba[:, :, 3:] + (1-rgba[:, :, 3:])
                rgb, alpha = rendered.rgb.float(), rendered.alpha.float()
                if w["l2_loss"]:
                    losses["l2_loss"] = F.mse_loss(rgb, target)
                    total = total+w["l2_loss"]*losses["l2_loss"]
                if w["opacity_loss"]:
                    losses["opacity_loss"] = F.mse_loss(alpha, rgba[:, :, 3:])
                    total = total+w["opacity_loss"]*losses["opacity_loss"]
                if w["lpips_loss"]:
                    losses["loss_lpips"] = self.lpips((rgb.flatten(0, 1)*2-1).clamp(-1, 1),
                                                    (target.flatten(0, 1)*2-1).clamp(-1, 1)).mean()
                    total = total+w["lpips_loss"]*losses["loss_lpips"]
                with torch.no_grad():
                    losses["psnr"] = -10*F.mse_loss(rgb.clamp(0, 1), target).clamp(min=1e-12).log10()
            if w["physical_loss"]:
                classes = F.one_hot(batch["physical_class"].long(), 14).float()
                losses["phys_loss"] = F.binary_cross_entropy_with_logits(prediction["phys_logits"].float(), classes)
                total = total+w["physical_loss"]*losses["phys_loss"]
                losses["material_accuracy"] = (prediction["phys_logits"].argmax(-1) == batch["physical_class"]).float().mean().detach()
            for field in ("E", "nu"):
                if w[field+"_loss"]:
                    mu, var, target = prediction[field+"_mu"], prediction[field+"_var"], batch[field]
                    losses[field+"_nll"] = robust_gaussian_nll(mu, var, target)
                    losses[field+"_mse"] = F.mse_loss(mu.float(), target.float())
                    losses[field+"_loss"] = losses[field+"_nll"]+losses[field+"_mse"]
                    total = total+w[field+"_loss"]*losses[field+"_loss"]
            losses["total_loss"] = total
        return losses
