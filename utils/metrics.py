import numpy as np
import torch
import torch.nn as nn

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

class RTALoss(nn.Module):

    def __init__(self, lambda_t=1.0, lambda_a=0.1, beta=1.0):
        super().__init__()
        self.lambda_t = lambda_t
        self.lambda_a = lambda_a
        self.huber = nn.SmoothL1Loss(beta=beta, reduction="mean")

    @staticmethod
    def _slope(x: torch.Tensor) -> torch.Tensor:

        batch, horizon, n_vars = x.shape
        t = torch.arange(horizon, device=x.device, dtype=x.dtype)
        t_centered = t - t.mean()
        denom = (t_centered ** 2).sum()

        x_centered = x - x.mean(dim=1, keepdim=True)
        numer = (x_centered * t_centered.view(1, horizon, 1)).sum(dim=1)
        return numer / denom

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        L_p = self.huber(pred, target)

        diff_pred = pred[:, 1:, :] - pred[:, :-1, :]
        diff_target = target[:, 1:, :] - target[:, :-1, :]
        L_t = self.huber(diff_pred, diff_target)

        mean_err = (pred.mean(dim=1) - target.mean(dim=1)).abs().mean()
        std_err = (
            pred.std(dim=1, unbiased=True)
            - target.std(dim=1, unbiased=True)
        ).abs().mean()
        slope_err = (self._slope(pred) - self._slope(target)).abs().mean()
        L_a = (mean_err + std_err + slope_err) / 3.0

        return L_p + self.lambda_t * L_t + self.lambda_a * L_a