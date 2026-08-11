"""Model ladder M0-M5 building blocks (design doc sections 9-11).

Feature sets:
  M1: price features only
  M2: M1 + raw physical fundamentals
  M3: M2 + engineered balance features (stock change, net flow, residual, days supply)
  M4: dual-branch (price / physical) + auxiliary next-week physical heads, no balance loss
  M5: M4 + soft petroleum-balance consistency loss
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PRICE_FEATURES = [
    "px_wti", "px_log_wti", "px_ret_1w", "px_ret_2w", "px_ret_4w",
    "px_vol_4w", "px_vol_8w", "px_vol_12w", "px_ma12_dist",
]
PHYS_RAW = [
    "production", "imports", "exports", "refinery_inputs",
    "stocks_commercial", "stocks_spr", "stocks_cushing", "utilization",
    "adjustment",
]
PHYS_DERIVED = [
    "d_stocks_commercial", "d_stocks_spr", "d_stocks_cushing",
    "net_flow_no_adj", "balance_residual", "days_supply",
]
AUX_TARGETS = [
    "aux_d_production_next", "aux_d_imports_next", "aux_d_exports_next",
    "aux_d_refinery_inputs_next", "aux_d_stocks_commercial_next",
    "aux_adjustment_next",
]
TARGETS = ["y_h1", "y_h2", "y_h4", "y_h8", "y_h12"]
VOL_TARGETS = ["yv_h1", "yv_h2", "yv_h4", "yv_h8", "yv_h12"]
RV_FEATURES = ["px_rv_1w", "px_rv_4w", "px_rv_12w"]

FEATURE_SETS = {
    "M1": PRICE_FEATURES,
    "M2": PRICE_FEATURES + PHYS_RAW,
    "M3": PRICE_FEATURES + PHYS_RAW + PHYS_DERIVED,
    # M4/M5 use branch-specific sets below
}
M45_PRICE = PRICE_FEATURES
M45_PHYS = PHYS_RAW + PHYS_DERIVED


@dataclass
class Scaler:
    """Standard scaler fit on training rows only."""
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)

    def fit(self, x: np.ndarray) -> "Scaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


class WindowDataset(Dataset):
    """Sequences of L consecutive weekly rows ending at row i (inclusive)."""

    def __init__(self, df: pd.DataFrame, idx: np.ndarray, lookback: int,
                 feat_arrays: dict, target_array: np.ndarray,
                 aux_array: np.ndarray = None, extra: dict = None):
        self.lookback = lookback
        self.idx = idx[idx >= lookback - 1]
        self.feat_arrays = feat_arrays          # name -> [N, F] scaled
        self.targets = target_array             # [N, H]
        self.aux = aux_array                    # [N, K] scaled or None
        self.extra = extra or {}                # name -> [N, ...] physical units

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        sl = slice(i - self.lookback + 1, i + 1)
        item = {name: torch.as_tensor(arr[sl], dtype=torch.float32)
                for name, arr in self.feat_arrays.items()}
        item["y"] = torch.as_tensor(self.targets[i], dtype=torch.float32)
        if self.aux is not None:
            item["aux"] = torch.as_tensor(self.aux[i], dtype=torch.float32)
        for name, arr in self.extra.items():
            item[name] = torch.as_tensor(arr[i], dtype=torch.float32)
        item["row"] = i
        return item


class GRUForecaster(nn.Module):
    """Single-branch GRU with one head per horizon (M1-M3)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.1, n_horizons: int = 3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=layers,
                          batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_horizons)

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.drop(out[:, -1])
        return self.head(h)


class DualBranchForecaster(nn.Module):
    """M4/M5: price branch + physical branch + auxiliary physical head."""

    def __init__(self, n_price: int, n_phys: int, hidden: int = 64,
                 phys_hidden: int = 32, layers: int = 1, dropout: float = 0.1,
                 n_horizons: int = 3, n_aux: int = 6):
        super().__init__()
        self.price_gru = nn.GRU(n_price, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
        self.phys_gru = nn.GRU(n_phys, phys_hidden, num_layers=layers,
                               batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.aux_head = nn.Linear(phys_hidden, n_aux)
        self.price_head = nn.Sequential(
            nn.Linear(hidden + phys_hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_horizons),
        )

    def forward(self, x_price, x_phys):
        hp, _ = self.price_gru(x_price)
        hf, _ = self.phys_gru(x_phys)
        hp, hf = self.drop(hp[:, -1]), self.drop(hf[:, -1])
        aux = self.aux_head(hf)
        y = self.price_head(torch.cat([hp, hf], dim=1))
        return y, aux


def balance_terms(aux_pred_std: torch.Tensor, aux_scaler: Scaler,
                  flow_t: torch.Tensor, d_spr_next: torch.Tensor):
    """Differentiably inverse-transform auxiliary CHANGE predictions to
    physical units, reconstruct next-week levels from week-t levels, and
    return (predicted stock change, predicted net accumulation) in kb/week.

    aux order (standardized): d_production, d_imports, d_exports,
                              d_refinery_inputs, d_stocks_commercial, adjustment
    flow_t columns: production, imports, exports, refinery_inputs (levels, kb/d)
    """
    mean = torch.as_tensor(aux_scaler.mean, dtype=aux_pred_std.dtype,
                           device=aux_pred_std.device)
    std = torch.as_tensor(aux_scaler.std, dtype=aux_pred_std.dtype,
                          device=aux_pred_std.device)
    phys = aux_pred_std * std + mean
    d_prod, d_imp, d_exp, d_ref, d_stock_c, adj = phys.unbind(dim=1)
    prod_t, imp_t, exp_t, ref_t = flow_t.unbind(dim=1)
    net_acc = 7.0 * ((prod_t + d_prod) + (imp_t + d_imp) + adj
                     - (ref_t + d_ref) - (exp_t + d_exp))
    d_stock = d_stock_c + d_spr_next
    return d_stock, net_acc


def balance_loss(aux_pred_std: torch.Tensor, aux_scaler: Scaler,
                 flow_t: torch.Tensor, d_spr_next: torch.Tensor,
                 scale: float) -> torch.Tensor:
    """Soft balance consistency loss. `scale` (kb/week, e.g. train std of the
    observed stock change) normalizes the residual so Huber delta=1 and the
    loss magnitude are comparable to the standardized price loss."""
    d_stock, net_acc = balance_terms(aux_pred_std, aux_scaler,
                                     flow_t, d_spr_next)
    return nn.functional.huber_loss(d_stock / scale, net_acc / scale, delta=1.0)
