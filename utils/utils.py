import torch
import numpy as np
import yaml
from pymatgen.core.composition import Composition
import pandas as pd

def load_yaml_config(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

def normalize_formulas(df: pd.DataFrame, formula_column: str = 'formula') -> pd.DataFrame:
    """Normalize chemical formulas to IUPAC format, removing duplicates due to structural representations.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'formula' column.
        formula_column"""
    formula=[]
    for form in df[formula_column].values:
        formula.append(Composition(Composition(form).get_integer_formula_and_factor()[0]).iupac_formula)
    df[formula_column]=formula
    return df 

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def RobustL1Loss(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2Loss(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    # NOTE can we scale log_std by something sensible to improve the OOD behaviour?
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    reg_term = torch.mean(log_std**2)
    return torch.mean(loss)+ 10 * reg_term

def QuantileLoss(output, target, quantile=0.5):
    """
    Quantile loss function.
    """
    error = output - target
    return torch.mean(torch.max((quantile - 1) * error, quantile * error))

def StudentTLoss(output, log_std, target, nu=3):
    """
    Student's t-distribution negative log-likelihood loss.
    
    Parameters:
        output: predicted mean (mu), shape (N,)
        log_std: predicted log of scale (sigma), shape (N,)
        target: true target values, shape (N,)
        nu: degrees of freedom for t-distribution (scalar, >0)

    Returns:
        Scalar loss (mean over batch)
    """
    scale = torch.exp(log_std)  # sigma
    resid = target - output
    loss = log_std + 0.5 * (nu + 1) * torch.log1p((resid**2) / (nu * scale**2))
    return loss.mean()

def IntervalScoreLoss(y_low, y_high, target, quantile):
    """Implementation of Interval score loss with boundaries 
       q_high = 1 + 0.5quantile, q_low = 1 - 0.5quantile
       loss = (y_high-y_low)+ 2/quantile * (y_low-y_target)*1(y<y_low)+2/quantile*(y_target-y_high)*1(y>y_high)
       target [batch] = y_target
    """
    width = y_high - y_low
    below = (target < y_low).float()
    above = (target > y_high).float()

    penalty_below = (y_low - target) * below
    penalty_above = (target - y_high) * above

    penalty = (2.0 / quantile) * (penalty_below + penalty_above)
    loss = width + penalty
    return loss.mean()

