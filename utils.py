import matplotlib.pyplot as plt
import pickle

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

with open('data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
    RIG_SCALER = pickle.load(f)


def plot_losses(train_losses, val_losses, save_name="losses"):
    print(train_losses)
    print(val_losses)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"{save_name}.png")
    plt.close()


def create_gaussian_diffusion(args):
    # default params
    sigma_small = True
    predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diff_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

import math
import torch
import torch.nn as nn

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def create_alignment_mask(seq_length, k=3):
    memory_mask = torch.zeros(seq_length, seq_length)
    
    for i in range(seq_length):
        start_j = max(0, i - k)
        end_j = min(seq_length, i + k + 1)
        memory_mask[i, start_j:end_j] = 1
    
    memory_mask = memory_mask.bool()
    return memory_mask
    