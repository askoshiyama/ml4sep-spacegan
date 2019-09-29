import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
import spacegan_utils


# Generator architecture
class Generator(nn.Module):
    def __init__(self, cond_dim, noise_dim, output_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(cond_dim + noise_dim, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )

    def forward(self, z, cond_info):
        input_data = torch.cat([z, cond_info], dim=1).float()
        data = self.model(input_data).float()
        return data


# Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, output_dim, cond_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(output_dim + cond_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, data, cond_info):
        input_data = torch.cat([data, cond_info], dim=1).float()
        validity = self.model(input_data).float()
        return validity

# dataset
df = pd.read_csv("grid.csv")
coord_vars = ["longitude", "latitude"]
cond_vars = ["z"] + coord_vars
output_vars = ["y"]
neighbours = 8

# problem configuration
results_path = "./Results/"
prob_config = {"epochs": 1001,
               "batch_size": 100,
               "device": torch.device("cuda"),
               "cond_dim": len(cond_vars) + neighbours,  # conditional information size
               "output_dim": len(output_vars),  # size of output
               "noise_dim": len(cond_vars) + neighbours,  # size of noise
               "noise_type": spacegan_utils.gaussian,  # type of noise and dimension used
               "noise_params": None,  # other params for noise (loc, scale, etc.) pass as a dict
               "scale_x": StandardScaler(),  # a sklearn.preprocessing scaling method
               "scale_y": StandardScaler(),  # a sklearn.preprocessing scaling method
               "print_results": False
               }

prob_config["gen_opt"] = torch.optim.SGD
prob_config["gen_opt_params"] = {"lr": 0.01}
prob_config["disc_opt"] = torch.optim.SGD
prob_config["disc_opt_params"] = {"lr": 0.01}
prob_config["adversarial_loss"] = torch.nn.BCELoss()

# checkpointing configuration
check_config = {
    "check_interval": 100,  # for model checkpointing
    "generate_image": False,
    "n_samples": 100,
    "perf_metrics": {"RMSE": spacegan_utils.rmse,
                     "MIE": spacegan_utils.mie,
                     },
    "pf_metrics_setting": {
        "RMSE": {"metric_level": "agg_metrics",
             "rank_function": np.argmin,
             "agg_function": lambda x: np.array(x)
             },
        "MIE": {"metric_level": "agg_metrics",
                "rank_function": np.argmin,
                "agg_function": lambda x: np.array(x)
               },
    },
    "agg_funcs": {"avg": np.mean,
                  "std": np.std
                 },
    "sample_metrics": False,
    "agg_metrics": True
}

