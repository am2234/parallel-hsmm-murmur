#!/usr/bin/env python

# train_model_hparams.py is © 2022, University of Cambridge
#
# train_model_hparams.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# train_model_hparams.py is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.
from team_code import train_challenge_model_full
import datetime
import numpy as np


class RangeHyperparam:
    def __init__(self, min, max, step=None, decimals=None):
        self.min = min
        self.max = max
        self.step = step
        self.decimals = decimals

    def sample(self, random_state):
        if self.step is not None:
            val = (
                random_state.randint(0, int(round((self.max - self.min) / self.step) + 1))
                * self.step
                + self.min
            )
        else:
            val = self.min + (self.max - self.min) * random_state.uniform()

        if self.decimals is not None:
            val = round(val, self.decimals)
        return val


def sample_hps(hyperparams, state):
    return {k: v.sample(state) for k, v in hyperparams.items()}


hps = {
    "rnn_hidden_size": RangeHyperparam(30, 100, 10),
    "rnn_num_layers": RangeHyperparam(1, 3, 1),
    "rnn_dropout": RangeHyperparam(0, 0.5, 0.1, 1),
    "ann_hidden_size": RangeHyperparam(20, 120, 10),
    "ann_dropout": RangeHyperparam(0, 0.5, 0.1, 1),
    "batch_size": RangeHyperparam(32, 96, 32),
    "lr": RangeHyperparam(0.5e-4, 1.5e-4, 0.5e-4),
}


if __name__ == "__main__":
    seed = 30  # tbd not reproducible anymore.
    state = np.random.RandomState(seed)

    for i in range(100):
        sampled_hparams = sample_hps(hps, state)
        print(sampled_hparams)

        current_time = datetime.datetime.now()
        filename = current_time.strftime("%Y-%m-%d__%H-%M-%S")

        folder_name = f"model_hparam_runs/{i}_{filename}"

        murmur_score, decision_murmur_score, outcome_score, val_loss = train_challenge_model_full(
            data_folder,
            folder_name,
            verbose=1,
            hparams=sampled_hparams,
        )

        with open("model_hparam_runs/results.txt", "a") as myfile:
            myfile.write(
                f"\n{folder_name}\t\t{murmur_score:.3f}\t{decision_murmur_score:.3f}\t{outcome_score:.0f}\t{val_loss:.3f}"
            )
