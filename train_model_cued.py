#!/usr/bin/env python

# train_model_cued.py is © 2022, University of Cambridge
#
# train_model_cued.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# train_model_cued.py is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.
from team_code import train_challenge_model_full
import datetime
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--prev_run", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.prev_run is None:
        current_time = datetime.datetime.now()
        filename = current_time.strftime("%Y-%m-%d__%H-%M-%S")
    else:
        filename = args.prev_run
    model_folder = "model_runs/" + filename

    murmur_score, decision_murmur_score, outcome_score, val_loss = train_challenge_model_full(
        args.data,
        model_folder,
        args.verbose,
        gpu=not args.cpu,
        load_old_file=args.prev_run is not None,
        quick=args.quick,
    )

    if not args.prev_run:
        with open("model/results.txt", "a") as myfile:
            myfile.write(
                f"\n{model_folder}\t\t{murmur_score:.3f}\t{outcome_score:.0f}\t{'q' if args.quick else ''}"
            )
