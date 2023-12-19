# neural_networks.py is © 2022, University of Cambridge
#
# neural_networks.py is published and distributed under the GAP Available Source License v1.0 (ASL).
#
# neural_networks.py is distributed in the hope that it will be useful for non-commercial academic
# research, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the ASL for more details.
#
# You should have received a copy of the ASL along with this program; if not, write to
# am2234 (at) cam (dot) ac (dot) uk.
from collections import defaultdict
import pathlib
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy.io as spio


WINDOW_STEP = 0.020  # seconds
WINDOW_LENGTH = 0.050  # seconds
FREQUENCY_SAMPLING = 4000  # Hz, fixed for all recordings
FREQUENCY_HIGH = 800  # Hz
TRAINING_SEQUENCE_LENGTH = 6  # seconds
TRAINING_BATCH_SIZE = 32
TRAINING_VAL_BATCH_SIZE = 128
TRAINING_LR = 1e-4
TRAINING_PATIENCE = 10
TRAINING_MAX_EPOCHS = 1000


def train_and_validate_model(
    model_folder: pathlib.Path,
    fold: int,
    data_folder: pathlib.Path,
    train_files: List[str],
    train_labels: List[str],
    train_timings,
    val_files: List[str],
    val_labels: List[str],
    val_timings,
    shared_feature_cache: Dict[str, torch.Tensor],
    verbose: int,
    gpu: bool,
    hparams,
    quick: bool = False,
):
    train_dataset = RecordingDataset(
        data_folder,
        train_files,
        train_labels,
        train_timings,
        sequence_length=int(TRAINING_SEQUENCE_LENGTH / WINDOW_STEP),
        cache=shared_feature_cache,
    )

    valid_dataset = RecordingDataset(
        data_folder,
        val_files,
        val_labels,
        val_timings,
        sequence_length=None,
        cache=shared_feature_cache,
    )

    val_features, val_labels, val_lengths = batch_whole_dataset(valid_dataset)

    model, val_loss = train_model(
        model_folder,
        fold,
        train_dataset,
        valid_dataset,
        use_gpu=gpu,
        verbose=verbose,
        quick=quick,
        hparams=hparams,
    )

    fold_posteriors = predict_single_model(model, val_features, val_lengths, gpu=gpu)
    return model, val_loss, {n: p for n, p in zip(val_files, fold_posteriors)}


def predict_files(model, data_folder, files, labels, timings, cache, gpu):
    dataset = RecordingDataset(
        data_folder, files, labels, timings, sequence_length=None, cache=cache
    )
    features, labels, lengths = batch_whole_dataset(dataset)
    posteriors = predict_single_model(model, features, lengths, gpu=gpu)
    return {n: p for n, p in zip(files, posteriors)}


def train_model(
    model_folder, fold, train_dataset, valid_dataset, use_gpu, verbose, quick, hparams
):
    class_weights = calculate_class_weights(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True, collate_fn=collate_fn
    )

    val_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=TRAINING_VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Train the model.
    model = instantiate_model(hparams)
    if use_gpu:
        # Move models and data to GPU. Move model before optimiser created.
        model = model.cuda()
        if class_weights is not None:
            class_weights = class_weights.cuda()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training model with {num_parameters} params")

    optimiser = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

    best_val_loss = np.inf
    early_stop_counter = 0

    for ep in range(TRAINING_MAX_EPOCHS if not quick else 2):
        running_loss = 0.0

        model.train()
        for i, (train_features, train_labels, train_lengths) in enumerate(train_dataloader):
            if use_gpu:
                train_features = train_features.cuda()
                train_labels = train_labels.cuda()

            # Train features is [B, F, T] whereas labels is [B]
            optimiser.zero_grad()

            out = model(train_features, train_lengths)
            loss = criterion(out, train_labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

        with torch.no_grad():
            model.eval()
            val_losses = []
            for val_features, val_labels, val_lengths in val_dataloader:
                if use_gpu:
                    val_features = val_features.cuda()
                    val_labels = val_labels.cuda()
                out = model(val_features, val_lengths)
                loss = criterion(out, val_labels)
                val_losses.append(loss * val_features.shape[0])
            loss = sum(val_losses) / len(valid_dataset)

        if verbose >= 2:
            print(
                f"{ep:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss.item():.3f} | {early_stop_counter:02d}"
            )
        elif verbose >= 1:
            print(
                f"\r{ep:04d}/{TRAINING_MAX_EPOCHS} | {running_loss / i:.3f} | {loss.item():.3f} | {early_stop_counter:02d} ",
                end="",
            )

        if loss < best_val_loss:
            # Save best model and reset patience
            save_challenge_model(model_folder, model, fold)
            best_val_loss = loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > TRAINING_PATIENCE:
                break

    print("\nFinished training neural network.")

    # Now run best model on validation set
    #    Re-load best model
    model = load_single_network_fold(model_folder, fold, hparams)
    if use_gpu:
        model = model.cuda()
    return model, best_val_loss


def predict_single_model(model, features, lengths, gpu):
    if gpu:
        features = features.cuda()

    results = []
    with torch.no_grad():
        # Compute HSMM observations using neural network
        out = model(features, lengths)
        out = out.cpu()

        for posteriors, lengths in zip(out, lengths):
            posteriors = posteriors[:, :lengths]
            posteriors = torch.nn.functional.softmax(posteriors, dim=0)
            results.append(posteriors)

    return results


def calc_frequency_bins():
    return int(np.ceil(FREQUENCY_HIGH * WINDOW_LENGTH))


def calculate_features(recording: torch.Tensor, fs: int, norm=True) -> Tuple[torch.Tensor, int]:
    assert fs == FREQUENCY_SAMPLING

    # Zero-mean and normalise by peak amplitude
    recording = recording.float()
    recording -= recording.mean()
    recording /= recording.abs().max()

    # Calculate spectrogram
    window_length = int(WINDOW_LENGTH * fs)
    window_step = int(WINDOW_STEP * fs)
    spectrogram = (
        torch.stft(
            recording,
            n_fft=window_length,
            hop_length=window_step,
            window=torch.hann_window(window_length),
            center=False,
            return_complex=False,
        )
        .pow(2)
        .sum(dim=-1)
    )

    # Remove high frequencies above FREQUENCY_HIGH Hz
    spectrogram = spectrogram[: calc_frequency_bins()]

    # Log and z-normalise
    spectrogram = torch.log(spectrogram)

    if norm:
        spectrogram = (spectrogram - spectrogram.mean(dim=-1, keepdims=True)) / spectrogram.std(
            dim=-1, keepdims=True
        )

    features_fs = int(1 / WINDOW_STEP)
    return spectrogram, features_fs


# Save your trained model.
def save_challenge_model(model_folder, model, fold):
    model_path = pathlib.Path(model_folder) / f"model_{fold}.pt"
    torch.save(model.state_dict(), model_path)


def instantiate_model(hparams):
    rnn_params = {k: v for k, v in hparams.items() if k.startswith(("rnn_", "ann_"))}
    return RecurrentNetworkModel(**rnn_params)


def load_single_network_fold(model_folder, fold, hparams):
    model = instantiate_model(hparams)
    model.load_state_dict(torch.load(pathlib.Path(model_folder) / f"model_{fold}.pt"))
    model.eval()
    return model


class RecurrentNetworkModel(nn.Module):
    def __init__(
        self, rnn_hidden_size, rnn_num_layers, rnn_dropout, ann_hidden_size, ann_dropout
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=calc_frequency_bins(),
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=rnn_dropout,
        )
        self.linear = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, ann_hidden_size[0]),
            nn.Tanh(),
            nn.Dropout(ann_dropout),
            nn.Linear(ann_hidden_size[0], ann_hidden_size[1]),
            nn.Tanh(),
            nn.Dropout(ann_dropout),
            nn.Linear(ann_hidden_size[1], 5),
        )

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # [B, C, T] to [B, T, C] for pack_padded_sequence

        sorted_lengths, sorted_indices = lengths.sort(descending=True)  # CPU- >CUDA-> CPU
        sorted_input = x[sorted_indices, :, :]

        packed_input = nn.utils.rnn.pack_padded_sequence(
            sorted_input, sorted_lengths, batch_first=True
        )
        output, h_n = self.rnn(packed_input)

        output, _ = nn.utils.rnn.pad_packed_sequence(
            output, total_length=x.shape[1], batch_first=True
        )  # check not batch first?
        output = output[sorted_indices.argsort()]

        return self.linear(output).permute(0, 2, 1)  # back to [B, C, T]

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)  # RNN doesn't downsample


class NetworkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 100),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(50, 5),
        )

    def forward(self, x, lengths):
        # x is features, size [B x 20 x T]
        x = x.permute(0, 2, 1)
        return self.network(x).permute(0, 2, 1)

    @property
    def output_fs(self):
        return int(1 / WINDOW_STEP)  # ANN doesn't downsample


def collate_fn(x):
    max_length = 0
    all_features = []
    all_labels = []

    actual_lengths = []

    for features, labels in x:
        all_features.append(features)
        all_labels.append(labels)
        max_length = max(max_length, len(labels))
        actual_lengths.append(len(labels))

    padded_features = all_features[0].new_full((len(x), all_features[0].shape[0], max_length), 0.0)
    for i, features in enumerate(all_features):
        padded_features[i, ..., : features.shape[-1]] = features

    padded_labels = all_labels[0].new_full((len(x), max_length), -1)
    for i, labels in enumerate(all_labels):
        padded_labels[i, : labels.shape[-1]] = labels

    return padded_features, padded_labels, torch.as_tensor(actual_lengths)


def calculate_class_weights(dataset):
    old_sequence_length = dataset.sequence_length
    dataset.sequence_length = None

    class_counts = defaultdict(int)
    for i in range(len(dataset)):
        _, label = dataset[i]
        unique, counts = torch.unique(label, return_counts=True)
        for uniq, count in zip(unique, counts):
            class_counts[uniq.item()] += count

    # May want to further adjust these weights as detecting murmur is v. important
    total_sum = sum(class_counts.values())
    class_weights = torch.zeros(5)
    for k, v in class_counts.items():
        class_weights[k] = total_sum / v

    # Restore sequence length
    dataset.sequence_length = old_sequence_length

    return class_weights


class RecordingDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, recording_paths, labels, timings, sequence_length, cache=None):
        self.data_folder = pathlib.Path(data_folder)
        self.recording_paths = recording_paths
        self.labels = labels
        self.timings = timings
        self.sequence_length = sequence_length
        assert len(self.recording_paths) == len(self.labels)

        self.cache = cache

    def __len__(self):
        return len(self.recording_paths)

    def __getitem__(self, idx):
        filename = self.recording_paths[idx]
        label = self.labels[idx]
        timing = self.timings[idx]

        if self.cache is None or filename not in self.cache:
            filepath = self.data_folder / filename
            fs, recording = spio.wavfile.read(filepath.with_suffix(".wav"))

            recording = torch.as_tensor(recording.copy())
            features, fs_features = calculate_features(recording, fs)

            segmentation_path = filepath.with_suffix(".tsv")
            segmentation_df = pd.read_csv(segmentation_path, sep="\t", header=None)
            segmentation_df.columns = ["start", "end", "state"]
            segmentation_df["start"] = (segmentation_df["start"] * fs_features).astype(int)
            segmentation_df["end"] = (segmentation_df["end"] * fs_features).astype(int)

            # num classes = 3
            segmentation_label = torch.zeros(features.shape[-1], dtype=torch.long)
            for row in segmentation_df.itertuples():
                # IN:  noise =  0, S1 = 1, systole = 2, S2 = 3, diastole = 4
                # OUT: noise = -1, S1 = 0, systole = 1, S2 = 2, diastole = 3, murmur = 4
                if row.state == 0:
                    segmentation_state = -1
                if row.state == 1:
                    segmentation_state = 0
                elif row.state == 3:
                    segmentation_state = 2
                elif row.state == 2:
                    segmentation_state = 1
                elif row.state == 4:
                    # Diastole
                    segmentation_state = 3

                segmentation_label[row.start : row.end] = segmentation_state

                if (row.state == 2) and (label == "Present"):
                    if timing == "Early-systolic":
                        portion = [0, 0.5]
                    elif timing == "Holosystolic":
                        portion = [0, 1]
                    elif timing == "Mid-systolic":
                        portion = [0.25, 0.75]
                    elif timing == "Late-systolic":
                        portion = [0.5, 1]
                    else:
                        portion = [0, 0]
                        print(f"Warn: Got timing {timing} for file {filename}")
                        # because diastolic

                    state_duration = row.end - row.start
                    start = int(row.start + portion[0] * state_duration)
                    end = int(np.ceil(row.start + portion[1] * state_duration))
                    segmentation_label[start:end] = 4

            indices_to_keep = segmentation_label != -1
            segmentation_label = segmentation_label[indices_to_keep]
            features = features[..., indices_to_keep]

            self.cache[filename] = (features, segmentation_label)
        else:
            features, segmentation_label = self.cache[filename]

        if self.sequence_length is not None:
            random_start = torch.randint(
                low=0, high=max(features.shape[-1] - self.sequence_length, 1), size=(1,)
            ).item()
            features = features[..., random_start : random_start + self.sequence_length]
            segmentation_label = segmentation_label[
                random_start : random_start + self.sequence_length
            ]

        return features, segmentation_label


def batch_whole_dataset(dataset: RecordingDataset):
    num_examples = len(dataset)
    return collate_fn([dataset[i] for i in range(num_examples)])
