from collections import namedtuple

import numpy as np

import torch
import torch.utils.data as data_utils


def create_loader(data, batch_size=500, validate=False):
    """Create a dataloader."""

    if isinstance(data, np.ndarray):
        dataset = data_utils.TensorDataset(
            *[
                torch.Tensor(data[:, :2048]),
                torch.Tensor(data[:, 2048:2058]),
                torch.Tensor(data[:, 2058:]),
            ]
        )
        if validate:
            batch_size = data.shape[0]
    else:
        dataset = data_utils.TensorDataset(
            *[
                torch.tensor(getattr(data, i)).float()
                for i in ["lsr", "feats", "scores"]
            ]
        )
        if validate:
            batch_size = data.lsr.shape[0]

    loader = data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def load_features(split=True, nt=False):
    """Load and combine training data from both train and dev datasets."""
    trainlsr = np.load("saved_features/train_lsr.npy", allow_pickle=True)
    trainnlp = np.load("saved_features/train_nlp.npy", allow_pickle=True)
    trainsc = np.load("saved_features/train_scores.npy", allow_pickle=True)

    devlsr = np.load("saved_features/dev_lsr.npy", allow_pickle=True)
    devnlp = np.load("saved_features/dev_nlp.npy", allow_pickle=True)
    devsc = np.load("saved_features/dev_scores.npy", allow_pickle=True)
    trainlsr = trainlsr.reshape(-1, 2048)
    devlsr = devlsr.reshape(-1, 2048)

    train = namedtuple("res", ["lsr", "feats", "scores"])(
        lsr=trainlsr, feats=trainnlp, scores=trainsc
    )
    dev = namedtuple("res", ["lsr", "feats", "scores"])(
        lsr=devlsr, feats=devnlp, scores=devsc
    )

    if split:
        return train, dev

    all_train_lsr = np.append(trainlsr, devlsr, axis=0)
    all_train_nlp = np.append(trainnlp, devnlp, axis=0)
    all_train_sc = np.append(trainsc, devsc, axis=0)

    if nt:
        res = namedtuple("res", ["lsr", "feats", "scores"])(
            lsr=all_train_lsr, feats=all_train_nlp, scores=all_train_sc
        )
    else:
        res = np.concatenate(
            (all_train_lsr, all_train_nlp, all_train_sc.reshape(-1, 1)), axis=1
        )
    return res, dev
