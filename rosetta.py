# Standard
import argparse
from collections import namedtuple
import datetime

# ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DL
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import datetime

# In house
from models import RecursiveNN, RecursiveNN_Linear, ModelBlock, weights_init
from feature_extraction import FeatureExtractor
from utils import load_features, create_loader

logdir = "./logs/"


def train_model(
    model, train_loader, optimizer, epoch, log_interval=100, scheduler=None, writer=None
):
    """Manage the training process of the model for one epoch."""
    tloss = 0

    """SMOOTH L1 LOSS: Creates a criterion that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.
    It is less sensitive to outliers than the `MSELoss` and in some cases
    prevents exploding gradients """

    for batch_idx, (lsr, feats, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(lsr, feats)

        loss = F.smooth_l1_loss(outputs, targets.view(-1))
        # loss = F.mse_loss(outputs, targets.view(-1))
        tloss += loss.item()

        loss.backward()

        optimizer.step()

    # print(
    #     "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
    #         epoch, batch_idx, tloss
    #     )
    # )

    # Write loss to tensorboard

    if writer != None:
        writer.add_scalar("Train/Loss", tloss, epoch)
    if scheduler is not None:
        scheduler.step()


def test_model(model, test_loader, epoch, writer=None, score=False):
    """Output test loss and/or score for one epoch."""

    test_loss = 0.0

    with torch.no_grad():
        for lsr, feats, targets in test_loader:
            outputs = model(lsr, feats)
            test_loss += F.mse_loss(outputs, targets.view(-1)).item()

    # print("\nTest set: Average loss: {:.4f}\n".format(test_loss))

    # Write loss to tensorboard
    if writer != None:
        writer.add_scalar("Test/Loss", test_loss, epoch)
    if score:  # For evolutionary algorithms
        df = pd.DataFrame({"real": targets.view(-1), "preds": outputs}).fillna(0)
        df = df.corr().fillna(0)
        score = abs(df["preds"]["real"])
        return test_loss, score
    else:
        return test_loss


class Rosetta:
    """Rosetta stone regressor.
    Main class orchestrating whole regression pipeline."""

    def __init__(self, mode="extract", bSave="T", bUseConv=False):
        self.mode = mode

        if bSave is "T":
            self.bSave = False
        else:
            self.bSave = False

        self.bUseConv = bUseConv

        # Saved model
        self.model = None
        self.full_data = True

        # Define all hyperparameters
        if self.bUseConv:
            self.params = {
                "step_size": 5,
                "gamma": 0.8,
                "batch_size_train": 64,
                "batch_size_test": 128,
                "lr": 4e-04,
                "epochs": 40,
                "NBaseline": 10,
                "upsampling_factor": 3000,
                "upsample": False,
                "conv_dict": {
                    "InChannels": [2],
                    "OutChannels": [2],
                    "Ksze": [1],
                    "Stride": [1],
                    "Padding": [0],
                    "MaxPoolDim": 1,
                    "MaxPoolBool": False,
                },
                "conv_ffnn_dict": {
                    "laser_hidden_layers": [64, 16],
                    "mixture_hidden_layers": [32, 32, 1],
                },
            }
        else:
            self.params = {
                "N1": 40,
                "N2": 20,
                "batch_size_test": 100,
                "batch_size_train": 500,
                "dropout": 0,
                "epochs": 30,
                "leaky_relu": True,
                "lr": 0.0003,
                "out_features": 27,
                "score": 0.09012854763115952,
                "upsample": False,
                "upsampling_factor": 5000,
                "step_size": 40,
                "gamma": 0.5,
            }

    def upsample(self, data):
        """Upsample data to make score distribution more uniform."""

        nlp = data.feats
        scores = data.scores
        lsr = data.lsr

        if self.params["upsample"]:
            # Define parameters
            alpha = 0.45
            beta = 15
            gamma = 0.05

            if self.bUseConv:
                lsr = lsr.reshape(-1, 2048)

            # Retrieve score distribution in 15 bins
            n, bins, _ = plt.hist(
                scores, 15, density=True, range=(-1, 1), facecolor="g", alpha=0.75
            )

            # Create upsampling distribution
            prob_dist = np.ones(len(n)) - n * alpha
            prob_dist = prob_dist ** beta / sum(prob_dist)

            # Assign upsampling distribution to each score
            probs = np.ones(len(scores))
            scores = scores.ravel()
            for idx in range(len(bins) - 1):
                probs[(scores > bins[idx]) & (scores < bins[idx + 1])] = (
                    1 * prob_dist[idx]
                )
            scaled_probs = probs / sum(probs)

            # Select indices to upsample
            idxs = np.random.choice(
                list(range(len(scores))),
                p=scaled_probs,
                size=self.params["upsampling_factor"],
            )

            # Create upsampling data subset with random noise
            augmented_lsr = np.zeros((len(idxs), lsr.shape[1]))
            augemented_nlp = np.zeros((len(idxs), nlp.shape[1]))
            augmented_scores = np.zeros((len(idxs), scores.shape[1]))
            lsr_std = lsr.std(axis=0)
            nlp_std = nlp.std(axis=0)
            scores_std = scores.std(axis=0)
            for i, value in enumerate(idxs):
                augmented_lsr[i, :] = lsr[value, :] + np.random.normal(
                    0, lsr_std * gamma, lsr.shape[1]
                )
                augemented_nlp[i, :] = nlp[value, :] + np.random.normal(
                    0, nlp_std * gamma, nlp.shape[1]
                )
                augmented_scores[i, :] = scores[value, :] + np.random.normal(
                    0, scores_std * gamma, scores.shape[1]
                )

            # Concatenate initial data with upsampled data
            final_lsr = np.concatenate([lsr, augmented_lsr], axis=0)
            final_nlp = np.concatenate([nlp, augemented_nlp], axis=0)
            final_scores = np.concatenate([scores, augmented_scores], axis=0)

            if self.bUseConv:
                final_lsr = final_lsr.reshape(-1, 2, 1024)

        else:
            final_lsr = lsr
            final_nlp = nlp
            final_scores = scores
        res = namedtuple("res", ["lsr", "feats", "scores"])(
            lsr=final_lsr, feats=final_nlp, scores=final_scores
        )

        return res

    def write_predictions(self):
        """Output the predictions to a text file."""

        res = FeatureExtractor("test").run()
        model = torch.load("model.pt")
        test = namedtuple("res", ["lsr", "feats", "scores"])(
            lsr=res.lsr.reshape(-1, 2048), feats=res.feats, scores=res.scores
        )
        dev_ = data_utils.TensorDataset(
            *[
                torch.tensor(getattr(test, i)).float()
                for i in ["lsr", "feats", "scores"]
            ]
        )
        with torch.no_grad():
            preds = model.forward(*dev_.tensors[:2]).cpu().numpy()
        np.set_printoptions(suppress=True)
        np.savetxt("predictions.txt", preds.astype(float), delimiter="\n", fmt="%f")
        print("Predictions saved to predictions.txt")

    def run(self):
        """Run whole data loading, feature extraction, model training and regressing pipeline."""
        if self.mode == "extract":
            print("Extracting features")
            train = FeatureExtractor("train").run()
            dev = FeatureExtractor("dev").run()

            print("Saving features")
            np.save("saved_features/train_lsr", train.lsr)
            np.save("saved_features/train_nlp", train.feats)
            np.save("saved_features/train_scores", train.scores)
            np.save("saved_features/dev_lsr", dev.lsr)
            np.save("saved_features/dev_nlp", dev.feats)
            np.save("saved_features/dev_scores", dev.scores)
        else:  # Load saved extracted features
            print("Loading saved features")
            split = False if self.full_data else True
            train, dev = load_features(split=split, nt=True)

        if self.params["upsample"]:
            train = self.upsample(train)

        train_loader = create_loader(train, self.params["batch_size_train"])
        dev_loader = create_loader(dev, validate=True)

        # We set a random seed to ensure that results are reproducible.
        # Also set a cuda GPU if available
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            GPU = True
        else:
            GPU = False
        device_idx = 0
        if GPU:
            device = torch.device(
                "cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu"
            )
        else:
            device = torch.device("cpu")
        print(f"Running on {device}")

        if self.bUseConv:
            model = RecursiveNN(
                ModelBlock,
                self.params["conv_dict"],
                self.params["conv_ffnn_dict"],
                BASELINE_dim=self.params["NBaseline"],
            )
        else:
            model = RecursiveNN_Linear(
                in_features=2048,
                N1=self.params["N1"],
                N2=self.params["N2"],
                out_features=self.params["out_features"],
                dropout=self.params["dropout"],
                leaky_relu=self.params["leaky_relu"],
            )

        model = model.to(device)

        weights_initialiser = True
        if weights_initialiser:
            model.apply(weights_init)
        params_net = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Total number of parameters in Model is: {}".format(params_net))
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=self.params["lr"])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.params["step_size"], gamma=self.params["gamma"]
        )

        date_string = (
            str(datetime.datetime.now())[:16].replace(":", "-").replace(" ", "-")
        )
        writer = SummaryWriter(logdir + date_string)
        print("Running model")
        for epoch in range(self.params["epochs"]):
            train_model(
                model,
                train_loader,
                optimizer,
                epoch,
                log_interval=1000,
                scheduler=scheduler,
                writer=writer,
            )
            test_loss = test_model(model, dev_loader, epoch, writer=writer)

        torch.save(model, "model.pt")
        self.model = model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("mode", type=str, nargs="+", help="extract or no-extract")
    parser.add_argument("save", type=str, nargs="+", help="T / F for save or not save")

    args = parser.parse_args().__dict__

    Rosetta(args["mode"][0], args["save"][0]).run()
