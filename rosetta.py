# Standard
import argparse
from collections import namedtuple
import datetime
# ML
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# DL
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter
# In house
from models import RecursiveNN, RecursiveNN_Linear, ModelBlock, weights_init
from feature_extraction import FeatureExtractor

logdir = "./logs/"


def train_model(
    model, train_loader, optimizer, epoch, log_interval=100, scheduler=None, writer=None
):
    """Manage the training process of the model for one epoch."""
    tloss = 0

    for batch_idx, (lsr, feats, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(lsr, feats)
        loss = F.mse_loss(outputs, targets.view(-1))
        tloss += loss.item()

        loss.backward()

        optimizer.step()

    print(
        "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
            epoch, batch_idx, tloss
        )
    )

    # Write loss to tensorboard
    if writer != None:
        writer.add_scalar("Train/Loss", tloss, epoch)
    if scheduler is not None:
        scheduler.step()


def test_model(
    model, test_loader, epoch, writer=None, scaler=None, upsample=False, score=False
):
    """Output test loss and/or score for one epoch."""

    test_loss = 0.0

    with torch.no_grad():
        for lsr, feats, targets in test_loader:
            outputs = model(lsr, feats)
            test_loss += F.mse_loss(outputs, targets).item()

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))

    # Write loss to tensorboard
    if writer != None:
        writer.add_scalar("Test/Loss", test_loss, epoch)
    if score: # For evolutionary algorithms
        df = pd.DataFrame({"real": targets, "preds": outputs}).fillna(0)
        df = df.corr().fillna(0)
        score = df["preds"]["real"]
        return test_loss, score
    else:
        return test_loss


class Rosetta:
    """Rosetta stone regressor.
    Main class orchestrating whole regression pipeline."""

    def __init__(self, mode="extract", bSave="T", bUseConv=False):
        self.mode = mode
        self.scaler = None

        if bSave is "T":
            self.bSave = False
        else:
            self.bSave = False

        # Laser latent space
        self.latent_space_laser = 24
        # Use Convolutional neural network
        self.bUseConv = bUseConv

        # Saved model
        self.model = None


        # Define all hyperparameters
        if self.bUseConv:
            self.params = {
                "step_size": 2,
                "gamma": 0.9,
                "batch_size_train": 512,
                "batch_size_test": 256,
                "lr": 2e-04,
                "epochs": 50,
                "NBaseline": 10,
                "upsampling_factor": 7000,
                "upsample": True,
                "conv_dict": {
                    "InChannels": [2, 8],
                    "OutChannels": [8, 1],
                    "Ksze": [2, 2],
                    "Stride": [2, 2],
                    "Padding": [1, 1],
                    "MaxPoolDim": 1,
                    "MaxPoolBool": True,
                },
                "conv_ffnn_dict": {
                    "laser_hidden_layers": [24, self.latent_space_laser],
                    "mixture_hidden_layers": [12, 1],
                },
            }
        else:
            self.params = {
                "N1": 32,
                "N2": 32,
                "lr": 3e-4,
                "step_size": 80,
                "gamma": 0.5,
                "batch_size_train": 300,
                "batch_size_test": 100,
                "epochs": 50,
                "upsampling_factor": 1000,
                "upsample": False,
            }

    def upsample(self, scores, lsr, nlp):
        """Upsample data."""

        if self.params["upsample"]:
            # Define parameters
            alpha = 0.45
            beta = 15
            gamma = 0.05

            if self.bUseConv:
                lsr = lsr.reshape(-1, 2048)

            # Retrieve score distribution in 15 bins
            n, bins, _ = plt.hist(
                scores,
                15,
                density=True,
                range=(-1, 1),
                facecolor="g",
                alpha=0.75,
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
            trainlsr = np.load("saved_features/train_lsr.npy", allow_pickle=True)
            trainnlp = np.load("saved_features/train_nlp.npy", allow_pickle=True)
            trainsc = np.load("saved_features/train_scores.npy", allow_pickle=True)

            devlsr = np.load("saved_features/dev_lsr.npy", allow_pickle=True)
            devnlp = np.load("saved_features/dev_nlp.npy", allow_pickle=True)
            devsc = np.load("saved_features/dev_scores.npy", allow_pickle=True)

            if not self.bUseConv:
                trainlsr = trainlsr.reshape(-1, 2048)
                devlsr = devlsr.reshape(-1, 2048)

        dev = namedtuple("res", ["lsr", "feats", "scores"])(
            lsr=devlsr, feats=devnlp, scores=devsc
        )

        # Upsample training set if necessary
        train = self.upsample(trainsc, trainlsr, trainnlp)

        train_ = data_utils.TensorDataset(
            *[
                torch.tensor(getattr(train, i)).float()
                for i in ["lsr", "feats", "scores"]
            ]
        )
        train_loader = data_utils.DataLoader(
            train_, batch_size=self.params["batch_size_train"], shuffle=True
        )

        dev_ = data_utils.TensorDataset(
            *[torch.tensor(getattr(dev, i)).float() for i in ["lsr", "feats", "scores"]]
        )
        dev_loader = data_utils.DataLoader(
            dev_, batch_size=self.params["batch_size_test"], shuffle=True
        )

        # test_ = data_utils.TensorDataset(*[torch.tensor(getattr(test, i)) for i in ['lsr', 'feats', 'scores']])
        # test_loader = data_utils.DataLoader(test_, batch_size = batch_size_test, shuffle = True)

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
                in_features=2048, N1=self.params["N1"], N2=self.params["N2"], out_features=5
            )  # 2048

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
            test_loss = test_model(
                model,
                dev_loader,
                epoch,
                writer=writer,
                scaler=self.scaler,
                upsample=self.self.params["upsample"],
            )

        torch.save(model, "model.pt")
        self.model = model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("mode", type=str, nargs="+", help="extract or no-extract")
    parser.add_argument("save", type=str, nargs="+", help="T / F for save or not save")

    args = parser.parse_args().__dict__

    Rosetta(args["mode"][0], args["save"][0]).run()
