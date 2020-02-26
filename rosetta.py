# Standard
import argparse
from collections import namedtuple
import datetime
from joblib import dump

import numpy as np
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models import RecursiveNN, RecursiveNN_Linear, ModelBlock, weights_init
from feature_extraction import FeatureExtractor

logdir = "./logs/"


def train_model(
    model, train_loader, optimizer, epoch, log_interval=100, scheduler=None, writer=None
):
    tloss = 0

    for batch_idx, (lsr, feats, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(lsr, feats)
        loss = F.mse_loss(outputs, targets.view(-1))
        tloss += loss.item()

        loss.backward()

        optimizer.step()

        # if batch_idx % log_interval == 0:

    # tloss /= len(train_loader.dataset)
    print(
        "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
            epoch, batch_idx, tloss
        )
    )

    if writer != None:
        writer.add_scalar("Train/Loss", tloss, epoch)
    if scheduler is not None:
        scheduler.step()


def test_model(
    model, test_loader, epoch, writer=None, scaler=None, upsample=False, score=False
):

    test_loss = 0.0

    with torch.no_grad():
        for lsr, feats, targets in test_loader:
            outputs = model(lsr, feats)
            if upsample:
                # targets = torch.tensor(scaler.transform(targets.reshape(-1, 1)).ravel())
                # targets = targets.float()
                # targets = torch.tensor(scaler.transform(targets.reshape(-1, 1)).ravel())
                # targets = targets.float()
                pass
            test_loss += F.mse_loss(outputs, targets).item()

            # pred = outputs.argmax(dim=1, keepdim=True)

    # test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    if writer != None:
        writer.add_scalar("Test/Loss", test_loss, epoch)
    if score:
        df = pd.DataFrame({"real": targets, "preds": outputs}).fillna(0)
        # display(df)
        df = df.corr().fillna(0)
        # display(df)
        score = df["preds"]["real"]
        return test_loss, score
    else:
        return test_loss


class Rosetta:
    """Rosetta stone classifier"""

    def __init__(self, mode="extract", bSave="T", bUseConv=False):
        self.mode = mode
        self.scaler = None

        if bSave is "T":
            print("GONE SAVE")
            self.bSave = False
        else:
            print("AINT GONE SAVE")
            print("Please specify F or T next time, setting false .......")
            self.bSave = False

        self.latent_space_laser = 24
        self.bUseConv = bUseConv

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
                "lr": 5e-5,
                "step_size": 60,
                "gamma": 0.5,
                "batch_size_train": 300,
                "batch_size_test": 100,
                "epochs": 100,
                "upsampling_factor": 1000,
                "upsample": False,
            }

    def preprocess(self, scores, lsr, nlp):

        if self.params["upsample"]:

            if self.bUseConv:
                lsr = lsr.reshape(-1, 2048)
            scores = scores.reshape(-1, 1)

            # idxs = np.nonzero(
            #     (scores.ravel() < 1.6) & (scores.ravel() > -2.5)
            # )  # Get indices to keep
            # filtered_lsr = lsr[idxs]
            # filtered_nlp = nlp[idxs]
            # filtered_scores = scores[idxs]
            scaler = None
            # self.scaler = MinMaxScaler((-1, 1))
            # self.scaler.fit(filtered_scores)
            # scaled_scores = self.scaler.transform(filtered_scores)
            filtered_lsr = lsr
            filtered_nlp = nlp
            filtered_scores = scores
            scaled_scores= filtered_scores

            n, bins, _ = plt.hist(
                scaled_scores,
                15,
                density=True,
                range=(-1, 1),
                facecolor="g",
                alpha=0.75,
            )

            prob_dist = np.ones(len(n)) - n * 0.45
            prob_dist = prob_dist ** 15 / sum(prob_dist)

            # dump(self.scaler, "scaler.joblib")

            probs = np.ones(len(scaled_scores))
            scaled_scores = scaled_scores.ravel()
            for idx in range(len(bins) - 1):
                probs[(scaled_scores > bins[idx]) & (scaled_scores < bins[idx + 1])] = (
                    1 * prob_dist[idx]
                )
            scaled_probs = probs / sum(probs)

            idxs = np.random.choice(
                list(range(len(scaled_scores))),
                p=scaled_probs,
                size=self.params["upsampling_factor"],
            )

            augmented_lsr = np.zeros((len(idxs), lsr.shape[1]))
            augemented_nlp = np.zeros((len(idxs), nlp.shape[1]))
            augmented_scores = np.zeros((len(idxs), scores.shape[1]))
            lsr_std = filtered_lsr.std(axis=0)
            nlp_std = filtered_nlp.std(axis=0)
            scores_std = filtered_scores.std(axis=0)
            for i, value in enumerate(idxs):
                augmented_lsr[i, :] = filtered_lsr[value, :] + np.random.normal(
                    0, lsr_std * 0.05, lsr.shape[1]
                )
                augemented_nlp[i, :] = filtered_nlp[value, :] + np.random.normal(
                    0, nlp_std * 0.05, nlp.shape[1]
                )
                augmented_scores[i, :] = filtered_scores[value, :] + np.random.normal(
                    0, scores_std * 0.05, scores.shape[1]
                )

            final_lsr = np.concatenate([filtered_lsr, augmented_lsr], axis=0)
            final_nlp = np.concatenate([filtered_nlp, augemented_nlp], axis=0)
            final_scores = np.concatenate([filtered_scores, augmented_scores], axis=0)

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
        if self.mode == "extract":
            print("Extracting features")
            train = FeatureExtractor("train").run()
            dev = FeatureExtractor("dev").run()
            # test = FeatureExtractor('test').run()

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

        train = self.preprocess(trainsc, trainlsr, trainnlp)

        params = self.params

        train_ = data_utils.TensorDataset(
            *[
                torch.tensor(getattr(train, i)).float()
                for i in ["lsr", "feats", "scores"]
            ]
        )
        train_loader = data_utils.DataLoader(
            train_, batch_size=params["batch_size_train"], shuffle=True
        )

        dev_ = data_utils.TensorDataset(
            *[torch.tensor(getattr(dev, i)).float() for i in ["lsr", "feats", "scores"]]
        )
        dev_loader = data_utils.DataLoader(
            dev_, batch_size=params["batch_size_test"], shuffle=True
        )

        # test_ = data_utils.TensorDataset(*[torch.tensor(getattr(test, i)) for i in ['lsr', 'feats', 'scores']])
        # test_loader = data_utils.DataLoader(test_, batch_size = batch_size_test, shuffle = True)

        # We set a random seed to ensure that your results are reproducible.
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
        print(device)

        if self.bUseConv:
            model = RecursiveNN(
                ModelBlock,
                params["conv_dict"],
                params["conv_ffnn_dict"],
                BASELINE_dim=params["NBaseline"],
            )
        else:
            model = RecursiveNN_Linear(
                in_features=2048, N1=params["N1"], N2=params["N2"], out_features=5
            )  # 2048

        model = model.to(device)

        weights_initialiser = True
        if weights_initialiser:
            model.apply(weights_init)
        params_net = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters in Model is: {}".format(params_net))
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params["step_size"], gamma=params["gamma"]
        )
        # scheduler = None

        date_string = (
            str(datetime.datetime.now())[:16].replace(":", "-").replace(" ", "-")
        )
        writer = SummaryWriter(logdir + date_string)
        running_test_loss = 1000
        print("Running model")
        for epoch in range(params["epochs"]):
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
                upsample=self.params["upsample"],
            )
            if (test_loss - running_test_loss) > 0.01:
                break
            running_test_loss = test_loss
        # os.mkdir("./models/" + date_string)
        # torch.save(model, "./models/"+date_string+"/model.pt")
        # with open("./models/"+ date_string+'/params.txt', 'w+') as json_file:
        #     json.dump(params, json_file)
        torch.save(model, "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input args")
    parser.add_argument("mode", type=str, nargs="+", help="extract or no-extract")
    args = parser.parse_args().__dict__

    parser.add_argument("save", type=str, nargs="+", help="T / F for save or not save")

    args = parser.parse_args().__dict__

    Rosetta(args["mode"][0], args["save"][0]).run()

# OTher features
# Count numbers, count capital words
