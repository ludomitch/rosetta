# Standard
import string
import csv
from collections import namedtuple

# Data science
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# NLP
from textblob_de import TextBlobDE as TBD
from textblob import TextBlob as TBE
import spacy
import language_check
from laserembeddings import Laser


def spacy_parser(x, y, mode="pos_"):
    """Parse POS tags and entities to retrieve features."""
    whitelist = ["PER", "PERSON", "LOC", "ORG"]
    if mode in ["ents"]:
        mode = "label_"
        x = x.ents
        y = y.ents
    x = [getattr(i, mode) for i in x]
    x = {k: x.count(k) for k in x if k}
    y = [getattr(i, mode) for i in y]
    y = {k: y.count(k) for k in y if k}
    if mode in ["label_"]:
        if "PERSON" in x:
            x["PER"] = x.pop("PERSON")
        x = {k: v for k, v in x.items() if k in whitelist}
        y = {k: v for k, v in y.items() if k in whitelist}

    if len(x) > len(y):
        it = x
        nit = y
    else:
        it = y
        nit = x
    res = 0
    for pos in it:
        if pos in nit:
            res += abs(it[pos] - nit[pos])
        else:
            res += it[pos]
    return res


class FeatureExtractor:
    def __init__(self, mode="train"):
        self.mode = mode

        self.src = None
        self.tgt = None
        self.scores = None

        self.df = None

        self.laser = Laser()

    def load_data(self):
        # Base df with three columns
        path = f"en-de/{self.mode}.ende"
        src = pd.read_csv(
            f"{path}.src",
            sep="\n",
            error_bad_lines=False,
            quoting=csv.QUOTE_NONE,
            header=None,
        )
        target = pd.read_csv(
            f"{path}.mt",
            sep="\n",
            error_bad_lines=False,
            quoting=csv.QUOTE_NONE,
            header=None,
        )

        df = src.rename(columns={0: "src"})

        if self.mode != "test":
            scores = pd.read_csv(
                f"{path}.scores",
                sep="\n",
                error_bad_lines=False,
                quoting=csv.QUOTE_NONE,
                header=None,
            )
            df["scores"] = scores
        else:
            df["scores"] = [
                0 for _ in range(len(target))
            ]  # just placeholder, not used for test
        df["tgt"] = target
        setattr(self, "df", df)
        return df

    def laser_embeddings(self):
        """Extract laser embeddings and reshape appropriately."""
        src = self.laser.embed_sentences(
            self.df["src"].tolist(), lang="en"
        )  # (N, 1024)
        tgt = self.laser.embed_sentences(
            self.df["tgt"].tolist(), lang="de"
        )  # (N, 1024)
        res = np.zeros((src.shape[0], 2, 1024))  # (N, 2, 1024) ndarray
        res[:, 0, :] = src
        res[:, 1, :] = tgt

        # Scale embeddings
        res = MinMaxScaler().fit_transform(res)

        return res

    def features(self):
        """Extract baseline features"""
        sp_en = spacy.load("en")
        sp_de = spacy.load("de")
        en_checker = language_check.LanguageTool("en-GB")
        ge_checker = language_check.LanguageTool("de-DE")

        ft = self.df.copy()
        # Sentences without punctuation
        ft[["src_p", "tgt_p"]] = ft[["src", "tgt"]].applymap(
            lambda x: x.lower().translate(str.maketrans("", "", string.punctuation))
        )
        # Number of tokens
        ft["src_len"] = ft["src_p"].apply(lambda x: len(x.split(" ")))
        ft["tgt_len"] = ft["tgt_p"].apply(lambda x: len(x.split(" ")))
        count = lambda l1, l2: sum([1 for x in l1 if x in l2])
        # Number of non alphanumeric characters
        ft["src_#punc"] = ft["src"].apply(lambda x: count(x, set(string.punctuation)))
        ft["tgt_#punc"] = ft["tgt"].apply(lambda x: count(x, set(string.punctuation)))
        # Sentiment analysis
        ft["tgt_polar"] = ft["tgt"].apply(lambda x: TBD(x).sentiment.polarity)
        ft["src_polar"] = ft["src"].apply(lambda x: TBE(x).sentiment.polarity)
        ft["polar_ftf"] = (ft["tgt_polar"] - ft["src_polar"]).abs()
        # Spacy encoding
        ft["src_sp"] = ft["src"].apply(lambda x: sp_en(x))
        ft["tgt_sp"] = ft["tgt"].apply(lambda x: sp_de(x))
        # Proofread errors
        ft["sp_pos_diff"] = [
            spacy_parser(x, y, "pos_") for x, y in zip(ft["src_sp"], ft["tgt_sp"])
        ]
        ft["sp_ent_diff"] = [
            spacy_parser(x, y, "ents") for x, y in zip(ft["src_sp"], ft["tgt_sp"])
        ]
        ft["src_gram_err"] = ft["src"].apply(lambda x: len(en_checker.check(x)))
        ft["tgt_gram_err"] = ft["tgt"].apply(lambda x: len(ge_checker.check(x)))
        # Features of interest
        foi = [
            "src_len",
            "tgt_len",
            "src_#punc",
            "tgt_#punc",
            "tgt_polar",
            "src_polar",
            "src_gram_err",
            "tgt_gram_err",
            "sp_pos_diff",
            "sp_ent_diff",
        ]  # Features of interest

        features = ft[foi].values
        scaled_features = MinMaxScaler().fit_transform(features)

        return scaled_features

    def run(self):
        """Run feature extraction pipeline."""
        print("Loading data")
        self.load_data()
        print("Extracting Laser Embeddings")
        laser_embeds = self.laser_embeddings()
        print(f"Laser features extracted, shape: {laser_embeds.shape}")
        print("Extracting NLP features")
        features = self.features()
        print(f"NLP features extracted, shape: {features.shape}")
        res = namedtuple("res", ["lsr", "feats", "scores"])(
            lsr=laser_embeds, feats=features, scores=self.df["scores"].values
        )
        return res
