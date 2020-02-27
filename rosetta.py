# Standard
import argparse
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import csv
import string
import datetime
import os
from joblib import dump, load
import matplotlib.pyplot as plt

# NLP
# from textblob_de import TextBlobDE as TBD
# from textblob import TextBlob as TBE
# import spacy
# import language_check
# from laserembeddings import Laser

# ML
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from models import RecursiveNN, RecursiveNN_Linear, ModelBlock, View, weights_init
# Unused stuff
# import nltk
# from scipy.spatial.distance import cdist
# import kiwi
# import utils

logdir="./logs/"


def spacy_parser(x,y, mode='pos_'):
    # Models don't have the same entities
    whitelist = ['PER', 'PERSON', 'LOC', 'ORG']
    if mode in ['ents']:
        mode = 'label_'
        x = x.ents
        y = y.ents
    x = [getattr(i,mode) for i in x]
    x = {k:x.count(k) for k in x if k}
    y = [getattr(i, mode) for i in y]
    y = {k:y.count(k) for k in y if k}
    if mode in ['label_']:
        if 'PERSON' in x:
            x['PER'] = x.pop('PERSON')
        x = {k:v for k,v in x.items() if k in whitelist}
        y = {k:v for k,v in y.items() if k in whitelist}

    if len(x)>len(y):
        it = x
        nit = y
    else:
        it = y
        nit = x
    res = 0
    for pos in it:
        if pos in nit:
            res += abs(it[pos]-nit[pos])
        else:
            res += it[pos]
    return res

class FeatureExtractor:

    def __init__(self, mode='train'):
        self.mode = mode

        self.src = None
        self.tgt= None
        self.scores = None

        self.df = None

        self.laser = Laser()

    def load_data(self):
        # Base df with three columns
        path = f'en-de/{self.mode}.ende'
        src = pd.read_csv(f'{path}.src', sep="\n", error_bad_lines=False, quoting=csv.QUOTE_NONE, header=None)
        target = pd.read_csv(f'{path}.mt', sep="\n", error_bad_lines=False,quoting=csv.QUOTE_NONE, header=None)
        scores = pd.read_csv(f'{path}.scores', sep="\n", error_bad_lines=False,quoting=csv.QUOTE_NONE, header=None)
        df = src.rename(columns={0:'src'})
        df['tgt'] = target
        df['scores'] = scores
        setattr(self, 'df', df)
        return df

    def laser_embeddings(self):
        src = self.laser.embed_sentences(self.df['src'].tolist(), lang='en') # (N, 1024)
        tgt = self.laser.embed_sentences(self.df['tgt'].tolist(), lang='de') # (N, 1024)
        res = np.zeros((src.shape[0],2,1024)) # (N, 2, 1024) ndarray
        res[:,0,:]= src
        res[:,1,:] = tgt
        return res

    def features(self):
        sp_en = spacy.load("en")
        sp_de = spacy.load("de")
        en_checker = language_check.LanguageTool('en-GB')
        ge_checker = language_check.LanguageTool('de-DE')

        ft = self.df.copy()
        ft[['src_p', 'tgt_p']] = ft[['src', 'tgt']].applymap(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)))
        ft['src_len'] = ft['src_p'].apply(lambda x: len(x.split(' ')))
        ft['tgt_len'] = ft['tgt_p'].apply(lambda x: len(x.split(' ')))
        count = lambda l1,l2: sum([1 for x in l1 if x in l2])
        ft['src_#punc'] = ft['src'].apply(lambda x: count(x,set(string.punctuation)) )
        ft['tgt_#punc'] = ft['tgt'].apply(lambda x: count(x,set(string.punctuation)) )
        ft['tgt_polar'] = ft['tgt'].apply(lambda x: TBD(x).sentiment.polarity) # Already does lemmatization
        ft['src_polar'] = ft['src'].apply(lambda x: TBE(' '.join([i.lemmatize() for i in TBE(x).words])).sentiment.polarity)
        # ft['polar_ftf'] = (ft['tgt_polar']-ft['src_polar']).abs()
        ft['src_sp'] = ft['src'].apply(lambda x: sp_en(x))
        ft['tgt_sp'] = ft['tgt'].apply(lambda x: sp_de(x))
        ft['src_gram_err'] = ft['src'].apply(lambda x: len(en_checker.check(x)))
        ft['tgt_gram_err'] = ft['tgt'].apply(lambda x: len(ge_checker.check(x)))
        ft['sp_pos_diff'] = [spacy_parser(x,y, 'pos_') for x,y in zip(ft['src_sp'], ft['tgt_sp'])]
        ft['sp_ent_diff'] = [spacy_parser(x,y, 'ents') for x,y in zip(ft['src_sp'], ft['tgt_sp'])]
        foi = [ 'src_len', 'tgt_len', 'src_#punc',
               'tgt_#punc', 'tgt_polar', 'src_polar',
               'src_gram_err', 'tgt_gram_err', 'sp_pos_diff', 'sp_ent_diff'] # Features of interest

        features = ft[foi].values
        normalized_features = MinMaxScaler((-1,1)).fit_transform(features)

        return normalized_features

    def run(self):
        print("Loading data")
        self.load_data()
        print("Extracting Laser Embeddings")
        laser_embeds = self.laser_embeddings()
        print(f"Laser features extracted, shape: {laser_embeds.shape}")
        print("Extracting NLP features")
        features = self.features()
        print(f"NLP features extracted, shape: {features.shape}")
        res = namedtuple("res", ['lsr', 'feats', 'scores'])(
            lsr=laser_embeds, feats=features, scores=self.df['scores'].values)
        return res

def train_model(model, train_loader, optimizer, epoch, log_interval=100, scheduler=None, writer=None):
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
        tloss +=loss.item()

        loss.backward()

        optimizer.step()

        # if batch_idx % log_interval == 0:

    # tloss /= len(train_loader.dataset)
    print(
        "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
            epoch,
            batch_idx,
            tloss,
        )
    )

    if writer != None:
        writer.add_scalar('Train/Loss', tloss, epoch)
    if scheduler is not None:
        scheduler.step()


def test_model(model, test_loader, epoch, writer = None, scaler=None, upsample=False, score=True):

    test_loss = 0.0

    with torch.no_grad():
        for lsr, feats, targets in test_loader:
            outputs = model(lsr, feats)
            if upsample:
                targets = torch.tensor(scaler.transform(targets.reshape(-1,1)).ravel())
                targets = targets.float()
            test_loss += F.smooth_l1_loss(outputs, targets).item()

            # pred = outputs.argmax(dim=1, keepdim=True)

    # test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}\n".format(
            test_loss
        )
    )
    if writer != None:
        writer.add_scalar('Test/Loss', test_loss, epoch)

    return test_loss

class Rosetta:
    """Rosetta stone classifier"""
    def __init__(self, mode='extract', bSave = 'T', bUseConv = False):
        self.mode = mode
        self.scaler = None

        if bSave is 'T':
            print('GONE SAVE')
            self.bSave = False
        else:
            print('AINT GONE SAVE')
            print('Please specify F or T next time, setting false .......')
            self.bSave = False

        self.latent_space_laser = 16
        self.bUseConv = bUseConv

        if self.bUseConv:
            self.params = {
                "step_size" : 5,
                "gamma" : 0.8,
                "batch_size_train" : 64,
                "batch_size_test": 128,
                "lr" :4e-04,
                "epochs":40,
                "NBaseline":10,
                'upsampling_factor':3000,
                'upsample': False,

                "conv_dict":{
                'InChannels': [2],
                'OutChannels': [2],
                'Ksze': [1],
                'Stride': [1],
                'Padding': [0],
                'MaxPoolDim':1,
                'MaxPoolBool':False},

                "conv_ffnn_dict":{
                    'laser_hidden_layers':[64, self.latent_space_laser],
                    'mixture_hidden_layers':[32, 32, 1]}
                }
        else:
            self.params = {
                "N1" : 64,
                "N2" : 32,
                "lr" : 2e-4,
                "step_size" : 5,
                "gamma" : 0.8,
                "batch_size_train" : 32,
                "batch_size_test": 128,
                "epochs": 30,
                'upsampling_factor':5000,
                'upsample': False
            }

    def preprocess(self, scores, lsr, nlp):

        if self.params['upsample']:

            if self.bUseConv:
                lsr = lsr.reshape(-1, 2048)
            scores = scores.reshape(-1,1)

            idxs = np.nonzero((scores.ravel()<1.6)&(scores.ravel()>-3.5)) # Get indices to keep
            filtered_lsr = lsr[idxs]
            filtered_nlp = nlp[idxs]
            filtered_scores = scores[idxs]

            self.scaler = MinMaxScaler((-1,1))
            self.scaler.fit(filtered_scores)
            scaled_scores = self.scaler.transform(filtered_scores)

            n, bins, patches = plt.hist(scaled_scores, 15, density=True, range=(-1, 1), facecolor='g', alpha=0.75)

            prob_dist = np.ones(len(n))-n*0.35
            prob_dist = prob_dist**15/sum(prob_dist)

            dump(self.scaler, 'scaler.joblib')

            probs = np.ones(len(scaled_scores))
            scaled_scores = scaled_scores.ravel()
            for idx in range(len(bins)-1):
                probs[(scaled_scores>bins[idx])&(scaled_scores<bins[idx+1])] = 1*prob_dist[idx]
            scaled_probs = probs/sum(probs)

            idxs = np.random.choice(list(range(len(scaled_scores))), p=scaled_probs, size=self.params['upsampling_factor'])

            augmented_lsr = np.zeros((len(idxs), lsr.shape[1]))
            augemented_nlp = np.zeros((len(idxs), nlp.shape[1]))
            augmented_scores = np.zeros((len(idxs), scores.shape[1]))
            lsr_std = filtered_lsr.std(axis=0)
            nlp_std = filtered_nlp.std(axis=0)
            scores_std = filtered_scores.std(axis=0)
            for i, value in enumerate(idxs):
                augmented_lsr[i,:] = filtered_lsr[value, :]+ np.random.normal(0, lsr_std*0.05, lsr.shape[1])
                augemented_nlp[i,:] = filtered_nlp[value, :]+ np.random.normal(0, nlp_std*0.05, nlp.shape[1])
                augmented_scores[i,:] = filtered_scores[value, :]+ np.random.normal(0, scores_std*.05, scores.shape[1])

            final_lsr = np.concatenate([filtered_lsr, augmented_lsr],axis=0)
            final_nlp = np.concatenate([filtered_nlp, augemented_nlp],axis=0)
            final_scores = np.concatenate([filtered_scores, augmented_scores],axis=0)

            if self.bUseConv:
                final_lsr = final_lsr.reshape(-1, 2, 1024)

        else:
            final_lsr = lsr
            final_nlp = nlp
            final_scores = scores
        res = namedtuple("res", ['lsr', 'feats', 'scores'])(
                    lsr=final_lsr, feats=final_nlp, scores=final_scores)

        return res

    def normalise(self, dataset):
        return (dataset - self.min_train)/(self.max_train-self.min_train)

    def de_normalise(self, dataset):
        return dataset*(self.max_train-self.min_train) + self.min_train

    def run(self):
        if self.mode == 'extract':
            print("Extracting features")
            train = FeatureExtractor('train').run()
            dev = FeatureExtractor('dev').run()
            # test = FeatureExtractor('test').run()

            print("Saving features")
            np.save("saved_features/train_lsr", train.lsr)
            np.save("saved_features/train_nlp", train.feats)
            np.save("saved_features/train_scores", train.scores)
            np.save("saved_features/dev_lsr", dev.lsr)
            np.save("saved_features/dev_nlp", dev.feats)
            np.save("saved_features/dev_scores", dev.scores)
        else: # Load saved extracted features
            print("Loading saved features")
            trainlsr = np.load("saved_features/train_lsr.npy", allow_pickle=True)
            trainnlp = np.load("saved_features/train_nlp.npy", allow_pickle=True)
            trainsc = np.load("saved_features/train_scores.npy", allow_pickle=True)

            devlsr = np.load("saved_features/dev_lsr.npy",  allow_pickle=True)
            devnlp = np.load("saved_features/dev_nlp.npy", allow_pickle=True)
            devsc = np.load("saved_features/dev_scores.npy", allow_pickle=True)

            self.min_train = np.min(trainlsr,axis = 0)
            self.max_train = np.max(trainlsr,axis = 0)

            trainlsr = self.normalise(trainlsr)
            devlsr = self.normalise(devlsr)

            if not self.bUseConv:
                trainlsr = trainlsr.reshape(-1, 2048)
                devlsr = devlsr.reshape(-1, 2048)

        dev = namedtuple("res", ['lsr', 'feats', 'scores'])(
            lsr=devlsr, feats=devnlp, scores=devsc)

        train = self.preprocess(trainsc, trainlsr, trainnlp)

        params = self.params

        train_ = data_utils.TensorDataset(*[torch.tensor(getattr(train, i)).float() for i in ['lsr', 'feats', 'scores']])
        train_loader = data_utils.DataLoader(train_, batch_size = params['batch_size_train'], shuffle = True)

        dev_ = data_utils.TensorDataset(*[torch.tensor(getattr(dev, i)).float() for i in ['lsr', 'feats', 'scores']])
        dev_loader = data_utils.DataLoader(dev_, batch_size = params['batch_size_test'], shuffle = True)

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
            device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        print(device)

        if self.bUseConv:
            model = RecursiveNN(ModelBlock, params["conv_dict"], params["conv_ffnn_dict"], BASELINE_dim=params["NBaseline"])
        else:
            model = RecursiveNN_Linear(in_features=2048, N1=params["N1"], N2=params["N2"], out_features=16) # 2048

        model = model.to(device)

        weights_initialiser = True
        if weights_initialiser:
            model.apply(weights_init)
        params_net = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters in Model is: {}".format(params_net))
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"],gamma=params["gamma"])
        # scheduler = None

        date_string = str(datetime.datetime.now())[:16].replace(":", "-").replace(" ", "-")
        writer = SummaryWriter(logdir + date_string)
        running_test_loss = 1000
        print('Running model')
        for epoch in range(params['epochs']):
            model.train()
            train_model(model, train_loader, optimizer, epoch, log_interval=1000,scheduler=scheduler, writer = writer)
            model.eval()
            test_loss = test_model(model, dev_loader, epoch, writer = writer, scaler=self.scaler, upsample=self.params['upsample'], score=False)
            if (test_loss-running_test_loss)>0.05:
                break
            else:
                running_test_loss=test_loss
        # os.mkdir("./models/" + date_string)
        # torch.save(model, "./models/"+date_string+"/model.pt")
        # with open("./models/"+ date_string+'/params.txt', 'w+') as json_file:
        #     json.dump(params, json_file)
        torch.save(model, 'model.pt')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process input args')
    parser.add_argument('mode', type=str, nargs='+',
                        help='extract or no-extract')
    args = parser.parse_args().__dict__

    parser.add_argument('save', type=str, nargs='+',
                        help='T / F for save or not save')

    args = parser.parse_args().__dict__

    ros = Rosetta(args['mode'][0], args['save'][0]).run()

# OTher features
# Count numbers, count capital words
