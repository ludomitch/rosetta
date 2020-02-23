# Standard
import argparse
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import csv
import string

# NLP
from textblob_de import TextBlobDE as TBD
from textblob import TextBlob as TBE
import spacy
import language_check
from laserembeddings import Laser

# ML
from sklearn.preprocessing import Normalizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch import optim
from torch.utils.tensorboard import SummaryWriter

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
        ft['tgt_polar'] = ft['tgt'].apply(lambda x: TBD(x).sentiment.polarity)
        ft['src_polar'] = ft['src'].apply(lambda x: TBE(x).sentiment.polarity)
        ft['polar_ftf'] = (ft['tgt_polar']-ft['src_polar']).abs()
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
        normalized_features = Normalizer().fit_transform(features)

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

# MODULES
class ModelBlock(nn.Module):
    def __init__(self, block):
        super(ModelBlock, self).__init__()
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class RecursiveNN(nn.Module):
    def __init__(self, ModelBlock, conv_diction, ffnn_diction, BASELINE_dim = 10):
        super(RecursiveNN, self).__init__()

        self.BASELINE_dim = BASELINE_dim
        self.Hin = [1024]
        self.Hout = []
        self.Win = [1]
        self.Dropout_p = 0.05

        # Convolution Variables
        self.conv_dict = conv_diction

        self.Kszes = conv_diction['Ksze']
        self.InChannels = conv_diction['InChannels']
        self.OutChannels = conv_diction['OutChannels']
        self.Strides = conv_diction['Stride']
        self.Paddings = conv_diction['Padding']

        self.PoolingDim = self.conv_dict.pop('MaxPoolDim')
        self.PoolingBool = self.conv_dict.pop('MaxPoolBool')

        # FFNN Variables
        self.ffnn_dict = ffnn_diction

        self.hidden_laser = ffnn_diction['laser_hidden_layers']
        self.hidden_mixture = ffnn_diction['mixture_hidden_layers']

        # Convolution of LASER embeddings
        conv_seq = self.make_conv_layer(ModelBlock)

        # FFNN
        # Joint features vector from LASER embeddings and BASELINE features
        ffnn_seq = self.make_ffnn_layer(ModelBlock)

        # Define the model
        self.conv_seq = conv_seq
        self.ffnn_seq = ffnn_seq

        # PRINT TO VISUALISE DURING INITIALISATION
        # print(self.conv_seq)
        # print()
        # print(self.ffnn_seq)

    def make_conv_layer(self, ModelBlock):
        layers = []

        # Create a fully convolutional layer
        for idx in range(len(self.Strides)):

            self.Hout.append(int((self.Hin[idx]-self.Kszes[idx]+2*self.Paddings[idx])/(self.Strides[idx]) + 1))
            if idx is not len(self.Strides):
                self.Hin.append(int(self.Hout[idx]))

            layer_subset = [self.conv_dict[feat][idx] for feat in self.conv_dict.keys()]
            block = [nn.Conv1d(*layer_subset),
                     nn.BatchNorm1d(self.OutChannels[idx]),
                     nn.ReLU(inplace=True),
                     nn.Dropout(p=self.Dropout_p)]
            module_block = ModelBlock(block)
            layers.append(module_block)

        if self.PoolingBool:
            layers.append(nn.MaxPool1d(self.PoolingDim, self.PoolingDim))
            self.Hout.append(self.Hout[-1]/self.PoolingDim)

        nfc = int(self.Hout[-1])*int(self.OutChannels[-1])
        self.hidden_laser.insert(0, nfc)

        layers.append(View((-1,nfc)))

        # Now make a FFNN from convolutional layer output into latent space size
        for idx in range(len(self.hidden_laser)-1):
            block = [nn.Linear(self.hidden_laser[idx],self.hidden_laser[idx + 1], bias = True),
                     nn.ReLU(inplace = True)]
            module_block = ModelBlock(block)
            layers.append(module_block)

        return nn.Sequential(*layers)

    def make_ffnn_layer(self, ModelBlock):
        layers = []

        # Add the mixture FFNN combining baseline with convolution from LASER
        for idx in range(len(self.hidden_mixture)):

            if idx == 0:
                block = [nn.Linear(self.hidden_laser[-1] + self.BASELINE_dim, self.hidden_mixture[idx], bias = True),
                     nn.ReLU(inplace = True)]
            else:
                block = [nn.Linear(self.hidden_mixture[idx - 1], self.hidden_mixture[idx], bias = True),
                    nn.ReLU(inplace = True)]

            module_block = ModelBlock(block)
            layers.append(module_block)

        return nn.Sequential(*layers)

    def forward(self, laser_inputs, baseline_features):

        out = self.conv_seq(laser_inputs)
        out = torch.cat([out, baseline_features], dim = 1)
        out = self.ffnn_seq(out)

        return out.view(-1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def loss_function(out, label):
    loss = criterion(out, label)
    return loss

def train_model(model, train_loader, optimizer, epoch, log_interval=100, scheduler=None, writer=None):
    tloss = 0

    for batch_idx, (lsr, feats, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(lsr, feats)

        loss = F.mse_loss(outputs, targets)
        tloss +=loss.item()

        loss.backward()

        optimizer.step()

        # if batch_idx % log_interval == 0:
        #     print(
        #         "Train Epoch: {:02d} -- Batch: {:03d} -- Loss: {:.4f}".format(
        #             epoch,
        #             batch_idx,
        #             loss.item(),
        #         )
        #     )

    tloss /= 100
    if writer != None:
        writer.add_scalar('Train/Loss', tloss, epoch)
    if scheduler != None:
        scheduler.step()


def test_model(model, test_loader, epoch, writer = None):

    test_loss = 0.0

    with torch.no_grad():
        for lsr, feats, targets in test_loader:
            outputs = model(lsr, feats)

            test_loss += F.mse_loss(outputs, targets, reduction="sum").item()

            # pred = outputs.argmax(dim=1, keepdim=True)

    test_loss /= len(test_loader.dataset)

    # print(
    #     "\nTest set: Average loss: {:.4f}\n".format(
    #         test_loss
    #     )
    # )

    if writer != None:
        writer.add_scalar('Test/Loss', test_loss, epoch)

class Rosetta:
    """Rosetta stone classifier"""
    def __init__(self, mode='extract'):
        self.mode = mode

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

            train = namedtuple("res", ['lsr', 'feats', 'scores'])(
                        lsr=trainlsr, feats=trainnlp, scores=trainsc)
            dev = namedtuple("res", ['lsr', 'feats', 'scores'])(
                lsr=devlsr, feats=devnlp, scores=devsc)

        # Feature Extractor Size from LASER Embeddings
        latent_space_laser = 10

        hyperParams = {
            "step_size" : 2,
            "gamma" : 0.8,
            "normalising" : False,
            "batch_size_train" : 128,
            "batch_size_test":16,
            "lr" :5e-04,
            "n_epochs":10,
            "NBaseline":10,
            "conv_dict":{
            'InChannels': [2, 8, 16, 32],
            'OutChannels': [8, 16, 32, 64],
            'Ksze': [4, 4, 4, 4],
            'Stride': [2, 2, 2, 2],
            'Padding': [1, 1, 1, 1],
            'MaxPoolDim':2,
            'MaxPoolBool':True
            },

            "conv_ffnn_dict":{
                'laser_hidden_layers':[40, 20, latent_space_laser],
                'mixture_hidden_layers':[8, 1]
            }

        }

        train_ = data_utils.TensorDataset(*[torch.tensor(getattr(train, i)).float() for i in ['lsr', 'feats', 'scores']])
        train_loader = data_utils.DataLoader(train_, batch_size = hyperParams["batch_size_train"], shuffle = True)

        dev_ = data_utils.TensorDataset(*[torch.tensor(getattr(dev, i)).float() for i in ['lsr', 'feats', 'scores']])
        dev_loader = data_utils.DataLoader(dev_, batch_size = hyperParams["batch_size_test"], shuffle = True)

        # test_ = data_utils.TensorDataset(*[torch.tensor(getattr(test, i)) for i in ['lsr', 'feats', 'scores']])
        # test_loader = data_utils.DataLoader(test_, batch_size = batch_size_test, shuffle = True)

        weights_initialiser = True

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

        model = RecursiveNN(ModelBlock, hyperParams["conv_dict"], hyperParams["conv_ffnn_dict"], BASELINE_dim=hyperParams["NBaseline"])
        model = model.to(device)

        if weights_initialiser:
            model.apply(weights_init)
        params_net = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters in Model is: {}".format(params_net))
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=hyperParams["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperParams["step_size"],gamma=hyperParams["gamma"])
        writer = SummaryWriter(logdir + "8")

        # Main Training Loop
        for epoch in range(hyperParams['n_epochs']):
            train_model(model, train_loader, optimizer, epoch, log_interval=1000, scheduler=scheduler, writer = writer)
            test_model(model, dev_loader, epoch, writer = writer)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Process input args')
    parser.add_argument('mode', type=str, nargs='+',
                        help='extract or no-extract')
    args = parser.parse_args().__dict__

    ros = Rosetta(args['mode'][0]).run()

# OTher features
# Count numbers, count capital words
