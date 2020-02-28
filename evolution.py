# Standard
import numpy as np
import datetime
import random
import copy

# ML
import torch
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from models import RecursiveNN_Linear
from rosetta import Rosetta, train_model, test_model

logdir = "./logs/"

folds = 3


def load_data():
    """Load and combine training data from both train and dev datasets."""
    trainlsr = np.load("saved_features/train_lsr.npy", allow_pickle=True)
    trainnlp = np.load("saved_features/train_nlp.npy", allow_pickle=True)
    trainsc = np.load("saved_features/train_scores.npy", allow_pickle=True)

    devlsr = np.load("saved_features/dev_lsr.npy", allow_pickle=True)
    devnlp = np.load("saved_features/dev_nlp.npy", allow_pickle=True)
    devsc = np.load("saved_features/dev_scores.npy", allow_pickle=True)
    trainlsr = trainlsr.reshape(-1, 2048)
    devlsr = devlsr.reshape(-1, 2048)

    all_train_lsr = np.append(trainlsr, devlsr, axis=0)
    all_train_nlp = np.append(trainnlp, devnlp, axis=0)
    all_train_sc = np.append(trainsc, devsc, axis=0)
    ros = Rosetta(mode="no_extract")
    train = ros.upsample(all_train_sc, all_train_lsr, all_train_nlp)
    train_ = np.concatenate(
        (train.lsr, train.feats, train.scores.reshape(-1, 1)), axis=1
    )
    return train_

def create_loader(data, individual, validate=False):
    """Create a dataloader."""

    if validate:
        batch_size = data.shape[0]
    else:
        batch_size = individual["batch_size_train"]
    dataset = data_utils.TensorDataset(
        *[
            torch.Tensor(data[:, :2048]),
            torch.Tensor(data[:, 2048:2058]),
            torch.Tensor(data[:, 2058:]),
        ]
    )
    loader = data_utils.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return loader

dataset = load_data()


def population_generator(pop, pop_size):
    """Generate a random population of size pop_size."""
    for _ in range(pop_size + 1):
        epochs = np.random.randint(low=1, high=100)
        pop.append(
            {
                "N1": np.random.randint(low=4, high=64),
                "N2": np.random.randint(low=4, high=64),
                "lr": np.random.randint(low=1, high=10)*1e-4,
                "step_size": np.random.randint(low=0, high=epochs),
                "gamma": np.random.random_sample(),
                "batch_size_train": np.random.randint(low=32, high=512),
                "epochs": epochs,
                "out_features": np.random.randint(low=1, high=15),
                "leaky_relu": bool(random.getrandbits(1)),
                "dropout": np.random.random_sample()*0.7
            }
        )
    return pop


def evolve(pop, lamda, mutation_rate, crossover_rate):
    """This function contains the evolution pipeline from one generation to the next."""

    # Shuffle population and randomly select some.
    new_pop = copy.deepcopy(pop[: int(lamda * len(pop))])  # adding lambda_best
    np.random.shuffle(pop)
    new_pop += list(np.random.choice(pop, int((1 - lamda) * len(pop)), replace=False))

    # Mutate
    if mutation_rate != None:
        for individual in new_pop:
            individual.pop("score")

            mutate_param = np.random.choice(
                [
                    "N1",
                    "N2",
                    "lr",
                    "step_size",
                    "gamma",
                    "batch_size_train",
                    "epochs",
                    "leaky_relu",
                    "out_features",
                    "dropout",
                    None,
                ],
                p=[(mutation_rate) / 10 for i in range(1,10)] + [1 - mutation_rate],
            )
            if mutate_param != None:
                print("Mutating {}:{}".format(mutate_param, individual[mutate_param]))
                if isinstance(individual[mutate_param], np.bool_):
                    if individual[mutate_param] == True:
                        m = 1
                    else:
                        m = 0

                    m = np.random.choice(np.random.normal(loc=m, size=10000))
                    if m > 0:
                        m = min(1, round(m))
                    if m < 0:
                        m = max(0, round(m))
                    if m <= 0:
                        individual[mutate_param] = False
                    else:
                        individual[mutate_param] = True

                else:
                    mutated = np.random.choice(
                        np.random.normal(loc=individual[mutate_param], size=10000)
                    )
                    if (individual[mutate_param] / 1).is_integer():
                        individual[mutate_param] = int(mutated)
                print(
                    "Mutated to {}:{}\n".format(mutate_param, individual[mutate_param])
                )

    # Crossover
    if crossover_rate != None:
        nb = int((crossover_rate / 2) * len(pop))
        params = [
                    "N1",
                    "N2",
                    "lr",
                    "step_size",
                    "gamma",
                    "batch_size_train",
                    "epochs",
                    "leaky_relu",
                    "out_features",
                    "dropout",
        ]
        cross1 = [new_pop.pop(random.randrange(len(new_pop))) for _ in range(nb)]
        cross2 = [new_pop.pop(random.randrange(len(new_pop))) for _ in range(nb)]
        for individual in range(len(cross1)):
            cross_param = np.random.choice(params)
            idx = params.index(cross_param)
            print(
                "Crossing {}\nwith\n{}\nAt position {}:{}\n".format(
                    cross1[individual], cross2[individual], idx, params[idx]
                )
            )

            for i in range(idx, len(params)):
                cross1[individual][params[i]], cross2[individual][params[i]] = (
                    cross2[individual][params[i]],
                    cross1[individual][params[i]],
                )
            print(
                "obtained {}\nAnd\n{}\n\n".format(
                    cross1[individual], cross2[individual]
                )
            )
            new_pop += [cross1[individual], cross2[individual]]

    # Reshuffle population
    np.random.shuffle(new_pop)

    return new_pop

def init_params(train, validate, individual):
    """Initialise model parameters"""

    model = RecursiveNN_Linear(
        2048,
        N1=individual["N1"],
        N2=individual["N2"],
        out_features=individual["out_features"],
        leaky_relu=individual['leaky_relu'],
        dropout=individual['dropout']
    )  # Create model with hyperparmeters "N1" and "N2"

    # Create clean loaders here
    loader_train = create_loader(train, individual)
    loader_val = create_loader(validate, individual, validate=True)

    return model, loader_train, loader_val

def split(arr, pos, n):
    """takes an array, splits it in 2 uneven arrays.
    arr : array : The array to be split
    pos : int : the position on which the split occurs
    n : int : the number of rows to take when splitting
    the upper matrix is of size n, the lower one is (size of arr) - N
    returns upper, lower"""
    upper = arr[pos : pos + n]
    lower = np.vstack((arr[:pos], arr[pos + n :]))

    return upper, lower


def cross_val(individual):
    """Conduct k fold cross validation for a single individual to find fitness."""
    
    # copy data to avoid damaging the dataset
    split_size = int(dataset.shape[0]/folds)
    # Shuffle data
    np.random.shuffle(dataset)

    # Init Score
    score = 0

    # Splitting validation from training
    for i in range(folds):
        print(f"-------------------- Validate/Train separation {i} --------------------")

        # Split data into validation and training set
        validate, train = split(dataset, i * split_size, split_size)

        # Initialise model and dataloaders
        model, train_loader, val_loader = init_params(train, validate, individual)

        # Initialise optimiser
        optimizer = torch.optim.Adam(
            model.parameters(), lr=individual["lr"], betas=(0.9, 0.999)
        )  # add beta to genotype?
        # Initialise scheduler
        #             scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = individual["step_size"], gamma = individual["gamma"])

        # Create Tensorboard logs
        date_string = (
            str(datetime.datetime.now())[:16].replace(":", "-").replace(" ", "-")
        )
        writer = SummaryWriter(logdir + date_string)


        scheduler = None

        # Train model
        for epoch in range(individual["epochs"]):
            train_model(
                model,
                train_loader,
                optimizer,
                epoch,
                log_interval=1000,
                scheduler=scheduler,
                writer=writer,
            )

        # test model on val set
        score += test_model(
            model,
            val_loader,
            epoch=0,
            writer=None,
            score=True,
        )[1]
    individual_score = score / folds
    print("Score ", individual_score)
    return individual_score


def run():
    """Run the whole evolutionary algorithm pipeline for a given number of iterations."""
    iterations = 10
    population_size = 25
    lamda = 0.1
    mutation_rate = 1
    crossover_rate = 1

    population = population_generator([], population_size)
    pop_scores = {k:[] for k in range(0,iterations)}
    for iteration in range(0,iterations):
        for individual in population:
            print(f"Individual params: {individual}")
            score = cross_val(individual)
            pop_scores[iteration].append(score)
            individual["score"] = score
        new_pop = sorted(population, key=lambda k: k["score"])
        new_pop = evolve(new_pop, lamda, mutation_rate, crossover_rate)
    print(new_pop)
    return new_pop, pop_scores
