Rosetta Quality Estimation

## Requirements

This project requires Python v3.6.x and pip3 to run.

## Setup

Requirements:
- Python 3
- pip3
- Java 6.0+

1. Install dependencies

Dependency management is done through pip.

`pip3 install -r requirements.txt`

2. Configuring tree

You can change the hyperparameters explored as well as which dataset you wish to use in `config.py`. Should you want to use your own data, all you have to do is include it in the root directory named as `<dataset_name>_dataset.txt` and then change the `DATASET` variable in config.py to <dataset_name>.

The `TEST_DATASET` variable in config.py is only needed when you use the `full_run.py` and will be used as the unseen data on which you will finally evaluate the performance of the tree.

## Command line interface

```
Rosetta.

Usage:
  rosetta.py <mode> <save>

Options:
  -h --help     Show this screen.
```

## Available Scripts
From the root directory, you can run:

`python3 -m rosetta <mode> <save>` to run the rosetta training pipeline.
`<mode>` accepts the values 'no-extract' or 'extract' and decides respectively whether to load pre-generated features saved in the `/saved_features` directory or whether to extract the features from scratch.
`<save>` accepts 'T' of 'F'. T for if you want to save the model as model.pt and F if you do not want to save the model generated.

`python3 -m evolution` to run the evolutionary manager pipeline. No options are required as all config options are hard-coded into the python file.



