# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

# How To Run

## Setup Local Workspace

```bash
# Create a new Conda project
$ conda crate -n udacity-aipnd-project

# Activate the Conda project
$ conda activate udacity_aipnd_project

# Install required packages
$ conda install pytorch torchvision torchaudio matplotlib pytorch-cuda=12.4 -c pytorch -c nvidia
```

Download and put the data under `data` directory.

## Run The Notebook

First run the jupyter server.

```bash
$ jupyter notebook --no-browser

Jupyter Server 2.14.1 is running at:
    http://localhost:8888/tree?token=<token>
```

Configure VSCode to connect to jupyter with the given token, alternatively run it directly

```bash
$ jupyter notebook notebook.ipynb
```

## Run the Training

To start to train the model

```bash
$ ./train.py -a densenet121 -e 5 -d data/flowers --gpu
```

It will generate a checkpoint at `checkpoint` directory with naming `<timedate>_<model_architecture>.pth` that later can be use as parameter for prediction.

## Run the Predicition

```bash
$ ./predict.py -m checkpoint/<timedate>_<model_architecture>.pth -i data/flowers/valid/1/image_06739.jpg

```
