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

# Logging

There are several logs under this project that located under `logs` directory

- console.log, logs from the notebook.
- train.log, logs from the train CLI script.
- predict.log, logs from the prediction CLI script.

# References

https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplot2grid.html
https://github.com/atan4583/aipnd-project/tree/master
https://github.com/cjimti/aipnd-project/tree/master
