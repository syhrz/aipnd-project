#!/usr/bin/env python3

import logging
# import os
import sys

import torch
from PIL import Image
# from torch import nn, optim
from torchvision import models, transforms  # datasets,

from helper import Config, read_predict_args

# from datetime import datetime


# Initialize Logger
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs/predict.log")
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def checkpoint_loader(filepath, device):
    """
    Load a model and its state from a checkpoint file.

    Parameters:
    - filepath (str): Path to the checkpoint file.
    - device (torch.device): Device to load the model onto (CPU or GPU).

    Returns:
    - model (nn.Module): The model with the loaded state.
    - args (Namespace): The arguments used for model configuration.
    """

    # Load checkpoint to the appropriate device
    checkpoint = torch.load(filepath, map_location=device)

    # Extract model architecture and initialize
    architecture = checkpoint['architecture']
    model = models.__dict__[architecture.lower()](pretrained=True)

    # Update model with the checkpoint state
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    # Update args with the architecture from checkpoint
    args.architecture = architecture
    args.dropout_probability = checkpoint['dropout_probability']
    args.learning_rate = checkpoint['learning_rate']
    args.epochs = checkpoint['epochs']
    args.train_batch_size = checkpoint['train_batch_size']
    args.valid_batch_size = checkpoint['valid_batch_size']

    return model, args


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    means = [0.485, 0.456, 0.406]
    std_devs = [0.229, 0.224, 0.225]

    pil_image = Image.open(image_path).convert("RGB")

    # Any reason not to let transforms do all the work here?
    in_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, std_devs)])

    pil_image = in_transforms(pil_image)

    return pil_image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using
    a trained deep learning model.
    '''

    # # Set model to evaluation mode
    # model.eval()

    # # Ensure the model is on the same device as the image tensor
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # Preprocess and load the image (assume the function returns a tensor)
    image_tensor = process_image(image_path).unsqueeze(0).to(device)

    # Perform forward pass to get predictions
    with torch.no_grad():
        output = model(image_tensor)
        top_probabilities, top_indices = torch.topk(output, topk)

        # Convert probabilities to exponential form (softmax)
        top_probabilities = top_probabilities.exp()

    # Convert tensor results to numpy arrays
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probabilities, top_classes


if __name__ == "__main__":
    # get input arguments and print
    args = read_predict_args()

    logger.debug(f"args: {args}")

    # Create a Config instance
    config = Config()

    # Update config based on arguments
    config.init_from_predict_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, args = checkpoint_loader(config.model_checkpoint_path, device)
    config.update_from_predict_args(args)

    logger.info(f"System Information: {sys.version}")
    logger.info(f"PyTorch Version {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"Selected device: {device}")
    logger.info(f"args: {args}")
    logger.info(f"config: {config}")

    top_probabilities, top_classes = predict(
        config.image_path,
        config.model_checkpoint_path
    )

    # logger.info(f"Correct Class: {correct_class}")
    logger.info(f"Top Probabilities: {top_probabilities}")
    logger.info(f"Top Classes: {top_classes}")
