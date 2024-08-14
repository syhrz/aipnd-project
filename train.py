#!/usr/bin/env python3

import logging
import os
import sys
from datetime import datetime

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

from helper import Config, read_train_args

# Initialize Logger
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
)
logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs/train.log")
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def load_data(config):
    """Loads and preprocesses image data for training, validation, and testing.

    Args:
        args (Namespace): Parsed command-line arguments containing paths
                          to the data directories and other parameters.

    Returns:
        tuple: A tuple containing:
            - image_datasets (dict): Datasets for training,
              validation, and testing.
            - dataloaders (dict): Dataloaders for training,
              validation, and testing.
    """

    train_dir = config.data_dir + "/train"
    valid_dir = config.data_dir + "/valid"
    test_dir = config.data_dir + "/test"

    # Ensure that the required directories are provided
    if not all([config.data_dir]):
        raise ValueError("Data directories must be specified.")

    # Define normalization values (mean and standard deviation)
    norm = transforms.Normalize(mean=config.means, std=config.std_devs)

    # Define data transformations for training, validation, and testing
    data_transformation = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(config.img_max_size),
                transforms.CenterCrop(config.img_center_crop),
                transforms.ToTensor(),
                norm,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(config.img_max_size),
                transforms.CenterCrop(config.img_center_crop),
                transforms.ToTensor(),
                norm,
            ]
        ),
    }

    # Load the datasets using ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(
            train_dir, transform=data_transformation["train"]
        ),
        "valid": datasets.ImageFolder(
            valid_dir, transform=data_transformation["valid"]
        ),
        "test": datasets.ImageFolder(
            test_dir, transform=data_transformation["test"]
        ),
    }

    # Define the dataloaders for each dataset
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"],
            batch_size=config.img_batch_size,
            shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"],
            batch_size=config.img_batch_size,
            shuffle=False,  # No need to shuffle validation data
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"],
            batch_size=config.img_batch_size,
            shuffle=False,  # No need to shuffle test data
        ),
    }

    return image_datasets, dataloaders


def classifier_builder(model, config, dataloaders):
    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Set the dropout probability
    dropout_probability = (
        config.dropout_probability if config.dropout_probability else 0.5
    )

    # Determine the output size based on the number of classes
    output_size = len(dataloaders["train"].dataset.classes)

    # Activation function, dropout layer, and output layer
    relu = nn.ReLU()
    dropout = nn.Dropout(dropout_probability)
    output = nn.LogSoftmax(dim=1)

    # Determine the hidden layer sizes
    if config.hidden_layers:
        hidden_sizes = list(map(int, config.hidden_layers.split(",")))
    else:
        hidden_sizes = config.hidden_layer_size[config.architecture.lower()]

    # Building the classifier layers
    layers = [
        nn.Linear(
            config.input_feature_size[config.architecture.lower()],
            hidden_sizes[0]
        ),
        relu,
    ]

    if "vgg" in config.architecture:
        layers.append(dropout)

    if len(hidden_sizes) > 1:
        for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            layers.append(nn.Linear(h1, h2))
            layers.append(relu)
            if "vgg" in config.architecture:
                layers.append(dropout)

    # Add the final output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    layers.append(output)

    # Assign the classifier to the model
    model.classifier = nn.Sequential(*layers)

    return model


# Function to validate model
def validate(model, dataloaders, criterion, device):
    model.eval()  # Set model to evaluation mode
    accuracy = 0
    validation_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in dataloaders["valid"]:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            validation_loss += criterion(outputs, labels).item()

            # Calculate accuracy
            probabilities = torch.exp(outputs)
            predictions = probabilities.max(dim=1)[1]
            equality = predictions == labels
            accuracy += equality.type(torch.FloatTensor).mean().item()

    # Compute average loss and accuracy
    validation_loss /= len(dataloaders["valid"])
    accuracy /= len(dataloaders["valid"])

    logger.debug(f"Validation Loss: {validation_loss:.3f}")
    logger.debug(f"Accuracy: {accuracy*100:.2f}%")

    return validation_loss, accuracy


# Train
def train(
    model,
    dataloaders,
    optimizer,
    criterion,
    epochs,
    log_frequency,
    learning_rate
):
    logger.info(f"Training model on {device}")

    model.to(device)
    start_time = datetime.now()

    logger.info(f"epochs: {epochs}")
    logger.info(f"log_frequency: {log_frequency}")
    logger.info(f"learning_rate: {learning_rate}")

    counter = 0

    for i in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in iter(dataloaders["train"]):
            counter += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if counter % log_frequency == 0:
                model.eval()

                with torch.no_grad():
                    validation_loss, accuracy = validate(
                        model, dataloaders, criterion, device
                    )

                valid_data_length = len(dataloaders["valid"])

                logger.info(f"Epoch: {i+1} / {epochs}")
                logger.info(f"Training Loss: {running_loss/log_frequency:.3f}")
                logger.info(
                    f"Validation Loss: {validation_loss/valid_data_length:.3f}"
                )
                logger.info(f"Accuracy: {accuracy*100:.2f}%")

                running_loss = 0

                model.train()

    run_time = datetime.now() - start_time

    logger.info(f"Training completed after {run_time}")

    return model


# Test
def test(model, dataloaders, criterion, device):
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    start_time = datetime.now()

    with torch.no_grad():  # Disable gradient calculation in testing
        for images, labels in dataloaders["test"]:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate average loss and accuracy
    test_loss /= len(dataloaders["test"])
    accuracy = correct / total

    # Log the results
    logger.info(f"Test Loss: {test_loss:.3f}")
    logger.info(f"Correct Prediction: {correct} / {total}")
    logger.info(f"Test Accuracy: {accuracy*100:.2f}%")

    run_time = datetime.now() - start_time
    logger.info(f"Test completed in {run_time}")

    # Store accuracy in the model object if needed
    model.test_accuracy = accuracy

    return model


def save_model(model, config):
    # Save the model checkpoint
    model.class_to_idx = dataloaders["train"].dataset.class_to_idx

    # Prepare checkpoint data
    checkpoint = {
        "architecture": config.architecture,
        "classifier": model.classifier,
        "epochs": config.epochs,
        "dropout_probability": config.dropout_probability,
        "learning_rate": config.learning_rate,
        "train_batch_size": dataloaders["train"].batch_size,
        "valid_batch_size": dataloaders["valid"].batch_size,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_idx": model.class_to_idx,
    }

    # Generate checkpoint filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"{timestamp}_{args.architecture}.pth"
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_filename)

    # Ensure the target directory exists
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Save the checkpoint
    try:
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            f"Checkpoint saved successfully: {checkpoint_filename} "
            f"in {config.checkpoint_dir}"
        )
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")


# Call for main function
if __name__ == "__main__":
    # Define supported models
    supported_models = ["densenet121", "densenet161", "vgg16"]

    # get input arguments and print
    args = read_train_args(supported_models=supported_models)

    logger.info(f"args: {args}")

    # Create a Config instance
    config = Config()

    # Update config based on arguments
    config.update_from_train_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f"System Information: {sys.version}")
    logger.info(f"PyTorch Version {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    logger.info(f"Selected device: {device}")
    logger.info(f"args: {args}")
    logger.info(f"config: {config}")

    # Load and transform the data sets
    image_datasets, dataloaders = load_data(config)

    # Load pre-trained model and replace with custom classifier
    model = models.__dict__[args.architecture](pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model = classifier_builder(model, config, dataloaders)

    logger.info(f"Model Architecture: {args.architecture}")
    logger.info(f"Classifier: {model.classifier}")
    logger.info(f"Training model with {device.type}")

    # set training criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), config.learning_rate)

    # Train the model
    model = train(
        model,
        dataloaders,
        optimizer,
        criterion,
        config.epochs,
        config.log_frequency,
        config.learning_rate,
    )

    # Test the model
    model = test(model, dataloaders, criterion, device)

    # Save the model checkpoint
    save_model(model, config)
