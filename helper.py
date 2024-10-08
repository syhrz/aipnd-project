import argparse


class Config:
    def __init__(self):
        self.architecture = "densenet121"
        self.data_dir = "data"
        self.checkpoint_dir = "checkpoint"
        self.hidden_layers = None
        self.learning_rate = 0.001
        self.dropout_probability = 0.3
        self.epochs = 5
        self.log_frequency = 50

        self.input_feature_size = {
            "densenet121": 1024,
            "densenet161": 2208,
            "vgg16": 25088,
        }

        self.hidden_layer_size = {
            "densenet121": [500],
            "densenet161": [1000, 500],
            "vgg16": [4096, 4096, 1000],
        }

        self.means = [0.485, 0.456, 0.406]
        self.std_devs = [0.229, 0.224, 0.225]
        self.img_max_size = 256
        self.img_center_crop = 224
        self.img_batch_size = 64

        self.model_checkpoint_path = ""
        self.image_path = ""
        self.train_batch_size = 0
        self.valid_batch_size = 0

    def update_from_train_args(self, args):
        """Update configuration from parsed command-line arguments."""
        self.architecture = args.architecture
        self.data_dir = args.data_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.hidden_layers = (
            [int(x) for x in args.hidden_layers.split(",")]
            if args.hidden_layers
            else None
        )
        self.learning_rate = args.learning_rate
        self.dropout_probability = args.dropout_probability
        self.epochs = args.epochs
        self.log_frequency = args.log_frequency

        # Validate configuration settings
        self._validate()

    def init_from_predict_args(self, args):
        """Update configuration from parsed command-line
        arguments for prediction.
        """
        self.model_checkpoint_path = args.model_checkpoint_path
        self.image_path = args.image_path

    def update_from_predict_args(self, args):
        """Update configuration from parsed command-line
        arguments for prediction.
        """

        # Optionally override architecture settings if provided
        if args.architecture:
            self.architecture = args.architecture
        if args.dropout_probability:
            self.dropout_probability = args.dropout_probability
        if args.learning_rate:
            self.learning_rate = args.learning_rate
        if args.epochs:
            self.epochs = args.epochs
        if args.train_batch_size:
            self.train_batch_size = args.train_batch_size
        if args.valid_batch_size:
            self.valid_batch_size = args.valid_batch_size

        # Validate configuration settings
        self._validate()

    def _validate(self):
        """Private method to validate the configuration settings."""
        if self.hidden_layers and len(self.hidden_layers) != len(
            self.hidden_layer_size.get(self.architecture, [])
        ):
            raise ValueError(
                "The number of hidden layers doesn't match "
                "the expected size for the chosen architecture."
            )
        if self.learning_rate <= 0:
            msg = "The learning rate must be greater than 0."
            raise ValueError(msg)
        if not (0 < self.dropout_probability < 1):
            msg = ("The dropout probability must be between 0 and 1.")
            raise ValueError(msg)
        if self.epochs <= 0:
            msg = "The number of epochs must be greater than 0."
            raise ValueError(msg)
        if self.log_frequency <= 0:
            msg = "Log frequency must be greater than 0."
            raise ValueError(msg)


def read_train_args(supported_models=None):
    """Parse command-line arguments and validate them."""
    if supported_models is None:
        raise ValueError("supported_models must be provided.")

    # Create parser
    parser = argparse.ArgumentParser(
        description="Parse input arguments for model training"
    )

    # Positional argument for data directory
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        nargs="?",
        default="data",
        help="Path to the dataset directory (default: ./data)",
    )

    # Optional argument for checkpoint directory
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoint",
        help="Path to the checkpoint directory (default: ./checkpoint)",
    )

    # Optional argument for model architecture
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        default="densenet121",
        choices=supported_models,
        help=(
            f"Model architecture (default: densenet121). "
            f"Available options: {", ".join(supported_models)}",
        ),
    )

    # Optional argument for learning rate
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    # Optional argument for dropout probability
    parser.add_argument(
        "-dp",
        "--dropout_probability",
        type=float,
        default=0.3,
        help="Dropout probability (default: 0.3)",
    )

    # Optional argument for hidden layers
    parser.add_argument(
        "-hl",
        "--hidden_layers",
        type=str,
        default=None,
        help=(
            "Hidden layers configuration as a comma-separated "
            "list (e.g., '500' or '1000,500')"
        ),
    )

    # Optional argument for epochs
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs to run (default: 5)",
    )

    # Optional argument for log frequency
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=50,
        help="Frequency of logging (default: 50)",
    )

    # Optional argument to enable GPU mode
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU mode (default: False)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Validate arguments
    if args.epochs < 1:
        msg = "The number of epochs must be greater than or equal to 1."
        parser.error(msg)
    if not (0 < args.dropout_probability < 1):
        msg = "The dropout probability must be between 0 and 1."
        parser.error(msg)
    if args.learning_rate <= 0:
        msg = "The learning rate must be greater than 0."
        parser.error()

    return args


def read_predict_args():
    # Create parser
    parser = argparse.ArgumentParser(
        description="Parse input arguments for model prediction"
    )

    # Required argument for checkpoint directory
    parser.add_argument(
        "-m", "--model_checkpoint_path",
        type=str,
        default=False,
        help=(
            "path to JSON file for mapping between "
            "class values and category names"
        ),
    )

    # Required argument for image path
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        default=False,
        help="Path to an image file",
    )

    # Optional argument to enable GPU mode
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU mode (default: False)",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args
