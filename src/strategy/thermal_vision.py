import configparser
import sys

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import image_dataset_from_directory

from rotating_logger import RotatingLogger


class InvalidVisionConfig(Exception):
    pass


class ThermalVision:
    def __init__(self, config_path: str):
        """
        Initialize the thermal vision model
        :param config_path: path to config file
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.logger = RotatingLogger()
        self.logger.info("Initializing ThermalVision")

    def __load_data(self):
        """
        Load heatmaps into datasets
        """
        self.logger.info("Loading heatmaps")
        dataset = image_dataset_from_directory(
            self.config.get("loader", "heatmaps_path"),
            labels="inferred",
            label_mode="int",
            color_mode="grayscale",
            shuffle=True
        )
        self.logger.info("Creating train and test datasets")
        # Split into train and validation sets
        test_size = int(len(dataset) * self.config.getfloat("loader", "test_split"))
        train_size = len(dataset) - test_size
        train_ds = dataset.take(train_size)
        test_ds = dataset.skip(train_size)
        # Normalize
        normalization_layer = layers.Rescaling(1. / 255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        # Prefetch
        self.train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.class_names = dataset.class_names

    def __load_model(self) -> None:
        """
        Load model
        """
        self.logger.info("Loading model from file")
        self.model = models.load_model(self.config.get("cnn", "model_load_path"))

    def __build_model(self):
        """
        Build vision model
        """
        self.logger.info("Building model")
        model = models.Sequential([
            layers.Input((256, 256, 1)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        self.model = model

    def __train_model(self):
        """
        Train vision model
        """
        self.logger.info("Training model")
        if self.model is None:
            self.__build_model()
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.history = self.model.fit(self.train_ds, validation_data=self.test_ds,
                                      epochs=self.config.getint("cnn", "epochs"))

    def __save_model(self) -> None:
        """
        Save model
        """
        self.logger.info("Saving model to file")
        self.model.save(self.config.get("cnn", "model_save_path"))

    def __plot_training_history(self) -> None:
        """
        Plot training history
        """
        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 8))
        sns.lineplot(self.history.history['accuracy'], label='accuracy')
        sns.lineplot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.title(f'Model training history', fontsize=16)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.config.get("cnn", "training_history_path"))
        plt.close()

    def train(self):
        """
        Train the model
        """
        if bool(self.config.getint("cnn", "train")):
            self.__load_data()
            self.__build_model()
            self.__train_model()
            self.__plot_training_history()
        elif bool(self.config.getint("cnn", "load")):
            self.__load_model()
        else:
            self.logger.info("Configuration does not allow training or loading a model")
            raise InvalidVisionConfig

    def predict(self, image) -> str | None:
        if self.model is None:
            self.logger.error("No model available")
            return None
        return self.model.predict(image)
