import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import multiprocessing
from joblib import Parallel, delayed
# Load metadata
train_df = pd.read_csv('D:\Projects\BirdCLF\BirdCLF_Data\\train.csv')
taxonomy_df = pd.read_csv('D:\Projects\BirdCLF\BirdCLF_Data\\taxonomy.csv')
# def process_audio(audio_path, output_dir, sr=32000, n_mels=128):
#     try:
#         y, sr = librosa.load(audio_path, sr=sr)
#         length_y = len(y)
#         i = 0
#         splitted_input = []
#
#         while i < length_y:
#             y_5sec = y[i:i + 5 * sr]
#             i += 5 * sr
#             if len(y_5sec) <= 5 * sr:
#                 y_5sec = np.pad(y_5sec, (0, (5 * sr - len(y_5sec))))
#             splitted_input.append(y_5sec)
#
#         mel_spec_db_list = []
#         for j in splitted_input:
#             mel_spec = librosa.feature.melspectrogram(y=j, sr=sr, n_mels=n_mels)
#             mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
#             mel_spec_db_list.append(mel_spec_db)
#
#         # Save as numpy arrays
#         folder_name = Path(audio_path).parent.name
#         file_name = Path(audio_path).stem
#         output_folder = Path(output_dir) / folder_name / file_name
#         output_folder.mkdir(parents=True, exist_ok=True)
#
#         for k, item in enumerate(mel_spec_db_list):
#             output_file = output_folder / f"mel_{k}.npy"
#             np.save(output_file, item)
#
#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
#
# def process_directory(input_dir, output_dir):
#     audio_files = list(Path(input_dir).rglob("*.ogg"))
#     Parallel(n_jobs=multiprocessing.cpu_count())(
#         delayed(process_audio)(file, output_dir) for file in audio_files
#     )
#
# input_directory = "D:\Projects\BirdCLF\BirdCLF_Data"
# output_directory = "D:\Projects\BirdCLF\Mel_Sepectogram_Output"
#
# process_directory(input_directory, output_directory)

def load_data(data_dir):
    """
    Loads mel spectrogram data from the directory and creates labels.

    Args:
        data_dir (str): Path to the main directory containing species folders.

    Returns:
        tuple: A tuple containing the data (numpy array) and labels (list).
    """
    data = []
    labels = []
    species_names = sorted(os.listdir(data_dir))
    label_encoder = LabelEncoder()
    label_encoder.fit(species_names)
    target_length = 24
    for species_name in species_names:
        species_dir = os.path.join(data_dir, species_name)
        if not os.path.isdir(species_dir):
            continue  # Skip non-directory files
        for subfolder in os.listdir(species_dir):
          subfolder_dir = os.path.join(species_dir,subfolder)
          if not os.path.isdir(subfolder_dir):
            continue
          each_file = []
          for file_name in os.listdir(subfolder_dir):
              if file_name.endswith('.npy'):
                  file_path = os.path.join(subfolder_dir, file_name)
                  mel_spectrogram = np.load(file_path)
                  each_file.append(mel_spectrogram)
          if len(each_file) <= target_length:
              each_file_ar = np.pad(each_file,(0,(target_length-len(each_file))))
              data.append(each_file_ar)
              labels.append(species_name)
    encoded_labels = label_encoder.transform(labels)
    max_freq = max(x.shape[1] for x in data)
    max_time = max(x.shape[2] for x in data)
    def pad_to_shape(arr, target_shape):
        padded = np.zeros(target_shape, dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        padded[slices] = arr[slices]
        return padded
    target_shape = (24, max_freq, max_time)
    padded_data = [pad_to_shape(x, target_shape) for x in data]
    return np.array(padded_data),np.array(encoded_labels),label_encoder.classes_
def create_model(input_shape, num_classes):
    """
    Creates a simple convolutional neural network model.

    Args:
        input_shape (tuple): Shape of the input mel spectrograms.
        num_classes (int): Number of species classes.

    Returns:
        tf.keras.Model: The created model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
def train_model(data_dir, epochs=10, batch_size=32, validation_split=0.2):
    """
    Trains the neural network model.

    Args:
        data_dir (str): Path to the main directory containing species folders.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
    """
    data, labels, species_names = load_data(data_dir)

    # Reshape the data to add a channel dimension
    # data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=validation_split, random_state=42)

    # Create the model
    input_shape = X_train.shape[1:]
    num_classes = len(species_names)
    model = create_model(input_shape, num_classes)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')
    model.save('myModel.h5')
    return model, species_names

data_directory = 'D:\Projects\BirdCLF\Mel_Sepectogram_Output_test'
trained_model, species_names = train_model(data_directory, epochs=5)