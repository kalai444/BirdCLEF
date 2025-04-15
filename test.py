import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def load_data(data_dir):
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

data_directory = 'D:\Projects\BirdCLF\Mel_Sepectogram_Output_test'

data_1,encoded_lables,actual_name = load_data(data_directory)

print(data_1.shape)
# for s in zip (data_1,encoded_lables):
#     print(s.shape)