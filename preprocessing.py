import os
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler_params = np.load('scaler_params.npy', allow_pickle=True).item()

# Initialize the scaler with the loaded parameters
scaler = StandardScaler()
scaler.mean_ = scaler_params['mean']
scaler.scale_ = scaler_params['scale']

# Path to your train folder containing data files
train_folder = 'train'

scaled_data_list = []

# Iterate through each file in the train folder
for filename in os.listdir(train_folder):
    if filename.endswith('.npy'):  # Assuming data files are in numpy format
        file_path = os.path.join(train_folder, filename)
        
        # Load data from file
        data = np.load(file_path)
        
        # Reshape and apply scaling transformation
        scaled_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        
        # Append scaled data to the list
        scaled_data_list.append(scaled_data)

# Stack the scaled data along the first dimension
train_data_normalized = np.stack(scaled_data_list, axis=0)

# Save the scaled data as "train_data_normalized.npy" in the same train folder
save_path = os.path.join(train_folder, 'train_data_normalized.npy')
np.save(save_path, train_data_normalized)

# Define the directory where numpy arrays are saved
# array_output_directory = "coarsened_array_outputs"

# # Load the filenames and extract time step indices
# filenames = os.listdir(array_output_directory)
# indices = [int(filename.split('_t')[1].split('.')[0]) for filename in filenames]

# # Shuffle the indices
# np.random.shuffle(indices)

# # Calculate the sizes of each subset
# total_samples = len(indices)
# train_size = int(0.8 * total_samples)
# test_size = int(0.1 * total_samples)
# val_size = total_samples - train_size - test_size

# # Assign indices to each subset
# train_indices = np.array(indices[:train_size])
# test_indices = np.array(indices[train_size:train_size + test_size])
# val_indices = np.array(indices[train_size + test_size:])

# print(train_indices)
# print(test_indices)
# print(val_indices)

# output_dir_train = "train"
# output_dir_test = "test"
# output_dir_val = "val"
# os.makedirs(output_dir_train, exist_ok=True)
# os.makedirs(output_dir_test, exist_ok=True)
# os.makedirs(output_dir_val, exist_ok=True)

# # Move the corresponding numpy arrays to their respective folders
# for index in train_indices:
#     filename = f'wind_speed_coarse_t{index}.npy'
#     src_path = os.path.join(array_output_directory, filename)
#     dst_path = os.path.join(output_dir_train, filename)
#     os.rename(src_path, dst_path)

# for index in test_indices:
#     filename = f'wind_speed_coarse_t{index}.npy'
#     src_path = os.path.join(array_output_directory, filename)
#     dst_path = os.path.join(output_dir_test, filename)
#     os.rename(src_path, dst_path)

# for index in val_indices:
#     filename = f'wind_speed_coarse_t{index}.npy'
#     src_path = os.path.join(array_output_directory, filename)
#     dst_path = os.path.join(output_dir_val, filename)
#     os.rename(src_path, dst_path)

# train_data = np.array([np.load(os.path.join(output_dir_train, filename)) for filename in os.listdir(output_dir_train)])
# test_data = np.array([np.load(os.path.join(output_dir_test, filename)) for filename in os.listdir(output_dir_test)])
# val_data = np.array([np.load(os.path.join(output_dir_val, filename)) for filename in os.listdir(output_dir_val)])

# # Perform z-score normalization on the training data
# scaler = StandardScaler()
# train_data_normalized = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)

# # Save the scaler parameters
# scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}

# # Apply the same transformation to the testing and validation sets
# test_data_normalized = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
# val_data_normalized = scaler.transform(val_data.reshape(-1, val_data.shape[-1])).reshape(val_data.shape)

# # Save the normalized datasets
# np.save(os.path.join(output_dir_train, "train_data_normalized.npy"), train_data_normalized)
# np.save(os.path.join(output_dir_test, "test_data_normalized.npy"), test_data_normalized)
# np.save(os.path.join(output_dir_val, "val_data_normalized.npy"), val_data_normalized)
