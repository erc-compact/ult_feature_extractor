import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import argparse
import time, sys
import numpy as np
import configparser
from PFDFile import PFD
import presto.prepfold as pp
import matplotlib.pyplot as plt
from PFDFeatureExtractor import PFDFeatureExtractor

def custom_normalize(data, method='standard', norm_type='l2'):
    """
    Normalizes the data based on the selected method: 
    'minmax', 'standard' (default), 'l1', or 'l2' normalization.
    
    Parameters:
    - data (np.array): Input 1D or 2D NumPy array to be normalized.
    - method (str): Normalization method to apply. 
      Options: 'minmax', 'standard', 'l1', 'l2'. Default is 'standard'.
    - norm_type (str): The type of norm to use if 'l1' or 'l2' is chosen. Default is 'l2'.
    
    Returns:
    - np.array: Normalized array.
    """
    
    # Handle 1D data by reshaping it to 2D for consistency
    is_1D = False
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        is_1D = True
    
    # Choose normalization method
    if method == 'minmax':
        # Apply min-max normalization to scale data between 0 and 1
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data)
    
    elif method == 'standard':
        # Apply standardization (0 mean, unit variance)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    
    elif method == 'l1' or method == 'l2':
        # Apply L1 or L2 norm-based normalization
        normalized_data = normalize(data, norm=norm_type, axis=1)
    
    else:
        raise ValueError("Unsupported normalization method. Choose 'minmax', 'standard', 'l1', or 'l2'.")
    
    # Reshape back to 1D if the input was 1D
    if is_1D:
        normalized_data = normalized_data.flatten()
    
    return normalized_data

def plot_data(original_data, normalized_data, plot_file_name="normalized_plot.png"):
    """
    Plots the original and normalized data side by side.
    
    Parameters:
    - original_data (np.array): The original data before normalization.
    - normalized_data (np.array): The normalized data.
    - plot_file_name (str): File name to save the plot. Default is 'normalized_plot.png'.
    """
    # Create a figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the original data
    axes[0].plot(original_data, 'o-', label="Original Data", color='blue')
    axes[0].set_title("Original Data")
    axes[0].legend()
    
    # Plot the normalized data
    axes[1].plot(normalized_data, 'o-', label="Normalized Data", color='green')
    axes[1].set_title("Normalized Data")
    axes[1].legend()
    
    # Add a common title and save the plot
    plt.suptitle("Original vs Normalized Data", fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_file_name)
    #plt.show()

def main():
    # Example 2D data (3 samples, 3 features)
    data = np.array([[10, 20, 30],
                     [15, 25, 35],
                     [20, 30, 40]])
                      self.pfd_instance.plot_subbands()
    
    # Specify the normalization method ('minmax', 'standard', 'l1', 'l2')
    method = 'l2'  # Change this to test different methods
    
    # Normalize the data using the specified method
    normalized_data = custom_normalize(data, method=method, norm_type='l2')
    
    # Plot and save the original and normalized data
    plot_data(data, normalized_data, plot_file_name="normalized_comparison.png")

if __name__ == "__main__":
    main()
