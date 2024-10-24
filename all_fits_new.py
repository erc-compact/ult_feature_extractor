import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import presto.prepfold as pp
import argparse
import numpy as np
from PFDFile import PFD
from PFDFeatureExtractor import PFDFeatureExtractor
from FeatureExtractor import FeatureExtractor
import configparser
from samples import normalize, downsample
import time, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erf
import itertools
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
import os

def down(data, bins):
    down_data= downsample(data,bins)
    return down_data


# Define sine function
def sine_function(x, A, phi, C):
    omega=2*np.pi*(1/64)
    return A * np.sin(omega * x + phi) + C

# Define sine squared function
def sine_squared_function(x, A, phi, C):
    omega=2*np.pi*(1/64)
    return A * (np.sin(omega * x + phi))**2 + C

# Define Gaussian function
def gaussian_function(x, A, mu, sigma, C):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

# Define double Gaussian function
def double_gaussian_function(x, A1, mu1, sigma1, A2, mu2, sigma2, C):
    gaussian1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    gaussian2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return gaussian1 + gaussian2 + C

# Function to plot and fit for a given data and function, with optional suffix and directory
def fit_and_plot(x_data, y_data, fit_function, initial_guess, plot_title, base_filename, suffix='', directory=''):
    # Fit the data using the provided fit function and initial guess
    params, covariance = curve_fit(fit_function, x_data, y_data, p0=initial_guess, maxfev=30000)
    # print("Final parameters:")
    # print(f"Amplitude (A): {params[0]}")
    # print(f"Angular Frequency (omega): {params[1]}")
    # print(f"Phase (phi): {params[2]}")
    # print(f"Offset (C): {params[3]}")
    # Generate fitted data
    y_fit = fit_function(x_data, *params)
    

    # Calculate chi-square
    chi_square = np.sum(((y_data - y_fit) ** 2))  # Assuming equal weights (no error provided)
    #return chi_square
    print(f"Chi-square value for {plot_title}: {chi_square:.2f}")

    # Plot the data and the fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(y_data, label='Data', color='blue')
    plt.plot(y_fit, label=f'Fitted Curve (Chi² = {chi_square:.2f})', color='red', linewidth=2)
    plt.title(f'{plot_title} {suffix}')
    plt.xlabel('Bin')
    plt.ylabel('Intensity')
    plt.legend()

    # Construct the save path with directory, filename, and suffix
    if directory and not os.path.exists(directory):
        os.makedirs(directory)  # Create directory if it doesn't exist
    save_path = os.path.join(directory, f'{base_filename}_{suffix}.png' if suffix else f'{base_filename}.png')

    # Save the plot to the file
    plt.savefig(save_path)
    print(f"Plot saved as '{save_path}'")

def print_exe(output):
    os.system("echo " + str(output))

# Sine fit
def sine_fit(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude, frequency, phase, offset]
    amplitude_guess = (max(data) - min(data)) / 2
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess, 0, offset_guess]

    # Call the fitting and plotting function
    fit_and_plot(x_data, normalized_data, sine_function, initial_guess, 'Sine Fit', 'sine_fit_plot', suffix, directory)

# Sine-squared fit
def sine_squared_fit(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude, frequency, phase, offset]
    amplitude_guess = (max(data) - min(data)) / 2
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess, 0, offset_guess]


    # Call the fitting and plotting function
    fit_and_plot(x_data, normalized_data, sine_squared_function, initial_guess,'Sine Squared Fit', 'sine_squared_fit_plot', suffix, directory)
                

# Gaussian fit
def gaussian_fit(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude, mean, standard deviation, offset]
    amplitude_guess = max(data) - min(data)
    mean_guess = np.mean(x_data)
    sigma_guess = np.std(x_data)
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess, mean_guess, sigma_guess, offset_guess]

    # Call the fitting and plotting function
    fit_and_plot(x_data, normalized_data, gaussian_function, initial_guess,'Gaussian Fit', 'gaussian_fit_plot', suffix, directory)

# Double Gaussian fit
def double_gaussian_fit(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude1, mean1, sigma1, amplitude2, mean2, sigma2, offset]
    amplitude_guess1 = max(data) - min(data)
    amplitude_guess2 = amplitude_guess1 / 2  # Second Gaussian with a smaller amplitude
    mean_guess1 = np.mean(x_data)
    mean_guess2 = mean_guess1 + len(x_data) / 4  # Offset the second Gaussian peak
    sigma_guess1 = np.std(x_data)
    sigma_guess2 = sigma_guess1 / 2  # Narrower second Gaussian
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess1, mean_guess1, sigma_guess1, amplitude_guess2, mean_guess2, sigma_guess2, offset_guess]

    # Call the fitting and plotting function
    fit_and_plot(x_data, normalized_data, double_gaussian_function, initial_guess,'Double Gaussian Fit', 'double_gaussian_fit_plot', suffix, directory)
# Function to get dedispersed and summed profile
def get_dedispersed_profile(pfd_contents):
    #start_time = time.time()
    pfd_contents.dedisperse()
    result = pfd_contents.sumprof
    result=downsample(result,64)
    #elapsed_time = time.time() - start_time
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="pfd file", required=True)
    parser.add_argument("--tag", help="Name for folder and prefix", required=True)  # New argument
    #parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    debugFlag=True
    FE=FeatureExtractor(debugFlag)

    test_pfd = args.file
    tag = args.tag 
    #config_file = args.config

    # Load configuration
    #features_to_extract = load_config(config_file)
    
    # Load PFD file based on presto
    pfd_contents = pp.pfd(test_pfd)
    summed_profile = get_dedispersed_profile(pfd_contents)
    #plot_summed_profile(summed_profile)
    #sine_fit(summed_profile, tag, directory=f'/hercules/scratch/dbhatnagar/Classifier_data_samples/chi_square_15Oct/{tag}')
    sine_fit(summed_profile, tag, directory=f'/hercules/scratch/dbhatnagar/Classifier_data_samples/22Oct_fixedF/clustering_pfds/fit_pics/{tag}')
    sine_squared_fit(summed_profile, tag, directory=f'/hercules/scratch/dbhatnagar/Classifier_data_samples/22Oct_fixedF/clustering_pfds/fit_pics/{tag}')
    gaussian_fit(summed_profile, tag, directory=f'/hercules/scratch/dbhatnagar/Classifier_data_samples/22Oct_fixedF/clustering_pfds/fit_pics/{tag}')
    double_gaussian_fit(summed_profile, tag, directory=f'/hercules/scratch/dbhatnagar/Classifier_data_samples/22Oct_fixedF/clustering_pfds/fit_pics/{tag}')
    # sine_squared_fit(sumprof)
    # gaussian_fit(sumprof)
    # double_gaussian_fit(sumprof)

if __name__ == "__main__":
    main()