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
from samples import normalize, downsample
import configparser
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
from scipy.stats import kurtosis, skew

def down(data, bins):
    down_data= downsample(data,bins)
    return down_data

 
def one_file(files, cpu, directory='.', tag='yes'):
    df = pd.DataFrame(columns=['file', 'chi2_sine', 'chi2_sine_square', 'chi2_gauss', 'chi2_dble','sigma_val', 'sigma1_val', 'sigma2_val','mean_IP','std_IP','skew_IP','kurt_IP','mean_DM','std_DM','skew_DM','kurt_DM'])
    
    for i, file in enumerate(files):
        print_exe(f'{i}/{len(files)} x 48') if cpu == 0 else None
        try:
            pfd_contents = pp.pfd(file)
            summed_profile = get_dedispersed_profile(pfd_contents)
            lodm = pfd_contents.dms[0]
            hidm = pfd_contents.dms[-1]
            chis,DMs = pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=64)
            DOF=float(pfd_contents.DOFcor)
            DOF=63 * pfd_contents.DOF_corr()
            print('chis shape: ', chis.shape)
            #print(DOF)

            # Perform fits using the provided tag and directory
            chi2_sine = sine_fit1(summed_profile, tag, directory=directory)
            chi2_sine_square = sine_squared_fit(summed_profile, tag, directory=directory)
            chi2_gauss, sigma_val= gaussian_fit(summed_profile, tag, directory=directory)
            chi2_dble,sigma1_val, sigma2_val= double_gaussian_fit(summed_profile, tag, directory=directory)
            mean_IFP= calculate_mean(summed_profile)
            std_IFP=calculate_std(summed_profile)
            skew_IFP=calculate_skewness(summed_profile)
            kurt_IFP=calculate_excess_kurtosis(summed_profile)
            mean_DM=calculate_mean(chis)
            std_DM=calculate_std(chis)
            skew_DM=calculate_skewness(chis)
            kurt_DM=calculate_excess_kurtosis(chis)

        except Exception:
            # In case of error, set chi2 values to 0
            df.loc[i] = [file, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
        else:
            # Otherwise, record the chi2 values for each file
            # print('DOF:', DOF)
            # print('gauss/DOF: ', chi2_gauss/DOF) 
            # print('gauss:', chi2_gauss)
            df.loc[i] = [file, chi2_sine/DOF, chi2_sine_square/DOF, chi2_gauss/DOF, chi2_dble/DOF, sigma_val, sigma1_val, sigma2_val, mean_IFP, std_IFP, skew_IFP,kurt_IFP, mean_DM, std_DM, skew_DM, kurt_DM]

    # Save the results to a CSV file
    df.to_csv(f'/tmp/dbhatnagar/temp_{cpu}.tmpcsv')

def reshape_into_cpu(files, ncpu):
    new_files = [[] for _ in range(ncpu)]
    for index, item in zip(itertools.cycle(range(ncpu)), files):
        new_files[index].append(item)
    return new_files

# Define sine function
def sine_function(x, A, omega, phi, C):
    return A * np.sin(omega * x + phi) + C

def sine_function_fixedF(x, A, phi, C):
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
def fit_and_plot(x_data, y_data, fit_function, initial_guess, plot_title, base_filename, suffix='', directory='',csv_maker=False):
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
    if csv_maker==True:
        return chi_square, params
    else:
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


def sine_fit1(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude, frequency, phase, offset]
    amplitude_guess = (max(data) - min(data)) / 2
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess, 0, offset_guess]

    #csv maker
    chi_square,params= fit_and_plot(x_data, normalized_data, sine_function_fixedF, initial_guess,'Sine Fit', 'sine_fit_plot', suffix, directory, csv_maker=True)
    return chi_square
    #fit_and_plot(x_data, normalized_data, sine_function_fixedF, initial_guess, 'Sine Fit fixed F', 'sine_fit_plot_fixedF', suffix, directory)

# Sine-squared fit
def sine_squared_fit(sumprof, suffix='', directory=''):
    # Prepare the data
    data = sumprof.flatten()
    normalized_data = (data - np.mean(data)) / np.std(data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for [amplitude, frequency, phase, offset]
    amplitude_guess = (max(data) - min(data)) / 2
    offset_guess = np.mean(data)
    #initial_guess = [amplitude_guess, 2 * np.pi / len(sumprof), 0, offset_guess]
    initial_guess = [amplitude_guess, 0, offset_guess]


    #csv maker
    chi_square,params=fit_and_plot(x_data, normalized_data, sine_squared_function, initial_guess,'Sine Squared Fit', 'sine_squared_fit_plot', suffix, directory,csv_maker=True)
    return chi_square

    # fit_and_plot(x_data, normalized_data, sine_squared_function, initial_guess,'Sine Squared Fit', 'sine_squared_fit_plot', suffix, directory)
                

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

    
    # Fit and return chi-square and fitted parameters (including sigma)
    chi_square, params = fit_and_plot(x_data, normalized_data, gaussian_function, initial_guess, 'Gaussian Fit', 'gaussian_fit_plot', suffix, directory, csv_maker=True)

    # Extract the sigma (third parameter) from the fitted params
    fitted_sigma = params[2]

    return chi_square, fitted_sigma

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

    # Fit and return chi-square and fitted parameters (including sigmas)
    chi_square, params = fit_and_plot(x_data, normalized_data, double_gaussian_function, initial_guess, 'Double Gaussian Fit', 'double_gaussian_fit_plot', suffix, directory, csv_maker=True)

    # Extract the sigmas (third and sixth parameters) from the fitted params
    fitted_sigma1 = params[2]
    fitted_sigma2 = params[5]

    return chi_square, fitted_sigma1, fitted_sigma2

# Function to get dedispersed and summed profile
def get_dedispersed_profile(pfd_contents):
    #start_time = time.time()
    pfd_contents.dedisperse()
    result = pfd_contents.sumprof
    #print(result.shape)
    #print(downsample(result,64).shape)
    result=downsample(result,64)
    #elapsed_time = time.time() - start_time
    return result

def calculate_excess_kurtosis(arr):
    """
    Calculate the excess kurtosis of a 1D numpy array.
    Excess kurtosis is calculated as kurtosis - 3.
    """
    return kurtosis(arr, fisher=True)

def calculate_mean(arr):
    """
    Calculate the mean of a 1D numpy array.
    """
    return np.mean(arr)

def calculate_std(arr):
    """
    Calculate the standard deviation of a 1D numpy array.
    """
    return np.std(arr, ddof=0)  # population std deviation; for sample std use ddof=1

def calculate_skewness(arr):
    """
    Calculate the skewness of a 1D numpy array.
    """
    return skew(arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="pfd directory", required=True)
    parser.add_argument("--tag", help="Tag for fits")  # Added tag argument
    parser.add_argument("--output_dir", help="Directory to save results", default="/tmp/dbhatnagar")  # Added output directory argument
    args = parser.parse_args()

    directory = Path(args.dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    files = [str(file) for file in directory.glob("*.pfd")]

    ncpus = 48
    files_cpu_split = reshape_into_cpu(files, ncpus)
    data = [(files_cpu_split[i], i, output_dir, args.tag) for i in range(ncpus)]  # Passing directory and tag to one_file

    # Use multiprocessing to process files in parallel
    with Pool(ncpus) as p:
        p.starmap(one_file, data)

    # Combine the temporary CSV files from all CPUs into a single DataFrame
    
    final_df = pd.DataFrame(columns=['file', 'chi2_sine', 'chi2_sine_square', 'chi2_gauss', 'chi2_dble','sigma_val', 'sigma1_val', 'sigma2_val','mean_IP','std_IP','skew_IP','kurt_IP','mean_DM','std_DM','skew_DM','kurt_DM'])
    
    for cpu in range(ncpus):
        df_tmp = pd.read_csv(output_dir / f'temp_{cpu}.tmpcsv', index_col=0)  # Read from output_dir
        final_df = pd.concat([final_df, df_tmp], ignore_index=True)

    # Save the final combined DataFrame as a CSV file in the output directory
    final_df.to_csv(output_dir / f'{directory.name}.csv')

if __name__ == "__main__":
    main()