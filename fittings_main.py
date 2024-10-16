import presto.prepfold as pp
import argparse
import numpy as np
from PFDFile import PFD
from PFDFeatureExtractor import PFDFeatureExtractor
from FeatureExtractor import FeatureExtractor
import configparser
import time, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erf


# Function to load configuration
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['FIT_FEATURES']


# Function to get dedispersed and summed profile
def get_dedispersed_profile(pfd_contents):
    #start_time = time.time()
    pfd_contents.dedisperse()
    result = pfd_contents.sumprof
    #elapsed_time = time.time() - start_time
    return result


# def sine_function(x, amplitude, frequency, phase, offset):
#     return amplitude * np.sin(frequency * x + phase) + offset

# def plot_summed_profile(sumprof):
#     # Obtain the length of the summed profile
#     length_of_sumprof = len(sumprof)
#     print("Length of summed profile: {:>10}".format(length_of_sumprof))

#     data = sumprof.flatten()
#     normalized_data = (data - np.mean(data)) / np.std(data)

#     # Define x values for fitting (same as the indices of normalized data)
#     x_data = np.arange(len(normalized_data))

#     # Initial guess for sine function parameters: [amplitude, frequency, phase, offset]
#     offset = abs(max(data) - min(data)) / 2
#     initial_guess = [1, 0, 0, offset]

#     # Fit the sine function to the normalized data
#     params, covariance = curve_fit(sine_function, x_data, normalized_data, p0=initial_guess)

#     # Generate fitted sine data using the fitted parameters
#     y_fit = sine_function(x_data, *params)

#     # Plot the normalized data and the fitted sine curve
#     plt.figure(figsize=(10, 6))
#     plt.plot(normalized_data, label='Normalized Data', color='blue')
#     plt.plot(y_fit, label='Fitted Sine Curve', color='red', linewidth=2)
#     plt.title('Dedispersed Summed Profile with Sine Fit')
#     plt.xlabel('Bin')
#     plt.ylabel('Intensity')
#     plt.legend()

#     # Save the plot to a file
#     plt.savefig('/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/fits_test/summed_profile_plot6.png')
#     print("Plot saved as 'summed_profile_plot.png'")



# Define the sine function to be used for fitting
def sine_function(x, A, omega, phi, C):
    return A * np.sin(omega * x + phi) + C

def plot_summed_profile(sumprof):
    # Obtain the length of the summed profile
    length_of_sumprof = len(sumprof)
    print(f"Length of summed profile: {length_of_sumprof:>10}")

    # Flatten the summed profile in case it is multidimensional
    data = sumprof.flatten()

    # Normalize the data
    normalized_data = (data - np.mean(data)) / np.std(data)

    # Define x values for fitting (corresponds to the indices of normalized data)
    x_data = np.arange(len(normalized_data))

    # Initial guess for sine function parameters: [amplitude, frequency, phase, offset]
    amplitude_guess = (max(data) - min(data)) / 2
    offset_guess = np.mean(data)
    initial_guess = [amplitude_guess, 2 * np.pi / length_of_sumprof, 0, offset_guess]

    # Fit the sine function to the normalized data
    params, covariance = curve_fit(sine_function, x_data, normalized_data, p0=initial_guess)

    # Generate fitted sine data using the fitted parameters
    y_fit = sine_function(x_data, *params)

    # Calculate chi-square
    chi_square = np.sum(((normalized_data - y_fit) ** 2))  # No errors provided, assuming equal weighting
    print(f"Chi-square value: {chi_square:.2f}")

    # Plot the normalized data and the fitted sine curve
    plt.figure(figsize=(10, 6))
    plt.plot(normalized_data, label='Normalized Data', color='blue')
    plt.plot(y_fit, label=f'Fitted Sine Curve (Chi² = {chi_square:.2f})', color='red', linewidth=2)
    plt.title('Dedispersed Summed Profile with Sine Fit')
    plt.xlabel('Bin')
    plt.ylabel('Intensity')
    plt.legend()

    # Save the plot to a file
    plot_path = '/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/fits_test/summed_profile_plot_new8.png'
    plt.savefig(plot_path)
    print(f"Plot saved as '{plot_path}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="pfd file", required=True)
    #parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()
    debugFlag=True
    FE=FeatureExtractor(debugFlag)

    test_pfd = args.file
    #config_file = args.config

    # Load configuration
    #features_to_extract = load_config(config_file)
    
    # Load PFD file based on presto
    pfd_contents = pp.pfd(test_pfd)
    summed_profile = get_dedispersed_profile(pfd_contents)
    plot_summed_profile(summed_profile)
    #chisq_sine, chisq_sine_sqr, maxima_diff, sum_residuals = FE.getSinusoidFittings(summed_profile)
    # print("{:<30} {:>10.4f}".format("Chi-Squared Sine Fit:", chisq_sine))
    # print("{:<30} {:>10.4f}".format("Chi-Squared Sine-Squared Fit:", chisq_sine_sqr))
    # print("{:<30} {:>10.4f}".format("Difference between maxima:", maxima_diff))
    # print("{:<30} {:>10.4f}".format("Sum over residuals:", sum_residuals))


if __name__ == "__main__":
    main()