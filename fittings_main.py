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


def plot_summed_profile(sumprof):
    sumprof=sumprof
    # Obtain the length of the summed profile
    length_of_sumprof = len(sumprof)
    #print(f"Length of summed profile: {length_of_sumprof}")
    print("Length of summed profile: {:>10}".format(length_of_sumprof))

    # Plot the summed profile
    plt.figure(figsize=(10, 6))
    plt.plot(sumprof)
    plt.title('Dedispersed Summed Profile')
    plt.xlabel('Bin')
    plt.ylabel('Intensity')

    # Save the plot to a file
    plt.savefig('summed_profile_plot.png')
    print("Plot saved as 'summed_profile_plot.png'")

    # Show the plot (optional)
    #plt.show()


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
    chisq_sine, chisq_sine_sqr, maxima_diff, sum_residuals = FE.getSinusoidFittings(summed_profile)
    print("{:<30} {:>10.4f}".format("Chi-Squared Sine Fit:", chisq_sine))
    print("{:<30} {:>10.4f}".format("Chi-Squared Sine-Squared Fit:", chisq_sine_sqr))
    print("{:<30} {:>10.4f}".format("Difference between maxima:", maxima_diff))
    print("{:<30} {:>10.4f}".format("Sum over residuals:", sum_residuals))


if __name__ == "__main__":
    main()