import presto.prepfold as pp
import argparse
import numpy as np
from PFDFile import PFD
from PFDFeatureExtractor import PFDFeatureExtractor
import configparser
import time, sys
#import matplotlib
#matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
#Tk
#matplotlib.use('PyQt6')

# Function to load configuration
def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config['FEATURES']

# Function to get time vs phase
def get_time_vs_phase(pfd_contents):
    start_time = time.time()
    result = pfd_contents.time_vs_phase()
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Function to get dedispersed and summed profile
def get_dedispersed_profile(pfd_contents):
    start_time = time.time()
    pfd_contents.dedisperse()
    result = pfd_contents.sumprof
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Function to get frequency vs phase
def get_freq_vs_phase(pfd_instance):
    start_time = time.time()
    result = pfd_instance.plot_subbands()
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Function to get chi2 vs DM curve
def get_dm_curve(pfd_instance):
    start_time = time.time()
    result = pfd_instance.DM_curve_Data()
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Type 6 Feature Functions
def compute_type_6_if_not_done(pfd_instance):  #checks if the Lyon features have already been computed
    if 'MEAN_IFP' not in pfd_instance.features1: #can be any feature instead of MEAN_IFP as an indicator
        pfd_instance.computeType_6()

def get_mean_ifp(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['MEAN_IFP']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_std_ifp(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['STD_IFP']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_skw_ifp(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['SKW_IFP']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_kurt_ifp(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['KURT_IFP']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_mean_dm(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['MEAN_DM']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_std_dm(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['STD_DM']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_skw_dm(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['SKW_DM']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

def get_kurt_dm(pfd_instance):
    compute_type_6_if_not_done(pfd_instance)
    start_time = time.time()
    result = pfd_instance.features1['KURT_DM']
    elapsed_time = time.time() - start_time
    return result, elapsed_time

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="pfd file", required=True)
    parser.add_argument("--config", help="config file", required=True)
    args = parser.parse_args()

    test_pfd = args.file
    config_file = args.config

    # Load configuration
    features_to_extract = load_config(config_file)
    
    # Load PFD file based on presto
    pfd_contents = pp.pfd(test_pfd)
    
    # PFDFile instance based on Pulsar Features Lab
    pfd_instance = PFD(False, test_pfd)

    # Extract and print features as per config
    if features_to_extract.getboolean('time_vs_phase'):
        time_phase, time_taken = get_time_vs_phase(pfd_contents)
        print(f'time vs phase: {time_phase} (Time taken: {time_taken:.4f} seconds)')

    if features_to_extract.getboolean('dedispersed_profile'):
        summed_profile, time_taken = get_dedispersed_profile(pfd_contents)
        print(f'dedispersed and summed profile: {summed_profile} (Time taken: {time_taken:.4f} seconds)')

    if features_to_extract.getboolean('freq_vs_phase'):
        freq_vs_phase, time_taken = get_freq_vs_phase(pfd_instance)
        print(f'freq vs phase: {freq_vs_phase} (Time taken: {time_taken:.4f} seconds)')

    if features_to_extract.getboolean('dm_curve'):
        #dm_curve, time_taken = get_dm_curve(pfd_instance)
        #data = pfd_instance.plot_chi2_vs_DM(loDM=0, hiDM=40)
        lodm = pfd_contents.dms[0]
        hidm = pfd_contents.dms[-1]
        (chis, DMs) = pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=100,device='/PS')

# Optionally, you can print the output
        print("Chi^2 values:", chis)
        print("DM values:", DMs)

        
        
        # plt.plot(data[1], data[0])
        # plt.savefig('devika_check.png')
        #print(f'DM curve: {dm_curve} (Time taken: {time_taken:.4f} seconds)')

        # x=dm_curve[0]
        # y=dm_curve[1]


        #x = x - min(x)
        # print(delta_x, y)
        #plt.plot(y, x)
        # print(np.shape(x), np.shape(y))
        #plt.savefig('devika.png')
        # plt.show()
        # print(x)
        # print(y)
        # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
        #print(dm_curve)
        sys.exit()
        #print(dm_curve_array)
        #plt.plot(dm_curve_array[:,0],dm_curve_array[:,1])
        #plt.plot(dm_curve_array,y)
        #plt.plot(dm_curve[:,0],dm_curve[:,1])

    # Type 6 Features
    if features_to_extract.getboolean('mean_ifp'):
        mean_ifp, time_taken = get_mean_ifp(pfd_instance)
        print(f'Mean_IFP: {mean_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('std_ifp'):
        std_ifp, time_taken = get_std_ifp(pfd_instance)
        print(f'STD_IFP: {std_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('skw_ifp'):
        skw_ifp, time_taken = get_skw_ifp(pfd_instance) # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
        print(f'SKW_IFP: {skw_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('kurt_ifp'):
        kurt_ifp, time_taken = get_kurt_ifp(pfd_instance)
        print(f'KURT_IFP: {kurt_ifp} (Time taken: {time_taken * 1000:.4f} ms)')
 # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
    if features_to_extract.getboolean('mean_dm'):
        mean_dm, time_taken = get_mean_dm(pfd_instance)
        print(f'Mean_DM: {mean_dm} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('std_dm'):
        std_dm, time_taken = get_std_dm(pfd_instance) # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
        print(f'STD_DM: {std_dm} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('skw_dm'):
        skw_dm, time_taken = get_skw_dm(pfd_instance)
        print(f'SKW_DM: {skw_dm} (Time taken: {time_taken * 1000:.4f} ms)')

    if features_to_extract.getboolean('kurt_dm'):
        kurt_dm, time_taken = get_kurt_dm(pfd_instance)
        print(f'KURT_DM: {kurt_dm} (Time taken: {time_taken * 1000:.4f} ms)')

if __name__ == "__main__":
    main()
