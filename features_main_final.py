import argparse
import time, sys
import numpy as np
import configparser
from normalization_functions import Norm
from PFDFile import PFD
import presto.prepfold as pp
import matplotlib.pyplot as plt
from PFDFeatureExtractor import PFDFeatureExtractor
from samples import normalize, downsample
#from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

# Decorator to time functions
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time            
        return result, elapsed_time
    return wrapper
        
        # if self.clock: # devika update this later
        #     
        # else:
        #     return func

class UFE:
    def __init__(self, debug=False, clock=False):
        self.debug = debug
        self.clock = clock

        self.arg_parser()

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", help="pfd file", required=True)
        parser.add_argument("--config", help="config file", required=True)
        args = parser.parse_args()

        self.test_pfd = args.file
        self.config_file = args.config

        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.features_to_extract = config['FEATURES']
        self.pfd_contents = pp.pfd(self.test_pfd) # presto
        self.pfd_instance = PFD(False, self.test_pfd) # pfl
 
        
    @time_it
    def get_time_vs_phase(self):
        result = self.pfd_contents.time_vs_phase()
        #np.save('timephase_PALFANP.npy',result)
        return result

    # Function to get dedispersed and summed profile #intensity profile
    @time_it
    def get_dedispersed_profile(self):
        self.pfd_contents.dedisperse()
        result = self.pfd_contents.sumprof
        #np.save('intensity_PALFANP.npy',result)
        return result
    
    # Function to get frequency vs phase
    @time_it
    def get_freq_vs_phase(self):
        result = self.pfd_instance.plot_subbands()
        #np.save('freqphase_PALFANP.npy',result)
        return result
    
    # Function to get chi2 vs DM curve
    @time_it
    def get_dm_curve(self):
        result = self.pfd_instance.DM_curve_Data()
        #np.save('dmcurve_PALFANP.npy',result)
        return result
    
    # Type 6 Feature Functions
    def compute_type_6_if_not_done(self):  #checks if the Lyon features have already been computed
        if 'MEAN_IFP' not in self.pfd_instance.features1: #can be any feature instead of MEAN_IFP as an indicator
            self.pfd_instance.computeType_6()

    @time_it
    def get_mean_ifp(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['MEAN_IFP']
        return result

    @time_it
    def get_std_ifp(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['STD_IFP']
        return result

    @time_it
    def get_skw_ifp(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['SKW_IFP']
        return result

    @time_it
    def get_kurt_ifp(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['KURT_IFP']
        return result
    
    @time_it
    def get_mean_dm(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['MEAN_DM']
        return result

    @time_it
    def get_std_dm(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['STD_DM']
        return result

    @time_it
    def get_skw_dm(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['SKW_DM']
        return result

    @time_it
    def get_kurt_dm(self):
        self.compute_type_6_if_not_done()
        result = self.pfd_instance.features1['KURT_DM']
        return result


    def custom_normalize(self, data, method='standard', norm_type='l2'):
        """
        Normalizes the data based on the selected method:
        'minmax', 'standard', 'l1', or 'l2' normalization.
        """
        is_1D = False
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            is_1D = True

        norm_instance = Norm(data)

        if method == 'minmax':
            normalized_data =  norm_instance.min_max_scaler()
        elif method == 'standard':
            normalized_data = norm_instance.standard_scaler()
        elif method == 'l1':
            normalized_data = norm_instance.l1_normalize()
        elif method == 'l2':
            normalized_data = norm_instance.l2_normalize()
        else:
            raise ValueError("Unsupported normalization method. Choose 'minmax', 'standard', 'l1', or 'l2'.")

        if is_1D:
            normalized_data = normalized_data.flatten()

        return normalized_data

    def down(self, data, bins):
        down_data= downsample(data,bins)
        return down_data
        

    def data_saver(self, data, data_feature_name, bins, normalize_method='standard', save_plots=False):
        # Save original data
        np.save(f'{data_feature_name}.npy', data)
        
        # Downsample the data
        down_data = self.down(data, bins)
        np.save(f'{data_feature_name}_down.npy', down_data)
        
        # Normalize the downsampled data
        norm = self.custom_normalize(down_data, method=normalize_method)
        np.save(f'{data_feature_name}_down_norm.npy', norm)

        # If save_plots is True, save the plots
        if save_plots:
            # Plot the original data
            plt.figure()
            #plt.plot(data)
            plt.imshow(data)
            plt.title(f'Original Data - {data_feature_name}')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            plt.savefig(f'{data_feature_name}_original.png')
            plt.close()

            # Plot the downsampled data
            plt.figure()
            plt.imshow(down_data)
            plt.title(f'Downsampled Data - {data_feature_name}')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            plt.savefig(f'{data_feature_name}_downsampled.png')
            plt.close()

            # Plot the normalized downsampled data
            plt.figure()
            plt.imshow(norm)
            plt.title(f'Normalized Downsampled Data - {data_feature_name}')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            plt.savefig(f'{data_feature_name}_downsampled_normalized.png')
            plt.close()

    def executer(self):
        #self.clock = True
        if self.features_to_extract.getboolean('time_vs_phase'):
            time_phase, time_taken = self.get_time_vs_phase()
            print(time_phase.shape)
            self.data_saver(time_phase,'FAST_time_phase', 64, save_plots=True)
          
            print(f'time vs phase: {time_phase} (Time taken: {time_taken:.4f} seconds)')

        if self.features_to_extract.getboolean('dedispersed_profile'):
            summed_profile, time_taken = self.get_dedispersed_profile()
            print(f'dedispersed and summed profile: {summed_profile} (Time taken: {time_taken:.4f} seconds)')

        if self.features_to_extract.getboolean('freq_vs_phase'):
            freq_vs_phase, time_taken = self.get_freq_vs_phase()
            print(f'freq vs phase: {freq_vs_phase} (Time taken: {time_taken:.4f} seconds)')

        if self.features_to_extract.getboolean('dm_curve'): #need function for time
            lodm = self.pfd_contents.dms[0]
            hidm = self.pfd_contents.dms[-1]
            (chis, DMs) = self.pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=100,device='dev.ps/ps')
            #print(chis, DMs)
            #print(f'chis,: {freq_vs_phase} (Time taken: {time_taken:.4f} seconds)')

        # Type 6 Features
        if self.features_to_extract.getboolean('mean_ifp'):
            mean_ifp, time_taken = self.get_mean_ifp()
            print(f'Mean_IFP: {mean_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('std_ifp'):
            std_ifp, time_taken = self.get_std_ifp()
            print(f'STD_IFP: {std_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('skw_ifp'):
            skw_ifp, time_taken = self.get_skw_ifp() # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
            print(f'SKW_IFP: {skw_ifp} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('kurt_ifp'):
            kurt_ifp, time_taken = self.get_kurt_ifp()
            print(f'KURT_IFP: {kurt_ifp} (Time taken: {time_taken * 1000:.4f} ms)')
 # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
        if self.features_to_extract.getboolean('mean_dm'):
            mean_dm, time_taken = self.get_mean_dm()
            print(f'Mean_DM: {mean_dm} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('std_dm'):
            std_dm, time_taken = self.get_std_dm() # print('dm curve array', type(dm_curve_array))
        # print('dm curve type', type(dm_curve))
            print(f'STD_DM: {std_dm} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('skw_dm'):
            skw_dm, time_taken = self.get_skw_dm()
            print(f'SKW_DM: {skw_dm} (Time taken: {time_taken * 1000:.4f} ms)')

        if self.features_to_extract.getboolean('kurt_dm'):
            kurt_dm, time_taken = self.get_kurt_dm()
            print(f'KURT_DM: {kurt_dm} (Time taken: {time_taken * 1000:.4f} ms)')

       #self.clock = False


if __name__ == "__main__":
    myufe = UFE(debug=True, clock=True)
   
    myufe.executer()