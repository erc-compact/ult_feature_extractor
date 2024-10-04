import argparse
import time, sys
import numpy as np
import configparser
from normalization_functions import Norm
from PFDFile import PFD
import presto.prepfold as pp
import matplotlib.pyplot as plt
import os
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
        print(result.shape)
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
        #result = self.pfd_instance.plot_subbands()
        #result=self.pfd_contents.plot_subbands(device='/NULL')
        self.pfd_contents.dedisperse()
        result = self.pfd_contents.profs.sum(0)
        print(result.shape)
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


    def custom_normalize(self, data, method='standard',intensity=False):
        """
        Normalizes the data based on the selected method:
        'minmax', 'standard', 'l1', or 'l2' normalization.
        """

        norm_instance = Norm(data)

        if method == 'minmax':
            normalized_data =  norm_instance.min_max_scaler()
        elif method == 'standard':
            normalized_data = norm_instance.standard_scaler()
        elif method=='pics':
            normalized_data=normalize(data)
        elif method == 'l1':
            normalized_data = norm_instance.l1_normalize()
        elif method == 'l2':
            normalized_data = norm_instance.l2_normalize()
            
        elif method=='maxNorm':
            # print(data.shape)
            # sys.exit()
            #data=data.flatten()
            normalized_data= data/np.max(data)
        elif method=='meanSub':
            if intensity==True:
                data=data.flatten()
                print('yes')
                print('check1',data.shape)
                normalized_data= (data-np.mean(data))/np.std(data)
            else:
                mean = np.mean(data, axis=1, keepdims=True)
                print('check2')
                std = np.std(data, axis=1, keepdims=True)
                normalized_data= (data - mean) / std
        else:
            raise ValueError("Unsupported normalization method")
        return normalized_data

        

    def down(self, data, bins):
        down_data= downsample(data,bins)
        return down_data

    def data_saver(self, data, data_feature_name, bins, normalize_method='standard', save_npy=False, save_plots=False, save_directory='.',intensity1=False):
        # Create the directory if it doesn't exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Construct the full file paths for saving
        def get_file_path(filename):
            return os.path.join(save_directory, filename)
        
        # Save original data as .npy if save_npy is True
        if save_npy:
            np.save(get_file_path(f'{data_feature_name}.npy'), data)
        
        # Downsample the data
        down_data = self.down(data, bins)
        
        # Save downsampled data as .npy if save_npy is True
        if save_npy:
            np.save(get_file_path(f'{data_feature_name}_down.npy'), down_data)
        
        # Normalize the downsampled data
        norm = self.custom_normalize(down_data, method=normalize_method,intensity=intensity1)
        
        # Save normalized downsampled data as .npy if save_npy is True
        if save_npy:
            np.save(get_file_path(f'{data_feature_name}_down_norm.npy'), norm)

        
        if save_plots:
            # Plot the original data
            plt.figure()
            if data.ndim == 1:
                plt.plot(data)
            else:
                img = plt.imshow(data, cmap='viridis')
                plt.colorbar(img)
            plt.title(f'Original Data - {data_feature_name}')
            plt.savefig(get_file_path(f'{data_feature_name}_original.png'))
            plt.close()

            # Plot the downsampled data
            plt.figure()
            if down_data.ndim == 1:
                plt.plot(down_data)
            else:
                img = plt.imshow(down_data, cmap='viridis')
                plt.colorbar(img)
            plt.title(f'Downsampled Data - {data_feature_name}')
            plt.savefig(get_file_path(f'{data_feature_name}_downsampled.png'))
            plt.close()

            # Plot the normalized downsampled data
            plt.figure()
            if norm.ndim == 1:
                plt.plot(norm)
            else:
                img = plt.imshow(norm, cmap='viridis')
                plt.colorbar(img)
            plt.title(f'Normalized Downsampled Data - {data_feature_name}')
            plt.savefig(get_file_path(f'{data_feature_name}_downsampled_normalized.png'))
            plt.close()

    def executer(self):
        #self.clock = True
        if self.features_to_extract.getboolean('time_vs_phase'):
            time_phase, time_taken = self.get_time_vs_phase()
            print('TP',time_phase.shape)
            self.data_saver(time_phase,'PALFA2_P_time_phase_new_meanSub', 48, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/test2', save_npy=False)
          
            #print(f'time vs phase: {time_phase} (Time taken: {time_taken:.4f} seconds)')

        if self.features_to_extract.getboolean('dedispersed_profile'):
            summed_profile, time_taken = self.get_dedispersed_profile()
            #print(f'dedispersed and summed profile: {summed_profile} (Time taken: {time_taken:.4f} seconds)')
            print('intensity',summed_profile.shape)
            self.data_saver(summed_profile,'PALFA2_P_intensity_new_meanSub', 64, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/test2', save_npy=False,intensity1=True)

        if self.features_to_extract.getboolean('freq_vs_phase'):
            freq_vs_phase, time_taken = self.get_freq_vs_phase()
            #print(f'freq vs phase: {freq_vs_phase} (Time taken: {time_taken:.4f} seconds)')
            print('FP',freq_vs_phase.shape)
            self.data_saver(freq_vs_phase,'PALFA2_P_freq_phase_new_meanSub', 64, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/test2', save_npy=False)

        if self.features_to_extract.getboolean('dm_curve'): #need function for time
            lodm = self.pfd_contents.dms[0]
            hidm = self.pfd_contents.dms[-1]
            (chis, DMs) = self.pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=100,device='dev.png/png')
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