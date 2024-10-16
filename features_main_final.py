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
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

# Decorator to time functions
def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time            
        return result, elapsed_time
    return wrapper

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
        #result = self.pfd_contents.time_vs_phase(interp=1)
        #self.pfd_contents.plot_intervals(device='time_phase_test.png/png')
        #print(result.shape)
        #np.save('timephase_PALFANP.npy',result)
        #return result
        self.pfd_contents.dedisperse()
        result = self.pfd_contents.profs.sum(1)
        # print(result.shape)
        #np.save('freqphase_PALFANP.npy',result)
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
    def get_dm_curve(self,bins):

        #result = self.pfd_instance.DM_curve_Data()
        #np.save('dmcurve_PALFANP.npy',result)
        lodm = self.pfd_contents.dms[0]
        hidm = self.pfd_contents.dms[-1]
        (chis,DMs) = self.pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=bins)
        result=(chis,DMs)
        print(type(result))
        print(type(result[0]))
        print(type(result[1]))
        #result = (chis, DMs)
        #print(result)

        return chis,DMs
    
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
                #print('check1',data.shape)
                normalized_data= (data-np.median(data))/np.std(data)

            else:
                print('data mean',np.mean(data.flatten()))
                print('data median',np.median(data.flatten()))
                mean = np.median(data, axis=1, keepdims=True)
                print('check2')
                std = np.std(data, axis=1, keepdims=True)
                # mean=np.mean(data.flatten())
                # std=np.std(data.flatten())
                normalized_data= (data - mean) / std
                print('normalized data mean',np.mean(normalized_data))
                print('normalized data median',np.median(normalized_data))
        elif method=='none':
            normalized_data=data
        else:
            raise ValueError("Unsupported normalization method")
        return normalized_data

        

    def down(self, data, bins):
        down_data= downsample(data,bins)
        return down_data

    def data_saver(self, data, data_feature_name, bins, normalize_method='standard', save_npy=False, save_plots=True, save_directory='.',intensity1=False):
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
                plt.plot(data,'black')
            else:
                img = plt.imshow(data, cmap='viridis')
                plt.colorbar(img, shrink=0.3)
            plt.title(f'Original Data - {data_feature_name}')
            #plt.ylabel('Intensity')
            plt.savefig(get_file_path(f'{data_feature_name}_original.png'))
            plt.close()

            # Plot the downsampled data
            plt.figure()
            if down_data.ndim == 1:
                plt.plot(down_data,'black')
            else:
                img = plt.imshow(down_data, cmap='viridis')
                plt.colorbar(img)
                # Use make_axes_locatable to create a new axis for the colorbar
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and padding

                #     # Add colorbar to the new axis
                # plt.colorbar(img, cax=cax)
            plt.title(f'Downsampled Data - {data_feature_name}')
            plt.savefig(get_file_path(f'{data_feature_name}_downsampled.png'))
            plt.close()

            # Plot the normalized downsampled data
            plt.figure()
            if norm.ndim == 1:
                plt.plot(norm,'black')
            else:
                img = plt.imshow(norm, cmap='viridis')
                plt.colorbar(img)
            plt.title(f'Normalized Downsampled Data - {data_feature_name}')
            plt.savefig(get_file_path(f'{data_feature_name}_downsampled_normalized.png'))
            plt.close()

    #def DM_plot_saver(self, chis, DMs, bins, save_npy=False, save_plots=False, save_directory='.'):


    def zero_DM(self, plot_zero_vals=False, plot_subtracted=False, save_dir='.', file_prefix='', subtract_first=False):
        # Calculate zero-DM dedispersed values
        self.pfd_contents.dedisperse(DM=0)
        intensity_zero_OG = self.pfd_contents.sumprof
        intensity_zero=self.custom_normalize(self.down(intensity_zero_OG,64),method='meanSub', intensity=True)
        time_phase_zero_OG = self.pfd_contents.profs.sum(1)
        time_phase_zero=self.custom_normalize(self.down(time_phase_zero_OG,64),method='meanSub')
        freq_phase_zero_OG = self.pfd_contents.profs.sum(0)
        freq_phase_zero=self.custom_normalize(self.down(freq_phase_zero_OG,64),method='meanSub')

        # Calculate the other dedispersed values
        time_phase1, time_taken = self.get_time_vs_phase()
        time_phase2=self.down(time_phase1,64)
        time_phase=self.custom_normalize(time_phase2,method='meanSub')
        freq_phase1, time_taken = self.get_freq_vs_phase()
        freq_phase2=self.down(freq_phase1,64)
        freq_phase=self.custom_normalize(freq_phase2,method='meanSub')
        summed_profile1, time_taken = self.get_dedispersed_profile()
        summed_profile2=self.down(summed_profile1,64)
        summed_profile=self.custom_normalize(summed_profile2,method='meanSub', intensity=True)

        # Prepare directory for saving plots
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Plot zero-DM values if requested
        if plot_zero_vals:
            # Plot intensity_zero (1D)
            plt.figure(figsize=(8, 6))
            plt.plot(intensity_zero, label='Intensity Zero DM')
            plt.title('Intensity (Zero DM)')
            plt.xlabel('Phase bins')
            plt.ylabel('Intensity')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{file_prefix}intensity_zero_dm.png'))
            plt.show()

            # Plot time_phase_zero (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(time_phase_zero, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Time vs Phase (Zero DM)')
            plt.xlabel('Phase bins')
            plt.ylabel('Time bins')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}time_vs_phase_zero_dm.png'))
            plt.show()

            # Plot freq_phase_zero (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(freq_phase_zero, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Frequency vs Phase (Zero DM)')
            plt.xlabel('Phase bins')
            plt.ylabel('Frequency channels')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}freq_vs_phase_zero_dm.png'))
            plt.show()

        # Plot subtracted values if requested
        if plot_subtracted:
            # Plot summed_profile - intensity_zero (1D)
            plt.figure(figsize=(8, 6))
            plt.plot(summed_profile - intensity_zero, label='Summed - Intensity Zero')
            plt.title('Summed Profile - Intensity Zero DM')
            plt.xlabel('Phase bins')
            plt.ylabel('Subtracted Intensity')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{file_prefix}summed_minus_intensity_zero_dm.png'))
            #plt.show()

            # Plot time_phase - time_phase_zero (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(time_phase - time_phase_zero, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Time vs Phase (Subtracted)')
            plt.xlabel('Phase bins')
            plt.ylabel('Time bins')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}time_vs_phase_subtracted.png'))
            #plt.show()

            # Plot freq_vs_phase - freq_phase_zero (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(freq_phase - freq_phase_zero, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('Frequency vs Phase (Subtracted)')
            plt.xlabel('Phase bins')
            plt.ylabel('Frequency channels')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}freq_vs_phase_subtracted.png'))
            #plt.show()

        if subtract_first:
            # **Step 1: Subtract Zero-DM values from the dedispersed values**
            intensity_subtracted = summed_profile1 - intensity_zero_OG
            time_phase_subtracted = time_phase1 - time_phase_zero_OG
            freq_phase_subtracted = freq_phase1 - freq_phase_zero_OG

            # **Step 2: Downsample the subtracted data**
            intensity_downsampled = self.down(intensity_subtracted, 64)
            time_phase_downsampled = self.down(time_phase_subtracted, 64)
            freq_phase_downsampled = self.down(freq_phase_subtracted, 64)

            # **Step 3: Normalize the downsampled data**
            intensity_normalized = self.custom_normalize(intensity_downsampled, method='meanSub', intensity=True)
            time_phase_normalized = self.custom_normalize(time_phase_downsampled, method='meanSub')
            freq_phase_normalized = self.custom_normalize(freq_phase_downsampled, method='meanSub')

            # **Plot and save the results**
            
            # Plot normalized intensity (1D)
            plt.figure(figsize=(8, 6))
            plt.plot(intensity_normalized, label='Normalized Intensity (Subtracted)')
            plt.title('Intensity(Best)-Intensity(0 DM)')
            plt.xlabel('Phase bins')
            plt.ylabel('Normalized Intensity')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'{file_prefix}_intensity_Sfirst.png'))
            plt.show()

            # Plot normalized time-phase (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(time_phase_normalized, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('TP(Best)-TP(0 DM)')
            plt.xlabel('Phase bins')
            plt.ylabel('Time bins')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}_time_vs_phase_Sfirst.png'))
            plt.show()

            # Plot normalized freq-phase (2D) with colorbar
            plt.figure(figsize=(8, 6))
            plt.imshow(freq_phase_normalized, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar()
            plt.title('FP(Best)-FP(0 DM))')
            plt.xlabel('Phase bins')
            plt.ylabel('Frequency channels')
            plt.savefig(os.path.join(save_dir, f'{file_prefix}_freq_vs_phase_Sfirst.png'))
            plt.show()



    def executer(self):
        #self.clock = True
        if self.features_to_extract.getboolean('time_vs_phase'):
            time_phase, time_taken = self.get_time_vs_phase()
            #print('TP',time_phase.shape)
            self.data_saver(time_phase,'Time_Phase', 64, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/Zero_DM_tests/GBNCC_RFI_2', save_npy=False)
          
            #print(f'time vs phase: {time_phase} (Time taken: {time_taken:.4f} seconds)')

        if self.features_to_extract.getboolean('dedispersed_profile'):
            summed_profile, time_taken = self.get_dedispersed_profile()
            #print(f'dedispersed and summed profile: {summed_profile} (Time taken: {time_taken:.4f} seconds)')
            #print('intensity',summed_profile.shape)
            self.data_saver(summed_profile,'Intensity', 64, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/Zero_DM_tests/GBNCC_RFI_2', save_npy=False,intensity1=True)

        if self.features_to_extract.getboolean('freq_vs_phase'):
            freq_vs_phase, time_taken = self.get_freq_vs_phase()
            #print(f'freq vs phase: {freq_vs_phase} (Time taken: {time_taken:.4f} seconds)')
            print('FP',freq_vs_phase.shape)
            self.data_saver(freq_vs_phase,'Freq_Phase', 64, save_plots=True, normalize_method='meanSub', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/Zero_DM_tests/GBNCC_RFI_2', save_npy=False)

        if self.features_to_extract.getboolean('dm_curve'): #need function for time
            lodm = self.pfd_contents.dms[0]
            hidm = self.pfd_contents.dms[-1]
            chis,DMs = self.pfd_contents.plot_chi2_vs_DM(loDM=lodm, hiDM=hidm, N=64)
            #np.save('dm_curve_data.npy', (chis, DMs))

            #data = np.vstack(chis_DMs_tuple)
            #self.data_saver(data,'FAST2_P_DMcurve', 64, save_plots=True, normalize_method='none', save_directory='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/test3', save_npy=False)

            #(chis,DMs) = self.get_dm_curve(bins=64)
            #self.DM_plot_saver(chis, DMs,'FAST_DM_curve')
            # print(chis)
            # print(DMs)

            # check=(chis,DMs)
            # print(type(dm_chis))
            # print(type(dm_chis[0]))
            # print(type(dm_chis[1]))
            # print(len(dm_chis[0]))
            # print((dm_chis[1]))

            #print(type(dm_chis[1]))
            # print(check[0].shape)
            # print(check[1].shape)

            # #to plot
            plt.plot(DMs, chis,'black')
            plt.xlabel('Dispersion Measure (DM)')
            plt.ylabel('Reduced Chi-squared')
            plt.title('Chi-squared vs Dispersion Measure')

            # Save the plot to a file
            plt.savefig('/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/FAST_P_1/chi2_vs_DM_plot.png')  # Save as a PNG file


            #plt.figure()
          
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

        if self.features_to_extract.getboolean('zero_dm'):
            #self.zero_DM(plot_zero_vals=True, plot_subtracted=True, save_dir='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/Zero_DM_tests/GBNCC_RFI_2',file_prefix='GBNCC_RFI_2')
            self.zero_DM(subtract_first=True,save_dir='/hercules/u/dbhatnagar/PulsarFeatureLab/PFL_Python3_Src/Zero_DM_tests/gbncc_rfi_2_s1',file_prefix='gbncc_rfi_2_s1')

       #self.clock = False


if __name__ == "__main__":
    myufe = UFE(debug=True, clock=True)
   
    myufe.executer()