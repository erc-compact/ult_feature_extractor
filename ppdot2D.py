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

class PPDOT:
    def __init__(self):
        self.arg_parser()

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", help="pfd file", required=True)
        parser.add_argument("--tag", help="Tag for fits")  
        args = parser.parse_args()

        self.test_pfd = args.file
        self.tag = args.tag 
        
        self.pfd_contents = pp.pfd(self.test_pfd) # presto


    def ppdot_calculator(self):

        self.len_period=len(self.pfd_contents.periods)
        print('total periods: ', self.len_period)
        
        self.len_pdots=len(self.pfd_contents.pdots)
        print('total pdots: ', self.len_pdots)
        self.ppdot2D= np.zeros((self.len_period, self.len_pdots))
        for i in range(self.len_period):
        #for i in range(5):
            print(f'Starting calculations for period index {i}')
            for j in range(self.len_pdots):
            #for j in range(5):
                #print('starting for new pdot')
                p=self.pfd_contents.periods[i]
                pdot=self.pfd_contents.pdots[j]
                self.pfd_contents.adjust_period(p=p, pd=pdot)
                self.avg_prof_new=(self.pfd_contents.profs/self.pfd_contents.proflen).sum()
                self.ppdot2D[i, j] = self.pfd_contents.calc_redchi2(avg=self.avg_prof_new)
                #print(f'Period: {p}, Pdot: {pdot} (for indices i={i}, j={j}), chi_val={self.ppdot2D[i,j]}')
        return self.ppdot2D
    

    def plot_ppdot(self, data_2d, filename='test_ppdot.png', xlabel="Pdot Index", ylabel="Period Index", title="Reduced Chi-Square"):
    
        plt.figure(figsize=(8, 6))
        plt.imshow(data_2d, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Reduced Chi-Square Value')
        
        # Labeling
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        # Save the plot to a file
        plt.savefig(filename, format='png', dpi=300)
        plt.close()  # Close the figure to free memory

        
if __name__ == "__main__":
    ppdot_instance = PPDOT()
    ppdot_data = ppdot_instance.ppdot_calculator()
    ppdot_instance.plot_ppdot(data_2d=ppdot_data, filename=f"/hercules/scratch/dbhatnagar/PPDOT_tests/{ppdot_instance.tag}.png")









    



 