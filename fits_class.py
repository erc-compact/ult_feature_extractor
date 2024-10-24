import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import presto.prepfold as pp
import presto.bestprof as bestprof
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

class FitSit:
    def __init__(self, debug=False):
        self.debug=debug
        self.arg_parser()

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--pfd_file", help="pfd file", required=True)
        parser.add_argument("--bestprof_file", help="bestprof file", required=True)
        parser.add_argument("--config", help="config file", required=True)
        args = parser.parse_args()

        self.test_pfd = args.pfd_file
        self.config_file = args.config
        self.test_bestprof= args.bestprof_file

        config = configparser.ConfigParser()
        config.read(self.config_file)

        self.features_to_extract = config['FITS']
        self.pfd_contents = pp.pfd(self.test_pfd) # presto
        self.bestprof_contents=bestprof.bestprof(self.test_bestprof)
        self.period=self.bestprof_contents.p0_topo
        self.freq=1/self.period
        
    def down(self,data,bins):
        down_data= downsample(data,bins)
        return down_data

    # Define sine function
    def sine_function(self,x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C

    #sine with fixed Frequency
    def sine_function_fixedF(self,x, A, phi, C):
        omega=2*np.pi*self.freq
        return A * np.sin(omega * x + phi) + C  

    # Define sine squared function
    def sine_squared_function(self,x, A, omega, phi, C):
        return A * (np.sin(omega * x + phi))**2 + C

    # Define Gaussian function
    def gaussian_function(self,x, A, mu, sigma, C):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + C

    # Define double Gaussian function
    def double_gaussian_function(self,x, A1, mu1, sigma1, A2, mu2, sigma2, C):
        gaussian1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
        gaussian2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
        return gaussian1 + gaussian2 + C

    

     
