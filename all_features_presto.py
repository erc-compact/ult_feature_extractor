import presto.prepfold as pp
import argparse
import numpy as np
from PFDFile import PFD
from PFDFeatureExtractor import PFDFeatureExtractor
parser = argparse.ArgumentParser()
parser.add_argument("--file", help="pfd file", required=True)
args = parser.parse_args()

test_pfd= args.file
print('success')
pfd_contents=pp.pfd(test_pfd)
#time_phase
time_phase=pfd_contents.time_vs_phase()
print('time vs phase: ', time_phase)

#best DM is calculcated by presto
dedispersed=pfd_contents.dedisperse()

summed_profile=pfd_contents.sumprof
print('dedispersed and summed profile: ', summed_profile)

 #Freq vs phase by PFL
debugFlag = False
candidateName = test_pfd
pfd_instance = PFD(debugFlag, candidateName)
print('checkpoint')
# Step 3: Call the plot_subbands method
result = pfd_instance.plot_subbands()
print('freq vs phase: ', result)

#chi2 vs DM by PFL
DM_result= pfd_instance.DM_curve_Data()

print('DM curve: ',DM_result)

# print('check1: ',(DM_result[0].shape))
# print('check1: ',(DM_result[1].shape))

pfd_instance.computeType_6()


print('Mean_IFP: ', pfd_instance.features1['MEAN_IFP'])
print('STD_IFP: ', pfd_instance.features1['STD_IFP'])
print('SKW_IFP: ', pfd_instance.features1['SKW_IFP'])
print('KURT_IFP: ', pfd_instance.features1['KURT_IFP'])

print('Mean_DM: ', pfd_instance.features1['MEAN_DM'])
print('STD_DM: ', pfd_instance.features1['STD_DM'])
print('SKW_DM: ', pfd_instance.features1['SKW_DM'])
print('KURT_DM: ', pfd_instance.features1['KURT_DM'])