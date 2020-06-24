import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
df_ro161 = pd.read_csv(r"S:\eshape\Pilot 1\results\Metrics\VHVV\VHVV_fields_calval_ro161.csv").reset_index(drop = True).drop(columns = ['Date'])
df_ro110 = pd.read_csv(r"S:\eshape\Pilot 1\results\Metrics\VHVV\VHVV_fields_calval_ro110.csv").reset_index(drop = True).drop(columns = ['Date'])
df_ro110_np = df_ro110.to_numpy()
df_ro110_np = df_ro110_np[~np.isnan(df_ro110_np)]
df_ro161_np = df_ro161.to_numpy()
df_ro161_np = df_ro161_np[~np.isnan(df_ro161_np)]

# concatenate two arrays
VH_VV_dataset = np.concatenate((df_ro161_np,df_ro110_np))
plt.hist(VH_VV_dataset, density= True, bins = 30)
plt.ylabel('Probability')
plt.xlabel('VHVV ratio (dB)')
plt.savefig(r'S:\eshape\Pilot 1\results\Metrics\VHVV\VHVV_density_distribution.png')
plt.close()
