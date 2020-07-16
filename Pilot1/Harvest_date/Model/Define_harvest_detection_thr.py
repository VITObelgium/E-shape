import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
outdir = r'S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\Test10\6_daily_window_data\RMSE_plots_thresholds'

df_RMSE = pd.read_csv(r"S:\eshape\Pilot 1\data\model_harvest_detection\model_Kasper\accuracy\Test10\6_daily_window_data\RMSE_ascending_descending_harvest_date_prediction_thresholds_TAP_excl_ro_combined.csv")
Models = df_RMSE.Model_name.to_list()
number_models = len(list(set([item.split('_')[1] for item in Models])))
thresholds = sorted(list(set([item.split('_')[-1] for item in Models])))
thresholds = [round(float(item),2) for item in thresholds]
for p in range(number_models):
    Model_extract = [item for item in Models if item.split('_')[1] == str(p)]
    df_model_extract = df_RMSE.loc[df_RMSE.Model_name.isin(Model_extract)]
    df_model_extract.sort_values(by='Model_name', inplace = True)
    fig, (ax1) = plt.subplots(1, figsize = (15,10))
    ax1.plot(thresholds, df_model_extract.RMSE_ro110, color = 'red', label = 'ro110_RMSE')
    ax1.plot(thresholds, df_model_extract.RMSE_ro161, color = 'blue', label = 'ro161_RMSE')
    ax1.plot(thresholds,df_model_extract.RMSE_ro_combined, color = 'black', linestyle = 'dashed', label = 'ro_combined')
    ax1.set_xlabel(r'Harvest_prob_threshold')
    ax1.set_ylim([4,12])
    ax1.set_ylabel(r'RMSE')
    ax1.legend(loc = 'upper right')
    if not os.path.exists(os.path.join(outdir,'TAP_excl')) : os.makedirs(os.path.join(outdir,'TAP_excl'))
    fig.savefig(os.path.join(outdir,'TAP_excl','Model_{}_RMSE_trend_asc_desc_thresholds_ro_combined.png'.format(str(p))))
    plt.close()