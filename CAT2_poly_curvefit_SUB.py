#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import seaborn as sns
from data_processing import load_and_clean_data_noDIS_old,load_and_clean_data_old,load_and_clean_data_noDIS,load_and_clean_data, process_subject_data, perform_fft, extract_peaks
from modeling import calculate_r_squared,polynomial, param_estimator, log_likelihood, calculate_ic
from visualization import norm_fft, plot_aic_bic,construct_text, plot_SUBandFit, plot_box_swarm, plot_polar_histogram, plot_error, plot_3d_surface, compute_histogram, plot_3d_histogram

# Constants
BIN_SIZE = 0.02
NUM_PEAKS = 1
PARAM_FIT = 2 #CHANGE
fitfunction = polynomial 

def process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all):
    """
    define the individualities of the plot
    """
    print(' used n_subs imported',num_subjects.max())

    curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
    for ii, subject in enumerate(num_subjects):
        if ii == (len(num_subjects)-1):
            print('ii=',ii)

        bin_t,bin_x, t, x = process_subject_data(df, subject, BIN_SIZE,df_all,Zc=True)
        p0 = [0.5,1] 
        curve_params[ii,:] = param_estimator(t, x,fitfunction,p0,bounds=(-np.inf, np.inf))

        R_squared_array[ii] = calculate_r_squared(x, curve_params[ii, -1])
    
        plot_SUBandFit(t, x, bin_t, bin_x, curve_params[ii,:],param_names,fitfunction,subject,[-2,5])
        fitaccSUBS[:,ii]=calculate_ic(t,x,PARAM_FIT,curve_params[ii,:],fitfunction)
    return bin_t, bin_x, t, x,curve_params,fitaccSUBS


#Data Processing 
df = load_and_clean_data('PAT22_summaryall.xlsx') 
df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx') #those have all conditions and serve to find z score

num_subjects = df['sub_idx'].unique()  # you can change that to vary the number of subjects
curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
fitaccSUBS = np.zeros((2,len(num_subjects))) # calculate the AIC and BIC per subjects
parameters = ['a','b']
colors = ['aquamarine','seashell' ]
param_names = ['a','b','SSR']
bin_width_freq = 3 #not that usefull
# Initialize R_squared_array 
R_squared_array = np.zeros(len(num_subjects))

#Processing and Modeling of parameters
bin_t, bin_x, t, x,curve_params,fitaccSUBS = process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all)
#init elements for Visuals
a = curve_params[:,0].flatten()
b = curve_params[:,1].flatten()
values = np.column_stack((a,b))
plot_box_swarm(values, parameters, colors)
mean_aic,mean_bic,mad_aic,mad_bic= plot_aic_bic(fitaccSUBS, num_subjects)
SSR = curve_params[:, -1]
mean_ssr = np.mean(SSR)
mad_ssr = np.mean(np.abs(SSR - mean_ssr))
print(f"Model Type: {fitfunction}\n"
      f"Mean AIC: {mean_aic:.2f}\n"
      f"MAD of AIC: {mad_aic:.2f}\n"
      f"Mean BIC: {mean_bic:.2f}\n"
      f"MAD of BIC: {mad_bic:.2f}\n"
      f"Mean SSR: {mean_ssr:.2f}\n"
      f"MAD of SSR: {mad_ssr:.2f}")

mean_r_squared = np.mean(R_squared_array)
mad_r_squared = np.mean(np.abs(R_squared_array - mean_r_squared))
print(f"\nMean R^2: {mean_r_squared:.2f}\n"
      f"MAD of R^2: {mad_r_squared:.2f}")
 
# %% PLOT Young vs OLD
    

# Define function to load data
def load_data(ageset):
    if ageset == "old":
        df_all, df_noDIS = load_and_clean_data_noDIS_old('PrepAtt22_behav_table4R_all.xlsx')
        df = load_and_clean_data_old('PrepAtt22_behav_table4R_all.xlsx')
    elif ageset == "young":
        df_all, df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
        df = load_and_clean_data('PAT22_summaryall.xlsx')
    return df_all, df_noDIS, df
fft_results = []

# Define function to process data
def process_EVA(df,df_all):
    num_subjects = df['sub_idx'].unique()
    print('# subjects:',len(num_subjects))
    parameters = np.zeros((len(num_subjects), PARAM_FIT+1))
    ICs = np.zeros((2, len(num_subjects)))
    print('IC init. lenght:',ICs.shape[1])
    SSRs =  np.zeros((1, len(num_subjects)))
    bin_t, bin_x, t, x, parameters, ICs = process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all)
    SSRs = parameters[:, -1]
    return ICs, SSRs

old_results, young_results = None, None

# Loop through both datasets (old and young)
for ageset in ["old", "young"]:
    results_matrix = []
    df_all, df_noDIS, df = load_data(ageset)
    ICs, SSRs = process_EVA(df, df_all)
    print('lenght fit ACC:',ICs.shape[1],'\nwith ageset:',ageset)
    # Construct the result matrix for this age set
    results_matrix = np.column_stack((ICs[0,:],ICs[1,:],SSRs))
    
    if ageset == "old":
        old_results = results_matrix
    else:
        young_results = results_matrix

model_name = "poly"
print('old results:', len(old_results))

# Convert arrays to DataFrames
columns = ['AIC', 'BIC', 'SSR']
old_df = pd.DataFrame(old_results, columns=columns)
young_df = pd.DataFrame(young_results, columns=columns)

# Add model name column to both DataFrames
old_df['Model'] = model_name
young_df['Model'] = model_name
print('old df:', len(old_df))

# Export DataFrames to CSV files (you can also export to Excel if needed)
old_df.to_csv(f'old_results_{model_name}.csv', index=False)
young_df.to_csv(f'young_results_{model_name}.csv', index=False)
print('done exporting')

# %% PLOT BIC AIC for each model
# Extracting BIC values from the results for old and young data
old_BIC = old_results[:, 1]  # The second column is the BIC values for old dataset
young_BIC = young_results[:, 1]  # The second column is the BIC values for young dataset

# old_SSR = old_results[0][:, 2]
# young_SSR = young_results[0][:, 2]

# Calculating the Mean Absolute Deviation (MAD) for old and young BIC values
mad_old_BIC = np.median(np.abs(old_BIC - np.median(old_BIC)))
mad_young_BIC = np.median(np.abs(young_BIC - np.median(young_BIC)))

# Combine data for violinplot
data = pd.DataFrame({
    'Age': ['Old'] * len(old_BIC) + ['Young'] * len(young_BIC),
    'BIC': np.concatenate([old_BIC, young_BIC])
})

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Age', y='BIC', data=data, inner='quartile', palette="muted")

# Add mean and MAD as error bars
plt.errorbar(x=[0, 1], 
             y=[np.mean(old_BIC), np.mean(young_BIC)], 
             yerr=[mad_old_BIC, mad_young_BIC], 
             fmt='rx', markersize=10)

plt.legend()
plt.title('PolyFit: Violin plot of SSR values for each group of participants (Old and Young )')
plt.show()
# %%
