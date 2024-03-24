#%% Import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import seaborn as sns
from data_processing import load_and_clean_data_noDIS_old,load_and_clean_data_old,load_and_clean_data_noDIS, load_and_clean_data, process_subject_data, perform_fft, extract_peaks
from modeling import calculate_r_squared,sin_fit, param_estimator, log_likelihood, calculate_ic
from visualization import plot_params_vs_error,norm_fft,plot_aic_bic,construct_text, plot_SUBandFit, plot_box_swarm, plot_polar_histogram, plot_error, plot_3d_surface, compute_histogram, plot_3d_histogram
from scipy.stats import zscore, pearsonr


def process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all,curve_params,fitaccSUBS ):
    """
    define the individualities of the plot
    """
    for ii, subject in enumerate(num_subjects):

        bin_t, bin_x, t, x = process_subject_data(df, subject,BIN_SIZE,df_noDIS,Zc=True)
        #std_values[ii] = np.std(x)  
        f, fft_result, mask = perform_fft(t, x)
        fft_results.append(fft_result)
        top_peak_freqs = extract_peaks(f, fft_result)
        p0 = [0,top_peak_freqs[0],0]
        print(curve_params.shape,'subject num(ii):',ii)
        curve_params[ii,:] = param_estimator(t, x,fitfunction,p0,bounds=(-np.inf, np.inf))
        R_squared_array[ii] = calculate_r_squared(x, curve_params[ii, -1])
        plot_SUBandFit(t, x, bin_t, bin_x, curve_params[ii,:],param_names,fitfunction,subject,ysize=[-2,5])
        fitaccSUBS[:,ii] = calculate_ic(t,x,PARAM_FIT,curve_params[ii,:],fitfunction)
    return bin_t, bin_x, t, x,f, fft_result,f, fft_results,p0,curve_params,std_values,fitaccSUBS

def plot_data(f, fft_results, bin_width_freq, num_subjects, curve_params, fitaccSUBS, values, parameters, colors):
    """
    Plot various data visualizations.
    """

    # Plot the 3D surface
    plot_3d_surface(num_subjects, f, norm_fft_results,' 3D View of FFT Across Participants')
    #plot Errors estimations
    mean_aic,mean_bic,mad_aic,mad_bic  = plot_aic_bic(fitaccSUBS, num_subjects)
    #plot_params_vs_error(curve_params,'Comparison: Freq vs Phase')
    plot_SD_per_subs(curve_params,SSR,num_subjects,'Corr. between SSR and SD')
    # Plot diff. parameters
    plot_polar_histogram(curve_params[:,-2])
    plot_box_swarm(values, parameters, colors)
    return mean_aic, mad_aic, mean_bic, mad_bic

def plot_SD_per_subs(curve_params,SSR,num_subjects,title):
    amplitude = zscore(np.abs(curve_params[:, 0].flatten() ))
    frequency = zscore(curve_params[:, 1].flatten())
    phase = zscore(curve_params[:, 2].flatten()  % (2*np.pi))
    SSR = zscore(SSR)
    r_corr, _ = pearsonr(SSR, amplitude)

    # After processing all subjects, plot the standard deviations
    plt.figure()
    #plt.plot(num_subjects, amplitude, 'o-', label='amp')
    #plt.plot(num_subjects, frequency, 'o-', label='freq')
    plt.plot(num_subjects, amplitude, 'o-', label='phase')
    plt.plot(num_subjects, SSR,'o-', label='residual error')
    plt.xlabel('Subject')
    plt.ylabel('SSR+Amplitude')
    plt.title(f"{title}\nPearson Corr. Coeff: {r_corr:.2f}")   
    plt.xticks(np.arange(min(num_subjects), max(num_subjects)+1, 1.0))
    plt.grid(True)
    plt.legend()
    plt.show()
# Constants

BIN_SIZE = 0.021
NUM_PEAKS = 1
PARAM_FIT = 3
fitfunction = sin_fit

# Init Datas for Processing...
df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
df = load_and_clean_data('PAT22_summaryall.xlsx')
num_subjects = df['sub_idx'].unique() #np.array([2,7,10,11,12]) #np.array([ 1,  3,  4,  5,  6,  8,  9, 13, 14, 15, 16, 17,18, 19, 20, 21]) you can change that to vary the number of subjects
fit_freq = np.zeros((len(num_subjects)))
curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
fft_results = []  # List to store FFT results for each subject
fitaccSUBS = np.zeros((2,len(num_subjects))) # calculate the AIC and BIC per subjects
param_names = ['Amp','Freq','Phi','SSR']
parameters = ['Amplitude', 'Frequency']
colors = ['skyblue', 'lightgreen']
bin_width_freq = 3
std_values = np.zeros(len(num_subjects))
R_squared_array = np.zeros(len(num_subjects))

#Processing and Modeling of parameters
bin_t, bin_x, t, x,f, fft_result,f, fft_results,p0,curve_params,std_values,fitaccSUBS = process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all,curve_params, fitaccSUBS)



# init elements for Visuals
amplitude = np.abs(curve_params[:, 0].flatten())
frequency = curve_params[:, 1].flatten()
phase = ( curve_params[:, 2].flatten()) % (2*np.pi)
values = np.column_stack((amplitude, frequency))
fft_results = np.array(fft_results)
norm_fft_results = norm_fft(fft_results)
SSR = curve_params[:, -1]


#visualization
mean_aic, mad_aic, mean_bic, mad_bic = plot_data(f, fft_results, bin_width_freq, num_subjects, curve_params, fitaccSUBS, values, parameters, colors)
# Calculate the mean and MAD of SSR
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

# %%
#plot_params_vs_error(std_values,amplitude, 'correlation of SD and Amplitude','SD','Amplitude')
# Freq Phase with all subs corr: 0.02
# Freq Amp with all subs corr: 0.14

# %%

# # Init Datas for Processing...

# df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
# num_subjects = df['sub_idx'].unique() #np.array([2,7,10,11,12]) #np.array([ 1,  3,  4,  5,  6,  8,  9, 13, 14, 15, 16, 17,18, 19, 20, 21]) you can change that to vary the number of subjects
# fit_freq = np.zeros((len(num_subjects)))
# curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
# fft_results = []  # List to store FFT results for each subject
# fitaccSUBS = np.zeros((2,len(num_subjects))) # calculate the AIC and BIC per subjects
# param_names = ['Amp','Freq','Phi','SSR']
# parameters = ['Amplitude', 'Frequency']
# colors = ['skyblue', 'lightgreen']
# bin_width_freq = 3
# std_all_values = np.zeros(len(num_subjects))
# std_noDIS_values = np.zeros(len(num_subjects))

# #Processing and Modeling of parameters
# # This code plots the correlatio n 
# for ii, subject in enumerate(num_subjects):
#     bin_t, bin_x, t, x_all = process_subject_data(df_all, subject,BIN_SIZE,Zc=False)
#     bin_t, bin_x, t, x_noDIS = process_subject_data(df_noDIS, subject,BIN_SIZE,Zc=False)
#     std_all_values[ii] = np.std(x_all)
#     std_noDIS_values[ii] = np.std(x_noDIS)
# plt.figure(figsize=(10, 6))
# plt.scatter(std_noDIS_values, std_all_values, marker='o', c='b', label='Subjects')
# plt.title('Correlation between SD of noDIS trials and SD of all trials')
# plt.xlabel('Standard Deviation of noDIS ')
# plt.ylabel('Standard Deviation of all trials')
# plt.grid(True)
# plt.legend()
# plt.show()
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
    fft_results = [] 
    curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
    fitaccSUBS = np.zeros((2, len(num_subjects)))
    bin_t, bin_x, t, x,f, fft_result,f, fft_results,p0,curve_params,std_values,fitaccSUBS = process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all,curve_params,fitaccSUBS)
    SSR = curve_params[:, -1]
    return fitaccSUBS, SSR

old_results, young_results = None, None

# Loop through both datasets (old and young)
for ageset in ["old", "young"]:
    df_all, df_noDIS, df = load_data(ageset)
    fitaccSUBS, SSR = process_EVA(df, df_all)
    
    # Construct the result matrix for this age set
    results_matrix = np.column_stack((fitaccSUBS[0,:],fitaccSUBS[1,:],SSR))
    
    if ageset == "old":
        old_results = results_matrix
    else:
        young_results = results_matrix
# Convert lists to numpy arrays
old_results = np.array(old_results)
young_results = np.array(young_results)

# Export them for the model comparison
model_name = "sin"

# Convert arrays to DataFrames
columns = ['AIC', 'BIC', 'SSR']
old_df = pd.DataFrame(old_results, columns=columns)
young_df = pd.DataFrame(young_results, columns=columns)

# Add model name column to both DataFrames
old_df['Model'] = model_name
young_df['Model'] = model_name

# Export DataFrames to CSV files (you can also export to Excel if needed)
old_df.to_csv(f'old_results_{model_name}.csv', index=False)
young_df.to_csv(f'young_results_{model_name}.csv', index=False)
print('done exporting')

# %% PLOT BIC AIC for each model
# Extracting BIC values from the results for old and young data
old_BIC = old_results[:, 2]  # The second column is the BIC values for old dataset
young_BIC = young_results[:, 2]  # The second column is the BIC values for young dataset

# old_SSR = old_results[0][:, 2]
# young_SSR = young_results[0][:, 2]

# Calculating the Mean Absolute Deviation (MAD) for old and young BIC values
mad_old_BIC = np.median(np.abs(old_BIC - np.median(old_BIC)))
mad_young_BIC = np.median(np.abs(young_BIC - np.median(young_BIC)))

# Combine data for violinplot
data = pd.DataFrame({
    'Age': ['Old'] * len(old_BIC) + ['Young'] * len(young_BIC),
    'SSR': np.concatenate([old_BIC, young_BIC])
})

# Violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(x='Age', y='SSR', data=data, inner='quartile', palette="muted")

# Add mean and MAD as error bars
plt.errorbar(x=[0, 1], 
             y=[np.mean(old_BIC), np.mean(young_BIC)], 
             yerr=[mad_old_BIC, mad_young_BIC], 
             fmt='rx', markersize=10)

plt.legend()
plt.title('SinFit: Violin plot of SSR values for each group of participants (Old and Young )')
plt.show()
 # %%
