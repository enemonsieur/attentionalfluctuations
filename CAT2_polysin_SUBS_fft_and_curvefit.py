#%% Import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import seaborn as sns
from data_processing import calculate_ssr,generate_gaussian,fit_gaussian, load_and_clean_data_noDIS_old,load_and_clean_data_old,load_and_clean_data_noDIS,load_and_clean_data, process_subject_data, perform_fft, extract_peaks
from modeling import calculate_r_squared, polysin, param_estimator, log_likelihood, calculate_ic
from visualization import plot_params_vs_error,norm_fft, plot_aic_bic,construct_text, plot_SUBandFit, plot_box_swarm, plot_polar_histogram, plot_error, plot_3d_surface, compute_histogram, plot_3d_histogram

# Constants
BIN_SIZE = 0.02
NUM_PEAKS = 1
PARAM_FIT = 5
fitfunction = polysin


def process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all):
    """
    define the individualities of the plot
    """
    for ii, subject in enumerate(num_subjects):

        bin_t, bin_x, t, x = process_subject_data(df, subject,BIN_SIZE,df_all,Zc=True)
        std_values[ii] = np.std(x)  
        f, fft_result, _ = perform_fft(t, x)
        fft_results.append(fft_result)
        top_peak_freqs = extract_peaks(f, fft_result)
        p0 = [0,0,0, top_peak_freqs[0],0] #to adapt
        curve_params[ii,:] = param_estimator(t, x,fitfunction,p0,bounds=(-np.inf, np.inf))
        #R_squared_array[ii] = calculate_r_squared(x, curve_params[ii, -1])


        #ssr_gaussian_values[ii] = calculate_ssr(x, predicted_x_gauss)

        plot_SUBandFit(t, x, bin_t, bin_x, curve_params[ii,:],param_names,fitfunction,subject,[-2,5])
        fitaccSUBS[:,ii] = calculate_ic(t,x,PARAM_FIT, curve_params[ii,:],fitfunction)
    return bin_t, bin_x, t, x, f, fft_result, f, fft_results, curve_params, fitaccSUBS

def plot_data(f, fft_results, bin_width_freq, num_subjects, curve_params, fitaccSUBS, values, parameters, colors):
    """
    Plot various data visualizations.
    """

    # Plot the 3D surface
    #plot_3d_histogram(f,fft_results,bin_width_freq)
    plot_3d_surface(num_subjects, f, fft_results,' 3D View of FFT Across Participants')

    #plot Errors estimations
    mean_aic,mean_bic,mad_aic,mad_bic = plot_aic_bic(fitaccSUBS, num_subjects)
    #plot_error(curve_params,num_subjects)
    #plot_params_vs_error(curve_params,'Freq to Phase Correlation in Polysin')

    # Plot diff. parameters
    plot_polar_histogram(curve_params[:,-2])
    plot_box_swarm(values, parameters, colors)
    return mean_aic, mad_aic, mean_bic, mad_bic

#Data Processing 
df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
df = load_and_clean_data('PAT22_summaryall.xlsx')
num_subjects = df['sub_idx'].unique()  # you can change that to vary the number of subjects
fit_freq = np.zeros((len(num_subjects)))
curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
fft_results = []  # List to store FFT results for each subject
fitaccSUBS = np.zeros((2,len(num_subjects))) # calculate the AIC and BIC per subjects
std_values = np.zeros(len(num_subjects))


parameters = ['a','b','Amplitude', 'Frequency']
colors = ['aquamarine','seashell','skyblue', 'lightgreen']
param_names = ['a','b','Amp','Freq','Phi','SSR']
bin_width_freq = 3

#Processing and Modeling of parameters
bin_t, bin_x, t, x, f, fft_result, f, fft_results, curve_params, fitaccSUBS = process_and_plot_subject_data(df, num_subjects, BIN_SIZE, df_all)

# init elements for Visuals
fft_results = np.array(fft_results)
a = curve_params[:,0].flatten()
b = curve_params[:,1].flatten()
amplitude = np.abs(curve_params[:,2].flatten())
frequency = curve_params[:,3].flatten()
values = np.column_stack((a, b, amplitude, frequency))
SSR = curve_params[:, -1]

# Visualization
mean_aic, mad_aic, mean_bic, mad_bic = plot_data(f, fft_results, bin_width_freq, num_subjects, curve_params, fitaccSUBS, values, parameters, colors)
mean_ssr = np.mean(SSR)
mad_ssr = np.mean(np.abs(SSR - mean_ssr))
print(f"Model Type: {fitfunction}\n"
      f"Mean AIC: {mean_aic:.2f}\n"
      f"MAD of AIC: {mad_aic:.2f}\n"
      f"Mean BIC: {mean_bic:.2f}\n"
      f"MAD of BIC: {mad_bic:.2f}\n"
      f"Mean SSR: {mean_ssr:.2f}\n"
      f"MAD of SSR: {mad_ssr:.2f}")
plot_params_vs_error(SSR,std_values, 'SD across participants','Subjects','Standard Deviation','blue')
# mean_r_squared = np.mean(R_squared_array)
# mad_r_squared = np.mean(np.abs(R_squared_array - mean_r_squared))
# print(f"\nMean R^2: {mean_r_squared:.2f}\n"
#       f"MAD of R^2: {mad_r_squared:.2f}")
#%% PLOT OLD VS YOUNG
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
    print(num_subjects)
    fft_results = [] 
    curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
    fitaccSUBS = np.zeros((2, len(num_subjects)))
    bin_t, bin_x, t, x, f, fft_result, f, fft_results, curve_params, fitaccSUBS = process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_all)
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

model_name = "polysin"

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
plt.title('PolySin: Violin plot of BIC values for each group of participants (Old and Young )')
plt.show()
 # %%




#plot Box Swarm + Bee Plot


# %% PLotting the Before and After of the people 
import matplotlib.pyplot as plt
def plot_SUBandFit(df, subject, BIN_SIZE, df_all, subject_num):
    """
    Plot each polynomial fit for each peak in curve params for non-Z-scored data.
    """
    
    # Run your data processing function
    bin_t, bin_x, t, x = process_subject_data(df, subject, BIN_SIZE, df_all, Zc=False)
    
    # Plot the trials data
    plt.plot(t, x, 'o', alpha=0.5, color='lightblue', label='individual trials')
    plt.plot(bin_t, bin_x, '-', alpha=0.5, color='black', label='35 ms binned trials')

    # Add labels and titles
    plt.xlabel('Distractor-Target interval (ms)', fontsize=18)
    plt.ylabel('Reaction Time (ms)', fontsize=18)
    plt.title(f'Subject {subject_num}', fontsize=20)
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_SUBandFit(df, 19, BIN_SIZE, df_all, 6)

# %%
