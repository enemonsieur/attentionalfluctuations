#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_processing import load_and_clean_data_noDIS_old,load_and_clean_data_old,load_and_clean_data_noDIS,load_and_clean_data, process_subject_data, perform_fft, extract_peaks
from modeling import calculate_r_squared, ACA_fit, param_estimator, log_likelihood, calculate_ic,perform_permutations,calculate_p_value,calculate_cdf
from visualization import plot_params_vs_error,norm_fft, plot_aic_bic,construct_text, plot_SUBandFit, plot_box_swarm, plot_polar_histogram, plot_error, plot_3d_surface, compute_histogram, plot_3d_histogram
#%%
# Constants
BIN_SIZE = 0.035
PARAM_FIT = 4
fitfunction = ACA_fit 
NUM_PERMUTATIONS =100
#Data Processing 
df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
df = load_and_clean_data('PAT22_summaryall.xlsx')
num_subjects = df['sub_idx'].unique()  # you can change that to vary the number of subjects
curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
fitaccSUBS = np.zeros((2,len(num_subjects)))
parameters = ['Amp Capture','Decay Capture','Amp Arousal', 'Duration Arousal','SSR']
title='Comparing Amp_Arousal with Decay_Arousal'
std_values = np.zeros(len(num_subjects))

# Initialize R_squared_array 
R_squared_array = np.zeros(len(num_subjects))

for ii, subject in enumerate(num_subjects):
    bin_t,bin_x, t, x = process_subject_data(df, subject, BIN_SIZE,df_noDIS,Zc=True)

    # std_values[ii] = np.std(x) #Pretty obvious aint it?
    # Data modelling
    # ASSUMPTIONS
    p0 = [0.9, 0.16, 0.21, 1.63]#[0.2, 0.16, 0.21, 0.63] 
    # add bounds
    bounds = ([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf]) #([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    curve_params[ii,:] = param_estimator(t, x,fitfunction,p0, bounds=bounds,maxfev=10000)
    # Estimate Information Criteria
    fitaccSUBS[:,ii] = calculate_ic(t,x,PARAM_FIT,curve_params[ii,:],fitfunction)
    # find how much % of error the opt param explain
    R_squared_array[ii] = calculate_r_squared(x, curve_params[ii, -1])
    # Data Visualization
    plot_SUBandFit(t, x, bin_t, bin_x, curve_params[ii,:],parameters,fitfunction,subject,[-2,5])

#plot Errors estimations
mean_aic,mean_bic,mad_aic,mad_bic = plot_aic_bic(fitaccSUBS, num_subjects)
plot_error(curve_params,num_subjects)
#plot_params_vs_error(curve_params,title)
#plot Box Swarm + Bee Plot
CapAmp = curve_params[:,0].flatten()
CapDec = curve_params[:,1].flatten()
ArAmp = curve_params[:,2].flatten()
ArDec = curve_params[:,3].flatten()
SSR = curve_params[:, -1]
values = np.column_stack((CapAmp,CapDec,ArAmp, ArDec))
parameters = ['Amp Capture','Decay Capture','Amp Arousal', 'Duration Arousal']
colors = ['aquamarine','seashell','skyblue','lightgreen']
plot_box_swarm(values, parameters, colors)
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

#[0.2, 0.16473684210526315, 0.21052631578947367, 0.631578947368421] 
#%% PERMUTATIONS ANALYSES
NUM_SUBJECTS = len(num_subjects)
NUM_PERMUTATIONS = 1000
# Initialize 2D array to store permuted R² values for all subjects
permuted_r_squared_matrix = np.zeros((NUM_SUBJECTS, NUM_PERMUTATIONS))

for ii, subject in enumerate(num_subjects):
  # Process subject data
  bin_t, bin_x, t, x = process_subject_data(df, subject, BIN_SIZE, df_noDIS, Zc=True)

  # Estimate model parameters for the original data
  curve_params[ii,:] = param_estimator(t, x, fitfunction, p0, bounds=bounds, maxfev=10000)

  # Perform permutations and calculate R² for each permutation
  for permutation in range(NUM_PERMUTATIONS):
    # Shuffle the data
    permuted_x = np.random.permutation(x)

    # Estimate model parameters for the permuted data
    permuted_curve_params = param_estimator(t, permuted_x, fitfunction, p0, bounds=bounds, maxfev=10000)

    # Calculate R² for the permuted data
    permuted_r_squared = calculate_r_squared(permuted_x, permuted_curve_params[-1])

    # Store permuted R² in the matrix
    permuted_r_squared_matrix[ii, permutation] = permuted_r_squared

import scipy.stats as stats

# Calculate the p-value for each subject
p_values = []

for ii, subject in enumerate(num_subjects):
  # Extract permuted R² values for the current subject
  permuted_r_squared_subject = permuted_r_squared_matrix[ii, :]

  # Calculate the p-value
  p_values.append(calculate_p_value(R_squared_array[ii], permuted_r_squared_subject))

# Print the p-values for all subjects
print("P-values:")
print(p_values)
# Create a figure with subplots
fig, axes = plt.subplots(7, 3, figsize=(15, 10))

axes = axes.flatten()  # Flatten the axes array for easy indexing

# Iterate over subjects and create histograms in subplots
for ii in range(len(num_subjects)):
    # Extract permuted R² values for the current subject
    permuted_r_squared_subject = permuted_r_squared_matrix[ii, :]

    # Create a histogram of the permuted R² values
    axes[ii].hist(permuted_r_squared_subject, color='blue')

    # Add the true R² value as a vertical red line
    axes[ii].axvline(R_squared_array[ii], color='red')

    # Set labels and title
    axes[ii].set_xlabel("R² Value")
    axes[ii].set_ylabel("Frequency")
    axes[ii].set_title(f"Distribution of Permuted R² Values for Subject {ii + 1}")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Calculate the median of the permuted R² values for all subjects
mean_permuted_r_squared_array = np.median(permuted_r_squared_matrix, axis=0)

# Calculate the median of the real R² values
mean_real_r_squared = np.median(R_squared_array)

# Create a histogram of the mean permuted R² values
plt.hist(mean_permuted_r_squared_array, color='blue', label='Median Permuted R²')

# Add the mean real R² value as a vertical red line
plt.axvline(mean_real_r_squared, color='red', label='Median Real R²')

# Set labels and title
plt.xlabel("R² Value")
plt.ylabel("Frequency")
plt.title("Distribution of Median Permuted R² Values and Mean Real R²")
plt.legend()

# Display the plot
plt.show()

#%% Plot OLD VS YOUNG
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
    curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
    fitaccSUBS = np.zeros((2, len(num_subjects)))
    SSRs =  np.zeros((1, len(num_subjects)))
    for ii, subject in enumerate(num_subjects):
        bin_t,bin_x, t, x = process_subject_data(df, subject, BIN_SIZE,df_all,Zc=True)

        std_values[ii] = np.std(x) #Pretty obvious aint it?

        # Data modelling
        # Find other params
        p0 = [0.9, 0.16, 0.21, -1.63]
        # add bounds
        bounds = ([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf]) #([-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
        curve_params[ii,:] = param_estimator(t, x,fitfunction,p0, bounds=bounds,maxfev=100000)
        # Estimate Information Criteria
        fitaccSUBS[:,ii] = calculate_ic(t,x,PARAM_FIT,curve_params[ii,:],fitfunction)
        SSRs = curve_params[:, -1]

    return fitaccSUBS,SSRs,num_subjects,curve_params
rm_dataset_param = []

# Loop through both datasets (old and young)
for ageset in ["old", "young"]:
    curve_params = np.zeros((len(num_subjects), PARAM_FIT+1))
    df_all, df_noDIS, df = load_data(ageset)
    fitaccSUBS, SSRs,num_subjects,curve_params = process_EVA(df, df_all)
    
    # Construct the result matrix for this age set
    results_matrix = np.column_stack((fitaccSUBS[0,:],fitaccSUBS[1,:],SSRs))
    dataset_param_col = np.array([ageset]*len(num_subjects))

    if ageset == "old":
        plt.plot(results_matrix[0,:])
        print('mean AIC',np.mean(results_matrix[0,:]))
        old_results = results_matrix
    else:
        young_results = results_matrix
    ageset_rm_dataset_param = np.column_stack((dataset_param_col, curve_params[:,:-1], num_subjects))
    rm_dataset_param.append(ageset_rm_dataset_param)
#%%
# Combine the results from old and young datasets
rm_dataset_param = np.vstack(rm_dataset_param)

# Convert to pandas DataFrame
columns = ['Dataset', 'Amp Capture','Decay Capture','Amp Arousal', 'Duration Arousal', 'SubjectID']
df_rm_dataset_param = pd.DataFrame(rm_dataset_param, columns=columns)
from statsmodels.stats.anova import AnovaRM

dependent_vars = ['Amp Capture','Decay Capture','Amp Arousal', 'Duration Arousal']

for depvar in dependent_vars:
    aovrm = AnovaRM(df_rm_dataset_param, depvar=depvar, subject='SubjectID', within=['Dataset'])
    fit = aovrm.fit()
    print(f"ANOVA Results for {depvar}:")
    print(fit)





#%%
# Export them for the model comparison
model_name = "ACA"

# Convert arrays to DataFrames
columns = ['AIC', 'BIC', 'SSR']
old_df = pd.DataFrame(old_results, columns=columns)
young_df = pd.DataFrame(young_results, columns=columns)

# Add model name column to both DataFrames
old_df['Model'] = model_name
young_df['Model'] = model_name

# Export DataFrames to CSV files (you can also export to Excel if needed)
old_df.to_csv(f'old_results_{model_name}2.csv', index=False)
young_df.to_csv(f'young_results_{model_name}2.csv', index=False)
print('saved')
#%% PLOT BIC AIC for each model
# Extracting BIC values from the results for old and young data
old_BIC = old_results[:, 1]  # The second column is the BIC values for old dataset
young_BIC = young_results[:, 1]  # The second column is the BIC values for young dataset

metrics = ["AIC", "BIC", "SSR"]
metric_pvalues = {
    "AIC": 0.211,
    "BIC": 0.211,
    "SSR": 0.108
}

plt.figure(figsize=(15, 8* len(metrics)))

# Loop through each metric
for i, metric in enumerate(metrics):
    old_metric_values = old_results[:, i]
    young_metric_values = young_results[:, i]
    
    # Creating the DataFrame
    data_old = pd.DataFrame({"Age": "Older", metric: old_metric_values})
    data_young = pd.DataFrame({"Age": "Younger", metric: young_metric_values})
    data = pd.concat([data_old, data_young], axis=0)

    # Violin plot comparing Old vs Young for the current metric
    plt.subplot(len(metrics), 1, i+1)
    sns.violinplot(x="Age", y=metric, data=data, palette="muted")
    plt.title(f"Violin plot for {metric} values comparing Older vs Younger data set")

    # Starting y-position for significance line
    max_y = data[metric].max() + 65  # Arbitrarily added offset

    # If the p-value is significant, draw a line indicating it
    if metric_pvalues[metric] < 0.5:
        plt.hlines(max_y, xmin=-0.5, xmax=1.5, color="black", linewidth=1)
        marker_y = max_y + 5
        star_marker = '***' if metric_pvalues[metric] <= 0.001 else ' '
        plt.text(0.5, marker_y, star_marker, ha='center', va='bottom', color='black')

    plt.tight_layout()
    plt.title(f"{metric} ", fontsize=50)


plt.show()
# %%
# %%
