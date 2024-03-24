#%% Import Required Libraries
import pandas as pd
import numpy as np
from data_processing import perform_fft, extract_peaks, load_and_clean_data_noDIS_old, load_and_clean_data_old, load_and_clean_data_noDIS, load_and_clean_data, process_subject_data
from modeling import polynomial, sin_fit, polysin, ACA_fit, calculate_r_squared, param_estimator, log_likelihood, calculate_ic
from visualization import plot_aic_bic

# Define Constants
BIN_SIZE = 0.02
fitfunction = polynomial  # Replace this with the actual function if it's not a string

def get_function_settings(func_name):
    if func_name == "polynomial":
        return polynomial, [0.5, 1], 2
    elif func_name == "sin_fit":
        f, fft_result, _ = perform_fft(t, x)
        top_peak_freqs = extract_peaks(f, fft_result)
        return sin_fit, [0, top_peak_freqs[0], 0], 3
    elif func_name == "polysin":
        f, fft_result, _ = perform_fft(t, x)
        top_peak_freqs = extract_peaks(f, fft_result)
        return polysin, [0, 0, 0, top_peak_freqs[0], 0], 5
    elif func_name == "ACA_fit":
        return ACA_fit, [0.9, 0.16, 0.21, 1.63], 4
    else:
        raise ValueError("Unknown function name")

def process_and_plot_subject_data(df, num_subjects, BIN_SIZE, df_all, func_name):
    fit_function, p0, param_count = get_function_settings(func_name)
    
    print('used n_subs imported', num_subjects.max())
    fitaccSUBS = np.zeros((2, len(num_subjects)))
    curve_params = np.zeros((len(num_subjects), param_count + 1))

    for ii, subject in enumerate(num_subjects):
        bin_t, bin_x, t, x = process_subject_data(df, subject, BIN_SIZE, df_all, Zc=True)
        print(curve_params[ii, :].shape)
        print(param_estimator(t, x, fit_function, p0, bounds=(-np.inf, np.inf)).shape)

        curve_params[ii, :] = param_estimator(t, x, fit_function, p0, bounds=(-np.inf, np.inf))
        fitaccSUBS[:, ii] = calculate_ic(t, x, param_count, curve_params[ii, :], fit_function)
        
    return bin_t, bin_x, t, x, curve_params, fitaccSUBS

def create_dataframe(results_matrix, label):
    """Create a DataFrame from results_matrix and add a 'Dataset' column with the given label."""
    df = pd.DataFrame(results_matrix, columns=['AIC', 'BIC', 'SSR'])
    df['Dataset'] = label
    return df

def load_data(ageset):
    if ageset == "old":
        df_all, df_noDIS = load_and_clean_data_noDIS_old('PrepAtt22_behav_table4R_all.xlsx')
        df = load_and_clean_data_old('PrepAtt22_behav_table4R_all.xlsx')
    elif ageset == "young":
        df_all, df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
        df = load_and_clean_data('PAT22_summaryall.xlsx')
    return df_all, df_noDIS, df

old_results, young_results = None, None
    
# Main stuff

# Initialize an empty dictionary to store DataFrames for each function
results_dfs = {}

# List of function names you're interested in
func_names = ["polynomial", "sin_fit", "polysin", "ACA_fit"]

for func_name in func_names: #loop through fxns
    master_results = []
    for ageset in ["old", "young"]:


        df_all, df_noDIS, df = load_data(ageset)
        bin_t, bin_x, t, x, curve_params, fit_metrics = process_and_plot_subject_data(df, df['sub_idx'].unique(), BIN_SIZE, df_all, func_name)
        
        SSRs = curve_params[:, -1]
        results_matrix = np.column_stack((fit_metrics[0, :], fit_metrics[1, :], SSRs))
        
        results_df = pd.DataFrame(results_matrix, columns=['AIC', 'BIC', 'SSR'])
        results_df['Dataset'] = ageset
        
        master_results.append(results_df)

    # Combine the old and young DataFrames
    master_df = pd.concat(master_results, ignore_index=True)
    
    # Store in dictionary
    results_dfs[func_name] = master_df

# Save the DataFrame to a CSV file
for func_name, df in results_dfs.items():
    df.to_csv(f"{func_name}_results.csv", index=False)
print('dataframe saved')

# %% Export between groups DF for each Metric
# Initialize a dictionary to store DataFrames for each metric
dfs_all_metrics = {}

# List of metrics we are interested in
metrics = ['AIC', 'BIC', 'SSR']

# Loop through each metric
for metric in metrics:

    # Initialize an empty DataFrame to store values for all models
    df_metric = pd.DataFrame()

    # Loop through each model's DataFrame stored in results_dfs
    for func_name, df in results_dfs.items():
        print(f"Debug: Working on {func_name} for {metric}")
        print(df.head())
        # Extract only the metric and 'Dataset' columns
        df_subset = df[[metric, 'Dataset']]
        # Rename the metric column to the model name for better identification
        df_subset = df_subset.rename(columns={metric: func_name})

        # Add a subject counter for merging
        df_subset['Subject'] = df_subset.groupby('Dataset').cumcount() + 1

        if df_metric.empty:
            df_metric = df_subset
        else:
            df_metric = pd.merge(df_metric, df_subset, on=['Dataset', 'Subject'], how='outer')

    print("After Merge")
    print("df_metric shape:", df_metric.shape)
    df_metrics = df_metric[['Dataset', 'Subject', 'polynomial', 'sin_fit', 'polysin', 'ACA_fit']]

    # Store in dictionary
    dfs_all_metrics[metric] = df_metrics

    # Optional: Save df_metric to a CSV file
    df_metrics.to_csv(f"{metric}_allmodels_YoungvsOld.csv", index=False)


# %%
