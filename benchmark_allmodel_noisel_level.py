#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from modeling import sin_fit, polynomial, polysin, ACA_fit
from data_processing import exgaussian_noise, load_and_clean_data_noDIS, load_and_clean_data, process_subject_data, perform_fft, extract_peaks

#%%

# functions to take the t-values of all subjects to make sure our 
# synthetic data matchs the experimental data. 

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


def process_and_plot_subject_data(df, num_subjects, BIN_SIZE,df_noDIS):
    """
    define the individualities of the plot
    """
    for ii, subject in enumerate(num_subjects):
        bin_t, bin_x, t, x = process_subject_data(df, subject,BIN_SIZE,df_noDIS,Zc=False)
        
    return bin_t, bin_x, t, x

# def create_synthetic_time_vector(original_time_vector):
#     """Create a synthetic time vector with the same sampling irregularities as the original."""
#     # Calculate the intervals between time points
#     intervals = np.diff(np.sort(original_time_vector))

#     # Randomly choose intervals, but maintain the original interval distribution
#     synthetic_intervals = np.random.choice(intervals, size=len(intervals), replace=True)

#     # Cumulative sum to create time vector, start from the original min time
#     synthetic_time_vector = np.cumsum(np.insert(synthetic_intervals, 0, original_time_vector[0]))

#     # Normalize the synthetic time vector to match the range of the original
#     synthetic_range = synthetic_time_vector[-1] - synthetic_time_vector[0]
#     original_range = original_time_vector[-1] - original_time_vector[0]
#     synthetic_time_vector = synthetic_time_vector * (original_range / synthetic_range)
#     synthetic_time_vector += (original_time_vector[0] - synthetic_time_vector[0])

#     return synthetic_time_vector

#load params to ensure you can extract the t vector
BIN_SIZE = 0.021
NUM_PEAKS = 1
PARAM_FIT = 3



# Function to estimate curve parameters
def param_estimator(t, x, fit_function, p0, bounds=(-np.inf, np.inf)):
    popt, _ = curve_fit(fit_function, t, x, p0=p0, bounds=bounds,maxfev=50000)
    ssr = np.sum((x - fit_function(t, *popt)) ** 2)
    return popt, ssr

# Dict of the init parameters
param_steps = {
    'polynomial': {'a': np.linspace(-1.5, 0.5, 5), 'b': np.linspace(-1.5, 0.5, 5)},
    'sin_fit': {'Amp': np.linspace(0, 0.5, 5), 'freq': np.linspace(2,15, 5), 'Phi': np.linspace(0, np.pi, 5)},
    'polysin': {'a': np.linspace(-1.5, 0.5, 5), 'b': np.linspace(-1.5, 0.5, 5), 'Amp': np.linspace(0, 0.5, 5), 'freq': np.linspace(2, 15, 5), 'Phi': np.linspace(0, np.pi, 5)},
    'ACA_fit': {'capture_amplitude': np.linspace(0.2, 1.8, 5), 'capture_decay': np.linspace(0.2, 0.2, 5), 'arousal_amplitude': np.linspace(0.02, 0.3, 5), 'arousal_duration': np.linspace(0.2, 1.2, 5)}
}

# Choose model
model = 'polysin'
fit_function = polysin  # Change this to the function you want to use

# Using the dict, init the original params, and the loop of the diff p0 params
default_params = {key: np.mean(val) for key, val in param_steps[model].items()}
initial_params_steps = [val for val in param_steps[model].values()]

# stack the p0 arrays
initial_params = np.stack(initial_params_steps, axis=1)


# Gen other prametrs

# Create a unequal sampling
df_all,df_noDIS = load_and_clean_data_noDIS('PAT22_summaryall.xlsx')
df = load_and_clean_data('PAT22_summaryall.xlsx')
num_subjects = df['sub_idx'].unique()
bin_t, bin_x, t_true, x_true = process_and_plot_subject_data(df, num_subjects, BIN_SIZE, df_all)
#synthetic_t = create_synthetic_time_vector(t_true)

t = np.linspace(start=0.4, stop=0.9, num=1000)
#synthetic_t

noise_levels = np.arange(0, 1, 0.025)
num_params = len(default_params)
curve_params = np.zeros((len(noise_levels), num_params))
SSRs = np.zeros(len(noise_levels))


# Number of iterations for variance estimate
num_iterations = 10

# Initialize storage matrices to hold results from each iteration
curve_params_iterations = np.zeros((num_iterations, len(noise_levels), num_params))
SSRs_iterations = np.zeros((num_iterations, len(noise_levels)))


#  iterations loop to estimate variance
for iter in range(num_iterations):
    # Loop through each noise level
    for ii, noise_level in enumerate(noise_levels):
        temp_curve_params = np.zeros((len(initial_params), num_params))
        x = fit_function(t, **default_params) + np.random.normal(0, noise_level, len(t))
        
        # Loop through each set of init  params
        for jj, init_param in enumerate(initial_params):
            p0 = init_param
            if fit_function.__name__ == 'sin_fit' or fit_function.__name__ == 'polysin':  
                f, fft_result, _ = perform_fft(t, x)
                top_peak_freqs = extract_peaks(f, fft_result)
                p0[-2] = top_peak_freqs[0] if top_peak_freqs.size > 0 else 0
            popt, ssr = param_estimator(t, x, fit_function, p0)
            ### BIG ASSUMPTION HERE, Do a graph of the correlation between the sign
            if fit_function.__name__ == 'sin_fit' or fit_function.__name__ == 'polysin':  
                popt[-1] = np.abs(popt[-1] )
                popt[-3] = np.abs(popt[-3] )
            temp_curve_params[jj, :] = popt
            if ii == 0 and iter == 0:  # only for the first noise level
                default_params_values = np.array(list(default_params.values()))  # convert default params to an array
                diff = np.abs(default_params_values - popt)  # calculate the absolute difference
                #print(f"Iteration {iter + 1}, init_param set {jj + 1}: fitParam: {popt}, Absolute difference between default and fitted params: {diff}")

        # Store mean params for noise level and its
        curve_params_iterations[iter, ii, :] = np.mean(temp_curve_params, axis=0)
        SSRs_iterations[iter, ii] = np.mean(ssr)

# mean across iterations
curve_params = np.mean(curve_params_iterations, axis=0)
curve_params_std = np.std(curve_params_iterations, axis=0)

SSRs = np.mean(SSRs_iterations, axis=0)

if model == 'sin_fit' or model == 'polysin':
    # For models with phase parameter
    delta_params_non_phase = np.abs(curve_params[:, :-1] - np.array(list(default_params.values()))[:-1])
    delta_Phi = np.arccos(np.cos(curve_params[:, -1] - default_params['Phi']))
    delta_params = np.column_stack((delta_params_non_phase, delta_Phi))
else:
    # For models without a phase parameter
    delta_params = np.abs(curve_params - np.array(list(default_params.values())))

fig, axs = plt.subplots(num_params, 1, figsize=(10, 5 * num_params))

param_names = list(default_params.keys())
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown']
SSRs_std = np.std(SSRs_iterations, axis=0)

threshold_met = {}

for i in range(num_params):
    axs[i].set_xlabel('Noise Level', fontsize=20)
    axs[i].set_ylabel(f"Delta_{param_names[i]}", color=colors[i], fontsize=24)

    # Plot the mean line
    axs[i].plot(noise_levels, delta_params[:, i], color=colors[i], linestyle='--', alpha=0.8)

    # Add a red line at 50% of the default parameter value
    threshold = np.abs(0.5 * default_params[param_names[i]])
    axs[i].axhline(y=threshold, color='red', linewidth=2)
    # Fill between mean +/- standard deviation, lower limit at 0
    axs[i].fill_between(noise_levels, delta_params[:, i], delta_params[:, i] + curve_params_std[:, i], color=colors[i], alpha=0.2)
    axs[i].tick_params(axis='both', labelsize=18)

        # Find the noise level at which the delta param crosses the threshold
    for noise_level, delta_param_value in zip(noise_levels, delta_params[:, i]):
        if delta_param_value >= threshold:
            threshold_met[param_names[i]] = noise_level
            break

#plt.suptitle(f"{model}: Difference in fitting and original parameters, and SSRs at different Noise Levels")
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
print("Noise levels at which each parameter crosses the threshold:", threshold_met)


#%% Plotting the SSRs of differents models
models = {
    "sin_fit": sin_fit,
    "polynomial": polynomial,
    "ACA_fit": ACA_fit,
    "polysin": polysin
}

SSRs_all_models = {}
from scipy.stats import ttest_1samp
import random


# Number of permutations
num_permutations = 1000

# Array to store permutation test statistics
permutation_stats = np.zeros((len(noise_levels), num_permutations))

# Loop through all the models
for model_name, fit_function in models.items():

    # Using the dict, init the original params, and the loop of the diff p0 params
    default_params = {key: np.mean(val) for key, val in param_steps[model_name].items()}
    initial_params_steps = [val for val in param_steps[model_name].values()]
    initial_params = np.stack(initial_params_steps, axis=1)
    # Gen other prametrs
    t = np.linspace(0.4, 0.9, 1000)
    noise_levels = np.arange(0, 1, 0.025)
    num_params = len(default_params)
    curve_params = np.zeros((len(noise_levels), num_params))
    SSRs = np.zeros(len(noise_levels))
    # Number of iterations for variance estimate
    num_iterations = 10

    # Initialize storage matrices to hold results from each iteration
    curve_params_iterations = np.zeros((num_iterations, len(noise_levels), num_params))
    SSRs_iterations = np.zeros((num_iterations, len(noise_levels)))

    #  iterations loop to estimate variance
    for iter in range(num_iterations):
        # Loop through each noise level
        for ii, noise_level in enumerate(noise_levels):
            temp_curve_params = np.zeros((len(initial_params), num_params))
            temp_ssr = np.zeros((len(initial_params), 1))
            x = fit_function(t, **default_params) + np.random.normal(0, noise_level, len(t))
            

            #perform permutations
            # Inside the iteration loop, after fitting the model with original data
            if iter == 0:  # Perform permutation test in the first iteration only
                for perm in range(num_permutations):
                    # Shuffle the labels
                    np.random.shuffle(x)

                    # Fit the model and calculate test statistic for shuffled data
                    shuffled_stats = []
                    for jj, init_param in enumerate(initial_params):
                        p0 = init_param
                        # Fit the model with shuffled data
                        shuffled_popt, _ = param_estimator(t, x, fit_function, p0)
                        
                        # Adjust default_params_values to match the length of shuffled_popt
                        if len(shuffled_popt) != len(default_params_values):
                            adjusted_default_params = np.resize(default_params_values, len(shuffled_popt))
                        else:
                            adjusted_default_params = default_params_values

                        # Use adjusted_default_params for the comparison
                        shuffled_diff = np.abs(adjusted_default_params - shuffled_popt)
                        shuffled_stats.append(shuffled_diff)

                    # Store the mean test statistic for this permutation
                    permutation_stats[ii, perm] = np.mean(shuffled_stats, axis=0)
# Loop through each set of init  params
            for jj, init_param in enumerate(initial_params):
                p0 = init_param
                if fit_function.__name__ == 'sin_fit' or fit_function.__name__ == 'polysin':  
                    f, fft_result, _ = perform_fft(t, x)
                    top_peak_freqs = extract_peaks(f, fft_result)
                    p0[-2] = top_peak_freqs[0] if top_peak_freqs.size > 0 else 0
                #print('p0 size:',len(popt))
                popt, ssr = param_estimator(t, x, fit_function, p0)
                ### BIG ASSUMPTION HERE, Do a graph of the correlation between the sign
                if fit_function.__name__ == 'sin_fit' or fit_function.__name__ == 'polysin':  
                    popt[-1] = np.abs(popt[-1] )
                    popt[-3] = np.abs(popt[-3] )
                temp_curve_params[jj, :] = popt
                temp_ssr[jj] = ssr
                if ii == 0 and iter == 0:  # only for the first noise level
                    default_params_values = np.array(list(default_params.values()))  # convert default params to an array
                    diff = np.abs(default_params_values - popt)  # calculate the absolute difference
                    #print(f"Iteration {iter + 1}, init_param set {jj + 1}: fitParam: {popt}, Absolute difference between default and fitted params: {diff}")

            # Store mean param s fornoise level and its
            curve_params_iterations[iter, ii, :] = np.mean(temp_curve_params, axis=0)
            SSRs_iterations[iter, ii] = np.mean(temp_ssr)


    # Store the SSRs for the current model
    SSRs_all_models[model_name] = np.mean(SSRs_iterations, axis=0)

# Now, outside the loop, plot the SSRs for all the models
plt.figure(figsize=(10, 6))
for model_name, SSRs in SSRs_all_models.items():
    plt.plot(noise_levels, SSRs, label=model_name, alpha=0.7)

    # Fill between mean +/- standard deviation
    plt.fill_between(noise_levels, SSRs, SSRs + np.std(SSRs_iterations, axis=0), alpha=0.2)

plt.xlabel('Noise Level',fontsize = 20)
plt.ylabel('Mean SSR',fontsize = 20)
plt.tick_params(axis='both', labelsize=18)
#plt.title('SSR variation with Noise Level for different models')
plt.legend(fontsize=20)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


#perform the TTEST perms
# After the loop, perform the one-sided t-test for each noise level
p_values = np.zeros(len(noise_levels))
for ii in range(len(noise_levels)):
    observed_stat = np.mean(curve_params_iterations[:, ii, :], axis=0)  # Observed statistic
    p_values[ii] = ttest_1samp(permutation_stats[ii, :], observed_stat, alternative='greater').pvalue

#%%
def param_estimator(t, x, fit_function, p0, bounds=(-np.inf, np.inf)):
    # Ensure that the initial guess (p0) matches the number of parameters expected by the fit function
    p0 = p0[:fit_function.__code__.co_argcount - 1]  # Adjusting p0 length
    popt, _ = curve_fit(fit_function, t, x, p0=p0, bounds=bounds, maxfev=50000)
    ssr = np.sum((x - fit_function(t, *popt)) ** 2)
    return popt, ssr

def perform_permutation_test(t, x_true, default_params, fit_function, initial_params, num_permutations=1000):
    num_params = len(default_params)
    print("Initial parameters:", initial_params)
    true_params, _ = param_estimator(t, x_true, fit_function, initial_params)
    true_delta_params = np.abs(true_params - np.array(list(default_params.values())))

    perm_delta_params = np.zeros((num_permutations, num_params))

    for i in range(num_permutations):
        permuted_x = np.random.permutation(x_true)
        perm_params, _ = param_estimator(t, permuted_x, fit_function, initial_params)
        perm_delta_params[i, :] = np.abs(perm_params - np.array(list(default_params.values())))

    t_stats = np.zeros(num_params)
    p_values = np.zeros(num_params)

    for j in range(num_params):
        t_stat, p_value = ttest_1samp(perm_delta_params[:, j], true_delta_params[j], alternative='greater')
        t_stats[j] = t_stat
        p_values[j] = p_value

    return t_stats, p_values

# Example usage
t_stats, p_values = perform_permutation_test(t, x, default_params, polysin, initial_params, num_permutations=1000)

# Output the results
for param, t_stat, p_value in zip(default_params.keys(), t_stats, p_values):
    print(f"Parameter: {param}, t-statistic: {t_stat:.2f}, p-value: {p_value:.4f}")

# %% Benchmarking Noise levels

# Plotting data before the loops
plt.figure(figsize=(10, 6))
# Generate clean data based on the chosen fit function
clean_data = fit_function(t, **default_params)
plt.plot(t, clean_data+0.35, label='Clean Data (0 noise)', color='black', linewidth=3)

# Initialize the colormap
colormap = plt.cm.get_cmap('viridis', len(np.arange(0.1, 0.7, 0.2)))

# Loop through noise levels from 0.1 to 1
for i, noise_level in enumerate(np.arange(0.1, 0.7, 0.2)):
    noisy_data = clean_data+0.35 + np.random.normal(0, noise_level, len(t))
    plt.scatter(t, noisy_data, color=colormap(i), alpha=0.2, label=f'Noise level {noise_level:.2f}')
model = 'PolySin'
# Labels and legend
plt.xlabel('Distractor-Target interval (s)', fontsize=18)
plt.ylabel('Reaction Time', fontsize=18)
#plt.title(f'Data Visualization for the {model} Model')

# Create a custom legend for the noise levels
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Noise level {x:.2f}', markersize=5, markerfacecolor=colormap(i)) 
                   for i, x in enumerate(np.arange(0.1, 0.7, 0.2))]
plt.legend(handles=legend_elements, loc='upper right', fontsize=18)

plt.show()

# %% PLOT OF THE 4 MODELS AS AN EXAMPLE FIGURE
# Time and Reaction Time data, modify this according to your needs
import numpy as np
import matplotlib.pyplot as plt
t = 1- np.linspace(0.4, 0.9, 100)
RT = np.zeros_like(t)  # replace this with actual RT centered around 0

# Parameter steps
param_steps = {
    'polynomial': {'a': -1, 'b': 0.3},
    'sin_fit': {'Amp': 0.25, 'freq': 8, 'Phi': np.pi},
    'polysin': {'a': -1, 'b': 0.05, 'Amp': 0.25, 'freq': 8, 'Phi': np.pi},
    'ACA_fit': {'capture_amplitude': 1, 'capture_decay': 0.2, 'arousal_amplitude': 0.15, 'arousal_duration':.7}
}

for model in ['polynomial', 'sin_fit', 'polysin', 'ACA_fit']:
    # Choose default parameters for each model
    default_params = param_steps[model]
    
    # Initialize the function to be used
    if model == 'polynomial':
        fit_function = polynomial
    elif model == 'sin_fit':
        fit_function = sin_fit
    elif model == 'polysin':
        fit_function = polysin
    elif model == 'ACA_fit':
        fit_function = ACA_fit
    
    # Compute curve
    curve = fit_function(t, **default_params)
    
    # Plot
    plt.figure()
    plt.title(f"Model: {model}",fontsize=18)
    plt.plot(t, curve, 'k', label="Model Curve")
    plt.plot(t, RT, label=" mean RT ", color='orange', linestyle='--')
    plt.xlabel("Distractor-Target Interval",fontsize=16)
    plt.ylabel("Reaction Times (normalized)",fontsize=16)
    plt.ylim(-1,1)
    # Annotate plot with parameters
    #textstr = '\n'.join([f"{key}: {val}" for key, val in default_params.items()])
    #plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.legend()
    plt.show()

# %%
#load the real da