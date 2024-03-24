
# #%% import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_model_comparison(models, aic_values, bic_values):
#     # Number of models
#     num_models = len(models)

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot AIC and BIC values
#     ax.plot(models, aic_values, marker='o', linestyle='-', color='blue', label='AIC')
#     ax.plot(models, bic_values, marker='o', linestyle='-', color='green', label='BIC')

#     # Add value labels
#     for i in range(num_models):
#         ax.text(i, aic_values[i], f'{aic_values[i]:.2f}', ha='center', va='bottom', color='blue')
#         ax.text(i, bic_values[i], f'{bic_values[i]:.2f}', ha='center', va='top', color='green')

#     # Set labels
#     ax.set_xlabel('Model')
#     ax.set_ylabel('Information Criterion Value')

#     # Add legend
#     ax.legend()
#     plt.title('CAT1 Model Comparison')
#     # Display the plot
#     plt.show()

# from scipy import stats

# def plot_model_comparison_subs(models, aic_values, bic_values,mad_aic,mad_bic):
#     # Number of models
#     num_models = len(models)

#     # Create figure and axis
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Plot AIC and BIC values + error bar
#     ax.errorbar(models, aic_values, yerr=np.abs(mad_aic), fmt='o', color='blue', label='AIC')
#     ax.errorbar(models, bic_values, yerr=np.abs(mad_bic), fmt='o', color='green', label='BIC')

#     # Add lines  
#     ax.plot(models, aic_values, color='blue', alpha=0.5)
#     ax.plot(models, bic_values, color='green', alpha=0.5)
#     # Add value labels
#     for i in range(num_models):
#         ax.text(i, aic_values[i], f'{aic_values[i]:.2f}', ha='center', va='bottom', color='blue')
#         ax.text(i, bic_values[i], f'{bic_values[i]:.2f}', ha='center', va='top', color='green')

#     # Set labels
#     ax.set_xlabel('Model')
#     ax.set_ylabel('Information Criterion Value')

#     # Add legend
#     ax.legend()
#     plt.title('CAT2 Model Comparison')

#     # Display the plot
#     plt.show()

# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Define models and their AIC and BIC values
# models_young = ['Sin', 'Polysin', 'Polynomial', 'ACA']
# aic_values_young = [479.00, 442.23, 445.04, 439.51]
# bic_values_young = [488.16, 448.38, 460.41, 451.80]
# mad_aic_young = [31.24, 36.83, 36.77, 48.89]
# mad_bic_young = [31.27, 36.83, 36.77, 48.89]
# mean_ssr_young = [192.10, 150.30, 147.29, 147.87]
# mad_ssr_young = [34.44, 34.95, 34.10, 45.75]
# # Create a DataFrame from the above lists
# df_young = pd.DataFrame({
#     'Model': models_young,
#     'AIC': aic_values_young,
#     'BIC': bic_values_young,
#     'MAD_AIC': mad_aic_young,
#     'MAD_BIC': mad_bic_young,
#     'SSR': mean_ssr_young,
#     'MAD_SSR': mad_ssr_young
# })
# models_old = ['Sin', 'Polysin', 'Polynomial', 'ACA']
# aic_values_old = [479.00, 445.60, 442.74, 441.04]
# bic_values_old = [488.16, 460.96, 448.89, 453.34]
# mad_aic_old = [31.24, 49.00, 49.07, 49.46]
# mad_bic_old = [31.27, 49.00, 49.07, 49.46]
# mean_ssr_old = [192.10, 151.89, 154.97, 149.57]
# mad_ssr_old = [34.44, 47.28, 48.39, 46.94]

# # Create a DataFrame from the above lists
# df_old = pd.DataFrame({
#     'Model': models_old,
#     'AIC': aic_values_old,
#     'BIC': bic_values_old,
#     'MAD_AIC': mad_aic_old,
#     'MAD_BIC': mad_bic_old,
#     'SSR': mean_ssr_old,
#     'MAD_SSR': mad_ssr_old
# })

# #select the model and values
# columns = ['Model', 'AIC', 'BIC', 'MAD_AIC', 'MAD_BIC', 'SSR', 'MAD_SSR']
# models, aic_values, bic_values, mad_aic, mad_bic, mean_ssr, mad_ssr = [df_young[col].tolist() for col in columns]
# # Configuration
# ind = np.arange(len(models))
# width = 0.25

# # Find index of minimum values
# min_aic_idx = np.argmin(aic_values)
# min_bic_idx = np.argmin(bic_values)
# min_ssr_idx = np.argmin(mean_ssr)

# # violin plots comparing the AIC, BIC, and SSR metrics across the four model
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Create subplots for each metric
# fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# # Violin plot for AIC
# sns.violinplot(x='Model', y='AIC', data=df_young, ax=axs[0], palette="muted")
# axs[0].set_title('AIC values across models')
# axs[0].hlines(df_young['AIC'].min(), xmin=-0.5, xmax=len(df_young['Model']) - 0.5, colors='blue', linestyles='dotted')

# # Violin plot for BIC
# sns.violinplot(x='Model', y='BIC', data=df_young, ax=axs[1], palette="muted")
# axs[1].set_title('BIC values across models')
# axs[1].hlines(df_young['BIC'].min(), xmin=-0.5, xmax=len(df_young['Model']) - 0.5, colors='green', linestyles='dotted')

# # Violin plot for SSR
# sns.violinplot(x='Model', y='SSR', data=df_young, ax=axs[2], palette="muted")
# axs[2].set_title('SSR values across models')
# axs[2].hlines(df_young['SSR'].min(), xmin=-0.5, xmax=len(df_young['Model']) - 0.5, colors='red', linestyles='dotted')

# plt.tight_layout()
# plt.show()

# %% PLOT TO COMPARE YOUNG AND OLD
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

 
# # Extract BIC values for ACA model
# bic_young_aca = df_young[df_young['Model'] == 'ACA']['AIC'].values[0]
# mad_bic_young_aca = df_young[df_young['Model'] == 'ACA']['MAD_AIC'].values[0]

# bic_old_aca = df_old[df_old['Model'] == 'ACA']['AIC'].values[0]
# mad_bic_old_aca = df_old[df_old['Model'] == 'ACA']['MAD_AIC'].values[0]

# # Plotting
# fig, ax = plt.subplots(figsize=(8, 8))

# # Adjust y limits for more margin
# ax.set_ylim(min(bic_young_aca, bic_old_aca) - 10, max(bic_young_aca, bic_old_aca) + 10)
# ax.set_xlim(-0.2, 1 + 0.2)

# # Plot BIC values for Young and Old with error bars
# ax.errorbar(['Young'], bic_young_aca, yerr=mad_bic_young_aca/10, color='black', marker='x', capsize=10, label='Young BIC')
# ax.errorbar(['Old'], bic_old_aca, yerr=mad_bic_old_aca/10, color='black', marker='x', capsize=10, label='Old BIC')

# # Connect the points
# ax.plot(['Young', 'Old'], [bic_young_aca, bic_old_aca], color='grey', linestyle='--')

# # Labels and Legends
# ax.set_ylabel('BIC Values')
# ax.set_title('Comparison of AIC values for ACA Model: Young vs. Old')
# ax.legend()
# plt.tight_layout()
# plt.show()




# %% All metrics of all datasets
import pandas as pd

# List of models
model_names = ["sin", "poly", "polysin", "ACA"]

# Dictionary to store the results
results_data = {}

# Loop through each model and import the corresponding files
for model in model_names:
    old_filename = f"old_results_{model}.csv"
    young_filename = f"young_results_{model}.csv"
    
    # Import the data
    old_data = pd.read_csv(old_filename)
    young_data = pd.read_csv(young_filename)
    
    # Store in the dictionary
    results_data[model] = {
        "old": old_data,
        "young": young_data
    }

# Now, the results_data dictionary has the loaded data. 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#%% List of models



# Create an aggregated DataFrame for violin plot
# Create an aggregated DataFrame for violin plot
agg_data = []

# Extracting young data for each model and metric and aggregating
for model in model_names:
    for metric in ["AIC", "BIC", "SSR"]:
        data_subset = results_data[model]["young"][metric]
        for value in data_subset:
            agg_data.append([model, metric, value])

# Convert to DataFrame
df_agg = pd.DataFrame(agg_data, columns=["Model", "Metric", "Value"])

# Create a dictionary to store the fake p-values for each metric
# Create a dictionary of p-values for all pairwise comparisons
metric_pairwise_pvalues = {
    "AIC": {
        ("ACA", "poly"): 0.173,
        ("ACA", "polysin"): 0.010,
        ("ACA", "sin"):  .001,   # Given that p < .001, this value is approximate
        ("poly", "polysin"): 0.162,
        ("poly", "sin"): .001,  # Given that p < .001, this value is approximate
        ("polysin", "sin"): 0.015,
    },
    "BIC": {
        ("ACA", "poly"): 0.030,
        ("ACA", "polysin"): .001,  # Given that p < .001, this value is approximate
        ("ACA", "sin"): .001,  # Given that p < .001, this value is approximate
        ("poly", "polysin"): .001,  # Given that p < .001, this value is approximate
        ("poly", "sin"): .001,  # Given that p < .001, this value is approximate
        ("polysin", "sin"): 0.337,
    },
    "SSR": {
        ("ACA", "poly"): .001,  # Given that p < .001, this value is approximate
        ("ACA", "polysin"): 0.077,
        ("ACA", "sin"): .001,  # Given that p < .001, this value is approximate
        ("poly", "polysin"): 0.077,
        ("poly", "sin"): 0.005,
        ("polysin", "sin"): .001,  #

    },
}
# Create the violin plot
plt.figure(figsize=(15, 20))

# Define some custom settings for visibility
medianprops = dict(linestyle='-', linewidth=10, color='navy')  # Properties for the median line
boxprops = dict(linestyle='-', linewidth=10, color='darkorange')  # Properties for the box (quartiles)

# Set default font sizes
plt.rcParams.update({'axes.titlesize': 30, 'axes.labelsize': 30,
                     'xtick.labelsize': 30, 'ytick.labelsize': 30,
                     'legend.fontsize': 30, 'font.size': 30})

# Set default font sizes
plt.rcParams['axes.titlesize'] = 30  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size
plt.rcParams['font.size'] = 30        # Default font size for text
metrics_name = ['AIC', 'BIC', 'SSR']
for i, metric in enumerate(["AIC", "BIC", "SSR"], 1):
    plt.subplot(3, 1, i)
    metric_data = df_agg[df_agg["Metric"] == metric]
    means = metric_data.groupby("Model")["Value"].mean()
    palette = {model: "orange" if model == means.idxmin() else "lightgray" for model in model_names}

    # Draw the violin plot with custom median and box properties
    sns.violinplot(x="Model", y="Value", data=metric_data, palette=palette,
                   medianprops=medianprops, boxprops=boxprops, linewidth = 4)

    plt.title(f"{metric}", fontsize=37)

    # Determine the maximum y-value across all the data points for each metric for an initial offset
    max_y = metric_data["Value"].max() + 25  # Adding an arbitrary offset to start

    # Create an incremental offset for each significant line
    offset_increment = 25

    # Add horizontal lines for each pairwise comparison
    for (model_name, other_model_name), p_value in metric_pairwise_pvalues[metric].items():
        if p_value < 0.05:
            plt.hlines(
                max_y,
                xmin=model_names.index(model_name) - 0.25,
                xmax=model_names.index(other_model_name) + 0.25,
                color="black",
                linewidth=1,
            )

            # Adjust the y-value for the significance marker based on p-value
            marker_y = max_y + (5 if p_value <= 0.001 else 3)

            if p_value <= 0.001:
                plt.text((model_names.index(model_name) + model_names.index(other_model_name)) / 2, marker_y, '**', ha='center', va='bottom', color='black',fontsize=18)
            else:
                plt.text((model_names.index(model_name) + model_names.index(other_model_name)) / 2, marker_y, '*', ha='center', va='bottom', color='black',fontsize=18)

            # Increase the max_y for the next line to avoid overlap
            max_y += offset_increment

plt.tight_layout()
plt.show()
# %%
# Define a function to calculate the desired statistics for each metric
def calculate_statistics(data):
    return {
        'Mean': data.mean(),
        'Std. Dev.': data.std(),
        'Min': data.min(),
        'Max': data.max()
    }

# Dictionary to store statistics
statistics_data = {}

# Loop through all models and datasets
for model in model_names:
    statistics_data[model] = {}
    for dataset in ['old', 'young']:
        statistics_data[model][dataset] = {}
        for metric in ["AIC", "BIC", "SSR"]:
            statistics_data[model][dataset][metric] = calculate_statistics(results_data[model][dataset][metric])

# Convert nested dictionary to multi-index DataFrame
df_statistics = pd.DataFrame.from_dict({(i, j, k): statistics_data[i][j][k] 
                                        for i in statistics_data.keys() 
                                        for j in statistics_data[i].keys()
                                        for k in statistics_data[i][j].keys()},
                                       orient='index')

# Display the DataFrame
print(df_statistics)

# If you want to save this DataFrame as a CSV:
# df_statistics.to_csv("statistics_table.csv")

# %% DIFFERENCE IN MEANS OF AIC
from scipy.stats import ttest_rel
import numpy as np

# Model of interest
model_of_interest = "ACA"

# Collect metrics to assess
metrics = ["AIC", "BIC", "SSR"]

# Store results
comparison_results = {}

# Loop through each metric
for metric in metrics:
    comparison_results[metric] = {}
    
    # Extract the values for the model of interest
    model_values = results_data[model_of_interest]["young"][metric]
    
    # Loop through all other models
    for model in model_names:
        if model != model_of_interest:
            
            # Extract the values for the current model
            comparison_values = results_data[model]["young"][metric]
            
            # Perform paired t-test
            t_stat, p_value = ttest_rel(model_values, comparison_values)
            
            # Store the results
            comparison_results[metric][model] = {
                't-statistic': t_stat,
                'p-value': p_value,
                'Difference in means': np.mean(model_values) - np.mean(comparison_values)
            }

# Convert results to a DataFrame for easy visualization
df_comparison = pd.DataFrame.from_dict({(i, j): comparison_results[i][j] 
                                        for i in comparison_results.keys() 
                                        for j in comparison_results[i].keys()},
                                       orient='index')

print(df_comparison)


# %% DATASET COMPARISON

# Extract ACA data for both datasets
old_aca_data = results_data["poly"]["old"]
young_aca_data = results_data["poly"]["young"]

# Calculate statistics for ACA model in both datasets
stats_old = old_aca_data.describe()
stats_young = young_aca_data.describe()

# Calculate the difference between young and old datasets
diff_stats = stats_young - stats_old

# Filter to include only mean, standard deviation, and other desired statistics
filtered_stats_old = stats_old.loc[['mean', 'std', '25%', '50%', '75%']]
filtered_stats_young = stats_young.loc[['mean', 'std', '25%', '50%', '75%']]
filtered_diff_stats = diff_stats.loc[['mean', 'std', '25%', '50%', '75%']]

# Display the tables
print("Statistics for Old Dataset (ACA Model):")
print(filtered_stats_old)
print("\nStatistics for Young Dataset (ACA Model):")
print(filtered_stats_young)
print("\nDifferences between Young and Old Datasets (ACA Model):")
print(filtered_diff_stats)

# %% ANOVA
# List of models
model_names = ["sin", "poly", "polysin", "ACA"]

# Dictionary to store the results
results_data = {}

# Loop through each model and import the corresponding files
for model in model_names:
    old_filename = f"old_results_{model}.csv"
    young_filename = f"young_results_{model}.csv"
    
    # Import the data
    old_data = pd.read_csv(old_filename)
    young_data = pd.read_csv(young_filename)
    
    # Store in the dictionary
    results_data[model] = {
        "old": old_data,
        "young": young_data
    }

# Create an aggregated DataFrame for violin plot
agg_data = []

# Extracting young data for each model and metric and aggregating
for model in model_names:
    for idx, metric in enumerate(["AIC", "BIC", "SSR"]):
        data_subset = results_data[model]["young"][metric]
        for subj_idx, value in enumerate(data_subset):
            agg_data.append([model, metric, value, subj_idx + 1])  # We use subj_idx + 1 to start subjects from 1

# Convert to DataFrame
df_agg = pd.DataFrame(agg_data, columns=["Model", "Metric", "Value", "Subject"])

#%% APPLY ANOVA

import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
# Filter the DataFrame for rows where Metric is 'AIC'
df_aic = df_agg[df_agg['Metric'] == 'SSR']
# Collect results in a list
results = []

# Conduct rANOVA

# Print results
#print(res)
# %%
# Importing the Pandas library
import pandas as pd

# List of metrics
metrics = ["AIC", "BIC", "SSR"]
model_names = ["sin", "poly", "polysin", "ACA"]

# Loop through each metric
for metric in metrics:
    agg_data = []
    
    # Extract data for each model
    for model in model_names:
        # Data for "old" group
        old_data = results_data[model]['old'][metric].tolist()
        # Data for "young" group
        young_data = results_data[model]['young'][metric].tolist()
        
        # Aggregate the data
        for i, val in enumerate(old_data):
            agg_data.append(['old', i+1, model, val])
        for i, val in enumerate(young_data):
            agg_data.append(['young', i+1, model, val])
    
    # Convert the aggregated data to a DataFrame
    df = pd.DataFrame(agg_data, columns=['Dataset', 'Subject', 'Model', metric])
    
    # Pivot the dataframe to have models as columns
    df_pivot = df.pivot_table(index=['Dataset', 'Subject'], columns='Model', values=metric).reset_index()
    
    # Assign the DataFrame to a specific variable based on the metric
    if metric == 'AIC':
        df_AIC = df_pivot
    elif metric == 'BIC':
        df_BIC = df_pivot
    elif metric == 'SSR':
        df_SSR = df_pivot

# Display the first few rows of each DataFrame
print("AIC DataFrame:")
print(df_AIC.head())

print("\nBIC DataFrame:")
print(df_BIC.head())

print("\nSSR DataFrame:")
print(df_SSR.head())

#%%
# Conduct rANOVA
df_aic = dfs["AIC"]
aovrm = AnovaRM(df_aic, depvar="Value", subject="Subject", within=["Model"])
res = aovrm.fit()

#%%
