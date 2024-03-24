#%% Import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.signal as signal
import seaborn as sns

def plot_SUBandFit(t, x, bin_t, bin_x, params, param_names, fitfunction, subject_num,ysize):
    """
    Plot each polynomial fit for each peaks in curve params.

    """
    t = 1-t
    bin_t = 1 - bin_t
    tmin = np.min(t)
    tmax = np.max(t)
    plt.plot(t, x, 'o', alpha=0.5, color='lightblue',label='trials')
    plt.plot(bin_t, bin_x, '-k', alpha=0.5,label='binned')
    popt = params[:-1]
    param_text = construct_text(params, param_names)
    plt.plot(bin_t, fitfunction(bin_t, *popt), '-r', alpha=0.2,label='fit')
    plt.text(0.8, 0.1, param_text , transform=plt.gca().transAxes)
    plt.xlabel('Distractor-Target Interval ')
    plt.ylabel('Reaction Times (normalized)')
    plt.ylim(ysize)
    plt.xlim(tmax, tmin) 
    plt.title(f'Subject {subject_num}')
    plt.legend()
    plt.show()

def construct_text(params, param_names):
    """ construct appropriate text for each names:
        param_names = ["a", "b", "Amp", "Freq", "Phase"]
        popt_text = construct_text(popt, param_names)
    """
    text = ""
    for i, name in enumerate(param_names):
        text += f"{name}: {params[i]:.2f}\n"
    return text

import numpy as np
import matplotlib.pyplot as plt

# def plot_polar_histogram(phase):
#     """
#     Polar histogram of the Phi (phase) of the sine.
#     """
#     phase = phase % (2 * np.pi)  # Ensure phase is between 0 and 2*pi
#     hist, bin_edges = np.histogram(phase, 8)  # Calculate histogram with bin edges
#     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Find bin centers

#     ax = plt.subplot(111, polar=True)
#     ax.bar(bin_centers, hist, width=(2 * np.pi) / 8, bottom=0, alpha=0.7)

#     # Calculate and plot the mean phase
#     mean_phase = np.arctan2(np.sin(phase).mean(), np.cos(phase).mean())
    
#     # Calculate the resultant vector length (R)
#     R = np.sqrt((np.cos(phase).sum())**2 + (np.sin(phase).sum())**2) / len(phase)
    
#     # Calculate circular standard deviation
#     circular_sd = np.sqrt(-2 * np.log(R))
    
#     # Convert to degrees for easier interpretation
#     mean_phase_deg = np.rad2deg(mean_phase)
#     circular_sd_deg = np.rad2deg(circular_sd)
    
#     ax.bar(mean_phase, np.max(hist)/2, width=(2*np.pi)/32, color='r', alpha=0.8, 
#            label=f"Mean: {mean_phase_deg:.2f}°, SD: {circular_sd_deg:.2f}°")  # Plot mean phase bar

#     plt.legend()
#     #plt.title('Polar Histogram of the Phase Across Participants')
#     plt.show()


def plot_polar_histogram(phase):
    # Wrap the phase within the range [0, 2π)
    phase = phase % (2 * np.pi)

    # Adjust phase values: [0, π] stay the same, [π, 2π) mapped to [-π, 0)
    adjusted_phase = np.where(phase <= np.pi, phase, phase - 2 * np.pi)

    # Histogram calculation
    hist, bin_edges = np.histogram(adjusted_phase, 8)  # 8 bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create polar plot
    ax = plt.subplot(111, polar=True)
    ax.bar(bin_centers, hist, width=(2 * np.pi) / 8, bottom=0, alpha=0.7)

    # Calculate the median of the adjusted data
    median_phase_adjusted = np.percentile(adjusted_phase, 50)

    # Convert the median back to a [0, 2π) range and then to degrees
    median_phase = (median_phase_adjusted + 2 * np.pi) % (2 * np.pi)
    median_phase_deg = np.rad2deg(median_phase)


    # Calculate circular standard deviation
    R = np.sqrt((np.cos(phase).sum())**2 + (np.sin(phase).sum())**2) / len(phase)
    circular_sd = np.sqrt(-2 * np.log(R))
    circular_sd_deg = np.rad2deg(circular_sd)

    # Plot the median as a bar
    ax.bar(median_phase, np.max(hist)/2, width=(2*np.pi)/32, color='r', alpha=0.8, 
           label=f"Median: {median_phase_deg:.2f}°, SD: {circular_sd_deg:.2f}°")  # Plot mean phase bar

    # Set plot properties
    ax.set_theta_zero_location('N')  # Zero at the top
    ax.grid(True)
    ax.set_yticklabels([])

    plt.legend(loc='upper right')
    plt.title('Polar Histogram of the Phase Across Participants')

    # Display the plot
    plt.show()



# def plot_box_swarm(values, parameters, colors, percentile_range=(10, 90), swarm_color='black', swarm_size=4):
#     # Check number of parameters and colors
#     if values.shape[1] != len(parameters) or len(parameters) != len(colors):
#         raise ValueError("Mismatch between the number of parameters and the shape of values or colors.")

#     # Check number of parameters and colors
#     if values.shape[1] != len(parameters) or len(parameters) != len(colors):
#         raise ValueError("Mismatch between the number of parameters and the shape of values or colors.")

#     # Get the outlier bounds
#     prctile_lb, prctile_ub = (10, 90)
#     outlier_lb = np.percentile(values, prctile_lb, axis=0)
#     outlier_ub = np.percentile(values, prctile_ub, axis=0)

#     # Filter out the outliers
#     values_filt = values[np.all(values >= outlier_lb, axis=1) & np.all(values <= outlier_ub, axis=1)]

#     num_params = len(parameters)
#     num_cols = 2  
#     num_rows = num_params // num_cols
#     if num_params % num_cols:
#         num_rows += 1

#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
#     axes = axes.flatten() if num_params > 1 else [axes]

#     for i, ax in enumerate(axes[:num_params]):
#         sns.boxplot(x=values_filt[:, i], ax=ax, color=colors[i], showfliers=False)  
        
#         ax.text(0.5, 0.9, 'mean: {:.2f}'.format(np.mean(values[:,i])), transform=ax.transAxes, fontsize=18)
        
#         ax.text(0.5, 0.77, 'SD: {:.2f}'.format(np.std(values[:,i])), transform=ax.transAxes, fontsize=18)
#         ax.set_xlabel(parameters[i], fontsize=22)  
        
#         sns.swarmplot(x=values_filt[:, i], ax=ax, color=swarm_color, size=swarm_size, alpha=0.7)

#     for ax in axes[num_params:]:
#         ax.remove()  # remove any unused subplots

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     #plt.suptitle('Box Plot and Beeswarm of Parameters', fontsize=16)
#     plt.show()



def plot_box_swarm(values, parameters, colors, percentile_range=(5, 95), swarm_color='black', swarm_size=5):
    # Check number of parameters and colors
    if values.shape[1] != len(parameters) or len(parameters) != len(colors):
        raise ValueError("Mismatch between the number of parameters and the shape of values or colors.")

    # Get the outlier bounds
    prctile_lb, prctile_ub = percentile_range
    outlier_lb = np.percentile(values, prctile_lb, axis=0)
    outlier_ub = np.percentile(values, prctile_ub, axis=0)

    # Filter out the outliers
    values_filt = values[np.all(values >= outlier_lb, axis=1) & np.all(values <= outlier_ub, axis=1)]

    num_params = len(parameters)
    num_cols = 2  # Adjust the number of columns based on your needs
    num_rows = num_params // num_cols
    if num_params % num_cols:
        num_rows += 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    axes = axes.flatten() if num_params > 1 else [axes]

    for i, ax in enumerate(axes[:num_params]):
        sns.boxplot(x=values_filt[:, i], ax=ax, color=colors[i], showfliers=True)  # Show outliers

        # Calculate and display the median instead of mean
        median = np.median(values[:, i])
        ax.text(0.5, 0.9, 'median: {:.2f}'.format(median), transform=ax.transAxes, fontsize=18)

        # Display SD
        std = np.std(values[:, i])
        ax.text(0.5, 0.77, 'SD: {:.2f}'.format(std), transform=ax.transAxes, fontsize=18)

        # Set x-label
        ax.set_xlabel(parameters[i], fontsize=22)

        # Display swarm plot
        sns.swarmplot(x=values_filt[:, i], ax=ax, color=swarm_color, size=swarm_size, alpha=0.7)

    # Remove any unused subplots
    for ax in axes[num_params:]:
        ax.remove()

    # Tight layout and adjust subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def mean_average_deviation(values):
    """
    Function to calculate the Mean Average Deviation (MAD) of an array.
    """
    mean_value = np.mean(values)
    mad = np.mean(np.abs(values - mean_value))
    return mad

def plot_aic_bic(fitaccSUBS, num_subjects):
    """
    Function to plot AIC and BIC values across subjects.
    """
    aic_values = fitaccSUBS[0, :]
    bic_values = fitaccSUBS[1, :]
    
    plt.figure(figsize=(10, 6))
    
    # Plot AIC and BIC values
    plt.plot(num_subjects, aic_values, color='blue', alpha=0.7, linewidth=2, label='AIC')
    plt.plot(num_subjects, bic_values, color='green', alpha=0.7, linewidth=2, label='BIC')

    # Adding labels and title
    plt.xlabel('Subjects')
    plt.ylabel('Values')
    plt.title('AIC and BIC values across subjects')

    # Add xticks
    plt.xticks(np.arange(min(num_subjects), max(num_subjects)+1, 1.0))

    # Add gridlines
    plt.grid(True, linestyle='--', alpha=0.6)
    # Add text annotations for mean AIC and BIC
    mean_aic = np.mean(aic_values)
    mean_bic = np.mean(bic_values)
    mad_aic = mean_average_deviation(aic_values)
    mad_bic = mean_average_deviation(bic_values)
    plt.text(0.02, 0.98, f'Mean AIC: {mean_aic:.2f}\nMean BIC: {mean_bic:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top')

    # Add legend
    plt.legend()
    plt.show()
    return mean_aic,mean_bic,mad_aic,mad_bic 

def plot_params_vs_error(param1,param2,title,param1title,param2title,color):
    from scipy.stats import   pearsonr

    """
    Look for corr. between two params
    """
    param1 =  param1 
    param2 =  param2
    #calc the corr
    r_corr, _ = pearsonr(param1, param2)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(param1, param2, 'x', color=color,alpha=0.9 )
    #plt.plot(, SSR_values, 'o', color='orange',alpha=0.7,label='SSR of param2')
    plt.xlabel(f"{param1title}")
    plt.ylabel(f"{param2title}")
    plt.title(f"{title} \nPearson Corr. Coeff: {r_corr:.2f}")   #\nPearson Corr. Coeff: {r_corr:.2f
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    #plt.xticks(np.arange(min(param1), max(param1)+1, 1.0))

    plt.show()

def plot_params_sin_vs_error(params):
    from scipy.stats import zscore, pearsonr

    """
    Look for corr. between SSR and Freq
    """
    SSR_values = params[:, -1]
    freq_values = zscore(params[:, -3])
    phase_values = zscore((params[:, -2])  % (2*np.pi))
    
    #calc the corr
    r_corr, _ = pearsonr(freq_values, phase_values)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_values, SSR_values, 'o', color='blue',alpha=0.7,label='SSR vs Frequency')
    plt.plot(phase_values, SSR_values, 'o', color='orange',alpha=0.7,label='SSR vs Phase')

    plt.xlabel('Zscore value of Freq and Phase')
    plt.ylabel('Sum of Squared Residuals (SSR)')
    plt.title(f'SSR vs Frequency and Phase \n'
              f'Pearson Corr. Coeff: {r_corr:.2f}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# def plot_phase_vs_error(params):
    # """
    # Look for corr. between SSR and Freq
    # """
    # SSR_values = params[:, -1]
    # freq_values = params[:, -3]  # Make sure freq is in the same order as the rows in params

    # # Create the plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(freq_values, SSR_values, 'o', color='black')
    # plt.plot(freq_values, SSR_values, 'o', color='black')

    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Sum of Squared Residuals (SSR)')
    # plt.title('SSR vs Frequency for each peak')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()

def plot_error(params,num_subjects):
    """
    Plot the curve of frequency versus SSR across all subjects and peaks.
    """
    y = params[:,-1].flatten()  # Flatten the SSR values
   
    # Sort the arrays based on increasing frequencies

    plt.plot(num_subjects, y, '-', color='black')
    plt.xlabel('Subject')
    plt.ylabel('SSR')
    plt.title('Error accros participants')
    plt.xticks(np.arange(min(num_subjects), max(num_subjects)+1, 1.0))
    plt.show()

import matplotlib.pyplot as plt
import numpy as np


# def plot_3d_surface(subjects, f, fft_results,title):
#     # f are the x-values of FFT (the frequencies)
#     # Create a new figure for the plot

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Create a grid of (subject, frequency) coordinates
#     subject_grid, frequency_grid = np.meshgrid(np.arange(len(subjects)), f)

#     # Plot the surface 3D
#     ax.plot_surface(subject_grid, frequency_grid, fft_results.T, cmap='Greens', alpha=0.7, edgecolor='none')

#     # Calculate the peak frequency for each FFT result
#     peak_indices = np.argmax(fft_results, axis=1)
#     peak_freqs = f[peak_indices]

#     # Plot these peak frequencies
#     ax.scatter(np.arange(len(subjects)), peak_freqs, fft_results.max(axis=1), color='red', s=50, label='Peak Frequencies')
#     surface = ax.plot_surface(subject_grid, frequency_grid, fft_results.T, cmap='Greens', alpha=0.7)

#     # Add labels
#     ax.set_xlabel('Subject')
#     ax.set_ylabel('Frequency')
#     ax.set_zlabel('Magnitude')
#     ax.set_title(title)

#     # Add tick labels for subjects
#     ax.set_xticks(np.arange(len(subjects)))
#     ax.set_xticklabels(subjects)

#     # Add a color bar
#     fig.colorbar(surface, ax=ax, pad=0.1, orientation='vertical', label='Magnitude')

#     # Set viewing angle for better visibility
#     ax.view_init(30, 45)

#     plt.show()

# Function to plot 3D surface
# Function to plot 3D surface
# def plot_3d_surface(subjects, f, fft_results, title):
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Correct the meshgrid variables: subject_grid must be first for the X-axis, frequency_grid for the Y-axis
#     subject_grid, frequency_grid = np.meshgrid(np.arange(len(subjects)), f)

#     # Make sure fft_results is the Z-axis data and correctly oriented
#     ax.plot_surface(subject_grid, frequency_grid, fft_results.T, cmap='Greens', alpha=0.8)


#     mean_freq = np.mean(peak_freqs)
#     print("Plotting band at mean frequency:", mean_freq)
#     # Correct the coordinates to create a surface for the band
#     x = np.array([[0, 0], [len(subjects) - 1, len(subjects) - 1]])
#     y = np.array([[mean_freq, mean_freq], [mean_freq, mean_freq]])
#     z = np.array([[min_magnitude, max_magnitude], [min_magnitude, max_magnitude]])
#     ax.plot_surface(x, y, z, color='orange', alpha=0.3, linewidth=0, antialiased=False,label='Mean Freq.')

#     # Set the labels with the correct font sizes
#     ax.set_xlabel('Subject', fontsize=15)
#     ax.set_ylabel('Frequency', fontsize=15)
#     ax.set_zlabel('Magnitude', fontsize=15)

#     # Set the title with the correct font size
#     ax.set_title(title, fontsize=18)

#     # Set the font size for the tick labels
#     ax.tick_params(axis='both', which='major', labelsize=10)

#     # Set the ticks for the subjects - adjust if the labels are actual names or other strings

#     total_subjects = len(subjects)
#     desired_ticks = 11

#     # Generate evenly spaced tick positions
#     tick_positions = np.linspace(1, total_subjects , desired_ticks, dtype=int)

#     # Apply the tick positions to the x-axis
#     ax.set_xticks(tick_positions)
#     ax.set_xticklabels(subjects, fontsize=15)  # Rotate if needed
#     # Set the legend with appropriate font size
#     ax.legend(fontsize=15)
#     ax.view_init(30, 60)
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# def plot_3d_surface(subjects, f, fft_results, title):
#     from matplotlib.patches import Patch

#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     subject_grid, frequency_grid = np.meshgrid(np.arange(len(subjects)), f)

#     # Transpose fft_results to align with the subject and frequency grids
#     ax.plot_surface(subject_grid, frequency_grid, fft_results.T, cmap='Greens', alpha=0.8, zorder=1)
    
#     peak_indices = np.argmax(fft_results, axis=1)
#     peak_freqs = f[peak_indices]

#     # Scatter plot for peak frequencies should use subject_grid for X-axis data
#     ax.scatter(subject_grid[0], peak_freqs, fft_results.max(axis=1), color='red', s=50, label='Peak Frequencies', zorder=3)

#     peak_indices = np.argmax(fft_results, axis=1)
#     peak_freqs = f[peak_indices]

#     max_magnitude = np.max(fft_results)
#     min_magnitude = np.min(fft_results)

#     mean_freq = np.mean(peak_freqs)
#     print("Plotting band at mean frequency:", mean_freq)

#     # Create a rectangle at the mean frequency across all subjects
#     x = np.array([[0, len(subjects) - 1], [0, len(subjects) - 1]])
#     y = np.array([[mean_freq, mean_freq], [mean_freq, mean_freq]])
#     z = np.array([[min_magnitude, min_magnitude], [max_magnitude, max_magnitude]])
#     ax.plot_surface(x, y, z, color='orange', alpha=0.3, linewidth=0, antialiased=False, label='Mean freq.', zorder=2)
#     total_subjects = len(subjects)
#     tick_positions = np.arange(1, total_subjects+1, 1) 
#     ax.set_xticks(tick_positions )

#     ax.set_xlabel('Subject', fontsize=18)
#     ax.set_ylabel('Frequency', fontsize=18)
#     ax.set_zlabel('Magnitude', fontsize=18)
#     #ax.set_title(title, fontsize=18)

#     # Create custom legend entries
#     legend_elements = [
#         Patch(facecolor='red', edgecolor='red', alpha=0.5, label='Peak freq. per Subject'),
#         Patch(facecolor='orange', edgecolor='orange', alpha=0.3, label='Mean Frequency Band')
#     ]
#     ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=18)

#     ax.view_init(30, 300)
#     plt.show()
def plot_3d_surface(subjects, f, fft_results, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Reverse the order of subjects and corresponding FFT results
    subjects_reversed = subjects[::-1]
    fft_results_reversed = fft_results[:, ::-1]

    # Now, create the subject grid for the reversed subject order
    subject_grid, frequency_grid = np.meshgrid(np.arange(len(subjects_reversed)), f)

    # Transpose fft_results to align with the subject and frequency grids
    surf = ax.plot_surface(subject_grid, frequency_grid, fft_results_reversed.T, cmap='Greens', alpha=1, zorder=5)

    peak_indices = np.argmax(fft_results_reversed, axis=1)
    peak_freqs = f[peak_indices]

    # Scatter plot for peak frequencies
    ax.scatter(subject_grid[0], peak_freqs, fft_results_reversed.max(axis=1), color='red', alpha=0.5, s=50, marker='o', zorder=3)

    max_magnitude = np.max(fft_results_reversed)
    min_magnitude = np.min(fft_results_reversed)

    mean_freq = np.median(peak_freqs)

    # Rectangle at the mean frequency across all subjects
    x = np.array([[0, len(subjects_reversed) - 1], [0, len(subjects_reversed) - 1]])
    y = np.array([[mean_freq, mean_freq], [mean_freq, mean_freq]])
    z = np.array([[min_magnitude, min_magnitude], [max_magnitude, max_magnitude]])
    ax.plot_surface(x, y, z, color='red', alpha=0.1, linewidth=0, antialiased=False, zorder=0)

    # Set custom ticks for frequency
    ax.set_yticks([2, 5, 8, 10, 12, 15, 20, 25])
    # Adjust x-axis (subject axis) ticks and labels
    ax.set_xticks(np.arange(len(subjects_reversed)))
    ax.set_xticklabels([str(subject) for subject in subjects_reversed])

    ax.set_xlabel('Subject', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_zlabel('Magnitude', fontsize=18)

    # Add a colorbar based on the information from the surface plot
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Magnitude', fontsize=18)  # Set the label for the colorbar

    # legend entries
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Peak Freq. per Subject',
               markerfacecolor='red', markersize=10, alpha=0.5),
        Patch(facecolor='red', edgecolor='red', alpha=0.3, label='median Frequency Band')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)

    # Change view angle to focus on frequency axis
    ax.view_init(30, 45)
    plt.tight_layout()
    plt.show()
 
def norm_fft(fft_results):
    normalized_fft_results = np.zeros_like(fft_results)
    # Loop through the subjects
    for i in range(fft_results.shape[0]):
        # Calculate the mean and standard deviation of the FFT results for the subject
        mean = np.mean(fft_results[i])
        std = np.std(fft_results[i])
        # Compute the Z-score normalization
        normalized_fft_results[i] = (fft_results[i] - mean) / std
    return normalized_fft_results
    
def compute_histogram(freqs, fft_data, bin_width):
    # Dummy histogram computation for the example
    # Replace this with your actual histogram computation logic
    return freqs, np.abs(fft_data)

def plot_3d_histogram(freqs, fft_results, bin_width):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('Blues')
    min_value = fft_results.min()
    max_value = fft_results.max()

    for ii, fft_data in enumerate(fft_results):
        bin_centers, hist = compute_histogram(freqs, fft_data, bin_width)
        subject_arr = np.full(hist.shape, ii)
        colors = cmap((hist - min_value) / (max_value - min_value))
        ax.bar(bin_centers, hist, zs=subject_arr, zdir='y', color=colors, alpha=0.8, edgecolor='gray')

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Subject ID')
    ax.set_zlabel('FFT Amplitude')
    ax.set_yticks(range(len(fft_results)))  # Assuming fft_results is a list of FFTs, one per subject
    ax.set_yticklabels([f'Sub-{i+1}' for i in range(len(fft_results))])  # Replace with actual subject names if available

    ax.grid(True)
    ax.set_title('3D Histogram of FFT Across Participants')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_value, max_value))
    sm.set_array([])
    plt.colorbar(sm, label='FFT Amplitude')
    plt.show()


def calculate_circular_mean(phase_values):
    """
    Calculate   circ mean phase.
    """


    # Compute the circular mean phase
    circular_mean = np.angle(np.mean(np.exp(1j * phase_values)))

    # Convert the circular mean phase back to degrees
    circular_mean_degrees = np.rad2deg(circular_mean)
    return circular_mean_degrees
 

# %%
