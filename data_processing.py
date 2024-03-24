#%% Import the dependencies
import pandas as pd
import numpy as np
import scipy.signal as signal
import numpy as np
from scipy.stats import exponnorm
from scipy.optimize import minimize
def load_and_clean_data_CAT1(file_path):
# Print the first 5 rows of the DataFrame
    try:
        # Read dataset
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    #drop the NaNs
    df_filtered = df_raw.dropna(subset=['TCD'])
    df_filtered = df_raw.dropna(subset=['TCT'])
    df_filtered = df_raw.dropna(subset=['TTR'])

    # drop negative RT
    df_filtered = df_raw[df_raw['TTR'] > 0]
    #drop kids
    df_filtered = df_raw[df_raw['age'] >=18]
    # find the DT
    df_filtered.loc[:, 'DT'] = df_filtered['TCT'] - df_filtered['TCD']
    df_filtered = df_filtered.rename(columns={'suj': 'sub_idx'})
    # Select relevant columns
    df_subset = df_filtered[['sub_idx','CueSide', 'DT','DisType', 'TTR']]
    df = df_subset.rename(columns={'CueSide': 'CUE', 'DisType': 'DIS', 'TTR': 'RT'})
    df = df.dropna(subset=['DT'])
    df = df.dropna(subset=['RT'])

    return df

def load_and_clean_data_noDIS(file_path):
    """
    We try to load the data from the CAT 1 OR cat 2 file and take only what we need.
    Returns a DataFrame.
    """
    try:
        # Read dataset
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    df_filtered = df_raw.dropna(subset=['CD'])
    df_filtered = df_filtered[df_filtered['RT'] > 0]
    #df_filtered = df_filtered[df_filtered['DT'] > 0]
    df_noDIS = df_filtered[df_filtered['DT'] == 0]

    # Select relevant columns
    df = df_filtered[['sub_idx','CUE', 'DT','CORR', 'RT','DIS']]
    df_noDIS = df_noDIS[['sub_idx','CUE', 'DT','CORR', 'RT','DIS']]

    return df,df_noDIS

def load_and_clean_data(file_path):
    """
    We try to load the data from the CAT 1 OR cat 2 file and take only what we need.
    Returns a DataFrame.
    """
    try:
        # Read dataset
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    df_filtered = df_raw.dropna(subset=['CD'])
    df_filtered = df_filtered[df_filtered['RT'] > 0]
    df_filtered = df_filtered[df_filtered['DT'] > 0]

    # Select relevant columns
    df = df_filtered[['sub_idx','CUE', 'DT','CORR', 'RT','DIS']]

    return df

def load_and_clean_data(file_path):
    """
    We try to load the data from the CAT 1 OR cat 2 file and take only what we need.
    Returns a DataFrame.
    """
    try:
        # Read dataset
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    df_filtered = df_raw.dropna(subset=['CD'])
    df_filtered = df_filtered[df_filtered['RT'] > 0]
    df_filtered = df_filtered[df_filtered['DT'] > 0]

    # Select relevant columns
    df = df_filtered[['sub_idx','CUE', 'DT','CORR', 'RT','DIS']]

    return df

def load_and_clean_data_noDIS_old(file_path):
    """
    Takes the noDIS trials for old data.
    """
    try:
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Filter data
    df_filtered = df_raw[(df_raw['RT'] > 0) & (df_raw['sub_idx'].str.startswith('oc'))]
    df_noDIS = df_filtered[df_filtered['DT'] == 0]

    # Remove 'oc' prefix and convert sub_idx to int
    for df in [df_filtered, df_noDIS]:
        df.loc[:, 'sub_idx'] = df['sub_idx'].str.replace('oc', '').astype(int)


    # Select relevant columns
    columns = ['sub_idx', 'CUE', 'DT', 'CORR', 'RT', 'DIS']
    df_filtered = df_filtered[columns]
    df_noDIS = df_noDIS[columns]

    return df_filtered, df_noDIS


def load_and_clean_data_old(file_path):
    """
    Load the data from the CAT 1 OR cat 2 file and take only what we need.
    Returns a DataFrame.
    """
    try:
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Filter data
    df_filtered = df_raw[
        (df_raw['RT'] > 0) & 
        (df_raw['DT'] > 0) & 
        (df_raw['sub_idx'].str.startswith('oc'))
    ]

    # Remove 'oc' prefix and convert sub_idx to int
    df_filtered.loc[:, 'sub_idx'] = df_filtered['sub_idx'].str.replace('oc', '').astype(int)

    # Select relevant columns
    columns = ['sub_idx', 'CUE', 'DT', 'CORR', 'RT', 'DIS']
    df_filtered = df_filtered[columns]

    return df_filtered

def process_POP_data(df, num_subjects,BIN_SIZE):
    """
    Function to z-score RT and DT for each subject, bin the data, and append them to x and t.
    """
    x = []
    t = []
    bin_x = []
    bin_t = []

    for ii, subject in enumerate(num_subjects):
        # Extract subset for each subject
        df_subject = df[df['sub_idx'] == subject]

        # Z-score 'RT' and 'DT' for each subject
        xraw = np.array(df_subject['RT']) / 1000
        traw = np.array(df_subject['DT']) / 1000

        x_subject = (xraw - np.mean(xraw)) 

        # Append to x and t
        x.append(x_subject)
        t.append(traw)
    # Flatten the lists
    x = np.concatenate(x)
    t = np.concatenate(t)

    # Bin those for clarity
    n_bins = int((t.max() - t.min()) / BIN_SIZE)

    # 1st let's create bins steps (from min +10, etc... until max)
    bins = np.linspace(t.min(), t.max(), n_bins + 1)

    # Now we have to Bin the values of x and t
    bin_idx = np.digitize(t, bins) 
    # find out for each trial in t from 0 to 566 say, where they belong in the bins.
    # Using the info of bin_idx, let's group the x values with each other
    bin_x = np.array([x[bin_idx == i+1].mean() for i in range(n_bins)]) #what the hell is that, 
    # I can hear. Lemme explain: first upi create bin x size n_bins. then you loop through the
    # values that are the number ofbins of bin_x, say, 5. then you look for at the x index from say 1 to 566, 
    # and each of them have their corresponding bin_x position between 1 to 5. ()
    bin_t = (bins[1:] + bins[:-1]) / 2
    return bin_t, bin_x,t,x

def process_POP_data_trial(df, num_subjects, TRIAL_NUM):
    """
    Bins the data into bins of 100 trials.

    Args:
        t (np.array): The time array.
        x (np.array): The response array.

    Returns:
        bin_t (np.array): The binned time array.
        bin_x (np.array): The binned response array.
    """
    x = []
    t = []
 

    for ii, subject in enumerate(num_subjects):
        # Extract subset for each subject
        df_subject = df[df['sub_idx'] == subject]

        # Z-score 'RT' and 'DT' for each subject
        xraw = np.array(df_subject['RT']) / 1000
        traw = np.array(df_subject['DT']) / 1000

        x_subject = (xraw - np.mean(xraw)) 

        # Append to x and t
        x.append(x_subject)
        t.append(traw)
    # Flatten the lists
    x = np.concatenate(x)
    t = np.concatenate(t)

    # Sort the data

    t_sorted = np.sort(t)
    # get ind of the t vector aswell
    t_sorted_indices = np.argsort(t)

    # Sort  x vector acording to sorted indices oft vector
    x_sorted = x[t_sorted_indices]
    # Initialize the binned response array
    bin_x = np.empty(int(len(t) / TRIAL_NUM))
    bin_t = np.empty(int(len(t) / TRIAL_NUM))

    # Iterate over the bins
    for i in range(int(len(t) / TRIAL_NUM)):
        # Calculate the mean of the current bin
        bin_x[i] = x_sorted[i * TRIAL_NUM:(i + 1) * TRIAL_NUM].mean()
        bin_t[i] = t_sorted[i * TRIAL_NUM:(i + 1) * TRIAL_NUM].mean()
    return t,x, bin_t, bin_x



def process_subject_data_noDIS(df,df_nodis, subject,BIN_SIZE):

    """
    Function to z-score RT and DT for each subject, bin the data, and append them to x and t.
    """

    # Extract infos for each subject
    df_subject = df[df['sub_idx'] == subject]
    df_snoDIS = df_nodis[df_nodis['sub_idx'] == subject]

    # turn em into arrays
    xraw = np.array(df_subject['RT']) / 1000
    traw = np.array(df_subject['DT']) / 1000
    xnoDIS = np.array(df_snoDIS['RT']) / 1000 
    # find zscore of the x parts
    x = (xraw - np.mean(xnoDIS))  / np.std(xraw)
    t = traw

    # fin how many trials the bin size makes
    n_bins = int((t.max() - t.min()) / BIN_SIZE)

    # 1st let's create bins steps (from min +10, etc... until max)
    bins = np.linspace(t.min(), t.max(), n_bins + 1)

    # NUse digitize to turn the trials into their corresponding bins (say you have 5 bins, put each trials of t in bin ix= 0,1,2,3 or 4)
    bin_idx = np.digitize(t, bins) 

    # Using the info of bin_idx, let's group the x values with each other
    bin_x = np.array([x[bin_idx == i+1].mean() for i in range(n_bins)])
    # you index for ex. all the x's trial that correspond to sec 0.4 to 0.5 (if b)
    bin_t = (bins[1:] + bins[:-1]) / 2
    return bin_t, bin_x,t,x


def process_subject_data(df, subject,BIN_SIZE,df_all,Zc):

    """
    Function to z-score RT and DT for each subject, bin the data, and append them to x and t.
    """

    # Extract infos for each subject
    df_subject = df[df['sub_idx'] == subject]
    df_subject_all = df_all[df_all['sub_idx'] == subject]

    x_all = np.array(df_subject_all['RT']) / 1000

    # turn em into arrays
    xraw = np.array(df_subject['RT']) / 1000
    traw = np.array(df_subject['DT']) / 1000

    if Zc == True:
    # find zscore of the x parts
        x = (xraw - np.mean(x_all)) / np.std(x_all)
    else:
        x = (xraw - np.mean(xraw)) 

    t = traw

    # fin how many trials the bin size makes
    n_bins = int((t.max() - t.min()) / BIN_SIZE)

    # 1st let's create bins steps (from min +10, etc... until max)
    bins = np.linspace(t.min(), t.max(), n_bins + 1)

    # NUse digitize to turn the trials into their corresponding bins (say you have 5 bins, put each trials of t in bin ix= 0,1,2,3 or 4)
    bin_idx = np.digitize(t, bins) 

    # Using the info of bin_idx, let's group the x values with each other
    bin_x = np.array([x[bin_idx == i+1].mean() for i in range(n_bins)])
    # you index for ex. all the x's trial that correspond to sec 0.4 to 0.5 (if b)
    bin_t = (bins[1:] + bins[:-1]) / 2
    return bin_t, bin_x,t,x

from scipy.stats import exponnorm
from scipy.stats import norm

def fit_gaussian(x):
    """Fit a Gaussian distribution to the data."""
    mu, sigma = np.mean(x), np.std(x)
    return mu, sigma

def generate_gaussian(mu, sigma, size):
    """Generate Gaussian distributed values."""
    return norm.rvs(loc=mu, scale=sigma, size=size)

def calculate_ssr(actual, predicted):
    """Calculate the sum of squared residuals."""
    return np.sum((actual - predicted) ** 2)

def generate_exgaussian(mu, sigma, tau, size):
    k = tau / sigma  # Convert tau to the 'k' parameter for exponnorm
    return exponnorm.rvs(k, loc=mu, scale=sigma, size=size)

def calculate_ssr_exgaussian(actual, predicted):
    return np.sum((actual - predicted) ** 2)

def exgaussian_noise(mu, sigma, tau, size):
    normal_noise = np.random.normal(mu, sigma, size)
    exponential_noise = np.random.exponential(tau, size)
    return normal_noise + exponential_noise
def exgaussian_pdf(x, mu, sigma, k):

    """Define the PDF of the Ex-Gaussian distribution."""
    # Exponnorm takes a 'k' parameter which is related to tau
    return exponnorm.pdf(x, k, loc=mu, scale=sigma)

def neg_log_likelihood(params, x):
    """Calculate the negative log likelihood for the Ex-Gaussian."""
    mu, sigma, k = params
    likelihood = exgaussian_pdf(x, mu, sigma, k)
    return -np.sum(np.log(likelihood))

def fit_exgaussian(x):
    
    """Fit the Ex-Gaussian distribution to the data."""
    initial_guess = [np.mean(x), np.std(x), 1]  # Initial guess for mu, sigma, k
    bounds = [(None, None), (0, None), (0, None)]  # mu is unbounded, sigma and k are positive
    result = minimize(neg_log_likelihood, initial_guess, args=(x), bounds=bounds)
    mu, sigma, k = result.x
    return mu, sigma, k


def fakedata(t,params,noise_level,fitfunction,BIN_SIZE):
    """ Function that use fake parameters to generate a data distribution"""
    t = np.linspace(t.min(), t.max(), 1000)
    xraw = fitfunction(t,*params[:-1]) + np.random.normal(0, noise_level, len(t))
    x = xraw - np.mean(xraw)
    # Bin those for clarity
    n_bins = int((t.max() - t.min()) / BIN_SIZE)

    # 1st let's create bins steps (from min +10, etc... until max)
    bins = np.linspace(t.min(), t.max(), n_bins + 1)

    # Now we have to Bin the values of x and t
    bin_idx = np.digitize(t, bins) 

    # Using the info of bin_idx, let's group the x values with each other
    bin_x = np.array([x[bin_idx == i+1].mean() for i in range(n_bins)])

    bin_t = (bins[1:] + bins[:-1]) / 2
    return t,x, bin_t,bin_x


# def perform_fft(t, x):
#     """
#     Perform FFT analysis.
#     Returns f, fft_result, and mask.
#     """
#     # Find the min and max freq
#     duration = t.max() - t.min()
#     min_resolvable_freq = 1 / duration  # Lowest freq we can find
#     sampling_freq = (1 / np.mean(np.abs(np.diff(np.sort(t)))))  # Highest
    
#     # Apply low-pass filter
#     b, a = signal.butter(4, sampling_freq/2-1, fs=sampling_freq, btype='low') #sampling_freq/2-1
#     filtered_x = x #signal.filtfilt(b, a, x)
#     # Perform FFT analysis
#     n = len(t)  # Length of the time series
#     f = np.fft.fftfreq(n, d=1/sampling_freq)  # Frequency distribution

#     mask = (f >= min_resolvable_freq) & (f <= 25)  # Apply frequency constraint up to 25 Hz
#     f = f[mask]
#     fft_result = np.abs(np.fft.fft(filtered_x)[mask])  # Estimate the power for each freq

#     return f, fft_result, mask

import numpy as np

def gaussian_kernel(sigma, step):
    # Define the time range for the kernel, centered around zero
    length = int(10 * sigma / step)  # 10 times the sigma
    t = np.arange(-length * step / 2, length * step / 2, step)
    kernel = np.exp(-t**2 / (2 * sigma**2))
    # Normalize the kernel
    return kernel / sum(kernel)

def perform_fft(t, x, sigma=0.01, step=0.001):
    from scipy.signal import convolve
    from numpy import hanning
    """
    Perform FFT analysis with Gaussian smoothing and a Hanning window.
    Returns f, fft_result, and relevant frequencies.
    """
    # Convolve with Gaussian kernel
    g_kernel = gaussian_kernel(sigma, step)
    smoothed_x = convolve(x, g_kernel, mode='same')

    # Apply Hanning window
    windowed_x = smoothed_x * hanning(len(smoothed_x))

    # Find the Sampling frequency
    duration = t.max() - t.min()
    min_resolvable_freq = 1 / duration  # Lowest freq we can find
    fs = (1 / np.mean(np.abs(np.diff(np.sort(t)))))  # Highest
    
    # Perform FFT analysis
    n = len(windowed_x)
    f = np.fft.fftfreq(n, d=1 / fs)
    fft_result = np.abs(np.fft.fft(windowed_x))

    # Scale FFT result (if necessary)
    fft_result = fft_result * 2
    # take only usable freq
    mask = (f >= min_resolvable_freq) & (f <= 25)  # Apply frequency constraint up to 25 Hz
    f = f[mask]
    fft_result = np.abs(np.fft.fft(x)[mask])  # Estimate the power for each freq
    return f, fft_result, mask



def extract_peaks(f, fft_result):
    """
        Extract the top peaks from the FFT result.
        Returns the frequencies of the top peaks.
    """
    top_peak_indices = np.argsort(np.abs(fft_result))[-1:] #take the biggest value
    # Extract the top NUM_PEAKS peaks - sort the highest values. This may not be the bes tmethod but stil
    top_peak_freqs = f[top_peak_indices]

    return top_peak_freqs



# %%
