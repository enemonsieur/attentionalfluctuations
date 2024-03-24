#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to estimate curve parameters
def param_estimator(t, x, fit_function, p0, bounds=(-np.inf, np.inf)):
    popt, _ = curve_fit(fit_function, t, x, p0=p0, bounds=bounds,maxfev = 50000)
    ssr = np.sum((x - fit_function(t, *popt)) ** 2)
    return popt, ssr

def ACA_fit(t, capture_amplitude, capture_decay, arousal_amplitude, arousal_decay_rate):
    Attcapture = capture_amplitude * np.exp(-(t-t.min())**2 / (capture_decay**2))
    arousal = arousal_amplitude * np.exp(-arousal_decay_rate * t)
    return Attcapture + arousal

t = np.linspace(0.4, 0.9, 1000)
noise_levels = np.arange(0, 1, 0.025)

# Initialize the default parameters for the ACA_fit function
default_params = {'capture_amplitude': 0.7, 'capture_decay': 0.2, 'arousal_amplitude': 0.2, 'arousal_decay_rate': 1}
num_params = len(default_params)

# Initialize arrays for initial parameters
capture_amplitude_steps = np.linspace(0, 1.5, 10)
capture_decay_steps = np.linspace(0.05, 0.5, 10)
arousal_amplitude_steps = np.linspace(0, 0.5, 10)
arousal_decay_rate_steps = np.linspace(0.5, 5, 10)

# Create a stack for initial parameters
initial_params = np.stack((capture_amplitude_steps, capture_decay_steps, arousal_amplitude_steps, arousal_decay_rate_steps), axis=1)

# Initialize storage matrices
curve_params = np.zeros((len(noise_levels), num_params))
SSRs = np.zeros(len(noise_levels))

# Loop through each noise level
for ii, noise_level in enumerate(noise_levels):
    temp_curve_params = np.zeros((len(initial_params), num_params))
    x = ACA_fit(t, **default_params) + np.random.normal(0, noise_level, len(t))
    
    for jj, init_param in enumerate(initial_params):
        p0 = init_param  
        popt, ssr = param_estimator(t, x, ACA_fit, p0)
        temp_curve_params[jj, :] = popt
    
    curve_params[ii, :] = np.mean(temp_curve_params, axis=0)
    SSRs[ii] = np.mean(ssr)

delta_params = np.abs(curve_params - np.array(list(default_params.values())))

fig, axs = plt.subplots(num_params, 1, figsize=(10, 5 * num_params))

param_names = list(default_params.keys())
colors = ['tab:orange', 'tab:blue', 'tab:green', 'tab:purple']

for i in range(num_params):
    axs[i].set_xlabel('Noise Level')
    axs[i].set_ylabel(param_names[i], color=colors[i])
    axs[i].plot(noise_levels, delta_params[:, i], color=colors[i])
    axs[i].tick_params(axis='y', labelcolor=colors[i])
    
    ax2 = axs[i].twinx()
    ax2.set_ylabel('SSR', color='tab:red')
    ax2.plot(noise_levels, SSRs, color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

plt.suptitle('ACA Fit: Difference in fitting and original parameters, and SSRs at different Noise Levels')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# %%
