#%%
# import the dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
#from ipywidgets import interact, FloatSlider


def get_num_params(fit_type):
    if fit_type == 'poly':
        num_params = 2
    elif fit_type == 'sin':
        num_params = 4
    elif fit_type == 'both':
        num_params = 6
    else:
        raise ValueError(f'Unknown fit_type: {fit_type}')
    return num_params
def poly_fit(x, a, b):
    return a * np.power(x, 2) + b * x 
def sin_fit(x, Amp, freq, Phi,Vs):
    return Amp*np.sin(freq*x + Phi) +Vs
def combined_fit(x, a, b, Amp, freq, Phi,Vs):
    return sin_fit(x, Amp, freq, Phi,Vs) + poly_fit(x,  a, b)
def mean_abs_deviation(data): #find mean, substract to each param and find mean of deviation
    deviations = np.abs(data - np.mean(data))
    return np.mean(deviations)
# load the dataset
df_raw = pd.read_excel('summary_all2.xlsx')
# Print the first 5 rows of the DataFrame
print(df_raw.head())

# filter the relevant part of the dataset (TDT and TRT)
df_sorted = df_raw.sort_values('DisType', ascending=False)

# drop the NaNs
df_filtered = df_raw.dropna(subset=['TCD'])

# drop rows where TTR is negative
df_filtered = df_filtered[df_filtered['TTR'] >= 0]

# filter based on Age>=18 and TCD
df_filtered = df_filtered[(df_filtered['age'] >= 18)] # & (df_filtered['TCD'])
df_filtered.loc[:, 'TDT'] = df_filtered['TCT'] - df_filtered['TCD']

#Maximum value of TDT: 1020.0
#Minimum value of TDT: 363.0
df_temp = df_filtered[['suj','DisType','TTR', 'TDT']]


#%% 

# CHOOSE DIS (put it upwards so that one can define all the params at once)
selected_dis_type = 'DIS1' #'DIS1', 'DIS2', 'both'
selected_subject = 'all' #'all', 'sujX', ["s5", "s10", "s15"] 
bin_size = 0.01 #from 0.001 to 0.5
fit_type = 'both' #'sin' or 'poly' or both 

print('selected distractor  is', selected_dis_type) 
# filter the dataframe based on the selected dis type
if selected_dis_type == 'DIS1':
    df_dis_type  = df_temp[df_temp['DisType'] == 2] #cut the bolean series where its true
elif selected_dis_type == 'DIS2':
    df_dis_type  = df_temp[df_temp['DisType'] == 1]
elif selected_dis_type == 'both':
    df_dis_type = df_temp[(df_temp['DisType'] == 1) | (df_temp['DisType'] == 2)] # select all the data
else: # CHANGE THIS IF WE DONT CARE ABOUT THE DISTRACTIOR
    print("Invalid dis type selected.")


# CHOOSE subjects
if selected_subject == 'all':
    df_subset = df_dis_type.copy()  # Select all subjects
elif isinstance(selected_subject, list):
    df_subset = df_dis_type[df_dis_type['suj'].isin(selected_subject)]  # Select multiple subjects
elif isinstance(selected_subject, str):
    df_subset = df_dis_type[df_dis_type['suj'] == selected_subject]  # Select single subject
else:
    print("Invalid subject selected.")


print('selected distractor  is', selected_dis_type) 
print('selected dis type is', selected_dis_type)
print('selected bin size is', bin_size)
print('selected fit function is', fit_type)

#%% Params module
num_subjects = df_temp['suj']#.unique() #unique subjects
num_params = get_num_params(fit_type) #number of parameter of the fit you choosed
sub_index = 0 #init the index of the subject
curve_params = np.zeros((len(num_subjects), num_params)) #init the dataframe with the number of subjects and the number of parameters


#loop through the subjects 
sub_index = 0
for i in range(1):#subject in num_subjects: #loops through all the sub
    sub_index += 1
 
    # extract subset for each subject
    #df_subject = df_subset[df_subset['suj'] == subject]
    #find average value for each subject
    # x = np.array(df_subject['TTR']) /1000
    # t = np.array(df_subject['TDT']) /1000
    
    # If you want to plot all sunjcets on the same plot, comment those lines out
    x = np.array(df_subset['TTR']) /1000
    t = np.array(df_subset['TDT']) /1000

    # find the # of bins you need
    n_bins = int((t.max() - t.min()) / bin_size)
    # Now let's create bins steps/edges (from min +10, etc... until max)
    bins = np.linspace(t.min(), t.max(), n_bins + 1)
    #now we have to Bin the values of x and t
    bin_idx = np.digitize(t, bins)
    #givesthe t[ind] (so the relevant times), that correspond to the bins we're creating
    bin_x = np.zeros(n_bins)
    #using the info of bin_idx, let's group the x values with each other
    for i in range(n_bins):
        bin_x[i] = x[bin_idx == i+1].mean()
    bin_t = (bins[1:] + bins[:-1]) / 2


    # plot the data
    # PUT ABOVE: Choose whether to show the plot or not
    show_plot = False

    if show_plot:
        plt.plot(t,x , 'o', alpha=0.2, color='lightblue')
        plt.plot(bin_t, bin_x, 'ok')
        plt.xlabel('Cue-Distractor time')
        plt.ylabel('Target-Responses time')
        plt.title('RTs of Trails for each Cue-Distractor condition in the CAT paradigm')
        plt.legend(['Independants RTs', 'Mean TRT for each 1ms interval'])
        plt.ylim(0, 0.7)
        plt.show()


    # FIT THE DATA 

    fit_functions = {'poly': poly_fit, 'sin': sin_fit, 'both': combined_fit}
    fitfunction = fit_functions[fit_type]
    #define the init params
    # add the starting params
    if fit_type == 'poly':
        p0 = [0, 0]  # a=1, b=1
    elif fit_type == 'sin':
        p0 = [0.2, 6, 0,0.5]  # Amp=1, freq=1, Phi=0, Vs= 0.5
    elif fit_type == 'both':
        p0 = [0, 0, 0.2, 6, 0,0.3]    
        # a=1, b=1, Amp=1, freq=1, Phi=0, Vs= 0.5

    poptsin, pcov = curve_fit(sin_fit, bin_t, bin_x,maxfev = 100000 ) 
    poptpoly, pcov = curve_fit(poly_fit, bin_t, bin_x,maxfev = 100000 ) 
    poptcomb, pcov = curve_fit(combined_fit, bin_t, bin_x,maxfev = 100000 ) 

    #curve_params[sub_index-1] = popt
#print('params:', popt)
#%%
plt.plot(t,x , 'o', alpha=0.1, color='lightblue')
#plt.plot(bin_t, bin_x, '-k')
plt.plot(bin_t, sin_fit(bin_t, *poptsin),  color='orange',alpha=0.5,linewidth=2) #
plt.plot(bin_t, poly_fit(bin_t, *poptpoly), '-g',alpha=0.5,linewidth=2) #
plt.plot(bin_t, combined_fit(bin_t, *(poptcomb)), '-r',alpha=0.5,linewidth=2) #
plt.xlabel('Cue- Distractor time')
plt.ylabel('Target-Responses time')
plt.title('3 differents curves fitting')
plt.legend(['Independants RTs', 'Sin','Poly','Combined'])
#plt.ylim(0.1, 0.5)
#plt.xlim(1.35,1.57)
plt.show()
#%%# Define the ll
def log_likelihood(y_obs, y_pred, sigma):
    '''
    Calculate  the ll
    '''
    return -0.5 * np.sum((y_obs - y_pred)**2 / sigma**2 + np.log(2 * np.pi * sigma**2))
x_pred = fitfunction(t, *popt)
#lets calculate SD
residuals = x - fitfunction(t, *popt)

# Calculate the RMSE (which is like the estimate of sigma)
SSR = np.sum(residuals**2)
sd = np.sqrt(SSR / len(x))
print("RMSE:", sd)

# find the LL based on that
ll = log_likelihood(x, x_pred, sd)
print('LL:', ll)

#Estimate the BIC
bic = -2 * ll + num_params * np.log(len(x))
print('BIC:', bic)

# find the AIC
k = len(popt)
aic = -2 * ll + 2 * k

print("AIC:", aic)

# %%
plt.plot(t,x , 'o', alpha=0.2, color='lightblue')
plt.plot(bin_t, bin_x, '-k')
plt.plot(bin_t, fitfunction(bin_t, *popt), '-r',alpha=0.5) #
plt.xlabel('Cue- Distractor time')
plt.ylabel('Target-Responses time')
plt.title('Fitting of the opt params on the Cue-Distractor condition in the CAT paradigm')
plt.legend(['Independants RTs', 'Mean RT', 'Fitted curve'])
plt.ylim(0.1, 0.6)
#plt.xlim(1.35,1.57)
plt.show()
# %% NOW try including Cue to Target time
