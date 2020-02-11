
'''
 This file takes the raw data which is in the form {(x,y,key)} where,
 INPUT:
 key: Ad-campaign, Type: integer
 x: price, Type: float
 y: return on investment, Type: float
 
 OUTPUT:
 (key, point of diminishing returns), Type: (integer, float)
  
 The goal here is to find the point of diminishing returns for
 each Ad-campaign. 
 As point of diminishing returns is defined as point where the 
 marginal increase starts decreasing, we calculate the point 
 where the second derviative of y as a function of x changes sign. 
 
 The above is calculated for every campaign and proceedes by first 
 smoothing the signal. 
'''

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


File = 'https://github.com/bindhyeswari/heroku-deploy-kunal/blob/master/ml/input_data/Data.xlsx'

df = pd.read_excel(File)

dfc = df.copy()

# Get the keys
keys = dfc['Key'].unique()
print("The number of unique keys is: ", len(keys))

def builddf(data):
    key_data = {} # Build a dictionary
    for i,k in enumerate(keys):
        key_data[k] = data[data['Key']==k] # Datafrmae corresponding to the key.
    return key_data

# Calling the function.
dfs = builddf(dfc)


# Gets the keys
keys = keys.tolist()

# Take an example dataset and find point of diminishing returns for that dataset.

dfe = dfs[keys[100]]

# Make sure the dataset is not too small so the model could be extended.

print("Size of the data set: Rows = {} and columns = {}".format(dfe.shape[0],dfe.shape[1]))

# Find the points of diminishing return

# To take the derivative of y with respect to x, we
# set the differential in x to '0.01' which is constant
# through out the data set.

dx = 0.01

#Using numpy's 'gradient' to find the finite-difference (derivative).
y_1 = np.gradient(dfe['y'],dx) # first derivative.
dfe['y_1'] = y_1
y_2 = np.gradient(dfe['y_1'],dx)# second derivative.
dfe['y_2'] = y_2

def savgol_f(data):
    '''Apply the filter with window size 51 and polynomial degree 1. '''
    # We smoothen the data three time whcih in our analysis makes
    # the data sufficently smooth.
    # Results of the data are stored in 'ys'
    window = 0
    if data.shape[0] < 50:
        window = 3
    elif 50<= data.shape[0] < 500:
        window = 11
    else:
        window = 51
    data['ys1']= savgol_filter(data['y'].values, window, 1)
    data['ys2']= savgol_filter(data['ys1'].values, window, 1)
    data['ys3']= savgol_filter(data['ys2'].values, window, 1)


# Now finding the point of diminishing returns.
# For that firstly we compute the derivatives.
def podr(data):
    # Calculate the derivatives first.
    dx = 0.01  # Set the differntial in x.
    first = np.gradient(data['ys3'], dx)
    data['first'] = first
    second = np.gradient(data['first'], dx)
    data['second'] = second

    # Calculate the point of diminishing returns.
    inf = []
    ps = 0
    std_unb = data['second'].std() / np.sqrt(data.shape[0])  # Calculate unbiased standard deviation.
    cut_off = int(0.5 * std_unb)  # Take half of it as cutoff.
    for i in range(data.shape[0]):
        if ((data['second'].iloc[-i] < 0) == False) and (data['second'].iloc[-i] > cut_off):
            inf.append((data.iloc[-1 - i, 1], data.iloc[-1 - i, 2]))  # Add (x,y) values.
    # Return the first point where the "If" condition above is satisfied.
    if len(inf) == 0:
        return -1
    else:
        return inf[0]


# Program to find the point of moving average for the whole data set.

# Corresponding to each key, we find the point of diminishing return,
# whcih is then returned by the function.

def podr_full():
    points = {}
    for k in keys:
        data = dfs[k]  # Get the dataframe corresponding to the key.
        savgol_f(data) # Apply the filter to smoothen the data
        pts = podr(data) # Collect the points of diminishing returns.
        points[k] = pts
    return points


   
 # Build a non-linear regression model
 def func(x,a,b,c,d):
     return a*(np.exp(b*(x**c))) + d
 

if __name__ == '__main__':

    points_dim = podr_full()
    with open('Results_podr.txt', 'w') as f:
        for key, value in points_dim.items():
            f.write('%s:%s\n' % (key, value))
    popt, pcov = curve_fit(func, dfe['x'], dfe['y])

