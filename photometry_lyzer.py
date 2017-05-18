# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:33:03 2017

@author: baylorbrangers
"""

#Fiber photometry signal analysis

import numpy as np
import matplotlib.pyplot as plt

#f= np.loadtxt('/Users/baylorbrangers/Desktop/end_times.CSV',delimiter=',',skiprows=1, unpack=True)
fid = open('/Users/baylorbrangers/Desktop/test0.csv')
dim_data = np.fromfile(fid, dtype='float32')

#amount of data aquired
v_length_data=np.size(dim_data)
v_length_data=(int(v_length_data)/4)

#generate matrix with stimulation/response values, using Fortran ordering
m_stim_response = np.reshape(dim_data,(4,int(v_length_data)),order='F').T

#filtering function for channels

def signal_filtering(stimulation_matrix,filtered_cols,threshold):
    
    for x in filtered_cols:
        v_photometry_data=stimulation_matrix[:,x]
        i_cut_off=2.5*np.ndarray.max(v_photometry_data)
        v_photometry_data[v_photometry_data>i_cut_off]=0
        corrected_photometry=v_photometry_data
        stimulation_matrix[:,x]=corrected_photometry
        return stimulation_matrix

#corrects time column
def time_values_signal(stimulation_matrix,data_length, time_column):
        corrected_time=np.arange(0,data_length, 1)
        stimulation_matrix[:, time_column]=corrected_time
        
        return stimulation_matrix        

def signal_averaging(stimulation_matrix,avg_range):
    #define time steps, in ms chunks
     i_length_data=stimulation_matrix.shape[0]/avg_range
     #i_length_data=i_length_data/avg_range
     
     #intialize avg matrix for time, red and green
     
     avg_matrix=np.zeros([i_length_data,2])
 
     x=0
     while (x < i_length_data):
         #avg_matrix[x,:]=np.mean(stimulation_matrix[np.arange(x*avg_range, (x+1)*avg_range, 1),2:3+1], axis=0)
         avg_matrix[x,:]=np.mean(stimulation_matrix[x*avg_range:(x+1)*avg_range,2:], axis=0)
         #avg_matrix[x,1]=np.mean(stimulation_matrix[np.arange(x, (x+1)*avg_range, 1),2])
         x=x+1
     return avg_matrix

#==============================================================================
#     
#==============================================================================
time_corrected=time_values_signal(m_stim_response,v_length_data,1)
signal_filt=signal_filtering(time_corrected,(2,3),1)
t=signal_averaging(signal_filt,500)




## Plotting data!
# evenly sampled time at 200ms intervals
# red dashes, blue squares and green triangles
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.plot(signal_filt[[1,100000],1],(signal_filt[[1,100000],3]-signal_filt[[1,100000],2]),'r--',)


plt.plot(signal_filt[[1,300],1],signal_filt[[1,300],2],'r--')