# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:52:00 2019

@author: Bosman Peter
"""
import numpy as np
import copy as cp
import forwardmodel as fwdm
import inverse_modelling as im
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt
import shutil
import os
import multiprocessing
from joblib import Parallel, delayed
import csv
import pandas as pd
import math
import glob
import matplotlib.style as style
style.use('classic')

##################################
###### user input: settings ######
##################################
ana_deriv = True
use_backgr_in_cost = False
write_to_f = True
use_ensemble = False
if use_ensemble:
    nr_of_members = 2
    est_post_pdf = True
    plot_perturbed_obs = True
maxnr_of_restarts = 3 #only implemented for tnc method at the moment
imposeparambounds = True
remove_all_prev = True #Use with caution, be careful for other files in working directory!!
optim_method = 'tnc' #bfgs or tnc
run_multicore = False #only possible when using ensemble, not working?
stopcrit = 1.0
use_mean = True #switch for using mean of obs over several days
use_weights = False #weights for the cost function, to enlarge the importance of certain obs
if use_weights:
    weight_morninghrs = 1/4 # to change weight of obs in the mroning before 10, when everything less well mixed. 1 means equal weights
if (use_backgr_in_cost and use_weights):
    obs_vs_backgr_weight = 1.0 # a scaling factor for the importance of all the observations in the cost function
discard_nan_minims = False #if False, if in a minimisation nan is encountered, it will use the state from the best simulation so far, if True, the minimisation will result in a state with nan's
plt.rc('font', size=12) #plot font size
######################################
###### end user input: settings ######
######################################

#remove previous files
if remove_all_prev:
    filelist_list = []
    filelist_list += [glob.glob('Optimfile*')] #add Everything starting with 'Optimfile' to the list
    filelist_list += [glob.glob('Gradfile*')]
    filelist_list += [glob.glob('Optstatsfile*')]
    filelist_list += [glob.glob('pdf_posterior*')]
    filelist_list += [glob.glob('fig_fit*')]
    for filelist in filelist_list:
        for filename in filelist:
            if os.path.isfile(filename): #in case file occurs in two filelists in filelist_list, two attempts to remove would give error
                os.remove(filename)

##################################
###### user input: load obs ######
##################################
canopy_height = 17 #m

#read data
start = 1000 #line where we start in csv file (-1)
selectedyears = [2015.]
selectedmonths = [8.]
selectedday = 18. #only used when not use_mean = True
selecteddays = [18,20,21,22,23,24,25] #period over which we average
selectedminutes = [0]#range(0,60) #only for second data file. use an array!
starthour = 5 #utc, start of obs (6)
endhour = 16

directory = 'Hyytiala_obs'
with open(directory+'/Data_2015_20190201.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i in range(0,start):
        next(reader) #skips the header
    #next(reader, None)
    data =[]
    for row in reader:
        data += [row]

hour = np.zeros(len(data))
day = np.zeros(len(data))
month = np.zeros(len(data))
year = np.zeros(len(data))
COS_125 = np.zeros(len(data))
CO2_125 = np.zeros(len(data))
H2O_125 = np.zeros(len(data))
COS_23 = np.zeros(len(data))
CO2_23 = np.zeros(len(data))
H2O_23 = np.zeros(len(data))
Temp17 = np.zeros(len(data))
Press = np.zeros(len(data))
TsoilA = np.zeros(len(data))
Rd  = 287.04
F_EC_COS = np.zeros(len(data))
F_EC_COS_flag = np.zeros(len(data))
F_EC_CO2_flag = np.zeros(len(data))
F_EC_COS_flag = np.zeros(len(data))
F_EC_CO2_flag = np.zeros(len(data))
GPP_NEE = np.zeros(len(data))
ustar = np.zeros(len(data))
H_flux = np.zeros(len(data))
LE_flux = np.zeros(len(data))
soilCOS_flux1 = np.zeros(len(data))
soilCOS_flux2 = np.zeros(len(data))
soilCO2_flux1 = np.zeros(len(data))
soilCO2_flux2 = np.zeros(len(data))
for i in range(len(data)):
    hour[i]   = data[i][0][11:13]
    hour[i]   = hour[i] - 2 #convert to utc. STAY AWAY FROM HOURS AFTER MIDNIGHT,ERRORS IN THAT CASE!!
    day[i]    = data[i][0][8:10]
    month[i]  = data[i][0][5:7]
    year[i]   = data[i][0][0:4]
    if data[i][1] == 'NA':
        data[i][1] = float('NaN')
    if data[i][2] == 'NA':
        data[i][2] = float('NaN')
    if data[i][3] == 'NA':
        data[i][3] = float('NaN')
    if data[i][4] == 'NA':
        data[i][4] = float('NaN')
    if data[i][5] == 'NA':
        data[i][5] = float('NaN')
    if data[i][6] == 'NA':
        data[i][6] = float('NaN')
    if data[i][37] == 'NA':
        data[i][37] = float('NaN')
    if data[i][38] == 'NA':
        data[i][38] = float('NaN')
    if data[i][40] == 'NA':
        data[i][40] = float('NaN')
    if data[i][41] == 'NA':
        data[i][41] = float('NaN')
    if data[i][43] == 'NA':
        data[i][43] = float('NaN')
    if data[i][33] == 'NA':
        data[i][33] = float('NaN')
    if data[i][19] == 'NA':
        data[i][19] = float('NaN')
    if data[i][18] == 'NA':
        data[i][18] = float('NaN')
    if data[i][16] == 'NA':
        data[i][16] = float('NaN')
    if data[i][21] == 'NA':
        data[i][21] = float('NaN')
    if data[i][22] == 'NA':
        data[i][22] = float('NaN')
    if data[i][26] == 'NA':
        data[i][26] = float('NaN')
    if data[i][23] == 'NA':
        data[i][23] = float('NaN')
    if data[i][27] == 'NA':
        data[i][27] = float('NaN')
    COS_125[i]    = data[i][1]
    COS_125[i] = COS_125[i] / 1000 #ppb
    CO2_125[i]    = data[i][2] 
    CO2_125[i] = CO2_125[i] #ppm
    H2O_125[i] = data[i][3]
    H2O_125[i] = H2O_125[i] /100 #mole fraction
    H2O_125[i] = H2O_125[i] / 28.97e-3 * 18e-3 #mol_H2O/mol_air * mol_air/kg_air * kg_H2O/mol_H2O gives kg_H2O / kg_air
    COS_23[i]    = data[i][4]
    COS_23[i] = COS_23[i] / 1000 #ppb
    CO2_23[i]    = data[i][5] 
    CO2_23[i] = CO2_23[i] #ppm
    H2O_23[i] = data[i][6]
    H2O_23[i] = H2O_23[i] /100 #mole fraction
    H2O_23[i] = H2O_23[i] / 28.97e-3 * 18e-3 #mol_H2O/mol_air * mol_air/kg_air * kg_H2O/mol_H2O gives kg_H2O / kg_air
    Press[i] = data[i-1][38] #hPa ;i-1 due to error in obs file
    Press[i] = float(Press[i])
    ustar[i] = data[i-1][37] #m s-1 ;i-1 due to error in obs file
    ustar[i] = float(ustar[i])
    H_flux[i] = data[i][40] #W m-2
    H_flux[i] = float(H_flux[i])
    LE_flux[i] = data[i][41] #W m-2
    LE_flux[i] = float(LE_flux[i])
    TsoilA[i] = data[i-1][43] #W m-2
    TsoilA[i] = float(TsoilA[i]) + 273.15
    Temp17[i]  = data[i-1][33]
    Temp17[i]  = float(Temp17[i]) + 273.15 #float needed as temp17m not predefined as array of floats
    rho = (Press[i]*100-1.20188*9.81*17) / (Rd*Temp17[i]) # gas law, pressure converted to Pa (not fully correct, using pressure correction based on us standard atmosphere)
    #1.20188 is density at 181+17 m in us standard atmosphere https://www.digitaldutch.com/atmoscalc/   (we actually need rho at 23 m, but we dont have temp there)
    F_EC_COS_flag[i] = data[i][19]
    F_EC_CO2_flag[i] = data[i][18]
    F_EC_COS[i] = data[i][16] #pmol/m2/s
    F_EC_COS[i] = F_EC_COS[i] / rho * 28.97e-3 /1000   #pmol/m2/s * m3/kg_air * kg_air/mol_air = ppt m/s; /1000 gives ppb m/s; rho we used not fully correct
    GPP_NEE[i] = data[i][21] #µmol/m2/s
    GPP_NEE[i] = GPP_NEE[i] * 44 * 1000*1e-6 #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2
    rho_for_soil = (Press[i]*100) / (Rd*Temp17[i]) #temp not correct actually...
    soilCOS_flux1[i] = float(data[i][22]) / rho_for_soil * 28.97e-3 /1000 # * m3/kg_air * kg_air/mol_air = ppt m/s; /1000 gives ppb m/s; rho we used not fully correct
    soilCOS_flux2[i] = float(data[i][26]) / rho_for_soil * 28.97e-3 /1000 # * m3/kg_air * kg_air/mol_air = ppt m/s; /1000 gives ppb m/s; rho we used not fully correct
    soilCO2_flux1[i] = float(data[i][23]) * 44 * 1000*1e-6   #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2 
    soilCO2_flux2[i] = float(data[i][27]) * 44 * 1000*1e-6   #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2 

if use_mean != True:
    COS_125_selected = np.zeros(endhour-starthour+1)
    CO2_125_selected = np.zeros(endhour-starthour+1)
    H2O_125_selected = np.zeros(endhour-starthour+1)
    COS_23_selected = np.zeros(endhour-starthour+1)
    CO2_23_selected = np.zeros(endhour-starthour+1)
    H2O_23_selected = np.zeros(endhour-starthour+1)
    hour_selected = np.zeros(endhour-starthour+1)
    F_EC_COS_selected = np.zeros(endhour-starthour+1)
    GPP_NEE_selected = np.zeros(endhour-starthour+1)
    Temp17_selected = np.zeros(endhour-starthour+1)
    Press_selected = np.zeros(endhour-starthour+1)
    ustar_selected = np.zeros(endhour-starthour+1)
    H_flux_selected = np.zeros(endhour-starthour+1)
    LE_flux_selected = np.zeros(endhour-starthour+1)
    TsoilA_selected = np.zeros(endhour-starthour+1)
    soilCOS_flux_selected = np.zeros(endhour-starthour+1)
    soilCO2_flux_selected = np.zeros(endhour-starthour+1)
    j = 0
    for i in range(len(data)):
        if ((year[i] in selectedyears) and (month[i] in selectedmonths)) and ((day[i] == selectedday) and (starthour<=hour[i]<=endhour)):
            COS_125_selected[j] = COS_125[i]
            CO2_125_selected[j] = CO2_125[i]
            H2O_125_selected[j] = H2O_125[i]
            COS_23_selected[j] = COS_23[i]
            CO2_23_selected[j] = CO2_23[i]
            H2O_23_selected[j] = H2O_23[i]
            hour_selected[j]    = hour[i]
            F_EC_COS_selected[j] = F_EC_COS[i]
            if F_EC_COS_flag[i] not in [0,1]:
                F_EC_COS_selected[j] = float('NaN')
            GPP_NEE_selected[j] = GPP_NEE[i]
            if F_EC_CO2_flag[i] not in [0,1]:
                GPP_NEE_selected[j] = float('NaN')
            Temp17_selected[j] = Temp17[i]
            Press_selected[j] = Press[i]
            ustar_selected[j] = ustar[i]
            H_flux_selected[j] = H_flux[i]
            LE_flux_selected[j] = LE_flux[i]
            TsoilA_selected[j] = TsoilA[i]
            soilCOS_flux_selected[j] = np.nanmean(soilCOS_flux1[i],soilCOS_flux2[i])
            soilCO2_flux_selected[j] = np.nanmean(soilCO2_flux1[i],soilCO2_flux2[i])
            j +=1
else:
    COS_125_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    CO2_125_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    H2O_125_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    COS_23_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    CO2_23_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    H2O_23_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    F_EC_COS_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    GPP_NEE_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    Temp17_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    Press_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    ustar_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    H_flux_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    LE_flux_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    TsoilA_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    soilCOS_flux_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    soilCO2_flux_selected = np.zeros((len(selecteddays),endhour-starthour+1))
    hour_selected = np.zeros(endhour-starthour+1)
    daycounter = 0
    hourcounter = 0
    for i in range(len(data)):
        if ((year[i] in selectedyears) and (month[i] in selectedmonths)) and ((day[i] in selecteddays) and (starthour<=hour[i]<=endhour)):
            COS_125_selected[daycounter,hourcounter] = COS_125[i]
            CO2_125_selected[daycounter,hourcounter] = CO2_125[i]
            H2O_125_selected[daycounter,hourcounter] = H2O_125[i]
            COS_23_selected[daycounter,hourcounter] = COS_23[i]
            CO2_23_selected[daycounter,hourcounter] = CO2_23[i]
            H2O_23_selected[daycounter,hourcounter] = H2O_23[i]
            F_EC_COS_selected[daycounter,hourcounter] = F_EC_COS[i]
            if F_EC_COS_flag[i] not in [0,1]:
                F_EC_COS_selected[daycounter,hourcounter] = float('NaN')
            GPP_NEE_selected[daycounter,hourcounter] = GPP_NEE[i]
            if F_EC_CO2_flag[i] not in [0,1]:
                GPP_NEE_selected[daycounter,hourcounter] = float('NaN')
            hour_selected[hourcounter] = hour[i]
            Temp17_selected[daycounter,hourcounter] = Temp17[i]
            Press_selected[daycounter,hourcounter] = Press[i]
            ustar_selected[daycounter,hourcounter] = ustar[i]
            H_flux_selected[daycounter,hourcounter] = H_flux[i]
            LE_flux_selected[daycounter,hourcounter] = LE_flux[i]
            TsoilA_selected[daycounter,hourcounter] = TsoilA[i]
            soilCOS_flux_selected[daycounter,hourcounter] = np.nanmean((soilCOS_flux1[i],soilCOS_flux2[i]))
            soilCO2_flux_selected[daycounter,hourcounter] = np.nanmean((soilCO2_flux1[i],soilCO2_flux2[i]))
            hourcounter +=1
            if hour[i] == endhour: #if the hour would be missing in the obs file we have a problem!!
                daycounter += 1
                hourcounter = 0
    COS_125_mean = np.zeros(endhour-starthour+1)
    CO2_125_mean = np.zeros(endhour-starthour+1)
    H2O_125_mean = np.zeros(endhour-starthour+1)
    COS_23_mean = np.zeros(endhour-starthour+1)
    CO2_23_mean = np.zeros(endhour-starthour+1)
    H2O_23_mean = np.zeros(endhour-starthour+1)
    F_EC_COS_mean = np.zeros(endhour-starthour+1)
    GPP_NEE_mean = np.zeros(endhour-starthour+1)
    Temp17_mean = np.zeros(endhour-starthour+1)
    Press_mean = np.zeros(endhour-starthour+1)
    ustar_mean = np.zeros(endhour-starthour+1)
    H_flux_mean = np.zeros(endhour-starthour+1)
    LE_flux_mean = np.zeros(endhour-starthour+1)
    TsoilA_mean = np.zeros(endhour-starthour+1)
    soilCOS_flux_mean = np.zeros(endhour-starthour+1)
    soilCO2_flux_mean = np.zeros(endhour-starthour+1)
    for i in range(0,endhour-starthour+1):
        COS_125_mean[i] = np.mean(COS_125_selected[:,i])
        CO2_125_mean[i] = np.mean(CO2_125_selected[:,i])
        H2O_125_mean[i] = np.mean(H2O_125_selected[:,i])
        COS_23_mean[i] = np.mean(COS_23_selected[:,i])
        CO2_23_mean[i] = np.mean(CO2_23_selected[:,i])
        H2O_23_mean[i] = np.mean(H2O_23_selected[:,i])
        F_EC_COS_mean[i] = np.nanmean(F_EC_COS_selected[:,i])
        GPP_NEE_mean[i] = np.nanmean(GPP_NEE_selected[:,i])
        Temp17_mean[i] = np.nanmean(Temp17_selected[:,i]) 
        Press_mean[i] = np.nanmean(Press_selected[:,i])
        ustar_mean[i] = np.nanmean(ustar_selected[:,i])
        H_flux_mean[i] = np.nanmean(H_flux_selected[:,i])
        LE_flux_mean[i] = np.nanmean(LE_flux_selected[:,i])
        TsoilA_mean[i] = np.nanmean(TsoilA_selected[:,i])
        soilCOS_flux_mean[i] = np.nanmean(soilCOS_flux_selected[:,i])
        soilCO2_flux_mean[i] = np.nanmean(soilCO2_flux_selected[:,i])
        
#stdev measurements
if use_mean:
    stdevCOS_125_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevCOS_125_hourly)):
        stdevCOS_125_hourly[i] = np.std(COS_125_selected[:,i])
    stdevCO2_125_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevCO2_125_hourly)):
        stdevCO2_125_hourly[i] = np.std(CO2_125_selected[:,i])
    stdevH2O_125_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevH2O_125_hourly)):
        stdevH2O_125_hourly[i] = np.std(H2O_125_selected[:,i])
    stdevCOS_23_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevCOS_23_hourly)):
        stdevCOS_23_hourly[i] = np.std(COS_23_selected[:,i])
    stdevCO2_23_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevCO2_23_hourly)):
        stdevCO2_23_hourly[i] = np.std(CO2_23_selected[:,i])
    stdevH2O_23_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevH2O_23_hourly)):
        stdevH2O_23_hourly[i] = np.std(H2O_23_selected[:,i])
    stdevCOSflux_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevCOSflux_hourly)):
        stdevCOSflux_hourly[i] = np.nanstd(F_EC_COS_selected[:,i]) #! avoid nan!!
    stdevGPP_NEE_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevGPP_NEE_hourly)):
        stdevGPP_NEE_hourly[i] = np.std(GPP_NEE_selected[:,i])
    stdevTemp17_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevTemp17_hourly)):
        stdevTemp17_hourly[i] = np.std(Temp17_selected[:,i])
    stdevPress_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevPress_hourly)):
        stdevPress_hourly[i] = np.std(Press_selected[:,i])
    stdevustar_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevustar_hourly)):
        stdevustar_hourly[i] = np.std(ustar_selected[:,i]) 
    stdevH_flux_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevH_flux_hourly)):
        stdevH_flux_hourly[i] = np.nanstd(H_flux_selected[:,i]) #! avoid nan!!
    stdevLE_flux_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevLE_flux_hourly)):
        stdevLE_flux_hourly[i] = np.nanstd(LE_flux_selected[:,i]) #! avoid nan!!
    stdevsoilCOS_flux_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevsoilCOS_flux_hourly)):
        stdevsoilCOS_flux_hourly[i] = np.nanstd(soilCOS_flux_selected[:,i]) #! avoid nan!!
    stdevsoilCO2_flux_hourly = np.zeros(endhour-starthour+1)
    for i in range(len(stdevsoilCO2_flux_hourly)):
        stdevsoilCO2_flux_hourly[i] = np.nanstd(soilCO2_flux_selected[:,i]) #! avoid nan!!

data2 = pd.read_csv(directory+'/'+'smeardata_20150801120000.csv', skiprows=3)
minute2 = data2['Minute'] #
hour2 = data2['Hour'] # 
day2 = data2['Day'] #
month2 = data2['Month'] #
year2 = data2['Year'] #

stdevTemp67_minutely = np.zeros((endhour-starthour+1,len(selectedminutes))) #stdev measurements
if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day2 == selectedday,np.logical_and(month2.isin(selectedmonths),year2.isin(selectedyears))),np.logical_and(np.logical_and(hour2>=starthour,hour2<=endhour),minute2.isin(selectedminutes)))
    Temp67_selected = data2[timeselection]['HYY_META.T672'] +273.15
else:
    timeselection = np.logical_and(np.logical_and(day2.isin(selecteddays),np.logical_and(month2.isin(selectedmonths),year2.isin(selectedyears))),np.logical_and(np.logical_and(hour2>=starthour,hour2<=endhour),minute2.isin(selectedminutes)))
    Temp67_selected = data2[timeselection]['HYY_META.T672'] +273.15
    Temp67_mean = np.zeros((endhour-starthour+1,len(selectedminutes)))
    
    for i in range(endhour-starthour+1):
        for j in range(len(selectedminutes)):
            timeselection2 = np.logical_and(np.logical_and(hour2>=starthour+i,hour2<starthour+i+1),minute2 == selectedminutes[j])
            Temp67toaverage = Temp67_selected[timeselection2]
            Temp67_mean[i][j] = np.mean(Temp67toaverage)
            stdevTemp67_minutely[i][j] = np.std(Temp67toaverage)
######################################
###### end user input: load obs ######
######################################


#optimisation
priormodinput = fwdm.model_input()
###########################################
###### user input: prior model param ######
########################################### 
priormodinput.COS        = 0.400 #ppb
priormodinput.COSmeasuring_height = 125. - canopy_height
priormodinput.COSmeasuring_height2 = 23 - canopy_height
priormodinput.CO2measuring_height = 125. - canopy_height
priormodinput.CO2measuring_height2 = 23 - canopy_height
priormodinput.Tmeasuring_height = 67.2 - canopy_height #0 would be a problem
priormodinput.qmeasuring_height = 125. - canopy_height
priormodinput.qmeasuring_height2 = 23. - canopy_height
priormodinput.alfa_sto = 1
priormodinput.gciCOS = 0.2 /(1.2*1000) * 28.9
priormodinput.ags_C_mode = 'surf' 
priormodinput.sw_useWilson  = True
priormodinput.dt         = 30       # time step [s]
priormodinput.tstart     = starthour - 0.5   # time of the day [h UTC]
priormodinput.runtime    = (endhour-priormodinput.tstart)*3600 + priormodinput.dt   # total run time [s]
priormodinput.sw_ml      = True      # mixed-layer model switch
priormodinput.sw_shearwe = False     # shear growth mixed-layer switch
priormodinput.sw_fixft   = False     # Fix the free-troposphere switch
priormodinput.h          = 600.      # initial ABL height [m]
priormodinput.Ps         = Press_mean[0]*100   # surface pressure [Pa]
priormodinput.divU       = 0.00        # horizontal large-scale divergence of wind [s-1]
priormodinput.fc         = 1.e-4     # Coriolis parameter [m s-1],not used at the moment
priormodinput.theta      = 288.      # initial mixed-layer potential temperature [K]
priormodinput.deltatheta = 1.00       # initial temperature jump at h [K]
priormodinput.gammatheta = 0.008     # free atmosphere potential temperature lapse rate [K m-1]
priormodinput.advtheta   = 0.        # advection of heat [K s-1]
priormodinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
priormodinput.wtheta     = 0.17       # surface kinematic heat flux [K m s-1]
priormodinput.q          = 0.006     # initial mixed-layer specific humidity [kg kg-1]
priormodinput.deltaq     = -0.001    # initial specific humidity jump at h [kg kg-1]
priormodinput.gammaq     = 0.        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priormodinput.advq       = 0.        # advection of moisture [kg kg-1 s-1]
priormodinput.wq         = 0.07e-3    # surface kinematic moisture flux [kg kg-1 m s-1] 
priormodinput.CO2        = 410.      # initial mixed-layer CO2 [ppm]
priormodinput.deltaCO2   = -44.      # initial CO2 jump at h [ppm]
priormodinput.deltaCOS   = 0.050      # initial COS jump at h [ppb]
priormodinput.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
priormodinput.gammaCOS   = 0.00        # free atmosphere COS lapse rate [ppb m-1]
priormodinput.advCO2     = 0.        # advection of CO2 [ppm s-1]
priormodinput.advCOS     = 0.        # advection of COS [ppb s-1]
priormodinput.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]
priormodinput.wCOS       = 0.01        # surface kinematic COS flux [ppb m s-1]
priormodinput.sw_wind    = False     # prognostic wind switch
priormodinput.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
priormodinput.deltau     = 4.        # initial u-wind jump at h [m s-1]
priormodinput.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
priormodinput.advu       = 0.        # advection of u-wind [m s-2]
priormodinput.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
priormodinput.deltav     = 4.0       # initial u-wind jump at h [m s-1]
priormodinput.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
priormodinput.advv       = 0.        # advection of v-wind [m s-2]
priormodinput.sw_sl      = True     # surface layer switch
priormodinput.ustar      = 0.3       # surface friction velocity [m s-1]
priormodinput.z0m        = 1.0      # roughness length for momentum [m]
priormodinput.z0h        = 1.0     # roughness length for scalars [m]
priormodinput.sw_rad     = True     # radiation switch
priormodinput.lat        = 61.85     # latitude [deg]
priormodinput.lon        = 24.28     # longitude [deg]
priormodinput.doy        = 220.      # day of the year [-]
priormodinput.cc         = 0.0       # cloud cover fraction [-]
priormodinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priormodinput.sw_ls      = True     # land surface switch
priormodinput.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
priormodinput.wg         = 0.20      # volumetric water content top soil layer [m3 m-3]
priormodinput.w2         = 0.25      # volumetric water content deeper soil layer [m3 m-3]
priormodinput.cveg       = 0.9      # vegetation fraction [-]
priormodinput.Tsoil      = 290.      # temperature top soil layer [K]
priormodinput.T2         = TsoilA_mean[0]      # temperature deeper soil layer [K]
priormodinput.a          = 0.219     # Clapp and Hornberger retention curve parameter a
priormodinput.b          = 4.90      # Clapp and Hornberger retention curve parameter b
priormodinput.p          = 4.        # Clapp and Hornberger retention curve parameter c
priormodinput.CGsat      = 3.56e-6   # saturated soil conductivity for heat
priormodinput.wsat       = 0.61     # saturated volumetric water content (Sun 2017)
priormodinput.wfc        = 0.4     # volumetric water content field capacity [-]
priormodinput.wwilt      = 0.10     # volumetric water content wilting point [-]
priormodinput.C1sat      = 0.132     
priormodinput.C2ref      = 1.8
priormodinput.LAI        = 2.        # leaf area index [-]
priormodinput.gD         = None       # correction factor transpiration for VPD [-]
priormodinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]
priormodinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priormodinput.alpha      = 0.2      # surface albedo [-]
priormodinput.Ts         = 290.      # initial surface temperature [K]
priormodinput.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
priormodinput.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]
priormodinput.Lambda     = 5.9       # thermal diffusivity skin layer [-]
priormodinput.c3c4       = 'c3'      # Plant type ('c3' or 'c4')
priormodinput.sw_cu      = False     # Cumulus parameterization switch
priormodinput.dz_h       = 150.      # Transition layer thickness [m]
priormodinput.Cs         = 1e12      # drag coefficient for scalars [-]
priormodinput.sw_dynamicsl_border = True
priormodinput.sw_model_stable_con = True

#soil COS model
priormodinput.soilCOSmodeltype   = 'Sun_Ogee' #can be set to None or 'Sun_Ogee'
priormodinput.uptakemodel = 'Ogee'
priormodinput.sw_soilmoisture    = 'simple'
priormodinput.sw_soiltemp    = 'simple'
priormodinput.kH_type         = 'Sun'
priormodinput.Diffus_type     = 'Sun'
priormodinput.b_sCOSm = 5.3
priormodinput.fCA = 1e3
priormodinput.nr_nodes     = 26
priormodinput.s_moist_opt  = 0.20
priormodinput.Vspmax        = 1.e-10
priormodinput.Q10             = 3.
priormodinput.layer1_2division = 0.3
priormodinput.write_soilCOS_to_f = False
priormodinput.nr_nodes_for_filewr = 5

###############################################
###### end user input: prior model param ######
###############################################

#run priormodel to initialise properly
priormodel = fwdm.model(priormodinput)
priormodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
priorinput = cp.deepcopy(priormodinput) #below we can add some input necessary for the state in the optimisation, that is not part of the model input (a scale for some of the observations in the costfunction if desired).

###############################################################
###### user input: obs scales,state and list of used obs ######
###############################################################
state=['q','h','deltatheta','deltaCOS','theta','COS','gammatheta','gammaq','alfa_sto','CO2','deltaCO2','deltaq']
obslist =['COSmh','COSmh2','Tmh','qmh','qmh2','CO2mh','CO2mh2']
###################################################################
###### end user input: obs scales,state and list of used obs ######
###################################################################

if use_backgr_in_cost or use_ensemble:
    priorvar = {}   
###########################################################
###### user input: prior information (if used) ############
###########################################################
    #if not optim.use_backgr_in_cost, than these are only used for perturbing the ensemble (when use_ensemble = True)
    #prior variances of the items in the state:
    priorvar['alpha'] = 0.2**2
    priorvar['gammatheta'] = 0.003**2 
    priorvar['gammaq'] = (0.005e-3)**2 
    priorvar['deltatheta'] = 0.75**2
    priorvar['deltaq'] = 0.004**2
    priorvar['theta'] = 2**2
    priorvar['wtheta'] = 0.2**2
    priorvar['h'] = 300**2
    priorvar['q'] = 0.004**2
    priorvar['wg'] = 0.2**2
    priorvar['deltaCOS'] = 0.02**2
    priorvar['deltaCO2'] = 50**2
    priorvar['COS'] = 0.1**2
    priorvar['CO2'] = 80**2
    priorvar['alfa_sto'] = 0.3**2
    priorvar['fCA'] = 1.e3**2
    pars_priorvar = np.zeros(len(state))
    i = 0
    for item in state:
        pars_priorvar[i] = priorvar[item]
        i += 1
    b_cov = np.diag(pars_priorvar)  #b_cov stands for background covariance matrix, b already exists as model parameter
    #here we can specify covariances as well, in the background information matrix
    #this can be done by e.g. b_cov[5,1] = 0.5
###########################################################
###### end user input: prior information (if used) ########
###########################################################             
else:
     b_cov = None         

#create inverse modelling framework, do check,...
optim = im.adjoint_modelling(priormodel,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,imposeparambounds=True,state=state,pri_err_cov_matr=b_cov)
optim.obs = obslist
for item in priorinput.__dict__: #just a check
    if item.startswith('obs_sca_cf') and (item not in state):
        raise Exception(item +' given in priorinput, but not part of state. Remove from priorinput or add '+item+' to the state')
Hx_prior = {}
for item in optim.obs:
    Hx_prior[item] = priormodel.out.__dict__[item]
checkpoint_prior = priormodel.cpx
checkpoint_init_prior = priormodel.cpx_init    

#The observations
obs_times = {}
obs_weights = {}
obs_units = {}
display_names = {}
for item in optim.obs:
###########################################################
###### user input: observation information ################
########################################################### 
    #for each of the variables provided in the observation list, link the model output variable 
    #to the correct observations that were read in. Also, specify the times and observational errors, and optional weights 
    #Optionally, you can provide a display name here, a name which name will be shown for the observations in the plots
    #please use np.array or list as datastructure for the obs, obs errors, observation times or weights
    obs_times[item] = hour_selected * 3600
    if use_mean:
        if item == 'COSmh':
            optim.__dict__['obs_'+item] = COS_125_mean
            optim.__dict__['error_obs_' + item] = stdevCOS_125_hourly
            if use_weights:
                obs_weights[item] = [1.0 for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'COSmh2':
            optim.__dict__['obs_'+item] = COS_23_mean
            optim.__dict__['error_obs_' + item] = stdevCOS_23_hourly
#        elif item == 'Tmh':
#            optim.__dict__['obs_'+item] = Temp17_mean
#            optim.__dict__['error_obs_' + item] = stdevTemp17_hourly
        elif item == 'Tmh':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp67_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp67_minutely)
        elif item == 'CO2mh':
            optim.__dict__['obs_'+item] = CO2_125_mean
            optim.__dict__['error_obs_' + item] = stdevCO2_125_hourly
        elif item == 'CO2mh2':
            optim.__dict__['obs_'+item] = CO2_23_mean
            optim.__dict__['error_obs_' + item] = stdevCO2_23_hourly
        elif item == 'wCOS':
            optim.__dict__['obs_'+item] = F_EC_COS_mean
            optim.__dict__['error_obs_' + item] = stdevCOSflux_hourly
        elif item == 'wCOSS':
            optim.__dict__['obs_'+item] = soilCOS_flux_mean
            optim.__dict__['error_obs_' + item] = stdevsoilCOS_flux_hourly
        elif item == 'wCO2A':
            optim.__dict__['obs_'+item] = GPP_NEE_mean
            optim.__dict__['error_obs_' + item] = stdevGPP_NEE_hourly
        elif item == 'wCO2R':
            optim.__dict__['obs_'+item] = soilCO2_flux_mean
            optim.__dict__['error_obs_' + item] = stdevsoilCO2_flux_hourly
        elif item == 'qmh':
            optim.__dict__['obs_'+item] = H2O_125_mean
            optim.__dict__['error_obs_' + item] = stdevH2O_125_hourly
        elif item == 'qmh2':
            optim.__dict__['obs_'+item] = H2O_23_mean
            optim.__dict__['error_obs_' + item] = stdevH2O_23_hourly
        elif item == 'ustar':
            optim.__dict__['obs_'+item] = ustar_mean
            optim.__dict__['error_obs_' + item] = stdevustar_hourly
        elif item == 'H':
            optim.__dict__['obs_'+item] = H_flux_mean
            optim.__dict__['error_obs_' + item] = stdevH_flux_hourly
        elif item == 'LE':
            optim.__dict__['obs_'+item] = LE_flux_mean
            optim.__dict__['error_obs_' + item] = stdevLE_flux_hourly
###########################################################
###### end user input: observation information ############
########################################################### 
    if (not hasattr(optim,'obs_'+item) or not hasattr(optim,'error_obs_'+item)): #a check to see wether all info is specified
        raise Exception('Incomplete or no information on obs of ' + item)
    if type(optim.__dict__['obs_'+item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
        raise Exception('Please convert observation data into type \'numpy.ndarray\' or list!')
    if type(optim.__dict__['error_obs_'+item]) not in [np.ndarray,list]:
        raise Exception('Please convert observation error data into type \'numpy.ndarray\' or list!')
    if type(obs_times[item]) not in [np.ndarray,list]:
        raise Exception('Please convert observation time data into type \'numpy.ndarray\' or list!')
    if use_weights and item in obs_weights:
        if type(obs_weights[item]) not in [np.ndarray,list]:
            raise Exception('Please convert observation weight data into type \'numpy.ndarray\' or list!')
    itoremove = []
    for i in range(len(optim.__dict__['obs_'+item])):
        if math.isnan(optim.__dict__['obs_'+item][i]):
            itoremove += [i]
    optim.__dict__['obs_'+item] = np.delete(optim.__dict__['obs_'+item],itoremove) #exclude the nan obs
    optim.__dict__['error_obs_'+item] = np.delete(optim.__dict__['error_obs_'+item],itoremove) #as a side effect, this turns the array into an numpy.ndarray if not already the case (or gives error).
    obs_times[item] = np.delete(obs_times[item],itoremove)#exclude the times,errors and weights as well (of the nan obs)
    if item in obs_weights:
        obs_weights[item] = np.delete(obs_weights[item],itoremove)        
    if (use_backgr_in_cost and use_weights): #add weight of obs vs prior (identical for every obs) in the cost function
        if item in obs_weights: #if already a weight specified for the specific type of obs
            obs_weights[item] = [x * obs_vs_backgr_weight for x in obs_weights[item]]
        else:
            obs_weights[item] = [obs_vs_backgr_weight for x in range(len(optim.__dict__['obs_'+item]))] #nans are already excluded in the obs at this stage, so no problem with nan
    if use_weights:
        if item in obs_weights: #if already a weight specified for the specific type of obs
            for i in range(len(obs_times[item])):
                if obs_times[item][i] < 10 * 3600:
                    obs_weights[item][i] = obs_weights[item][i] * weight_morninghrs
        else:
            obs_weights[item] = np.ones(len(optim.__dict__['obs_'+item]))
            for i in range(len(obs_times[item])):
                if obs_times[item][i] < 10 * 3600:
                    obs_weights[item][i] = weight_morninghrs #nans are already excluded in the obs at this stage, so no problem with nan

        
    if len(obs_times[item]) != len(optim.__dict__['obs_'+item]):
        raise Exception('Error: size of obs and obstimes inconsistent!')
    for num in obs_times[item]:
        if round(num, 10) not in [round(num2, 10) for num2 in priormodel.out.t * 3600]:
            raise Exception('Error: obs occuring at a time that is not modelled (' + str(item) +')')
    
print('total number of obs:')
number_of_obs = 0
for item in optim.obs:
    number_of_obs += len(optim.__dict__['obs_'+item])
print(number_of_obs)
if use_weights:
    sum_of_weights = 0
    for item in optim.obs: 
        if item in obs_weights:
            sum_of_weights += np.sum(obs_weights[item])#need sum, the weights are an array for every item
        else:
            sum_of_weights += len(optim.__dict__['obs_'+item])
    print('total number of obs, corrected for weights:')
    print(sum_of_weights)
print('number of params to optimise:')
number_of_params = len(state)
print(number_of_params)
########################################
obs_sca_cf = {}
optim.pstate = []
for item in state:
    optim.pstate.append(priorinput.__dict__[item])
    if item.startswith('obs_sca_cf_'):
        obsname = item.split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
        obs_sca_cf[obsname] = cp.deepcopy(priorinput.__dict__[item])
optim.pstate = np.array(optim.pstate)
optiminput = cp.deepcopy(priorinput) #deepcopy!
params = tuple([optiminput,state,obs_times,obs_weights])
optim.checkpoint = cp.deepcopy(checkpoint_prior) #needed, as first thing optimizer does is calculating the gradient
optim.checkpoint_init = cp.deepcopy(checkpoint_init_prior) #needed, as first thing optimizer does is calculating the gradient
for item in optim.obs:
    if item in obs_sca_cf:
        obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
    else:
        obs_scale = 1.0 
    weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
    k = 0 #index for observations
    for ti in range(priormodel.tsteps):
        if round(priormodel.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[item]]: #so if we are at a time where we have an obs        
            if item in obs_weights:
                weight = obs_weights[item][k]
            forcing = weight * (Hx_prior[item][ti] - obs_scale * optim.__dict__['obs_'+item][k])/(optim.__dict__['error_obs_' + item][k]**2)
            optim.forcing[ti][item] = forcing
            k += 1
if optim_method == 'bfgs':
    if ana_deriv:
        minimisation = optimize.fmin_bfgs(optim.min_func,optim.pstate,fprime=optim.ana_deriv,args=params,gtol=1.e-9,full_output=True)
    else:
        minimisation = optimize.fmin_bfgs(optim.min_func,optim.pstate,fprime=optim.num_deriv,args=params,gtol=1.e-9,full_output=True)
    state_opt0 = minimisation[0]
    min_costf = minimisation[1]
elif optim_method == 'tnc':
    bounds = []
    for i in range(len(state)):
        for key in optim.boundedvars:
            if key == state[i]:
                bounds.append((optim.boundedvars[key][0],optim.boundedvars[key][1]))
        if state[i] not in optim.boundedvars:
            bounds.append((None,None)) #bounds need something
    try:
        if ana_deriv:
            minimisation = optimize.fmin_tnc(optim.min_func,optim.pstate,fprime=optim.ana_deriv,args=params,bounds=bounds,maxfun=None)
        else:
            minimisation = optimize.fmin_tnc(optim.min_func,optim.pstate,fprime=optim.num_deriv,args=params,bounds=bounds,maxfun=None)
        state_opt0 = minimisation[0]
        min_costf = optim.cost_func(state_opt0,optiminput,state,obs_times,obs_weights)
    except (im.nan_incostfError):
        print('Minimisation aborted due to nan')
        open('Optimfile.txt','a').write('\n')
        open('Optimfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
        open('Gradfile.txt','a').write('\n')
        open('Gradfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
        if (discard_nan_minims == False and len(optim.Statelist) > 0): #len(optim.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
            min_costf = np.min(optim.Costflist)
            min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the number of the simulation where costf was minimal
            state_opt0 = optim.Statelist[min_costf_ind]
            optim.stop = True
        else:
            state_opt0 = np.array([np.nan for x in range(len(state))])
            min_costf = np.nan
    except (im.static_costfError):
        print('Minimisation aborted as it proceeded too slow')
        open('Optimfile.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
        open('Gradfile.txt','a').write('\nMinimisation aborted as it proceeded too slow')
        min_costf = np.min(optim.Costflist)
        min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the number of the simulation where costf was minimal
        state_opt0 = optim.Statelist[min_costf_ind]
    for i in range(maxnr_of_restarts):
        if (min_costf > stopcrit and (not hasattr(optim,'stop'))): #will evaluate to False if min_costf is equal to nan
            optim.nr_of_sim_bef_restart = optim.sim_nr
            open('Optimfile.txt','a').write('\n')
            open('Optimfile.txt','a').write('{0:>25s}'.format('restart'))
            open('Gradfile.txt','a').write('\n')
            open('Gradfile.txt','a').write('{0:>25s}'.format('restart'))
            try:
                if ana_deriv:
                    minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                else:
                    minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                state_opt0 = minimisation[0]
            except (im.nan_incostfError):
                print('Minimisation aborted due to nan, no restart')
                open('Optimfile.txt','a').write('\nnan reached, no restart')
                open('Gradfile.txt','a').write('\nnan reached, no restart')
                if discard_nan_minims == False:
                    min_costf = np.min(optim.Costflist)
                    min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the number of the simulation where costf was minimal
                    state_opt0 = optim.Statelist[min_costf_ind]
                else:
                    state_opt0 = np.array([np.nan for x in range(len(state))])
                    min_costf = np.nan
                break
            except (im.static_costfError):
                print('Minimisation aborted as it proceeded too slow')
                open('Optimfile.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                open('Gradfile.txt','a').write('\nMinimisation aborted as it proceeded too slow')
                min_costf = np.min(optim.Costflist)
                min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the number of the simulation where costf was minimal
                state_opt0 = optim.Statelist[min_costf_ind]
            for j in range (len(state)):
                optiminput.__dict__[state[j]] = state_opt0[j]
            min_costf = optim.cost_func(state_opt0,optiminput,state,obs_times,obs_weights)
    open('Optimfile.txt','a').write('\n')
    open('Optimfile.txt','a').write('{0:>25s}'.format('finished'))
print('optimal state without ensemble='+str(state_opt0))

def run_ensemble_member(counter,seed):
    priorinput_mem = cp.deepcopy(priorinput)
    for j in range(len(state)):
        np.random.seed(seed) #VERY IMPORTANT! You have to explicitly set the seed (to None is ok), otherwise multicore implementation will use same random number for all ensemble members. 
        rand_nr_norm_distr = np.random.normal(0,np.sqrt(pars_priorvar[j]))
        priorinput_mem.__dict__[state[j]] += rand_nr_norm_distr
        if optim.imposeparambounds:
            if state[j] in optim.boundedvars:
                if priorinput_mem.__dict__[state[j]] < optim.boundedvars[state[j]][0]: #lower than lower bound
                    priorinput_mem.__dict__[state[j]] = optim.boundedvars[state[j]][0] #so to make it within the bounds
                elif priorinput_mem.__dict__[state[j]] > optim.boundedvars[state[j]][1]: #higher than upper bound
                    priorinput_mem.__dict__[state[j]] = optim.boundedvars[state[j]][1]
    priormodel_mem = fwdm.model(priorinput_mem)
#    open('errorfile'+str(counter)+'.txt','w').write('{0:>25s}'.format('state'))
#    open('errorfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n'))
#    for j in range(len(state)):
#        open('errorfile'+str(counter)+'.txt','a').write('{0:>25s}'.format(state[j]))
#    open('errorfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n'))
#    for j in range(len(state)):
#        open('errorfile'+str(counter)+'.txt','a').write('{0:>25s}'.format(str(priorinput_mem.__dict__[state[j]])))
    priormodel_mem.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
    optim_mem = im.adjoint_modelling(priormodel_mem,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,imposeparambounds=imposeparambounds,state=state,Optimfile='Optimfile'+str(counter)+'.txt',Gradfile='Gradfile'+str(counter)+'.txt',pri_err_cov_matr=b_cov)
    optim_mem.obs = cp.deepcopy(optim.obs)
    for item in optim_mem.obs:
        optim_mem.__dict__['obs_'+item] = cp.deepcopy(optim.__dict__['obs_'+item])
    Hx_prior_mem = {}
    for item in optim_mem.obs:
        Hx_prior_mem[item] = priormodel_mem.out.__dict__[item]
        optim_mem.__dict__['error_obs_' + item] = cp.deepcopy(optim.__dict__['error_obs_' + item])
    if est_post_pdf:
        for item in optim_mem.obs:
            np.random.seed(seed)  
            rand_nr_list = ([np.random.normal(0,optim_mem.__dict__['error_obs_' + item][i]) for i in range(len(optim_mem.__dict__['error_obs_' + item]))])
            optim_mem.__dict__['obs_'+item] += rand_nr_list
        if plot_perturbed_obs:
            for i in range(len(optim.obs)):
                plt.figure()
                plt.plot(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black',label = 'orig')
                plt.errorbar(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]],yerr=optim.__dict__['error_obs_'+optim.obs[i]],ecolor='black',fmt='None')
                plt.plot(obs_times[optim_mem.obs[i]]/3600,optim_mem.__dict__['obs_'+optim_mem.obs[i]], linestyle=' ', marker='o',color = 'red',label = 'pert')
                if optim.obs[i] in display_names:
                    plt.ylabel(display_names[optim.obs[i]] +' ('+ obs_units[optim.obs[i]] + ')')
                else:
                    plt.ylabel(optim.obs[i] +' ('+ obs_units[optim.obs[i]] + ')')
                plt.xlabel('time (h)')
                plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
                plt.legend()
                if write_to_f:
                    plt.savefig('fig_obs_'+optim.obs[i]+'_mem'+str(counter)+'.png', format='png')
    checkpoint_prior_mem = priormodel_mem.cpx
    checkpoint_init_prior_mem = priormodel_mem.cpx_init
    optim_mem.pstate = []
    for item in state:
        optim_mem.pstate.append(priorinput_mem.__dict__[item])
    optim_mem.pstate = np.array(optim_mem.pstate)
    optiminput_mem = cp.deepcopy(priorinput_mem) #deepcopy!
    params = tuple([optiminput_mem,state,obs_times,obs_weights])
    optim_mem.checkpoint = cp.deepcopy(checkpoint_prior_mem) #needed, as first thing optimizer does is calculating the gradient
    optim_mem.checkpoint_init = cp.deepcopy(checkpoint_init_prior_mem) #needed, as first thing optimizer does is calculating the gradient
    for item in optim_mem.obs:
        weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
        k = 0 #counter for the observations (specific for each type of obs) 
        for ti in range(priormodel_mem.tsteps):
            if round(priormodel_mem.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[item]]: #so if we are at a time where we have an obs
                if item in obs_weights:
                    weight = obs_weights[item][k]
                forcing = weight * (Hx_prior_mem[item][ti]-optim_mem.__dict__['obs_'+item][k])/(optim_mem.__dict__['error_obs_' + item][k]**2)
                optim_mem.forcing[ti][item] = forcing
                k += 1
    if optim_method == 'bfgs':
        if ana_deriv:
            minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params,gtol=1.e-9,full_output=True)
        else:
            minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params,gtol=1.e-9,full_output=True)
        state_opt_mem = minimisation_mem[0]
        min_costf_mem = minimisation_mem[1]
    elif optim_method == 'tnc':
        try:
            if ana_deriv:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params,bounds=bounds,maxfun=None)
            else:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params,bounds=bounds,maxfun=None)
            state_opt_mem = minimisation_mem[0]
            min_costf_mem = optim_mem.cost_func(state_opt_mem,optiminput_mem,state,obs_times,obs_weights)    
        except (im.nan_incostfError):
            print('Minimisation aborted due to nan')
            open('Optimfile'+str(counter)+'.txt','a').write('\n')
            open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
            open('Gradfile'+str(counter)+'.txt','a').write('\n')
            open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
            if (discard_nan_minims == False and len(optim_mem.Statelist) > 0): #len(optim_mem.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf_mem = np.min(optim_mem.Costflist)
                min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the number of the simulation where costf was minimal
                state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                optim_mem.stop = True
            else:
                state_opt_mem = np.array([np.nan for x in range(len(state))])
                min_costf_mem = np.nan
        except (im.static_costfError):
            print('Minimisation aborted as it proceeded too slow')
            open('Optimfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
            open('Gradfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow')
            min_costf_mem = np.min(optim_mem.Costflist)
            min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the number of the simulation where costf was minimal
            state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
        for i in range(maxnr_of_restarts):
            if (min_costf_mem > stopcrit and (not hasattr(optim_mem,'stop'))): #will evaluate to False if min_costf_mem is equal to nan
                optim_mem.nr_of_sim_bef_restart = optim_mem.sim_nr
                open('Optimfile'+str(counter)+'.txt','a').write('\n')
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('restart'))
                open('Gradfile'+str(counter)+'.txt','a').write('\n')
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('restart'))
                try:
                    if ana_deriv:
                        minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                    else:
                        minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                    state_opt_mem = minimisation_mem[0]
                except (im.nan_incostfError):
                    print('Minimisation aborted due to nan, no restart for this member')
                    open('Optimfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                    open('Gradfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                    if discard_nan_minims == False:
                        min_costf_mem = np.min(optim_mem.Costflist)
                        min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the number of the simulation where costf was minimal
                        state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                    else:
                        state_opt_mem = np.array([np.nan for x in range(len(state))])
                        min_costf_mem = np.nan
                    break
                except (im.static_costfError):
                    print('Minimisation aborted as it proceeded too slow')
                    open('Optimfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                    open('Gradfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow')
                    min_costf_mem = np.min(optim_mem.Costflist)
                    min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the number of the simulation where costf was minimal
                    state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                for j in range (len(state)):
                    optiminput_mem.__dict__[state[j]] = state_opt_mem[j]
                min_costf_mem = optim_mem.cost_func(state_opt_mem,optiminput_mem,state,obs_times,obs_weights)
    open('Optimfile'+str(counter)+'.txt','a').write('\n')
    open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('finished'))
    return min_costf_mem,state_opt_mem
   
if use_ensemble:
    ensemble = []
    for i in range(0,nr_of_members): #the zeroth is the one done before
        ensemble.append({})
    ensemble[0]['min_costf'] = min_costf
    ensemble[0]['state_opt'] = state_opt0   
    shutil.copyfile('Optimfile.txt', 'Optimfile_'+str(0)+'.txt')
    shutil.copyfile('Gradfile.txt', 'Gradfile_'+str(0)+'.txt')
    if run_multicore:
        num_cores = multiprocessing.cpu_count()
        result_array = Parallel(n_jobs=num_cores)(delayed(run_ensemble_member)(i,None)  for i in range(1,nr_of_members)) #, prefer="threads" makes it work, but probably not multiprocess. None is for the seed
        #the above returns a list of tuples
        for j in range(1,nr_of_members):
            ensemble[j]['min_costf'] = result_array[j-1][0] #-1 due to the fact that the zeroth ensemble member is not part of the result_array, while it is part of ensemble
            ensemble[j]['state_opt'] = result_array[j-1][1]
    else:
        for i in range(1,nr_of_members):
            ensemble[i]['min_costf'],ensemble[i]['state_opt'] =  run_ensemble_member(i,None)
    print('whole ensemble:')
    print(ensemble)
    seq = np.array([x['min_costf'] for x in ensemble]) #iterate over the dictionaries
    min_costf_ensemble = np.nanmin(seq)
    opt_sim_nr = np.where(seq == min_costf_ensemble)[0][0]
    state_opt = ensemble[opt_sim_nr]['state_opt']
    print('optimal state ensemble '+ str(state) +':')
    print(state_opt)
    print('index of member with the best state:')
    print(opt_sim_nr)
    if est_post_pdf:
        stdev_post = np.zeros(len(state))
        for i in range(len(state)):
            seq = np.array([x['state_opt'][i] for x in ensemble]) #iterate over the dictionaries,gives array 
            mean_state_i_opt = np.nanmean(seq) 
            stdev_post[i] = np.nanstd(seq)
            x = np.linspace(mean_state_i_opt - 3*stdev_post[i], mean_state_i_opt + 3*stdev_post[i], 100)
            fig = plt.figure()
            plt.plot(x,stats.norm.pdf(x, mean_state_i_opt, stdev_post[i]), linestyle='-', marker='None',color='black',label='post')
            plt.ylabel('pdf (-)')
            plt.xlabel(state[i])
            #add prior pdf
            x2 = np.linspace(priorinput.__dict__[state[i]] - 3*np.sqrt(pars_priorvar[i]), priorinput.__dict__[state[i]] + 3*np.sqrt(pars_priorvar[i]), 100)
            plt.plot(x2,stats.norm.pdf(x2, priorinput.__dict__[state[i]], np.sqrt(pars_priorvar[i])), linestyle='-', marker='None',color='yellow',label='prior')
            plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
            plt.legend(loc=0, frameon=True,prop={'size':20})
            if write_to_f:
                plt.savefig('pdf_posterior_'+state[i]+'.png', format='png')

optimalinput = cp.deepcopy(priorinput)
i = 0
for item in state:
    if use_ensemble:
        optimalinput.__dict__[item] = state_opt[i]
    else:
        optimalinput.__dict__[item] = state_opt0[i]
    i += 1
optimalmodel = fwdm.model(optimalinput)
optimalmodel.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)

############################
#stats file
############################
if use_ensemble:
    if use_backgr_in_cost:
        if use_weights:
            chi_sq = min_costf_ensemble/(sum_of_weights + number_of_params)
        else:
            chi_sq = min_costf_ensemble/(number_of_obs + number_of_params) #calculation of chi squared statistic
    else:
        if use_weights:
            chi_sq = min_costf_ensemble/(sum_of_weights)
        else:
            chi_sq = min_costf_ensemble/(number_of_obs)
else:
    if use_backgr_in_cost:
        if use_weights:
            chi_sq = min_costf/(sum_of_weights + number_of_params)
        else:
            chi_sq = min_costf/(number_of_obs + number_of_params) #calculation of chi squared statistic
    else:
        if use_weights:
            chi_sq = min_costf/(sum_of_weights)
        else:
            chi_sq = min_costf/(number_of_obs)
if write_to_f:
    open('Optstatsfile.txt','w').write('{0:>9s}'.format('nr of obs')) #here we make the file   
    if use_weights:
        open('Optstatsfile.txt','a').write('{0:>40s}'.format('total nr obs, corrected for weights'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('number of params to optimise'))
    open('Optstatsfile.txt','a').write('{0:>25s}'.format('chi squared'))
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(' (of member with lowest costf)'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>9s}'.format(str(number_of_obs)))
    if use_weights:
        open('Optstatsfile.txt','a').write('{0:>40s}'.format(str(sum_of_weights)))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format(str(number_of_params)))
    open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(chi_sq)))
    open('Optstatsfile.txt','a').write('\n\n')
    open('Optstatsfile.txt','a').write('{0:>31s}'.format('optimal state without ensemble:'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('      ')
    for item in state:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('      ')
    for item in state_opt0:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
    open('Optstatsfile.txt','a').write('\n')
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>31s}'.format('optimal state with ensemble:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('      ')
        for item in state_opt:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>31s}'.format('index member with best state:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>31s}'.format(str(opt_sim_nr)))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>9s}'.format('ensemble members:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('minimum costf'))
        for item in state:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Optstatsfile.txt','a').write('\n')
        for item in ensemble:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item['min_costf'])))
            for param in item['state_opt']:
                open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(param)))
            open('Optstatsfile.txt','a').write('\n')

for i in range(len(obslist)):
    fig = plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow')
    plt.plot(priormodel.out.t,optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red')
    plt.plot(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black')
    plt.errorbar(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]],yerr=optim.__dict__['error_obs_'+optim.obs[i]],ecolor='black',fmt='None')
    if 'obs_sca_cf_'+obslist[i] in state: #plot the obs scaled with the scaling factors (if determined)
        plt.plot(obs_times[obslist[i]]/3600,optimalinput.__dict__['obs_sca_cf_'+obslist[i]]*optim.__dict__['obs_'+obslist[i]], linestyle=' ', marker='o',color = 'red')
    plt.ylabel(optim.obs[i])
    plt.xlabel('time (h)')
    plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    if write_to_f:
        plt.savefig('fig_fit_'+optim.obs[i]+'.png', format='png')

########################################################
###### user input: additional plotting etc. ############
######################################################## 

#fig = plt.figure()
#plt.plot(optimalmodel.out.t,optimalmodel.out.COSsurf, linestyle='--', marker='o',color='red')
#plt.plot(optimalmodel.out.t,optimalmodel.out.COSmh, linestyle='--', marker='o',color='green')
#plt.plot(optimalmodel.out.t,optimalmodel.out.COSmh2, linestyle='--', marker='o',color='orange')
#plt.plot(optimalmodel.out.t,optimalmodel.out.COS, linestyle='--', marker='o',color='blue')
#
#fig = plt.figure()
#plt.plot(optimalmodel.out.t,optimalmodel.out.CO2surf, linestyle='--', marker='o',color='red')
#plt.plot(optimalmodel.out.t,optimalmodel.out.CO2mh, linestyle='--', marker='o',color='green')
#plt.plot(optimalmodel.out.t,optimalmodel.out.CO2mh2, linestyle='--', marker='o',color='orange')
#plt.plot(optimalmodel.out.t,optimalmodel.out.CO2, linestyle='--', marker='o',color='blue')
#
#fig = plt.figure()
#plt.plot(optimalmodel.out.t,optimalmodel.out.T2m, linestyle='--', marker='o',color='yellow')
#plt.plot(optimalmodel.out.t,optimalmodel.out.Tmh, linestyle='--', marker='o',color='red')
#plt.plot(optimalmodel.out.t,optimalmodel.out.Tsurf, linestyle='--', marker='o',color='blue')
#
fig = plt.figure()
plt.plot(optimalmodel.out.t,priormodel.out.wCOS, linestyle='--', marker='o',color='yellow')
plt.plot(optimalmodel.out.t,optimalmodel.out.wCOS, linestyle='--', marker='o',color='red')
plt.plot(hour_selected,F_EC_COS_mean, linestyle=' ', marker='o',color='black')
plt.ylabel('COS surface flux (ppb m s-1)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCOS.png', format='png')

fig = plt.figure()
plt.plot(optimalmodel.out.t,priormodel.out.wCO2A, linestyle='--', marker='o',color='yellow')
plt.plot(optimalmodel.out.t,optimalmodel.out.wCO2A, linestyle='--', marker='o',color='red')
plt.plot(hour_selected,GPP_NEE_mean, linestyle=' ', marker='o',color='black')
plt.ylabel('GPP (mg CO2/m2/s)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCO2A.png', format='png')

fig = plt.figure()
plt.plot(optimalmodel.out.t,priormodel.out.wCO2R, linestyle='--', marker='o',color='yellow')
plt.plot(optimalmodel.out.t,optimalmodel.out.wCO2R, linestyle='--', marker='o',color='red')
plt.plot(hour_selected,soilCO2_flux_mean, linestyle=' ', marker='o',color='black')
plt.ylabel('Resp (mg CO2/m2/s)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCO2R.png', format='png')
    
