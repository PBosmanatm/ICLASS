# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:12:49 2019

@author: Bosman Peter
"""

import forwardmodel as fwdm
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time

use_mean=True
write_to_f = False
Avogadro = 6.022141 *1e23
Planck   = 6.62607 *1e-34
light_vel = 299792458
wavel_PAR = 0.55e-6

#read data
start = 1000 #line where we start in csv file (-1)
selectedyears = [2015.]
selectedmonths = [8.]
selectedday = 18. #only used when not use_mean = True
#selecteddays = [18,20,21,22,23,24,25] #period over which we average, this is a sunny period
selecteddays = [18,20,21,22,23,24,25]
selectedminutes = [0]#range(0,60) #only for second data file. use an array!
starthour = 5 #utc, start of obs (6)
endhour = 16#16
endhour_mod = endhour 

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
q_125 = np.zeros(len(data))
COS_23 = np.zeros(len(data))
CO2_23 = np.zeros(len(data))
H2O_23 = np.zeros(len(data))
Temp17 = np.zeros(len(data))
Press = np.zeros(len(data))
TsoilA = np.zeros(len(data))
Rd  = 287.04
F_EC_COS = np.zeros(len(data))
F_EC_CO2 = np.zeros(len(data))
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
PAR = np.zeros(len(data))
solarang = np.zeros(len(data))
for i in range(len(data)):
    hour[i]   = data[i][0][11:13]
    hour[i]   = hour[i] - 2 +0.5 #convert to utc. STAY AWAY FROM HOURS AFTER MIDNIGHT,ERRORS IN THAT CASE!! 0.5 because data are an average over the following hour
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
    if data[i][17] == 'NA':
        data[i][17] = float('NaN')
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
    if data[i][31] == 'NA':
        data[i][31] = float('NaN')
    COS_125[i]    = data[i][1]
    COS_125[i] = COS_125[i] / 1000 #ppb
    CO2_125[i]    = data[i][2] 
    CO2_125[i] = CO2_125[i] #ppm
    H2O_125[i] = data[i][3]
    H2O_125[i] = H2O_125[i] /100 #mole fraction
    q_125[i] = H2O_125[i] / 28.97e-3 * 18e-3 #mol_H2O/mol_air * mol_air/kg_air * kg_H2O/mol_H2O gives kg_H2O / kg_air
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
    F_EC_CO2[i] = data[i][17] #umol/m2/s
    F_EC_CO2[i] = F_EC_CO2[i] / rho * 28.97e-3    #µmol/m2/s * m3/kg_air * kg_air/mol_air = ppm m/s; rho we used not fully correct
    GPP_NEE[i] = data[i][21] #µmol/m2/s
    GPP_NEE[i] = GPP_NEE[i] * 44 * 1000*1e-6 #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2
    rho_for_soil = (Press[i]*100) / (Rd*Temp17[i]) #temp not correct actually...
    soilCOS_flux1[i] = float(data[i][22]) / rho_for_soil * 28.97e-3 /1000 # * m3/kg_air * kg_air/mol_air = ppt m/s; /1000 gives ppb m/s; rho we used not fully correct
    soilCOS_flux2[i] = float(data[i][26]) / rho_for_soil * 28.97e-3 /1000 # * m3/kg_air * kg_air/mol_air = ppt m/s; /1000 gives ppb m/s; rho we used not fully correct
    soilCO2_flux1[i] = float(data[i][23]) * 44 * 1000*1e-6   #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2 
    soilCO2_flux2[i] = float(data[i][27]) * 44 * 1000*1e-6   #mg CO2/m2/s = umol/m2/s * g_co2/mol_co2 * mg/g * mol_co2/umol_co2 
    PAR[i]      = data[i-1][31] #micromol/m²/s
    PAR[i]      = PAR[i] /1e6 * Avogadro * Planck * light_vel / wavel_PAR #in W/m2
    solarang[i]      = data[i][30] #degrees; the solarangle from the measurement file, though not an actual measurement probably

if use_mean != True:
    COS_125_selected = np.zeros(endhour-starthour)
    CO2_125_selected = np.zeros(endhour-starthour)
    H2O_125_selected = np.zeros(endhour-starthour)
    q_125_selected = np.zeros(endhour-starthour)
    COS_23_selected = np.zeros(endhour-starthour)
    CO2_23_selected = np.zeros(endhour-starthour)
    H2O_23_selected = np.zeros(endhour-starthour)
    hour_selected = np.zeros(endhour-starthour)
    F_EC_COS_selected = np.zeros(endhour-starthour)
    F_EC_CO2_selected = np.zeros(endhour-starthour)
    GPP_NEE_selected = np.zeros(endhour-starthour)
    Temp17_selected = np.zeros(endhour-starthour)
    Press_selected = np.zeros(endhour-starthour)
    ustar_selected = np.zeros(endhour-starthour)
    H_flux_selected = np.zeros(endhour-starthour)
    LE_flux_selected = np.zeros(endhour-starthour)
    TsoilA_selected = np.zeros(endhour-starthour)
    soilCOS_flux_selected = np.zeros(endhour-starthour)
    soilCO2_flux_selected = np.zeros(endhour-starthour)
    PAR_selected = np.zeros(endhour-starthour)
    solarang_selected = np.zeros(endhour-starthour)
    j = 0
    for i in range(len(data)):
        if ((year[i] in selectedyears) and (month[i] in selectedmonths)) and ((day[i] == selectedday) and (starthour<=hour[i]<=endhour)):
            COS_125_selected[j] = COS_125[i]
            CO2_125_selected[j] = CO2_125[i]
            H2O_125_selected[j] = H2O_125[i]
            q_125_selected[j]  = q_125[i]
            COS_23_selected[j] = COS_23[i]
            CO2_23_selected[j] = CO2_23[i]
            H2O_23_selected[j] = H2O_23[i]
            hour_selected[j]    = hour[i]
            F_EC_COS_selected[j] = F_EC_COS[i]
            F_EC_CO2_selected[j] = F_EC_CO2[i]
            if F_EC_COS_flag[i] not in [0,1]:
                F_EC_COS_selected[j] = float('NaN')
            GPP_NEE_selected[j] = GPP_NEE[i]
            if F_EC_CO2_flag[i] not in [0,1]:
                GPP_NEE_selected[j] = float('NaN')
                F_EC_CO2_selected[j] = float('NaN')
            Temp17_selected[j] = Temp17[i]
            Press_selected[j] = Press[i]
            ustar_selected[j] = ustar[i]
            H_flux_selected[j] = H_flux[i]
            LE_flux_selected[j] = LE_flux[i]
            TsoilA_selected[j] = TsoilA[i]
            soilCOS_flux_selected[j] = np.nanmean([soilCOS_flux1[i],soilCOS_flux2[i]])
            soilCO2_flux_selected[j] = np.nanmean([soilCO2_flux1[i],soilCO2_flux2[i]])
            PAR_selected[j] = PAR[i]
            solarang_selected[j] = solarang[i]
            j +=1
else:
    COS_125_selected = np.zeros((len(selecteddays),endhour-starthour))
    CO2_125_selected = np.zeros((len(selecteddays),endhour-starthour))
    H2O_125_selected = np.zeros((len(selecteddays),endhour-starthour))
    q_125_selected = np.zeros((len(selecteddays),endhour-starthour))
    COS_23_selected = np.zeros((len(selecteddays),endhour-starthour))
    CO2_23_selected = np.zeros((len(selecteddays),endhour-starthour))
    H2O_23_selected = np.zeros((len(selecteddays),endhour-starthour))
    F_EC_COS_selected = np.zeros((len(selecteddays),endhour-starthour))
    F_EC_CO2_selected = np.zeros((len(selecteddays),endhour-starthour))
    GPP_NEE_selected = np.zeros((len(selecteddays),endhour-starthour))
    Temp17_selected = np.zeros((len(selecteddays),endhour-starthour))
    Press_selected = np.zeros((len(selecteddays),endhour-starthour))
    ustar_selected = np.zeros((len(selecteddays),endhour-starthour))
    H_flux_selected = np.zeros((len(selecteddays),endhour-starthour))
    LE_flux_selected = np.zeros((len(selecteddays),endhour-starthour))
    TsoilA_selected = np.zeros((len(selecteddays),endhour-starthour))
    soilCOS_flux_selected = np.zeros((len(selecteddays),endhour-starthour))
    soilCO2_flux_selected = np.zeros((len(selecteddays),endhour-starthour))
    PAR_selected = np.zeros((len(selecteddays),endhour-starthour))
    solarang_selected = np.zeros((len(selecteddays),endhour-starthour))
    hour_selected = np.zeros(endhour-starthour)
    daycounter = 0
    hourcounter = 0
    for i in range(len(data)):
        if ((year[i] in selectedyears) and (month[i] in selectedmonths)) and ((day[i] in selecteddays) and (starthour<=hour[i]<=endhour)):
            COS_125_selected[daycounter,hourcounter] = COS_125[i]
            CO2_125_selected[daycounter,hourcounter] = CO2_125[i]
            H2O_125_selected[daycounter,hourcounter] = H2O_125[i]
            q_125_selected[daycounter,hourcounter] = q_125[i]
            COS_23_selected[daycounter,hourcounter] = COS_23[i]
            CO2_23_selected[daycounter,hourcounter] = CO2_23[i]
            H2O_23_selected[daycounter,hourcounter] = H2O_23[i]
            F_EC_COS_selected[daycounter,hourcounter] = F_EC_COS[i]
            F_EC_CO2_selected[daycounter,hourcounter] = F_EC_CO2[i]
            if F_EC_COS_flag[i] not in [0,1]:
                F_EC_COS_selected[daycounter,hourcounter] = float('NaN')
            GPP_NEE_selected[daycounter,hourcounter] = GPP_NEE[i]
            if F_EC_CO2_flag[i] not in [0,1]:
                GPP_NEE_selected[daycounter,hourcounter] = float('NaN')
                F_EC_CO2_selected[daycounter,hourcounter] = float('NaN')
            hour_selected[hourcounter] = hour[i]
            Temp17_selected[daycounter,hourcounter] = Temp17[i]
            Press_selected[daycounter,hourcounter] = Press[i]
            ustar_selected[daycounter,hourcounter] = ustar[i]
            H_flux_selected[daycounter,hourcounter] = H_flux[i]
            LE_flux_selected[daycounter,hourcounter] = LE_flux[i]
            TsoilA_selected[daycounter,hourcounter] = TsoilA[i]
            soilCOS_flux_selected[daycounter,hourcounter] = np.nanmean((soilCOS_flux1[i],soilCOS_flux2[i]))
            soilCO2_flux_selected[daycounter,hourcounter] = np.nanmean((soilCO2_flux1[i],soilCO2_flux2[i]))
            PAR_selected[daycounter,hourcounter] = PAR[i]
            solarang_selected[daycounter,hourcounter] = solarang[i]
            hourcounter +=1
            if hour[i]+1 >= endhour:#obs at every half hour, so if next is a new day increase daycounter#if the hour would be missing in the obs file we have a problem!!
                daycounter += 1
                hourcounter = 0
    COS_125_mean = np.zeros(endhour-starthour)
    CO2_125_mean = np.zeros(endhour-starthour)
    H2O_125_mean = np.zeros(endhour-starthour)
    q_125_mean = np.zeros(endhour-starthour)
    COS_23_mean = np.zeros(endhour-starthour)
    CO2_23_mean = np.zeros(endhour-starthour)
    H2O_23_mean = np.zeros(endhour-starthour)
    F_EC_COS_mean = np.zeros(endhour-starthour)
    F_EC_CO2_mean = np.zeros(endhour-starthour)
    GPP_NEE_mean = np.zeros(endhour-starthour)
    Temp17_mean = np.zeros(endhour-starthour)
    Press_mean = np.zeros(endhour-starthour)
    ustar_mean = np.zeros(endhour-starthour)
    H_flux_mean = np.zeros(endhour-starthour)
    LE_flux_mean = np.zeros(endhour-starthour)
    TsoilA_mean = np.zeros(endhour-starthour)
    soilCOS_flux_mean = np.zeros(endhour-starthour)
    soilCO2_flux_mean = np.zeros(endhour-starthour)
    PAR_mean = np.zeros(endhour-starthour)
    solarang_mean = np.zeros(endhour-starthour)
    for i in range(0,endhour-starthour):
        COS_125_mean[i] = np.mean(COS_125_selected[:,i])
        CO2_125_mean[i] = np.mean(CO2_125_selected[:,i])
        H2O_125_mean[i] = np.mean(H2O_125_selected[:,i])
        q_125_mean[i] = np.mean(q_125_selected[:,i])
        COS_23_mean[i] = np.mean(COS_23_selected[:,i])
        CO2_23_mean[i] = np.mean(CO2_23_selected[:,i])
        H2O_23_mean[i] = np.mean(H2O_23_selected[:,i])
        F_EC_COS_mean[i] = np.nanmean(F_EC_COS_selected[:,i])
        F_EC_CO2_mean[i] = np.nanmean(F_EC_CO2_selected[:,i])
        GPP_NEE_mean[i] = np.nanmean(GPP_NEE_selected[:,i])
        Temp17_mean[i] = np.nanmean(Temp17_selected[:,i]) 
        Press_mean[i] = np.nanmean(Press_selected[:,i])
        ustar_mean[i] = np.nanmean(ustar_selected[:,i])
        H_flux_mean[i] = np.nanmean(H_flux_selected[:,i])
        LE_flux_mean[i] = np.nanmean(LE_flux_selected[:,i])
        TsoilA_mean[i] = np.nanmean(TsoilA_selected[:,i])
        soilCOS_flux_mean[i] = np.nanmean(soilCOS_flux_selected[:,i])
        soilCO2_flux_mean[i] = np.nanmean(soilCO2_flux_selected[:,i])
        PAR_mean[i] = np.nanmean(PAR_selected[:,i])
        solarang_mean[i] = np.nanmean(solarang_selected[:,i])
        
#stdev measurements
if use_mean:
    stdevCOS_125_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCOS_125_hourly)):
        stdevCOS_125_hourly[i] = np.std(COS_125_selected[:,i])
    stdevCO2_125_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCO2_125_hourly)):
        stdevCO2_125_hourly[i] = np.std(CO2_125_selected[:,i])
    stdevH2O_125_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevH2O_125_hourly)):
        stdevH2O_125_hourly[i] = np.std(H2O_125_selected[:,i])
    stdevCOS_23_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCOS_23_hourly)):
        stdevCOS_23_hourly[i] = np.std(COS_23_selected[:,i])
    stdevCO2_23_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCO2_23_hourly)):
        stdevCO2_23_hourly[i] = np.std(CO2_23_selected[:,i])
    stdevH2O_23_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevH2O_23_hourly)):
        stdevH2O_23_hourly[i] = np.std(H2O_23_selected[:,i])
    stdevCOSflux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCOSflux_hourly)):
        stdevCOSflux_hourly[i] = np.nanstd(F_EC_COS_selected[:,i]) #! avoid nan!!
    stdevCO2flux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevCO2flux_hourly)):
        stdevCO2flux_hourly[i] = np.nanstd(F_EC_CO2_selected[:,i]) #! avoid nan!!
    stdevGPP_NEE_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevGPP_NEE_hourly)):
        stdevGPP_NEE_hourly[i] = np.std(GPP_NEE_selected[:,i])
    stdevTemp17_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevTemp17_hourly)):
        stdevTemp17_hourly[i] = np.std(Temp17_selected[:,i])
    stdevPress_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevPress_hourly)):
        stdevPress_hourly[i] = np.std(Press_selected[:,i])
    stdevustar_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevustar_hourly)):
        stdevustar_hourly[i] = np.std(ustar_selected[:,i]) 
    stdevH_flux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevH_flux_hourly)):
        stdevH_flux_hourly[i] = np.nanstd(H_flux_selected[:,i]) #! avoid nan!!
    stdevLE_flux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevLE_flux_hourly)):
        stdevLE_flux_hourly[i] = np.nanstd(LE_flux_selected[:,i]) #! avoid nan!!
    stdevsoilCOS_flux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevsoilCOS_flux_hourly)):
        stdevsoilCOS_flux_hourly[i] = np.nanstd(soilCOS_flux_selected[:,i]) #! avoid nan!!
    stdevsoilCO2_flux_hourly = np.zeros(endhour-starthour)
    for i in range(len(stdevsoilCO2_flux_hourly)):
        stdevsoilCO2_flux_hourly[i] = np.nanstd(soilCO2_flux_selected[:,i]) #! avoid nan!!

data2 = pd.read_csv(directory+'/'+'smeardata_20150801120000.csv', skiprows=3)
minute2 = data2['Minute'] #
hour2 = data2['Hour'] -2 #utc #this dataset is minutely, not averaged over the hour!
day2 = data2['Day'] ##Stay away from hours around midnight, will give errors!!
month2 = data2['Month'] #
year2 = data2['Year'] #
hour2_selected = np.zeros(endhour-starthour+1)

stdevTemp67_minutely = np.zeros((endhour-starthour+1,len(selectedminutes))) #stdev measurements
if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day2 == selectedday,np.logical_and(month2.isin(selectedmonths),year2.isin(selectedyears))),np.logical_and(np.logical_and(hour2>=starthour,hour2<=endhour),minute2.isin(selectedminutes)))
    Temp67_selected = data2[timeselection]['HYY_META.T672'] +273.15
    hour2_selected  = np.unique(hour2[timeselection])
else:
    timeselection = np.logical_and(np.logical_and(day2.isin(selecteddays),np.logical_and(month2.isin(selectedmonths),year2.isin(selectedyears))),np.logical_and(np.logical_and(hour2>=starthour,hour2<=endhour),minute2.isin(selectedminutes)))
    Temp67_selected = data2[timeselection]['HYY_META.T672'] +273.15
    Temp67_mean = np.zeros((endhour-starthour+1,len(selectedminutes)))
    hour2_selected  = np.unique(hour2[timeselection])
    
    for i in range(endhour-starthour+1):
        for j in range(len(selectedminutes)):
            timeselection2 = np.logical_and(np.logical_and(hour2>=starthour+i,hour2<starthour+i+1),minute2 == selectedminutes[j])
            Temp67toaverage = Temp67_selected[timeselection2]
            Temp67_mean[i][j] = np.mean(Temp67toaverage)
            stdevTemp67_minutely[i][j] = np.std(Temp67toaverage)
            
data3 = pd.read_csv(directory+'/'+'Hyytiala_rad.csv',skiprows=4)
hour3 = data3['Hour']-2+0.5 # 0.5 since it is the average over the coming hour, 2 to correct to UTC
day3 = data3['Day'] #
month3 = data3['Month'] #
year3 = data3['Year'] #

if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day3 == selectedday,np.logical_and(month3.isin(selectedmonths),year3.isin(selectedyears))),np.logical_and(hour3>=starthour,hour3<=endhour))
    netrad67_selected = data3[timeselection]['netrad_67']
    PARdata3_selected = data3[timeselection]['PAR'] /1e6 * Avogadro * Planck * light_vel / wavel_PAR #from micromol/m2/s to W/m2
    PARdiff_selected = data3[timeselection]['HYY_META.diffPAR'] /1e6 * Avogadro * Planck * light_vel / wavel_PAR #from micromol/m2/s to W/m2
    LWin_selected = data3[timeselection]['HYY_META.LWin']
    LWout_selected = data3[timeselection]['HYY_META.LWout']
    SWin_selected = data3[timeselection]['HYY_META.Globmast']
    SWout_selected = data3[timeselection]['HYY_META.RGlob125']
    hour3_selected = np.unique(hour3[timeselection]) #unique to remove duplicates 
else:
    timeselection = np.logical_and(np.logical_and(day3.isin(selecteddays),np.logical_and(month3.isin(selectedmonths),year3.isin(selectedyears))),np.logical_and(hour3>=starthour,hour3<=endhour))
    hour3_selected = np.unique(hour3[timeselection]) #unique to remove duplicates 
    netrad67_selected = data3[timeselection]['netrad_67'] 
    netrad67_mean = np.zeros(len(hour3_selected))    
    PARdata3_selected = data3[timeselection]['PAR'] /1e6 * Avogadro * Planck * light_vel / wavel_PAR #from micromol/m2/s to W/m2
    PARdiff_selected = data3[timeselection]['HYY_META.diffPAR'] /1e6 * Avogadro * Planck * light_vel / wavel_PAR #from micromol/m2/s to W/m2
    PARdata3_mean = np.zeros(len(hour3_selected))
    PARdiff_mean = np.zeros(len(hour3_selected))
    LWin_selected = data3[timeselection]['HYY_META.LWin']
    LWin_mean = np.zeros(len(hour3_selected))
    LWout_selected = data3[timeselection]['HYY_META.LWout']
    LWout_mean = np.zeros(len(hour3_selected))
    SWin_selected = data3[timeselection]['HYY_META.Globmast']
    SWin_mean = np.zeros(len(hour3_selected))
    SWout_selected = data3[timeselection]['HYY_META.RGlob125']
    SWout_mean = np.zeros(len(hour3_selected))
    for i in range(len(hour3_selected)):
        timeselection3 = np.logical_and(hour3>=starthour+i,hour3<starthour+i+1)
        netrad67toaverage = netrad67_selected[timeselection3]
        PARdata3toaverage = PARdata3_selected[timeselection3]
        PARdifftoaverage = PARdiff_selected[timeselection3]
        LWintoaverage = LWin_selected[timeselection3]
        LWouttoaverage = LWout_selected[timeselection3]
        SWintoaverage = SWin_selected[timeselection3]
        SWouttoaverage = SWout_selected[timeselection3]
        netrad67_mean[i] = np.mean(netrad67toaverage)
        PARdata3_mean[i] = np.mean(PARdata3toaverage)
        PARdiff_mean[i] = np.mean(PARdifftoaverage)
        LWin_mean[i] = np.mean(LWintoaverage)
        LWout_mean[i] = np.mean(LWouttoaverage)
        SWin_mean[i] = np.mean(SWintoaverage)
        SWout_mean[i] = np.mean(SWouttoaverage)
        
data4 = pd.read_csv(directory+'/'+'Hyytiala_wind.csv', skiprows=1)
hour4 = data4['Hour'] -2+0.5 #utc 
day4 = data4['Day'] ##Stay away from hours around midnight, will give errors!!
month4 = data4['Month'] #
year4 = data4['Year'] #

if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day4 == selectedday,np.logical_and(month4.isin(selectedmonths),year4.isin(selectedyears))),np.logical_and(hour4>=starthour,hour4<=endhour))
    Windsp8_4_selected = data4[timeselection]['HYY_META.WSU84']
    Windsp16_8_selected = data4[timeselection]['HYY_META.WSU168']
    Windsp33_6_selected = data4[timeselection]['HYY_META.WSU336']
    hour4_selected = np.unique(hour4[timeselection]) #unique to remove duplicates 
else:
    timeselection = np.logical_and(np.logical_and(day4.isin(selecteddays),np.logical_and(month4.isin(selectedmonths),year4.isin(selectedyears))),np.logical_and(hour4>=starthour,hour4<=endhour))
    hour4_selected = np.unique(hour4[timeselection]) #unique to remove duplicates 
    Windsp8_4_selected = data4[timeselection]['HYY_META.WSU84']
    Windsp16_8_selected = data4[timeselection]['HYY_META.WSU168']
    Windsp33_6_selected = data4[timeselection]['HYY_META.WSU336']
    Windsp8_4_mean = np.zeros(len(hour4_selected))   
    Windsp16_8_mean = np.zeros(len(hour4_selected)) 
    Windsp33_6_mean = np.zeros(len(hour4_selected)) 
    for i in range(len(hour4_selected)):
        timeselection4 = np.logical_and(hour4>=starthour+i,hour4<starthour+i+1)
        Windsp8_4toaverage = Windsp8_4_selected[timeselection4]
        Windsp8_4_mean[i] = np.mean(Windsp8_4toaverage)
        Windsp16_8toaverage = Windsp16_8_selected[timeselection4]
        Windsp16_8_mean[i] = np.mean(Windsp16_8toaverage)
        Windsp33_6toaverage = Windsp33_6_selected[timeselection4]
        Windsp33_6_mean[i] = np.mean(Windsp33_6toaverage)

#model settings
priorinput = fwdm.model_input()
priorinput.hc = 17
priorinput.COS        = 0.400 #ppb
priorinput.COSmeasuring_height = 125. - priorinput.hc
priorinput.COSmeasuring_height2 = 23 - priorinput.hc
priorinput.CO2measuring_height = 125. - priorinput.hc
priorinput.CO2measuring_height2 = 23 - priorinput.hc
priorinput.Tmeasuring_height = 67.2 - priorinput.hc #0 would be a problem
priorinput.Tmeasuring_height2 = 18 - priorinput.hc #0 would be a problem
priorinput.qmeasuring_height = 125. - priorinput.hc
priorinput.qmeasuring_height2 = 23. - priorinput.hc
priorinput.sw_useWilson  = False
priorinput.dt         = 10.0       # time step [s]
priorinput.tstart     = starthour - 0.5   # time of the day [h UTC]
priorinput.runtime    = (endhour_mod-priorinput.tstart)*3600 + priorinput.dt   # total run time [s]
priorinput.sw_ml      = True      # mixed-layer model switch
priorinput.sw_shearwe = False     # shear growth mixed-layer switch
priorinput.sw_fixft   = False     # Fix the free-troposphere switch
priorinput.h          = 400.      # initial ABL height [m]
if use_mean:
    priorinput.Ps         = Press_mean[0]*100   # surface pressure [Pa]
else:
    priorinput.Ps         = Press_selected[0]*100   # surface pressure [Pa]
priorinput.divU       = 0.00        # horizontal large-scale divergence of wind [s-1]
priorinput.theta      = 288.5      # initial mixed-layer potential temperature [K]
priorinput.deltatheta = 1.00       # initial temperature jump at h [K]
priorinput.gammatheta = 0.008     # free atmosphere potential temperature lapse rate [K m-1]
priorinput.advtheta   = 0.        # advection of heat [K s-1]
priorinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
priorinput.wtheta     = 20/1.2/1005       # initial surface kinematic heat flux [K m s-1]
priorinput.q          = 0.007     # initial mixed-layer specific humidity [kg kg-1]
priorinput.deltaq     = -0.0015    # initial specific humidity jump at h [kg kg-1]
priorinput.gammaq     = -0.000005        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priorinput.advq       = 0.        # advection of moisture [kg kg-1 s-1]
priorinput.wq         = 20/1.2/2.5e6    # initial surface kinematic moisture flux [kg kg-1 m s-1] 
priorinput.CO2        = 400.      # initial mixed-layer CO2 [ppm]
priorinput.deltaCO2   = -10.      # initial CO2 jump at h [ppm]
priorinput.deltaCOS   = 0.050      # initial COS jump at h [ppb]
priorinput.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
priorinput.gammaCOS   = 0.00        # free atmosphere COS lapse rate [ppb m-1]
priorinput.advCO2     = 0.        # advection of CO2 [ppm s-1]
priorinput.advCOS     = 0.        # advection of COS [ppb s-1]
priorinput.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]
priorinput.wCOS       = 0.0       # surface kinematic COS flux [ppb m s-1]
priorinput.sw_wind    = True     # prognostic wind switch
priorinput.u          = 4.5        # initial mixed-layer u-wind speed [m s-1]
priorinput.deltau     = 4.        # initial u-wind jump at h [m s-1]
priorinput.gammau     = 0.005        # free atmosphere u-wind speed lapse rate [s-1]
priorinput.advu       = 0.        # advection of u-wind [m s-2]
priorinput.v          = -4.5      # initial mixed-layer u-wind speed [m s-1]
priorinput.deltav     = 4.0       # initial u-wind jump at h [m s-1]
priorinput.gammav     = -0.005        # free atmosphere v-wind speed lapse rate [s-1]
priorinput.advv       = 0.        # advection of v-wind [m s-2]
priorinput.sw_sl      = True     # surface layer switch
priorinput.ustar      = 1.0       # surface friction velocity [m s-1]
priorinput.z0m        = 1.0      # roughness length for momentum [m]
priorinput.z0h        = 1.0     # roughness length for scalars [m]
priorinput.sw_rad     = True     # radiation switch
priorinput.lat        = 61.85     # latitude [deg]
priorinput.lon        = 24.28     # longitude [deg]
priorinput.fc         = 2 * 7.292e-5 * np.sin(priorinput.lat * 2 * np.pi / 360.)     # Coriolis parameter [m s-1]
if use_mean:
    priorinput.doy        = 31 + 28 +31 +30 +31 +30 +31 + int(np.mean(selecteddays))      # day of the year [-]
else:
    priorinput.doy        = 31 + 28 +31 +30 +31 +30 +31 + selectedday
priorinput.cc         = 0.5       # cloud cover fraction [-]
priorinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priorinput.sw_ls      = True     # land surface switch
priorinput.ls_type    = 'canopy_model'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs or 'canopy_model')
priorinput.wg         = 0.25      # volumetric water content top soil layer [m3 m-3]
priorinput.w2         = 0.25      # volumetric water content deeper soil layer [m3 m-3]
#here
priorinput.Tsoil      = priorinput.theta      # temperature top soil layer [K]
if use_mean:
    priorinput.T2         = TsoilA_mean[0]      # temperature deeper soil layer [K]
else:
    priorinput.T2         = TsoilA_selected[0]      # temperature deeper soil layer [K]
priorinput.a          = 0.219     # Clapp and Hornberger retention curve parameter a
priorinput.b          = 4.90      # Clapp and Hornberger retention curve parameter b
priorinput.p          = 4.        # Clapp and Hornberger retention curve parameter c
priorinput.CGsat      = 3.56e-6   # saturated soil conductivity for heat
priorinput.wsat       = 0.61     # saturated volumetric water content (Sun 2017)
priorinput.wfc        = 0.4     # volumetric water content field capacity [-]
priorinput.wwilt      = 0.10     # volumetric water content wilting point [-]
priorinput.C1sat      = 0.132     
priorinput.C2ref      = 1.8
priorinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priorinput.alpha      = np.mean(SWout_mean/SWin_mean)      # surface albedo [-]
priorinput.Ts         = priorinput.theta      # initial surface temperature [K]
priorinput.Wmax       = 0.0002    # max thickness of water layer on wet vegetation [m]
priorinput.Lambda     = 5.9       # thermal diffusivity skin layer [-]
priorinput.c3c4       = 'c3'      # Plant type ('c3' or 'c4')
priorinput.sw_cu      = False     # Cumulus parameterization switch
priorinput.dz_h       = 150.      # Transition layer thickness [m]
priorinput.Cs         = 1e12      # initial drag coefficient for scalars [-]
priorinput.sw_dynamicsl_border = True
priorinput.sw_model_stable_con = True
priorinput.R10 = 0.115
if priorinput.ls_type != 'canopy_model':
    priorinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]

#for bulk a-gs only:
#priorinput.alfa_plant = 1
#priorinput.gciCOS = 0.2 /(1.2*1000) * 28.9
#priorinput.ags_C_mode = 'MXL' 
#priorinput.cveg       = 1.0     # vegetation fraction [-]
#priorinput.LAI        = 0.01        # leaf area index [-]

#soil COS model
priorinput.soilCOSmodeltype   = 'Sun_Ogee' #can be set to None or 'Sun_Ogee'
priorinput.uptakemodel = 'Ogee'
priorinput.sw_soilmoisture    = 'simple'
priorinput.sw_soiltemp    = 'simple'
priorinput.kH_type         = 'Sun'
priorinput.Diffus_type     = 'Sun'
priorinput.b_sCOSm = 5.3
priorinput.fCA = 1e4
priorinput.nr_nodes     = 26
priorinput.s_moist_opt  = 0.20
priorinput.Vspmax        = 1.e-10
priorinput.Q10             = 3.
priorinput.layer1_2division = 0.3
priorinput.write_soilCOS_to_f = False
priorinput.nr_nodes_for_filewr = 5

#canopy model
priorinput.nr_nodes_veg = 20
priorinput.z_veglay = np.linspace(priorinput.hc*0.01,priorinput.hc*0.99,priorinput.nr_nodes_veg)
priorinput.gliCOS = 0.2 /(1.2*1000) * 28.9/100.  #/100 just for the fit
priorinput.lad_veglay = np.zeros(priorinput.nr_nodes_veg)
a_max = 1.7 #Launianen 2007 fig 1
for lay in range(priorinput.nr_nodes_veg):
    if priorinput.z_veglay[lay]/priorinput.hc < 0.1:
        priorinput.lad_veglay[lay] = 0.8*a_max #Launianen 2011 fig 1
    elif priorinput.z_veglay[lay]/priorinput.hc >= 0.1 and priorinput.z_veglay[lay]/priorinput.hc <= 0.4:
        priorinput.lad_veglay[lay] = 0.01*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc > 0.4 and priorinput.z_veglay[lay]/priorinput.hc < 0.5:
        priorinput.lad_veglay[lay] = 0.1*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc >= 0.5 and priorinput.z_veglay[lay]/priorinput.hc <= 0.6:
        priorinput.lad_veglay[lay] = 0.3*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc > 0.6 and priorinput.z_veglay[lay]/priorinput.hc < 0.7:
        priorinput.lad_veglay[lay] = 0.4*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc >= 0.7 and priorinput.z_veglay[lay]/priorinput.hc <= 0.8:
        priorinput.lad_veglay[lay] = 0.25*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc > 0.8 and priorinput.z_veglay[lay]/priorinput.hc < 0.9:
        priorinput.lad_veglay[lay] = 0.1*a_max
    elif priorinput.z_veglay[lay]/priorinput.hc >= 0.9 and priorinput.z_veglay[lay]/priorinput.hc <= 1.0:
        priorinput.lad_veglay[lay] = 0.05*a_max#0.05
#priorinput.lad_veglay = np.array([0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7])
priorinput.rws_COS_leaf = 1.e5 #wet skin resistance (resistance for water layer)
priorinput.rws_CO2_leaf = 1.e5
priorinput.C_CO2_veglayer_init_ppm = np.linspace(400,400,priorinput.nr_nodes_veg)#[400,400,400,400,400,400,400,400,400,400,400,400,400,400,400]
priorinput.C_COS_veglayer_init_ppb = np.linspace(0.4,0.4,priorinput.nr_nodes_veg)#[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
H2O_pct = priorinput.q / 18 * 28.97 * 100 #kg/kg to molar pct: g_h2O/g_air * mol_h2o/g_h2o * g_air/mol_air = mol_h2o_mol_air, times 100 for %
priorinput.C_H2O_veglayer_init_pct = np.linspace(H2O_pct,H2O_pct,priorinput.nr_nodes_veg)
priorinput.lad_dependence_u = True
#priorinput.ra_veg = 5.
priorinput.K_mode = 'Launianen' #'int_resistance' or 'Launianen'
priorinput.U_mode = 'Cionco'
priorinput.alfa_sto = 1.0
priorinput.dt_can = 2.0
priorinput.calc_sun_shad = True
priorinput.Kdif = 0.7
priorinput.prescr_fPARdif = False
#priorinput.fPARdif = 0.3
tsteps = int(np.floor(priorinput.runtime / priorinput.dt))
fPARdif = np.zeros(tsteps)
xp = (hour3_selected - priorinput.tstart) / (priorinput.dt / 3600)
priorinput.fPARdif = np.interp(range(tsteps),xp,PARdiff_mean/PARdata3_mean)
priorinput.frac_green = 0.9
priorinput.Ts_mode = 'CLASS_en_bal'
priorinput.Wl         = np.linspace(0.0000,0.0000,priorinput.nr_nodes_veg)    # equivalent water layer depth for wet vegetation [m]
priorinput.K_scale = 1.0
priorinput.U_ext_scale = 0.5

#for a-gs in the canopy model
#priorinput.T1gm = [288.,    286.   ]
priorinput.T2gm = [288.,    309.   ]
#leon Casse settings
#priorinput.Ammax298 = [1.0]
#priorinput.gm298 = [1.0]
#priorinput.f0 = [0.80]

priormodel = fwdm.model(priorinput)
priormodel.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model


plt.figure()
plt.plot(priormodel.out.t,priormodel.out.COS)
plt.ylabel('COS_conc ML (ppb)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.CO2)
plt.ylabel('CO2_conc ML (ppm)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.theta)
plt.ylabel('theta (K)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.LE)
plt.ylabel('LE (W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.LEveg)
plt.ylabel('LE veg(W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.enbalerr)
plt.ylabel('energy bal error(W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.Swin*0.5,label = 'model')
if use_mean:
    plt.plot(hour_selected,PAR_mean,label = 'obs')
    plt.plot(hour3_selected,PARdata3_mean,label = 'obs data3')
    plt.plot(hour3_selected,PARdiff_mean,label = 'obs diff')
else:
    plt.plot(hour_selected,PAR_selected,label = 'obs')
    plt.plot(hour3_selected,PARdata3_selected,label = 'obs data3')
    plt.plot(hour3_selected,PARdiff_selected,label = 'obs diff')
plt.legend()
plt.ylabel('PAR (W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.fPARdif,label = 'model')
if use_mean:
    plt.plot(hour3_selected,PARdiff_mean/PARdata3_mean,label = 'obs',linestyle="",marker="o")
else:
    plt.plot(hour3_selected,PARdiff_selected/PARdata3_selected,label = 'obs',linestyle="",marker="o")
plt.legend()
plt.ylabel('PAR diff fract')
plt.xlabel('time')

plt.figure()
ind8_4 = np.where(np.abs(priormodel.canopy_model.z_veglay - 8.4) == np.min(np.abs(priormodel.canopy_model.z_veglay - 8.4)))[0][0]#index closest to 8.4 obs height
ind16_8 = np.where(np.abs(priormodel.canopy_model.z_veglay - 16.8) == np.min(np.abs(priormodel.canopy_model.z_veglay - 16.8)))[0][0]#index closest to 16.8 obs height
plt.plot(priormodel.out.t,priormodel.out.U_veg[:,ind8_4],label = 'model '+str(round(priormodel.canopy_model.z_veglay[ind8_4],2)))
plt.plot(priormodel.out.t,priormodel.out.U_veg[:,ind16_8],label = 'model '+str(round(priormodel.canopy_model.z_veglay[ind16_8],2)))
if use_mean:
    plt.plot(hour4_selected,Windsp8_4_mean,label = 'obs 8.4')
    plt.plot(hour4_selected,Windsp16_8_mean,label = 'obs 16.8')
else:
    plt.plot(hour4_selected,Windsp8_4_selected,label = 'obs 8.4')
    plt.plot(hour4_selected,Windsp16_8_selected,label = 'obs 16.8')
plt.legend()
plt.ylabel('Wind sp canopy (ms-1)')
plt.xlabel('time')

#plt.figure()
#plt.plot(priormodel.out.t,priorinput.fPARdif,label = 'model')
#if use_mean:
#    plt.plot(hour3_selected,PARdiff_mean/PARdata3_mean,label = 'obs')
#else:
#    plt.plot(hour3_selected,PARdiff_selected/PARdata3_selected,label = 'obs')
#plt.legend()
#plt.ylabel('PAR diff fract input')
#plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.Q,label = 'model')
if use_mean:
    plt.plot(hour3_selected,netrad67_mean,label = 'obs')
else:
    plt.plot(hour3_selected,netrad67_selected,label = 'obs')
plt.legend()
plt.ylabel('Q (W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.Lwout,label = 'model out')
plt.plot(priormodel.out.t,priormodel.out.Lwin,label = 'model in')
if use_mean:
    plt.plot(hour3_selected,LWout_mean,label = 'obs out')
    plt.plot(hour3_selected,LWin_mean,label = 'obs in')
else:
    plt.plot(hour3_selected,LWout_selected,label = 'obs out')
    plt.plot(hour3_selected,LWin_selected,label = 'obs in')
plt.legend()
plt.ylabel('LW rad (W/m2)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.Swout,label = 'model out')
plt.plot(priormodel.out.t,priormodel.out.Swin,label = 'model in')
if use_mean:
    plt.plot(hour3_selected,SWout_mean,label = 'obs out')
    plt.plot(hour3_selected,SWin_mean,label = 'obs in')
else:
    plt.plot(hour3_selected,SWout_selected,label = 'obs out')
    plt.plot(hour3_selected,SWin_selected,label = 'obs in')
plt.legend()
plt.ylabel('SW rad (W/m2)')
plt.xlabel('time')

plt.figure()
model_solarang = np.arcsin(priormodel.out.sinlea) * 360 / (2*np.pi) 
plt.plot(priormodel.out.t,model_solarang,label = 'model')
if use_mean:
    plt.plot(hour_selected,solarang_mean,label = 'obs')
else:
    plt.plot(hour_selected,solarang_selected,label = 'obs')
plt.legend()
plt.ylabel('solar angle (deg)')
plt.xlabel('time')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.h)
plt.ylabel('h (m)')
plt.xlabel('time')

fig = plt.figure()
plt.rc('font', size=16)
plt.plot(priormodel.out.t,priormodel.out.wCOS, linestyle='--', marker='o',color='yellow')
if use_mean:
    plt.plot(hour_selected,F_EC_COS_mean, linestyle=' ', marker='o',color='black')
else:
    plt.plot(hour_selected,F_EC_COS_selected, linestyle=' ', marker='o',color='black')
plt.ylabel('COS surface flux (ppb m s-1)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCOS.png', format='png')
    
fig = plt.figure()
plt.rc('font', size=16)
plt.plot(priormodel.out.t,priormodel.out.wCOSP, linestyle='--', marker='o',color='yellow')
if use_mean:
    plt.plot(hour_selected,F_EC_COS_mean, linestyle=' ', marker='o',color='black')
else:
    plt.plot(hour_selected,F_EC_COS_selected, linestyle=' ', marker='o',color='black')
plt.ylabel('COS plant flux (ppb m s-1)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCOSP.png', format='png')

fig = plt.figure()
plt.rc('font', size=16)
plt.plot(priormodel.out.t,priormodel.out.wCO2A, linestyle='--', marker='o',color='yellow')
if use_mean:
    plt.plot(hour_selected,GPP_NEE_mean, linestyle=' ', marker='o',color='black')
else:
    plt.plot(hour_selected,GPP_NEE_selected, linestyle=' ', marker='o',color='black')
plt.ylabel('GPP (mg CO2/m2/s)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCO2A.png', format='png')
    
fig = plt.figure()
plt.rc('font', size=16)
plt.plot(priormodel.out.t[25:],priormodel.out.wCO2[25:], linestyle='--', marker='o',color='yellow')
if use_mean:
    plt.plot(hour_selected,F_EC_CO2_mean, linestyle=' ', marker='o',color='black')
else:
    plt.plot(hour_selected,F_EC_CO2_selected, linestyle=' ', marker='o',color='black')
plt.ylabel('flux_surlay_canopy (ppm m/s)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCO2.png', format='png')
    
fig = plt.figure()
plt.rc('font', size=16)
plt.plot(priormodel.out.t,priormodel.out.wCO2R, linestyle='--', marker='o',color='yellow')
if use_mean:
    plt.plot(hour_selected,soilCO2_flux_mean, linestyle=' ', marker='o',color='black')
else:
    plt.plot(hour_selected,soilCO2_flux_selected, linestyle=' ', marker='o',color='black')
plt.ylabel('Resp (mg CO2/m2/s)')
plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_wCO2R.png', format='png')

plt.rc('font', size=16)
plt.figure()
plt.plot(priormodel.out.C_CO2_veglayer_ppm[-1,:],priorinput.z_veglay)
plt.ylabel('height')
plt.xlabel('CO2_conc at t = ' + str(priormodel.t) +' (ppm)')

plt.figure()
plt.plot(priormodel.out.C_COS_veglayer_ppb[-1,:],priorinput.z_veglay)
plt.ylabel('height')
plt.xlabel('COS_conc at t = ' + str(priormodel.t) +' (ppb)')

plt.figure()
plt.plot(priormodel.out.C_H2O_veglayer_pct[-1,:],priorinput.z_veglay)
plt.ylabel('height')
plt.xlabel('H2O_conc at t = ' + str(priormodel.t) +' (pct)')

plt.figure()
plt.plot(priormodel.canopy_model.veg_exchange_COS/priormodel.canopy_model.dz_veglay,priorinput.z_veglay)
plt.ylabel('height')
plt.xlabel('COS_upt at t = ' + str(priormodel.t) +' (mol m-3 s-1)')

plt.figure()
plt.plot(priormodel.canopy_model.veg_exchange_CO2/priormodel.canopy_model.dz_veglay,priorinput.z_veglay)
plt.ylabel('height')
plt.xlabel('CO2_upt at t = ' + str(priormodel.t) +' (mol m-3 s-1)')
 
plt.figure()
plt.plot(priormodel.out.t,priormodel.out.C_CO2_veglayer_ppm[:,0] )
plt.xlabel('time')
plt.ylabel('CO2_conc lowest veg layer (ppm)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.C_CO2_veglayer_ppm[:,4] )
plt.xlabel('time')
plt.ylabel('CO2_conc fifth veg layer (ppm)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.C_CO2_veglayer_ppm[:,-1] )
plt.xlabel('time')
plt.ylabel('CO2_conc highest veg layer (ppm)')

if not priorinput.calc_sun_shad:
    plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.gsco2_leaf[:,-1] )
    plt.xlabel('time')
    plt.ylabel('stom cond leaf highest veg layer (m s-1)')
    #
    plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.gsco2_leaf[:,0])
    plt.xlabel('time')
    plt.ylabel('stom cond leaf lowest veg layer (m s-1)')
    
    plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.gsco2_leaf[:,9])
    plt.xlabel('time')
    plt.ylabel('stom cond leaf tenth veg layer (m s-1)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.ci_co2[:,-1]/priormodel.rho/priormodel.mco2*priormodel.mair * 1e6 * 0.001*0.001  )
plt.xlabel('time')
plt.ylabel('ci highest veg layer (ppm)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.ci_co2[:,-1]/priormodel.rho/priormodel.mco2*priormodel.mair * 1e6 * 0.001*0.001 / priormodel.out.C_CO2_veglayer_ppm[:,-1])
plt.xlabel('time')
plt.ylabel('ci/ca highest veg layer (-)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.ci_co2[:,0]/priormodel.rho/priormodel.mco2*priormodel.mair * 1e6 * 0.001*0.001 )
plt.xlabel('time')
plt.ylabel('ci lowest veg layer (ppm)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.C_CO2_veglayer_ppm[:,-1] - priormodel.out.ci_co2[:,-1]/priormodel.rho/priormodel.mco2*priormodel.mair * 1e6 * 0.001*0.001 ) #ppm = mumol_co2 / mol_air = mg_co2 / m3 * m3 / kg_air * mol_co2 / g_co2 * g_air / mol_air * mumol_co2 / mol_co2 * kg_air/g_air * g_co2 / mg_co2
plt.xlabel('time')
plt.ylabel('co2 - ci highest veg layer (ppm)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.CO2plantflux[:,-1] )
plt.xlabel('time')
plt.ylabel('GPP highest veg layer (mol m-3 s-1)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.COSplantflux[:,-1] )
plt.xlabel('time')
plt.ylabel('COS upt highest veg layer (mol m-3 s-1)')

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.rbveg_CO2[:,-1] )
plt.xlabel('time')
plt.ylabel('leaf BL resist highest veg layer (s m-1)')

Dsleaf = np.zeros(len(priormodel.out.t))
for i in range(len(priormodel.out.t)):
    Dsleaf[i] = fwdm.esat(priormodel.out.Ts[i]) - priormodel.out.e[i]
plt.figure()
plt.plot(priormodel.out.t,Dsleaf)
plt.xlabel('time')
plt.ylabel('2m Ds (Pa)')

plt.figure()
if use_mean:
    plt.plot(hour_selected,Temp17_mean,label='obs')
plt.plot(priormodel.out.t,priormodel.out.Tmh2,label='model')
plt.xlabel('time')
plt.ylabel('Temp 17 m (K)')
plt.legend()

plt.figure()
if use_mean:
    plt.plot(hour2_selected,Temp67_mean,label='obs')
plt.plot(priormodel.out.t,priormodel.out.Tmh,label='model')
plt.xlabel('time')
plt.ylabel('Temp 67 m (K)')
plt.legend()

plt.figure()
if use_mean:
    plt.plot(hour_selected,q_125_mean,label='obs')
plt.plot(priormodel.out.t,priormodel.out.qmh,label='model')
plt.xlabel('time')
plt.ylabel('q 125 m (kg/kg)')
plt.legend()

plt.figure()
if use_mean:
    plt.plot(hour_selected,LE_flux_mean,label='obs')
plt.plot(priormodel.out.t,priormodel.out.LE,label='model')
plt.plot(priormodel.out.t,priormodel.out.LEliq+priormodel.out.LEsoil+priormodel.out.LEveg,label='model instant')
plt.xlabel('time')
plt.ylabel('LE (W/m2)')
plt.legend()

plt.figure()
if use_mean:
    plt.plot(hour_selected,H_flux_mean,label='obs')
plt.plot(priormodel.out.t,priormodel.out.H,label='model')
plt.xlabel('time')
plt.ylabel('H (W/m2)')
plt.legend()

plt.figure()
plt.plot(priormodel.out.t,priormodel.out.G,label='model')
plt.xlabel('time')
plt.ylabel('G (W/m2)')
plt.legend()

if priorinput.calc_sun_shad == True:
    plt.figure()
    plt.plot(priormodel.canopy_model.gsco2_leaf_sun,priorinput.z_veglay,label='sun')
    plt.plot(priormodel.canopy_model.gsco2_leaf_sha,priorinput.z_veglay,label='sha')
    plt.ylabel('height')
    plt.xlabel('stom cond leaf at t = ' + str(priormodel.t) +' (m s-1)')
    plt.legend()
    
    plt.figure()
    plt.plot(priormodel.canopy_model.PAR_sun,priorinput.z_veglay,label='sun')
    plt.plot(priormodel.canopy_model.PAR_sha,priorinput.z_veglay,label='sha')
    plt.ylabel('height')
    plt.xlabel('PAR at t = ' + str(priormodel.t) +' (W m-2)')
    plt.legend()
    
    plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.PAR_sun_abs[:,-1],label='sun')
    plt.plot(priormodel.out.t,priormodel.out.PAR_sha_abs[:,-1],label='sha')
    plt.xlabel('time')
    plt.ylabel('PAR absorbed highest veg layer' +' (W m-2)')
    plt.legend()
    
#    plt.figure()
#    plt.plot(priormodel.out.t,priormodel.out.aapsun[:,-1],label='sun')
#    plt.plot(priormodel.out.t,priormodel.out.aapsha[:,-1],label='sha')
#    plt.xlabel('time')
#    plt.ylabel('pexp')
#    plt.legend()
    
    plt.figure()
    plt.plot(priormodel.out.t,priormodel.out.gsco2_leaf_sun[:,-1],label='sun')
    plt.plot(priormodel.out.t,priormodel.out.gsco2_leaf_sha[:,-1],label='sha')
    plt.xlabel('time')
    plt.ylabel('stom cond leaf highest veg layer (m s-1)')
    plt.legend()