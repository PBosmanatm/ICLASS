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
import pandas as pd
import math
import glob
import matplotlib.style as style
style.use('classic')

##################################
###### user input: settings ######
##################################
ana_deriv = True
use_backgr_in_cost = True
write_to_f = True
use_ensemble = False
if use_ensemble:
    nr_of_members = 28
    run_multicore = True #only possible when using ensemble
    est_post_pdf = True
    plot_perturbed_obs = False #only for posterior pdf, obs are only perturbed there
maxnr_of_restarts = 3 #only implemented for tnc method at the moment
imposeparambounds = True
remove_all_prev = True #Use with caution, be careful for other files in working directory!!
optim_method = 'tnc' #bfgs or tnc
stopcrit = 1.0
use_mean = False #switch for using mean of obs over several days
use_weights = True #weights for the cost function, to enlarge the importance of certain obs (one weight per type of obs)
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
canopy_height = 0 #m
#constants
Eps = 18.005/28.964 #p3.3 intro atm
g = 9.81
    
selectedyears = [2003.]
selectedmonths = [9]
selectedday = 25 #only used when not use_mean = True
selecteddays = [24,25,26]#[24,25,26] #period over which we average
selectedminutes = [5,15,25,35,45,55]#use an array! Do not include the same element twice
starthour = 9 #utc, start of obs  (use an integer!)
endhour = 15 #(use an integer!)

directory = 'Cabauw_obs'
data_Temp = pd.read_csv(directory+'/'+'caboper_air_temperature_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
data_Temp.loc[data_Temp['TA200'] <= -9999 ,'TA200'] = np.nan #set -9999 to nan
data_Temp.loc[data_Temp['TA140'] <= -9999 ,'TA140'] = np.nan
data_Temp.loc[data_Temp['TA080'] <= -9999 ,'TA080'] = np.nan
data_Temp.loc[data_Temp['TA040'] <= -9999 ,'TA040'] = np.nan
data_Temp.loc[data_Temp['TA020'] <= -9999 ,'TA020'] = np.nan
data_Temp.loc[data_Temp['TA010'] <= -9999 ,'TA010'] = np.nan
data_Temp.loc[data_Temp['TA002'] <= -9999 ,'TA002'] = np.nan
data_Press = pd.read_csv(directory+'/'+'caboper_surface_pressure_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
data_DewP = pd.read_csv(directory+'/'+'caboper_dew_point_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
data_DewP.loc[data_DewP['TD200'] <= -9999 ,'TD200'] = np.nan
data_DewP.loc[data_DewP['TD140'] <= -9999 ,'TD140'] = np.nan
data_DewP.loc[data_DewP['TD080'] <= -9999 ,'TD080'] = np.nan
data_DewP.loc[data_DewP['TD040'] <= -9999 ,'TD040'] = np.nan
data_DewP.loc[data_DewP['TD020'] <= -9999 ,'TD020'] = np.nan
data_DewP.loc[data_DewP['TD010'] <= -9999 ,'TD010'] = np.nan
data_DewP.loc[data_DewP['TD002'] <= -9999 ,'TD002'] = np.nan
data_Flux = pd.read_csv(directory+'/'+'cabsurf_surface_flux_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
data_Flux.loc[data_Flux['HSON'] <= -9999 ,'HSON'] = np.nan
data_Flux.loc[data_Flux['LEED'] <= -9999 ,'LEED'] = np.nan
data_Flux.loc[data_Flux['FG0'] <= -9999 ,'FG0'] = np.nan
data_Flux.loc[data_Flux['G05'] <= -9999 ,'G05'] = np.nan
data_Flux.loc[data_Flux['G10'] <= -9999 ,'G10'] = np.nan
data_Flux.loc[data_Flux['FCED'] <= -9999 ,'FCED'] = np.nan
data_Flux.loc[data_Flux['USTED'] <= -9999 ,'USTED'] = np.nan
data_Rad = pd.read_csv(directory+'/'+'caboper_radiation_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
data_Rad.loc[data_Rad['SWU'] <= -9999 ,'SWU'] = np.nan
data_Rad.loc[data_Rad['SWD'] <= -9999 ,'SWD'] = np.nan
data_Rad.loc[data_Rad['LWU'] <= -9999 ,'LWU'] = np.nan
data_Rad.loc[data_Rad['LWD'] <= -9999 ,'LWD'] = np.nan
data_Windsp = pd.read_csv(directory+'/'+'caboper_wind_speed_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True) 
data_Windsp.loc[data_Windsp['F200'] <= -9999 ,'F200'] = np.nan
data_Windsp.loc[data_Windsp['SF200'] <= -9999 ,'SF200'] = np.nan #standard dev 200m, see Cabauw_TR.pdf
data_Windsp.loc[data_Windsp['F010'] <= -9999 ,'F010'] = np.nan
data_Windsp.loc[data_Windsp['SF010'] <= -9999 ,'SF010'] = np.nan
data_Winddir = pd.read_csv(directory+'/'+'caboper_wind_direction_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True) 
data_Winddir.loc[data_Winddir['D200'] <= -9999 ,'D200'] = np.nan
data_Winddir.loc[data_Winddir['SD200'] <= -9999 ,'SD200'] = np.nan #see Cabauw_TR.pdf
data_Winddir.loc[data_Winddir['D010'] <= -9999 ,'D010'] = np.nan
data_Winddir.loc[data_Winddir['SD010'] <= -9999 ,'SD010'] = np.nan
date = data_Temp['day'] #
btime = data_Temp['btime']#begintime
etime = data_Temp['etime']#endtime
btimehour = np.zeros(len(btime))
btimeminute = np.zeros(len(btime))
for i in range(len(btime)):
    if len(str(btime[i])) <= 2:
        btimehour[i] = 0
        btimeminute[i] = btime[i]
    else:
        btimehour[i] = float(str(btime[i])[0:-2])
        btimeminute[i] = float(str(btime[i])[-2:])
etimehour = np.zeros(len(etime))
etimeminute = np.zeros(len(etime))
for i in range(len(etime)):
    if len(str(etime[i])) <= 2:
        etimehour[i] = 0
        etimeminute[i] = etime[i]
    else:
        etimehour[i] = float(str(etime[i])[0:-2])
        etimeminute[i] = float(str(etime[i])[-2:])
hour = np.zeros(len(btime),dtype = int)
minute = np.zeros(len(btime),dtype = int)
for i in range(len(btime)):
    if etimeminute[i] == 0 and (btimeminute[i] == 50 and etimehour[i] == btimehour[i] + 1):
        minute[i] = 55
        hour[i] = btimehour[i]
    else:
        minute[i] = (btimeminute[i] + etimeminute[i]) / 2
        hour[i] = btimehour[i]
year = np.array([int(str(date[i])[0:4]) for i in range(len(date))])
month = np.array([int(str(date[i])[4:6]) for i in range(len(date))])
day = np.array([int(str(date[i])[6:]) for i in range(len(date))])
minute_True_False = np.zeros(len(btime),dtype=bool)
month_True_False = np.zeros(len(btime),dtype=bool)
year_True_False = np.zeros(len(btime),dtype=bool)
day_True_False = np.zeros(len(btime),dtype=bool)
for i in range(len(minute_True_False)):
    if minute[i] in selectedminutes:
        minute_True_False[i] = True
    if month[i] in selectedmonths:
        month_True_False[i] = True
    if year[i] in selectedyears:
        year_True_False[i] = True
    if day[i] in selecteddays:
        day_True_False[i] = True #True for all selected days only

if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day == selectedday,np.logical_and(month_True_False,year_True_False)),np.logical_and(np.logical_and(hour>=starthour,hour<endhour),minute_True_False))
    Temp200_selected = data_Temp[timeselection]['TA200'] +273.15
    Temp140_selected = data_Temp[timeselection]['TA140'] +273.15
    Temp80_selected = data_Temp[timeselection]['TA080'] +273.15
    Temp40_selected = data_Temp[timeselection]['TA040'] +273.15
    Temp20_selected = data_Temp[timeselection]['TA020'] +273.15
    Temp10_selected = data_Temp[timeselection]['TA010'] +273.15
    Temp2_selected = data_Temp[timeselection]['TA002'] +273.15
    TD200_selected = data_DewP[timeselection]['TD200'] +273.15
    TD140_selected = data_DewP[timeselection]['TD140'] +273.15
    TD80_selected = data_DewP[timeselection]['TD080'] +273.15
    TD40_selected = data_DewP[timeselection]['TD040'] +273.15
    TD20_selected = data_DewP[timeselection]['TD020'] +273.15
    TD10_selected = data_DewP[timeselection]['TD010'] +273.15
    TD2_selected = data_DewP[timeselection]['TD002'] +273.15
    Press_selected = data_Press[timeselection]['AP0']*100. #Pa
    e200_selected = 610.7 * np.exp(17.2694*(TD200_selected - 273.16) / (TD200_selected - 35.86)) #eq 3.3 intro atm
    e140_selected = 610.7 * np.exp(17.2694*(TD140_selected - 273.16) / (TD140_selected - 35.86)) #eq 3.3 intro atm
    e80_selected = 610.7 * np.exp(17.2694*(TD80_selected - 273.16) / (TD80_selected - 35.86)) #eq 3.3 intro atm
    e40_selected = 610.7 * np.exp(17.2694*(TD40_selected - 273.16) / (TD40_selected - 35.86)) #eq 3.3 intro atm
    e20_selected = 610.7 * np.exp(17.2694*(TD20_selected - 273.16) / (TD20_selected - 35.86)) #eq 3.3 intro atm
    e10_selected = 610.7 * np.exp(17.2694*(TD10_selected - 273.16) / (TD10_selected - 35.86)) #eq 3.3 intro atm
    e2_selected = 610.7 * np.exp(17.2694*(TD2_selected - 273.16) / (TD2_selected - 35.86)) #eq 3.3 intro atm
    rho80 = (Press_selected - 1.22030 * g * 80) / (287.04 * Temp80_selected) #1.22030 from us standard atmopsphere 40m
    q200_selected = Eps * e200_selected / (Press_selected - rho80 * g * 200 - (1 - Eps) * e200_selected) #eq 3.4 intro atm
    q140_selected = Eps * e140_selected / (Press_selected - rho80 * g * 140 - (1 - Eps) * e140_selected) #eq 3.4 intro atm
    q80_selected = Eps * e80_selected / (Press_selected - rho80 * g * 80 - (1 - Eps) * e80_selected) #eq 3.4 intro atm
    q40_selected = Eps * e40_selected / (Press_selected - rho80 * g * 40 - (1 - Eps) * e40_selected) #eq 3.4 intro atm
    q20_selected = Eps * e20_selected / (Press_selected - rho80 * g * 20 - (1 - Eps) * e20_selected) #eq 3.4 intro atm
    q10_selected = Eps * e10_selected / (Press_selected - rho80 * g * 10 - (1 - Eps) * e10_selected) #eq 3.4 intro atm
    q2_selected = Eps * e2_selected / (Press_selected - rho80 * g * 2 - (1 - Eps) * e2_selected) #eq 3.4 intro atm
    H_selected = data_Flux[timeselection]['HSON'] #W/m2
    LE_selected = data_Flux[timeselection]['LEED'] #W/m2
    G_selected = data_Flux[timeselection]['FG0'] #W/m2
    ustar_selected = data_Flux[timeselection]['USTED'] #W/m2
    wCO2_selected = data_Flux[timeselection]['FCED'] #mg CO2/m2/s
    SWU_selected = data_Rad[timeselection]['SWU'] #W m-2
    SWD_selected = data_Rad[timeselection]['SWD'] #W m-2
    LWU_selected = data_Rad[timeselection]['LWU'] #W m-2
    LWD_selected = data_Rad[timeselection]['LWD'] #W m-2
    Windsp200_selected =  data_Windsp[timeselection]['F200'] #m s-1
    Windsp10_selected =  data_Windsp[timeselection]['F010'] #m s-1
    stdevWindsp200_selected = data_Windsp[timeselection]['SF200'] #m s-1
    stdevWindsp10_selected = data_Windsp[timeselection]['SF010'] #m s-1
    Winddir200_selected =  data_Winddir[timeselection]['D200'] #deg
    Winddir10_selected =  data_Winddir[timeselection]['D010'] #deg
    stdevWinddir200_selected = data_Winddir[timeselection]['SD200'] #deg
    stdevWinddir10_selected = data_Winddir[timeselection]['SD010'] #deg
    u200_selected = Windsp200_selected * np.cos((270. - Winddir200_selected)*2*np.pi/360) #2*np.pi/360 for conversion to rad. Make drawing to see the 270 minus wind dir
    v200_selected = Windsp200_selected * np.sin((270. - Winddir200_selected)*2*np.pi/360)
    u10_selected = Windsp10_selected * np.cos((270. - Winddir10_selected)*2*np.pi/360) #2*np.pi/360 for conversion to rad. Make drawing to see the 270 minus wind dir
    v10_selected = Windsp10_selected * np.sin((270. - Winddir10_selected)*2*np.pi/360)
else:    
    stdevTemp200_hourly = np.zeros((endhour-starthour))
    stdevTemp140_hourly = np.zeros((endhour-starthour))
    stdevTemp80_hourly = np.zeros((endhour-starthour))
    stdevTemp40_hourly = np.zeros((endhour-starthour))
    stdevTemp20_hourly = np.zeros((endhour-starthour))
    stdevTemp10_hourly = np.zeros((endhour-starthour))
    stdevTemp2_hourly = np.zeros((endhour-starthour))
    stdevq200_hourly = np.zeros((endhour-starthour))
    stdevq140_hourly = np.zeros((endhour-starthour))
    stdevq80_hourly = np.zeros((endhour-starthour))
    stdevq40_hourly = np.zeros((endhour-starthour))
    stdevq20_hourly = np.zeros((endhour-starthour))
    stdevq10_hourly = np.zeros((endhour-starthour))
    stdevq2_hourly = np.zeros((endhour-starthour))
    stdevPress_hourly = np.zeros((endhour-starthour))
    stdevH_hourly = np.zeros((endhour-starthour))
    stdevLE_hourly = np.zeros((endhour-starthour))
    stdevG_hourly = np.zeros((endhour-starthour))
    stdevustar_hourly = np.zeros((endhour-starthour))
    stdevwCO2_hourly = np.zeros((endhour-starthour))
    Temp200_mean = np.zeros((endhour-starthour))
    Temp140_mean = np.zeros((endhour-starthour))
    Temp80_mean = np.zeros((endhour-starthour))
    Temp40_mean = np.zeros((endhour-starthour))
    Temp20_mean = np.zeros((endhour-starthour))
    Temp10_mean = np.zeros((endhour-starthour))
    Temp2_mean = np.zeros((endhour-starthour))
    q200_mean = np.zeros((endhour-starthour))
    q140_mean = np.zeros((endhour-starthour))
    q80_mean = np.zeros((endhour-starthour))
    q40_mean = np.zeros((endhour-starthour))
    q20_mean = np.zeros((endhour-starthour))
    q10_mean = np.zeros((endhour-starthour))
    q2_mean = np.zeros((endhour-starthour))
    Press_mean = np.zeros((endhour-starthour))
    hours_mean = np.zeros((endhour-starthour))
    H_mean = np.zeros((endhour-starthour))
    LE_mean = np.zeros((endhour-starthour))
    G_mean = np.zeros((endhour-starthour))
    ustar_mean = np.zeros((endhour-starthour))
    wCO2_mean = np.zeros((endhour-starthour))
    Windsp200_mean = np.zeros((endhour-starthour))
    Windsp10_mean = np.zeros((endhour-starthour))
    
    for i in range(endhour-starthour):
        timeselection2 = np.logical_and(np.logical_and(day_True_False,np.logical_and(month_True_False,year_True_False)),np.logical_and(np.logical_and(hour>=starthour,hour<endhour),np.logical_and(np.logical_and(hour>=starthour+i,hour<starthour+i+1),minute_True_False)))
        Temp200toaverage = data_Temp[timeselection2]['TA200'] +273.15
        Temp140toaverage = data_Temp[timeselection2]['TA140'] +273.15
        Temp80toaverage = data_Temp[timeselection2]['TA080'] +273.15
        Temp40toaverage = data_Temp[timeselection2]['TA040'] +273.15
        Temp20toaverage = data_Temp[timeselection2]['TA020'] +273.15
        Temp10toaverage = data_Temp[timeselection2]['TA010'] +273.15
        Temp2toaverage = data_Temp[timeselection2]['TA002'] +273.15
        TD200_selected = data_DewP[timeselection2]['TD200'] +273.15
        TD140_selected = data_DewP[timeselection2]['TD140'] +273.15
        TD80_selected = data_DewP[timeselection2]['TD080'] +273.15
        TD40_selected = data_DewP[timeselection2]['TD040'] +273.15
        TD20_selected = data_DewP[timeselection2]['TD020'] +273.15
        TD10_selected = data_DewP[timeselection2]['TD010'] +273.15
        TD2_selected = data_DewP[timeselection2]['TD002'] +273.15
        Presstoaverage = data_Press[timeselection2]['AP0']*100 #Pa
        e200_selected = 610.7 * np.exp(17.2694*(TD200_selected - 273.16) / (TD200_selected - 35.86)) #eq 3.3 intro atm
        e140_selected = 610.7 * np.exp(17.2694*(TD140_selected - 273.16) / (TD140_selected - 35.86)) #eq 3.3 intro atm
        e80_selected = 610.7 * np.exp(17.2694*(TD80_selected - 273.16) / (TD80_selected - 35.86)) #eq 3.3 intro atm
        e40_selected = 610.7 * np.exp(17.2694*(TD40_selected - 273.16) / (TD40_selected - 35.86)) #eq 3.3 intro atm
        e20_selected = 610.7 * np.exp(17.2694*(TD20_selected - 273.16) / (TD20_selected - 35.86)) #eq 3.3 intro atm
        e10_selected = 610.7 * np.exp(17.2694*(TD10_selected - 273.16) / (TD10_selected - 35.86)) #eq 3.3 intro atm
        e2_selected = 610.7 * np.exp(17.2694*(TD2_selected - 273.16) / (TD2_selected - 35.86)) #eq 3.3 intro atm
        rho80 = (Presstoaverage - 1.22030 * g * 80) / (287.04 * Temp80toaverage) #1.22030 from us standard atmopsphere 40m
        q200toaverage = Eps * e200_selected / (Presstoaverage - rho80 * g * 200 - (1 - Eps) * e200_selected) #eq 3.4 intro atm
        q140toaverage = Eps * e140_selected / (Presstoaverage - rho80 * g * 140 - (1 - Eps) * e140_selected) #eq 3.4 intro atm
        q80toaverage = Eps * e80_selected / (Presstoaverage - rho80 * g * 80 - (1 - Eps) * e80_selected) #eq 3.4 intro atm
        q40toaverage = Eps * e40_selected / (Presstoaverage - rho80 * g * 40 - (1 - Eps) * e40_selected) #eq 3.4 intro atm
        q20toaverage = Eps * e20_selected / (Presstoaverage - rho80 * g * 20 - (1 - Eps) * e20_selected) #eq 3.4 intro atm
        q10toaverage = Eps * e10_selected / (Presstoaverage - rho80 * g * 10 - (1 - Eps) * e10_selected) #eq 3.4 intro atm
        q2toaverage = Eps * e2_selected / (Presstoaverage - rho80 * g * 2 - (1 - Eps) * e2_selected) #eq 3.4 intro atm
        
        Htoaverage = data_Flux[timeselection2]['HSON'] #W/m2
        LEtoaverage = data_Flux[timeselection2]['LEED'] #W/m2
        Gtoaverage = data_Flux[timeselection2]['FG0'] #W/m2
        ustartoaverage = data_Flux[timeselection2]['FG0'] #m/s
        wCO2toaverage = data_Flux[timeselection2]['FCED']
        Windsp200toaverage = data_Windsp[timeselection2]['F200'] #m s-1
        Windsp10toaverage = data_Windsp[timeselection2]['F010'] #m s-1
        Temp200_mean[i] = np.mean(Temp200toaverage)
        Temp140_mean[i] = np.mean(Temp140toaverage)
        Temp80_mean[i] = np.mean(Temp80toaverage)
        Temp40_mean[i] = np.mean(Temp40toaverage)
        Temp20_mean[i] = np.mean(Temp20toaverage)
        Temp10_mean[i] = np.mean(Temp10toaverage)
        Temp2_mean[i] = np.mean(Temp2toaverage)
        q200_mean[i] = np.mean(q200toaverage)
        q140_mean[i] = np.mean(q140toaverage)
        q80_mean[i] = np.mean(q80toaverage)
        q40_mean[i] = np.mean(q40toaverage)
        q20_mean[i] = np.mean(q20toaverage)
        q10_mean[i] = np.mean(q10toaverage)
        q2_mean[i] = np.mean(q2toaverage)
        Windsp200_mean[i] = np.mean(Windsp200toaverage)
        Windsp10_mean[i] = np.mean(Windsp10toaverage)
        
        Press_mean[i] = np.mean(Presstoaverage)
        H_mean[i] = np.mean(Htoaverage)
        LE_mean[i] = np.mean(LEtoaverage)
        G_mean[i] = np.mean(Gtoaverage)
        ustar_mean[i] = np.mean(ustartoaverage)
        wCO2_mean[i] = np.mean(wCO2toaverage)
        stdevTemp200_hourly[i] = np.std(Temp200toaverage)
        stdevTemp140_hourly[i] = np.std(Temp140toaverage)
        stdevTemp80_hourly[i] = np.std(Temp80toaverage)
        stdevTemp40_hourly[i] = np.std(Temp40toaverage)
        stdevTemp20_hourly[i] = np.std(Temp20toaverage)
        stdevTemp10_hourly[i] = np.std(Temp10toaverage)
        stdevTemp2_hourly[i] = np.std(Temp2toaverage)
        stdevq200_hourly[i] = np.std(q200toaverage)
        stdevq140_hourly[i] = np.std(q140toaverage)
        stdevq80_hourly[i] = np.std(q80toaverage)
        stdevq40_hourly[i] = np.std(q40toaverage)
        stdevq20_hourly[i] = np.std(q20toaverage)
        stdevq10_hourly[i] = np.std(q10toaverage)
        stdevq2_hourly[i] = np.std(q2toaverage)
        stdevH_hourly[i] = np.std(Htoaverage)
        stdevLE_hourly[i] = np.std(LEtoaverage)
        stdevG_hourly[i] = np.std(Gtoaverage)
        stdevustar_hourly[i] = np.std(ustartoaverage)
        stdevwCO2_hourly[i] = np.std(wCO2toaverage)
        hours_mean[i] = starthour + i + 0.5

#this dataset contains less
if use_mean:
    if (2003 in selectedyears and (9 in selectedmonths and 25 in selecteddays)):
        data_BLH = pd.read_csv(directory+'/'+'BLheight.txt',delim_whitespace=True)
        date_BLH = data_BLH['Date'] #
        year_BLH = np.array([int(str(date_BLH[i])[0:4]) for i in range(len(date_BLH))])
        month_BLH = np.array([int(str(date_BLH[i])[4:6]) for i in range(len(date_BLH))])
        day_BLH = np.array([int(str(date_BLH[i])[6:]) for i in range(len(date_BLH))])
        dhour_BLH = data_BLH['dhour'] #
        BLH_mean = np.zeros((endhour-starthour))
        dhour_BLH_mean = np.zeros((endhour-starthour))
        for i in range(endhour-starthour):
            timeselection_BLH = np.logical_and(np.logical_and(dhour_BLH>=starthour,dhour_BLH<endhour),np.logical_and(dhour_BLH>=starthour+i,dhour_BLH<starthour+i+1))
            BLH_toaverage = data_BLH[timeselection_BLH]['BLH'].astype(float) #astype(float), as it can give error if it are integers or strings
            BLH_mean[i] = np.mean(BLH_toaverage)
            dhour_BLH_toaverage = dhour_BLH[timeselection_BLH]
            dhour_BLH_mean[i] = np.mean(dhour_BLH_toaverage)
            
            
else:
    if (2003 in selectedyears and (9 in selectedmonths and selectedday == 25)):
        data_BLH = pd.read_csv(directory+'/'+'BLheight.txt',delim_whitespace=True)
        date_BLH = data_BLH['Date'] #
        year_BLH = np.array([int(str(date_BLH[i])[0:4]) for i in range(len(date_BLH))])
        month_BLH = np.array([int(str(date_BLH[i])[4:6]) for i in range(len(date_BLH))])
        day_BLH = np.array([int(str(date_BLH[i])[6:]) for i in range(len(date_BLH))])
        dhour_BLH = data_BLH['dhour'] #
        timeselection_BLH = np.logical_and(dhour_BLH>=starthour,dhour_BLH<endhour)
        BLH_selected = data_BLH[timeselection_BLH]['BLH'].astype(float) #astype(float), as it can give error if it are integers or strings
        dhour_BLH_selected = dhour_BLH[timeselection_BLH]
    
#CO2 datafile
data_CO2 = pd.read_csv(directory+'/'+'CO2-September2003.txt',skiprows = 3,delim_whitespace=True)    
date_CO2 = data_CO2['Date'] #
time_CO2 = data_CO2['Time'] #
hour_CO2 = np.zeros(len(time_CO2))
day_CO2 = np.zeros(len(time_CO2),dtype=int)
month_CO2 = np.zeros(len(time_CO2),dtype=int)
year_CO2 = np.zeros(len(time_CO2),dtype=int)
for i in range(len(date_CO2)):
    day_CO2[i] = int(str(date_CO2[i])[0:2])
    month_CO2[i] = int(str(date_CO2[i])[3:5])
    year_CO2[i] = int(str(date_CO2[i])[6:])     
for i in range(len(time_CO2)):
    hour_CO2[i] = float(str(time_CO2[i])[-5:-3])
    if hour_CO2[i] != 0: #Date/time indicates end of 1 hour averaging interval in UTC in the datafile
        hour_CO2[i] = hour_CO2[i] - 0.5
    else:
        hour_CO2[i] = 23.5
        day_CO2[i] = day_CO2[i] - 1 #than it is from previous day
month_True_False = np.zeros(len(time_CO2),dtype=bool)
year_True_False = np.zeros(len(time_CO2),dtype=bool)
day_True_False = np.zeros(len(time_CO2),dtype=bool)
for i in range(len(month_True_False)):
    if month_CO2[i] in selectedmonths:
        month_True_False[i] = True
    if year_CO2[i] in selectedyears:
        year_True_False[i] = True
    if day_CO2[i] in selecteddays:
        day_True_False[i] = True #True for all selected days only      
if use_mean != True:
    timeselection = np.logical_and(np.logical_and(day_CO2 == selectedday,np.logical_and(month_True_False,year_True_False)),np.logical_and(hour_CO2>=starthour,hour_CO2<endhour))
    CO2_200_selected = data_CO2[timeselection]['CO2_200'] #ppm
    CO2_120_selected = data_CO2[timeselection]['CO2_120'] #ppm
    CO2_60_selected = data_CO2[timeselection]['CO2_60'] #ppm
    CO2_20_selected = data_CO2[timeselection]['CO2_20'] #ppm
    hour_CO2_selected = hour_CO2[timeselection]
else:
    stdevCO2_200_hourly = np.zeros((endhour-starthour))
    stdevCO2_120_hourly = np.zeros((endhour-starthour))
    stdevCO2_60_hourly = np.zeros((endhour-starthour))
    stdevCO2_20_hourly = np.zeros((endhour-starthour))
    CO2_200_mean = np.zeros((endhour-starthour))
    CO2_120_mean = np.zeros((endhour-starthour))
    CO2_60_mean = np.zeros((endhour-starthour))
    CO2_20_mean = np.zeros((endhour-starthour))
    for i in range(endhour-starthour):
        timeselection2 = np.logical_and(np.logical_and(day_True_False,np.logical_and(month_True_False,year_True_False)),np.logical_and(np.logical_and(hour_CO2>=starthour,hour_CO2<endhour),np.logical_and(hour_CO2>=starthour+i,hour_CO2<starthour+i+1)))
        CO2_200toaverage = data_CO2[timeselection2]['CO2_200'] #ppm
        CO2_120toaverage = data_CO2[timeselection2]['CO2_120'] #ppm
        CO2_60toaverage = data_CO2[timeselection2]['CO2_60'] #ppm
        CO2_20toaverage = data_CO2[timeselection2]['CO2_20'] #ppm
        CO2_200_mean[i] = np.mean(CO2_200toaverage)
        CO2_120_mean[i] = np.mean(CO2_120toaverage)
        CO2_60_mean[i] = np.mean(CO2_60toaverage)
        CO2_20_mean[i] = np.mean(CO2_20toaverage)
        stdevCO2_200_hourly[i] = np.std(CO2_200toaverage)
        stdevCO2_120_hourly[i] = np.std(CO2_120toaverage)
        stdevCO2_60_hourly[i] = np.std(CO2_60toaverage)
        stdevCO2_20_hourly[i] = np.std(CO2_20toaverage)
######################################
###### end user input: load obs ######
######################################    

#optimisation
priormodinput = fwdm.model_input()
###########################################
###### user input: prior model param ######
########################################### 
priormodinput.COS        = 0.400 #ppb
priormodinput.CO2measuring_height = 200. - canopy_height
priormodinput.CO2measuring_height2 = 120 - canopy_height
priormodinput.CO2measuring_height3 = 60 - canopy_height
priormodinput.CO2measuring_height4 = 20 - canopy_height
priormodinput.Tmeasuring_height = 200 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height2 = 140 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height3 = 80 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height4 = 40 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height5 = 20 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height6 = 10 - canopy_height #0 would be a problem
priormodinput.Tmeasuring_height7 = 2 - canopy_height #0 would be a problem
priormodinput.qmeasuring_height = 200. - canopy_height
priormodinput.qmeasuring_height2 = 140. - canopy_height
priormodinput.qmeasuring_height3 = 80. - canopy_height
priormodinput.qmeasuring_height4 = 40. - canopy_height
priormodinput.qmeasuring_height5 = 20. - canopy_height
priormodinput.qmeasuring_height6 = 10. - canopy_height
priormodinput.qmeasuring_height7 = 2. - canopy_height
priormodinput.alfa_sto = 1
priormodinput.gciCOS = 0.2 /(1.2*1000) * 28.9
priormodinput.ags_C_mode = 'MXL' 
priormodinput.sw_useWilson  = False
priormodinput.dt         = 60       # time step [s]
priormodinput.tstart     = starthour   # time of the day [h UTC]
priormodinput.runtime    = (endhour-priormodinput.tstart)*3600 + priormodinput.dt   # total run time [s]
priormodinput.sw_ml      = True      # mixed-layer model switch
priormodinput.sw_shearwe = False     # shear growth mixed-layer switch
priormodinput.sw_fixft   = False     # Fix the free-troposphere switch
priormodinput.h          = np.array(BLH_selected)[0]      # initial ABL height [m]
if use_mean:
    priormodinput.Ps         = Press_mean[0]   # surface pressure [Pa]
else:
    priormodinput.Ps         = np.array(Press_selected)[0]   # surface pressure [Pa]
priormodinput.divU       = 0.00        # horizontal large-scale divergence of wind [s-1]
if use_mean:
     priormodinput.theta      = 284
else:
    priormodinput.theta      = np.array(Temp200_selected)[0]*((np.array(Press_selected)[0]-200*9.81*np.array(rho80)[0])/100000)**(-287.04/1005)    # initial mixed-layer potential temperature [K]
priormodinput.deltatheta = 4.20       # initial temperature jump at h [K]
priormodinput.gammatheta = 0.0036     # free atmosphere potential temperature lapse rate [K m-1]
priormodinput.gammatheta2 = 0.015     # free atmosphere potential temperature lapse rate for h > htrans [K m-1]
priormodinput.htrans = 95000 #height of BL above which to use gammatheta2 instead of gammatheta
priormodinput.advtheta   = 0        # advection of heat [K s-1]
#priormodinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
if use_mean:
    priormodinput.q          = 0.0049     # initial mixed-layer specific humidity [kg kg-1]
else:
    priormodinput.q          = np.array(q200_selected)[0]
priormodinput.deltaq     = -0.0008    # initial specific humidity jump at h [kg kg-1]
priormodinput.gammaq     = -1.2e-6        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priormodinput.advq       = 0        # advection of moisture [kg kg-1 s-1] 
if use_mean:
    priormodinput.CO2        = 422.      # initial mixed-layer CO2 [ppm]
else:
    priormodinput.CO2        = np.array(CO2_200_selected)[0]      # initial mixed-layer CO2 [ppm]
priormodinput.deltaCO2   = -44.      # initial CO2 jump at h [ppm]
priormodinput.deltaCOS   = 0.050      # initial COS jump at h [ppb]
priormodinput.gammaCO2   = -0.000        # free atmosphere CO2 lapse rate [ppm m-1]
priormodinput.gammaCOS   = 0.00        # free atmosphere COS lapse rate [ppb m-1]
priormodinput.advCO2     = 0         # advection of CO2 [ppm s-1]
priormodinput.advCOS     = 0.        # advection of COS [ppb s-1]
priormodinput.wCOS       = 0.01        # surface kinematic COS flux [ppb m s-1]
priormodinput.sw_wind    = True     # prognostic wind switch
priormodinput.u          = np.array(Windsp200_selected)[0]        # initial mixed-layer u-wind speed [m s-1]
priormodinput.deltau     = 3.        # initial u-wind jump at h [m s-1]
priormodinput.gammau     = 0.002      # free atmosphere u-wind speed lapse rate [s-1]
priormodinput.advu       = 0.        # advection of u-wind [m s-2]
priormodinput.v          = 0      # initial mixed-layer u-wind speed [m s-1]
priormodinput.deltav     = 0       # initial u-wind jump at h [m s-1]
priormodinput.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
priormodinput.advv       = 0.        # advection of v-wind [m s-2]
priormodinput.sw_sl      = True     # surface layer switch
priormodinput.ustar      = 0.3       # surface friction velocity [m s-1]
priormodinput.z0m        = 0.05      # roughness length for momentum [m]
priormodinput.z0h        = 0.01     # roughness length for scalars [m]
priormodinput.sw_rad     = True     # radiation switch
priormodinput.lat        = 51.971     # latitude [deg] #https://icdc.cen.uni-hamburg.de/1/daten/atmosphere/weathermast-cabauw.html
priormodinput.lon        = 4.927     # longitude [deg]
priormodinput.doy        = 268.      # day of the year [-]
priormodinput.fc         = 2 * 7.2921e-5 * np.sin(priormodinput.lat*2*np.pi/360.)     # Coriolis parameter [m s-1]
priormodinput.cc         = 0.0       # cloud cover fraction [-]
priormodinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priormodinput.sw_ls      = True     # land surface switch
priormodinput.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
priormodinput.wg         = 0.48      # volumetric water content top soil layer [m3 m-3]
priormodinput.w2         = 0.48      # volumetric water content deeper soil layer [m3 m-3]
priormodinput.cveg       = 0.9      # vegetation fraction [-]
priormodinput.Tsoil      = 282.      # temperature top soil layer [K]
priormodinput.T2         = 285      # temperature deeper soil layer [K]
priormodinput.a          = 0.083     # Clapp and Hornberger retention curve parameter a
priormodinput.b          = 11.4      # Clapp and Hornberger retention curve parameter b
priormodinput.p          = 12.        # Clapp and Hornberger retention curve parameter c
priormodinput.CGsat      = 3.6e-6   # saturated soil conductivity for heat
priormodinput.wsat       = 0.6     # saturated volumetric water content (Sun 2017)
priormodinput.wfc        = 0.491     # volumetric water content field capacity [-]
priormodinput.wwilt      = 0.314     # volumetric water content wilting point [-]
priormodinput.C1sat      = 0.342     
priormodinput.C2ref      = 0.3
priormodinput.LAI        = 2.        # leaf area index [-]
priormodinput.gD         = None       # correction factor transpiration for VPD [-]
priormodinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]
priormodinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priormodinput.alpha      = 0.25      # surface albedo [-]
priormodinput.Ts         = np.array(Temp10_selected)[0]      # initial surface temperature [K]
priormodinput.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
priormodinput.Wl         = 0.00014    # equivalent water layer depth for wet vegetation [m]
priormodinput.Lambda     = 5.9       # thermal diffusivity skin layer [-]
priormodinput.c3c4       = 'c3'      # Plant type ('c3' or 'c4')
priormodinput.sw_cu      = False     # Cumulus parameterization switch
priormodinput.dz_h       = 150.      # Transition layer thickness [m]
priormodinput.Cs         = 1e12      # drag coefficient for scalars [-]
priormodinput.sw_dynamicsl_border = True
priormodinput.sw_model_stable_con = True
priormodinput.sw_printwarnings = False
priormodinput.sw_use_ribtol = True
priormodinput.sw_advfp = True #prescribed advection to take place over full profile (also in Free troposphere), only in ML if FALSE
priormodinput.sw_dyn_beta = True
#tsteps = int(np.floor(priormodinput.runtime / priormodinput.dt)) #the number of timesteps, used below
#priormodinput.wtheta_input = np.zeros(tsteps)
#priormodinput.wq_input = np.zeros(tsteps)
#priormodinput.wCO2_input = np.zeros(tsteps)
#for t in range(tsteps):
#    if (t*priormodinput.dt >= 1.5*3600) and (t*priormodinput.dt <= 9*3600):
#        priormodinput.wtheta_input[t] = 0.08 * np.sin(np.pi*(t*priormodinput.dt-5400)/27000)
#    else:
#        priormodinput.wtheta_input[t] = 0
#    priormodinput.wq_input[t] = 0.087 * np.sin(np.pi * t*priormodinput.dt/43200) /1000 #/1000 for conversion to kg/kg m s-1
#    if (t*priormodinput.dt >= 2*3600) and (t*priormodinput.dt <= 9.5*3600):
#        priormodinput.wCO2_input[t] = -0.1*np.sin(np.pi*(t*priormodinput.dt-7200)/27000)
#    else:
#        priormodinput.wCO2_input[t] = 0.
#priormodinput.wCO2       = priormodinput.wCO2_input[0]        # surface kinematic CO2 flux [ppm m s-1]
#priormodinput.wq         = priormodinput.wq_input[0]     # surface kinematic moisture flux [kg kg-1 m s-1]
#priormodinput.wtheta     = priormodinput.wtheta_input[0]       # surface kinematic heat flux [K m s-1]
priormodinput.wCO2       = 0.0000001
priormodinput.wq       = 0.0000001
priormodinput.wtheta       = 0.0000001

#soil COS model
priormodinput.soilCOSmodeltype   = None #can be set to None or 'Sun_Ogee'
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
priorinput.obs_sca_cf_H = 1.0 #Always use this format, 'obs_sca_cf_' plus the observation type you want to scale
#e.g. priorinput.obs_sca_cf_H = 1.5 means that in the cost function all obs of H will be multiplied with 1.5. But only if obs_sca_cf_H also in the state!
priorinput.obs_sca_cf_LE = 1.0
#e.g. state=['h','q','theta','gammatheta','deltatheta','deltaq','alpha','gammaq','wg','wtheta','z0h','z0m','ustar','wq','divU']
state=['advtheta','advq','advCO2','deltatheta','gammatheta','deltaq','gammaq','deltaCO2','gammaCO2','alfa_sto','alpha','obs_sca_cf_H','obs_sca_cf_LE','z0m','z0h']
obslist =['Tmh','Tmh2','Tmh3','Tmh4','Tmh5','Tmh6','Tmh7','qmh','qmh2','qmh3','qmh4','qmh5','qmh6','qmh7','CO2mh','CO2mh2','CO2mh3','CO2mh4','h','LE','H','wCO2','Swout']

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
    priorvar['alpha'] = 0.1**2
    priorvar['gammatheta'] = 0.003**2 
    priorvar['gammatheta2'] = 0.003**2 
    priorvar['gammaq'] = (0.003e-3)**2
    priorvar['gammaCO2'] = (60.e-3)**2 
    priorvar['deltatheta'] = 1.5**2
    priorvar['deltaq'] = 0.002**2
    priorvar['theta'] = 1**2
    priorvar['deltaCO2'] = 40**2
    priorvar['CO2'] = 30**2
    priorvar['alfa_sto'] = 0.25**2
    priorvar['z0m'] = 0.1**2
    priorvar['z0h'] = 0.1**2
    priorvar['advtheta'] = (2/3600)**2 
    priorvar['advq'] = (0.002/3600)**2 
    priorvar['advCO2'] = (40/3600)**2
    priorvar['obs_sca_cf_H'] = 0.4**2 
    priorvar['obs_sca_cf_LE'] = 0.4**2 
#    priorvar['wtheta'] = (150/1.1/1005)**2
#    priorvar['wq'] = (0.1e-3)**2
#    priorvar['ustar'] = (0.7)**2
#    priorvar['h'] = 200**2
#    priorvar['q'] = 0.003**2
#    priorvar['wg'] = 0.2**2
#    priorvar['deltaCOS'] = 0.02**2
#    priorvar['COS'] = 0.1**2
#    priorvar['fCA'] = 1.e3**2
#    priorvar['divU'] = 0.0001**2
#    priorvar['u'] = 1.5**2
#    priorvar['v'] = 1.5**2
#    priorvar['deltau'] = 1.5**2
#    priorvar['deltav'] = 1.5**2
#    priorvar['gammau'] = 0.02**2
#    priorvar['gammav'] = 0.02**2
#    priorvar['advu'] = 0.0006**2
#    priorvar['advv'] = 0.0006**2
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
    if use_mean:
        if item == 'Tmh':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp200_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp200_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
            display_names[item] = 'T_200'
        if item == 'Tmh2':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp140_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp140_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        if item == 'Tmh3':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp80_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp80_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        if item == 'Tmh4':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp40_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp40_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        if item == 'Tmh5':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp20_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp20_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        if item == 'Tmh6':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp10_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp10_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        if item == 'Tmh7':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp2_mean) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = np.ndarray.flatten(stdevTemp2_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'K'
        elif item == 'CO2mh':
            optim.__dict__['obs_'+item] = np.array(CO2_200_mean)
            optim.__dict__['error_obs_' + item] = stdevCO2_200_hourly
            obs_times[item] = hours_mean * 3600. #same as for temperature
            obs_units[item] = 'ppm'
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'CO2mh2':
            optim.__dict__['obs_'+item] = np.array(CO2_120_mean)
            optim.__dict__['error_obs_' + item] = stdevCO2_120_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
        elif item == 'CO2mh3':
            optim.__dict__['obs_'+item] = np.array(CO2_60_mean)
            optim.__dict__['error_obs_' + item] = stdevCO2_60_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
        elif item == 'CO2mh4':
            optim.__dict__['obs_'+item] = np.array(CO2_20_mean)
            optim.__dict__['error_obs_' + item] = stdevCO2_20_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
        elif item == 'wCO2':
            optim.__dict__['obs_'+item] = np.array(wCO2_mean)
            optim.__dict__['error_obs_' + item] = stdevwCO2_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'mg CO2 m-2 s-1'
        elif item == 'h':
            optim.__dict__['obs_'+item] = np.array(BLH_mean) #we just have one day
            print('WARNING: the obs of h are not really a mean...')
            optim.__dict__['error_obs_' + item] = [100 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(dhour_BLH_mean) * 3600.
            for i in range(len(obs_times[item])):
                obs_times[item][i] = round(obs_times[item][i],0) #Very important, since otherwise the obs will be a fraction of a second off and will not be used
            obs_units[item] = 'm'
        elif item == 'qmh':
            optim.__dict__['obs_'+item] = np.array(q200_mean)
            optim.__dict__['error_obs_' + item] = np.array(stdevq200_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh2':
            optim.__dict__['obs_'+item] = np.array(q140_mean)
            optim.__dict__['error_obs_' + item] = np.array(stdevq140_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh3':
            optim.__dict__['obs_'+item] = np.array(q80_mean)
            optim.__dict__['error_obs_' + item] = np.array(stdevq80_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh4':
            optim.__dict__['obs_'+item] = np.array(q40_mean)
            optim.__dict__['error_obs_' + item] = np.array(stdevq40_hourly)
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh5':
            optim.__dict__['obs_'+item] = np.array(q20_mean)
            optim.__dict__['error_obs_' + item] = stdevq20_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh6':
            optim.__dict__['obs_'+item] = q10_mean
            optim.__dict__['error_obs_' + item] = stdevq10_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'qmh7':
            optim.__dict__['obs_'+item] = q2_mean
            optim.__dict__['error_obs_' + item] = stdevq2_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'kg kg-1'
        elif item == 'ustar':
            optim.__dict__['obs_'+item] = ustar_mean
            optim.__dict__['error_obs_' + item] = stdevustar_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'm s-1'
        elif item == 'H':
            optim.__dict__['obs_'+item] = H_mean
            optim.__dict__['error_obs_' + item] = stdevH_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'W m-2'
        elif item == 'LE':
            optim.__dict__['obs_'+item] = LE_mean
            optim.__dict__['error_obs_' + item] = stdevLE_hourly
            obs_times[item] = hours_mean * 3600.
            obs_units[item] = 'W m-2'
    else:
        refnumobs = np.sum(~np.isnan(np.array(BLH_selected))) #only for the weights, a reference number of the number of non-nan obs, chosen to be based on h here. When e.g. wco2 has more obs than this number,
        #it will be given a lower weight per individual observation than h
        obstimes_T = []
        for i in range(starthour,endhour):
            for minute in selectedminutes:
                obstimes_T.append(i * 3600. + minute * 60.)
        if item == 'Tmh':
            optim.__dict__['obs_'+item] = np.array(Temp200_selected) #np.array since it is a pandas dataframe
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_200'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
                # np.sum(~np.isnan(optim.__dict__['obs_'+item])) used here instead of len(optim.__dict__['obs_'+item]), since nan data should not count for the length of the observation array. ~ inverts the np.isnan array. 
        if item == 'Tmh2':
            optim.__dict__['obs_'+item] = np.array(Temp140_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_140'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh3':
            optim.__dict__['obs_'+item] = np.array(Temp80_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_80'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh4':
            optim.__dict__['obs_'+item] = np.array(Temp40_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_40'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh5':
            optim.__dict__['obs_'+item] = np.array(Temp20_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_20'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh6':
            optim.__dict__['obs_'+item] = np.array(Temp10_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_10'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh7':
            optim.__dict__['obs_'+item] = np.array(Temp2_selected) #flatten becuase it is from the second datafile, which has a different structure
            optim.__dict__['error_obs_' + item] = [0.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'K'
            display_names[item] = 'T_2'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'CO2mh':
            optim.__dict__['obs_'+item] = np.array(CO2_200_selected)
            optim.__dict__['error_obs_' + item] = [5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
            display_names[item] = 'CO2_200'
        elif item == 'CO2mh2':
            optim.__dict__['obs_'+item] = np.array(CO2_120_selected)
            optim.__dict__['error_obs_' + item] = [5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
            display_names[item] = 'CO2_120'
        elif item == 'CO2mh3':
            optim.__dict__['obs_'+item] = np.array(CO2_60_selected)
            optim.__dict__['error_obs_' + item] = [5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
            display_names[item] = 'CO2_60'
        elif item == 'CO2mh4':
            optim.__dict__['obs_'+item] = np.array(CO2_20_selected)
            optim.__dict__['error_obs_' + item] = [5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            obs_units[item] = 'ppm'
            display_names[item] = 'CO2_20'
        elif item == 'wCO2':
            optim.__dict__['obs_'+item] = np.array(wCO2_selected)
            optim.__dict__['error_obs_' + item] = [0.15 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'mg CO2 m-2 s-1'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'h':
            optim.__dict__['obs_'+item] = np.array(BLH_selected) #we just have one day
            optim.__dict__['error_obs_' + item] = [120 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(dhour_BLH_selected) * 3600
            for i in range(len(obs_times[item])):
                obs_times[item][i] = round(obs_times[item][i],0) #Very important, since otherwise the obs will be a fraction of a second off and will not be used
            obs_units[item] = 'm'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh':
            optim.__dict__['obs_'+item] = np.array(q200_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_200'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh2':
            optim.__dict__['obs_'+item] = np.array(q140_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_140'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh3':
            optim.__dict__['obs_'+item] = np.array(q80_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_80'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh4':
            optim.__dict__['obs_'+item] = np.array(q40_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_40'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh5':
            optim.__dict__['obs_'+item] = np.array(q20_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_20'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh6':
            optim.__dict__['obs_'+item] = np.array(q10_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_10'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh7':
            optim.__dict__['obs_'+item] = np.array(q2_selected)
            optim.__dict__['error_obs_' + item] = [0.00025 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'kg kg-1'
            display_names[item] = 'q_2'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'ustar':
            optim.__dict__['obs_'+item] = np.array(ustar_selected)
            optim.__dict__['error_obs_' + item] = [0.15 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'm s-1'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'H':
            optim.__dict__['obs_'+item] = np.array(H_selected)
            optim.__dict__['error_obs_' + item] = [70 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'W m-2'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'LE':
            optim.__dict__['obs_'+item] = np.array(LE_selected)
            optim.__dict__['error_obs_' + item] = [70 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'W m-2'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'u':
            optim.__dict__['obs_'+item] = np.array(u200_selected)
            optim.__dict__['error_obs_' + item] = np.array(stdevWindsp200_selected) #not fully correct, includes variation in v...
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'm s-1'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'v':
            optim.__dict__['obs_'+item] = np.array(v200_selected)
            optim.__dict__['error_obs_' + item] = np.array(stdevWindsp200_selected)
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'm s-1'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'Swout':
            optim.__dict__['obs_'+item] = np.array(SWU_selected)
            optim.__dict__['error_obs_' + item] = [20 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            obs_units[item] = 'W m-2'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
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
optim.pstate = [] #initial state values
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
    obs_sca_cf_mem = {}
    optim_mem.pstate = []
    for item in state:
        optim_mem.pstate.append(priorinput_mem.__dict__[item])
        if item.startswith('obs_sca_cf_'):
            obsname = item.split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
            obs_sca_cf_mem[obsname] = cp.deepcopy(priorinput_mem.__dict__[item])
    optim_mem.pstate = np.array(optim_mem.pstate)
    optiminput_mem = cp.deepcopy(priorinput_mem) #deepcopy!
    params = tuple([optiminput_mem,state,obs_times,obs_weights])
    optim_mem.checkpoint = cp.deepcopy(checkpoint_prior_mem) #needed, as first thing optimizer does is calculating the gradient
    optim_mem.checkpoint_init = cp.deepcopy(checkpoint_init_prior_mem) #needed, as first thing optimizer does is calculating the gradient
    for item in optim_mem.obs:
        if item in obs_sca_cf_mem:
            obs_scale = obs_sca_cf_mem[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
        else:
            obs_scale = 1.0 
        weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
        k = 0 #counter for the observations (specific for each type of obs) 
        for ti in range(priormodel_mem.tsteps):
            if round(priormodel_mem.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[item]]: #so if we are at a time where we have an obs
                if item in obs_weights:
                    weight = obs_weights[item][k]
                forcing = weight * (Hx_prior_mem[item][ti] - obs_scale * optim_mem.__dict__['obs_'+item][k])/(optim_mem.__dict__['error_obs_' + item][k]**2) #don't include the backgorund term of the cost function in the forcing!
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
            print('Minimisation aborted due to nan') #discard_nan_minims == False allows to use last non-nan result in the optimisation, otherwise we throw away the optimisation
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
    if not optim.obs[i].startswith('qmh'): #q should be plotted differently for clarity
        plt.plot(priormodel.out.t,priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
        plt.plot(priormodel.out.t,optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red',label = 'post')
        plt.plot(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black')
        plt.errorbar(obs_times[optim.obs[i]]/3600,optim.__dict__['obs_'+optim.obs[i]],yerr=optim.__dict__['error_obs_' + optim.obs[i]],ecolor='black',fmt='None')
        if 'obs_sca_cf_'+obslist[i] in state: #plot the obs scaled with the scaling factors (if determined)
            plt.plot(obs_times[obslist[i]]/3600,optimalinput.__dict__['obs_sca_cf_'+obslist[i]]*optim.__dict__['obs_'+obslist[i]], linestyle=' ', marker='o',color = 'red')
        if optim.obs[i] in display_names:
            plt.ylabel(display_names[optim.obs[i]] +' ('+ obs_units[optim.obs[i]] + ')')
        else:
            plt.ylabel(optim.obs[i] +' ('+ obs_units[optim.obs[i]] + ')')
    else:
        plt.plot(priormodel.out.t,1000*priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
        plt.plot(priormodel.out.t,1000*optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red',label = 'post')
        plt.plot(obs_times[optim.obs[i]]/3600,1000*optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black')
        plt.errorbar(obs_times[optim.obs[i]]/3600,1000*optim.__dict__['obs_'+optim.obs[i]],yerr=1000*optim.__dict__['error_obs_' + optim.obs[i]],ecolor='black',fmt='None')
        if optim.obs[i] in display_names:
            plt.ylabel(display_names[optim.obs[i]] +' ('+ 'g/kg' + ')')
        else:
            plt.ylabel(optim.obs[i] +' ('+ 'g/kg' + ')')
    plt.xlabel('time (h)')
    plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    plt.legend()
    if write_to_f:
        if optim.obs[i] == 'h':
            plt.savefig('fig_fit_'+'BLheight(h)'+'.png', format='png') #renamed since Windows cannnot have a file 'fig_fit_h' and 'fig_fit_H' in the same folder          
        else:
            plt.savefig('fig_fit_'+optim.obs[i]+'.png', format='png')

    
########################################################
###### user input: additional plotting etc. ############
########################################################       

#fig = plt.figure()
#plt.plot(optimalmodel.out.t,priormodel.out.wCO2, linestyle='--', marker='o',color='yellow')
#plt.plot(optimalmodel.out.t,optimalmodel.out.wCO2, linestyle='--', marker='o',color='red')
#plt.plot(hours_mean,wCO2_mean, linestyle=' ', marker='o',color='black')
#plt.ylabel('CO2 flux (mg CO2/m2/s)')
#plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#if write_to_f:
#    plt.savefig('fig_wCO2.png', format='png')

fig = plt.figure()
if priormodel.sw_ls:
    plt.plot(priormodel.out.t,priormodel.out.__dict__['H'], linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['H'], linestyle=' ', marker='o',color='red',label = 'post')
else:
    plt.plot(priormodel.out.t,priormodel.out.__dict__['wtheta']*priormodel.rho*priormodel.cp, linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['wtheta']*priormodel.rho*priormodel.cp, linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,H_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('H (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'H'+'.png', format='png')
    
fig = plt.figure()
if priormodel.sw_ls:
    plt.plot(priormodel.out.t,priormodel.out.__dict__['LE'], linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['LE'], linestyle=' ', marker='o',color='red',label = 'post')
else:
    plt.plot(priormodel.out.t,priormodel.out.__dict__['wq']*priormodel.rho*priormodel.Lv, linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['wq']*priormodel.rho*priormodel.Lv, linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,LE_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('LE (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'LE'+'.png', format='png')

fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['Swin'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Swin'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,SWD_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('SWD (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'SWD'+'.png', format='png')
    
fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['Swout'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Swout'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,SWU_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('SWU (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'SWU'+'.png', format='png')

fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['Lwout'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Lwout'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,LWU_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('LWU (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'LWU'+'.png', format='png')
    
fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['Lwin'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Lwin'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,LWD_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('LWD (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'LWD'+'.png', format='png')
    
fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['G'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['G'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,G_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('G (W m-2)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'G'+'.png', format='png')
    
fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['theta'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.__dict__['theta'], linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(np.array(obstimes_T)/3600,Temp200_selected*((Press_selected-200*9.81*rho80)/100000)**(-287.04/1005), linestyle=' ', marker='o',color = 'black')
plt.ylabel('theta (K)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'theta'+'.png', format='png')
    
#fig = plt.figure()
#plt.plot(np.array(obstimes_T)/3600,Temp200_selected*((Press_selected-200*9.81*rho80)/100000)**(-287.04/1005), linestyle=' ', marker='o',color = 'black')
#plt.ylabel('theta (K)')
#plt.ylim = ([282,292])
#plt.xlim = ([7,17])
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)

fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.__dict__['h'], linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(np.array(dhour_BLH_selected),BLH_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('h (m)')
plt.xlabel('time (h)')
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
plt.legend()
if write_to_f:
    plt.savefig('fig_'+'h'+'.png', format='png')

profileheights = np.array([priorinput.CO2measuring_height4,priorinput.CO2measuring_height3,priorinput.CO2measuring_height2,priorinput.CO2measuring_height])    
colorlist = ['yellow','red','green','blue','orange','pink']
markerlist = ['x','v','s','p']

fig = plt.figure()
i = 0
for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
    color = colorlist[i]
    plt.plot(priormodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'pmod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
    plt.plot(priormodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
    plt.plot(priormodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
    plt.plot(priormodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
    i += 1
plt.ylabel('height (m)')
plt.xlabel('CO2 mixing ratio (ppm)')  
i = 0
for ti in range(0,len(obs_times['CO2mh']),2):
    marker = markerlist[i]
    color = colorlist[i]
    plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
    plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
    plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
    plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
    i += 1
plt.legend(fontsize=8)  
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_'+'CO2'+'_profile_prior.png', format='png')

fig = plt.figure()
i = 0
for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
    color = colorlist[i]
    plt.plot(optimalmodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'mod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
    plt.plot(optimalmodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
    plt.plot(optimalmodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
    plt.plot(optimalmodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
    i += 1
plt.ylabel('height (m)')
plt.xlabel('CO2 mixing ratio (ppm)')  
i = 0
for ti in range(0,len(obs_times['CO2mh']),2):
    marker = markerlist[i]
    color = colorlist[i]
    plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
    plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
    plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
    plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
    i += 1
plt.legend(fontsize=8)  
plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
if write_to_f:
    plt.savefig('fig_'+'CO2'+'_profile.png', format='png')
        
fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.H+priormodel.out.LE+priormodel.out.G, linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.H+optimalmodel.out.LE+optimalmodel.out.G, linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(obs_times['H']/3600,optim.obs_H+optim.obs_LE+G_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('H+LE+G')

fig = plt.figure()
plt.plot(priormodel.out.t,priormodel.out.Swin+priormodel.out.Lwin-priormodel.out.Swout-priormodel.out.Lwout, linestyle=' ', marker='o',color='yellow',label = 'prior')
plt.plot(priormodel.out.t,optimalmodel.out.Swin+optimalmodel.out.Lwin-optimalmodel.out.Swout-optimalmodel.out.Lwout, linestyle=' ', marker='o',color='red',label = 'post')
plt.plot(obs_times['H']/3600,SWD_selected+LWD_selected-SWU_selected-LWU_selected, linestyle=' ', marker='o',color = 'black')
plt.ylabel('Qnet (W/m²)')
plt.xlabel('time (h)')
