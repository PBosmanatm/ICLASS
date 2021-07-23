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
import matplotlib.pyplot as plt
import shutil
import os
from joblib import Parallel, delayed
import pandas as pd
import glob
import matplotlib.style as style
import pickle
style.use('classic')

##################################
###### user input: settings ######
##################################
ana_deriv = True #use analytical or numerical derivative
use_backgr_in_cost = True #include the background (prior) part of the cost function
write_to_f = True #write output and figures to files
use_ensemble = True #use an ensemble of optimisations
if use_ensemble:
    nr_of_members = 175
    use_covar_to_pert = False #whether to take prior covariance (if specified) into account when perturbing the ensemble 
    pert_non_state_param = False #perturb parameters that are not in the state
    est_post_pdf_covmatr = True #estimate the posterior pdf and covariance matrix of the state (and more)
    if est_post_pdf_covmatr:
        plot_perturbed_obs = False #Plot the perturbed observations of the ensemble members
        nr_bins = int(nr_of_members/10) #nr of bins for the pdfs
        succes_opt_crit = 1.7 #the chi squared at which an optimisation is considered successful (lower or equal to is succesfull)
    print_status_dur_ens = False #whether to print state etc info during ensemble of optimisations (during member 0 printing will always take place)
estimate_model_err = False #estimate the model error by perturbing specified non-state parameters
if estimate_model_err:
    nr_of_members_moderr = 30 #number of members for the ensemble that estimates the model error
imposeparambounds = True #force the optimisation to keep parameters within specified bounds (tnc only) and when using ensemble, keep priors within bounds (tnc and bfgs)
paramboundspenalty = False #add a penalty to the cost function when parameter bounds exceeded in the optimisation
if paramboundspenalty:
    setNanCostfOutBoundsTo0 = True #when cost function becomes nan when params outside specified bounds, set cost f to zero before adding penalty (nan + number gives nan)
    penalty_exp = 60 #exponent to use in the penalty function
remove_prev = True #Use with caution, be careful for other files in working directory! Removes (non-user specified) files that might have remained from previous optimisations. See manual for a list
optim_method = 'tnc' #bfgs or tnc, the chosen optimisation algorithm
if optim_method == 'tnc':
    maxnr_of_restarts = 1 #The number of times to restart the optimisation if the cost function is not as low as specified in stopcrit. Only implemented for tnc method at the moment. 
    if maxnr_of_restarts > 0:
        stopcrit = 40.0#If the cost function is equal or lower than this value, no restart will be attempted   
elif optim_method == 'bfgs':
    gtol = 1e-05 # A parameter for the bfgs algorithm. From scipy documentation: 'Gradient norm must be less than gtol before successful termination.'
if estimate_model_err or use_ensemble:
    run_multicore = True #Run part of the code on multiple cores simultaneously
    if run_multicore:
        max_num_cores = 'all' #'all' to use all available cores, otherwise specify an integer (without quotation marks)
    set_seed = True #Set the seed in case the output should be reproducable
    if set_seed:
        seedvalue = 18 #the chosen value of the seed. No floating point numbers and no negative numbers 
discard_nan_minims = False #if False, if in a minimisation nan is encountered, it will use the state from the best simulation so far, if True, the minimisation will result in a state with nans    
use_mean = False #switch for using mean of obs over several days
use_weights = True #weights for the cost function, to enlarge or reduce the importance of certain obs
if use_weights:
    weight_morninghrs = 1 #to change weights of obs in the morning (the hour at which the morning ends is specified in variable 'end_morninghrs'), when everything less well mixed. 1 means equal weights
    end_morninghrs = 10 #At all times smaller than this time (UTC, decimal hour), weight_morninghrs is applied
if (use_backgr_in_cost and use_weights):
    obs_vs_backgr_weight = 1.0 # a scaling factor for the importance of all the observations in the cost function
if write_to_f:
    wr_obj_to_pickle_files = True #write certain variables to files for possible postprocessing later
    figformat = 'eps' #the format in which you want figure output, e.g. 'png'
plotfontsize = 12 #plot font size, except for legend
legendsize = plotfontsize - 1
######################################
###### end user input: settings ######
######################################
plt.rc('font', size=plotfontsize)     
#some input checks
if use_ensemble:
    if (nr_of_members < 2 or type(nr_of_members) != int):
        raise Exception('When use_ensemble is True, nr_of_members should be an integer and > 1')
if type(figformat) != str:
    raise Exception('figformat should be of type str, e.g. \'png\'')
if use_ensemble or estimate_model_err:
    if run_multicore:
        if not (max_num_cores == 'all' or type(max_num_cores) == int):
            raise Exception('Invalid input for max_num_cores')
        elif type(max_num_cores) == int:
            if max_num_cores < 2:
                raise Exception('max_num_cores should be larger or equal than 2')

if write_to_f:
    if wr_obj_to_pickle_files:
        vars_to_pickle = ['priormodel','priorinput','obsvarlist','disp_units','disp_units_par','optim','obs_times','measurement_error','display_names','optimalinput','optimalinput_onsp','optimalmodel','optimalmodel_onsp'] #list of strings      
        for item in vars_to_pickle:
            if item in vars(): #check if variable exists, if so, delete so we do not write anything from a previous script/run to files
                del(vars()[item])
storefolder_objects = 'pickle_objects' #the folder where to store pickled variables when wr_obj_to_pickle_files == True
#remove previous files
if remove_prev:
    filelist_list = []
    filelist_list += [glob.glob('Optimfile*')] #add Everything starting with 'Optimfile' to the list
    filelist_list += [glob.glob('Gradfile*')]
    filelist_list += [glob.glob('Optstatsfile.txt')]
    filelist_list += [glob.glob('Modelerrorfile.txt')]
    filelist_list += [glob.glob('pdf_posterior*')]
    filelist_list += [glob.glob('pdf_nonstate_*')]
    filelist_list += [glob.glob('fig_fit*')]
    filelist_list += [glob.glob('fig_obs*')]
    filelist_list += [glob.glob('pp_*')]
    filelist_list += [glob.glob(storefolder_objects+'/*')]
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
#data_Temp.loc[data_Temp['TA200'] == -9999 ,'TA200'] = np.nan #set -9999 to nan
for column in data_Temp.drop(columns=['day','btime','etime']): #We leave these three columns out using the drop function, otherwise they would be converted to floats. They remain in the dataframe itself using this statement.
    data_Temp.loc[data_Temp[column] == -9999 ,column] = np.nan
data_Press = pd.read_csv(directory+'/'+'caboper_surface_pressure_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
for column in data_Press.drop(columns=['day','btime','etime']):
    data_Press.loc[data_Press[column] == -9999 ,column] = np.nan
data_DewP = pd.read_csv(directory+'/'+'caboper_dew_point_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
for column in data_DewP.drop(columns=['day','btime','etime']):
    data_DewP.loc[data_DewP[column] == -9999 ,column] = np.nan
data_Flux = pd.read_csv(directory+'/'+'cabsurf_surface_flux_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
for column in data_Flux.drop(columns=['day','btime','etime']):
    data_Flux.loc[data_Flux[column] == -9999 ,column] = np.nan
data_Rad = pd.read_csv(directory+'/'+'caboper_radiation_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True)
for column in data_Rad.drop(columns=['day','btime','etime']):
    data_Rad.loc[data_Rad[column] == -9999 ,column] = np.nan
data_Windsp = pd.read_csv(directory+'/'+'caboper_wind_speed_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True) 
for column in data_Windsp.drop(columns=['day','btime','etime']):
    data_Windsp.loc[data_Windsp[column] == -9999 ,column] = np.nan
data_Winddir = pd.read_csv(directory+'/'+'caboper_wind_direction_200309-24-25-26.lot', skiprows=[0,1,3],delim_whitespace=True) 
for column in data_Winddir.drop(columns=['day','btime','etime']):
    data_Winddir.loc[data_Winddir[column] == -9999 ,column] = np.nan
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
    stdevWindsp200_selected = data_Windsp[timeselection]['SF200'] #m s-1 #standard dev 200m, see Cabauw_TR.pdf
    stdevWindsp10_selected = data_Windsp[timeselection]['SF010'] #m s-1
    Winddir200_selected =  data_Winddir[timeselection]['D200'] #deg
    Winddir10_selected =  data_Winddir[timeselection]['D010'] #deg
    stdevWinddir200_selected = data_Winddir[timeselection]['SD200'] #deg  #see Cabauw_TR.pdf
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
priormodinput.sw_shearwe = True     # shear growth mixed-layer switch
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
priormodinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
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
priormodinput.wsat       = 0.6     # saturated volumetric water content ECMWF config [-]
priormodinput.wfc        = 0.491     # volumetric water content field capacity [-]
priormodinput.wwilt      = 0.314     # volumetric water content wilting point [-]
priormodinput.C1sat      = 0.342     
priormodinput.C2ref      = 0.3
priormodinput.LAI        = 2.        # leaf area index [-]
priormodinput.gD         = None       # correction factor transpiration for VPD [-]
priormodinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]
priormodinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priormodinput.alpha      = 0.25      # surface albedo [-]
priormodinput.Ts         = 284      # initial surface temperature [K]
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
priormodinput.R10 = 0.23
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
#priormodinput.wCO2       = priormodinput.wCO2_input[0]        # surface total CO2 flux [mgCO2 m-2 s-1]
#priormodinput.wq         = priormodinput.wq_input[0]     # surface kinematic moisture flux [kg kg-1 m s-1]
#priormodinput.wtheta     = priormodinput.wtheta_input[0]       # surface kinematic heat flux [K m s-1]
priormodinput.wCO2       = 0.0000001 # surface total CO2 flux [mgCO2 m-2 s-1]
priormodinput.wq       = 0.0000001 # surface kinematic moisture flux [kg kg-1 m s-1]
priormodinput.wtheta       = 0.0000001 # surface kinematic heat flux [K m s-1]

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
priorinput = cp.deepcopy(priormodinput) 

##########################################################################
###### user input: state, list of used obs and non-model priorinput ######
##########################################################################
#e.g. state=['h','q','theta','gammatheta','deltatheta','deltaq','alpha','gammaq','wg','wtheta','z0h','z0m','ustar','wq','divU']
state=['advtheta','advq','advCO2','deltatheta','gammatheta','deltaq','gammaq','deltaCO2','gammaCO2','alfa_sto','alpha','EnBalDiffObsHFrac','wg','R10']
obsvarlist =['Tmh','Tmh2','Tmh3','Tmh4','Tmh5','Tmh6','Tmh7','qmh','qmh2','qmh3','qmh4','qmh5','qmh6','qmh7','CO2mh','CO2mh2','CO2mh3','CO2mh4','h','LE','H','wCO2','Swout']
#below we can add some input necessary for the state in the optimisation, that is not part of the model input (a scale for some of the observations in the costfunction if desired). Or EnBalDiffObsHFrac
if 'EnBalDiffObsHFrac' in state:
    priorinput.EnBalDiffObsHFrac = 0.6
#priorinput.obs_sca_cf_H = 1.0 #Always use this format, 'obs_sca_cf_' plus the observation type you want to scale
##e.g. priorinput.obs_sca_cf_H = 1.5 means that in the cost function all obs of H will be multiplied with 1.5. But only if obs_sca_cf_H also in the state!
#priorinput.obs_sca_cf_LE = 1.0

##############################################################################
###### end user input: state, list of used obs and non-model priorinput ######
##############################################################################
if len(set(state)) != len(state):
    raise Exception('Mulitiple occurences of same item in state')
if len(set(obsvarlist)) != len(obsvarlist):
    raise Exception('Mulitiple occurences of same item in obsvarlist')
if ('EnBalDiffObsHFrac' in state and ('obs_sca_cf_H' in state or 'obs_sca_cf_LE' in state)):
    raise Exception('When EnBalDiffObsHFrac in state, you cannot include obs_sca_cf_H or obs_sca_cf_LE in state as well')
for item in state:
    if item.startswith('obs_sca_cf_'):
        obsname = item.split("obs_sca_cf_",1)[1]
        if obsname not in obsvarlist:
            raise Exception(item+' given in state, but '+obsname+' not given in obsvarlist')
    if item not in priorinput.__dict__ or priorinput.__dict__[item] is None:
        raise Exception(item +' given in state, but no prior given!')
for item in priorinput.__dict__: #just a check
    if item.startswith('obs_sca_cf') and (item not in state):
        raise Exception(item +' given in priorinput, but not part of state. Remove from priorinput or add '+item+' to the state')
    elif item == 'EnBalDiffObsHFrac' and (item not in state):
        raise Exception(item +' given in priorinput, but not part of state. Remove from priorinput or add '+item+' to the state')
for item in obsvarlist:
    if not hasattr(priormodel.out,item):
        raise Exception(item +' from obsvarlist is not a model variable occurring in class \'model_output\' in forwardmodel.py')
if len(state) < 1:
    raise Exception('Variable \'state\' is empty')
if len(obsvarlist) < 1:
    raise Exception('Variable \'obsvarlist\' is empty')
       
if use_backgr_in_cost or use_ensemble:
    priorvar = {}
    priorcovar={}
###########################################################
###### user input: prior variance/covar (if used) #########
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
    priorvar['wg'] = (0.2)**2
#    priorvar['obs_sca_cf_H'] = 0.4**2 
#    priorvar['obs_sca_cf_LE'] = 0.4**2 
    priorvar['EnBalDiffObsHFrac'] = 0.4**2 
    priorvar['cc'] = 0.2**2 
    priorvar['R10'] = 0.4**2 
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
    #below we can specify covariances as well, for the background information matrix. If covariances are not specified, they are taken as 0
    #e.g. priorcovar['gammatheta,gammaq'] = 5.
###########################################################
###### end user input: prior variance/covar (if used) #####
###########################################################                  
    for thing in priorvar:
        if thing not in priorinput.__dict__:
            raise Exception('Parameter \''+thing +'\' specified in priorvar, but does not exist in priorinput')
        if priorvar[thing] <= 0:
            raise Exception('Prior variance for '+thing+' should be greater than zero!')
    b_cov = np.diag(np.zeros(len(state)))
    i = 0
    for item in state:
        if item not in priorvar:
            raise Exception('No prior variance specified for '+item)
        b_cov[i][i] = priorvar[item] #b_cov stands for background covariance matrix, b already exists as model parameter
        i += 1
    #in b_cov, params should have same order as in state
    if bool(priorcovar):# check if covar matr not empty
        for thing in priorcovar:
            if thing.count(',') != 1:
                raise Exception('Invalid key \''+thing+'\' in priorcovar')
            if ''.join(thing.split()) != thing:
                raise Exception('Invalid key \''+thing+'\' in priorcovar')
            thing1,thing2 = thing.split(',')
            if thing1 not in priorinput.__dict__:
                raise Exception('Parameter \''+thing1 +'\' specified in priorcovar, but does not exist in priorinput')
            elif thing2 not in priorinput.__dict__:
                raise Exception('Parameter \''+thing2 +'\' specified in priorcovar, but does not exist in priorinput')
            if priorcovar[thing] > 1 * np.sqrt(priorvar[thing1])*np.sqrt(priorvar[thing2]) or priorcovar[thing] < -1 * np.sqrt(priorvar[thing1])*np.sqrt(priorvar[thing2]):
                raise Exception('Prior covariance of '+thing + ' inconsistent with specified variances (deduced correlation not in [-1,1]).')
        for i in range(len(state)):
            item = state[i]
            for item2 in np.delete(state,i): #exclude item2 == item, that is for variance, not covar
                if item+','+item2 in priorcovar:
                    b_cov[i][state.index(item2)] = priorcovar[item+','+item2] 
                    b_cov[state.index(item2)][i] = priorcovar[item+','+item2]
                elif item2+','+item in priorcovar:
                    b_cov[i][state.index(item2)] = priorcovar[item2+','+item] 
                    b_cov[state.index(item2)][i] = priorcovar[item2+','+item]
    if not np.all(np.linalg.eigvals(b_cov) > 0):
        raise Exception('Prior error covariance matrix is not positive definite, check the specified elements')#See page 12 and 13 of Brasseur and Jacob 2017                              
else:
     b_cov = None 

boundedvars = {}
if imposeparambounds or paramboundspenalty:
#############################################################
###### user input: parameter bounds #########################
#############################################################
    boundedvars['deltatheta'] = [0.2,7] #lower and upper bound
    boundedvars['deltaCO2'] = [-200,200]
    boundedvars['deltaq'] = [-0.009,0.009]
    boundedvars['alpha'] = [0.05,0.8] 
    boundedvars['alfa_sto'] = [0.1,5]
    boundedvars['wg'] = [priorinput.wwilt+0.001,priorinput.wsat-0.001]
    boundedvars['theta'] = [274,310]
    boundedvars['h'] = [50,3200]
    boundedvars['wtheta'] = [0.05,0.6]
    boundedvars['gammatheta'] = [0.0015,0.018]
    boundedvars['gammatheta2'] = [0.0015,0.018]
    boundedvars['gammaq'] = [-9e-6,9e-6]
    boundedvars['z0m'] = [0.0001,5]
    boundedvars['z0h'] = [0.0001,5]
    boundedvars['q'] = [0.002,0.020]
    boundedvars['divU'] = [0,1e-4]
    boundedvars['fCA'] = [0.1,1e8]
    boundedvars['CO2'] = [100,1000]
    boundedvars['ustar'] = [0.01,50]
    boundedvars['wq'] = [0,0.1] #negative flux seems problematic because L going to very small values
    boundedvars['EnBalDiffObsHFrac'] = [0,1]
    boundedvars['cc'] = [0,1]
    boundedvars['R10'] = [0,15]
#############################################################
###### end user input: parameter bounds  ####################
#############################################################    
    for param in boundedvars:    
        if not hasattr(priorinput,param):
            raise Exception('Parameter \''+ param + '\' in boundedvars does not occur in priorinput')

#create inverse modelling framework, do check,...
optim = im.inverse_modelling(priormodel,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,StateVarNames=state,obsvarlist=obsvarlist,
                             pri_err_cov_matr=b_cov,paramboundspenalty=paramboundspenalty,boundedvars=boundedvars)
Hx_prior = {}
for item in obsvarlist:
    Hx_prior[item] = priormodel.out.__dict__[item]


#The observations
obs_times = {}
obs_weights = {}
disp_units = {}
display_names = {}
measurement_error = {}
for item in obsvarlist:
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
            measurement_error[item] = np.ndarray.flatten(stdevTemp200_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
            display_names[item] = 'T_200'
        if item == 'Tmh2':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp140_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp140_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        if item == 'Tmh3':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp80_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp80_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        if item == 'Tmh4':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp40_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp40_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        if item == 'Tmh5':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp20_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp20_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        if item == 'Tmh6':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp10_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp10_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        if item == 'Tmh7':
            optim.__dict__['obs_'+item] = np.ndarray.flatten(Temp2_mean) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = np.ndarray.flatten(stdevTemp2_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'K'
        elif item == 'CO2mh':
            optim.__dict__['obs_'+item] = np.array(CO2_200_mean)
            measurement_error[item] = stdevCO2_200_hourly
            obs_times[item] = hours_mean * 3600. #same as for temperature
            disp_units[item] = 'ppm'
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'CO2mh2':
            optim.__dict__['obs_'+item] = np.array(CO2_120_mean)
            measurement_error[item] = stdevCO2_120_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
        elif item == 'CO2mh3':
            optim.__dict__['obs_'+item] = np.array(CO2_60_mean)
            measurement_error[item] = stdevCO2_60_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
        elif item == 'CO2mh4':
            optim.__dict__['obs_'+item] = np.array(CO2_20_mean)
            measurement_error[item] = stdevCO2_20_hourly
            obs_times[item] = hours_mean * 3600.
            if use_weights:
                obs_weights[item] = [1./4 for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
        elif item == 'wCO2':
            optim.__dict__['obs_'+item] = np.array(wCO2_mean)
            measurement_error[item] = stdevwCO2_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'mg CO2 m$^{-2}$s$^{-1}$'
        elif item == 'h':
            optim.__dict__['obs_'+item] = np.array(BLH_mean) #we just have one day
            print('WARNING: the obs of h are not really a mean...')
            measurement_error[item] = [100 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(dhour_BLH_mean) * 3600.
            for i in range(len(obs_times[item])):
                obs_times[item][i] = round(obs_times[item][i],0) #Very important, since otherwise the obs will be a fraction of a second off and will not be used
            disp_units[item] = 'm'
        elif item == 'qmh':
            optim.__dict__['obs_'+item] = np.array(q200_mean)
            measurement_error[item] = np.array(stdevq200_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh2':
            optim.__dict__['obs_'+item] = np.array(q140_mean)
            measurement_error[item] = np.array(stdevq140_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh3':
            optim.__dict__['obs_'+item] = np.array(q80_mean)
            measurement_error[item] = np.array(stdevq80_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh4':
            optim.__dict__['obs_'+item] = np.array(q40_mean)
            measurement_error[item] = np.array(stdevq40_hourly)
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh5':
            optim.__dict__['obs_'+item] = np.array(q20_mean)
            measurement_error[item] = stdevq20_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh6':
            optim.__dict__['obs_'+item] = q10_mean
            measurement_error[item] = stdevq10_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'qmh7':
            optim.__dict__['obs_'+item] = q2_mean
            measurement_error[item] = stdevq2_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'g kg$^{-1}$'
        elif item == 'ustar':
            optim.__dict__['obs_'+item] = ustar_mean
            measurement_error[item] = stdevustar_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'm s$^{-1}$'
        elif item == 'H':
            optim.__dict__['obs_'+item] = H_mean
            measurement_error[item] = stdevH_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'W m$^{-2}$'
        elif item == 'LE':
            optim.__dict__['obs_'+item] = LE_mean
            measurement_error[item] = stdevLE_hourly
            obs_times[item] = hours_mean * 3600.
            disp_units[item] = 'W m$^{-2}$'
    else:
        refnumobs = np.sum(~np.isnan(np.array(BLH_selected))) #only for the weights, a reference number of the number of non-nan obs, chosen to be based on h here. When e.g. wco2 has more obs than this number,
        #it will be given a lower weight per individual observation than h
        obstimes_T = []
        for i in range(starthour,endhour):
            for minute in selectedminutes:
                obstimes_T.append(i * 3600. + minute * 60.)
        if item == 'Tmh':
            optim.__dict__['obs_'+item] = np.array(Temp200_selected) #np.array since it is a pandas dataframe
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_200'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
                # np.sum(~np.isnan(optim.__dict__['obs_'+item])) used here instead of len(optim.__dict__['obs_'+item]), since nan data should not count for the length of the observation array. ~ inverts the np.isnan array. 
        if item == 'Tmh2':
            optim.__dict__['obs_'+item] = np.array(Temp140_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_140'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh3':
            optim.__dict__['obs_'+item] = np.array(Temp80_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_80'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh4':
            optim.__dict__['obs_'+item] = np.array(Temp40_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_40'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh5':
            optim.__dict__['obs_'+item] = np.array(Temp20_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_20'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh6':
            optim.__dict__['obs_'+item] = np.array(Temp10_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_10'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        if item == 'Tmh7':
            optim.__dict__['obs_'+item] = np.array(Temp2_selected) #flatten becuase it is from the second datafile, which has a different structure
            measurement_error[item] = [0.1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'K'
            display_names[item] = 'T_2'
            if use_weights:
                obs_weights[item] = [1./7*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'CO2mh':
            optim.__dict__['obs_'+item] = np.array(CO2_200_selected)
            measurement_error[item] = [1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
            display_names[item] = 'CO2_200'
        elif item == 'CO2mh2':
            optim.__dict__['obs_'+item] = np.array(CO2_120_selected)
            measurement_error[item] = [1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
            display_names[item] = 'CO2_120'
        elif item == 'CO2mh3':
            optim.__dict__['obs_'+item] = np.array(CO2_60_selected)
            measurement_error[item] = [1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
            display_names[item] = 'CO2_60'
        elif item == 'CO2mh4':
            optim.__dict__['obs_'+item] = np.array(CO2_20_selected)
            measurement_error[item] = [1 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = hour_CO2_selected * 3600
            if use_weights:
                obs_weights[item] = [1./4*refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
            disp_units[item] = 'ppm'
            display_names[item] = 'CO2_20'
        elif item == 'wCO2':
            optim.__dict__['obs_'+item] = np.array(wCO2_selected)
            measurement_error[item] = [0.08 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'mg CO2 m$^{-2}$s$^{-1}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'h':
            optim.__dict__['obs_'+item] = np.array(BLH_selected) #we just have one day
            measurement_error[item] = [120 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(dhour_BLH_selected) * 3600
            for i in range(len(obs_times[item])):
                obs_times[item][i] = round(obs_times[item][i],0) #Very important, since otherwise the obs will be a fraction of a second off and will not be used
            disp_units[item] = 'm'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh':
            optim.__dict__['obs_'+item] = np.array(q200_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_200'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh2':
            optim.__dict__['obs_'+item] = np.array(q140_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_140'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh3':
            optim.__dict__['obs_'+item] = np.array(q80_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_80'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh4':
            optim.__dict__['obs_'+item] = np.array(q40_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_40'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh5':
            optim.__dict__['obs_'+item] = np.array(q20_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_20'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh6':
            optim.__dict__['obs_'+item] = np.array(q10_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_10'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'qmh7':
            optim.__dict__['obs_'+item] = np.array(q2_selected)
            measurement_error[item] = [0.00008 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'g kg$^{-1}$'
            display_names[item] = 'q_2'
            if use_weights:
                obs_weights[item] = [refnumobs*1./7*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'ustar':
            optim.__dict__['obs_'+item] = np.array(ustar_selected)
            measurement_error[item] = [0.15 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'm s$^{-1}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'H':
            optim.__dict__['obs_'+item] = np.array(H_selected)
            measurement_error[item] = [15 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'W m$^{-2}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'LE':
            optim.__dict__['obs_'+item] = np.array(LE_selected)
            measurement_error[item] = [15 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'W m$^{-2}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'u':
            optim.__dict__['obs_'+item] = np.array(u200_selected)
            measurement_error[item] = np.array(stdevWindsp200_selected) #not fully correct, includes variation in v...
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'm s$^{-1}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'v':
            optim.__dict__['obs_'+item] = np.array(v200_selected)
            measurement_error[item] = np.array(stdevWindsp200_selected)
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'm s$^{-1}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
        elif item == 'Swout':
            optim.__dict__['obs_'+item] = np.array(SWU_selected)
            measurement_error[item] = [2.5 for j in range(len(optim.__dict__['obs_'+item]))]#we don't have info on this
            obs_times[item] = np.array(obstimes_T)
            disp_units[item] = 'W m$^{-2}$'
            if use_weights:
                obs_weights[item] = [refnumobs*1/np.sum(~np.isnan(optim.__dict__['obs_'+item])) for j in range(len(optim.__dict__['obs_'+item]))]
###########################################################
###### end user input: observation information ############
###########################################################
if use_ensemble:
    if est_post_pdf_covmatr:
        disp_units_par = {}         
##############################################################################
###### user input: units of parameters for pdf figures (optional) ############
##############################################################################
        disp_units_par['theta'] = 'K'
        disp_units_par['advtheta'] = 'Ks$^{-1}$'
        disp_units_par['advq'] = 'kg kg$^{-1}$s$^{-1}$'
        disp_units_par['advCO2'] = 'ppm s$^{-1}$'
        disp_units_par['deltatheta'] = 'K'
        disp_units_par['gammatheta'] = 'K m$^{-1}$'
        disp_units_par['deltaq'] = 'kg kg$^{-1}$'
        disp_units_par['gammaq'] = 'kg kg$^{-1}$m$^{-1}$'
        disp_units_par['deltaCO2'] = 'ppm'
        disp_units_par['gammaCO2'] = 'ppm m$^{-1}$'
        disp_units_par['alfa_sto'] = '-'
        disp_units_par['alpha'] = '-'
        disp_units_par['EnBalDiffObsHFrac'] = '-'
        disp_units_par['wg'] = '-'
        disp_units_par['cc'] = '-'
        disp_units_par['R10'] = 'mg CO2 m$^{-2}$s$^{-1}$'
    
##############################################################################
###### end user input: units of parameters for pdf figures (optional) ########
##############################################################################                
if 'EnBalDiffObsHFrac' in state:             
##################################################################
###### user input: energy balance information (if used) ##########
##################################################################
    #If H in obsvarlist, specify optim.EnBalDiffObs_atHtimes; If LE in obsvarlist, specify optim.EnBalDiffObs_atLEtimes. optim.EnBalDiffObs_atHtimes is the energy balance gap at the observation times of H
    optim.EnBalDiffObs_atHtimes = np.array((SWD_selected + LWD_selected - SWU_selected - LWU_selected) - (H_selected + LE_selected + G_selected))
    optim.EnBalDiffObs_atLEtimes = np.array((SWD_selected + LWD_selected - SWU_selected - LWU_selected) - (H_selected + LE_selected + G_selected))
        
##################################################################
###### end user input: energy balance information (if used) ######
##################################################################
    
    for item in ['H','LE']:
        if item in obsvarlist:
            if len(optim.__dict__['EnBalDiffObs_at'+item+'times']) != len(optim.__dict__['obs_'+item]):
                raise Exception('When including EnBalDiffObsHFrac in state and '+ item + ' in obsvarlist, an EnBalDiffObs_at' +item+'times value should correspond to every obs of ' + item)
            if type(optim.__dict__['EnBalDiffObs_at'+item+'times']) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert EnBalDiffObs_at'+item+'times data into type \'numpy.ndarray\' or list!')
            #below some checks that happen later for the other observations, but we need to do the ones below here already since we use them earlier
            if (not hasattr(optim,'obs_'+item) or item not in measurement_error): #a check to see wether all info is specified
                raise Exception('Incomplete or no information on obs of ' + item)
            if item not in obs_times:
                raise Exception('Please specify the observation times of '+item+'.')
            if type(measurement_error[item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert measurement_error data of '+item+' into type \'numpy.ndarray\' or list!')
            if type(optim.__dict__['obs_'+item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert observation data of '+item+' into type \'numpy.ndarray\' or list!')
            if type(obs_times[item]) not in [np.ndarray,list]:
                raise Exception('Please convert observation time data of '+item+' into type \'numpy.ndarray\' or list!')
            if use_weights and item in obs_weights:
                if type(obs_weights[item]) not in [np.ndarray,list]:
                    raise Exception('Please convert observation weight data of '+item+' into type \'numpy.ndarray\' or list!')
            
    for item in ['H','LE']:
        if item in obsvarlist:
            if not hasattr(optim,'EnBalDiffObs_at'+item+'times'):
                raise Exception('When including EnBalDiffObsHFrac in state and '+ item + ' in obsvarlist, \'optim.EnBalDiffObs_at'+item+'times\' should be specified!')
            itoremove = []
            for i in range(len(optim.__dict__['EnBalDiffObs_at'+item+'times'])):
                if np.isnan(optim.__dict__['EnBalDiffObs_at'+item+'times'][i]):
                    itoremove += [i]
            optim.__dict__['EnBalDiffObs_at'+item+'times'] = np.delete(optim.__dict__['EnBalDiffObs_at'+item+'times'],itoremove) #exclude the nan obs. 
            optim.__dict__['obs_'+item] = np.delete(optim.__dict__['obs_'+item],itoremove) #later in this script these (and other) obs are checked for nan
            #Also EnBalDiffObs_atHtimes and EnBalDiffObs_atLEtimes are checked again later in this script, then for nans in the obs of H and LE respectively . So if a nan occurs in LE, the EnBalDiffObs_atLEtimes value
            #at the time of the nan in LE will be discarded as well.
            measurement_error[item] = np.delete(measurement_error[item],itoremove)
            obs_times[item] = np.delete(obs_times[item],itoremove)
            if item in obs_weights:
                obs_weights[item] = np.delete(obs_weights[item],itoremove)
    
mod_error = {} #model error
repr_error = {} #representation error, see eq 11.11 in chapter inverse modelling Brasseur and Jacob 2017
if estimate_model_err:  
    me_paramdict = {} #dictionary of dictionaries, me means model error
########################################################################
###### user input: model and representation error ######################
########################################################################
    #in case the model error is estimated with a model ensemble (switch estimate_model_err), specify here the parameters to perturb for this estimation:
    me_paramdict['cveg'] = {'distr':'uniform','leftbound': 0.1,'rightbound': 1.0}
    me_paramdict['Lambda'] = {'distr':'normal','scale': 0.3}
else:
    pass
    #in case you want to specify directly the model errors (estimate_model_err = False), specify them here:
    #e.g. mod_error['theta'] = [0.5 for j in range(len(measurement_error['theta']))]
    mod_error['Tmh'],mod_error['Tmh2'],mod_error['Tmh3'],mod_error['Tmh4'],mod_error['Tmh5'],mod_error['Tmh6'],mod_error['Tmh7'] = [np.zeros(len(measurement_error['Tmh'])) for x in range(7)]
    for j in range(len(measurement_error['Tmh'])):
        if obs_times['Tmh'][j]/3600 > 10.5:
            mod_error['Tmh'][j] = 0.15
            mod_error['Tmh2'][j] = 0.15
            mod_error['Tmh3'][j] = 0.15
            mod_error['Tmh4'][j] = 0.15
            mod_error['Tmh5'][j] = 0.15
            mod_error['Tmh6'][j] = 0.15
            mod_error['Tmh7'][j] = 0.15
        else:
            mod_error['Tmh'][j] = 0.3
            mod_error['Tmh2'][j] = 0.3
            mod_error['Tmh3'][j] = 0.3
            mod_error['Tmh4'][j] = 0.3
            mod_error['Tmh5'][j] = 0.3
            mod_error['Tmh6'][j] = 0.3
            mod_error['Tmh7'][j] = 0.3
    mod_error['qmh'],mod_error['qmh2'],mod_error['qmh3'],mod_error['qmh4'],mod_error['qmh5'],mod_error['qmh6'],mod_error['qmh7'] = [np.zeros(len(measurement_error['qmh'])) for x in range(7)]
    for j in range(len(measurement_error['qmh'])):
        if obs_times['qmh'][j]/3600 > 10.5:
            mod_error['qmh'][j] = 0.00015
            mod_error['qmh2'][j] = 0.00015
            mod_error['qmh3'][j] = 0.00015
            mod_error['qmh4'][j] = 0.00015
            mod_error['qmh5'][j] = 0.00015
            mod_error['qmh6'][j] = 0.00015
            mod_error['qmh7'][j] = 0.00015
        else:
            mod_error['qmh'][j] = 0.0002
            mod_error['qmh2'][j] = 0.0002
            mod_error['qmh3'][j] = 0.0002
            mod_error['qmh4'][j] = 0.0002
            mod_error['qmh5'][j] = 0.0002
            mod_error['qmh6'][j] = 0.0002
            mod_error['qmh7'][j] = 0.0002
    mod_error['CO2mh'],mod_error['CO2mh2'],mod_error['CO2mh3'],mod_error['CO2mh4'] = [np.zeros(len(measurement_error['CO2mh'])) for x in range(4)]
    for j in range(len(measurement_error['CO2mh'])):
        if obs_times['CO2mh'][j]/3600 > 10.5:
            mod_error['CO2mh'][j] = 1
            mod_error['CO2mh2'][j] = 1
            mod_error['CO2mh3'][j] = 1
            mod_error['CO2mh4'][j] = 1
        else:
            mod_error['CO2mh'][j] = 2.5
            mod_error['CO2mh2'][j] = 2.5
            mod_error['CO2mh3'][j] = 2.5
            mod_error['CO2mh4'][j] = 2.5
    mod_error['h'] = [40 for j in range(len(measurement_error['h']))]
    mod_error['H'] = [np.abs(0.10*optim.obs_H[j]) for j in range(len(measurement_error['H']))]
    mod_error['LE'] = [np.abs(0.10*optim.obs_LE[j]) for j in range(len(measurement_error['LE']))]
    mod_error['wCO2'] = [np.abs(0.30*optim.obs_wCO2[j]) for j in range(len(measurement_error['wCO2']))]
    mod_error['Swout'] = np.zeros(len(measurement_error['Swout']))
    for j in range(len(measurement_error['Swout'])):
        if obs_times['Swout'][j]/3600 > 10.5 and obs_times['Swout'][j]/3600 < 13:
            mod_error['Swout'][j] = 13
        else:
            mod_error['Swout'][j] = 3
#specify the representation error here, if nothing specified it is assumed 0
#e.g. repr_error['theta'] = [0.3 for j in range(len(measurement_error['theta']))]
########################################################################
###### end user input: model and representation error ##################
######################################################################## 
if estimate_model_err:
    for param in cp.deepcopy(me_paramdict): #deepcopy to prevent 'dictionary changed size during iteration'
        if param in state:
            del me_paramdict[param] #delete parameters that are in state as well, they should not be used for estimating the model error
    if not bool(me_paramdict): #checks wether dictionary empty
        raise Exception('When estimate_model_err == True, include at least one parameter (that is not included in the state) in me_paramdict')
if use_ensemble:
    non_state_paramdict = {}
    if pert_non_state_param:          
################################################################################
###### user input: non-state parameters to perturb in ensemble (if used) #######
################################################################################
        #specify here which non-state params to perturb in the ensemble:
        #e.g. non_state_paramdict['cveg'] = {'distr':'uniform','leftbound': 0.1,'rightbound': 1.0}
        #or use parameter dictionary from the estimation of the model error (if estimate_model_err):
        non_state_paramdict = me_paramdict
############################################################################################################
###### end user input: non-state parameters to perturb in ensemble (if used) ###############################
###### User input ends here until the end of the file, where additional plotting etc can be done if desired#
############################################################################################################
        for param in cp.deepcopy(non_state_paramdict):    
            if not hasattr(priorinput,param):
                raise Exception('Parameter \''+ param + '\' in non_state_paramdict does not occur in priorinput')
            if param in state:
                del non_state_paramdict[param] #delete parameters that are both in state and in this dictionary, they should not be perturbed additionally
        if not bool(non_state_paramdict): #checks wether dictionary empty
            raise Exception('When pert_non_state_param == True, include at least one parameter (that is not included in the state) in non_state_paramdict')

for item in obsvarlist:
    if (not hasattr(optim,'obs_'+item) or item not in measurement_error): #a check to see wether all info is specified
        raise Exception('Incomplete or no information on obs of ' + item)
    if item not in repr_error:
        repr_error[item] = np.zeros(len(measurement_error[item]))
    if item not in obs_times:
        raise Exception('Please specify the observation times of '+item+'.')
    if type(measurement_error[item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
        raise Exception('Please convert measurement_error data of '+item+' into type \'numpy.ndarray\' or list!')
    if type(repr_error[item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
        raise Exception('Please convert repr_error data of '+item+' into type \'numpy.ndarray\' or list!')
    if type(optim.__dict__['obs_'+item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
        raise Exception('Please convert observation data of '+item+' into type \'numpy.ndarray\' or list!')
    if type(obs_times[item]) not in [np.ndarray,list]:
        raise Exception('Please convert observation time data of '+item+' into type \'numpy.ndarray\' or list!')
    if use_weights and item in obs_weights:
        if type(obs_weights[item]) not in [np.ndarray,list]:
            raise Exception('Please convert observation weight data of '+item+' into type \'numpy.ndarray\' or list!')
    if len(obs_times[item]) != len(optim.__dict__['obs_'+item]):
        raise Exception('Error: size of obs and obstimes inconsistent!') 
    if len(obs_times[item]) != len(measurement_error[item]):
        raise Exception('Error: size of measurement_error and obstimes inconsistent for '+item+'!')
    if len(obs_times[item]) != len(repr_error[item]):
        raise Exception('Error: size of repr_error and obstimes inconsistent for '+item+'!')
    itoremove = []
    for i in range(len(optim.__dict__['obs_'+item])):
        if np.isnan(optim.__dict__['obs_'+item][i]):
            itoremove += [i]
    optim.__dict__['obs_'+item] = np.delete(optim.__dict__['obs_'+item],itoremove) #exclude the nan obs
    measurement_error[item] = np.delete(measurement_error[item],itoremove) #as a side effect, this turns the array into an numpy.ndarray if not already the case (or gives error).
    if not estimate_model_err:
        if item in mod_error:
            if type(mod_error[item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert mod_error data of '+item+' into type \'numpy.ndarray\' or list!')
            mod_error[item] = np.delete(mod_error[item],itoremove)
    repr_error[item] = np.delete(repr_error[item],itoremove)
    obs_times[item] = np.delete(obs_times[item],itoremove)#exclude the times,errors and weights as well (of the nan obs)
    if item in obs_weights:
        obs_weights[item] = np.delete(obs_weights[item],itoremove)
    if 'EnBalDiffObsHFrac' in state and item in ['H','LE']: #This is in case of a nan in e.g. H that is not a nan in EnBalDiffObs_atHtimes yet!
        optim.__dict__['EnBalDiffObs_at'+item+'times'] = np.delete(optim.__dict__['EnBalDiffObs_at'+item+'times'],itoremove) #exclude the nan obs 
        
    if use_weights and item in obs_weights:
        if len(obs_times[item]) != len(obs_weights[item]):
            raise Exception('Error: size of weights and obstimes inconsistent for '+item+'!')
    if (use_backgr_in_cost and use_weights): #add weight of obs vs prior (identical for every obs) in the cost function
        if item in obs_weights: #if already a weight specified for the specific type of obs
            obs_weights[item] = [x * obs_vs_backgr_weight for x in obs_weights[item]]
        else:
            obs_weights[item] = [obs_vs_backgr_weight for x in range(len(optim.__dict__['obs_'+item]))] #nans are already excluded in the obs at this stage, so no problem with nan
    if use_weights:
        if item in obs_weights: #if already a weight specified for the specific type of obs
            for i in range(len(obs_times[item])):
                if obs_times[item][i] < end_morninghrs * 3600:
                    obs_weights[item][i] = obs_weights[item][i] * weight_morninghrs
        else:
            obs_weights[item] = np.ones(len(optim.__dict__['obs_'+item]))
            for i in range(len(obs_times[item])):
                if obs_times[item][i] < end_morninghrs * 3600:
                    obs_weights[item][i] = weight_morninghrs #nans are already excluded in the obs at this stage, so no problem with nan
    
    for num in obs_times[item]:
        if round(num, 10) not in [round(num2, 10) for num2 in priormodel.out.t * 3600]:
            raise Exception('Error: obs occuring at a time that is not modelled (' + str(item) +')')
    if item not in disp_units:
        disp_units[item] = ''
    if item not in display_names:
        display_names[item] = item
if use_ensemble:
    if est_post_pdf_covmatr:
        for item in state:
            if item not in disp_units_par:
                disp_units_par[item] = ''   
        if pert_non_state_param:
            for item in non_state_paramdict:
                if item not in disp_units_par:
                    disp_units_par[item] = ''

if estimate_model_err:
    for param in me_paramdict:    
        if not hasattr(priorinput,param):
            raise Exception('Parameter \''+ param + '\' in me_paramdict for estimating the model error does not occur in priorinput')            
    def run_mod_pert_par(counter,seed,modelinput,paramdict,obsvarlist,obstimes):
        modelinput_mem = cp.deepcopy(priorinput) 
        if seed != None:
            if use_ensemble:
                seed = seed + 3 * nr_of_members + counter#create a unique seed for every member and for anything in this file
            else:
                seed = seed + nr_of_members_moderr + counter#create a unique seed for every member and for anything in this file
        np.random.seed(seed) #VERY IMPORTANT! You have to explicitly set the seed (to None is ok), otherwise multicore implementation will use same random number for all ensemble members. 
        for param in paramdict:
            if paramdict[param]['distr'] == 'normal':
                rand_nr = np.random.normal(0,paramdict[param]['scale'])
            elif paramdict[param]['distr'] == 'bounded normal':
                counter_while_loop = 1
                rand_nr = np.random.normal(0,paramdict[param]['scale'])
                while (modelinput_mem.__dict__[param] + rand_nr < paramdict[param]['leftbound'] or modelinput_mem.__dict__[param] + rand_nr > paramdict[param]['rightbound']): #lower than lower bound or higher than upper bound
                    rand_nr = np.random.normal(0,paramdict[param]['scale'])
                    if counter_while_loop >= 100:
                        raise Exception('Problem for estimating model error: no parameter value within bounds obtained for an ensemble member for parameter '+param+' after '+str(counter_while_loop)+ ' attempts')
                    counter_while_loop += 1
            elif paramdict[param]['distr'] == 'uniform':
                rand_nr = np.random.uniform(paramdict[param]['leftbound'],paramdict[param]['rightbound'])
            elif paramdict[param]['distr'] == 'triangular':
                rand_nr = np.random.triangular(paramdict[param]['leftbound'],paramdict[param]['mode'],paramdict[param]['rightbound'])
            else:
                raise Exception('Problem for estimating model error: unknown distribtion for '+param)
            modelinput_mem.__dict__[param] += rand_nr
        model_mem = fwdm.model(modelinput_mem)
        model_mem.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=True,save_vars_indict=False)
        returndict = {}
        returndict['hasnans'] = False
        for item in obsvarlist:
            returndict[item] = []
            for t in range(len(model_mem.out.t)):
                if round(model_mem.out.t[t]*3600, 10) in [round(num2, 10) for num2 in obstimes[item]]:
                    returndict[item].append(model_mem.out.__dict__[item][t])
                    if np.isnan(model_mem.out.__dict__[item][t]):
                        returndict['hasnans'] = True
            if len(returndict[item]) < 1:
                raise Exception('No model output at the observation times of '+item)
        return returndict
    if not set_seed:
        seedvalue = None  
    print('Starting model error ensemble...')          
    if run_multicore:
        if max_num_cores == 'all':
            max_num_cores_ = -1
        else:
            max_num_cores_ = max_num_cores
        result_array = Parallel(n_jobs=max_num_cores_)(delayed(run_mod_pert_par)(i,seedvalue,priorinput,me_paramdict,obsvarlist,obs_times)  for i in range(0,nr_of_members_moderr)) #, prefer="threads" makes it work, but probably not multiprocess. None is for the seed
        #the above returns a list with one item for each member, this item itself is a dictionary
    else:
        result_array = np.zeros(nr_of_members_moderr,dtype=dict)
        for i in range(nr_of_members_moderr):
            result_array[i] =  run_mod_pert_par(i,seedvalue,priorinput,me_paramdict,obsvarlist,obs_times)
    for item in obsvarlist:
        mod_error[item] = np.zeros(len(obs_times[item]))
        for t in range(len(result_array[0][item])):
            seq = np.array([x[item][t] for x in result_array[0:]]) #in case of nan for the first member, the length does not change
            if np.sum(~np.isnan(seq)) < 2:
                raise Exception('Cannot estimate model error for '+item+' at t = '+str(obs_times[item][t]/3600)+ ' h, since less than 2 non-nan data points')
            mod_error[item][t] =  np.nanstd(seq,ddof = 1)
    if write_to_f:
        open('Modelerrorfile.txt','w').write('{0:>36s}'.format('nr of members in model err ensemble:'))
        open('Modelerrorfile.txt','a').write('{0:>50s}'.format('nr of non-nan members in model err ensemble:\n'))        
        nan_members = 0
        for member in result_array:
            if member['hasnans'] == True:
                nan_members += 1
        open('Modelerrorfile.txt','a').write('{0:>36s}'.format(str(nr_of_members_moderr)))
        open('Modelerrorfile.txt','a').write('{0:>49s}'.format(str(nr_of_members_moderr-nan_members)))
        open('Modelerrorfile.txt','a').write('\n')
        open('Modelerrorfile.txt','a').write('{0:>30s}'.format('Time-mean model errors on obs:\n'))
        open('Modelerrorfile.txt','a').write('     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Modelerrorfile.txt','a').write('\n     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(np.mean(mod_error[item]))))
        open('Modelerrorfile.txt','a').write('\n')
        open('Modelerrorfile.txt','a').write('{0:>31s}'.format('Median model errors on obs:\n'))
        open('Modelerrorfile.txt','a').write('     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Modelerrorfile.txt','a').write('\n     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(np.median(mod_error[item]))))
        open('Modelerrorfile.txt','a').write('\n')
        open('Modelerrorfile.txt','a').write('{0:>31s}'.format('Max model errors on obs:\n'))
        open('Modelerrorfile.txt','a').write('     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Modelerrorfile.txt','a').write('\n     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(np.max(mod_error[item]))))
        open('Modelerrorfile.txt','a').write('\n')
        open('Modelerrorfile.txt','a').write('{0:>31s}'.format('Min model errors on obs:\n'))
        open('Modelerrorfile.txt','a').write('     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Modelerrorfile.txt','a').write('\n     ')
        for item in obsvarlist:
            open('Modelerrorfile.txt','a').write('{0:>25s}'.format(str(np.min(mod_error[item]))))
        open('Modelerrorfile.txt','a').write('\n     ')
    print('Finished model error ensemble.') 
    
for item in obsvarlist: #some stuff involving mod_error
    if not estimate_model_err:
        if item not in mod_error:
            mod_error[item] = np.zeros(len(measurement_error[item]))
    if len(obs_times[item]) != len(mod_error[item]):
        raise Exception('Error: size of mod_error and obstimes inconsistent for '+item+'!')
    optim.__dict__['error_obs_' + item] = np.sqrt(np.array(measurement_error[item])**2 + np.array(mod_error[item])**2 + np.array(repr_error[item])**2)

print('total number of obs:')
number_of_obs = 0
for item in obsvarlist:
    number_of_obs += len(optim.__dict__['obs_'+item])
print(number_of_obs)
if use_weights:
    WeightsSums = {}
    tot_sum_of_weights = 0
    for item in obsvarlist: #if use_weights and no weight specified for a type of obs, the weights for the type of obs have been set to one before
        WeightsSums[item] = np.sum(obs_weights[item])#need sum, the weights are an array for every item
        tot_sum_of_weights += WeightsSums[item]
    print('total number of obs, corrected for weights:')
    print(tot_sum_of_weights)
print('number of params to optimise:')
number_of_params = len(state)
print(number_of_params)
########################################
obs_sca_cf = {}
optim.pstate = [] #initial state values, used also in background_costf in inverse_modelling.py
for item in state:
    optim.pstate.append(priorinput.__dict__[item])
    if item.startswith('obs_sca_cf_'):
        obsname = item.split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
        obs_sca_cf[obsname] = cp.deepcopy(priorinput.__dict__[item])
optim.pstate = np.array(optim.pstate)
inputcopy = cp.deepcopy(priorinput) #deepcopy!
params = tuple([inputcopy,state,obs_times,obs_weights])
if ana_deriv:
    optim.checkpoint = cp.deepcopy(priormodel.cpx) #needed, as first thing optimizer does is calculating the gradient (when using bfgs it seems)
    optim.checkpoint_init = cp.deepcopy(priormodel.cpx_init) #needed, as first thing optimizer does is calculating the gradient (when using bfgs it seems)
    for item in obsvarlist:
        if 'EnBalDiffObsHFrac' in state:
            if item not in ['H','LE']:
                observations_item = optim.__dict__['obs_'+item]
            elif item == 'H':
                observations_item = cp.deepcopy(optim.__dict__['obs_H']) + optim.pstate[state.index('EnBalDiffObsHFrac')] * optim.EnBalDiffObs_atHtimes
            elif item == 'LE':
                observations_item = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optim.pstate[state.index('EnBalDiffObsHFrac')]) * optim.EnBalDiffObs_atLEtimes  
        else:
            observations_item = optim.__dict__['obs_'+item]
        if item in obs_sca_cf:
            obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
        else:
            obs_scale = 1.0 
        weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
        k = 0 #counter for the observations (specific for each type of obs)
        for ti in range(priormodel.tsteps):
            if round(priormodel.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[item]]: #so if we are at a time where we have an obs        
                if item in obs_weights:
                    weight = obs_weights[item][k]
                forcing = weight * (Hx_prior[item][ti] - obs_scale * observations_item[k])/(optim.__dict__['error_obs_' + item][k]**2)
                optim.forcing[ti][item] = forcing
                k += 1
if paramboundspenalty:
    optim.setNanCostfOutBoundsTo0 = setNanCostfOutBoundsTo0
    optim.penalty_exp = penalty_exp
if optim_method == 'bfgs':
    try:
        if ana_deriv:
            minimisation = optimize.fmin_bfgs(optim.min_func,optim.pstate,fprime=optim.ana_deriv,args=params,gtol=gtol,full_output=True)
        else:
            minimisation = optimize.fmin_bfgs(optim.min_func,optim.pstate,fprime=optim.num_deriv,args=params,gtol=gtol,full_output=True)
        state_opt0 = minimisation[0]
        min_costf0 = minimisation[1]
    except (im.nan_incostfError):
        print('Minimisation aborted due to nan')
        if write_to_f:
            open('Optimfile.txt','a').write('\n')
            open('Optimfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
            open('Gradfile.txt','a').write('\n')
            open('Gradfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
        if (discard_nan_minims == False and len(optim.Statelist) > 0): #len(optim.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf0 = np.min(optim.Costflist)
                min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
                state_opt0 = optim.Statelist[min_costf_ind]
        else:
            state_opt0 = np.array([np.nan for x in range(len(state))])
            min_costf0 = np.nan
    except (im.static_costfError):
        print('Minimisation aborted as it proceeded too slow')
        if write_to_f:
            open('Optimfile.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
            open('Gradfile.txt','a').write('\nMinimisation aborted as it proceeded too slow')
        min_costf0 = np.min(optim.Costflist)
        min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
        state_opt0 = optim.Statelist[min_costf_ind]
        
elif optim_method == 'tnc':
    if imposeparambounds:
        bounds = []
        for i in range(len(state)):
            for key in boundedvars:
                if key == state[i]:
                    bounds.append((boundedvars[key][0],boundedvars[key][1]))
            if state[i] not in boundedvars:
                bounds.append((None,None)) #bounds need something
    else:
        bounds = [(None,None) for item in state]
    try:
        if ana_deriv:
            minimisation = optimize.fmin_tnc(optim.min_func,optim.pstate,fprime=optim.ana_deriv,args=params,bounds=bounds,maxfun=None)
        else:
            minimisation = optimize.fmin_tnc(optim.min_func,optim.pstate,fprime=optim.num_deriv,args=params,bounds=bounds,maxfun=None)
        state_opt0 = minimisation[0]
        min_costf0 = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights) #within cost_func, the values of the variables in inputcopy that are also state variables will be overwritten by the values of the variables in state_opt0
    except (im.nan_incostfError):
        print('Minimisation aborted due to nan')
        if write_to_f:
            open('Optimfile.txt','a').write('\n')
            open('Optimfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
            open('Gradfile.txt','a').write('\n')
            open('Gradfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
        if (discard_nan_minims == False and len(optim.Statelist) > 0): #len(optim.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
            min_costf0 = np.min(optim.Costflist)
            min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
            state_opt0 = optim.Statelist[min_costf_ind]
        else:
            state_opt0 = np.array([np.nan for x in range(len(state))])
            min_costf0 = np.nan
        optim.stop = True
    except (im.static_costfError):
        print('Minimisation aborted as it proceeded too slow')
        if write_to_f:
            open('Optimfile.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
            open('Gradfile.txt','a').write('\nMinimisation aborted as it proceeded too slow')
        min_costf0 = np.min(optim.Costflist)
        min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
        state_opt0 = optim.Statelist[min_costf_ind]
    if not hasattr(optim,'stop'):
        for i in range(maxnr_of_restarts): 
            if min_costf0 > stopcrit: #will evaluate to False if min_costf0 is equal to nan
                optim.nr_of_sim_bef_restart = optim.sim_nr
                if write_to_f:
                    open('Optimfile.txt','a').write('{0:>25s}'.format('\n restart'))
                    open('Gradfile.txt','a').write('{0:>25s}'.format('\n restart'))
                try:
                    if ana_deriv:
                        minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                    else:
                        minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                    state_opt0 = minimisation[0]
                except (im.nan_incostfError):
                    print('Minimisation aborted due to nan, no restart')
                    if write_to_f:
                        open('Optimfile.txt','a').write('\nnan reached, no restart')
                        open('Gradfile.txt','a').write('\nnan reached, no restart')
                    if discard_nan_minims == False:
                        min_costf0 = np.min(optim.Costflist)
                        min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
                        state_opt0 = optim.Statelist[min_costf_ind]
                    else:
                        state_opt0 = np.array([np.nan for x in range(len(state))])
                        min_costf0 = np.nan
                    break
                except (im.static_costfError):
                    print('Minimisation aborted as it proceeded too slow')
                    if write_to_f:
                        open('Optimfile.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                        open('Gradfile.txt','a').write('\nMinimisation aborted as it proceeded too slow')
                    min_costf0 = np.min(optim.Costflist)
                    min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the index number of the simulation where costf was minimal
                    state_opt0 = optim.Statelist[min_costf_ind]
                min_costf0 = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights)
    if write_to_f:
        open('Optimfile.txt','a').write('{0:>25s}'.format('\n finished'))
else:
    raise Exception('Unavailable optim_method \'' + str(optim_method) + '\' specified')
print('optimal state without ensemble='+str(state_opt0))
CostParts0 = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights,True)

def run_ensemble_member(counter,seed,non_state_paramdict={}):
    priorinput_mem = cp.deepcopy(priorinput)
    if seed != None:
        seed = seed + counter#create a unique seed for every member
    np.random.seed(seed) #VERY IMPORTANT! You have to explicitly set the seed (to None is ok), otherwise multicore implementation will use same random number for all ensemble members. 
    if np.count_nonzero(b_cov) == len(state) or not use_covar_to_pert: #than covariances all zero
        for j in range(len(state)):
            rand_nr_norm_distr = np.random.normal(0,np.sqrt(b_cov[j,j]))
            priorinput_mem.__dict__[state[j]] += rand_nr_norm_distr
            if imposeparambounds:
                if state[j] in boundedvars:
                    counter_while_loop = 1
                    while (priorinput_mem.__dict__[state[j]] < boundedvars[state[j]][0] or priorinput_mem.__dict__[state[j]] > boundedvars[state[j]][1]): #lower than lower bound or higher than upper bound
                        priorinput_mem.__dict__[state[j]] =  cp.deepcopy(priorinput.__dict__[state[j]])#so to make it within the bounds
                        rand_nr_norm_distr = np.random.normal(0,np.sqrt(b_cov[j,j]))
                        priorinput_mem.__dict__[state[j]] += rand_nr_norm_distr
                        if counter_while_loop >= 100:
                            raise Exception('No prior within bounds obtained for an ensemble member for state item '+state[j]+' after '+str(counter_while_loop)+ ' attempts')
                        counter_while_loop += 1                                                      
    else:
        if imposeparambounds:
            counter_while_loop = 1
            continueloop = True
            while continueloop:
                rand_nrs = np.random.multivariate_normal(np.zeros(len(state)),b_cov,check_valid='raise')
                continueloop = False            
                for j in range(len(state)):
                    if state[j] in boundedvars:
                        if priorinput_mem.__dict__[state[j]] + rand_nrs[j] < boundedvars[state[j]][0] or priorinput_mem.__dict__[state[j]] + rand_nrs[j] > boundedvars[state[j]][1]:
                            continueloop = True
                if counter_while_loop >= len(state)*100:
                    raise Exception('No prior within bounds obtained for an ensemble member after '+str(counter_while_loop)+ ' attempts')
                counter_while_loop += 1
        else:
            rand_nrs = np.random.multivariate_normal(np.zeros(len(state)),b_cov,check_valid='raise')
        for j in range(len(state)):
            priorinput_mem.__dict__[state[j]] += rand_nrs[j]
            
    non_state_pparamvals = {}
    if pert_non_state_param:
        for param in non_state_paramdict:
            if non_state_paramdict[param]['distr'] == 'normal':
                rand_nr = np.random.normal(0,non_state_paramdict[param]['scale'])
            elif non_state_paramdict[param]['distr'] == 'bounded normal':
                counter_while_loop = 1
                rand_nr = np.random.normal(0,non_state_paramdict[param]['scale'])
                while (priorinput_mem.__dict__[param] + rand_nr < non_state_paramdict[param]['leftbound'] or priorinput_mem.__dict__[param] + rand_nr > non_state_paramdict[param]['rightbound']): #lower than lower bound or higher than upper bound
                    rand_nr = np.random.normal(0,non_state_paramdict[param]['scale'])
                    if counter_while_loop >= 100:
                        raise Exception('No non_state parameter value within bounds obtained for an ensemble member for parameter '+param+' after '+str(counter_while_loop)+ ' attempts')
                    counter_while_loop += 1
            elif non_state_paramdict[param]['distr'] == 'uniform':
                rand_nr = np.random.uniform(non_state_paramdict[param]['leftbound'],non_state_paramdict[param]['rightbound'])
            elif non_state_paramdict[param]['distr'] == 'triangular':
                rand_nr = np.random.triangular(non_state_paramdict[param]['leftbound'],non_state_paramdict[param]['mode'],non_state_paramdict[param]['rightbound'])
            else:
                raise Exception('Problem for estimating model error: unknown distribtion for '+param)
            priorinput_mem.__dict__[param] += rand_nr
            non_state_pparamvals[param] = priorinput_mem.__dict__[param]
    priormodel_mem = fwdm.model(priorinput_mem)
    priormodel_mem.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
    optim_mem = im.inverse_modelling(priormodel_mem,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,StateVarNames=state,obsvarlist=obsvarlist,Optimfile='Optimfile'+str(counter)+'.txt',
                                     Gradfile='Gradfile'+str(counter)+'.txt',pri_err_cov_matr=b_cov,paramboundspenalty=paramboundspenalty,boundedvars=boundedvars)
    optim_mem.print = print_status_dur_ens
    Hx_prior_mem = {}
    for item in obsvarlist:
        Hx_prior_mem[item] = priormodel_mem.out.__dict__[item]
        optim_mem.__dict__['obs_'+item] = cp.deepcopy(optim.__dict__['obs_'+item])
        optim_mem.__dict__['error_obs_' + item] = cp.deepcopy(optim.__dict__['error_obs_' + item])
        if est_post_pdf_covmatr: 
            rand_nr_list = ([np.random.normal(0,measurement_error[item][i]) for i in range(len(measurement_error[item]))])
            optim_mem.__dict__['obs_'+item] += rand_nr_list
            if plot_perturbed_obs:
                unsca = 1 #a scale for plotting the obs with different units
                if (disp_units[item] == 'g/kg' or disp_units[item] == 'g kg$^{-1}$') and (item == 'q' or item.startswith('qmh')): #q can be plotted differently for clarity
                    unsca = 1000
                plt.figure()
                plt.plot(obs_times[item]/3600,unsca*optim.__dict__['obs_'+item], linestyle=' ', marker='*',ms=10,color = 'black',label = 'orig')
                plt.errorbar(obs_times[item]/3600,unsca*optim.__dict__['obs_'+item],yerr=unsca*measurement_error[item],ecolor='black',fmt='None')
                plt.plot(obs_times[item]/3600,unsca*optim_mem.__dict__['obs_'+item], linestyle=' ', marker='D',color = 'orange',label = 'pert')
                plt.ylabel(display_names[item] +' ('+ disp_units[item] + ')')
                plt.xlabel('time (h)')
                plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
                plt.legend(prop={'size':legendsize},loc=0)
                if write_to_f:
                    if not ('fig_obs_'+item+'_mem'+str(counter)+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system
                        plt.savefig('fig_obs_'+item+'_mem'+str(counter)+'.'+figformat, format=figformat)
                    else:
                        itemname = item + '_'
                        while ('fig_obs_'+itemname+'_mem'+str(counter)+'.'+figformat).lower() in [x.lower() for x in os.listdir()]:
                            itemname += '_'
                        #Windows cannnot have a file 'fig_obs_H_mem1.png' and 'fig_obs_h_mem1.png' in the same folder. The while loop can also handle e.g. the combination of variables Abc, ABC and abc                
                        plt.savefig('fig_obs_'+itemname+'_mem'+str(counter)+'.'+figformat, format=figformat)
    obs_sca_cf_mem = {}
    optim_mem.pstate = [] #needed also for use in the inverse_modelling object
    for item in state:
        optim_mem.pstate.append(priorinput_mem.__dict__[item])
        if item.startswith('obs_sca_cf_'):
            obsname = item.split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
            obs_sca_cf_mem[obsname] = cp.deepcopy(priorinput_mem.__dict__[item])
    optim_mem.pstate = np.array(optim_mem.pstate)
    inputcopy_mem = cp.deepcopy(priorinput_mem) #deepcopy!
    params = tuple([inputcopy_mem,state,obs_times,obs_weights]) 
    if 'EnBalDiffObsHFrac' in state:
        if 'H' in obsvarlist:
            optim_mem.EnBalDiffObs_atHtimes = optim.EnBalDiffObs_atHtimes
        if 'LE' in obsvarlist:
            optim_mem.EnBalDiffObs_atLEtimes = optim.EnBalDiffObs_atLEtimes
    if ana_deriv:
        optim_mem.checkpoint = cp.deepcopy(priormodel_mem.cpx) 
        optim_mem.checkpoint_init = cp.deepcopy(priormodel_mem.cpx_init)
        for item in obsvarlist:
            if 'EnBalDiffObsHFrac' in state:
                if item not in ['H','LE']:
                    observations_item = optim_mem.__dict__['obs_'+item]
                elif item == 'H':
                    observations_item = cp.deepcopy(optim_mem.__dict__['obs_H']) + optim_mem.pstate[state.index('EnBalDiffObsHFrac')] * optim_mem.EnBalDiffObs_atHtimes
                elif item == 'LE':
                    observations_item = cp.deepcopy(optim_mem.__dict__['obs_LE']) + (1 - optim_mem.pstate[state.index('EnBalDiffObsHFrac')]) * optim_mem.EnBalDiffObs_atLEtimes 
            else:
                observations_item = optim_mem.__dict__['obs_'+item]
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
                    forcing = weight * (Hx_prior_mem[item][ti] - obs_scale * observations_item[k])/(optim_mem.__dict__['error_obs_' + item][k]**2) #don't include the background term of the cost function in the forcing!
                    optim_mem.forcing[ti][item] = forcing
                    k += 1
    if paramboundspenalty:
        optim_mem.setNanCostfOutBoundsTo0 = setNanCostfOutBoundsTo0
        optim_mem.penalty_exp = penalty_exp
    if optim_method == 'bfgs':
        try:
            if ana_deriv:
                minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params,gtol=gtol,full_output=True)
            else:
                minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params,gtol=gtol,full_output=True)
            state_opt_mem = minimisation_mem[0]
            min_costf_mem = minimisation_mem[1]
        except (im.nan_incostfError):
            print('Minimisation aborted due to nan') #discard_nan_minims == False allows to use last non-nan result in the optimisation, otherwise we throw away the optimisation
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
            if (discard_nan_minims == False and len(optim_mem.Statelist) > 0): #len(optim_mem.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf_mem = np.min(optim_mem.Costflist)
                min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the index number of the simulation where costf was minimal
                state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
            else:
                state_opt_mem = np.array([np.nan for x in range(len(state))])
                min_costf_mem = np.nan
        except (im.static_costfError):
            print('Minimisation aborted as it proceeded too slow')
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                open('Gradfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow')
            min_costf_mem = np.min(optim_mem.Costflist)
            min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the index number of the simulation where costf was minimal
            state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
    elif optim_method == 'tnc':
        try:
            if ana_deriv:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params,bounds=bounds,maxfun=None)
            else:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params,bounds=bounds,maxfun=None)
            state_opt_mem = minimisation_mem[0]
            min_costf_mem = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights) #within cost_func, the values of the variables in inputcopy_mem that are also state variables will be overwritten by the values of the variables in state_opt_mem
        except (im.nan_incostfError):
            print('Minimisation aborted due to nan') #discard_nan_minims == False allows to use last non-nan result in the optimisation, otherwise we throw away the optimisation
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
            if (discard_nan_minims == False and len(optim_mem.Statelist) > 0): #len(optim_mem.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf_mem = np.min(optim_mem.Costflist)
                min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) 
                state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
            else:
                state_opt_mem = np.array([np.nan for x in range(len(state))])
                min_costf_mem = np.nan
            optim_mem.stop = True
        except (im.static_costfError):
            print('Minimisation aborted as it proceeded too slow')
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                open('Gradfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow')
            min_costf_mem = np.min(optim_mem.Costflist)
            min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) 
            state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
        if not hasattr(optim_mem,'stop'):
            for i in range(maxnr_of_restarts):
                if min_costf_mem > stopcrit: #will evaluate to False if min_costf_mem is equal to nan
                    optim_mem.nr_of_sim_bef_restart = optim_mem.sim_nr
                    if write_to_f:
                        open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n restart'))
                        open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n restart'))
                    try:
                        if ana_deriv:
                            minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                        else:
                            minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                        state_opt_mem = minimisation_mem[0]
                    except (im.nan_incostfError):
                        print('Minimisation aborted due to nan, no restart for this member')
                        if write_to_f:
                            open('Optimfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                            open('Gradfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                        if discard_nan_minims == False:
                            min_costf_mem = np.min(optim_mem.Costflist)
                            min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the index number of the simulation where costf was minimal
                            state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                        else:
                            state_opt_mem = np.array([np.nan for x in range(len(state))])
                            min_costf_mem = np.nan
                        break
                    except (im.static_costfError):
                        print('Minimisation aborted as it proceeded too slow')
                        if write_to_f:
                            open('Optimfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow') #\n to make it start on a new line
                            open('Gradfile'+str(counter)+'.txt','a').write('\nMinimisation aborted as it proceeded too slow')
                        min_costf_mem = np.min(optim_mem.Costflist)
                        min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the index number of the simulation where costf was minimal
                        state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                    min_costf_mem = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights)
    CostParts = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights,True)
    if write_to_f:
        open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n finished'))
    return min_costf_mem,state_opt_mem,optim_mem.pstate,non_state_pparamvals,CostParts
   
#chi squared denominator to be used later
if use_weights:
    denom_chisq = tot_sum_of_weights
else:
    denom_chisq = number_of_obs
if use_backgr_in_cost:
    denom_chisq += number_of_params

if use_ensemble:
    ensemble = []
    for i in range(0,nr_of_members): #the zeroth is the one done before
        ensemble.append({})
    ensemble[0]['min_costf'] = min_costf0
    ensemble[0]['state_opt'] = state_opt0   
    ensemble[0]['pstate'] = optim.pstate 
    ensemble[0]['CostParts'] = CostParts0
    if write_to_f:
        shutil.move('Optimfile.txt', 'Optimfile_'+str(0)+'.txt')
        shutil.move('Gradfile.txt', 'Gradfile_'+str(0)+'.txt')
    if not set_seed:
        seedvalue = None
    if run_multicore:
        if max_num_cores == 'all':
            max_num_cores_ = -1
        else:
            max_num_cores_ = max_num_cores
        result_array = Parallel(n_jobs=max_num_cores_)(delayed(run_ensemble_member)(i,seedvalue,non_state_paramdict)  for i in range(1,nr_of_members)) #, prefer="threads" makes it work, but probably not multiprocess. None is for the seed
        #the above returns a list of tuples
        for j in range(1,nr_of_members):
            ensemble[j]['min_costf'] = result_array[j-1][0] #-1 due to the fact that the zeroth ensemble member is not part of the result_array, while it is part of ensemble
            ensemble[j]['state_opt'] = result_array[j-1][1]
            ensemble[j]['pstate'] = result_array[j-1][2]
            ensemble[j]['nonstateppars'] = result_array[j-1][3] #an empty dictionary if not pert_non_state_param=True
            ensemble[j]['CostParts'] = result_array[j-1][4]
    else:
        for i in range(1,nr_of_members):
            ensemble[i]['min_costf'],ensemble[i]['state_opt'],ensemble[i]['pstate'],ensemble[i]['nonstateppars'],ensemble[i]['CostParts'] =  run_ensemble_member(i,seedvalue,non_state_paramdict)
    print('whole ensemble:')
    print(ensemble)
    seq_costf = np.array([x['min_costf'] for x in ensemble]) #iterate over the dictionaries
    min_costf_ensemble = np.nanmin(seq_costf)
    if np.isnan(min_costf_ensemble):
        raise Exception('All optimisations in ensemble resulted in nan!')
    opt_sim_nr = np.where(seq_costf == min_costf_ensemble)[0][0]
    state_opt = ensemble[opt_sim_nr]['state_opt']
    print('optimal state ensemble '+ str(state) +':')
    print(state_opt)
    print('index of member with the best state:')
    print(opt_sim_nr)
    if est_post_pdf_covmatr:
        mean_state_post = np.zeros(len(state))
        mean_state_prior = np.zeros(len(state))
        chi_sq = np.zeros(len(ensemble))
        success_ens = np.zeros(len(ensemble), dtype=bool) #True or False wether optimisation successful
        for i in range(len(ensemble)):
            if not np.isnan(seq_costf[i]):
                chi_sq[i] = seq_costf[i]/denom_chisq
                if chi_sq[i] <= succes_opt_crit:
                    success_ens[i] = True
        if np.sum(success_ens[1:]) > 1:
            for i in range(len(state)):
                seq = np.array([x['state_opt'][i] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
                seq_suc = np.array([seq[x] for x in range(len(seq)) if success_ens[1:][x]])
                mean_state_post[i] = np.mean(seq_suc) #np.nanmean not necessary since we filter already for successful optimisations
                nbins = np.linspace(np.min(seq_suc), np.max(seq_suc), nr_bins + 1)
                n, bins = np.histogram(seq_suc, nbins, density=1)
                pdfx = np.zeros(n.size)
                pdfy = np.zeros(n.size)
                for k in range(n.size):
                    pdfx[k] = 0.5*(bins[k]+bins[k+1])
                    pdfy[k] = n[k]
                fig = plt.figure()
                plt.plot(pdfx,pdfy, linestyle='-', linewidth=2,color='red',label='post')
                seq_p = np.array([x['pstate'][i] for x in ensemble[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
                seq_suc_p = np.array([seq_p[x] for x in range(len(seq_p)) if success_ens[1:][x]])
                mean_state_prior[i] = np.mean(seq_suc_p)
                nbins_p = np.linspace(np.min(seq_suc_p), np.max(seq_suc_p), nr_bins + 1)
                n_p, bins_p = np.histogram(seq_suc_p, nbins_p, density=1)
                pdfx = np.zeros(n_p.size)
                pdfy = np.zeros(n_p.size)
                for k in range(n_p.size):
                    pdfx[k] = 0.5*(bins_p[k]+bins_p[k+1])
                    pdfy[k] = n_p[k]
                plt.plot(pdfx,pdfy, linestyle='dashed', linewidth=2,color='gold',label='prior')
                plt.axvline(mean_state_post[i], linestyle='-',linewidth=2,color='red',label = 'mean post')
                plt.axvline(mean_state_prior[i], linestyle='dashed',linewidth=2,color='gold',label = 'mean prior')
                plt.xlabel(state[i] + ' ('+ disp_units_par[state[i]] +')')
                plt.ylabel('Probability density (-)')  
                plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
                plt.legend(loc=0, frameon=True,prop={'size':legendsize}) 
                if write_to_f:
                    if not ('pdf_posterior_'+state[i]+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system
                        plt.savefig('pdf_posterior_'+state[i]+'.'+figformat, format=figformat)
                    else:
                        itemname = state[i] + '_'
                        while ('pdf_posterior_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: 
                            itemname += '_'
                        plt.savefig('pdf_posterior_'+itemname+'.'+figformat, format=figformat)
            matr = np.zeros((len(state),np.sum(success_ens[1:]))) #exclude the first ensemble member even if it was successful, since the prior was not randomly sampled, adding it influences the variance.
            j = 0
            for i in range(len(ensemble[1:])):
                if success_ens[1:][i]:
                    matr[:,j] = ensemble[1:][i]['state_opt']
                    j += 1
            post_cov_matr = np.cov(matr,ddof=1)
            post_cor_matr = np.corrcoef(matr) #no ddof for np.corrcoef, gives DeprecationWarning
            #see https://stackoverflow.com/questions/21030668/why-do-numpy-cov-diagonal-elements-and-var-functions-have-different-values, np.cov does not give the same variance as np.var, except when setting the degrees of freedom to the same value.
            #note that the variance of a variable minus its mean is the same as the variance of the variable alone. So if the post error = variable minus its mean, the above estimates the posterior error covariance matrix
            if len(state) == 1:
                post_cov_matr = post_cov_matr.reshape(1,1) #needed, otherwise post_cov_matr[0] raises an error
                post_cor_matr = post_cor_matr.reshape(1,1)
            if pert_non_state_param:
                mean_nonstate_p = {}
                matr_incl_nonst = cp.deepcopy(matr)
                nonstparamlist = []
                for param in non_state_paramdict:
                    fig = plt.figure()
                    seq_ns = np.array([x['nonstateppars'][param] for x in ensemble[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
                    seq_suc_ns = np.array([seq_ns[x] for x in range(len(seq_ns)) if success_ens[1:][x]])
                    mean_nonstate_p[param] = np.mean(seq_suc_ns)
                    nbins_ns = np.linspace(np.min(seq_suc_ns), np.max(seq_suc_ns), nr_bins + 1)
                    n_ns, bins_ns = np.histogram(seq_suc_ns, nbins_ns, density=1)
                    pdfx = np.zeros(n_ns.size)
                    pdfy = np.zeros(n_ns.size)
                    for k in range(n_ns.size):
                        pdfx[k] = 0.5*(bins_ns[k]+bins_ns[k+1])
                        pdfy[k] = n_ns[k]
                    plt.plot(pdfx,pdfy, linestyle='-', linewidth=2,color='black',label='pdf')
                    plt.axvline(mean_nonstate_p[param], linestyle='dashed',linewidth=2,color='black',label = 'mean')
                    plt.xlabel(param + ' ('+ disp_units_par[param] +')')
                    plt.ylabel('Probability density (-)')  
                    plt.legend(prop={'size':legendsize},loc=0)
                    plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
                    if write_to_f:
                        if not ('pdf_nonstate_'+param+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system
                            plt.savefig('pdf_nonstate_'+param+'.'+figformat, format=figformat)
                        else:
                            itemname = param + '_'
                            while ('pdf_nonstate_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]:
                                itemname += '_'
                            plt.savefig('pdf_nonstate_'+itemname+'.'+figformat, format=figformat)
                    matr_incl_nonst = np.append(matr_incl_nonst,[seq_suc_ns],axis=0)
                    nonstparamlist.append(param)
                post_cov_matr_incl_nonst = np.cov(matr_incl_nonst,ddof=1) 
                post_cor_matr_incl_nonst = np.corrcoef(matr_incl_nonst) #no ddof for np.corrcoef, gives DeprecationWarning

if use_ensemble:
    optimalstate = cp.deepcopy(state_opt)
else:
    optimalstate = cp.deepcopy(state_opt0)            
optimalinput = cp.deepcopy(priorinput)
i = 0
for item in state:
    optimalinput.__dict__[item] = optimalstate[i]
    i += 1
optimalmodel = fwdm.model(optimalinput) #Note that this does not account for non-state parameters possibly changed in the ensemble
optimalmodel.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)

if use_ensemble:
    if pert_non_state_param and opt_sim_nr != 0:
        optimalinput_onsp = cp.deepcopy(optimalinput)#onsp means optimal non-state params
        for item in nonstparamlist:
            optimalinput_onsp.__dict__[item] = ensemble[opt_sim_nr]['nonstateppars'][item]
        optimalmodel_onsp = fwdm.model(optimalinput_onsp) 
        optimalmodel_onsp.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)

############################
#stats file
############################
chi_sq0 = min_costf0 / denom_chisq #calculation of chi squared statistic
if use_ensemble:
    chi_sq_ens = min_costf_ensemble / denom_chisq
    
if write_to_f:
    open('Optstatsfile.txt','w').write('{0:>9s}'.format('nr of obs')) #here we make the file   
    if use_weights:
        open('Optstatsfile.txt','a').write('{0:>40s}'.format('total nr obs, corrected for weights'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('number of params to optimise'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('chi squared without ensemble'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('prior costf without ensemble'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('post costf without ensemble'))
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>50s}'.format('chi squared of member with lowest post costf'))
        open('Optstatsfile.txt','a').write('{0:>50s}'.format('prior costf of member with lowest post costf'))
        open('Optstatsfile.txt','a').write('{0:>50s}'.format('post costf of member with lowest post costf'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>9s}'.format(str(number_of_obs)))
    if use_weights:
        open('Optstatsfile.txt','a').write('{0:>40s}'.format(str(tot_sum_of_weights)))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format(str(number_of_params)))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format(str(chi_sq0)))
    prior_costf = optim.cost_func(optim.pstate,inputcopy,state,obs_times,obs_weights)
    post_costf = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights)
    open('Optstatsfile.txt','a').write('{0:>35s}'.format(str(prior_costf)))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format(str(post_costf)))
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>50s}'.format(str(chi_sq_ens)))
        if opt_sim_nr != 0:
            filename = 'Optimfile'+str(opt_sim_nr)+'.txt'
        else:
            filename = 'Optimfile_0.txt'
        with open(filename,'r') as LowestCfFile:
            header_end = LowestCfFile.readline().split()[-1]
            if header_end != 'Costf':
                raise Exception('Optimfile'+str(opt_sim_nr)+'.txt does not have \'Costf\' as last column')
            prior_costf_ens = LowestCfFile.readline().split()[-1]
        open('Optstatsfile.txt','a').write('{0:>50s}'.format(str(prior_costf_ens)))
        open('Optstatsfile.txt','a').write('{0:>50s}'.format(str(ensemble[opt_sim_nr]['min_costf'])))
    open('Optstatsfile.txt','a').write('\n\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('optimal state without ensemble:'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('      ')
    for item in state:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('      ')
    for item in state_opt0:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
    open('Optstatsfile.txt','a').write('\n')
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('optimal state with ensemble:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('      ')
        for item in state_opt:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('index member with best state:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>31s}'.format(str(opt_sim_nr)))
        open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('costf parts best state:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    if use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('Background'))
    if paramboundspenalty:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('Penalty'))
    open('Optstatsfile.txt','a').write('\n      ')
    if use_ensemble:
        CPdictio = ensemble[opt_sim_nr]['CostParts']
    else:
        CPdictio = CostParts0
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio[obsvar])))
    if use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio['backgr'])))
    if paramboundspenalty:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio['penalty'])))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('chi squared for those parts:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        if use_weights:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio[obsvar] / WeightsSums[obsvar])))
        else:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio[obsvar] / len(optim.__dict__['obs_'+obsvar]))))
    if use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictio['backgr'] / number_of_params)))
    if use_ensemble or use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('Normalised deviation to unper-'))
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('\n    turbed prior for best state:'))
        reldev = np.zeros(len(state))
        for i in range(len(state)):
            reldev[i] = (optimalstate[i]-priorinput.__dict__[state[i]])/np.sqrt(priorvar[state[i]])
        open('Optstatsfile.txt','a').write('\n      ')
        for item in state:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
        open('Optstatsfile.txt','a').write('\n      ')
        for item in reldev:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('model_variance/obs_variance'))
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('\n                 for best state:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    outp_at_obstimes = {}
    obs_to_use = {}
    outp_at_obstimes_pr = {}
    obs_to_use_pr = {}
    var_ratio_pr = {}#different system since written later, so we need to store it
    for obsvar in obsvarlist:
        outp_at_obstimes[obsvar] = []
        outp_at_obstimes_pr[obsvar] = []
        for ti in range(priormodel.tsteps):
            if round(optimalmodel.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[obsvar]]:
                outp_at_obstimes[obsvar] += [optimalmodel.out.__dict__[obsvar][ti]]
                outp_at_obstimes_pr[obsvar] += [priormodel.out.__dict__[obsvar][ti]]
        numerator = np.var(outp_at_obstimes[obsvar])
        numerator_pr = np.var(outp_at_obstimes_pr[obsvar])
        obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_'+obsvar])
        obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_'+obsvar])
        if 'obs_sca_cf_'+obsvar in state:
            obs_to_use[obsvar] *= optimalinput.__dict__['obs_sca_cf_'+obsvar]
            obs_to_use_pr[obsvar] *= priorinput.__dict__['obs_sca_cf_'+obsvar]
        elif 'EnBalDiffObsHFrac' in state:
            if obsvar == 'H':
                obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_H']) + optimalstate[state.index('EnBalDiffObsHFrac')] * optim.EnBalDiffObs_atHtimes
                obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_H']) + optim.pstate[state.index('EnBalDiffObsHFrac')] * optim.EnBalDiffObs_atHtimes
            elif obsvar == 'LE':
                obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optimalstate[state.index('EnBalDiffObsHFrac')]) * optim.EnBalDiffObs_atLEtimes
                obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optim.pstate[state.index('EnBalDiffObsHFrac')]) * optim.EnBalDiffObs_atLEtimes
        denominator = np.var(obs_to_use[obsvar])
        denominator_pr = np.var(obs_to_use_pr[obsvar])
        var_ratio = numerator/denominator
        var_ratio_pr[obsvar] = numerator_pr/denominator_pr
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(var_ratio)))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('model_variance/obs_variance'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('for prior:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(var_ratio_pr[obsvar])))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('Mean bias error(mod-obs)'))
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('\n                 for best state:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        mbe = np.mean(outp_at_obstimes[obsvar]-obs_to_use[obsvar])
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(mbe)))        
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('Mean bias error prior(mod-obs):'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        mbe_pr = np.mean(outp_at_obstimes_pr[obsvar]-obs_to_use_pr[obsvar])
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(mbe_pr)))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('Root mean squared error'))
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('\n                 for best state:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        rmse = np.sqrt(np.mean((outp_at_obstimes[obsvar]-obs_to_use[obsvar])**2))
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(rmse)))  
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('Root mean squared error'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('for prior:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        rmse_pr = np.sqrt(np.mean((outp_at_obstimes_pr[obsvar]-obs_to_use_pr[obsvar])**2))
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(rmse_pr)))
    if use_ensemble:
        if est_post_pdf_covmatr:
            open('Optstatsfile.txt','a').write('\n\n')
            open('Optstatsfile.txt','a').write('{0:>32s}'.format('estim post state covar matrix:'))
            open('Optstatsfile.txt','a').write('\n')
            if np.sum(success_ens[1:]) > 1:
                if np.sum(success_ens[1:]) < 10:
                    open('Optstatsfile.txt','a').write('Warning: posterior state statistics estimated on basis of only '+str(np.sum(success_ens[1:]))+' ensemble members\n')
                open('Optstatsfile.txt','a').write('{0:>31s}'.format(' '))
                for item in state:
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                open('Optstatsfile.txt','a').write('\n')            
                for i in range(len(state)):
                    open('Optstatsfile.txt','a').write('{0:>31s}'.format(state[i]))
                    for item in post_cov_matr[i]:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
                    open('Optstatsfile.txt','a').write('\n\n')
                open('Optstatsfile.txt','a').write('{0:>32s}'.format('estim post state corr matrix:'))
                open('Optstatsfile.txt','a').write('\n')
                open('Optstatsfile.txt','a').write('{0:>31s}'.format(' '))
                for item in state:
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                open('Optstatsfile.txt','a').write('\n')            
                for i in range(len(state)):
                    open('Optstatsfile.txt','a').write('{0:>31s}'.format(state[i]))
                    for item in post_cor_matr[i]:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
                    open('Optstatsfile.txt','a').write('\n\n')
                    
                if pert_non_state_param:    
                    open('Optstatsfile.txt','a').write('{0:>32s}'.format('est cov mat post state+pert par:'))
                    open('Optstatsfile.txt','a').write('\n')
                    open('Optstatsfile.txt','a').write('{0:>31s}'.format(' '))
                    for item in state:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                    for param in nonstparamlist:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(param))
                    open('Optstatsfile.txt','a').write('\n')            
                    for i in range(len(post_cov_matr_incl_nonst)):
                        if i <= len(state) -1:
                            open('Optstatsfile.txt','a').write('{0:>31s}'.format(state[i]))
                        else:
                            open('Optstatsfile.txt','a').write('{0:>31s}'.format(nonstparamlist[i-len(state)]))
                        for item in post_cov_matr_incl_nonst[i]:
                            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
                        open('Optstatsfile.txt','a').write('\n\n')   
                        
                    open('Optstatsfile.txt','a').write('{0:>32s}'.format('est cor mat post state+pert par:'))
                    open('Optstatsfile.txt','a').write('\n')
                    open('Optstatsfile.txt','a').write('{0:>31s}'.format(' '))
                    for item in state:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                    for param in nonstparamlist:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(param))
                    open('Optstatsfile.txt','a').write('\n')            
                    for i in range(len(post_cor_matr_incl_nonst)):
                        if i <= len(state) -1:
                            open('Optstatsfile.txt','a').write('{0:>31s}'.format(state[i]))
                        else:
                            open('Optstatsfile.txt','a').write('{0:>31s}'.format(nonstparamlist[i-len(state)]))
                        for item in post_cor_matr_incl_nonst[i]:
                            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item)))
                        open('Optstatsfile.txt','a').write('\n\n')   
                    
                open('Optstatsfile.txt','a').write('{0:>32s}'.format('Mean posterior state elements:'))
                open('Optstatsfile.txt','a').write('\n')
                open('Optstatsfile.txt','a').write('{0:>6s}'.format(' '))
                for item in state:
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                open('Optstatsfile.txt','a').write('\n')            
                open('Optstatsfile.txt','a').write('{0:>6s}'.format(' '))
                for i in range(len(state)):
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(mean_state_post[i])))
                open('Optstatsfile.txt','a').write('\n\n')    
                open('Optstatsfile.txt','a').write('{0:>33s}'.format('post/prior variance rat ensemb:\n'))  
                open('Optstatsfile.txt','a').write('{0:>6s}'.format(' '))
                for item in state:
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
                open('Optstatsfile.txt','a').write('\n')
                open('Optstatsfile.txt','a').write('{0:>6s}'.format(' '))
                for i in range(len(state)):
                    pstate_seq = np.array([x['pstate'][i] for x in ensemble[1:]])
                    pstate_seq_suc = np.array([pstate_seq[x] for x in range(len(pstate_seq)) if success_ens[1:][x]])
                    priorvar_state_i = np.var(pstate_seq_suc,ddof=1)
                    ratio = post_cov_matr[i][i]/priorvar_state_i
                    open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(ratio))) 
            else:
                open('Optstatsfile.txt','a').write('Not enough successful optimisations (selected criterion chi squared <= '+str(succes_opt_crit)+') to estimate posterior error covariance matrix \n')
            open('Optstatsfile.txt','a').write('{0:>32s}'.format('\n\nNr of success mems (excl mem 0):\n'))
            open('Optstatsfile.txt','a').write('{0:>31s}'.format(str(np.sum(success_ens[1:]))))
        open('Optstatsfile.txt','a').write('\n\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('optimised ensemble members:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>6s}'.format('number'))
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('minimum costf'))
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('chi squared'))
        if est_post_pdf_covmatr:
            open('Optstatsfile.txt','a').write('{0:>14s}'.format('successful'))
        for item in state:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
        if pert_non_state_param: 
            open('Optstatsfile.txt','a').write('{0:>32s}'.format('perturbed non-state params:'))
            for param in nonstparamlist: 
                open('Optstatsfile.txt','a').write('{0:>25s}'.format(param))
        open('Optstatsfile.txt','a').write('\n')
        i = 0
        for item in ensemble:
            open('Optstatsfile.txt','a').write('{0:>6s}'.format(str(i)))
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item['min_costf'])))
            chisq_member = item['min_costf'] / denom_chisq
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(chisq_member)))
            if est_post_pdf_covmatr:
                open('Optstatsfile.txt','a').write('{0:>14s}'.format(str(success_ens[i])))
            for param in item['state_opt']:
                open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(param)))
            if pert_non_state_param: 
                open('Optstatsfile.txt','a').write('{0:>32s}'.format(' '))
                if i != 0:
                    for param in nonstparamlist:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item['nonstateppars'][param])))                
                else:
                    for param in nonstparamlist:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(priorinput.__dict__[param])))
            open('Optstatsfile.txt','a').write('\n')
            i += 1
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('prior ensemble members:'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>6s}'.format('number'))
        for item in state:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(item))
        open('Optstatsfile.txt','a').write('\n')
        i = 0
        for item in ensemble:
            open('Optstatsfile.txt','a').write('{0:>6s}'.format(str(i)))
            for param in item['pstate']:
                open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(param)))
            open('Optstatsfile.txt','a').write('\n')
            i += 1

           
for i in range(len(obsvarlist)): #Note that only the obs of member 0 (the real obs) are plotted here 
    unsca = 1 #a scale for plotting the obs with different units
    if (disp_units[obsvarlist[i]] == 'g/kg' or disp_units[obsvarlist[i]] == 'g kg$^{-1}$') and (obsvarlist[i] == 'q' or obsvarlist[i].startswith('qmh')): #q can be plotted differently for clarity
        unsca = 1000
    fig = plt.figure()
    plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]],yerr=unsca*optim.__dict__['error_obs_'+obsvarlist[i]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]],yerr=unsca*measurement_error[obsvarlist[i]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    plt.plot(priormodel.out.t,unsca*priormodel.out.__dict__[obsvarlist[i]], ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
    plt.plot(priormodel.out.t,unsca*optimalmodel.out.__dict__[obsvarlist[i]], linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
    if use_ensemble:
        if pert_non_state_param and opt_sim_nr != 0:
            plt.plot(priormodel.out.t,unsca*optimalmodel_onsp.out.__dict__[obsvarlist[i]], linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
    plt.plot(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='*',color = 'black',ms=10, label = 'obs')
    if 'obs_sca_cf_'+obsvarlist[i] in state: #plot the obs scaled with the scaling factors (if applicable)
        plt.plot(obs_times[obsvarlist[i]]/3600,optimalinput.__dict__['obs_sca_cf_'+obsvarlist[i]]*unsca*optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    plt.ylabel(display_names[obsvarlist[i]] +' ('+ disp_units[obsvarlist[i]] + ')')
    plt.xlabel('time (h)')
    plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    plt.legend(prop={'size':legendsize},loc=0)
    if write_to_f:
        if not ('fig_fit_'+obsvarlist[i]+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system 
            plt.savefig('fig_fit_'+obsvarlist[i]+'.'+figformat, format=figformat)
        else:
            itemname = obsvarlist[i] + '_'
            while ('fig_fit_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: 
                itemname += '_'
            plt.savefig('fig_fit_'+itemname+'.'+figformat, format=figformat)#Windows cannnot have a file 'fig_fit_h' and 'fig_fit_H' in the same folder. The while loop can also handle e.g. the combination of variables Abc, ABC and abc   

if 'EnBalDiffObsHFrac' in state:
    if 'H' in obsvarlist:
        enbal_corr_H = optim.obs_H + optimalinput.EnBalDiffObsHFrac * optim.EnBalDiffObs_atHtimes
        fig = plt.figure()
        plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=optim.__dict__['error_obs_H'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
        plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=measurement_error['H'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
        plt.plot(priormodel.out.t,priormodel.out.H, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
        plt.plot(optimalmodel.out.t,optimalmodel.out.H, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
        if use_ensemble:
            if pert_non_state_param and opt_sim_nr != 0:
                plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.H, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
        plt.plot(obs_times['H']/3600,optim.__dict__['obs_'+'H'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
        plt.plot(obs_times['H']/3600,enbal_corr_H, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
        plt.ylabel('H (' + disp_units['H']+')')
        plt.xlabel('time (h)')
        plt.legend(prop={'size':legendsize},loc=0)
        plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
        if write_to_f:
            plt.savefig('fig_fit_enbalcorrH.'+figformat, format=figformat)
    if 'LE' in obsvarlist:        
        enbal_corr_LE = optim.obs_LE + (1 - optimalinput.EnBalDiffObsHFrac) * optim.EnBalDiffObs_atLEtimes
        fig = plt.figure()
        plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=optim.__dict__['error_obs_LE'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
        plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=measurement_error['LE'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
        plt.plot(priormodel.out.t,priormodel.out.LE, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
        plt.plot(optimalmodel.out.t,optimalmodel.out.LE, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
        if use_ensemble:
            if pert_non_state_param and opt_sim_nr != 0:
                plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.LE, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
        plt.plot(obs_times['LE']/3600,optim.__dict__['obs_'+'LE'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
        plt.plot(obs_times['LE']/3600,enbal_corr_LE, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
        plt.ylabel('LE (' + disp_units['LE']+')')
        plt.xlabel('time (h)')
        plt.legend(prop={'size':legendsize},loc=0)
        plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
        if write_to_f:
            plt.savefig('fig_fit_enbalcorrLE.'+figformat, format=figformat)

if write_to_f:
    if wr_obj_to_pickle_files:
        if storefolder_objects not in os.listdir():
            os.mkdir(storefolder_objects)
        for item in vars_to_pickle:
            if item in vars(): #check if variable exists
                with open(storefolder_objects+'/'+item+'.pkl', 'wb') as output:
                    pickle.dump(vars()[item], output, pickle.HIGHEST_PROTOCOL)
########################################################
###### user input: additional plotting etc. ############
########################################################       
#The following code can be used to plot profiles of CO2 (adapt depending on the optimisation performed)
#profileheights = np.array([priorinput.CO2measuring_height4,priorinput.CO2measuring_height3,priorinput.CO2measuring_height2,priorinput.CO2measuring_height])    
#colorlist = ['red','gold','green','blue','orange','pink']
#markerlist = ['x','v','s','p']
#
#fig = plt.figure()
#i = 0
#for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
#    color = colorlist[i]
#    plt.plot(priormodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'pmod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
#    plt.plot(priormodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
#    plt.plot(priormodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
#    plt.plot(priormodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
#    i += 1
#plt.ylabel('height (m)')
#plt.xlabel('CO2 mixing ratio ('+disp_units['CO2mh']+')')
#plt.ylim([np.min(profileheights)-0.01*(np.max(profileheights)-np.min(profileheights)),np.max(profileheights)+0.01*(np.max(profileheights)-np.min(profileheights))]) 
#i = 0
#for ti in range(0,len(obs_times['CO2mh']),2):
#    marker = markerlist[i]
#    color = colorlist[i]
#    plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
#    plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
#    plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
#    plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
#    i += 1
#plt.legend(fontsize=8)  
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#if write_to_f:
#    plt.savefig('fig_'+'CO2'+'_profile_prior.'+figformat, format=figformat)
#
#fig = plt.figure()
#i = 0
#for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
#    color = colorlist[i]
#    plt.plot(optimalmodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'mod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
#    plt.plot(optimalmodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
#    plt.plot(optimalmodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
#    plt.plot(optimalmodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
#    i += 1
#plt.ylabel('height (m)')
#plt.xlabel('CO2 mixing ratio ('+disp_units['CO2mh']+')') 
#plt.ylim([np.min(profileheights)-0.01*(np.max(profileheights)-np.min(profileheights)),np.max(profileheights)+0.01*(np.max(profileheights)-np.min(profileheights))]) 
#i = 0
#for ti in range(0,len(obs_times['CO2mh']),2):
#    marker = markerlist[i]
#    color = colorlist[i]
#    plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
#    plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
#    plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
#    plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
#    i += 1
#plt.legend(fontsize=8)  
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#if write_to_f:
#    plt.savefig('fig_'+'CO2'+'_profile.'+figformat, format=figformat)
    
#fig = plt.figure()
#plt.plot(optimalmodel.out.t,priormodel.out.wCO2, linestyle='--', marker='o',color='yellow')
#plt.plot(optimalmodel.out.t,optimalmodel.out.wCO2, linestyle='--', marker='o',color='red')
#plt.plot(hours_mean,wCO2_mean, linestyle=' ', marker='o',color='black')
#plt.ylabel('CO2 flux (mg CO2/m2/s)')
#plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#if write_to_f:
#    plt.savefig('fig_wCO2.'+figformat, format=figformat)

#fig = plt.figure()
#if priormodel.sw_ls:
#    plt.plot(priormodel.out.t,priormodel.out.__dict__['H'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['H'], linestyle=' ', marker='o',color='red',label = 'post')
#else:
#    plt.plot(priormodel.out.t,priormodel.out.__dict__['wtheta']*priormodel.rho*priormodel.cp, linestyle=' ', marker='o',color='yellow',label = 'prior')
#    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['wtheta']*priormodel.rho*priormodel.cp, linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,H_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('H (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'H'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#if priormodel.sw_ls:
#    plt.plot(priormodel.out.t,priormodel.out.__dict__['LE'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['LE'], linestyle=' ', marker='o',color='red',label = 'post')
#else:
#    plt.plot(priormodel.out.t,priormodel.out.__dict__['wq']*priormodel.rho*priormodel.Lv, linestyle=' ', marker='o',color='yellow',label = 'prior')
#    plt.plot(priormodel.out.t,optimalmodel.out.__dict__['wq']*priormodel.rho*priormodel.Lv, linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,LE_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('LE (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'LE'+'.'+figformat, format=figformat)
#
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['Swin'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Swin'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,SWD_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('SWD (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'SWD'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['Swout'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Swout'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,SWU_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('SWU (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'SWU'+'.'+figformat, format=figformat)
#
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['Lwout'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Lwout'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,LWU_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('LWU (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'LWU'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['Lwin'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['Lwin'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,LWD_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('LWD (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'LWD'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['G'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['G'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,G_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('G (W m-2)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'G'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['theta'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.__dict__['theta'], linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(np.array(obstimes_T)/3600,Temp200_selected*((Press_selected-200*9.81*rho80)/100000)**(-287.04/1005), linestyle=' ', marker='o',color = 'black')
#plt.ylabel('theta (K)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'theta'+'.'+figformat, format=figformat)
#    
#fig = plt.figure()
#plt.plot(np.array(obstimes_T)/3600,Temp200_selected*((Press_selected-200*9.81*rho80)/100000)**(-287.04/1005), linestyle=' ', marker='o',color = 'black')
#plt.ylabel('theta (K)')
#plt.ylim = ([282,292])
#plt.xlim = ([7,17])
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)

#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.__dict__['h'], linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(np.array(dhour_BLH_selected),BLH_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('h (m)')
#plt.xlabel('time (h)')
#plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
#plt.legend()
#if write_to_f:
#    plt.savefig('fig_'+'h'+'.'+figformat, format=figformat)
#        
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.H+priormodel.out.LE+priormodel.out.G, linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.H+optimalmodel.out.LE+optimalmodel.out.G, linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(obs_times['H']/3600,H_selected+LE_selected+G_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('H+LE+G')
#
#fig = plt.figure()
#plt.plot(priormodel.out.t,priormodel.out.Swin+priormodel.out.Lwin-priormodel.out.Swout-priormodel.out.Lwout, linestyle=' ', marker='o',color='yellow',label = 'prior')
#plt.plot(priormodel.out.t,optimalmodel.out.Swin+optimalmodel.out.Lwin-optimalmodel.out.Swout-optimalmodel.out.Lwout, linestyle=' ', marker='o',color='red',label = 'post')
#plt.plot(obs_times['H']/3600,SWD_selected+LWD_selected-SWU_selected-LWU_selected, linestyle=' ', marker='o',color = 'black')
#plt.ylabel('Qnet (W/m²)')
#plt.xlabel('time (h)')
