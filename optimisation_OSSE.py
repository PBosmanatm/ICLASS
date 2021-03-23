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
import multiprocessing
from joblib import Parallel, delayed
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
    run_multicore = False #only possible when using ensemble
maxnr_of_restarts = 7 #only implemented for tnc method at the moment
imposeparambounds = True
remove_all_prev = True #Use with caution, be careful for other files in working directory!!
optim_method = 'tnc' #bfgs or tnc
stopcrit = 0.01
perturb_truth_obs = False
if perturb_truth_obs:
    set_seed = True
    if set_seed:
        seedvalue = 17
use_weights = False #weights for the cost function, to enlarge the importance of certain obs
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

#optimisation
priormodinput = fwdm.model_input()
###########################################
###### user input: prior model param ######
###########################################
priormodinput.wCOS       = 0.01
priormodinput.COS        = 500
priormodinput.COSmeasuring_height = 5.
priormodinput.COSmeasuring_height2 = 8.
priormodinput.alfa_sto = 1
priormodinput.gciCOS = 0.2 /(1.2*1000) * 28.9
priormodinput.ags_C_mode = 'surf' 
priormodinput.sw_useWilson  = True
priormodinput.dt         = 60       # time step [s]
priormodinput.runtime    = 4*3600   # total run time [s]
priormodinput.sw_ml      = True      # mixed-layer model switch
priormodinput.sw_shearwe = False     # shear growth mixed-layer switch
priormodinput.sw_fixft   = False     # Fix the free-troposphere switch
priormodinput.h          = 600.      # initial ABL height [m]
priormodinput.Ps         = 101300.   # surface pressure [Pa]
priormodinput.divU       = 0.000001  # horizontal large-scale divergence of wind [s-1]
priormodinput.fc         = 1.e-4     # Coriolis parameter [m s-1]
priormodinput.theta      = 278.      # initial mixed-layer potential temperature [K]
priormodinput.deltatheta = 2       # initial temperature jump at h [K]
priormodinput.gammatheta = 0.002     # free atmosphere potential temperature lapse rate [K m-1]
priormodinput.advtheta   = 0.        # advection of heat [K s-1]
priormodinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
priormodinput.wtheta     = 0.4       # surface kinematic heat flux [K m s-1]
priormodinput.q          = 0.008     # initial mixed-layer specific humidity [kg kg-1]
priormodinput.deltaq     = -0.001    # initial specific humidity jump at h [kg kg-1]
priormodinput.gammaq     = 0.        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priormodinput.advq       = 0.        # advection of moisture [kg kg-1 s-1]
priormodinput.wq         = 0.1e-3    # surface kinematic moisture flux [kg kg-1 m s-1] 
priormodinput.CO2        = 422.      # initial mixed-layer CO2 [ppm]
priormodinput.deltaCO2   = -44.      # initial CO2 jump at h [ppm]
priormodinput.deltaCOS   = 50.      # initial COS jump at h [ppb]
priormodinput.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
priormodinput.gammaCOS   = 1.        # free atmosphere CO2 lapse rate [ppb m-1]
priormodinput.advCO2     = 0.        # advection of CO2 [ppm s-1]
priormodinput.advCOS     = 0.        # advection of COS [ppb s-1]
priormodinput.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]
priormodinput.wCOS       = 0.5        # surface kinematic COS flux [ppb m s-1]
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
priormodinput.z0m        = 0.02      # roughness length for momentum [m]
priormodinput.z0h        = 0.02     # roughness length for scalars [m]
priormodinput.sw_rad     = True     # radiation switch
priormodinput.lat        = 31.97     # latitude [deg]
priormodinput.lon        = 0     # longitude [deg]
priormodinput.doy        = 185.      # day of the year [-]
priormodinput.tstart     = 10   # time of the day [h UTC]
priormodinput.cc         = 0.0       # cloud cover fraction [-]
#priormodinput.Q          = 600.      # net radiation [W m-2] 
priormodinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priormodinput.sw_ls      = True     # land surface switch
priormodinput.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
priormodinput.wg         = 0.27      # volumetric water content top soil layer [m3 m-3]
priormodinput.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
priormodinput.cveg       = 0.35      # vegetation fraction [-]
priormodinput.Tsoil      = 290.      # temperature top soil layer [K]
priormodinput.T2         = 285.      # temperature deeper soil layer [K]
priormodinput.a          = 0.219     # Clapp and Hornberger retention curve parameter a
priormodinput.b          = 4.90      # Clapp and Hornberger retention curve parameter b
priormodinput.p          = 4.        # Clapp and Hornberger retention curve parameter c
priormodinput.CGsat      = 3.56e-6   # saturated soil conductivity for heat
priormodinput.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
priormodinput.wfc        = 0.323     # volumetric water content field capacity [-]
priormodinput.wwilt      = 0.10     # volumetric water content wilting point [-]
priormodinput.C1sat      = 0.132     
priormodinput.C2ref      = 1.8
priormodinput.LAI        = 2.        # leaf area index [-]
priormodinput.gD         = 0.0       # correction factor transpiration for VPD [-]
priormodinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]
priormodinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priormodinput.alpha      = 0.45     # surface albedo [-]
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
priormodinput.sw_use_ribtol = True
priormodinput.sw_advfp = True #prescribed advection to take place over full profile (also in Free troposphere), only in ML if FALSE
priormodinput.sw_dyn_beta = True

priormodinput.soilCOSmodeltype   = 'Sun_Ogee' #can be set to None or 'Sun_Ogee'
priormodinput.uptakemodel = 'Ogee'
priormodinput.sw_soilmoisture    = 'simple'
priormodinput.sw_soiltemp    = 'simple'
priormodinput.kH_type         = 'Sun'
priormodinput.Diffus_type     = 'Sun'
priormodinput.b_sCOSm = 5.3
priormodinput.fCA = 3e4
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



#run testmodel to initialise properly
#What works well is the following:
#optim.obs=['h','q']
#state=['theta','h','deltatheta']
#truthinput.deltatheta = 1
#truthinput.theta = 288
#truthinput.h = 400

######################################################################
###### user input: obs scales,state and list of used pseudo-obs ######
######################################################################

state=['theta','h','deltatheta','advtheta','advq']
obslist=['h','q','Tmh']
#with 3, 1 and 0.25 for 'alfa_sto','deltatheta','alpha' it finds the truth (1, 2 ,0.2)back

##########################################################################
###### end user input: obs scales,state and list of used pseudo-obs ######
##########################################################################

if use_backgr_in_cost or use_ensemble:
    priorvar = {} 
###########################################################
###### user input: prior information (if used) ############
###########################################################
    #if not optim.use_backgr_in_cost, than these are only used for perturbing the ensemble (when use_ensemble = True)
    #prior variances of the items in the state: 
    priorvar['alpha'] = 0.2**2
    priorvar['gammatheta'] = 0.003**2 
    priorvar['deltatheta'] = 0.75**2
    priorvar['theta'] = 2**2
    priorvar['h'] = 200**2
    priorvar['wg'] = 0.15**2
    priorvar['advtheta'] = 0.0005**2
    priorvar['advq'] = 0.0000005**2
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
truthinput = cp.deepcopy(priorinput)
###############################################
###### user input: set the 'truth' ############
###############################################
#Items not specified here are taken over from priorinput
#truthinput.alfa_plant = 1.
#truthinput.alpha = 0.25
truthinput.deltatheta = 1
truthinput.theta = 288
truthinput.h = 400
truthinput.advtheta = 0.0002
truthinput.advq = 0.0000002
#truthinput.gammatheta = 0.006
#truthinput.wg = 0.17
###################################################
###### end user input: set the 'truth' ############
###################################################
#run the model with 'true' parameters
truthmodel = fwdm.model(truthinput)
truthmodel.run(checkpoint=False,updatevals_surf_lay=True)

#The pseudo observations
obs_times = {}
obs_weights = {}
obs_units = {}
display_names = {}
for item in optim.obs:
##################################################################
###### user input: pseudo-observation information ################
##################################################################
    #for each of the variables provided in the observation list, link the model output variable 
    #to the correct observations that were read in. Also, specify the times and observational errors, and optional weights 
    #Optionally, you can provide a display name here, a name which name will be shown for the observations in the plots
    #please use np.array or list as datastructure for the obs, obs errors, observation times or weights
    obs_times[item] = truthmodel.out.t[::6] * 3600
    optim.__dict__['obs_'+item] = truthmodel.out.__dict__[item][::6]
    if item == 'h':
        optim.__dict__['error_obs_' + item] = [110 for number in range(len(obs_times[item]))]
        if use_weights:
            obs_weights[item] = [1.0 for j in range(len(optim.__dict__['obs_'+item]))]
    if item == 'q':
        optim.__dict__['error_obs_' + item] = [60 for number in range(len(obs_times[item]))]
    if item == 'Tmh':
        optim.__dict__['error_obs_' + item] = [2 for number in range(len(obs_times[item]))]
        
    #Here we account for possible scaling factors for the observations
    if hasattr(truthinput,'obs_sca_cf_'+item):
        optim.__dict__['obs_'+item] /= truthinput.__dict__['obs_sca_cf_'+item]
    
######################################################################
###### end user input: pseudo-observation information ################
######################################################################
    if (not hasattr(optim,'obs_'+item) or not hasattr(optim,'error_obs_'+item)): #a check to see wether all info is specified
        raise Exception('Incomplete or no information on obs of ' + item)
    optim.__dict__['error_obs_' + item] = np.array(optim.__dict__['error_obs_' + item])
    if perturb_truth_obs:
        if set_seed:
            np.random.seed(seedvalue)
        else:
            np.random.seed(None)
        rand_nr_list = ([np.random.normal(0,optim.__dict__['error_obs_' + item][i]) for i in range(len(optim.__dict__['error_obs_' + item]))])
        optim.__dict__['obs_'+item] += rand_nr_list
    if (use_backgr_in_cost and use_weights): #add weight of obs vs prior (identical for every obs) in the cost function
        if item in obs_weights: #if already a weight specified for the specific type of obs
            obs_weights[item] = [x * obs_vs_backgr_weight for x in obs_weights[item]]
        else:
            obs_weights[item] = [obs_vs_backgr_weight for x in range(len(optim.__dict__['obs_'+item]))]         
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
    k = 0 #counter for the observations (specific for each type of obs)
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
            optim.nr_of_sim_bef_restart = optim.sim_nr #This statement is required in case the optimisation algorithm terminates succesfully, in case of static_costfError it was already ok
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
    priormodel_mem.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
    optim_mem = im.adjoint_modelling(priormodel_mem,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,imposeparambounds=imposeparambounds,state=state,Optimfile='Optimfile'+str(counter)+'.txt',Gradfile='Gradfile'+str(counter)+'.txt',pri_err_cov_matr=b_cov)
    optim_mem.obs = cp.deepcopy(optim.obs)
    for item in optim_mem.obs:
        optim_mem.__dict__['obs_'+item] = cp.deepcopy(optim.__dict__['obs_'+item]) #the truth obs
    Hx_prior_mem = {}
    for item in optim_mem.obs:
        Hx_prior_mem[item] = priormodel_mem.out.__dict__[item]
        optim_mem.__dict__['error_obs_' + item] = cp.deepcopy(optim.__dict__['error_obs_' + item])
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
        k = 0
        for ti in range(priormodel_mem.tsteps):
            if round(priormodel_mem.out.t[ti] * 3600,10) in [round(num, 10) for num in obs_times[item]]: #so if we are at a time where we have an obs
                if item in obs_weights:
                    weight = obs_weights[item][k]
                forcing = weight * (Hx_prior_mem[item][ti] - obs_scale * optim_mem.__dict__['obs_'+item][k])/(optim_mem.__dict__['error_obs_' + item][k]**2)
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
    print('ensemble:')
    print(ensemble)
    seq = np.array([x['min_costf'] for x in ensemble]) #iterate over the dictionaries
    min_costf_ensemble = np.nanmin(seq)
    opt_sim_nr = np.where(seq == min_costf_ensemble)[0][0]
    state_opt = ensemble[opt_sim_nr]['state_opt']
    print('index of member with the best state:')
    print(opt_sim_nr)


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

for i in range(len(optim.obs)):
    fig = plt.figure()
    plt.plot(priormodel.out.t-priorinput.tstart,priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(optimalmodel.out.t-priorinput.tstart,optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red',label = 'post')
    plt.plot(obs_times[optim.obs[i]]/3600-priorinput.tstart,optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='x',color = 'black',label = 'obs')
    plt.ylabel(optim.obs[i])
    plt.xlabel('time (h)')
    plt.legend()
    plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    if write_to_f:
        plt.savefig('fig_fit_'+optim.obs[i]+'.eps', format='eps')

########################################################
###### user input: additional plotting etc. ############
########################################################  
for i in range(len(optim.obs)):
    fig = plt.figure()
    plt.plot(priormodel.out.t*3600,priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(optimalmodel.out.t*3600,optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red',label = 'post')
    plt.plot(obs_times[optim.obs[i]],optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black')
    plt.ylabel(optim.obs[i])
    plt.xlabel('timestep')
    plt.legend()

