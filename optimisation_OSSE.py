# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:52:00 2019

@author: Bosman Peter
"""
import numpy as np
import copy as cp
import forwardmodel as fwdm
import adjoint_modelling as am
from scipy import optimize
import matplotlib.pyplot as plt
import shutil
import os
import time
import multiprocessing
from joblib import Parallel, delayed


#switches
ana_deriv = True
use_backgr_in_cost = False
write_to_f = True
use_ensemble = False
if use_ensemble:
    nr_of_members = 2
maxnr_of_restarts = 7 #only implemented for tnc method at the moment
imposeparambounds = True
remove_all_prev = True
optim_method = 'tnc' #bfgs or tnc
run_multicore = False #only possible when using ensemble
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

#remove previous files
if remove_all_prev:
    os.system('rm Optimfile*')
    os.system('rm Gradfile*')

#optimisation
priorinput = fwdm.model_input()
priorinput.wCOS       = 0.01
priorinput.COS        = 500
priorinput.COSmeasuring_height = 5.
priorinput.COSmeasuring_height2 = 8.
priorinput.alfa_sto = 1
priorinput.gciCOS = 0.2 /(1.2*1000) * 28.9
priorinput.ags_C_mode = 'surf' 
priorinput.sw_useWilson  = True
priorinput.dt         = 60       # time step [s]
priorinput.runtime    = 4*3600   # total run time [s]
priorinput.sw_ml      = True      # mixed-layer model switch
priorinput.sw_shearwe = False     # shear growth mixed-layer switch
priorinput.sw_fixft   = False     # Fix the free-troposphere switch
priorinput.h          = 600.      # initial ABL height [m]
priorinput.Ps         = 101300.   # surface pressure [Pa]
priorinput.divU       = 0.000001  # horizontal large-scale divergence of wind [s-1]
priorinput.fc         = 1.e-4     # Coriolis parameter [m s-1]
priorinput.theta      = 278.      # initial mixed-layer potential temperature [K]
priorinput.deltatheta = 2       # initial temperature jump at h [K]
priorinput.gammatheta = 0.002     # free atmosphere potential temperature lapse rate [K m-1]
priorinput.advtheta   = 0.        # advection of heat [K s-1]
priorinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
priorinput.wtheta     = 0.4       # surface kinematic heat flux [K m s-1]
priorinput.q          = 0.008     # initial mixed-layer specific humidity [kg kg-1]
priorinput.deltaq     = -0.001    # initial specific humidity jump at h [kg kg-1]
priorinput.gammaq     = 0.        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priorinput.advq       = 0.        # advection of moisture [kg kg-1 s-1]
priorinput.wq         = 0.1e-3    # surface kinematic moisture flux [kg kg-1 m s-1] 
priorinput.CO2        = 422.      # initial mixed-layer CO2 [ppm]
priorinput.deltaCO2   = -44.      # initial CO2 jump at h [ppm]
priorinput.deltaCOS   = 50.      # initial COS jump at h [ppb]
priorinput.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
priorinput.gammaCOS   = 1.        # free atmosphere CO2 lapse rate [ppb m-1]
priorinput.advCO2     = 0.        # advection of CO2 [ppm s-1]
priorinput.advCOS     = 0.        # advection of COS [ppb s-1]
priorinput.wCO2       = 0.        # surface kinematic CO2 flux [ppm m s-1]
priorinput.wCOS       = 0.5        # surface kinematic COS flux [ppb m s-1]
priorinput.sw_wind    = False     # prognostic wind switch
priorinput.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
priorinput.deltau     = 4.        # initial u-wind jump at h [m s-1]
priorinput.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
priorinput.advu       = 0.        # advection of u-wind [m s-2]
priorinput.v          = -4.0      # initial mixed-layer u-wind speed [m s-1]
priorinput.deltav     = 4.0       # initial u-wind jump at h [m s-1]
priorinput.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
priorinput.advv       = 0.        # advection of v-wind [m s-2]
priorinput.sw_sl      = True     # surface layer switch
priorinput.ustar      = 0.3       # surface friction velocity [m s-1]
priorinput.z0m        = 0.02      # roughness length for momentum [m]
priorinput.z0h        = 0.02     # roughness length for scalars [m]
priorinput.sw_rad     = True     # radiation switch
priorinput.lat        = 31.97     # latitude [deg]
priorinput.lon        = 0     # longitude [deg]
priorinput.doy        = 185.      # day of the year [-]
priorinput.tstart     = 10   # time of the day [h UTC]
priorinput.cc         = 0.0       # cloud cover fraction [-]
#priorinput.Q          = 600.      # net radiation [W m-2] 
priorinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priorinput.sw_ls      = True     # land surface switch
priorinput.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
priorinput.wg         = 0.27      # volumetric water content top soil layer [m3 m-3]
priorinput.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
priorinput.cveg       = 0.35      # vegetation fraction [-]
priorinput.Tsoil      = 290.      # temperature top soil layer [K]
priorinput.T2         = 285.      # temperature deeper soil layer [K]
priorinput.a          = 0.219     # Clapp and Hornberger retention curve parameter a
priorinput.b          = 4.90      # Clapp and Hornberger retention curve parameter b
priorinput.p          = 4.        # Clapp and Hornberger retention curve parameter c
priorinput.CGsat      = 3.56e-6   # saturated soil conductivity for heat
priorinput.wsat       = 0.472     # saturated volumetric water content ECMWF config [-]
priorinput.wfc        = 0.323     # volumetric water content field capacity [-]
priorinput.wwilt      = 0.10     # volumetric water content wilting point [-]
priorinput.C1sat      = 0.132     
priorinput.C2ref      = 1.8
priorinput.LAI        = 2.        # leaf area index [-]
priorinput.gD         = 0.0       # correction factor transpiration for VPD [-]
priorinput.rsmin      = 110.      # minimum resistance transpiration [s m-1]
priorinput.rssoilmin  = 50.       # minimun resistance soil evaporation [s m-1]
priorinput.alpha      = 0.45     # surface albedo [-]
priorinput.Ts         = 290.      # initial surface temperature [K]
priorinput.Wmax       = 0.0002    # thickness of water layer on wet vegetation [m]
priorinput.Wl         = 0.0000    # equivalent water layer depth for wet vegetation [m]
priorinput.Lambda     = 5.9       # thermal diffusivity skin layer [-]
priorinput.c3c4       = 'c3'      # Plant type ('c3' or 'c4')
priorinput.sw_cu      = False     # Cumulus parameterization switch
priorinput.dz_h       = 150.      # Transition layer thickness [m]
priorinput.Cs         = 1e12      # drag coefficient for scalars [-]
priorinput.sw_dynamicsl_border = True
priorinput.sw_model_stable_con = True
priorinput.sw_use_ribtol = True
priorinput.sw_advfp = True #prescribed advection to take place over full profile (also in Free troposphere), only in ML if FALSE
priorinput.sw_dyn_beta = True

priorinput.soilCOSmodeltype   = 'Sun_Ogee' #can be set to None or 'Sun_Ogee'
priorinput.uptakemodel = 'Ogee'
priorinput.sw_soilmoisture    = 'simple'
priorinput.sw_soiltemp    = 'simple'
priorinput.kH_type         = 'Sun'
priorinput.Diffus_type     = 'Sun'
priorinput.b_sCOSm = 5.3
priorinput.fCA = 3e4
priorinput.nr_nodes     = 26
priorinput.s_moist_opt  = 0.20
priorinput.Vspmax        = 1.e-10
priorinput.Q10             = 3.
priorinput.layer1_2division = 0.3
priorinput.write_soilCOS_to_f = False
priorinput.nr_nodes_for_filewr = 5

priormodel = fwdm.model(priorinput)
priormodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
#run testmodel to initialise properly
#What works well is the following:
#optim.obs=['h','q']
#state=['theta','h','deltatheta']
#truthinput.deltatheta = 1
#truthinput.theta = 288
#truthinput.h = 400

state=['theta','h','deltatheta','advtheta','advq']
optim = am.adjoint_modelling(priormodel,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,imposeparambounds=True,state=state)
#with 3, 1 and 0.25 for 'alfa_sto','deltatheta','alpha' it finds the truth (1, 2 ,0.2)back
optim.obs=['h','q','Tmh']
#optim.obs=['h']

Hx_prior = {}
for item in optim.obs:
    Hx_prior[item] = priormodel.out.__dict__[item]
checkpoint_prior = priormodel.cpx
checkpoint_init_prior = priormodel.cpx_init

if optim.use_backgr_in_cost or use_ensemble:
    priorvar = {}
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

if optim.use_backgr_in_cost:
    optim.b_cov = np.diag(pars_priorvar)  #b_cov stands for background covariance matrix, b already exists
    optim.binv = np.linalg.inv(optim.b_cov)
          

########################################
#Set the truth
########################################
truthinput = cp.deepcopy(priorinput)
#truthinput.alfa_plant = 1.
#truthinput.alpha = 0.25
truthinput.deltatheta = 1
truthinput.theta = 288
truthinput.h = 400
truthinput.advtheta = 0.0002
truthinput.advq = 0.0000002
#truthinput.gammatheta = 0.006
#truthinput.wg = 0.17
truthmodel = fwdm.model(truthinput)
truthmodel.run(checkpoint=False,updatevals_surf_lay=True)

obs_times = {}
obs_weights = {}
for item in optim.obs:
    obs_times[item] = truthmodel.out.t[::6] * 3600
    if item == 'h':
        optim.__dict__['error_obs_' + item] = [110 for number in range(len(obs_times[item]))]
    if item == 'q':
        optim.__dict__['error_obs_' + item] = [60 for number in range(len(obs_times[item]))]
    if item == 'Tmh':
        optim.__dict__['error_obs_' + item] = [2 for number in range(len(obs_times[item]))]
    optim.__dict__['error_obs_' + item] = np.array(optim.__dict__['error_obs_' + item])
    optim.__dict__['obs_'+item] = truthmodel.out.__dict__[item][::6]    
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
        if round(num, 3) not in [round(num2, 3) for num2 in priormodel.out.t * 3600]:
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
optim.pstate = []
for item in state:
    optim.pstate.append(priorinput.__dict__[item])
optim.pstate = np.array(optim.pstate)
optiminput = cp.deepcopy(priorinput) #deepcopy!
params = tuple([optiminput,state,obs_times,obs_weights])
optim.checkpoint = cp.deepcopy(checkpoint_prior) #needed, as first thing optimizer does is calculating the gradient
optim.checkpoint_init = cp.deepcopy(checkpoint_init_prior) #needed, as first thing optimizer does is calculating the gradient
for item in optim.obs:
    weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable
    k = 0 #counter for the observations (specific for each type of obs)
    for t in range(priormodel.tsteps):
        if round(priormodel.out.t[t] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
            if item in obs_weights:
                weight = obs_weights[item][k]
            forcing = weight * (Hx_prior[item][t]-optim.__dict__['obs_'+item][k])/(optim.__dict__['error_obs_' + item][k]**2)
            optim.forcing[t][item] = forcing
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
    except (am.nan_incostfError):
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
    except (am.static_costfError):
        print('Minimisation aborted as it proceeded too slow')
        state_opt0 = optim.Statelist[-1]
        min_costf = optim.Costflist[-1]
    for i in range(maxnr_of_restarts):
        if (min_costf > stopcrit and (not hasattr(optim,'stop'))): #will evaluate to False if min_costf is equal to nan
            optim.nr_of_restarted_sim = optim.sim_nr #This statement is required in case the optimisation algorithm terminates succesfully, in case of static_costfError it was already ok
            open('Optimfile.txt','a').write('\n')
            open('Optimfile.txt','a').write('{0:>25s}'.format('restart from end'))
            open('Gradfile.txt','a').write('\n')
            open('Gradfile.txt','a').write('{0:>25s}'.format('restart from end'))
            try:
                if ana_deriv:
                    minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from end point to make it better if costf still too large
                else:
                    minimisation = optimize.fmin_tnc(optim.min_func,state_opt0,fprime=optim.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from end point to make it better if costf still too large
                state_opt0 = minimisation[0]
            except (am.nan_incostfError):
                print('Minimisation aborted due to nan, no restart')
                open('Optimfile.txt','a').write('\n')
                open('Optimfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
                open('Gradfile.txt','a').write('\n')
                open('Gradfile.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
                if discard_nan_minims == False:
                    min_costf = np.min(optim.Costflist)
                    min_costf_ind = optim.Costflist.index(min(optim.Costflist)) #find the number of the simulation where costf was minimal
                    state_opt0 = optim.Statelist[min_costf_ind]
                else:
                    state_opt0 = np.array([np.nan for x in range(len(state))])
                    min_costf = np.nan
                break
            except (am.static_costfError):
                print('Minimisation aborted as it proceeded too slow')
                state_opt0 = optim.Statelist[-1]
                min_costf = optim.Costflist[-1]
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
    optim_mem = am.adjoint_modelling(priormodel_mem,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,imposeparambounds=imposeparambounds,state=state,Optimfile='Optimfile'+str(counter)+'.txt',Gradfile='Gradfile'+str(counter)+'.txt')
    optim_mem.obs = cp.deepcopy(optim.obs)
    for item in optim_mem.obs:
        optim_mem.__dict__['obs_'+item] = cp.deepcopy(optim.__dict__['obs_'+item]) #the truth obs
    Hx_prior_mem = {}
    for item in optim_mem.obs:
        Hx_prior_mem[item] = priormodel_mem.out.__dict__[item]
        optim_mem.__dict__['error_obs_' + item] = cp.deepcopy(optim.__dict__['error_obs_' + item])
    checkpoint_prior_mem = priormodel_mem.cpx
    checkpoint_init_prior_mem = priormodel_mem.cpx_init
    if use_backgr_in_cost: 
       optim_mem.binv = cp.deepcopy(optim.binv)
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
        k = 0
        for t in range(priormodel_mem.tsteps):
            if round(priormodel_mem.out.t[t] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
                if item in obs_weights:
                    weight = obs_weights[item][k]
                forcing = weight * (Hx_prior_mem[item][t]-optim_mem.__dict__['obs_'+item][k])/(optim_mem.__dict__['error_obs_' + item][k]**2)
                optim_mem.forcing[t][item] = forcing
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
        except (am.nan_incostfError):
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
        except (am.static_costfError):
            print('Minimisation aborted as it proceeded too slow')
            state_opt_mem = optim_mem.Statelist[-1]
            min_costf_mem = optim_mem.Costflist[-1]
        for i in range(maxnr_of_restarts):
            if (min_costf_mem > stopcrit and (not hasattr(optim_mem,'stop'))): #will evaluate to False if min_costf_mem is equal to nan
                optim_mem.nr_of_restarted_sim = optim_mem.sim_nr
                open('Optimfile'+str(counter)+'.txt','a').write('\n')
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('restart from end'))
                open('Gradfile'+str(counter)+'.txt','a').write('\n')
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('restart from end'))
                try:
                    if ana_deriv:
                        minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.ana_deriv,args=params,bounds=bounds,maxfun=None) #restart from end point to make it better if costf still too large
                    else:
                        minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.num_deriv,args=params,bounds=bounds,maxfun=None) #restart from end point to make it better if costf still too large
                    state_opt_mem = minimisation_mem[0]
                except (am.nan_incostfError):
                    print('Minimisation aborted due to nan, no restart for this member')
                    open('Optimfile'+str(counter)+'.txt','a').write('\n')
                    open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('nan reached, no restart'))
                    if discard_nan_minims == False:
                        min_costf_mem = np.min(optim_mem.Costflist)
                        min_costf_mem_ind = optim_mem.Costflist.index(min(optim_mem.Costflist)) #find the number of the simulation where costf was minimal
                        state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                    else:
                        state_opt_mem = np.array([np.nan for x in range(len(state))])
                        min_costf_mem = np.nan
                    break
                except (am.static_costfError):
                    print('Minimisation aborted as it proceeded too slow')
                    state_opt_mem = optim_mem.Statelist[-1]
                    min_costf_mem = optim_mem.Costflist[-1]
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
    opt_sim_nr = np.where(seq == np.nanmin(seq))[0][0]
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
for i in range(len(optim.obs)):
    fig = plt.figure()
    plt.rc('font', size=16)
    plt.plot(priormodel.out.t*3600,priormodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
    plt.plot(optimalmodel.out.t*3600,optimalmodel.out.__dict__[optim.obs[i]], linestyle=' ', marker='o',color='red',label = 'post')
    plt.plot(obs_times[optim.obs[i]],optim.__dict__['obs_'+optim.obs[i]], linestyle=' ', marker='o',color = 'black')
    plt.ylabel(optim.obs[i])
    plt.xlabel('timestep')
    plt.legend()

