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
import glob
import matplotlib.style as style
import pickle
style.use('classic')


##################################
###### user input: settings ######
##################################
ana_deriv = True #use analytical or numerical derivative
use_backgr_in_cost = False #include the background (prior) part of the cost function
write_to_f = True #write output and figures to files
use_ensemble = False #use an ensemble of optimisations
if use_ensemble:
    nr_of_members = 2 #number of members in the ensemble of optimisations (including the one with unperturbed prior)
    use_covar_to_pert = False #whether to take prior covariance (if specified) into account when perturbing the ensemble 
    pert_non_state_param = True #perturb parameters that are not in the state
    est_post_pdf_covmatr = True #estimate the posterior pdf and covariance matrix of the state (and more)
    if est_post_pdf_covmatr:
        nr_bins = 3 #nr of bins for the pdfs
        succes_opt_crit = 6 #the chi squared at which an optimisation is considered successfull (lower or equal to is succesfull)
    pert_obs_ens = False #Perturb observations of every ensemble member (except member 0)
    if pert_obs_ens:
        use_sigma_O = False #If True, the total observational error is used to perturb the obs, if False only the measurement error is used
        plot_perturbed_obs = True #Plot the perturbed observations of the ensemble members
    pert_Hx_min_sy_ens = True #Perturb the data part of the cost function (in every ensemble member except member 0), by perturbing H(x) - sy with a random number from a distribution with standard deviation sigma_O
    print_status_dur_ens = False #whether to print state etc info during ensemble of optimisations (during member 0 printing will always take place)
estimate_model_err = False #estimate the model error by perturbing specified non-state parameters
if estimate_model_err:
    nr_of_members_moderr = 30 #number of members for the ensemble that estimates the model error
imposeparambounds = True #force the optimisation to keep parameters within specified bounds (tnc only) and when using ensemble, keep priors within bounds (tnc and bfgs)
paramboundspenalty = False #add a penalty to the cost function when parameter bounds exceeded in the optimisation
if paramboundspenalty:
    setNanCostfOutBoundsTo0 = True #when cost function becomes nan when params outside specified bounds, set cost func to zero before adding penalty (nan + number gives nan)
    penalty_exp = 60 #exponent to use in the penalty function (see manual)
remove_prev = True #Use with caution, be careful for other files in working directory! Removes certain files that might have remained from previous optimisations. See manual for more info on what files are removed
abort_slow_minims = True #Abort minimisations that proceed too slow (can be followed by a restart)
optim_method = 'tnc' #bfgs or tnc, the chosen optimisation algorithm. tnc recommended
if optim_method == 'tnc':
    maxnr_of_restarts = 2 #The maximum number of times to restart the optimisation if the cost function is not as low as specified in stopcrit. Only implemented for tnc method at the moment. 
    if maxnr_of_restarts > 0:
        stopcrit = 0.00001#If the cost function is equal or lower than this value, no restart will be attempted   
elif optim_method == 'bfgs':
    gtol = 1e-05 # A parameter for the bfgs algorithm. From scipy documentation: 'Gradient norm must be less than gtol before successful termination.'
perturb_truth_obs = False #Perturb the 'true' observations. When using ensemble, obs of members will be perturbed twice, member 0 just once
if use_ensemble or estimate_model_err:
    run_multicore = False  #Run part of the code on multiple cores simultaneously
    if run_multicore:
        max_num_cores = 'all' #'all' to use all available cores, otherwise specify an integer (without quotation marks)
if (perturb_truth_obs or (use_ensemble or estimate_model_err)):
    set_seed = True #Set the seed in case the output should be reproducable
    if set_seed:
        seedvalue = 14 #the chosen value of the seed. No floating point numbers and no negative numbers 
discard_nan_minims = False #if False, if in a minimisation nan is encountered, it will use the state from the best simulation so far, if True, the minimisation will result in a state with nans 
use_weights = False #weights for the cost function, to enlarge or reduce the importance of certain obs, or to modify the relative importance of the obs vs the background part
if use_weights:
    weight_morninghrs = 1/4 #to change weights of obs in the morning (the hour at which the morning ends is specified in variable 'end_morninghrs'), when everything less well mixed. 1 means equal weights compared to the other parts of the day
    end_morninghrs = 10 #At all times smaller than this time (UTC, decimal hour), weight_morninghrs is applied
if (use_backgr_in_cost and use_weights):
    obs_vs_backgr_weight = 1.0 # a scaling factor for the importance of all the observations in the cost function 
if write_to_f:
    wr_obj_to_pickle_files = True #write certain variables to files for possible postprocessing later
    figformat = 'eps' #the format in which you want figure output, e.g. 'png'
plot_errbars = False #plot error bars in the figures
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
    if est_post_pdf_covmatr and not (pert_obs_ens or pert_Hx_min_sy_ens):
        raise Exception('est_post_pdf_covmatr is set to True, but both switches pert_obs_ens and pert_Hx_min_sy_ens are set to False')
    if pert_Hx_min_sy_ens and pert_obs_ens:
        raise Exception('pert_Hx_min_sy_ens and pert_obs_ens should not both be set to True')
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
        vars_to_pickle = ['priormodel','priorinput','obsvarlist','disp_units','disp_units_par','display_names','disp_nms_par','optim','obs_times','measurement_error','optimalinput','optimalinput_onsp','optimalmodel','optimalmodel_onsp','PertData_mems'] #list of strings      
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

#optimisation
priormodinput = fwdm.model_input()
###########################################
###### user input: prior model param ######
###########################################
priormodinput.wCOS       = 0.01
priormodinput.COS        = 500
priormodinput.COSmeasuring_height = 5.
priormodinput.COSmeasuring_height2 = 8.
priormodinput.CO2measuringheight = 20.
priormodinput.Tmeasuringheight = 2.
priormodinput.sca_sto = 0.5
priormodinput.gciCOS = 0.2 /(1.2*1000) * 28.9
priormodinput.ags_C_mode = 'MXL' 
priormodinput.sw_useWilson  = False
priormodinput.dt         = 60       # time step [s]
priormodinput.runtime    = 4*3600   # total run time [s]
priormodinput.sw_ml      = True      # mixed-layer model switch
priormodinput.sw_shearwe = True     # shear growth mixed-layer switch
priormodinput.sw_fixft   = False     # Fix the free-troposphere switch
priormodinput.h          = 650.      # initial ABL height [m]
priormodinput.Ps         = 101300.   # surface pressure [Pa]
priormodinput.divU       = 0.00  # horizontal large-scale divergence of wind [s-1]
priormodinput.fc         = 1.e-4     # Coriolis parameter [m s-1]
priormodinput.theta      = 282.      # initial mixed-layer potential temperature [K]
priormodinput.deltatheta = 2       # initial temperature jump at h [K]
priormodinput.gammatheta = 0.005     # free atmosphere potential temperature lapse rate [K m-1]
priormodinput.advtheta   = 0.        # advection of heat [K s-1]
priormodinput.beta       = 0.2       # entrainment ratio for virtual heat [-]
priormodinput.wtheta     = 0.1       # surface kinematic heat flux [K m s-1]
priormodinput.q          = 0.008     # initial mixed-layer specific humidity [kg kg-1]
priormodinput.deltaq     = -0.001    # initial specific humidity jump at h [kg kg-1]
priormodinput.gammaq     = -0.001e-3        # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
priormodinput.advq       = 0.        # advection of moisture [kg kg-1 s-1]
priormodinput.wq         = 0.1e-3    # surface kinematic moisture flux [kg kg-1 m s-1] 
priormodinput.CO2        = 422.      # initial mixed-layer CO2 [ppm]
priormodinput.deltaCO2   = -44.      # initial CO2 jump at h [ppm]
priormodinput.deltaCOS   = 50.      # initial COS jump at h [ppb]
priormodinput.gammaCO2   = 0.        # free atmosphere CO2 lapse rate [ppm m-1]
priormodinput.gammaCOS   = 1.        # free atmosphere COS lapse rate [ppb m-1]
priormodinput.advCO2     = 0.        # advection of CO2 [ppm s-1]
priormodinput.advCOS     = 0.        # advection of COS [ppb s-1]
priormodinput.wCO2       = 0.        # surface total CO2 flux [mgCO2 m-2 s-1]
priormodinput.wCOS       = 0.5        # surface kinematic COS flux [ppb m s-1]
priormodinput.sw_wind    = False     # prognostic wind switch
priormodinput.u          = 6.        # initial mixed-layer u-wind speed [m s-1]
priormodinput.deltau     = 4.        # initial u-wind jump at h [m s-1]
priormodinput.gammau     = 0.        # free atmosphere u-wind speed lapse rate [s-1]
priormodinput.advu       = 0.        # advection of u-wind [m s-2]
priormodinput.v          = -4.0      # initial mixed-layer v-wind speed [m s-1]
priormodinput.deltav     = 4.0       # initial v-wind jump at h [m s-1]
priormodinput.gammav     = 0.        # free atmosphere v-wind speed lapse rate [s-1]
priormodinput.advv       = 0.        # advection of v-wind [m s-2]
priormodinput.sw_sl      = True     # surface layer switch
priormodinput.ustar      = 0.3       # surface friction velocity [m s-1]
priormodinput.z0m        = 0.02      # roughness length for momentum [m]
priormodinput.z0h        = 0.02     # roughness length for scalars [m]
priormodinput.sw_rad     = True     # radiation switch
priormodinput.lat        = 41.97     # latitude [deg]
priormodinput.lon        = 0     # longitude [deg]
priormodinput.doy        = 185.      # day of the year [-]
priormodinput.tstart     = 10   # time of the day [h UTC]
priormodinput.cc         = 0.0       # cloud cover fraction [-]
#priormodinput.Q          = 600.      # net radiation [W m-2] 
priormodinput.dFz        = 0.        # cloud top radiative divergence [W m-2] 
priormodinput.sw_ls      = True     # land surface switch
priormodinput.ls_type    = 'ags'     # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
priormodinput.wg         = 0.14      # volumetric water content top soil layer [m3 m-3]
priormodinput.w2         = 0.21      # volumetric water content deeper soil layer [m3 m-3]
priormodinput.cveg       = 0.85      # vegetation fraction [-]
priormodinput.Tsoil      = 282.      # temperature top soil layer [K]
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
priormodinput.Ts         = 282.      # initial surface temperature [K]
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

#soil COS model
priormodinput.soilCOSmodeltype   = None #can be set to None or 'Sun_Ogee'
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
priorinput = cp.deepcopy(priormodinput) 



#run testmodel to initialise properly
#What works well is the following:
#obsvarlist=['h','q']
#state=['theta','h','deltatheta']
#truthinput.deltatheta = 1
#truthinput.theta = 288
#truthinput.h = 400

#################################################################################
###### user input: state, list of used pseudo-obs and non-model priorinput ######
#################################################################################
state=['h','alpha','sca_sto','wg','gammatheta']
obsvarlist=['h','q']#
#below we can add some input necessary for the state in the optimisation, that is not part of the model input (a scale for some of the observations in the costfunction if desired). Or FracH
if 'FracH' in state:
    priorinput.FracH = 0.6

#####################################################################################
###### end user input: state, list of used pseudo-obs and non-model priorinput ######
#####################################################################################
if len(set(state)) != len(state):
    raise Exception('Mulitiple occurences of same item in state')
if len(set(obsvarlist)) != len(obsvarlist):
    raise Exception('Mulitiple occurences of same item in obsvarlist')
if ('FracH' in state and ('obs_sca_cf_H' in state or 'obs_sca_cf_LE' in state)):
    raise Exception('When FracH in state, you cannot include obs_sca_cf_H or obs_sca_cf_LE in state as well')
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
    elif item == 'FracH' and (item not in state):
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
    priorvar['alpha'] = 0.2**2
    priorvar['gammatheta'] = 0.003**2 
    priorvar['gammaq'] = (0.003e-3)**2 
    priorvar['deltatheta'] = 0.75**2
    priorvar['theta'] = 2**2
    priorvar['h'] = 200**2
    priorvar['wg'] = 0.15**2
#    priorvar['obs_sca_cf_Ts'] = 0.4**2
    priorvar['advtheta'] = 0.0005**2
    priorvar['advq'] = 0.0000005**2
    priorvar['FracH'] = 0.3**2
    #below we can specify covariances as well, for the background information matrix. If covariances are not specified, they are taken as 0
    #e.g. priorcovar['gammatheta,gammaq'] = 5.
#    from scipy.stats import truncnorm
#    priorvar_norm['alpha'] = 0.2**2
#    priorvar['alpha'] = truncnorm.stats(optim.boundedvars['alpha'][0], optim.boundedvars['alpha'][1], loc=priorinput.alpha, scale=np.sqrt(priorvar_norm['alpha']), moments=’v’)
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
#    boundedvars['deltatheta'] = [0.2,7] #lower and upper bound
#    boundedvars['deltaCO2'] = [-200,200]
#    boundedvars['deltaq'] = [-0.008,0.008]
#    boundedvars['alpha'] = [0.05,0.8] 
#    boundedvars['sca_sto'] = [0.1,5]
#    boundedvars['wg'] = [priorinput.wwilt+0.001,priorinput.wsat-0.001]
#    boundedvars['theta'] = [274,310]
#    boundedvars['h'] = [50,3200]
#    boundedvars['wtheta'] = [0.05,0.6]
#    boundedvars['gammatheta'] = [0.002,0.018]
##    boundedvars['gammatheta2'] = [0.002,0.018]
#    boundedvars['gammaq'] = [-9e-6,9e-6]
#    boundedvars['z0m'] = [0.0001,5]
#    boundedvars['z0h'] = [0.0001,5]
#    boundedvars['q'] = [0.002,0.020]
#    boundedvars['divU'] = [0,1e-4]
#    boundedvars['fCA'] = [0.1,1e8]
#    boundedvars['CO2'] = [100,1000]
#    boundedvars['ustar'] = [0.01,50]
#    boundedvars['wq'] = [0,0.1] #negative flux seems problematic because L going to very small values
##    boundedvars['FracH'] = [0,1]
#############################################################
###### end user input: parameter bounds  ####################
#############################################################  
    for param in boundedvars:    
        if not hasattr(priorinput,param):
            raise Exception('Parameter \''+ param + '\' in boundedvars does not occur in priorinput')

#create inverse modelling framework, do check,...          
optim = im.inverse_modelling(priormodel,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,StateVarNames=state,obsvarlist=obsvarlist,
                             pri_err_cov_matr=b_cov,paramboundspenalty=paramboundspenalty,abort_slow_minims=abort_slow_minims,boundedvars=boundedvars)
Hx_prior = {}
for item in obsvarlist:
    Hx_prior[item] = priormodel.out.__dict__[item]
truthinput = cp.deepcopy(priorinput)
###############################################
###### user input: set the 'truth' ############
###############################################
#Items not specified here are taken over from priorinput
#truthinput.alfa_plant = 1.
truthinput.alpha = 0.20
#truthinput.deltatheta = 1
#truthinput.theta = 288
truthinput.h = 350
truthinput.sca_sto = 1.0
truthinput.gammatheta = 0.003
truthinput.wg = 0.27
#truthinput.advtheta = 0.0002
#truthinput.advq = 0.0000002
#truthinput.gammatheta = 0.006
#truthinput.wg = 0.17
if 'FracH' in state:
    truthinput.FracH = 0.35
###################################################
###### end user input: set the 'truth' ############
###################################################
#run the model with 'true' parameters
truthmodel = fwdm.model(truthinput)
truthmodel.run(checkpoint=False,updatevals_surf_lay=True)

#The pseudo observations
obs_times = {}
obs_weights = {}
disp_units = {}
display_names = {}
measurement_error = {}
for item in obsvarlist:
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
        measurement_error[item] = [100 for number in range(len(obs_times[item]))]
        if use_weights:
            obs_weights[item] = [1.0 for j in range(len(optim.__dict__['obs_'+item]))]
    if item == 'q':
        measurement_error[item] = [0.0004 for number in range(len(obs_times[item]))]
        disp_units[item] = 'g/kg'
    if item == 'Tmh':
        measurement_error[item] = [0.65 for number in range(len(obs_times[item]))]
    if item == 'Ts':
        measurement_error[item] = [2 for number in range(len(obs_times[item]))]
    if item == 'H':
        measurement_error[item] = [25 for number in range(len(obs_times[item]))]
    if item == 'LE':
        measurement_error[item] = [25 for number in range(len(obs_times[item]))]
    if item == 'wCO2':
        measurement_error[item] = [0.04 for number in range(len(obs_times[item]))]
    if item == 'CO2mh':
        measurement_error[item] = [2 for number in range(len(obs_times[item]))]
        
    #Here we account for possible scaling factors for the observations
    if hasattr(truthinput,'obs_sca_cf_'+item):
        optim.__dict__['obs_'+item] /= truthinput.__dict__['obs_sca_cf_'+item]
    
######################################################################
###### end user input: pseudo-observation information ################
######################################################################
if use_ensemble:
    if est_post_pdf_covmatr:
        disp_units_par = {}  
        disp_nms_par = {}           
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
        disp_units_par['sca_sto'] = '-'
        disp_units_par['alpha'] = '-'
        disp_units_par['FracH'] = '-'
        disp_units_par['wg'] = '-'
        
#        disp_nms_par['theta'] = r'$\theta$' #name for parameter theta
#        disp_nms_par['advtheta'] = r'$adv_{\theta}$'
#        disp_nms_par['advq'] = '$adv_{q}$'
#        disp_nms_par['advCO2'] = '$adv_{CO2}$'
#        disp_nms_par['deltatheta'] = r'$\Delta_{\theta}$'
#        disp_nms_par['gammatheta'] = r'$\gamma_{\theta}$'
#        disp_nms_par['deltaq'] = '$\Delta_{q}$'
#        disp_nms_par['gammaq'] = '$\gamma_{q}$'
#        disp_nms_par['deltaCO2'] = '$\Delta_{CO2}$'
#        disp_nms_par['deltaCO2'] = '$\Delta_{CO2}$'
#        disp_nms_par['gammaCO2'] = '$\gamma_{CO2}$'
#        disp_nms_par['sca_sto'] = r'$\alpha_{sto}$'
#        disp_nms_par['alpha'] = r'$\alpha_{rad}$'
#        disp_nms_par['FracH'] = '$Frac_{H}$'
#        disp_nms_par['wg'] = '$w_{g}$'
#        disp_nms_par['R10'] = '$R_{10}$'
    
##############################################################################
###### end user input: units of parameters for pdf figures (optional) ########
##############################################################################    
    
if 'FracH' in state:  
##################################################################
###### user input: energy balance information (if used) ##########
##################################################################
    #If H in obsvarlist, specify optim.EnBalDiffObs_atHtimes; If LE in obsvarlist, specify optim.EnBalDiffObs_atLEtimes. optim.EnBalDiffObs_atHtimes is the energy balance gap at the observation times of H
    obs_times['H'] = truthmodel.out.t[::4] * 3600
    optim.obs_H = truthmodel.out.H[::4]
    measurement_error['H'] = [30 for number in range(len(obs_times['H']))]
    optim.EnBalDiffObs_atHtimes = 0.25 * (truthmodel.out.H[::4] + truthmodel.out.LE[::4] + truthmodel.out.G[::4])
    optim.EnBalDiffObs_atLEtimes = 0.25 * (truthmodel.out.H[::6] + truthmodel.out.LE[::6] + truthmodel.out.G[::6])
    optim.obs_H = optim.obs_H - truthinput.FracH * optim.EnBalDiffObs_atHtimes  
    optim.obs_LE = optim.obs_LE - (1 - truthinput.FracH) * optim.EnBalDiffObs_atLEtimes 
            
##################################################################
###### end user input: energy balance information (if used) ######
##################################################################
    for item in ['H','LE']:
        if item in obsvarlist:
            if not hasattr(optim,'EnBalDiffObs_at'+item+'times'):
                raise Exception('When including FracH in state and '+ item + ' in obsvarlist, \'optim.EnBalDiffObs_at'+item+'times\' should be specified!')
            if len(optim.__dict__['EnBalDiffObs_at'+item+'times']) != len(optim.__dict__['obs_'+item]):
                raise Exception('When including FracH in state and '+ item + ' in obsvarlist, an EnBalDiffObs_at' +item+'times value should correspond to every obs of ' + item)
            if type(optim.__dict__['EnBalDiffObs_at'+item+'times']) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert EnBalDiffObs_at'+item+'times data into type \'numpy.ndarray\' or list!')

mod_error = {} #model error
repr_error = {} #representation error, see eq 11.11 in chapter inverse modelling Brasseur and Jacob 2017
if estimate_model_err:  
    me_paramdict = {} #dictionary of dictionaries, me means model error
########################################################################
###### user input: model and representation error ######################
########################################################################
    #in case the model error is estimated with a model ensemble (switch estimate_model_err), specify here the parameters to perturb for this estimation
    #and the distributions to sample random numbers from (to add to these parameters in the ensemble):
    me_paramdict['cveg'] = {'distr':'uniform','leftbound': 0.1,'rightbound': 1.0}
    me_paramdict['Lambda'] = {'distr':'normal','scale': 0.3}
else:
    pass
    #in case you want to specify directly the model errors (estimate_model_err = False), specify them here:
    #e.g. mod_error['theta'] = [0.5 for j in range(len(measurement_error['theta']))]
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

orig_obs = {} #if perturb_truth_obs we perturb the obs also without ensemble
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
        raise Exception('Error: size of obs and obstimes inconsistent for '+item+'!')
    if len(obs_times[item]) != len(measurement_error[item]):
        raise Exception('Error: size of measurement_error and obstimes inconsistent for '+item+'!')
    if len(obs_times[item]) != len(repr_error[item]):
        raise Exception('Error: size of repr_error and obstimes inconsistent for '+item+'!')
    if use_weights and item in obs_weights:
        if len(obs_times[item]) != len(obs_weights[item]):
            raise Exception('Error: size of weights and obstimes inconsistent for '+item+'!')
    if len(set([round(num2, 8) for num2 in obs_times[item]])) != len([round(num2, 8) for num2 in obs_times[item]]):
        raise Exception('Error: Observation times of '+item +', rounded to 8 decimal places, are not unique!')
    itoremove = []
    for i in range(len(optim.__dict__['obs_'+item])):
        if np.isnan(optim.__dict__['obs_'+item][i]):
            itoremove += [i]
    if 'FracH' in state and item in ['H','LE']: #Check also for a nan in optim.__dict__['EnBalDiffObs_at'+item+'times'], than the corresponding obs are discarded as well
        for j in range(len(optim.__dict__['obs_'+item])):
            if np.isnan(optim.__dict__['EnBalDiffObs_at'+item+'times'][j]):
                if j not in itoremove:
                    itoremove += [j]
    optim.__dict__['obs_'+item] = np.delete(optim.__dict__['obs_'+item],itoremove) #exclude the nan obs
    measurement_error[item] = np.delete(measurement_error[item],itoremove) #as a side effect, this turns the array into an numpy.ndarray if not already the case (or gives error).
    if not estimate_model_err:
        if item in mod_error:
            if type(mod_error[item]) not in [np.ndarray,list]: #a check to see whether data is of a correct type
                raise Exception('Please convert mod_error data of '+item+' into type \'numpy.ndarray\' or list!')
            if len(obs_times[item]) != len(mod_error[item]):
                raise Exception('Error: size of mod_error and obstimes inconsistent for '+item+'!')
            mod_error[item] = np.delete(mod_error[item],itoremove)
    repr_error[item] = np.delete(repr_error[item],itoremove)
    obs_times[item] = np.delete(obs_times[item],itoremove)#exclude the times,errors and weights as well (of the nan obs)
    if item in obs_weights:
        obs_weights[item] = np.delete(obs_weights[item],itoremove)
    if 'FracH' in state and item in ['H','LE']: #Remove the necessary entries in optim.__dict__['EnBalDiffObs_at'+item+'times'] as well
        optim.__dict__['EnBalDiffObs_at'+item+'times'] = np.delete(optim.__dict__['EnBalDiffObs_at'+item+'times'],itoremove) #exclude the nan obs. If a nan occurs in LE, the EnBalDiffObs_atLEtimes value
            #at the time of the nan in LE will be discarded as well. 
    orig_obs[item] = cp.deepcopy(optim.__dict__['obs_'+item])
    if perturb_truth_obs:
        if set_seed:
            if use_ensemble:
                seedvalue_pto = seedvalue + 2 * nr_of_members #to make sure this seed value does not occur in the ensemble as well
            else:
                seedvalue_pto = seedvalue
            np.random.seed(seedvalue_pto)
        else:
            np.random.seed(None)
        rand_nr_list = ([np.random.normal(0,measurement_error[item][i]) for i in range(len(measurement_error[item]))])
        optim.__dict__['obs_'+item] += rand_nr_list
    if (use_backgr_in_cost and use_weights): #add weight of obs vs prior (identical for every obs) in the cost function
        if item in obs_weights: #if already a weight specified for the specific type of obs
            obs_weights[item] = [x * obs_vs_backgr_weight for x in obs_weights[item]]
        else:
            obs_weights[item] = [obs_vs_backgr_weight for x in range(len(optim.__dict__['obs_'+item]))]         
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
        if round(num, 8) not in [round(num2, 8) for num2 in priormodel.out.t * 3600]:
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
            if item not in disp_nms_par:
                disp_nms_par[item] = item
        if pert_non_state_param:
            for item in non_state_paramdict:
                if item not in disp_units_par:
                    disp_units_par[item] = ''
                if item not in disp_nms_par:
                    disp_nms_par[item] = item

if estimate_model_err:  
    for param in me_paramdict:    
        if not hasattr(priorinput,param):
            raise Exception('Parameter \''+ param + '\' in me_paramdict for estimating the model error does not occur in priorinput')
    def run_mod_pert_par(counter,seed,modelinput,paramdict,obsvarlist,obstimes):
        modelinput_mem = cp.deepcopy(modelinput) 
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
                if round(model_mem.out.t[t]*3600, 8) in [round(num2, 8) for num2 in obstimes[item]]:
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
    optim.__dict__['error_obs_' + item] = np.sqrt(np.array(measurement_error[item])**2 + np.array(mod_error[item])**2 + np.array(repr_error[item])**2) #Eq 11.13 of Brasseur and Jacob 2017




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
        if 'FracH' in state:
            if item not in ['H','LE']:
                observations_item = optim.__dict__['obs_'+item]
            elif item == 'H':
                observations_item = cp.deepcopy(optim.__dict__['obs_H']) + optim.pstate[state.index('FracH')] * optim.EnBalDiffObs_atHtimes
            elif item == 'LE':
                observations_item = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optim.pstate[state.index('FracH')]) * optim.EnBalDiffObs_atLEtimes  
        else:
            observations_item = optim.__dict__['obs_'+item]
        if item in obs_sca_cf:
            obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
        else:
            obs_scale = 1.0 
        weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable
        k = 0 #counter for the observations (specific for each type of obs)
        for ti in range(priormodel.tsteps):
            if round(priormodel.out.t[ti] * 3600,8) in [round(num, 8) for num in obs_times[item]]: #so if we are at a time where we have an obs
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
                min_costf_ind = optim.Costflist.index(min_costf0) #find the index number of the simulation where costf was minimal
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
        min_costf_ind = optim.Costflist.index(min_costf0) #find the index number of the simulation where costf was minimal
        state_opt0 = optim.Statelist[min_costf_ind]
elif optim_method == 'tnc':
    if imposeparambounds:
        bounds = []
        for i in range(len(state)):
            if state[i] in boundedvars:
                bounds.append((boundedvars[state[i]][0],boundedvars[state[i]][1]))
            else:
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
            min_costf_ind = optim.Costflist.index(min_costf0) #find the index number of the simulation where costf was minimal
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
        min_costf_ind = optim.Costflist.index(min_costf0) #find the index number of the simulation where costf was minimal
        state_opt0 = optim.Statelist[min_costf_ind]
    if not hasattr(optim,'stop'):
        for i in range(maxnr_of_restarts):
            if min_costf0 > stopcrit: #will evaluate to False if min_costf0 is equal to nan
                optim.nr_of_sim_bef_restart = optim.sim_nr #This statement is required in case the optimisation algorithm terminates successfully, in case of static_costfError it was already ok
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
                        min_costf_ind = optim.Costflist.index(min_costf0) 
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
                    min_costf_ind = optim.Costflist.index(min_costf0) 
                    state_opt0 = optim.Statelist[min_costf_ind]
                min_costf0 = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights)
    if write_to_f:
        open('Optimfile.txt','a').write('{0:>25s}'.format('\n finished'))
else:
    raise Exception('Unavailable optim_method \'' + str(optim_method) + '\' specified')
print('optimal state without ensemble='+str(state_opt0))
CostParts0 = optim.cost_func(state_opt0,inputcopy,state,obs_times,obs_weights,RetCFParts=True)
CPdictiopr = optim.cost_func(optim.pstate,inputcopy,state,obs_times,obs_weights,RetCFParts=True)

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
                        priorinput_mem.__dict__[state[j]] =  cp.deepcopy(priorinput.__dict__[state[j]])#try to make parameter within the bounds
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
                raise Exception('Problem for perturbing non-state parameters: unknown distribtion for '+param)
            priorinput_mem.__dict__[param] += rand_nr
            non_state_pparamvals[param] = priorinput_mem.__dict__[param]
    priormodel_mem = fwdm.model(priorinput_mem)
    priormodel_mem.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False) #delete_at_end should be false, to keep tsteps of model
    optim_mem = im.inverse_modelling(priormodel_mem,write_to_file=write_to_f,use_backgr_in_cost=use_backgr_in_cost,StateVarNames=state,obsvarlist=obsvarlist,Optimfile='Optimfile'+str(counter)+'.txt',
                                     Gradfile='Gradfile'+str(counter)+'.txt',pri_err_cov_matr=b_cov,paramboundspenalty=paramboundspenalty,abort_slow_minims=abort_slow_minims,boundedvars=boundedvars)
    optim_mem.print = print_status_dur_ens
    Hx_prior_mem = {}
    PertDict = {} #To be passed as argument to min_func etc.
    PertDict_to_return = {} #The Perturbations dictionary to return in the return statement of the function. Cannot simply return PertDict, since it would return an empty dict when pert_obs_ens is True
    for item in obsvarlist:
        Hx_prior_mem[item] = priormodel_mem.out.__dict__[item]
        optim_mem.__dict__['obs_'+item] = cp.deepcopy(optim.__dict__['obs_'+item]) #the truth obs
        optim_mem.__dict__['error_obs_' + item] = cp.deepcopy(optim.__dict__['error_obs_' + item])
        if pert_Hx_min_sy_ens: #add a perturbation to H(x) - sy in the cost function, using sigma_O
            rand_nr_list = ([np.random.normal(0,optim_mem.__dict__['error_obs_' + item][i]) for i in range(len(measurement_error[item]))])
            PertDict[item] = rand_nr_list
        elif pert_obs_ens: 
            if use_sigma_O:
                rand_nr_list = ([np.random.normal(0,optim_mem.__dict__['error_obs_' + item][i]) for i in range(len(measurement_error[item]))])
            else:
                rand_nr_list = ([np.random.normal(0,measurement_error[item][i]) for i in range(len(measurement_error[item]))])
            PertDict_to_return[item] = rand_nr_list #Not PertDict, since than Hx_min_sy would additionally get perturbed
            optim_mem.__dict__['obs_'+item] += rand_nr_list
            if plot_perturbed_obs:
                unsca = 1 #a scale for plotting the obs with different units
                if (disp_units[item] == 'g/kg' or disp_units[item] == 'g kg$^{-1}$') and (item == 'q' or item.startswith('qmh')): #q can be plotted differently for clarity
                    unsca = 1000
                plt.figure()
                plt.plot(obs_times[item]/3600,unsca*orig_obs[item], linestyle=' ', marker='*',ms=10,color = 'black',label = 'orig')
                if perturb_truth_obs:
                    plt.plot(obs_times[item]/3600,unsca*optim.__dict__['obs_'+item], linestyle=' ', marker='o',color = 'blue',label = 'member 0')
                if plot_errbars:
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
    if pert_Hx_min_sy_ens:
        PertDict_to_return = cp.deepcopy(PertDict) #The Perturbations dictionary to return is the same as PertDict in this case
    obs_sca_cf_mem = {}
    optim_mem.pstate = [] #needed also for use in the inverse_modelling object
    for item in state:
        optim_mem.pstate.append(priorinput_mem.__dict__[item])
        if item.startswith('obs_sca_cf_'):
            obsname = item.split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
            obs_sca_cf_mem[obsname] = cp.deepcopy(priorinput_mem.__dict__[item])
    optim_mem.pstate = np.array(optim_mem.pstate)
    inputcopy_mem = cp.deepcopy(priorinput_mem) #deepcopy!
    params_mem = tuple([inputcopy_mem,state,obs_times,obs_weights,PertDict])
    if 'FracH' in state:
        if 'H' in obsvarlist:
            optim_mem.EnBalDiffObs_atHtimes = optim.EnBalDiffObs_atHtimes
        if 'LE' in obsvarlist:
            optim_mem.EnBalDiffObs_atLEtimes = optim.EnBalDiffObs_atLEtimes
    if ana_deriv:
        optim_mem.checkpoint = cp.deepcopy(priormodel_mem.cpx) 
        optim_mem.checkpoint_init = cp.deepcopy(priormodel_mem.cpx_init) 
        for item in obsvarlist:
            if 'FracH' in state:
                if item not in ['H','LE']:
                    observations_item = optim_mem.__dict__['obs_'+item]
                elif item == 'H':
                    observations_item = cp.deepcopy(optim_mem.__dict__['obs_H']) + optim_mem.pstate[state.index('FracH')] * optim_mem.EnBalDiffObs_atHtimes
                elif item == 'LE':
                    observations_item = cp.deepcopy(optim_mem.__dict__['obs_LE']) + (1 - optim_mem.pstate[state.index('FracH')]) * optim_mem.EnBalDiffObs_atLEtimes 
            else:
                observations_item = optim_mem.__dict__['obs_'+item]
            if item in obs_sca_cf_mem:
                obs_scale = obs_sca_cf_mem[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
            else:
                obs_scale = 1.0
            weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable
            pert = 0.0
            k = 0
            for ti in range(priormodel_mem.tsteps):
                if round(priormodel_mem.out.t[ti] * 3600,8) in [round(num, 8) for num in obs_times[item]]: #so if we are at a time where we have an obs
                    if item in obs_weights:
                        weight = obs_weights[item][k]
                    if item in PertDict:
                        pert = PertDict[item][k]
                    forcing = weight * (Hx_prior_mem[item][ti] - obs_scale * observations_item[k] + pert)/(optim_mem.__dict__['error_obs_' + item][k]**2) #don't include the background term of the cost function in the forcing!
                    optim_mem.forcing[ti][item] = forcing
                    k += 1
    if paramboundspenalty:
        optim_mem.setNanCostfOutBoundsTo0 = setNanCostfOutBoundsTo0
        optim_mem.penalty_exp = penalty_exp
    if optim_method == 'bfgs':
        try:
            if ana_deriv:
                minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params_mem,gtol=gtol,full_output=True)
            else:
                minimisation_mem = optimize.fmin_bfgs(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params_mem,gtol=gtol,full_output=True)
            state_opt_mem = minimisation_mem[0]
            min_costf_mem = minimisation_mem[1]
        except (im.nan_incostfError):
            print('Minimisation aborted due to nan') #discard_nan_minims == False allows to use last non-nan result in the optimisation, otherwise we throw away the optimisation
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
            if (discard_nan_minims == False and len(optim_mem.Statelist) > 0): #len(optim_mem.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf_mem = np.min(optim_mem.Costflist)
                min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
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
            min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
            state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
    elif optim_method == 'tnc':
        try:
            if ana_deriv:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.ana_deriv,args=params_mem,bounds=bounds,maxfun=None)
            else:
                minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,optim_mem.pstate,fprime=optim_mem.num_deriv,args=params_mem,bounds=bounds,maxfun=None)
            state_opt_mem = minimisation_mem[0]
            min_costf_mem = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights,PertDict) #within cost_func, the values of the variables in inputcopy_mem that are also state variables will be overwritten by the values of the variables in state_opt_mem   
        except (im.nan_incostfError):
            print('Minimisation aborted due to nan') #discard_nan_minims == False allows to use last non-nan result in the optimisation, otherwise we throw away the optimisation
            if write_to_f:
                open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
                open('Gradfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\nnan reached, no restart'))
            if (discard_nan_minims == False and len(optim_mem.Statelist) > 0): #len(optim_mem.Statelist) > 0 to check wether there is already a non-nan result in the optimisation, if not we choose nan as result
                min_costf_mem = np.min(optim_mem.Costflist)
                min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
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
            min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
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
                            minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.ana_deriv,args=params_mem,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                        else:
                            minimisation_mem = optimize.fmin_tnc(optim_mem.min_func,state_opt_mem,fprime=optim_mem.num_deriv,args=params_mem,bounds=bounds,maxfun=None) #restart from best sim so far to make it better if costf still too large
                        state_opt_mem = minimisation_mem[0]
                    except (im.nan_incostfError):
                        print('Minimisation aborted due to nan, no restart for this member')
                        if write_to_f:
                            open('Optimfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                            open('Gradfile'+str(counter)+'.txt','a').write('\nnan reached, no restart')
                        if discard_nan_minims == False:
                            min_costf_mem = np.min(optim_mem.Costflist)
                            min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
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
                        min_costf_mem_ind = optim_mem.Costflist.index(min_costf_mem) #find the index number of the simulation where costf was minimal
                        state_opt_mem = optim_mem.Statelist[min_costf_mem_ind]
                    min_costf_mem = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights,PertDict) 
    CostParts = optim_mem.cost_func(state_opt_mem,inputcopy_mem,state,obs_times,obs_weights,PertDict,True)
    if write_to_f:
        open('Optimfile'+str(counter)+'.txt','a').write('{0:>25s}'.format('\n finished'))
    return min_costf_mem,state_opt_mem,optim_mem.pstate,non_state_pparamvals,CostParts,PertDict_to_return

#chi squared denominator to be used later
if use_weights:
    denom_chisq = tot_sum_of_weights
else:
    denom_chisq = number_of_obs
if use_backgr_in_cost:
    denom_chisq += number_of_params
  
if use_ensemble:
    ensemble = []
    PertData_mems = []
    for i in range(0,nr_of_members): #the zeroth is the one done before
        ensemble.append({})
        PertData_mems.append({})
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
        result_array = Parallel(n_jobs=max_num_cores_)(delayed(run_ensemble_member)(i,seedvalue,non_state_paramdict)  for i in range(1,nr_of_members)) #, prefer="threads" makes it work, but probably not multiprocess. 
        #the above returns a list of tuples
        for j in range(1,nr_of_members):
            ensemble[j]['min_costf'] = result_array[j-1][0] #-1 due to the fact that the zeroth ensemble member is not part of the result_array, while it is part of ensemble
            ensemble[j]['state_opt'] = result_array[j-1][1]
            ensemble[j]['pstate'] = result_array[j-1][2]
            ensemble[j]['nonstateppars'] = result_array[j-1][3] #an empty dictionary if not pert_non_state_param=True
            ensemble[j]['CostParts'] = result_array[j-1][4]
            PertData_mems[j] = result_array[j-1][5] #we want to store this info seperately for writing as a pickle object
    else:
        for i in range(1,nr_of_members):
            ensemble[i]['min_costf'],ensemble[i]['state_opt'],ensemble[i]['pstate'],ensemble[i]['nonstateppars'],ensemble[i]['CostParts'],PertData_mems[i] =  run_ensemble_member(i,seedvalue,non_state_paramdict)
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
        pvar_state_ens = np.zeros(len(state)) #variance of the prior state elements calculated from the ensemble excl member 0
        chi_sq = np.zeros(len(ensemble))
        success_ens = np.zeros(len(ensemble), dtype=bool) #True or False wether optimisation successful
        for i in range(len(ensemble)):
            if not np.isnan(seq_costf[i]):
                chi_sq[i] = seq_costf[i]/denom_chisq
                if chi_sq[i] <= succes_opt_crit:
                    success_ens[i] = True
        if np.sum(success_ens[1:]) > 1:
            for i in range(len(state)):
                seq = np.array([x['state_opt'][i] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling,
                #as we choose the prior ourselves (unperturbed prior, not drawn from a distribution).
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
                seq_p = np.array([x['pstate'][i] for x in ensemble[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation
                seq_suc_p = np.array([seq_p[x] for x in range(len(seq_p)) if success_ens[1:][x]])
                mean_state_prior[i] = np.mean(seq_suc_p)
                pvar_state_ens[i] = np.var(seq_suc_p,ddof=1)
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
                plt.xlabel(disp_nms_par[state[i]] + ' ('+ disp_units_par[state[i]] +')')
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
                    seq_ns = np.array([x['nonstateppars'][param] for x in ensemble[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling
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
                    plt.xlabel(disp_nms_par[param] + ' ('+ disp_units_par[param] +')')
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
        for item in non_state_paramdict:
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
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('post chi squared without ensemble'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('prior costf without ensemble'))
    open('Optstatsfile.txt','a').write('{0:>35s}'.format('post costf without ensemble'))
    if use_ensemble:
        open('Optstatsfile.txt','a').write('{0:>51s}'.format('post chi squared of member with lowest post costf'))
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
        open('Optstatsfile.txt','a').write('{0:>51s}'.format(str(chi_sq_ens)))
        if opt_sim_nr != 0:
            filename = 'Optimfile'+str(opt_sim_nr)+'.txt'
            with open(filename,'r') as LowestCfFile:
                header_end = LowestCfFile.readline().split()[-1]
                if header_end != 'Costf':
                    raise Exception(filename +' does not have \'Costf\' as last column')
                prior_costf_ens = LowestCfFile.readline().split()[-1]
        else:
            prior_costf_ens = prior_costf
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
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('post costf parts of member'))
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('with lowest post costf:'))
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
    open('Optstatsfile.txt','a').write('\n')
    open('Optstatsfile.txt','a').write('{0:>32s}'.format('costf parts prior member 0:'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
    if use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('Background'))
    if paramboundspenalty:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format('Penalty'))
    open('Optstatsfile.txt','a').write('\n      ')
    for obsvar in obsvarlist:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictiopr[obsvar])))
    if use_backgr_in_cost:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictiopr['backgr'])))
    if paramboundspenalty:
        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CPdictiopr['penalty'])))
    open('Optstatsfile.txt','a').write('\n')
    if use_ensemble and opt_sim_nr != 0:
        CP_best_st_obsmem0 = optim.cost_func(state_opt,inputcopy,state,obs_times,obs_weights,RetCFParts=True)
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('costf parts best state with obs'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format(', prior and non-state'))
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('parameters of member 0:'))
        open('Optstatsfile.txt','a').write('\n      ')
        for obsvar in obsvarlist:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(obsvar))
        if use_backgr_in_cost:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format('Background'))
        if paramboundspenalty:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format('Penalty'))
        open('Optstatsfile.txt','a').write('\n      ')
        for obsvar in obsvarlist:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CP_best_st_obsmem0[obsvar])))
        if use_backgr_in_cost:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CP_best_st_obsmem0['backgr'])))
        if paramboundspenalty:
            open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(CP_best_st_obsmem0['penalty'])))
        open('Optstatsfile.txt','a').write('\n')    
    if use_ensemble or use_backgr_in_cost: #priorvar only defined if use_backgr_in_cost or if use_ensemble
        open('Optstatsfile.txt','a').write('\n')
        open('Optstatsfile.txt','a').write('{0:>32s}'.format('Normalised deviation from unper-'))
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
            if round(optimalmodel.out.t[ti] * 3600,8) in [round(num, 8) for num in obs_times[obsvar]]:
                outp_at_obstimes[obsvar] += [optimalmodel.out.__dict__[obsvar][ti]]
                outp_at_obstimes_pr[obsvar] += [priormodel.out.__dict__[obsvar][ti]]
        numerator = np.var(outp_at_obstimes[obsvar])#degrees of freedom not important, when numerator and denominator have same degrees of freedom
        numerator_pr = np.var(outp_at_obstimes_pr[obsvar])
        obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_'+obsvar])
        obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_'+obsvar])
        if 'obs_sca_cf_'+obsvar in state:
            obs_to_use[obsvar] *= optimalinput.__dict__['obs_sca_cf_'+obsvar]
            obs_to_use_pr[obsvar] *= priorinput.__dict__['obs_sca_cf_'+obsvar]
        elif 'FracH' in state: #obs_sca_cf_H or obs_sca_cf_LE will not be in the state together with FracH, the script would raise an Exception in that case
            if obsvar == 'H':
                obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_H']) + optimalstate[state.index('FracH')] * optim.EnBalDiffObs_atHtimes
                obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_H']) + optim.pstate[state.index('FracH')] * optim.EnBalDiffObs_atHtimes
            elif obsvar == 'LE':
                obs_to_use[obsvar] = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optimalstate[state.index('FracH')]) * optim.EnBalDiffObs_atLEtimes
                obs_to_use_pr[obsvar] = cp.deepcopy(optim.__dict__['obs_LE']) + (1 - optim.pstate[state.index('FracH')]) * optim.EnBalDiffObs_atLEtimes
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
                    ratio = post_cov_matr[i][i]/pvar_state_ens[i]
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
            for param in non_state_paramdict: 
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
                    for param in non_state_paramdict:
                        open('Optstatsfile.txt','a').write('{0:>25s}'.format(str(item['nonstateppars'][param])))                
                else:
                    for param in non_state_paramdict:
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

obslabel = 'obs' #Note that only the obs of member 0 are plotted here
for i in range(len(obsvarlist)):
    unsca = 1 #a scale for plotting the obs with different units
    if (disp_units[obsvarlist[i]] == 'g/kg' or disp_units[obsvarlist[i]] == 'g kg$^{-1}$') and (obsvarlist[i] == 'q' or obsvarlist[i].startswith('qmh')): #q can be plotted differently for clarity
        unsca = 1000
    fig = plt.figure()
    if plot_errbars:
        plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*orig_obs[obsvarlist[i]],yerr=unsca*optim.__dict__['error_obs_'+obsvarlist[i]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
        plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*orig_obs[obsvarlist[i]],yerr=unsca*measurement_error[obsvarlist[i]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    plt.plot(priormodel.out.t,unsca*priormodel.out.__dict__[obsvarlist[i]], ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
    plt.plot(priormodel.out.t,unsca*optimalmodel.out.__dict__[obsvarlist[i]], linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
    if use_ensemble:
        if pert_non_state_param and opt_sim_nr != 0:
            plt.plot(priormodel.out.t,unsca*optimalmodel_onsp.out.__dict__[obsvarlist[i]], linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
    plt.plot(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='*',color = 'black',ms=10, label = obslabel)
    if perturb_truth_obs:
        plt.plot(obs_times[obsvarlist[i]]/3600,unsca*orig_obs[obsvarlist[i]], linestyle=' ', marker='.',color = 'black',ms=10, label = 'unp obs') #unperturbed obs
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
        
if 'FracH' in state:
    if 'H' in obsvarlist:
        enbal_corr_H = optim.obs_H + optimalinput.FracH * optim.EnBalDiffObs_atHtimes
        fig = plt.figure()
        if plot_errbars:
            plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=optim.__dict__['error_obs_H'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=measurement_error['H'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
        plt.plot(priormodel.out.t,priormodel.out.H, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
        plt.plot(optimalmodel.out.t,optimalmodel.out.H, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
        if use_ensemble:
            if pert_non_state_param and opt_sim_nr != 0:
                plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.H, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
        plt.plot(obs_times['H']/3600,optim.__dict__['obs_'+'H'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs uncor')
        plt.plot(obs_times['H']/3600,enbal_corr_H, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
        plt.ylabel('H (' + disp_units['H']+')')
        plt.xlabel('time (h)')
        plt.legend(prop={'size':legendsize},loc=0)
        plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
        if write_to_f:
            plt.savefig('fig_fit_enbalcorrH.'+figformat, format=figformat)
    if 'LE' in obsvarlist:
        enbal_corr_LE = optim.obs_LE + (1 - optimalinput.FracH) * optim.EnBalDiffObs_atLEtimes
        fig = plt.figure()
        if plot_errbars:
            plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=optim.__dict__['error_obs_LE'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=measurement_error['LE'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
        plt.plot(priormodel.out.t,priormodel.out.LE, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
        plt.plot(optimalmodel.out.t,optimalmodel.out.LE, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
        if use_ensemble:
            if pert_non_state_param and opt_sim_nr != 0:
                plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.LE, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
        plt.plot(obs_times['LE']/3600,optim.__dict__['obs_'+'LE'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs uncor')
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
#for i in range(len(obsvarlist)):
#    fig = plt.figure()
#    plt.plot(priormodel.out.t*3600,priormodel.out.__dict__[obsvarlist[i]], linestyle=' ', marker='o',color='yellow',label = 'prior')
#    plt.plot(optimalmodel.out.t*3600,optimalmodel.out.__dict__[obsvarlist[i]], linestyle=' ', marker='o',color='red',label = 'post')
#    plt.plot(obs_times[obsvarlist[i]],optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='o',color = 'black')
#    plt.ylabel(obsvarlist[i])
#    plt.xlabel('timestep')
#    plt.legend(prop={'size':legendsize})

