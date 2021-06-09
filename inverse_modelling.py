# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:52:50 2019

@author: Bosman Peter
"""
import numpy as np
import copy as cp
import sys
import forwardmodel as fwdm
import math

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))
def desat(T,dT):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86)) * (-17.2694)*(35.86-273.16)/(T-35.86)**2 * dT
def dqsat_dT(T,p,dT):
    result = 0.622 * desat(T,dT) / p
    return result
def dqsat_dp(T,p,dp):
    result = 0.622 * esat(T) * (-1) * p**(-2) * dp
    return result

class nan_incostfError(Exception):
    pass

class static_costfError(Exception):
    pass

class inverse_modelling:
    
    def __init__(self,model,write_to_file=False,use_backgr_in_cost=False,StateVarNames=[],obsvarlist=[],Optimfile='Optimfile.txt',Gradfile='Gradfile.txt',pri_err_cov_matr=None,paramboundspenalty=False,boundedvars={}):                         
        self.adjointtestingags = False #adjoint test of one module at one single time step
        self.adjointtestingrun_surf_lay = False
        self.adjointtestingribtol = False
        self.adjointtestingrun_mixed_layer = False
        self.adjointtestingint_mixed_layer = False
        self.adjointtestingrun_radiation = False
        self.adjointtestingrun_land_surface = False
        self.adjointtestingint_land_surface = False
        self.adjointtestingstatistics = False
        self.adjointtestingrun_cumulus = False
        self.adjointtestingrun_soil_COS_mod = False
        self.adjointtestingstore = False
        self.adjointtestingjarvis_stewart = False
        self.manualadjointtesting = False
        self.adjointtesting = False
        self.gradienttesting = False
        self.paramboundspenalty = paramboundspenalty
        self.model = model
        self.boundedvars = cp.deepcopy(boundedvars)
        self.forcing = []
        self.checkpoint = [] #list of dictionaries
        for i in range(model.tsteps):
            self.checkpoint.append ({})
            self.forcing.append ({})
        self.obsvarlist = obsvarlist
        self.use_backgr_in_cost = use_backgr_in_cost
        if pri_err_cov_matr is not None:
            self.binv = np.linalg.inv(pri_err_cov_matr) #invert prior error covariance matrix
        self.write_to_f = write_to_file
        self.sim_nr = 0
        self.Costflist = []
        self.Statelist = []
        self.nr_sims_for_check = 10 #when the cost function changes too slowly, the minimisation will be aborted. This number determines how many of the most recent simulations (their cost function value) will be taken into account 
        self.nr_of_sim_bef_restart = 0 #the simulation nr reached before restart took place (if applicable, at this point in the code no restart has taken place). 
        if self.write_to_f:
            self.Optimf = Optimfile
            self.Gradf = Gradfile
            open(self.Optimf,'w').write('{0:>25s}'.format('sim_nr')) #here we make the file
            for item in StateVarNames:
                open(self.Optimf,'a').write('{0:>25s}'.format(item))
            open(self.Optimf,'a').write('{0:>25s}'.format('data'))
            if self.use_backgr_in_cost:
                open(self.Optimf,'a').write('{0:>25s}'.format('background'))
            open(self.Optimf,'a').write('{0:>25s}'.format('Costf'))
            
            open(self.Gradf,'w').write('{0:>25s}'.format('sim_nr')) #here we make the file
            for item in StateVarNames:
                open(self.Gradf,'a').write('{0:>25s}'.format(item))
            if self.use_backgr_in_cost:
                for item in StateVarNames:
                    open(self.Gradf,'a').write('{0:>30s}'.format('backgr_grad_'+item))
            for item in StateVarNames:
                open(self.Gradf,'a').write('{0:>30s}'.format('Costf_grad_'+item))
    
    def min_func(self,state_to_opt,inputdata,state_var_names,obs_times,obs_weights={}): #some dummy vars as arg list of min_func and deriv_func needs to be the same
        cost = 0
        self.sim_nr += 1
        print('accessing min func')
        print('state_to_opt'+str(state_to_opt))
        obs_sca_cf = {}
        for i in range(len(state_var_names)):
            inputdata.__dict__[state_var_names[i]] = state_to_opt[i]
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i]
        model = fwdm.model(inputdata)
        model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        for item in self.obsvarlist:
            if 'EnBalDiffObsHFrac' in state_var_names:
                if item not in ['H','LE']:
                    observations_item = self.__dict__['obs_'+item]
                elif item == 'H':
                    observations_item = cp.deepcopy(self.__dict__['obs_H']) + state_to_opt[state_var_names.index('EnBalDiffObsHFrac')] * self.EnBalDiffObs_atHtimes
                elif item == 'LE':
                    observations_item = cp.deepcopy(self.__dict__['obs_LE']) + (1 - state_to_opt[state_var_names.index('EnBalDiffObsHFrac')]) * self.EnBalDiffObs_atLEtimes  
            else:
                observations_item = self.__dict__['obs_'+item]
            if item in obs_sca_cf:
                obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
            else:
                obs_scale = 1.0 
            weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
            k = 0 #counter for the observations (specific for each type of obs)
            for i in range (model.tsteps):
                if round(model.out.t[i] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
                    if item in obs_weights:
                        weight = obs_weights[item][k]
                    cost += weight * (model.out.__dict__[item][i] - obs_scale * observations_item[k])**2/(self.__dict__['error_obs_' + item][k]**2)
                    forcing = weight * (model.out.__dict__[item][i]- obs_scale * observations_item[k])/(self.__dict__['error_obs_' + item][k]**2)
                    self.forcing[i][item] = forcing #so here the model observation mismatch is calculated for use in the next call of the analytical derivative calculation
                    k += 1
        for i in range (model.tsteps):
            self.checkpoint[i] = model.cpx[i]
        self.checkpoint_init = model.cpx_init
        if self.paramboundspenalty: #note that the code under this if statement will not perform any action if tnc method is used, since in case of tnc with parameter bounds, the state variables are always within the specified bounds. 
            for i in range(len(state_var_names)):
                if state_var_names[i] in self.boundedvars:
                    if state_to_opt[i] < self.boundedvars[state_var_names[i]][0]: #lower than lower bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(cost):
                                cost = 0 #to avoid ending up with a lot of nans. note that nan + a number gives nan. Assume nans only occur when outside of param bounds, than these statements will lead to a high costf...
                        cost += 1/self.penalty_exp*(2 - (state_to_opt[i] - self.boundedvars[state_var_names[i]][0]))**self.penalty_exp #note that this will always be positive
                    elif state_to_opt[i] > self.boundedvars[state_var_names[i]][1]: #higher than higher bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(cost):
                                cost = 0
                        cost += 1/self.penalty_exp*(2 + (state_to_opt[i] - self.boundedvars[state_var_names[i]][1]))**self.penalty_exp 
        if self.write_to_f:
            open(self.Optimf,'a').write('\n')
            open(self.Optimf,'a').write('{0:>25s}'.format(str(self.sim_nr)))
            for item in state_to_opt:
                open(self.Optimf,'a').write('{0:>25s}'.format(str(item)))
            open(self.Optimf,'a').write('{0:>25s}'.format(str(cost)))
        if self.use_backgr_in_cost:
            Jbackground = self.background_costf(state_to_opt)
            cost += Jbackground
            if self.write_to_f:
                open(self.Optimf,'a').write('{0:>25s}'.format(str(Jbackground)))
        print('costf='+str(cost))
        if self.write_to_f:
            open(self.Optimf,'a').write('{0:>25s}'.format(str(cost)))
        print('end min func')
        if math.isnan(cost):
            raise nan_incostfError('nan in costf')
        self.Costflist.append(cost)
        self.Statelist.append(state_to_opt)
        if self.sim_nr > 15 and (self.sim_nr - self.nr_of_sim_bef_restart > self.nr_sims_for_check): #second condition to prevent termination immediately after restart
            too_small_diff = True
            for i in range(len(self.Costflist)-1,len(self.Costflist)-1-self.nr_sims_for_check,-1): #so you look at last 10 sims
                diff = (self.Costflist[i] - self.Costflist[i-1])/self.Costflist[i-1]
                if abs(diff) > 0.001:
                    too_small_diff = False #prevent too many sims without a reasonable change in costf
            if too_small_diff == True:
                if self.write_to_f:
                    open(self.Optimf,'a').write('\n')
                    open(self.Optimf,'a').write('{0:>25s}'.format('aborted minimisation'))
                self.nr_of_sim_bef_restart = self.sim_nr 
                raise static_costfError('too slow progress in costf')
        return cost
    
    def cost_func(self,state_to_opt,inputdata,state_var_names,obs_times,obs_weights={},RetCFParts=False): #Similar to min func, but without file writing and changing forcings etc, so it can be called safely without altering an optimisation
        cost = 0
        obs_sca_cf = {}
        for i in range(len(state_var_names)):
            inputdata.__dict__[state_var_names[i]] = state_to_opt[i]
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i]
        model = fwdm.model(inputdata)
        model.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)
        CostParts = {}
        for item in self.obsvarlist: #do not provide obs at a time that is not modelled, this will lead to false results! (make a check for this in the optimisation file!)
            if RetCFParts: #means return cost function parts
                CostParts[item] = 0
            if 'EnBalDiffObsHFrac' in state_var_names:
                if item not in ['H','LE']:
                    observations_item = self.__dict__['obs_'+item]
                elif item == 'H':
                    observations_item = cp.deepcopy(self.__dict__['obs_H']) + state_to_opt[state_var_names.index('EnBalDiffObsHFrac')] * self.EnBalDiffObs_atHtimes
                elif item == 'LE':
                    observations_item = cp.deepcopy(self.__dict__['obs_LE']) + (1 - state_to_opt[state_var_names.index('EnBalDiffObsHFrac')]) * self.EnBalDiffObs_atLEtimes  
            else:
                observations_item = self.__dict__['obs_'+item]
            if item in obs_sca_cf:
                obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
            else:
                obs_scale = 1.0 
            weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
            k = 0 #counter for the observations (specific for each type of obs)
            for i in range (model.tsteps):
                if round(model.out.t[i] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
                    if item in obs_weights:
                        weight = obs_weights[item][k]
                    increment = weight * (model.out.__dict__[item][i]- obs_scale * observations_item[k])**2/(self.__dict__['error_obs_' + item][k]**2)
                    cost += increment
                    k += 1
                    if RetCFParts:
                        CostParts[item] += increment
            
        if self.paramboundspenalty: #note that the code under this if statement will not perform any action if tnc method is used, since in case of tnc with parameter bounds, the state variables are always within the specified bounds. 
            if RetCFParts:
                CostParts['penalty'] = 0
            for i in range(len(state_var_names)):
                if state_var_names[i] in self.boundedvars:
                    if state_to_opt[i] < self.boundedvars[state_var_names[i]][0]: #lower than lower bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(cost):
                                cost = 0 #to avoid ending up with a lot of nans. note that nan + a number gives nan. 
                                if RetCFParts:
                                    for key in RetCFParts:
                                       RetCFParts[key] = 0
                        increment = 1/self.penalty_exp*(2 - (state_to_opt[i] - self.boundedvars[state_var_names[i]][0]))**self.penalty_exp #note that this will always be positive
                        cost += increment
                        if RetCFParts:
                            CostParts['penalty'] += increment
                    elif state_to_opt[i] > self.boundedvars[state_var_names[i]][1]: #higher than higher bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(cost):
                                cost = 0 #to avoid ending up with a lot of nans. note that nan + a number gives nan. 
                                if RetCFParts:
                                    for key in RetCFParts:
                                       RetCFParts[key] = 0
                        increment = 1/self.penalty_exp*(2 + (state_to_opt[i] - self.boundedvars[state_var_names[i]][1]))**self.penalty_exp 
                        cost += increment
                        if RetCFParts:
                            CostParts['penalty'] += increment
        if self.use_backgr_in_cost:
            Jbackground = self.background_costf(state_to_opt)
            cost += Jbackground
            if RetCFParts:
                CostParts['backgr'] = Jbackground
        if RetCFParts:
            return CostParts 
        else:
            return cost
    
    def ana_deriv(self,state_to_opt,inputdata,state_var_names,obs_times,obs_weights={}): #some dummy vars as arg list of min_func and deriv_func needs to be the same
        model = self.model #just to ease notation later on
        print('accessing deriv_func')
        print ('state'+str(state_to_opt))
        if self.write_to_f:
            open(self.Gradf,'a').write('\n')
            open(self.Gradf,'a').write('{0:>25s}'.format(str(self.sim_nr)))
            for item in state_to_opt:
                open(self.Gradf,'a').write('{0:>25s}'.format(str(item)))
        gradient = np.zeros(len(state_to_opt))
        returnvariables = []
        for item in state_var_names:
            if not (item.startswith('obs_sca_cf_') or item == 'EnBalDiffObsHFrac'):
                returnvariables.append('ad'+item)
        checkpoint_init = self.checkpoint_init
        self.initialise_adjoint()
        HTy = self.adjoint(self.forcing,self.checkpoint,checkpoint_init,model,returnvariables=returnvariables)
        HTy_counter = 0
        for i in range(len(state_var_names)):
            if not (state_var_names[i].startswith('obs_sca_cf_') or state_var_names[i] == 'EnBalDiffObsHFrac'):
                gradient[i] += 2*HTy[HTy_counter] #see second part of 11.77 in chapter inverse modelling Brasseur and Jacob 2017, where adjoint is K^T, and the forcing is SO^-1*(F(x) - y)? 
                HTy_counter += 1
            #now add the gradients for the obs scaling that is possibly included in the state
            if state_var_names[i].startswith('obs_sca_cf_'):
                obsname = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                forcinglist = []
                for dictio in self.forcing:
                    if obsname in dictio: #there are possibly also model timesteps where we do not have an obs of the specific type we are looking for, in that case some dictionaries in self.forcing do not have the obsname item               
                        forcinglist.append(dictio[obsname])           
                gradient[i] += np.sum(np.array(forcinglist) * -2 * self.__dict__['obs_'+obsname][:]) #this is the derivative of the cost function to the obs scale parameter. 
                #Note that forcing is a part of this derivative (obs part of cost function for one obs = w*(Hx - obs_scale*y)²/sigma_y², derivative to obs_scale = 2w * -y * (Hx - obs_scale*y)/sigma_y² = -2y* forcing, total observation part of cost function is sum of cost functions for individual obs)
            elif state_var_names[i] == 'EnBalDiffObsHFrac':
                deriv_H_obs_to_EnBalDiffObsHFrac = []
                deriv_LE_obs_to_EnBalDiffObsHFrac = []
                forcinglist_H = []
                forcinglist_LE = []
                ti = 0
                ti2 = 0
                for dictio in self.forcing: #one dictionary for every timestep
                    if 'H' in dictio: #               
                        forcinglist_H.append(dictio['H'])
                        deriv_H_obs_to_EnBalDiffObsHFrac.append(self.EnBalDiffObs_atHtimes[ti]) #only add to deriv_to_H_obs_EnBalDiffObsHFrac if an obs of H is used at that timestep 
                        ti += 1 #EnBalDiffObs_atHtimes should be available at (and only at) the times of H. It is also allowed to use only H or LE instead of both
                    if 'LE' in dictio:
                        forcinglist_LE.append(dictio['LE'])
                        deriv_LE_obs_to_EnBalDiffObsHFrac.append(-1 * self.EnBalDiffObs_atLEtimes[ti2])
                        ti2 += 1
                gradient[i] += np.sum(np.array(forcinglist_H) * -2 * np.array(deriv_H_obs_to_EnBalDiffObsHFrac)) #derivative to EnBalDiffObsHFrac (via observations_item, called y here) is 
                #2w * (Hx - obs_scale*y)/sigma_y² * -obs_scale * d_y/d_EnBalDiffObsHFrac = -2*obsscale* forcing * d_y/d_EnBalDiffObsHFrac. obsscale always 1 for H and LE obs if EnBalDiffObsHFrac in state
                gradient[i] += np.sum(np.array(forcinglist_LE) * -2 * np.array(deriv_LE_obs_to_EnBalDiffObsHFrac))
            if self.paramboundspenalty: #note that the code under this if statement will not perform any action if tnc method is used, since in case of tnc with parameter bounds, the state variables are always within the specified bounds. 
                if state_var_names[i] in self.boundedvars:
                    if state_to_opt[i] < self.boundedvars[state_var_names[i]][0]: #lower than lower bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(gradient[i]): #note that nan + something = nan
                                gradient[i] = 0 #to avoid ending up with a lot of nans. Assume nans only occur when outside of param bounds.
                        gradient[i] += -(2 - (state_to_opt[i] - self.boundedvars[state_var_names[i]][0]))**(self.penalty_exp-1) #this is the derivative of what is in min_func. Note that also without nan we get a large gradient with this statement
                    elif state_to_opt[i] > self.boundedvars[state_var_names[i]][1]: #higher than higher bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(gradient[i]):
                                gradient[i] = 0
                        gradient[i] += (2 + (state_to_opt[i] - self.boundedvars[state_var_names[i]][1]))**(self.penalty_exp-1) 
        #add the background part of the cost function
        if self.use_backgr_in_cost:
            dcostf_dbackground = 2*np.matmul(self.binv,state_to_opt-self.pstate) #see first term of 11.77 in chapter inverse modelling Brasseur and Jacob 2017
            for i in range(len(state_to_opt)):
                gradient[i] += dcostf_dbackground[i] 
                if self.write_to_f:
                    open(self.Gradf,'a').write('{0:>30s}'.format(str(dcostf_dbackground[i])))
        if self.write_to_f:
            for item in gradient:
                open(self.Gradf,'a').write('{0:>30s}'.format(str(item)))
        print ('grad'+str(gradient))
        print('end deriv_func')
        return np.array(gradient) #!!!!must return an array!!
    
    def num_deriv(self,state_to_opt,inputdata,state_var_names,obs_times,obs_weights={}): #some dummy vars as arg list of min_func and deriv_func needs to be the same
        print('accessing num_deriv')
        if self.write_to_f:
            open(self.Gradf,'a').write('\n')
            open(self.Gradf,'a').write('{0:>25s}'.format(str(self.sim_nr)))
            for item in state_to_opt:
                open(self.Gradf,'a').write('{0:>25s}'.format(str(item)))
        gradient = np.zeros(len(state_to_opt))
        cost_forw = 0
        cost_backw = 0
        delta = 0.000001
        obs_sca_cf = {}
        for i in range(len(state_var_names)):
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i] #set the obsscales
        for i in range(len(state_var_names)):
            inputdata.__dict__[state_var_names[i]] = state_to_opt[i] + delta
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i] + delta 
            if state_var_names[i] == 'EnBalDiffObsHFrac': #no elif here
                EnBalDiffObsHFrac = state_to_opt[i] + delta
            elif 'EnBalDiffObsHFrac' in state_var_names:
                EnBalDiffObsHFrac = state_to_opt[state_var_names.index('EnBalDiffObsHFrac')]
            model = fwdm.model(inputdata)
            model.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)
            for item in self.obsvarlist:
                if 'EnBalDiffObsHFrac' in state_var_names:
                    if item not in ['H','LE']:
                        observations_item = self.__dict__['obs_'+item]
                    elif item == 'H':
                        observations_item = cp.deepcopy(self.__dict__['obs_H']) + EnBalDiffObsHFrac * self.EnBalDiffObs_atHtimes
                    elif item == 'LE':
                        observations_item = cp.deepcopy(self.__dict__['obs_LE']) + (1 - EnBalDiffObsHFrac) * self.EnBalDiffObs_atLEtimes  
                else:
                    observations_item = self.__dict__['obs_'+item]
                if item in obs_sca_cf:
                    obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
                else:
                    obs_scale = 1.0 
                weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
                r = 0 #counter for the observations (specific for each type of obs)
                for j in range (model.tsteps):
                    if round(model.out.t[j] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
                        if item in obs_weights:
                            weight = obs_weights[item][r]
                        cost_forw += weight * (model.out.__dict__[item][j]- obs_scale * observations_item[r])**2/(self.__dict__['error_obs_' + item][r]**2)
                        r += 1
            inputdata.__dict__[state_var_names[i]] = state_to_opt[i] - delta
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i] - delta
            elif state_var_names[i] == 'EnBalDiffObsHFrac': 
                EnBalDiffObsHFrac = state_to_opt[i] - delta
            model = fwdm.model(inputdata)
            model.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False)
            for item in self.obsvarlist:
                if 'EnBalDiffObsHFrac' in state_var_names:
                    if item not in ['H','LE']:
                        observations_item = self.__dict__['obs_'+item]
                    elif item == 'H':
                        observations_item = cp.deepcopy(self.__dict__['obs_H']) + EnBalDiffObsHFrac * self.EnBalDiffObs_atHtimes
                    elif item == 'LE':
                        observations_item = cp.deepcopy(self.__dict__['obs_LE']) + (1 - EnBalDiffObsHFrac) * self.EnBalDiffObs_atLEtimes  
                else:
                    observations_item = self.__dict__['obs_'+item]
                if item in obs_sca_cf:
                    obs_scale = obs_sca_cf[item] #a scale for increasing/decreasing the magnitude of the observation in the cost function, useful if observations are possibly biased (scale not time dependent).
                else:
                    obs_scale = 1.0 
                weight = 1.0 # a weight for the observations in the cost function, modified below if weights are specified. For each variable in the obs, provide either no weights or a weight for every time there is an observation for that variable 
                r = 0 #counter for the observations (specific for each type of obs)
                for k in range (model.tsteps):
                    if round(model.out.t[k] * 3600,3) in [round(num, 3) for num in obs_times[item]]: #so if we are at a time where we have an obs
                        if item in obs_weights:
                            weight = obs_weights[item][r]
                        cost_backw += weight * (model.out.__dict__[item][k]- obs_scale * observations_item[r])**2/(self.__dict__['error_obs_' + item][r]**2)
                        r += 1
            if state_var_names[i].startswith('obs_sca_cf_'):
                item = state_var_names[i].split("obs_sca_cf_",1)[1] #split so we get the part after obs_sca_cf_
                obs_sca_cf[item] = state_to_opt[i] #reset the obs scale since delta was subtracted, for use with the other parameters in the state
            gradient[i] = (cost_forw - cost_backw) / (2 * delta)
            if self.paramboundspenalty: #note that the code under this if statement will not perform any action if tnc method is used, since in case of tnc with parameter bounds, the state variables are always within the specified bounds. 
                if state_var_names[i] in self.boundedvars:
                    if state_to_opt[i] < self.boundedvars[state_var_names[i]][0]: #lower than lower bound
                        if self.setNanCostfOutBoundsTo0:
                            if math.isnan(gradient[i]): #note that nan + something = nan 
                                gradient[i] = 0 #to avoid ending up with a lot of nans. Assume nans only occur when outside of param bounds.
                        gradient[i] += -(2 - (state_to_opt[i] - self.boundedvars[state_var_names[i]][0]))**(self.penalty_exp-1) #this is the derivative of what is in min_func. Note that also without nan we get a large gradient with this statement
                    elif state_to_opt[i] > self.boundedvars[state_var_names[i]][1]: #higher than higher bound
                        if self.setNanCostfOutBoundsTo0:        
                            if math.isnan(gradient[i]):
                                gradient[i] = 0
                        gradient[i] += (2 + (state_to_opt[i] - self.boundedvars[state_var_names[i]][1]))**(self.penalty_exp-1) 
            if self.use_backgr_in_cost:
                forw_state = cp.deepcopy(state_to_opt)
                forw_state[i] += delta
                backgr_forw = self.background_costf(forw_state)
                backw_state = cp.deepcopy(state_to_opt)
                backw_state[i] -= delta
                backgr_backw = self.background_costf(backw_state)
                # in simple case:
                #backgr_forw = (state_to_opt[i] + delta - self.pstate[i])**2/(self.b_cov[i,i])
                #backgr_backw = (state_to_opt[i] - delta - self.pstate[i])**2/(self.b_cov[i,i])
                dcostf_dbackground = (backgr_forw - backgr_backw) / (2 * delta) #for one variable
                gradient[i] += dcostf_dbackground
                if self.write_to_f:
                    open(self.Gradf,'a').write('{0:>30s}'.format(str(dcostf_dbackground)))
        if self.write_to_f:
            for item in gradient:
                open(self.Gradf,'a').write('{0:>30s}'.format(str(item)))
        print ('grad'+str(gradient))
        print('end num_deriv')
        return np.array(gradient)
    
    def background_costf(self,state):
        '''function that returns the background part of the cost function'''
        Jb = np.matmul(np.matrix.transpose(np.array(state)-np.array(self.pstate)),np.matmul(self.binv,np.array(state)-np.array(self.pstate)))
        return Jb
    
    def dpsim(self, zeta, dzeta):
        if(zeta <= 0):
            if self.model.sw_useWilson:
                x     = (1. + 3.6 * (-1*zeta) ** (2./3.)) ** (-0.5)
                dx     = -0.5* (1. + 3.6 * (-1*zeta) ** (2./3.)) ** (-1.5) * 3.6*(2/3)*((-1*zeta) **(-1./3.)*-1) *dzeta
                dpsim = 3. * 1/((1. + 1. / x) / 2.) * 0.5 * -1 * x**(-2) * dx
            else: #businger-Dyer
                x     = (1. - 16. * zeta)**(0.25)
                dx     = 0.25 * (1. - 16. * zeta)**(0.25 - 1.) * -16 * dzeta
                dpsim = - 2. * 1 / (1+ x**2) * dx + 1 / ((1. + x)**2. * (1. + x**2.) / 8.) * ((1. + x**2.) / 8. * 2 * (1. + x) * dx + (1. + x)**2. / 8 * 2 * x * dx)
        else:
            if self.model.sw_model_stable_con:
                #psim  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - zeta - (10./3.) / 0.35
                dpsim =  -2./3. * np.exp(-0.35 * zeta) * dzeta - 2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) * -0.35 * dzeta - dzeta
            else:
                dpsim = np.nan 
        return dpsim
    
    def dpsih(self, zeta,dzeta):
        if(zeta <= 0):
            if self.model.sw_useWilson:
                x     = (1. + 7.9 * (-1*zeta) ** (2./3.)) ** (-0.5)
                dx     = (-0.5) * (1. + 7.9 * (-1*zeta) ** (2./3.)) ** (-1.5) * 7.9 * (2/3)* (-1*zeta) ** (-1./3.) *-1*dzeta
                dpsih  = 3. * 1/( (1. + 1. / x) / 2.) *0.5*-1*x**(-2)*dx
            else:
                x     = (1. - 16. * zeta)**(0.25)
                dx     = 0.25 * (1. - 16. * zeta)**(0.25 - 1.) * -16 * dzeta
                dpsih  = 2. * 1 / ((1. + x*x) / 2.) * 1/2*2*x*dx
                #raise Exception("Businger-Dyer not yet implemented")
        else:
            if self.model.sw_model_stable_con:
                #psih  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - (1. + (2./3.) * zeta) ** (1.5) - (10./3.) / 0.35 + 1.
                dpsih = -2./3. * np.exp(-0.35 * zeta) * dzeta - 2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) * -0.35 * dzeta - 1.5 * (1. + (2./3.) * zeta) ** (0.5) * (2./3.) * dzeta
            else:
                dpsih = np.nan 
        return dpsih
        
    def dzeta_dL(self,z,L): #in contrast to the other functions, this is a true derivative, not derivative * dL
        result = -z*L**(-2)
        return result
    
    def dE1(self,x):
    # Wikipedia
        return -np.exp(-x)/x
    
    def initialise_tl(self,dstate):
        #needed for some gradient tests, depending on whether they are perturbed (than not needed) or not
        #also needed for adjoint test
        #And variables not specified here will not propagate in time as self-variables in the gradient test
        #surf lay
        self.dtheta,self.dwtheta,self.dh,self.dwCOS,self.dCOS,self.dCs_start,self.drs,self.dustar_start= 0,0,0,0,0,0,0,0
        self.dwstar,self.dwthetav,self.dthetav,self.dq,self.dwq,self.dCOSmeasuring_height,self.dz0m,self.dz0h = 0,0,0,0,0,0,0,0
        #ribtol
        self.dzsl, self.dRib = 0,0
        #ags
        self.dalfa_sto,self.dthetasurf,self.dTs,self.devap,self.dCO2,self.dwg,self.dw2,self.dSwin,self.dra,self.dTsoil,self.dcveg,self.de,self.dwwilt,self.dwfc,self.dCOSsurf = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        self.dCO2surf,self.dgciCOS,self.dE0,self.dR10,self.dPARfract = 0,0,0,0,0 #
        #run mixed layer
        self.ddeltatheta,self.ddeltathetav,self.ddeltaq,self.ddeltaCO2,self.ddeltaCOS,self.dM,self.dadvtheta,self.dwqM,self.dadvq = 0,0,0,0,0,0,0,0,0
        self.dwCO2,self.dwCO2M,self.dadvCO2,self.dwCOSM,self.dadvCOS,self.dlcl,self.dustar,self.dgammatheta,self.dgammatheta2,self.dgammaq,self.dgammau,self.dgammav = 0,0,0,0,0,0,0,0,0,0,0,0
        self.dgammaCO2,self.dgammaCOS,self.ddivU,self.du,self.dv,self.dfc,self.ddeltau,self.ddeltav,self.dadvu,self.dadvv,self.ddFz,self.dbeta = 0,0,0,0,0,0,0,0,0,0,0,0
        #integrate mixed layer
        self.ddz_h,self.dhtend,self.dthetatend,self.ddeltathetatend,self.dqtend,self.ddeltaqtend,self.dCO2tend = 0,0,0,0,0,0,0
        self.dCOStend,self.ddeltaCO2tend,self.ddeltaCOStend,self.ddztend,self.dutend,self.dvtend,self.ddeltautend,self.ddeltavtend = 0,0,0,0,0,0,0,0
        #run radiation
        self.ddoy,self.dlat,self.dlon,self.dcc,self.dalpha = 0,0,0,0,0
        #run land surface
        self.dwstar,self.dCs,self.dtheta,self.dq,self.dwfc,self.dwwilt,self.dwg,self.dLAI,self.dWmax,self.dWl,self.dcveg = 0,0,0,0,0,0,0,0,0,0,0
        self.dQ,self.dLambda,self.dTsoil,self.drsmin,self.dwsat,self.dw2,self.dT2 = 0,0,0,0,0,0,0
        self.dCOSsurf, self.dTsurf, self.drssoilmin,self.dCGsat,self.db,self.dC1sat,self.dC2ref,self.da,self.dp = 0,0,0,0,0,0,0,0,0
        #integrate land surface
        self.dTsoiltend,self.dwgtend,self.dWltend = 0,0,0
        #run_cumulus
        self.dwqe,self.dwqM,self.ddeltaq,self.dT_h,self.dP_h,self.ddz_h,self.dwCOSe = 0,0,0,0,0,0,0 #
        #statistics
        self.dtheta,self.dq,self.dwtheta,self.dwq,self.ddeltatheta,self.ddeltaq,self.dh = 0,0,0,0,0,0,0
        #run_soil_COS_mod
        if self.model.input.soilCOSmodeltype == 'Sun_Ogee':
            self.dmol_rat_ocs_atm,self.dairtemp,self.dfCA,self.dVspmax = 0,0,0,0
            self.dC_soilair_current = np.zeros(self.model.input.nr_nodes)
            self.dQ10,self.db_sCOSm = 0,0
        #store
        self.dwthetae,self.dwthetave,self.dwqe,self.dwCO2A,self.dwCO2R,self.dwCO2e,self.dwCO2M,self.duw,self.dvw = 0,0,0,0,0,0,0,0,0
        self.dT2m,self.dthetamh,self.dthetamh2,self.dthetamh3,self.dthetamh4,self.dthetamh5,self.dthetamh6,self.dthetamh7 = 0,0,0,0,0,0,0,0
        self.dTmh,self.dTmh2,self.dTmh3,self.dTmh4,self.dTmh5,self.dTmh6,self.dTmh7,self.dq2m,self.dqmh,self.dqmh2,self.dqmh3,self.dqmh4,self.dqmh5,self.dqmh6,self.dqmh7,self.du2m,self.dv2m,self.de2m,self.desat2m = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        self.dCOSmh,self.dCOSmh2,self.dCOSmh3,self.dCO2mh,self.dCO2mh2,self.dCO2mh3,self.dCO2mh4,self.dCOS2m = 0,0,0,0,0,0,0,0#
        self.dthetavsurf,self.dqsurf,self.dSwout,self.dSwin,self.dLwin,self.dLwout,self.dH,self.dLE,self.dLEliq,self.dLEveg,self.dLEsoil = 0,0,0,0,0,0,0,0,0,0,0
        self.dLEpot,self.dLEref,self.dG,self.dRH_h,self.desatvar,self.dqsatvar,self.dwCOS,self.dwCOSP,self.dwCOSS = 0,0,0,0,0,0,0,0,0
        self.dac,self.dCO22m,self.dCm,self.dL = 0,0,0,0#
        #jarvis_stewart
        self.dgD = 0
        #timestep (tl_full_model)
        self.dwtheta_input = np.zeros(self.model.tsteps)
        self.dwq_input = np.zeros(self.model.tsteps)
        self.dwCO2_input = np.zeros(self.model.tsteps)
        self.dwCOS_input = np.zeros(self.model.tsteps)
        for key in dstate:
            self.__dict__[key] = dstate[key]
    
    def tl_full_model(self,model,checkpoint,checkpoint_init,Hx_variable=None,tl_dict=None,returnvariable=None):
        if self.adjointtesting:
            self.Hx = []
            for i in range(model.tsteps):
                self.Hx.append([]) #because it can be a scalar or something with more dimiensions
        self.tl_init(model,checkpoint_init)
        for i in range(0,model.tsteps):
            if not model.sw_ls: #
                if hasattr(model.input,'wtheta_input'):
                    self.dwtheta = self.dwtheta_input[i]
                if hasattr(model.input,'wq_input'):
                    self.dwq = self.dwq_input[i]
                if hasattr(model.input,'wCO2_input'):
                    self.dwCO2 = self.dwCO2_input[i]
                if hasattr(model.input,'wCOS_input'):
                    self.dwCOS = self.dwCOS_input[i]
            self.tl_statistics(model,checkpoint[i])
            if(model.sw_rad):
                self.tl_run_radiation(model,checkpoint[i])
            if(model.sw_sl):
                self.tl_run_surface_layer(model,checkpoint[i])
            if(model.sw_ls):
                self.tl_run_land_surface(model,checkpoint[i])
            if(model.sw_cu):
                self.tl_run_cumulus(model,checkpoint[i])
            if(model.sw_ml):
                self.tl_run_mixed_layer(model,checkpoint[i])
            self.tl_store(model,checkpoint[i])
            if(model.sw_ls):
                self.tl_integrate_land_surface(model,checkpoint[i])
            if(model.sw_ml):
                self.tl_integrate_mixed_layer(model,checkpoint[i])
            if self.adjointtesting:
                for key in self.__dict__[tl_dict]:
                    if key == Hx_variable:
                        self.Hx[i] = self.__dict__[tl_dict][Hx_variable]
        if self.gradienttesting:
            for key in self.__dict__[tl_dict]: #the dictionary is a self variable
                if key == returnvariable:
                    returnvar = self.__dict__[tl_dict][returnvariable]
                    return returnvar
            
    def tl_init(self,model,checkpoint_init,returnvariable=None): #the tangent linear of the init part)
        if not hasattr(model.input,'gammatheta2'):
            self.dgammatheta2 = self.dgammatheta #we need to include this part of the initialistion of the forward model values, since it does not simply come from input for this case
        fac = model.mair / (model.rho*model.mco2)
        self.dwCO2 = fac * self.dwCO2
        if not model.sw_ls:
            if hasattr(model.input,'wCO2_input'):
                self.dwCO2_input = self.dwCO2_input * fac
        self.tl_statistics(model,checkpoint_init[0]) #0 for all checkpoints, except surf layer, which has iterations
        if(model.sw_rad):
            self.tl_run_radiation(model,checkpoint_init[0]) 
        if(model.sw_sl):
            for i in range(model.nr_of_surf_lay_its): 
                self.tl_run_surface_layer(model,checkpoint_init[i])
        if(model.sw_ls):
            if self.model.input.soilCOSmodeltype == 'Sun_Ogee':
                #the tangent linear belonging to the initialisation of the soil_COS_mod object
                #note that we can make the call here already, since tl_run_land_surface does not calculate the variables we need as input
                self.tl_init_soil_COS_mod(model,checkpoint_init[0])
            self.tl_run_land_surface(model,checkpoint_init[0])
        if(model.sw_cu):
            self.tl_run_mixed_layer(model,checkpoint_init[0])
            self.tl_run_cumulus(model,checkpoint_init[0])        
        if(model.sw_ml):
            self.tl_run_mixed_layer(model,checkpoint_init[0])
            
    def tl_init_soil_COS_mod(self,model,checkpoint_init,returnvariable=None):
        self.Output_tl_isCm = {}
        airtemp = checkpoint_init['isCm_airtemp']
        Rgas = checkpoint_init['isCm_Rgas_end']
        pressure = checkpoint_init['isCm_pressure']
        mol_rat_ocs_atm = checkpoint_init['isCm_mol_rat_ocs_atm']
        dmol_rat_ocs_atm = self.dCOSsurf #this is because COSsurf is in the arg list, while mol_rat_ocs_atm is used in the soil COS model
        dairtemp = self.dTsurf
        dC_air_init = 1.e-9 * pressure / Rgas *(dmol_rat_ocs_atm / airtemp + mol_rat_ocs_atm * -1 * airtemp**(-2) * dairtemp)
        dC_soilair_current = np.zeros(model.soilCOSmodel.nr_nodes)
        for i in range(model.soilCOSmodel.nr_nodes):
            dC_soilair_current[i] = dC_air_init
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_isCm.update({variablename: the_locals[variablename]})  
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_isCm:
                if key in self.__dict__: #otherwise you get a lot of unnecessary vars in memory. If you forget to inititalise a variable in the tl inititalisation it will  give an error anyway
                    self.__dict__[key] = self.Output_tl_isCm[key]
        if returnvariable is not None:
            for key in self.Output_tl_isCm:
                if key == returnvariable:
                    returnvar = self.Output_tl_isCm[returnvariable]
                    return returnvar
                
    def tl_statistics(self,model,checkpoint,returnvariable=None): #tangent linear of the statistics part of the model
        self.Output_tl_stat = {}
        q = checkpoint['stat_q']
        theta = checkpoint['stat_theta']
        wq = checkpoint['stat_wq']
        deltaq = checkpoint['stat_deltaq']
        deltatheta = checkpoint['stat_deltatheta']
        qsat_variable = checkpoint['stat_qsat_variable_end']
        T_h = checkpoint['stat_T_h_end']
        P_h = checkpoint['stat_P_h_end']
        t = checkpoint['stat_t'] #note this is time!!
        it = checkpoint['stat_it_end']
        p_lcl = checkpoint['stat_p_lcl_end']
        T_lcl = checkpoint['stat_T_lcl_end']
        dthetav   = self.dtheta  + 0.61 * (self.dtheta * q + theta * self.dq)
        dwthetav  = self.dwtheta + 0.61 * (self.dtheta * wq + theta * self.dwq)
        ddeltathetav  = (self.dtheta + self.ddeltatheta) * (1. + 0.61 * (q + deltaq)) + (theta + deltatheta) * (0.61 * (self.dq + self.ddeltaq)) - (self.dtheta * (1. + 0.61 * q) + theta * 0.61 * self.dq)
        dP_h    = - model.rho * model.g * self.dh
        dT_h    = self.dtheta - model.g/model.cp * self.dh
        dqsat_variable_dT_H = dqsat_dT(T_h,P_h,dT_h)
        dqsat_variable_dP_H = dqsat_dp(T_h,P_h,dP_h)
        dqsat_variable = dqsat_variable_dT_H + dqsat_variable_dP_H
        dRH_h   = 1 / qsat_variable * self.dq + q * (-1) * qsat_variable**(-2) * dqsat_variable
        if(t == 0):
            dlcl = self.dh
            self.dlcl = dlcl
            dRHlcl = 0.
        else: 
            dRHlcl = 0
        self.dRHlcl = dRHlcl #since dRHlcl should be used in the while loop
        for iteration in range(it):
            dlcl_new     = self.dlcl + -1*self.dRHlcl*1000.
            dp_lcl       = - model.rho * model.g * dlcl_new
            dT_lcl       = self.dtheta - model.g/model.cp * dlcl_new
            dqsat_variable_dp_lcl = dqsat_dp(T_lcl[iteration],p_lcl[iteration],dp_lcl)
            dqsat_variable_dT_lcl = dqsat_dT(T_lcl[iteration],p_lcl[iteration],dT_lcl)
            dRHlcl        = self.dq / fwdm.qsat(T_lcl[iteration], p_lcl[iteration]) + q * -1 * fwdm.qsat(T_lcl[iteration], p_lcl[iteration])**-2 * (dqsat_variable_dp_lcl + dqsat_variable_dT_lcl)
            dlcl = dlcl_new #Don't combine the statements dlcl = dlcl_new and self.dlcl = dlcl into the statement self.dlcl = dlcl_new, because the statement self.__dict__[key] = self.Output_tl_stat[key] some lines below leads to self.dlcl = dlcl
            #(as dlcl defined earlier), which would overwrite the earlier statement self.dlcl = dlcl_new !!! 
            self.dlcl = dlcl #since dlcl should be used again within the for loop, at the first line of the for loop
            self.dRHlcl = dRHlcl
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_stat.update({variablename: the_locals[variablename]})  
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_stat:
                if key in self.__dict__: #otherwise you get a lot of unnecessary vars in memory
                    self.__dict__[key] = self.Output_tl_stat[key]
        if returnvariable is not None:
            for key in self.Output_tl_stat:
                if key == returnvariable:
                    returnvar = self.Output_tl_stat[returnvariable]
                    return returnvar
    
    def tl_run_radiation(self,model,checkpoint,returnvariable=None):
        self.Output_tl_rr = {}
        doy = checkpoint['rr_doy']
        sda = checkpoint['rr_sda_end']
        lat = checkpoint['rr_lat']
        sinlea_lon = checkpoint['rr_sinlea_lon_end']
        sinlea = checkpoint['rr_sinlea_middle'] #we use middle for the value of the first occurence of a variable that is calulated twice in the same module, end is for the value of the last calculation (or for the value of a variable that is only calculated once in a module)
        h = checkpoint['rr_h']
        theta = checkpoint['rr_theta']
        cc = checkpoint['rr_cc']
        Tr = checkpoint['rr_Tr_end']
        alpha = checkpoint['rr_alpha']
        Ts = checkpoint['rr_Ts']
        Ta = checkpoint['rr_Ta_end']
        t = checkpoint['rr_t']
        lon = checkpoint['rr_lon']
        dsda = 0.409 * -np.sin(2. * np.pi * (doy - 173.) / 365.) * 2. * np.pi / 365 * self.ddoy
        dpart1_sinlea = np.sin(sda) * np.cos(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.dlat + np.sin(2. * np.pi * lat / 360.) * np.cos(sda)*dsda
        dsinlea_lon = -np.sin(2. * np.pi * (t * model.dt + model.tstart * 3600.) / 86400. + 2. * np.pi * lon / 360.) * 2. * np.pi / 360. * self.dlon 
        dpart2_sinlea = np.cos(sda) * sinlea_lon * -np.sin(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.dlat + np.cos(2. * np.pi * lat / 360.) * sinlea_lon * -np.sin(sda) * dsda + np.cos(2. * np.pi * lat / 360.) * np.cos(sda) * dsinlea_lon
        dsinlea = dpart1_sinlea - dpart2_sinlea
        if sinlea < 0.0001:
            dsinlea = 0
        sinlea = checkpoint['rr_sinlea_end']
        dTa_dtheta = self.dtheta * ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp)
        dTa_dh = theta * (model.Rd / model.cp) * ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp - 1) * -1 * 0.1 * model.rho * model.g / model.Ps * self.dh
        dTa = dTa_dtheta + dTa_dh
        dTr  = (1. - 0.4 * cc) * 0.2 * dsinlea + (0.6 + 0.2 * sinlea) * - 0.4 * self.dcc
        dSwin = model.S0 * (dTr * sinlea + Tr * dsinlea)
        dSwout = model.S0 * (Tr * sinlea * self.dalpha + alpha * sinlea * dTr + alpha * Tr * dsinlea)
        dLwin = 0.8 * model.bolz * 4 * Ta ** 3. * dTa
        dLwout = model.bolz * 4 * Ts ** 3. * self.dTs
        dQ = dSwin - dSwout + dLwin - dLwout
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rr.update({variablename: the_locals[variablename]})  
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rr:
                if key in self.__dict__: #otherwise you get a lot of unnecessary vars in memory
                    self.__dict__[key] = self.Output_tl_rr[key]
        if returnvariable is not None:
            for key in self.Output_tl_rr:
                if key == returnvariable:
                    returnvar = self.Output_tl_rr[returnvariable]
                    return returnvar

    
    def tl_run_surface_layer(self,model,checkpoint,returnvariable=None):
        self.Output_tl_rsl = {}
        ueff = checkpoint['rsl_ueff_end']
        q = checkpoint['rsl_q']
        qsurf = checkpoint['rsl_qsurf_end']
        #from old qsurf implementation:
#        cq = checkpoint['rsl_cq_end']
#        rs = checkpoint['rsl_rs']
#        qsatsurf_rsl = checkpoint['rsl_qsatsurf_rsl_end']
        thetasurf = checkpoint['rsl_thetasurf_end']
        wq = checkpoint['rsl_wq']
        thetav = checkpoint['rsl_thetav']
        wthetav = checkpoint['rsl_wthetav']
        zsl = checkpoint['rsl_zsl_end']
        L = checkpoint['rsl_L_end']
        wtheta = checkpoint['rsl_wtheta']
        Cm = checkpoint['rsl_Cm_end']
        ustar = checkpoint['rsl_ustar_end']
        Cs_start = checkpoint['rsl_Cs']
        ustar_start = checkpoint['rsl_ustar']
        u = checkpoint['rsl_u']
        v = checkpoint['rsl_v']
        wstar = checkpoint['rsl_wstar']
        wCOS = checkpoint['rsl_wCOS']
        wCO2 = checkpoint['rsl_wCO2']
        uw = checkpoint['rsl_uw_end']
        vw = checkpoint['rsl_vw_end']
        T2m = checkpoint['rsl_T2m_end']
        COSmeasuring_height = checkpoint['rsl_COSmeasuring_height']
        z0m = checkpoint['rsl_z0m']
        z0h = checkpoint['rsl_z0h']
        thetavsurf = checkpoint['rsl_thetavsurf_end']
        if model.sw_use_ribtol:
            Rib = checkpoint['rsl_Rib_middle']
        if np.sqrt(u**2. + v**2. + wstar**2.) < 0.01:
            dueff = 0
        else:
            dueff = 0.5 * (u**2. + v**2. + wstar**2.)**(-0.5) * (2 * u * self.du + 2 * v * self.dv + 2 * wstar * self.dwstar)
        dCOSsurf_dwCOS = self.dwCOS / (Cs_start * ueff)
        dCOSsurf_dCOS = self.dCOS
        dCOSsurf_dCs_start = wCOS / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        dCOSsurf_dueff = wCOS / Cs_start * (-1) * ueff**(-2) * dueff
        dCOSsurf = dCOSsurf_dwCOS + dCOSsurf_dCOS + dCOSsurf_dCs_start + dCOSsurf_dueff 
        dCO2surf_dwCO2 = self.dwCO2 / (Cs_start * ueff)
        dCO2surf_dCO2 = self.dCO2
        dCO2surf_dCs_start = wCO2 / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        dCO2surf_dueff = wCO2 / Cs_start * (-1) * ueff**(-2) * dueff
        dCO2surf = dCO2surf_dwCO2 + dCO2surf_dCO2 + dCO2surf_dCs_start + dCO2surf_dueff 
        dthetasurf_dtheta = self.dtheta #dthetasurf_dtheta means dthetasurf/dtheta * dtheta actually
        
        dthetasurf_dwtheta = self.dwtheta/(Cs_start * ueff)
        dthetasurf_dCs_start = wtheta / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        dthetasurf_dueff = wtheta / Cs_start * (-1) * ueff**(-2) * dueff
        if self.manualadjointtesting:
            dthetasurf_dtheta = self.x
        dthetasurf = dthetasurf_dtheta + dthetasurf_dwtheta + dthetasurf_dCs_start + dthetasurf_dueff    
        dTsurf = dthetasurf * (100000/model.Ps)**(-model.Rd/model.cp)
        if self.manualadjointtesting:
            self.Hx = dTsurf
        #   Below the original, problematic way of calculating qsurf
#        dqsatsurf_rsl = dqsat_dT(thetasurf, model.Ps, dthetasurf)
#        dcq_dCs_start = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * rs * self.dCs_start
#        dcq_dueff = -1 * (1. + Cs_start * ueff * rs) ** -2. * Cs_start * rs * dueff
#        dcq_drs = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * Cs_start * self.drs
#        dcq = dcq_dCs_start + dcq_dueff + dcq_drs
#        dqsurf_dq    = (1. - cq) * self.dq
#        dqsurf_dcq = (-q + qsatsurf_rsl) * dcq
#        dqsurf_dqsatsurf_rsl = cq * dqsatsurf_rsl
#        dqsurf = dqsurf_dq + dqsurf_dcq + dqsurf_dqsatsurf_rsl
        #here the new way of calculating qsurf
        dqsurf_dq = self.dq
        dqsurf_dwq = self.dwq/(Cs_start * ueff)
        dqsurf_dCs_start = wq / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        dqsurf_dueff = wq / Cs_start * (-1) * ueff**(-2) * dueff
        dqsurf = dqsurf_dq + dqsurf_dwq + dqsurf_dCs_start + dqsurf_dueff 
        desurf = dqsurf * model.Ps / 0.622
        dthetavsurf = dthetasurf * (1. + 0.61 * qsurf) + 0.61 * thetasurf * dqsurf
        dzsl      = 0.1 * self.dh
        self.dzsl = dzsl #needed before we call tl_ribtol
        if model.sw_use_ribtol:
            dRib_dthetav = model.g * -1 * thetav**-2 * self.dthetav * zsl * (thetav - thetavsurf) / ueff**2. + model.g / thetav * zsl / ueff**2. * self.dthetav
            dRib_dzsl = model.g / thetav * (thetav - thetavsurf) / ueff**2. * dzsl
            dRib_dthetavsurf = model.g / thetav * zsl / ueff**2. * -1 * dthetavsurf
            dRib_dueff = model.g / thetav * zsl * (thetav - thetavsurf) * -2 * ueff**-3 * dueff
            if Rib > 0.2:
                dRib_dthetav = 0.
                dRib_dzsl = 0.
                dRib_dthetavsurf = 0.
                dRib_dueff = 0.
            dRib = dRib_dthetav + dRib_dzsl + dRib_dthetavsurf + dRib_dueff
            self.dRib = dRib #needed before we call tl_ribtol
            dL = self.tl_ribtol(model,checkpoint,returnvariable='dL')
        else:
            dL_dthetav = ustar_start**3 /(model.k * model.g * -1 * wthetav) * self.dthetav
            dL_dustar_start = thetav * 3*ustar_start**2 /(model.k * model.g * -1 * wthetav) * self.dustar_start
            dL_dwthetav = thetav * ustar_start**3 /(model.k * model.g * -1) * (-1) * wthetav**(-2) * self.dwthetav
            dL = dL_dthetav + dL_dustar_start + dL_dwthetav
        dpsim_term_for_dCm_dzsl = self.dpsim(zsl / L, 1/L*dzsl)
        dzeta_dL_z0m = self.dzeta_dL(z0m,L) * dL
        dzeta_dL_zsl = self.dzeta_dL(zsl,L) * dL
        dzeta_dL_z0h = self.dzeta_dL(z0h,L) * dL
        dzeta_dL_2 = self.dzeta_dL(2.,L) * dL
        dzeta_dL_Tmh = self.dzeta_dL(model.Tmeasuring_height,L) * dL
        dzeta_dL_Tmh2 = self.dzeta_dL(model.Tmeasuring_height2,L) * dL
        dzeta_dL_Tmh3 = self.dzeta_dL(model.Tmeasuring_height3,L) * dL
        dzeta_dL_Tmh4 = self.dzeta_dL(model.Tmeasuring_height4,L) * dL
        dzeta_dL_Tmh5 = self.dzeta_dL(model.Tmeasuring_height5,L) * dL
        dzeta_dL_Tmh6 = self.dzeta_dL(model.Tmeasuring_height6,L) * dL
        dzeta_dL_Tmh7 = self.dzeta_dL(model.Tmeasuring_height7,L) * dL
        dzeta_dL_qmh = self.dzeta_dL(model.qmeasuring_height,L) * dL
        dzeta_dL_qmh2 = self.dzeta_dL(model.qmeasuring_height2,L) * dL
        dzeta_dL_qmh3 = self.dzeta_dL(model.qmeasuring_height3,L) * dL
        dzeta_dL_qmh4 = self.dzeta_dL(model.qmeasuring_height4,L) * dL
        dzeta_dL_qmh5 = self.dzeta_dL(model.qmeasuring_height5,L) * dL
        dzeta_dL_qmh6 = self.dzeta_dL(model.qmeasuring_height6,L) * dL
        dzeta_dL_qmh7 = self.dzeta_dL(model.qmeasuring_height7,L) * dL
        dzeta_dL_COSmh = self.dzeta_dL(COSmeasuring_height,L) * dL
        dzeta_dL_COSmh2 = self.dzeta_dL(model.COSmeasuring_height2,L) * dL
        dzeta_dL_COSmh3 = self.dzeta_dL(model.COSmeasuring_height3,L) * dL
        dzeta_dL_CO2mh = self.dzeta_dL(model.CO2measuring_height,L) * dL
        dzeta_dL_CO2mh2 = self.dzeta_dL(model.CO2measuring_height2,L) * dL
        dzeta_dL_CO2mh3 = self.dzeta_dL(model.CO2measuring_height3,L) * dL
        dzeta_dL_CO2mh4 = self.dzeta_dL(model.CO2measuring_height4,L) * dL
        dpsimterm_for_Cm_zsl = self.dpsim(zsl / L,dzeta_dL_zsl)
        dpsimterm_for_Cm_z0m = self.dpsim(z0m / L,dzeta_dL_z0m)
        dpsimterm_for_dCm_dz0m = self.dpsim(z0m / L,1 / L * self.dz0m) 
        constant_for_Cm = model.k**2. *(-2) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) ** (-3)
        dCm_dzsl = constant_for_Cm * (1/(zsl / z0m) * 1/z0m * dzsl - dpsim_term_for_dCm_dzsl)
        dCm_dz0m = constant_for_Cm * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        dCm_dL = constant_for_Cm * (-1* dpsimterm_for_Cm_zsl + dpsimterm_for_Cm_z0m)
        dCm = dCm_dzsl + dCm_dz0m + dCm_dL
        dpsim_term_for_dCs_dzsl = self.dpsim(zsl / L, 1/L*dzsl)
        dpsih_term_for_dCs_dzsl = self.dpsih(zsl / L, 1/L*dzsl)
        constant_for_Cs = model.k**2. *(-1) * ((np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))) ** (-2)
        constant2_for_Cs = model.k**2. *(-1) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L))**-2 / (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))
        dCs_dzsl = constant_for_Cs * ((1/(zsl / z0m) * 1/z0m * dzsl - dpsim_term_for_dCs_dzsl) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) + (1/(zsl / z0h) * 1/z0h * dzsl - dpsih_term_for_dCs_dzsl ) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) )
        dCs_dz0m = constant2_for_Cs * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        dpsimterm_for_dCs_dL = (- self.dpsim(zsl / L,dzeta_dL_zsl)+self.dpsim(z0m / L,dzeta_dL_z0m))
        dpsihterm_for_dCs_dL = (- self.dpsih(zsl / L,dzeta_dL_zsl)+ self.dpsih(z0h / L,dzeta_dL_z0h))
        dCs_dL = constant_for_Cs * ((np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))*dpsimterm_for_dCs_dL + 
                      (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) *dpsihterm_for_dCs_dL)
        dpsihterm_for_dCs_dz0h = self.dpsih(z0h / L,1 / L * self.dz0h)
        constant3_for_Cs = model.k**2. / (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) * -1  * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) ** (-2)
        dCs_dz0h = constant3_for_Cs * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dCs = dCs_dzsl + dCs_dz0m + dCs_dL + dCs_dz0h
        
        dustar = 0.5*(Cm)**(-0.5) * ueff * dCm + Cm**(0.5) * dueff
        if model.updatevals_surf_lay:
            dCs_start = dCs
            dustar_start = dustar
        duw    = - (dCm * ueff * u + Cm * u * dueff + Cm * ueff * self.du)
        dvw    = - (dCm * ueff * v + Cm * v * dueff + Cm * ueff * self.dv)
        dT2m_dthetasurf = dthetasurf
        dT2m_dwtheta = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwtheta
        dT2m_dustar = - wtheta / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dT2m_dz0h = - wtheta / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_2_L = self.dpsih(2. / L,dzeta_dL_2)
        dpsih_z0h_L = self.dpsih(z0h / L,dzeta_dL_z0h)
        dT2m_dL = - wtheta / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        dT2m = dT2m_dthetasurf + dT2m_dwtheta + dT2m_dustar + dT2m_dz0h + dT2m_dL
        dthetamh_dthetasurf = dthetasurf
        dthetamh_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height / z0h) * model.Tmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh_L = self.dpsih(model.Tmeasuring_height / L,dzeta_dL_Tmh)
        dthetamh_dL = - wtheta / ustar / model.k * (- dpsih_Tmh_L + dpsih_z0h_L)
        dthetamh = dthetamh_dthetasurf + dthetamh_dwtheta + dthetamh_dustar + dthetamh_dz0h + dthetamh_dL
        dTmh = dthetamh * ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp)
        dthetamh2_dthetasurf = dthetasurf
        dthetamh2_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh2_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh2_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height2 / z0h) * model.Tmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh2_L = self.dpsih(model.Tmeasuring_height2 / L,dzeta_dL_Tmh2)
        dthetamh2_dL = - wtheta / ustar / model.k * (- dpsih_Tmh2_L + dpsih_z0h_L)
        dthetamh2 = dthetamh2_dthetasurf + dthetamh2_dwtheta + dthetamh2_dustar + dthetamh2_dz0h + dthetamh2_dL
        dTmh2 = dthetamh2 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp)
        dthetamh3_dthetasurf = dthetasurf
        dthetamh3_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh3_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh3_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height3 / z0h) * model.Tmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh3_L = self.dpsih(model.Tmeasuring_height3 / L,dzeta_dL_Tmh3)
        dthetamh3_dL = - wtheta / ustar / model.k * (- dpsih_Tmh3_L + dpsih_z0h_L)
        dthetamh3 = dthetamh3_dthetasurf + dthetamh3_dwtheta + dthetamh3_dustar + dthetamh3_dz0h + dthetamh3_dL
        dTmh3 = dthetamh3 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp)
        dthetamh4_dthetasurf = dthetasurf
        dthetamh4_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh4_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh4_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height4 / z0h) * model.Tmeasuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh4_L = self.dpsih(model.Tmeasuring_height4 / L,dzeta_dL_Tmh4)
        dthetamh4_dL = - wtheta / ustar / model.k * (- dpsih_Tmh4_L + dpsih_z0h_L)
        dthetamh4 = dthetamh4_dthetasurf + dthetamh4_dwtheta + dthetamh4_dustar + dthetamh4_dz0h + dthetamh4_dL
        dTmh4 = dthetamh4 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp)
        dthetamh5_dthetasurf = dthetasurf
        dthetamh5_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh5_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh5_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height5 / z0h) * model.Tmeasuring_height5 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh5_L = self.dpsih(model.Tmeasuring_height5 / L,dzeta_dL_Tmh5)
        dthetamh5_dL = - wtheta / ustar / model.k * (- dpsih_Tmh5_L + dpsih_z0h_L)
        dthetamh5 = dthetamh5_dthetasurf + dthetamh5_dwtheta + dthetamh5_dustar + dthetamh5_dz0h + dthetamh5_dL
        dTmh5 = dthetamh5 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp)
        dthetamh6_dthetasurf = dthetasurf
        dthetamh6_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh6_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh6_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height6 / z0h) * model.Tmeasuring_height6 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh6_L = self.dpsih(model.Tmeasuring_height6 / L,dzeta_dL_Tmh6)
        dthetamh6_dL = - wtheta / ustar / model.k * (- dpsih_Tmh6_L + dpsih_z0h_L)
        dthetamh6 = dthetamh6_dthetasurf + dthetamh6_dwtheta + dthetamh6_dustar + dthetamh6_dz0h + dthetamh6_dL
        dTmh6 = dthetamh6 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp)
        dthetamh7_dthetasurf = dthetasurf
        dthetamh7_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * self.dwtheta
        dthetamh7_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dthetamh7_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height7 / z0h) * model.Tmeasuring_height7 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_Tmh7_L = self.dpsih(model.Tmeasuring_height7 / L,dzeta_dL_Tmh7)
        dthetamh7_dL = - wtheta / ustar / model.k * (- dpsih_Tmh7_L + dpsih_z0h_L)
        dthetamh7 = dthetamh7_dthetasurf + dthetamh7_dwtheta + dthetamh7_dustar + dthetamh7_dz0h + dthetamh7_dL
        dTmh7 = dthetamh7 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp)
        dq2m_dqsurf = dqsurf
        dq2m_dwq = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwq
        #note that the following derivative is not a real derivative to ustar!!
#        #this because not only ustar is influenced by a change in ustar, but also L. With dq2m_dustar we mean here the derivative of q2m to a change in the variable ustar as occuring in the formula for q2m in the forward model,
#        #meaning that it does not include the indirect changes of q2m via L due to a change in ustar        
        dq2m_dustar = - wq / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dq2m_dz0h = - wq / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dq2m_dL = - wq / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        dq2m = dq2m_dqsurf + dq2m_dwq + dq2m_dustar + dq2m_dz0h + dq2m_dL
        dqmh_dqsurf = dqsurf
        dqmh_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * self.dwq
        dqmh_dustar = - wq / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height / z0h) * model.qmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh_L = self.dpsih(model.qmeasuring_height / L,dzeta_dL_qmh)
        dqmh_dL = - wq / ustar / model.k * (- dpsih_qmh_L + dpsih_z0h_L)
        dqmh = dqmh_dqsurf + dqmh_dwq + dqmh_dustar + dqmh_dz0h + dqmh_dL
        dqmh2_dqsurf = dqsurf
        dqmh2_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwq
        dqmh2_dustar = - wq / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh2_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height2 / z0h) * model.qmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh2_L = self.dpsih(model.qmeasuring_height2 / L,dzeta_dL_qmh2)
        dqmh2_dL = - wq / ustar / model.k * (- dpsih_qmh2_L + dpsih_z0h_L)
        dqmh2 = dqmh2_dqsurf + dqmh2_dwq + dqmh2_dustar + dqmh2_dz0h + dqmh2_dL
        dqmh3_dqsurf = dqsurf
        dqmh3_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwq
        dqmh3_dustar = - wq / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh3_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height3 / z0h) * model.qmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh3_L = self.dpsih(model.qmeasuring_height3 / L,dzeta_dL_qmh3)
        dqmh3_dL = - wq / ustar / model.k * (- dpsih_qmh3_L + dpsih_z0h_L)
        dqmh3 = dqmh3_dqsurf + dqmh3_dwq + dqmh3_dustar + dqmh3_dz0h + dqmh3_dL
        dqmh4_dqsurf = dqsurf
        dqmh4_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * self.dwq
        dqmh4_dustar = - wq / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh4_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height4 / z0h) * model.qmeasuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh4_L = self.dpsih(model.qmeasuring_height4 / L,dzeta_dL_qmh4)
        dqmh4_dL = - wq / ustar / model.k * (- dpsih_qmh4_L + dpsih_z0h_L)
        dqmh4 = dqmh4_dqsurf + dqmh4_dwq + dqmh4_dustar + dqmh4_dz0h + dqmh4_dL
        dqmh5_dqsurf = dqsurf
        dqmh5_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * self.dwq
        dqmh5_dustar = - wq / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh5_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height5 / z0h) * model.qmeasuring_height5 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh5_L = self.dpsih(model.qmeasuring_height5 / L,dzeta_dL_qmh5)
        dqmh5_dL = - wq / ustar / model.k * (- dpsih_qmh5_L + dpsih_z0h_L)
        dqmh5 = dqmh5_dqsurf + dqmh5_dwq + dqmh5_dustar + dqmh5_dz0h + dqmh5_dL
        dqmh6_dqsurf = dqsurf
        dqmh6_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * self.dwq
        dqmh6_dustar = - wq / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh6_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height6 / z0h) * model.qmeasuring_height6 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh6_L = self.dpsih(model.qmeasuring_height6 / L,dzeta_dL_qmh6)
        dqmh6_dL = - wq / ustar / model.k * (- dpsih_qmh6_L + dpsih_z0h_L)
        dqmh6 = dqmh6_dqsurf + dqmh6_dwq + dqmh6_dustar + dqmh6_dz0h + dqmh6_dL
        dqmh7_dqsurf = dqsurf
        dqmh7_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * self.dwq
        dqmh7_dustar = - wq / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dqmh7_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height7 / z0h) * model.qmeasuring_height7 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_qmh7_L = self.dpsih(model.qmeasuring_height7 / L,dzeta_dL_qmh7)
        dqmh7_dL = - wq / ustar / model.k * (- dpsih_qmh7_L + dpsih_z0h_L)
        dqmh7 = dqmh7_dqsurf + dqmh7_dwq + dqmh7_dustar + dqmh7_dz0h + dqmh7_dL
        dCOS2m_dCOSsurf = dCOSsurf
        dCOS2m_dwCOS = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwCOS
        dCOS2m_dustar = - wCOS / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCOS2m_dz0h = - wCOS / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dCOS2m_dL = - wCOS / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        dCOS2m = dCOS2m_dCOSsurf + dCOS2m_dwCOS + dCOS2m_dustar + dCOS2m_dz0h + dCOS2m_dL
        dCOSmh_dCOSsurf = dCOSsurf
        dCOSmh_dwCOS = - 1 / ustar / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * self.dwCOS
        dCOSmh_dustar = - wCOS / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dzeta_dCOSmh = 1 / L * self.dCOSmeasuring_height
        dpsih_COSmh_L_num = self.dpsih(COSmeasuring_height / L,dzeta_dCOSmh) #num means the variable to which we take the derivative is in the numerator
        dCOSmh_dCOSmeasuring_height = - wCOS / ustar / model.k * (1 / (COSmeasuring_height / z0h) * 1 / z0h * self.dCOSmeasuring_height - dpsih_COSmh_L_num)
        dCOSmh_dz0h = - wCOS / ustar / model.k * (1 / (COSmeasuring_height / z0h) * COSmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_COSmh_L = self.dpsih(COSmeasuring_height / L,dzeta_dL_COSmh)
        dCOSmh_dL = - wCOS / ustar / model.k * (- dpsih_COSmh_L + dpsih_z0h_L)
        dCOSmh = dCOSmh_dCOSsurf + dCOSmh_dwCOS + dCOSmh_dustar + dCOSmh_dCOSmeasuring_height + dCOSmh_dz0h + dCOSmh_dL
        dCOSmh2_dCOSsurf = dCOSsurf
        dCOSmh2_dwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwCOS
        dCOSmh2_dustar = - wCOS / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCOSmh2_dz0h = - wCOS / ustar / model.k * (1 / (model.COSmeasuring_height2 / z0h) * model.COSmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_COSmh2_L = self.dpsih(model.COSmeasuring_height2 / L,dzeta_dL_COSmh2)
        dCOSmh2_dL = - wCOS / ustar / model.k * (- dpsih_COSmh2_L + dpsih_z0h_L)
        dCOSmh2 = dCOSmh2_dCOSsurf + dCOSmh2_dwCOS + dCOSmh2_dustar + dCOSmh2_dz0h + dCOSmh2_dL
        dCOSmh3_dCOSsurf = dCOSsurf
        dCOSmh3_dwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwCOS
        dCOSmh3_dustar = - wCOS / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCOSmh3_dz0h = - wCOS / ustar / model.k * (1 / (model.COSmeasuring_height3 / z0h) * model.COSmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_COSmh3_L = self.dpsih(model.COSmeasuring_height3 / L,dzeta_dL_COSmh3)
        dCOSmh3_dL = - wCOS / ustar / model.k * (- dpsih_COSmh3_L + dpsih_z0h_L)
        dCOSmh3 = dCOSmh3_dCOSsurf + dCOSmh3_dwCOS + dCOSmh3_dustar + dCOSmh3_dz0h + dCOSmh3_dL
        dCO22m_dCO2surf = dCO2surf
        dCO22m_dwCO2 = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwCO2
        dCO22m_dustar = - wCO2 / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCO22m_dz0h = - wCO2 / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dCO22m_dL = - wCO2 / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        dCO22m = dCO22m_dCO2surf + dCO22m_dwCO2 + dCO22m_dustar + dCO22m_dz0h + dCO22m_dL
        dCO2mh_dCO2surf = dCO2surf
        dCO2mh_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * self.dwCO2
        dCO2mh_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCO2mh_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height / z0h) * model.CO2measuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_CO2mh_L = self.dpsih(model.CO2measuring_height / L,dzeta_dL_CO2mh)
        dCO2mh_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh_L + dpsih_z0h_L)
        dCO2mh = dCO2mh_dCO2surf + dCO2mh_dwCO2 + dCO2mh_dustar + dCO2mh_dz0h + dCO2mh_dL
        dCO2mh2_dCO2surf = dCO2surf
        dCO2mh2_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * self.dwCO2
        dCO2mh2_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCO2mh2_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height2 / z0h) * model.CO2measuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_CO2mh2_L = self.dpsih(model.CO2measuring_height2 / L,dzeta_dL_CO2mh2)
        dCO2mh2_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh2_L + dpsih_z0h_L)
        dCO2mh2 = dCO2mh2_dCO2surf + dCO2mh2_dwCO2 + dCO2mh2_dustar + dCO2mh2_dz0h + dCO2mh2_dL
        dCO2mh3_dCO2surf = dCO2surf
        dCO2mh3_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * self.dwCO2
        dCO2mh3_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCO2mh3_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height3 / z0h) * model.CO2measuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_CO2mh3_L = self.dpsih(model.CO2measuring_height3 / L,dzeta_dL_CO2mh3)
        dCO2mh3_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh3_L + dpsih_z0h_L)
        dCO2mh3 = dCO2mh3_dCO2surf + dCO2mh3_dwCO2 + dCO2mh3_dustar + dCO2mh3_dz0h + dCO2mh3_dL
        dCO2mh4_dCO2surf = dCO2surf
        dCO2mh4_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * self.dwCO2
        dCO2mh4_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        dCO2mh4_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height4 / z0h) * model.CO2measuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        dpsih_CO2mh4_L = self.dpsih(model.CO2measuring_height4 / L,dzeta_dL_CO2mh4)
        dCO2mh4_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh4_L + dpsih_z0h_L)
        dCO2mh4 = dCO2mh4_dCO2surf + dCO2mh4_dwCO2 + dCO2mh4_dustar + dCO2mh4_dz0h + dCO2mh4_dL        
        du2m_duw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * duw
        du2m_dustar = - uw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * dustar
        du2m_dz0m = - uw / ustar / model.k * (1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        dpsim_2_L = self.dpsim(2. / L,dzeta_dL_2)
        dpsim_z0m_L = self.dpsim(z0m / L,dzeta_dL_z0m)
        du2m_dL = - uw / ustar / model.k * (- dpsim_2_L + dpsim_z0m_L)
        du2m = du2m_duw + du2m_dustar + du2m_dz0m + du2m_dL
        dv2m_dvw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * dvw
        dv2m_dustar = - vw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * dustar
        dv2m_dz0m = - vw / ustar / model.k * (1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        dv2m_dL = - vw / ustar / model.k * (- dpsim_2_L + dpsim_z0m_L)
        dv2m = dv2m_dvw + dv2m_dustar + dv2m_dz0m + dv2m_dL
        desat2m = desat(T2m,dT2m)
        de2m = 1 * model.Ps / 0.622 * dq2m
        if model.sw_dynamicsl_border:
            if model.Tmeasuring_height > zsl:
                dthetamh = self.dtheta
                dTmh = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height2 > zsl:
                dthetamh2 = self.dtheta
                dTmh2 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height3 > zsl:
                dthetamh3 = self.dtheta
                dTmh3 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height4 > zsl:
                dthetamh4 = self.dtheta
                dTmh4 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height5 > zsl:
                dthetamh5 = self.dtheta
                dTmh5 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height6 > zsl:
                dthetamh6 = self.dtheta
                dTmh6 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp)
            if model.Tmeasuring_height7 > zsl:
                dthetamh7 = self.dtheta
                dTmh7 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp)
            if model.qmeasuring_height > zsl:
                dqmh = self.dq
            if model.qmeasuring_height2 > zsl:
                dqmh2 = self.dq
            if model.qmeasuring_height3 > zsl:
                dqmh3 = self.dq
            if model.qmeasuring_height4 > zsl:
                dqmh4 = self.dq
            if model.qmeasuring_height5 > zsl:
                dqmh5 = self.dq
            if model.qmeasuring_height6 > zsl:
                dqmh6 = self.dq
            if model.qmeasuring_height7 > zsl:
                dqmh7 = self.dq
            if COSmeasuring_height > zsl:
                dCOSmh = self.dCOS
            if model.COSmeasuring_height2 > zsl:
                dCOSmh2 = self.dCOS
            if model.COSmeasuring_height3 > zsl:
                dCOSmh3 = self.dCOS
            if model.CO2measuring_height > zsl:
                dCO2mh = self.dCO2
            if model.CO2measuring_height2 > zsl:
                dCO2mh2 = self.dCO2
            if model.CO2measuring_height3 > zsl:
                dCO2mh3 = self.dCO2
            if model.CO2measuring_height4 > zsl:
                dCO2mh4 = self.dCO2
            
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rsl.update({variablename: the_locals[variablename]})              
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rsl:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rsl[key]
        if returnvariable is not None:
            for key in self.Output_tl_rsl:
                if key == returnvariable:
                    returnvar = self.Output_tl_rsl[returnvariable]
                    return returnvar
    
    def tl_ribtol(self,model,checkpoint,returnvariable=None):
        self.Output_tl_rtl = {} #rtl means Rib to L
        it = checkpoint['rtl_it_end']
        zsl = checkpoint['rtl_zsl']
        z0m = checkpoint['rtl_z0m']
        z0h = checkpoint['rtl_z0h']
        L = checkpoint['rtl_L_middle']
        Lstart = checkpoint['rtl_Lstart_end']
        Lend = checkpoint['rtl_Lend_end']
        fxdif_part1 = checkpoint['rtl_fxdif_part1_end']
        fxdif_part2 = checkpoint['rtl_fxdif_part2_end']
        fx = checkpoint['rtl_fx_end']
        fxdif = checkpoint['rtl_fxdif_end']
        dL = 0.
        dL0 = 0.
        for i in range(it):
            dL0      = dL
            #fx      = Rib - zsl / L * (np.log(zsl / z0h) - self.psih(zsl / L) + self.psih(z0h / L)) / (np.log(zsl / z0m) - self.psim(zsl / L) + self.psim(z0m / L))**2.
            dfx_dRib = self.dRib
            dzeta_dL_zsl = self.dzeta_dL(zsl,L[i]) * dL
            dzeta_dL_z0h = self.dzeta_dL(z0h,L[i]) * dL
            dzeta_dL_z0m = self.dzeta_dL(z0m,L[i]) * dL
            dpsihterm_for_dfx_dzsl = self.dpsih(zsl / L[i],1 / L[i] * self.dzsl)
            dpsimterm_for_dfx_dzsl = self.dpsim(zsl / L[i],1 / L[i] * self.dzsl)
            dpsihterm1_for_dfx_dL = self.dpsih(zsl / L[i],dzeta_dL_zsl)
            dpsihterm2_for_dfx_dL = self.dpsih(z0h / L[i],dzeta_dL_z0h)
            dpsimterm1_for_dfx_dL = self.dpsim(zsl / L[i],dzeta_dL_zsl)
            dpsimterm2_for_dfx_dL = self.dpsim(z0m/ L[i],dzeta_dL_z0m)
            dpsimterm_for_dfx_dz0m = self.dpsim(z0m/ L[i],1 / L[i] * self.dz0m)
            dpsihterm_for_dfx_dz0h = self.dpsih(z0h / L[i],1 / L[i] * self.dz0h)
            dfx_dzsl = - self.dzsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. + \
                       - zsl / L[i] * ((1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfx_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. + \
                        (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfx_dzsl))
            dfx_dL   = - zsl * -1 * L[i]**-2 * dL * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. +\
                       - zsl / L[i] * ((- dpsihterm1_for_dfx_dL + dpsihterm2_for_dfx_dL) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. +
                        (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (- dpsimterm1_for_dfx_dL + dpsimterm2_for_dfx_dL))
            dfx_dz0m = - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfx_dz0m)
            dfx_dz0h = - zsl / L[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfx_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2
            dfx = dfx_dRib + dfx_dzsl + dfx_dL + dfx_dz0m + dfx_dz0h
            dLstart  = dL - 0.001*dL
            dLend    = dL + 0.001*dL
            dzeta_dLstart_zsl = self.dzeta_dL(zsl,Lstart[i]) * dLstart
            dzeta_dLstart_z0h = self.dzeta_dL(z0h,Lstart[i]) * dLstart
            dzeta_dLstart_z0m = self.dzeta_dL(z0m,Lstart[i]) * dLstart
            dpsihterm_for_dfxdif_part1_dzsl = self.dpsih(zsl / Lstart[i], 1 / Lstart[i] * self.dzsl)
            dpsimterm_for_dfxdif_part1_dzsl = self.dpsim(zsl / Lstart[i], 1 / Lstart[i] * self.dzsl)
            dpsihterm1_for_dfxdif_part1_dLstart = self.dpsih(zsl / Lstart[i], dzeta_dLstart_zsl)
            dpsihterm2_for_dfxdif_part1_dLstart = self.dpsih(z0h / Lstart[i], dzeta_dLstart_z0h)
            dpsimterm1_for_dfxdif_part1_dLstart = self.dpsim(zsl / Lstart[i], dzeta_dLstart_zsl)
            dpsimterm2_for_dfxdif_part1_dLstart = self.dpsim(z0m / Lstart[i], dzeta_dLstart_z0m)
            dpsihterm_for_dfxdif_part1_dz0h = self.dpsih(z0h / Lstart[i], 1 / Lstart[i] * self.dz0h)
            dpsimterm_for_dfxdif_part1_dz0m = self.dpsim(z0m / Lstart[i], 1 / Lstart[i] * self.dz0m)
            dfxdif_part1_dzsl = - self.dzsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                                - zsl / Lstart[i] * (1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfxdif_part1_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                                - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfxdif_part1_dzsl)
            dfxdif_part1_dLstart = - zsl * -1 * Lstart[i]**-2 * dLstart * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                                   - zsl / Lstart[i] * (-dpsihterm1_for_dfxdif_part1_dLstart + dpsihterm2_for_dfxdif_part1_dLstart) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. +\
                                   - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (-dpsimterm1_for_dfxdif_part1_dLstart + dpsimterm2_for_dfxdif_part1_dLstart)
            dfxdif_part1_dz0h = - zsl / Lstart[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfxdif_part1_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. 
            dfxdif_part1_dz0m = - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfxdif_part1_dz0m)
            dfxdif_part1 = dfxdif_part1_dzsl + dfxdif_part1_dLstart + dfxdif_part1_dz0h + dfxdif_part1_dz0m

            dzeta_dLend_zsl = self.dzeta_dL(zsl,Lend[i]) * dLend
            dzeta_dLend_z0h = self.dzeta_dL(z0h,Lend[i]) * dLend
            dzeta_dLend_z0m = self.dzeta_dL(z0m,Lend[i]) * dLend
            dpsihterm_for_dfxdif_part2_dzsl = self.dpsih(zsl / Lend[i], 1 / Lend[i] * self.dzsl)
            dpsimterm_for_dfxdif_part2_dzsl = self.dpsim(zsl / Lend[i], 1 / Lend[i] * self.dzsl)
            dpsihterm1_for_dfxdif_part2_dLend = self.dpsih(zsl / Lend[i], dzeta_dLend_zsl)
            dpsihterm2_for_dfxdif_part2_dLend = self.dpsih(z0h / Lend[i], dzeta_dLend_z0h)
            dpsimterm1_for_dfxdif_part2_dLend = self.dpsim(zsl / Lend[i], dzeta_dLend_zsl)
            dpsimterm2_for_dfxdif_part2_dLend = self.dpsim(z0m / Lend[i], dzeta_dLend_z0m)
            dpsihterm_for_dfxdif_part2_dz0h = self.dpsih(z0h / Lend[i], 1 / Lend[i] * self.dz0h)
            dpsimterm_for_dfxdif_part2_dz0m = self.dpsim(z0m / Lend[i], 1 / Lend[i] * self.dz0m)
            dfxdif_part2_dzsl = -1 * (- self.dzsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                                - zsl / Lend[i] * (1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfxdif_part2_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                                - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfxdif_part2_dzsl))
            dfxdif_part2_dLend = -1 * (- zsl * -1 * Lend[i]**-2 * dLend * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                                   - zsl / Lend[i] * (-dpsihterm1_for_dfxdif_part2_dLend + dpsihterm2_for_dfxdif_part2_dLend) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. +\
                                   - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (-dpsimterm1_for_dfxdif_part2_dLend + dpsimterm2_for_dfxdif_part2_dLend))
            dfxdif_part2_dz0h = -1 * - zsl / Lend[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfxdif_part2_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. 
            dfxdif_part2_dz0m = -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfxdif_part2_dz0m)
            dfxdif_part2 = dfxdif_part2_dzsl + dfxdif_part2_dLend + dfxdif_part2_dz0h + dfxdif_part2_dz0m
            dfxdif = (dfxdif_part1 + dfxdif_part2) / (Lstart[i] - Lend[i]) + (fxdif_part1[i] + fxdif_part2[i]) * -1 * (Lstart[i] - Lend[i])**-2 * (dLstart - dLend)
            dL_new       = dL - (dfx / fxdif[i] + fx[i] * -1 * fxdif[i]**-2 * dfxdif)
            dL = dL_new #trick for easier adjoint coding
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rtl.update({variablename: the_locals[variablename]})              
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rtl:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rtl[key]
        for key in self.Output_tl_rtl:
            if key == returnvariable:
                returnvar = self.Output_tl_rtl[returnvariable]
                return returnvar
    
    def tl_run_land_surface(self,model,checkpoint,returnvariable=None):
        self.Output_tl_rls = {}
        u = checkpoint['rls_u']
        v = checkpoint['rls_v']
        wstar = checkpoint['rls_wstar']
        Cs = checkpoint['rls_Cs']
        ueff = checkpoint['rls_ueff_end']
        theta = checkpoint['rls_theta']
        esatvar = checkpoint['rls_esatvar_end']
        wg = checkpoint['rls_wg']
        wfc = checkpoint['rls_wfc']
        wwilt = checkpoint['rls_wwilt']
        Wmax = checkpoint['rls_Wmax']
        LAI = checkpoint['rls_LAI']
        Wlmx = checkpoint['rls_Wlmx_end']
        Wl = checkpoint['rls_Wl']
        ra = checkpoint['rls_ra_end']
        cveg = checkpoint['rls_cveg']
        cliq = checkpoint['rls_cliq_end']
        dqsatdT = checkpoint['rls_dqsatdT_end']
        rssoil = checkpoint['rls_rssoil_end']
        Lambda = checkpoint['rls_Lambda']
        qsatvar = checkpoint['rls_qsatvar_end']
        q = checkpoint['rls_q']
        Tsoil = checkpoint['rls_Tsoil']
        p2_numerator_Ts = checkpoint['rls_p2_numerator_Ts_end']
        p3_numerator_Ts = checkpoint['rls_p3_numerator_Ts_end']
        p4_numerator_Ts = checkpoint['rls_p4_numerator_Ts_end']
        p5_numerator_Ts = checkpoint['rls_p5_numerator_Ts_end']
        p3_denominator_Ts = checkpoint['rls_p3_denominator_Ts_end']
        p4_denominator_Ts = checkpoint['rls_p4_denominator_Ts_end']
        numerator_Ts = checkpoint['rls_numerator_Ts_end']
        denominator_Ts = checkpoint['rls_denominator_Ts_end']
        p1_LEveg = checkpoint['rls_p1_LEveg_end']
        p2_LEveg_liq_soil = checkpoint['rls_p2_LEveg_liq_soil_end']
        Ts = checkpoint['rls_Ts_end']
        p1_LEliq = checkpoint['rls_p1_LEliq_end']
        p1_LEsoil = checkpoint['rls_p1_LEsoil_end']
        G = checkpoint['rls_G_end']
        Q = checkpoint['rls_Q']
        numerator_LEpot = checkpoint['rls_numerator_LEpot_end']
        rsmin = checkpoint['rls_rsmin']
        denominator_LEref = checkpoint['rls_denominator_LEref_end']
        numerator_LEref = checkpoint['rls_numerator_LEref_end']
        w2 = checkpoint['rls_w2']
        wsat = checkpoint['rls_wsat']
        CG = checkpoint['rls_CG_end']
        C1 = checkpoint['rls_C1_end']
        C2 = checkpoint['rls_C2_end']
        d1 = checkpoint['rls_d1_end']
        LEsoil = checkpoint['rls_LEsoil_end']
        wgeq = checkpoint['rls_wgeq_end']
        ustar = checkpoint['rls_ustar']
        rs = checkpoint['rls_rs_end'] #rs calculated in e.g. ags
        rssoilmin = checkpoint['rls_rssoilmin']
        f2 = checkpoint['rls_f2_end']
        CGsat = checkpoint['rls_CGsat']
        b = checkpoint['rls_b']
        C1sat = checkpoint['rls_C1sat']
        C2ref = checkpoint['rls_C2ref']
        a = checkpoint['rls_a']
        p = checkpoint['rls_p']
        if self.manualadjointtesting:
            self.dwstar,self.dCs,self.dtheta,self.dq,self.dwfc,self.dwwilt,self.dwg,self.dLAI,self.dWmax,self.dWl,self.dcveg,self.dQ,self.dLambda,self.dTsoil,self.drsmin,self.dwsat,self.dw2,self.dT2 =self.x
        dueff = 0.5*(u ** 2. + v ** 2. + wstar**2.)**(-1/2) * (2 * u * self.du + 2 * v * self.dv + 2 * wstar * self.dwstar)
        if(model.sw_sl):
            dra = -1 * (Cs * ueff)**-2. *(self.dCs * ueff + Cs * dueff)
        else:
            if ustar >= 1.e-3:
                dra = dueff / ustar**2. + ueff * -2 * ustar**(-3) * self.dustar 
            else:
                dra = dueff / (1.e-3)**2.
        desatvar    = desat(theta,self.dtheta)
        dqsatvar    = dqsat_dT(theta,model.Ps,self.dtheta)
        ddesatdT_dtheta = desatvar * (17.2694 / (theta - 35.86) - 17.2694 * (theta - 273.16) / (theta - 35.86)**2.) + esatvar * ((17.2694 * -1 * (theta - 35.86)**(-2)) * self.dtheta + -1 * 17.2694 * (self.dtheta/(theta - 35.86)**2. + (theta - 273.16) * (-2) * (theta - 35.86)**(-3) * self.dtheta))
        ddqsatdT_dtheta = 0.622 / model.Ps * ddesatdT_dtheta
        de = model.Ps / 0.622 * self.dq
        #now we need to call another module, which needs one of the variables above
        #therefore they need to become self variables
        the_locals = cp.deepcopy(locals())
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rls.update({variablename: the_locals[variablename]})
        for key in self.Output_tl_rls:
            self.__dict__[key] = self.Output_tl_rls[key]
        if(self.model.ls_type == 'js'): 
            drs = self.tl_jarvis_stewart(model,checkpoint,returnvariable='drs') 
        elif(self.model.ls_type == 'ags'):
            drs = self.tl_ags(model,checkpoint,returnvariable='drs')
            #this calculates drs which is used later!!
        elif(self.model.ls_type == 'canopy_model'):
            raise Exception('Canopy model not yet implemented') #to implement
        elif(self.ls_type == 'sib4'):
            raise Exception('sib4 not yet implemented') #to implement
        else:
            raise Exception('problem with ls switch')
        if(wg > wwilt):
          df2          = (self.dwfc - self.dwwilt) / (wg - wwilt) + (wfc - wwilt) * -1 * (wg - wwilt)**(-2) * (self.dwg - self.dwwilt)
        else:
          df2        = 0
        drssoil = rssoilmin * df2 + f2 * self.drssoilmin
        if(self.model.ls_type != 'canopy_model'):
            dWlmx = self.dLAI * Wmax + LAI * self.dWmax
            if Wl / Wlmx <= 1: #cliq = min(1., self.Wl / Wlmx)
                dcliq = self.dWl / Wlmx + Wl * (-1) * Wlmx ** (-2) * dWlmx
            else:
                dcliq = 0
            dp1_numerator_Ts = model.rho * model.cp *(self.dtheta / ra + theta * (-1) * ra**(-2) * dra) 
            dp2_numerator_Ts = self.dcveg * (1. - cliq) * model.rho * model.Lv / (ra + rs) + cveg * - dcliq * model.rho * model.Lv / (ra + rs) + cveg * (1. - cliq) * model.rho * model.Lv * (-1) * (ra + rs)**(-2) * (dra + drs)
            dp3_numerator_Ts = ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq
            dp4_numerator_Ts = model.rho * model.Lv * ((dqsatdT * theta - qsatvar + q)*(-1)*(ra + rssoil)**(-2)*(dra + drssoil) + 1 / (ra + rssoil) * (ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq))
            dp5_numerator_Ts = dcliq * model.rho * model.Lv /  ra * (dqsatdT * theta - qsatvar + q) + cliq * model.rho * model.Lv * ((dqsatdT * theta - qsatvar + q)*(-1)*ra**(-2)*dra + 1 / ra * (ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq))
            dnumerator_Ts = self.dQ + dp1_numerator_Ts + dp2_numerator_Ts * p3_numerator_Ts + p2_numerator_Ts * dp3_numerator_Ts + (1. - cveg) * dp4_numerator_Ts + p4_numerator_Ts * - self.dcveg + \
                            cveg * dp5_numerator_Ts + self.dcveg * p5_numerator_Ts + self.dLambda * Tsoil + Lambda * self.dTsoil
            dp1_denominator_Ts = model.rho * model.cp * (-1) * ra**(-2) * dra
            dp2_denominator_Ts = self.dcveg * (1. - cliq) * model.rho * model.Lv / (ra + rs) * dqsatdT + cveg * model.rho * model.Lv / (ra + rs) * dqsatdT * -dcliq + cveg * (1. - cliq) * model.rho * model.Lv * (dqsatdT * (-1) * (ra + rs)**(-2) * (dra + drs) + 1 / (ra + rs) * ddqsatdT_dtheta)
            dp3_denominator_Ts = model.rho * model.Lv * (dqsatdT * (-1) * (ra + rssoil)**(-2) * (dra + drssoil) + 1 / (ra + rssoil) * ddqsatdT_dtheta)
            dp4_denominator_Ts = model.rho * model.Lv * (dcliq * 1 / ra * dqsatdT + cliq * dqsatdT * (-1) * ra**(-2) * dra + cliq / ra * ddqsatdT_dtheta)
            ddenominator_Ts = dp1_denominator_Ts + dp2_denominator_Ts + (1. - cveg) * dp3_denominator_Ts + p3_denominator_Ts * -self.dcveg + cveg * dp4_denominator_Ts + p4_denominator_Ts * self.dcveg + self.dLambda
            dTs = dnumerator_Ts / denominator_Ts + numerator_Ts * (-1) * denominator_Ts**(-2) * ddenominator_Ts
            desatsurf      = desat(Ts,dTs)
            dqsatsurf = dqsat_dT(Ts,model.Ps,dTs)
            dp1_LEveg = model.rho * model.Lv * (- dcliq * cveg / (ra + rs) + (1 - cliq) / (ra + rs) * self.dcveg + (1 - cliq) * cveg * (-1) * (ra + rs)**(-2) * ((dra + drs)))
            dp2_LEveg_liq_soil = ddqsatdT_dtheta * (Ts - theta) + dqsatdT * (dTs - self.dtheta) + dqsatvar - self.dq
            if self.manualadjointtesting:
                self.Hx = dp2_LEveg_liq_soil
            dLEveg = dp1_LEveg * p2_LEveg_liq_soil + p1_LEveg * dp2_LEveg_liq_soil
            dp1_LEliq = model.rho * model.Lv * (dcliq * cveg / ra + cliq / ra * self.dcveg + cliq * cveg * (-1) * ra**(-2) * dra)    
            dLEliq = dp1_LEliq * p2_LEveg_liq_soil + p1_LEliq * dp2_LEveg_liq_soil
            dp1_LEsoil = model.rho * model.Lv * (-self.dcveg / (ra + rssoil) + (1. - cveg) * (-1) * (ra + rssoil)**(-2) * (dra + drssoil))
            dLEsoil = dp1_LEsoil * p2_LEveg_liq_soil + p1_LEsoil * dp2_LEveg_liq_soil
            dWltend      = - dLEliq / (model.rhow * model.Lv)
            dLE   = dLEsoil + dLEveg + dLEliq
            dH    = model.rho * model.cp * ((Ts - theta) * (-1) * (ra)**(-2) * dra + 1 / ra * (dTs - self.dtheta))
            dG      = self.dLambda * (Ts - Tsoil) + Lambda * (dTs - self.dTsoil)
            dp1_numerator_LEpot = ddqsatdT_dtheta * (Q - G) + dqsatdT * (self.dQ - dG)
            dp2_numerator_LEpot = model.rho * model.cp * ((qsatvar - q) * (-1) * ra**(-2) * dra + 1 / ra * (dqsatvar - self.dq))
            dnumerator_LEpot = dp1_numerator_LEpot + dp2_numerator_LEpot
            dLEpot = dnumerator_LEpot / (dqsatdT + model.cp / model.Lv) + numerator_LEpot * (-1) * (dqsatdT + model.cp / model.Lv)**(-2) * ddqsatdT_dtheta
            dp1_numerator_LEref = ddqsatdT_dtheta * (Q - G) + dqsatdT * (self.dQ - dG)
            dp2_numerator_LEref = model.rho * model.cp * ((qsatvar - q) * (-1) * ra**(-2) * dra + 1 / ra * (dqsatvar - self.dq))
            dnumerator_LEref = dp1_numerator_LEref + dp2_numerator_LEref
            ddenominator_LEref = ddqsatdT_dtheta + model.cp / model.Lv * (self.drsmin / LAI / ra + rsmin / ra * (-1) * LAI**(-2) * self.dLAI + rsmin / LAI * (-1) * ra**(-2) * dra)
            dLEref  = dnumerator_LEref / denominator_LEref + numerator_LEref * (-1) * denominator_LEref**(-2) * ddenominator_LEref
        else:
            pass #to implement
        dCG_dCGsat = (wsat / w2)**(b / (2. * np.log(10.))) * self.dCGsat
        dCG_dwsat = CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) * self.dwsat / w2
        dCG_dw2 = CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) *  wsat * (-1) * w2**(-2) * self.dw2
        dCG_db = CGsat * (wsat / w2)**(b / (2. * np.log(10.))) * np.log(wsat / w2) * 1 / (2. * np.log(10.)) * self.db #d(a^x)/dx = a^x * ln(a)
        dCG = dCG_dCGsat + dCG_dwsat + dCG_dw2 + dCG_db
        dTsoiltend_dCG = G * dCG
        dTsoiltend_dG   = CG * dG
        dTsoiltend_dTsoil = - 2. * np.pi / 86400. * self.dTsoil
        dTsoiltend_dT2 = 2. * np.pi / 86400. * self.dT2
        dTsoiltend = dTsoiltend_dCG + dTsoiltend_dG + dTsoiltend_dTsoil + dTsoiltend_dT2
        dC1_dC1sat = (wsat / wg) ** (b / 2. + 1.) * self.dC1sat
        dC1_dwsat  = C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) * self.dwsat / wg
        dC1_dwg    = C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) * wsat * (-1) * wg**(-2) * self.dwg
        dC1_db     = C1sat * (wsat / wg) ** (b / 2. + 1.) * np.log(wsat / wg) * 1 / 2. * self.db
        dC1 = dC1_dC1sat + dC1_dwsat + dC1_dwg + dC1_db             
        dC2_dC2ref = w2 / (wsat - w2) * self.dC2ref
        dC2_dw2    = C2ref * (self.dw2 / (wsat - w2) + w2 * (-1) * (wsat - w2)**(-2) * -self.dw2)
        dC2_dwsat  = C2ref * (w2 * (-1) * (wsat - w2)**(-2) * self.dwsat)
        dC2        = dC2_dC2ref + dC2_dw2 + dC2_dwsat        
        #wgeq        = self.w2 - self.wsat * self.a * ( (self.w2 / self.wsat) ** self.p * (1. - (self.w2 / self.wsat) ** (8. * self.p)) )
        dwgeq_dw2  = self.dw2 - wsat * a * (p * (w2 / wsat) ** (p - 1) * self.dw2 / wsat * (1. - (w2 / wsat) ** (8. * p)) + (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * self.dw2 / wsat)
        dwgeq_dwsat = -self.dwsat * a * ((w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p))) - wsat * a * (p * (w2 / wsat) ** (p - 1) * w2 * (-1) * wsat**(-2) * self.dwsat * (1. - (w2 / wsat) ** (8. * p)) + (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * w2 * (-1) * wsat**(-2) * self.dwsat)
        dwgeq_da = - wsat * (w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p)) * self.da
        dwgeq_dp = - wsat * a * ( (1. - (w2 / wsat) ** (8. * p)) * (w2 / wsat) ** p * np.log(w2 / wsat) + (w2 / wsat) ** p * -1 * (w2 / wsat) ** (8. * p) * np.log(w2 / wsat) * 8) * self.dp
        dwgeq      = dwgeq_dw2 + dwgeq_dwsat + dwgeq_da + dwgeq_dp
        dwgtend_dLEsoil = - C1 / (model.rhow * d1) / model.Lv * dLEsoil 
        dwgtend_dC1 = - dC1 / (model.rhow * d1) * LEsoil / model.Lv
        dwgtend_dC2 = - dC2 / 86400. * (wg - wgeq)
        dwgtend_dwg = - C2 / 86400. * self.dwg
        dwgtend_dwgeq = C2 / 86400. * dwgeq
        dwgtend = dwgtend_dLEsoil + dwgtend_dC1 + dwgtend_dC2 + dwgtend_dwg +dwgtend_dwgeq
        dwtheta   = dH  / (model.rho * model.cp)
        dwq       = dLE / (model.rho * model.Lv)
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rls.update({variablename: the_locals[variablename]})              
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rls:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rls[key]
        if returnvariable is not None:
            for key in self.Output_tl_rls:
                if key == returnvariable:
                    returnvar = self.Output_tl_rls[returnvariable]
                    return returnvar
            
    def tl_jarvis_stewart(self,model,checkpoint,returnvariable=None):
        self.Output_tl_js = {}
        Swin = checkpoint['js_Swin']
        w2 = checkpoint['js_w2']
        wwilt = checkpoint['js_wwilt']
        wfc = checkpoint['js_wfc']
        gD = checkpoint['js_gD']
        e = checkpoint['js_e']
        esatvar = checkpoint['js_esatvar']
        theta = checkpoint['js_theta']
        f2js = checkpoint['js_f2js_middle']
        LAI = checkpoint['js_LAI']
        f1 = checkpoint['js_f1_end']
        f3 = checkpoint['js_f3_end']
        f4 = checkpoint['js_f4_end']
        rsmin = checkpoint['js_rsmin']
        if(model.sw_rad):
            if (0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)) <= 1:
                df1 = -1 * ((0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)))**(-2) * (0.004 * self.dSwin / (0.81 * (0.004 * Swin + 1.)) + (0.004 * Swin + 0.05) * -1 * (0.81 * (0.004 * Swin + 1.))**(-2) * 0.81 * 0.004 * self.dSwin)
            else:
                df1 = 0
        else:
            df1 = 0
        if(w2 > wwilt):
            #f2js = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)
            df2js_dwfc = 1 / (w2 - wwilt) * self.dwfc
            df2js_dwwilt = 1 / (w2 - wwilt) * -1 * self.dwwilt + (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * -1 * self.dwwilt
            df2js_dw2 = (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * self.dw2
            df2js = df2js_dwfc + df2js_dwwilt + df2js_dw2
        else:
            df2js = 0
        if f2js < 1:
            df2js = 0
        #f3 = 1. / np.exp(- self.gD * (self.esat - self.e) / 100.)
        df3 = -1 * (np.exp(- gD * (esatvar - e) / 100.))**-2 * np.exp(- gD * (esatvar - e) / 100.) * (-self.dgD  * (esatvar - e) / 100 - 1 / 100 * gD * self.desatvar + 1 / 100 * gD * self.de)
        df4 = -1 * (1. - 0.0016 * (298.0-theta)**2.)**(-2) * - 0.0016 * 2 * (298.0 - theta) * -1 * self.dtheta
        f2js = checkpoint['js_f2js_end']
        #self.rs = self.rsmin / self.LAI * f1 * f2js * f3 * f4
        drs_drsmin = self.drsmin / LAI * f1 * f2js * f3 * f4 
        drs_dLAI = rsmin * f1 * f2js * f3 * f4 * -1 * LAI**(-2) * self.dLAI
        drs_df1 = rsmin / LAI * f2js * f3 * f4 * df1
        drs_df2js = rsmin / LAI * f1 * f3 * f4 * df2js
        drs_df3 = rsmin / LAI * f1 * f2js * f4 * df3
        drs_df4 = rsmin / LAI * f1 * f2js * f3 * df4
        drs = drs_drsmin + drs_dLAI + drs_df1 + drs_df2js + drs_df3 + drs_df4
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_js.update({variablename: the_locals[variablename]})  
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_js:
                if key in self.__dict__: #otherwise you get a lot of unnecessary vars in memory
                    self.__dict__[key] = self.Output_tl_js[key]
        for key in self.Output_tl_js:
            if key == returnvariable:
                returnvar = self.Output_tl_js[returnvariable]
                return returnvar
    
    def tl_ags(self,model,checkpoint,returnvariable=None):
        COS  = checkpoint['ags_COS']
        cveg = checkpoint['ags_cveg']
        LAI  = checkpoint['ags_LAI']
        alfa_sto  = checkpoint['ags_alfa_sto']
        thetasurf, Ts, CO2, wg, Swin, ra, Tsoil = checkpoint['ags_thetasurf'],checkpoint['ags_Ts'],checkpoint['ags_CO2'],checkpoint['ags_wg'],checkpoint['ags_Swin'],checkpoint['ags_ra'],checkpoint['ags_Tsoil']
        texp, fw, Ds, D0, Dstar, co2abs, CO2comp, rsCO2, ci, PAR, alphac, cfrac = checkpoint['ags_texp_end'],checkpoint['ags_fw_end'],checkpoint['ags_Ds_end'],checkpoint['ags_D0_end'],checkpoint['ags_Dstar_end'],checkpoint['ags_co2abs_end'],checkpoint['ags_CO2comp_end'],checkpoint['ags_rsCO2_end'],checkpoint['ags_ci_end'],checkpoint['ags_PAR_end'],checkpoint['ags_alphac_end'],checkpoint['ags_cfrac_end']
        gm, gm1,gm2,gm3,sqrtf,sqterm,fmin0 = checkpoint['ags_gm_end'],checkpoint['ags_gm1_end'],checkpoint['ags_gm2_end'],checkpoint['ags_gm3_end'],checkpoint['ags_sqrtf_end'],checkpoint['ags_sqterm_end'],checkpoint['ags_fmin0_end']
        Ammax1,Ammax2,Ammax3,fmin = checkpoint['ags_Ammax1_end'],checkpoint['ags_Ammax2_end'],checkpoint['ags_Ammax3_end'],checkpoint['ags_fmin_end']
        pexp, xdiv, aexp, Ammax = checkpoint['ags_pexp_end'],checkpoint['ags_xdiv_end'],checkpoint['ags_aexp_end'],checkpoint['ags_Ammax_end']
        y, y1, AmRdark, sy = checkpoint['ags_y_end'],checkpoint['ags_y1_end'],checkpoint['ags_AmRdark_end'],checkpoint['ags_sy_end']
        fstr, a1, div1, div2, An_temporary, part1,gcco2, a11 = checkpoint['ags_fstr_end'],checkpoint['ags_a1_end'],checkpoint['ags_div1_end'],checkpoint['ags_div2_end'],checkpoint['ags_An_temporary_end'],checkpoint['ags_part1_end'],checkpoint['ags_gcco2_end'],checkpoint['ags_a11_end']
        gciCOS = checkpoint['ags_gciCOS_end']
        gctCOS = checkpoint['ags_gctCOS_end']
        PARfract = checkpoint['ags_PARfract']
        wwilt = checkpoint['ags_wwilt']
        wfc = checkpoint['ags_wfc']
        w2 = checkpoint['ags_w2']
        R10,E0 = checkpoint['ags_R10'],checkpoint['ags_E0']
        self.Output_tl_ags = {}
        # output:  wCO2A   wCO2R  wCO2 rs 
        # Select index for plap.nt type
        if(model.c3c4 == 'c3'):
            c = 0
        elif(model.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('option \"%s\" for \"c3c4\" invalid'%model.c3c4)
        # cp: thetasurf!

        # derivative: d/dx (a^0.1*(x-c) = 0.1 * a^0.1*(x-c) ln(a)
        #CO2comp       = model.CO2comp298[c] * model.rho * pow(model.Q10CO2[c],(0.1 * (thetasurf - 298.)))  
        dCO2comp_dthetasurf = model.CO2comp298[c] * model.rho * np.log(model.Q10CO2[c])  \
                   * pow(model.Q10CO2[c],(0.1 * (thetasurf - 298.))) * 0.1* self.dthetasurf
        # calculate mesophyll conductance: to ease derivative, split in three: derivative to thetasurf only!
        dgm1_dthetasurf = 0.1* self.dthetasurf*np.log(model.Q10gm[c])* \
                model.gm298[c] *  pow(model.Q10gm[c],(0.1 * (thetasurf-298.)))  
        dgm2_dthetasurf =  -0.3*self.dthetasurf*np.exp(0.3 * (model.T1gm[c] - thetasurf))
        dgm3_dthetasurf =   0.3*self.dthetasurf*np.exp(0.3 * (thetasurf - model.T2gm[c]))
        dgm_dthetasurf  = dgm1_dthetasurf/(gm2*gm3) - gm1*dgm2_dthetasurf/(gm3*gm2**2) - gm1*dgm3_dthetasurf/(gm2*gm3**2)
        dgm_dthetasurf  = dgm_dthetasurf / 1000. # conversion from mm s-1 to m s-1
        dfmin0_dthetasurf        = -1./9.*dgm_dthetasurf
        dsqrtf_dthetasurf        = 2.*fmin0*dfmin0_dthetasurf +  4*model.gmin[c]/model.nuco2q *dgm_dthetasurf
        dsqterm_dthetasurf       = dsqrtf_dthetasurf*0.5*pow(sqrtf,-0.5)
        dfmin_dthetasurf         = -dfmin0_dthetasurf/(2.*gm) + dsqterm_dthetasurf/(2.*gm)-(-fmin0 + sqterm)*dgm_dthetasurf/(2*gm**2)
#        if self.manualadjointtesting:
#            dfmin_dthetasurf = self.x
        dD0_dthetasurf           = -dfmin_dthetasurf/model.ad[c]
        #dcfrac_dthetasurf        = model.f0[c]*dD0_dthetasurf*Ds/(D0**2) + dfmin_dthetasurf*(Ds/D0) - fmin*dD0_dthetasurf*Ds/(D0**2)  #zero??
        #the above statement is actually zero, see Notes 3. Gives errors if you include the above statement, due to rounding making it non-zero
#        if self.manualadjointtesting:
#            self.Hx = dD0_dthetasurf
        dDs_dTs       = (desat(Ts,self.dTs)-self.devap)/1000.       # note dTs and devap propagate similar....called dTs
        dDs_de        = -self.de / 1000
        #Ds            = (esat(Ts) - evap) / 1000. # kPa
        #D0            = (model.f0[c] - fmin) / model.ad[c]
        #cfrac         = model.f0[c] * (1. - (Ds / D0)) + fmin * (Ds / D0)
        dcfrac_dTs    = -model.f0[c]*(dDs_dTs/D0) + fmin*(dDs_dTs/D0)
        dcfrac_de    = -model.f0[c]*(dDs_de/D0) + fmin*(dDs_de/D0)        
        if model.ags_C_mode == 'MXL': 
            dco2abs_dCO2  = self.dCO2 * (model.mco2 / model.mair) * model.rho
            dco2abs = dco2abs_dCO2
        elif model.ags_C_mode == 'surf': 
            dco2abs_dCO2surf  = self.dCO2surf * (model.mco2 / model.mair) * model.rho
            dco2abs = dco2abs_dCO2surf
        else:
            raise Exception('wrong ags_C_mode switch')
        dci_dthetasurf           = dCO2comp_dthetasurf *(1.- cfrac)
        # ci            = cfrac * (co2abs - CO2comp) + CO2comp
        dci_dTs       = dcfrac_dTs * (co2abs - CO2comp)
        dci_de       = dcfrac_de * (co2abs - CO2comp)
        dci_dCO2      = cfrac * dco2abs #note that I make the naming inconsistent with this, as dci_dCO2 can be dci_dCO2surf, depending on a switch

        dAmmax1_dthetasurf        = 0.1*self.dthetasurf*np.log(model.Q10Am[c])* \
                            model.Ammax298[c] *  pow(model.Q10Am[c],(0.1 * (thetasurf - 298.))) 
        dAmmax2_dthetasurf        = -0.3*self.dthetasurf*np.exp(0.3 * (model.T1Am[c] - thetasurf))
        dAmmax3_dthetasurf        = +0.3*self.dthetasurf*np.exp(0.3 * (thetasurf - model.T2Am[c]))
        #Ammax         = Ammax1/(Ammax2*Ammax3) 
        dAmmax_dthetasurf  = dAmmax1_dthetasurf/(Ammax2*Ammax3) - Ammax1*dAmmax2_dthetasurf/(Ammax3*Ammax2**2) - Ammax1*dAmmax3_dthetasurf/(Ammax2*Ammax3**2)
        if (w2 - wwilt)/(wfc - wwilt) > 1 or (w2 - wwilt)/(wfc - wwilt) < 1e-3:
            dbetaw_dw2 = 0
            dbetaw_dwfc = 0
            dbetaw_dwwilt = 0
        else: 
            dbetaw_dw2 = self.dw2/(wfc - wwilt)
            dbetaw_dwfc = (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * self.dwfc
            dbetaw_dwwilt = -self.dwwilt / (wfc - wwilt) + (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * -self.dwwilt
        if (model.c_beta == 0):
            dfstr_dw2  = dbetaw_dw2
            dfstr_dwfc = dbetaw_dwfc
            dfstr_dwwilt = dbetaw_dwwilt
        else:
            P = checkpoint['ags_P_end']
            betaw = checkpoint['ags_betaw_end']
            #fstr = (1. - np.exp(-P * betaw)) / (1 - np.exp(-P))
            dfstr_dw2  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dw2
            dfstr_dwfc  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dwfc
            dfstr_dwwilt  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dwwilt
        #aexp          = -gm*ci/Ammax + gm*CO2comp/Ammax
        daexp_dthetasurf        = -dgm_dthetasurf*ci/Ammax - dci_dthetasurf*gm/Ammax + dAmmax_dthetasurf*gm*ci/(Ammax**2) + \
                         dgm_dthetasurf*CO2comp/Ammax + dCO2comp_dthetasurf*gm/Ammax - dAmmax_dthetasurf*gm*CO2comp/(Ammax**2)
        daexp_dTs     = -dci_dTs  * gm / Ammax
        daexp_de      = -dci_de  * gm / Ammax
        daexp_dCO2    = -dci_dCO2 * gm / Ammax

        Am            = Ammax * (1. - np.exp(aexp))
        dAm_dthetasurf           = dAmmax_dthetasurf * (1. - np.exp(aexp)) - Ammax*daexp_dthetasurf*np.exp(aexp)
        dAm_dTs       = -Ammax * daexp_dTs  * np.exp(aexp)
        dAm_de        = -Ammax * daexp_de  * np.exp(aexp)
        dAm_dCO2      = -Ammax * daexp_dCO2 * np.exp(aexp)
        
        Rdark        = (1. / 9.) * Am
        dRdark_dthetasurf       = (1. / 9.) * dAm_dthetasurf
        dRdark_dTs   = (1. / 9.) * dAm_dTs
        dRdark_de    = (1. / 9.) * dAm_de
        dRdark_dCO2  = (1. / 9.) * dAm_dCO2
        AmRdark       = Am + Rdark
        dAmRdark_dthetasurf      = dAm_dthetasurf + dRdark_dthetasurf
        dAmRdark_dTs  = dAm_dTs + dRdark_dTs
        dAmRdark_de   = dAm_de + dRdark_de
        dAmRdark_dCO2 = dAm_dCO2 + dRdark_dCO2  
        Swina        = Swin  * cveg
        dSwina_dSwin = self.dSwin * cveg
        dSwina_dcveg = Swin  * self.dcveg
        if Swin  * cveg < 1e-1:
            dPAR_dPARfract = 1e-1 * self.dPARfract
            dPAR_dSwin = 0
            dPAR_dcveg = 0
        else:
            dPAR_dPARfract = Swin * cveg * self.dPARfract
            dPAR_dSwin     = PARfract * dSwina_dSwin
            dPAR_dcveg     = PARfract * dSwina_dcveg
        #xdiv = co2abs + 2.*CO2comp
        dxdiv_dthetasurf      = 2.*dCO2comp_dthetasurf  # dco2abs/dthetav = 0  
        dxdiv_dCO2 = dco2abs 
        alphac       = model.alpha0[c] * (co2abs - CO2comp) / xdiv   # dco2abs/dthetav = 0       
        dalphac_dthetasurf      = model.alpha0[c] * ( (CO2comp-co2abs) * dxdiv_dthetasurf / (xdiv**2) - dCO2comp_dthetasurf/xdiv)
        dalphac_dCO2 = model.alpha0[c] * ( (CO2comp-co2abs) * dxdiv_dCO2 / (xdiv**2) + dco2abs/xdiv)
        #pexp         = -1 * alphac * PAR / (AmRdark)
        dpexp_dthetasurf        = -1 * dalphac_dthetasurf * PAR / (AmRdark) + dAmRdark_dthetasurf * alphac * PAR/(AmRdark**2)
        dpexp_dTs    = dAmRdark_dTs*alphac*PAR/(AmRdark**2)
        dpexp_de     = dAmRdark_de*alphac*PAR/(AmRdark**2)
        dpexp_dCO2   = -1 * dalphac_dCO2 * PAR / (AmRdark) + dAmRdark_dCO2 * alphac * PAR/(AmRdark**2)
        dpexp_dPARfract  = -1 * alphac * dPAR_dPARfract / (AmRdark)
        dpexp_dSwin  = -1 * alphac * dPAR_dSwin / (AmRdark)
        dpexp_dcveg  = -1 * alphac * dPAR_dcveg / (AmRdark)
        Ag           = AmRdark * (1. - np.exp(pexp))
        dAg          = ( dAmRdark_dthetasurf + dAmRdark_dTs + dAmRdark_de + dAmRdark_dCO2  ) * (1. - np.exp(pexp)) \
                      -  AmRdark * ( dpexp_dthetasurf + dpexp_dTs + dpexp_de + dpexp_dCO2 + dpexp_dPARfract + dpexp_dSwin + dpexp_dcveg) * np.exp(pexp)
        # calculation dAG closed 
        dy_dthetasurf=  model.Kx[c] * PAR * (dalphac_dthetasurf / AmRdark - dAmRdark_dthetasurf * alphac / (AmRdark**2))
        dy_dTs       = -model.Kx[c] * PAR * dAmRdark_dTs * alphac / (AmRdark**2)
        dy_de        = -model.Kx[c] * PAR * dAmRdark_de * alphac / (AmRdark**2)
        dy_dCO2      =  model.Kx[c] * PAR * (dalphac_dCO2 / AmRdark - dAmRdark_dCO2 * alphac / (AmRdark**2))
        dy_dPARfract     =  alphac * model.Kx[c] * dPAR_dPARfract / (AmRdark)
        dy_dSwin     =  alphac * model.Kx[c] * dPAR_dSwin / (AmRdark)
        dy_dcveg     =  alphac * model.Kx[c] * dPAR_dcveg / (AmRdark)

        dy1_dthetasurf          =  dy_dthetasurf * np.exp(-model.Kx[c] * LAI)
        dy1_dTs      =  dy_dTs   * np.exp(-model.Kx[c] * LAI)
        dy1_de       =  dy_de   * np.exp(-model.Kx[c] * LAI)
        dy1_dCO2     =  dy_dCO2  * np.exp(-model.Kx[c] * LAI)
        dy1_dPARfract =  dy_dPARfract * np.exp(-model.Kx[c] * LAI)
        dy1_dSwin    =  dy_dSwin * np.exp(-model.Kx[c] * LAI)
        dy1_dcveg    =  dy_dcveg * np.exp(-model.Kx[c] * LAI)
        dy1_dLAI     =  y * np.exp(-model.Kx[c] * LAI) * -model.Kx[c] * self.dLAI
        
        dsy_dthetasurf          =  (dy1_dthetasurf       * self.dE1(y1) - dy_dthetasurf       * self.dE1(y))/(model.Kx[c] * LAI)
        dsy_dTs      =  (dy1_dTs   * self.dE1(y1) - dy_dTs   * self.dE1(y))/(model.Kx[c] * LAI)
        dsy_de      =  (dy1_de   * self.dE1(y1)  - dy_de   * self.dE1(y))/(model.Kx[c] * LAI)
        dsy_dCO2     =  (dy1_dCO2  * self.dE1(y1) - dy_dCO2  * self.dE1(y))/(model.Kx[c] * LAI)
        if self.manualadjointtesting:
            dy1_dPARfract,dy_dPARfract = self.x
        dsy_dPARfract    =  (dy1_dPARfract * self.dE1(y1) - dy_dPARfract * self.dE1(y))/(model.Kx[c] * LAI)
        if self.manualadjointtesting:
            self.Hx = dsy_dPARfract
        dsy_dSwin    =  (dy1_dSwin * self.dE1(y1) - dy_dSwin * self.dE1(y))/(model.Kx[c] * LAI)
        dsy_dcveg    =  (dy1_dcveg * self.dE1(y1) - dy_dcveg * self.dE1(y))/(model.Kx[c] * LAI)
        dsy_dLAI    =  dy1_dLAI * self.dE1(y1) / (model.Kx[c] * LAI) + (model.E1(y1) - model.E1(y)) * -1 * (model.Kx[c] * LAI)**(-2) * model.Kx[c] * self.dLAI

        dAn_temporary_dthetasurf          = dAmRdark_dthetasurf * (1. - sy) - dsy_dthetasurf * AmRdark
        dAn_temporary_dTs      = dAmRdark_dTs * (1. - sy) - dsy_dTs * AmRdark
        dAn_temporary_de       = dAmRdark_de * (1. - sy) - dsy_de * AmRdark
        dAn_temporary_dCO2     = dAmRdark_dCO2 * (1. - sy) - dsy_dCO2 * AmRdark
        dAn_temporary_dPARfract    = - dsy_dPARfract * AmRdark
        dAn_temporary_dSwin    = - dsy_dSwin * AmRdark
        dAn_temporary_dcveg    = - dsy_dcveg * AmRdark
        dAn_temporary_dLAI    = - dsy_dLAI * AmRdark

        
        da11_dthetasurf         =  -dfmin_dthetasurf*a1
        #dDstar_dthetasurf       = dD0_dthetasurf/a11 - da11_dthetasurf*D0/(a11**2)  # is exactly zero, see notes 3
        #ddiv1_dthetasurf        =  -dDstar_dthetasurf*Ds/(Dstar**2) = 0
        ddiv1_dTs    =  dDs_dTs/Dstar
        ddiv1_de    =  dDs_de/Dstar
        ddiv2_dthetasurf        =  -dCO2comp_dthetasurf*div1 
        ddiv2_dTs    = (co2abs - CO2comp)*ddiv1_dTs
        ddiv2_de    = (co2abs - CO2comp)*ddiv1_de
        ddiv2_dCO2   = dco2abs * div1

        dpart1_dthetasurf       = a1 * fstr * dAn_temporary_dthetasurf / div2 - a1 * fstr * An_temporary * ddiv2_dthetasurf / (div2**2)
        dpart1_dTs   = a1 * fstr * dAn_temporary_dTs / div2 \
                     - a1 * fstr * An_temporary * ddiv2_dTs / (div2**2)
        dpart1_de    = a1 * fstr * dAn_temporary_de / div2 \
                     - a1 * fstr * An_temporary * ddiv2_de / (div2**2)
        dpart1_dCO2  = a1 * fstr * dAn_temporary_dCO2 / div2 \
                     - a1 * fstr * An_temporary * ddiv2_dCO2 / (div2**2)
        dpart1_dPARfract = a1 * fstr * dAn_temporary_dPARfract / div2
        dpart1_dSwin = a1 * fstr * dAn_temporary_dSwin / div2
        dpart1_dcveg = a1 * fstr * dAn_temporary_dcveg / div2
        #part1        = a1 * fstr * An / div2
        dpart1_dw2   = a1 * dfstr_dw2 * An_temporary / div2
        dpart1_dwfc  = a1 * dfstr_dwfc * An_temporary / div2
        dpart1_dwwilt  = a1 * dfstr_dwwilt * An_temporary / div2
        dpart1_dLAI  = a1 * fstr * dAn_temporary_dLAI / div2
        
        dgcco2_dthetasurf       = alfa_sto * LAI * dpart1_dthetasurf
        dgcco2_dTs   = alfa_sto * LAI * dpart1_dTs
        dgcco2_de    = alfa_sto * LAI * dpart1_de
        dgcco2_dCO2  = alfa_sto * LAI * dpart1_dCO2
        dgcco2_dPARfract = alfa_sto * LAI * dpart1_dPARfract
        dgcco2_dSwin = alfa_sto * LAI * dpart1_dSwin
        dgcco2_dcveg = alfa_sto * LAI * dpart1_dcveg
        dgcco2_dw2   = alfa_sto * LAI * dpart1_dw2
        dgcco2_dwfc   = alfa_sto * LAI * dpart1_dwfc
        dgcco2_dwwilt   = alfa_sto * LAI * dpart1_dwwilt
        dgcco2_dalfa_sto = LAI * (model.gmin[c] / model.nuco2q + part1) * self.dalfa_sto
        dgcco2_dLAI = alfa_sto * (model.gmin[c] / model.nuco2q + part1) * self.dLAI + alfa_sto * LAI * dpart1_dLAI
        dgcco2 = dgcco2_dthetasurf + dgcco2_dTs + dgcco2_de + dgcco2_dCO2 + dgcco2_dPARfract + dgcco2_dSwin + dgcco2_dcveg + dgcco2_dw2 + dgcco2_dwfc + dgcco2_dwwilt + dgcco2_dalfa_sto + dgcco2_dLAI
        
        dgctCOS_dthetasurf = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dthetasurf
        dgctCOS_dTs = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dTs
        dgctCOS_de  = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_de
        dgctCOS_dCO2 = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dCO2
        dgctCOS_dPARfract = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dPARfract
        dgctCOS_dSwin = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dSwin
        dgctCOS_dcveg = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dcveg
        dgctCOS_dw2 = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dw2
        dgctCOS_dwfc = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dwfc
        dgctCOS_dwwilt = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dwwilt
        dgctCOS_dalfa_sto = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dalfa_sto
        dgctCOS_dLAI = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dLAI
        dgctCOS_dgciCOS = -1 * (1 / gciCOS + 1.21/gcco2)**-2 * -1 * gciCOS**-2 * self.dgciCOS
        dgctCOS = dgctCOS_dthetasurf + dgctCOS_dTs + dgctCOS_de + dgctCOS_dCO2 + dgctCOS_dPARfract + dgctCOS_dSwin + dgctCOS_dcveg + dgctCOS_dw2 + dgctCOS_dwfc + dgctCOS_dwwilt + dgctCOS_dalfa_sto + dgctCOS_dLAI + dgctCOS_dgciCOS
        # calculate surface resistance for moisture and carbon dioxide
        
        drs     = - ( dgcco2_dthetasurf + dgcco2_dTs + dgcco2_de + dgcco2_dCO2 + dgcco2_dPARfract + dgcco2_dSwin + dgcco2_dcveg + dgcco2_dw2 + dgcco2_dwfc + dgcco2_dwwilt + dgcco2_dalfa_sto + dgcco2_dLAI) / (1.6 * gcco2**2)
        rsCO2   = 1. / gcco2
        drsCO2_dthetasurf       = - dgcco2_dthetasurf       / (gcco2**2)
        drsCO2_dTs   = - dgcco2_dTs   / (gcco2**2)
        drsCO2_de    = - dgcco2_de   / (gcco2**2)
        drsCO2_dCO2  = - dgcco2_dCO2  / (gcco2**2)
        drsCO2_dPARfract = - dgcco2_dPARfract / (gcco2**2)
        drsCO2_dSwin = - dgcco2_dSwin / (gcco2**2)
        drsCO2_dcveg = - dgcco2_dcveg / (gcco2**2)
        drsCO2_dw2   = - dgcco2_dw2   / (gcco2**2)
        drsCO2_dwfc   = - dgcco2_dwfc   / (gcco2**2)
        drsCO2_dwwilt   = - dgcco2_dwwilt   / (gcco2**2)
        drsCO2_dalfa_sto   = - dgcco2_dalfa_sto   / (gcco2**2)
        drsCO2_dLAI   = - dgcco2_dLAI   / (gcco2**2)
        
        An           = -(co2abs - ci) / (ra + rsCO2)
        dAn_dthetasurf          = dci_dthetasurf/ (ra + rsCO2) + drsCO2_dthetasurf*(co2abs-ci)/((ra + rsCO2)**2)
        dAn_dTs      = dci_dTs/ (ra + rsCO2) + drsCO2_dTs*(co2abs-ci)/((ra + rsCO2)**2)
        dAn_de       = dci_de/ (ra + rsCO2) + drsCO2_de*(co2abs-ci)/((ra + rsCO2)**2)
        dAn_dCO2     = -(dco2abs - dci_dCO2)/ (ra + rsCO2) + drsCO2_dCO2*(co2abs-ci)/((ra + rsCO2)**2)
        dAn_dra      = self.dra * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dPARfract    = drsCO2_dPARfract * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dSwin    = drsCO2_dSwin * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dcveg    = drsCO2_dcveg * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dw2      = drsCO2_dw2   * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dwfc      = drsCO2_dwfc   * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dwwilt      = drsCO2_dwwilt   * (co2abs-ci)/((ra + rsCO2)**2)
        dAn_dalfa_sto = -(co2abs - ci) * -1 * (ra + rsCO2)**(-2) * drsCO2_dalfa_sto
        dAn_dLAI      = drsCO2_dLAI   * (co2abs-ci)/((ra + rsCO2)**2)
        dfw_dwg      = -self.dwg* model.Cw * model.wmax / ((wg + model.wmin)**2)
        #texp = E0 / (283.15 * 8.314) * (1. - 283.15 / Tsoil) = E0 / (283.15 * 8.314) \
        #  - E0 / (283.15 * 8.314)*283.15/Tsoil = .. - E0/(8.314*Tsoil)
        dtexp_dE0 = self.dE0 / (283.15 * 8.314) * (1. - 283.15 / Tsoil)
        dtexp_dTsoil = self.dTsoil*E0/(8.314*Tsoil**2)
        #Resp         = R10 * (1. - fw) * np.exp(texp)
        dResp_dE0 = dtexp_dE0*R10 * (1. - fw) * np.exp(texp)
        dResp_dTsoil = dtexp_dTsoil*R10 * (1. - fw) * np.exp(texp)
        dResp_dwg    = -dfw_dwg*R10* np.exp(texp)
        dResp_dR10    = (1. - fw) * np.exp(texp) * self.dR10
        dwCO2A  = (dAn_dthetasurf + dAn_dTs + dAn_de + dAn_dCO2 + dAn_dra + dAn_dPARfract + dAn_dSwin + dAn_dcveg + dAn_dw2 + dAn_dwfc + dAn_dwwilt + dAn_dalfa_sto + dAn_dLAI)  * (model.mair / (model.rho * model.mco2))
        dwCO2R  = (dResp_dE0 + dResp_dTsoil + dResp_dwg + dResp_dR10)  * (model.mair / (model.rho * model.mco2))
        dwCO2   = dwCO2A + dwCO2R
        #self.hx = drs
        if model.ags_C_mode == 'MXL': 
            dwCOSP  = COS * (1 / gctCOS + ra)**(-2) * (-1 * gctCOS**(-2) * dgctCOS + self.dra) + -1 / (1 / gctCOS + ra) * self.dCOS
        elif model.ags_C_mode == 'surf':
            COSsurf = checkpoint['ags_COSsurf']
            dwCOSP  = COSsurf * (1 / gctCOS + ra)**(-2) * (-1 * gctCOS**(-2) * dgctCOS + self.dra) + -1 / (1 / gctCOS + ra) * self.dCOSsurf
        else:
            raise Exception('wrong ags_C_mode switch')
        if(model.input.soilCOSmodeltype == 'Sun_Ogee'):
            self.dmol_rat_ocs_atm = self.dCOSsurf #this is because COSsurf is given as argument to the function, while mol_rat_ocs_atm is used in the soil COS model
            self.dairtemp = self.dTsurf
            #self.dwsat = self.dwsat #not needed, but to show it goes for all the arguments, but e.g. this has an identical name
            dwCOSS_molm2s = self.tl_run_soil_COS_mod(model,checkpoint,returnvariable='dCOS_netuptake_soilsun')
            dwCOSS = dwCOSS_molm2s / model.rho * model.mair * 1.e-3 * 1.e9
        elif model.soilCOSmodeltype == None:
            dwCOSS = 0
        dwCOS   = dwCOSP + dwCOSS
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_ags.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_ags:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_ags[key]
        for key in self.Output_tl_ags:
            if key == returnvariable:
                returnvar = self.Output_tl_ags[returnvariable]
                return returnvar
        
    def tl_run_soil_COS_mod(self,model,checkpoint,returnvariable=None):
        sCOSm = model.soilCOSmodel #just for shorter notation
        self.Output_tl_rsCm = {}
        airtemp = checkpoint['rsCm_airtemp']
        mol_rat_ocs_atm = checkpoint['rsCm_mol_rat_ocs_atm']
        T_nodes = checkpoint['rsCm_T_nodes_end']
        s_moist = checkpoint['rsCm_s_moist_end']
        C_soilair_current = checkpoint['rsCm_C_soilair_current']
        Q10 = checkpoint['rsCm_Q10']
        SunTref = checkpoint['rsCm_SunTref']
        Vspmax = checkpoint['rsCm_Vspmax']
        wsat = checkpoint['rsCm_wsat']
        diffus_nodes = checkpoint['rsCm_diffus_nodes_end']
        D_a_0 = checkpoint['rsCm_D_a_0_end']
        C_air = checkpoint['rsCm_C_air_end']
        conduct = checkpoint['rsCm_conduct_end']
        dt = checkpoint['rsCm_dt']
        A_matr = checkpoint['rsCm_A_matr_end']
        B_matr = checkpoint['rsCm_B_matr_end']
        matr_3_eq12 = checkpoint['rsCm_matr_3_eq12_end']
        invmatreq12 = checkpoint['rsCm_invmatreq12_end']
        kH = checkpoint['rsCm_kH_end']
        Rgas = sCOSm.Rgas
        pressure = model.Ps
        
        dC_air = 1.e-9 * pressure / Rgas * (self.dmol_rat_ocs_atm / airtemp + mol_rat_ocs_atm * (-1) * airtemp**(-2) * self.dairtemp)
        dT_nodes = np.zeros(sCOSm.nr_nodes)
        ds_moist = np.zeros(sCOSm.nr_nodes)
        for i in range(0,sCOSm.nr_nodes):
            if sCOSm.sw_soiltemp == 'Sunpaper':
                dT_nodes[i] = 0
            elif sCOSm.sw_soiltemp == 'simple':
                if (sCOSm.z_soil[i] > sCOSm.layer1_2division):
                    dT_nodes[i] = self.dT2
                else:
                    dT_nodes[i] = self.dTsoil    
            elif sCOSm.sw_soiltemp == 'interpol': ##y= y1 + (x-x1) * (y2-y1)/(x2-x1); y1 is Tsoil, x1 is 0
                dT_nodes[i] = self.dTsoil + (sCOSm.z_soil[i] - 0) * (self.dT2 - self.dTsoil)/(1 - 0)    
            else:
                raise Exception('ERROR: Problem in soiltemp switch inputdata')
            if sCOSm.sw_soilmoisture == 'simple':
                if (sCOSm.z_soil[i] > sCOSm.layer1_2division):
                    ds_moist[i] = self.dw2
                else:
                    ds_moist[i] = self.dwg    
            elif sCOSm.sw_soilmoisture == 'interpol':
                ds_moist[i] = self.dwg + (sCOSm.z_soil[i] - 0) * (self.dw2 - self.dwg)/(1 - 0)
            else:
                raise Exception('ERROR: Problem in soilmoisture switch inputdata')
        #now the call to self.calckH()  
        if sCOSm.kH_type == 'Sun':
            alfa = checkpoint['rsCm_alfa_end']
            beta = checkpoint['rsCm_beta_end']
            K_eq20 = checkpoint['rsCm_K_eq20_end']
            dkH = (dT_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / T_nodes) + (T_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / T_nodes) * beta * K_eq20 * -1 *  T_nodes**(-2) * dT_nodes
        elif sCOSm.kH_type == 'Ogee':
            kHog = checkpoint['rsCm_kHog_end']
            dkHog = 0.021 * np.exp(24900/self.Rgas*(1/self.T_nodes-1/298.15)) * 24900/self.Rgas * -1 * T_nodes**(-2) * dT_nodes
            dkH = Rgas * 0.01 * (dkHog * T_nodes + kHog * dT_nodes)
        else:
            raise Exception('ERROR: Problem in kH_type switch inputdata')
        #now the call to soil_uptake
        if 	(sCOSm.uptakemodel == 'Sun'):
            raise Exception ('Sun uptake not implemented in TL')
        elif sCOSm.uptakemodel == 'Ogee':
            deltaHa = checkpoint['rsCm_deltaHa_end']
            deltaHd = checkpoint['rsCm_deltaHd_end']
            deltaSd = checkpoint['rsCm_deltaSd_end']
            xCA = checkpoint['rsCm_xCA_end']
            fCA = checkpoint['rsCm_fCA']
            kuncat_ref = checkpoint['rsCm_kuncat_ref_end']
            xCAref = checkpoint['rsCm_xCAref_end']
            ktot = checkpoint['rsCm_ktot_end']
            dxCA = np.exp(-deltaHa/(Rgas*T_nodes)) / (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas)) * -deltaHa/Rgas * -1 * T_nodes**(-2) * dT_nodes + np.exp(-deltaHa/(Rgas*T_nodes)) * -1 * (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas))**(-2) * np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas) * -deltaHd/Rgas* -1 * T_nodes**(-2) * dT_nodes
            dktot = kuncat_ref/xCAref * (self.dfCA*xCA + fCA*dxCA)
            ds_uptake = (s_moist * C_soilair_current) * (-dktot * kH - ktot * dkH) + (-ktot * kH) * (ds_moist * C_soilair_current + s_moist * self.dC_soilair_current)
        elif sCOSm.uptakemodel == 'newSun':
            raise Exception ('newSun uptake not implemented in TL')
        else:
            raise Exception('ERROR: Problem with uptake in switch inputdata')
        #now soil production
        ds_prod = self.dVspmax * Q10 **((T_nodes-SunTref)/10.0) + Vspmax * (((T_nodes - SunTref)/10.0) * Q10 **((T_nodes - SunTref)/10.0 - 1) * self.dQ10 + np.log(Q10) * Q10 **((T_nodes-SunTref)/10.0) * dT_nodes/10.0)
        #now calcD
        ddiffus = np.zeros(sCOSm.nr_nodes)
        if sCOSm.Diffus_type == ('Sun'):
            Dm = checkpoint['rsCm_Dm_end']
            n = checkpoint['rsCm_n_end']
            b_sCOSm = checkpoint['rsCm_b_sCOSm_end']
            D_a = checkpoint['rsCm_D_a_end']
            
            db_sCOSm = self.db_sCOSm
            #d(a^x)/dx = ln(a) * a^x
            #Dm * (wsat - self.s_moist)**2 * ((wsat - self.s_moist)/wsat)**(3./b_sCOSm) * (self.T_nodes/self.SunTref)**n
            ddiffus_nodes_dwsat = Dm * (T_nodes / SunTref)**n * (2 * (wsat - s_moist) * self.dwsat * ((wsat - s_moist)/wsat)**(3./b_sCOSm) + (wsat - s_moist)**2 * (3./b_sCOSm) * 
                                        ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * (1 / wsat * self.dwsat + (wsat - s_moist) * -1 * wsat**(-2) * self.dwsat))
            ddiffus_nodes_ds_moist = Dm * (T_nodes / SunTref)**n * (2 * (wsat - s_moist) * - ds_moist * ((wsat - s_moist)/wsat)**(3./b_sCOSm) + \
                                     (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * - ds_moist/wsat)
            ddiffus_nodes_db_sCOSm = Dm * (wsat - s_moist)**2 * np.log((wsat - s_moist)/wsat) * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * 3 * -1 * b_sCOSm**(-2) * db_sCOSm * (T_nodes/SunTref)**n
            ddiffus_nodes_dT_nodes = Dm * (wsat - s_moist)**2 * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * n * (T_nodes / SunTref)**(n-1) * 1 / SunTref * dT_nodes
            ddiffus_nodes = ddiffus_nodes_dwsat + ddiffus_nodes_ds_moist + ddiffus_nodes_db_sCOSm + ddiffus_nodes_dT_nodes
            dD_a = Dm * n * (airtemp/SunTref)**(n-1) * self.dairtemp/SunTref
            ddiffus[0] = 2. * -1 * (1./diffus_nodes[0] + 1./D_a)**(-2) * (-1 * (diffus_nodes[0])**(-2) * ddiffus_nodes[0] + -1 * D_a**(-2) * dD_a)
            for i in range(1,sCOSm.nr_nodes):
                ddiffus[i] = (ddiffus_nodes[i] + ddiffus_nodes[i-1])/2.
        elif sCOSm.Diffus_type == ('Ogee'):
            raise Exception('Ogee diffusion not yet implemented')
        else:
            raise Exception('Error in Diffus_type switch inputdata')
        #now calcG
        dconduct = np.zeros(sCOSm.nr_nodes)
        for i in range(1,sCOSm.nr_nodes):
            dconduct[i] = ddiffus[i] / (sCOSm.z_soil[i] - sCOSm.z_soil[i-1])
        dconduct[0] = ddiffus[0] / (sCOSm.z_soil[0] - 0)
        #now calcC
        C_soilair = checkpoint['rsCm_C_soilair_middle']
        dsource = (ds_uptake+ds_prod)*sCOSm.dz_soil
        dD_a_0 = ddiffus[0]
        dsource[0] = (ds_uptake[0]+ds_prod[0])*sCOSm.dz_soil[0] + 1 / sCOSm.z_soil[0] * (dD_a_0 * C_air + D_a_0 * dC_air)
        deta = dkH * s_moist + kH * ds_moist + (self.dwsat - ds_moist)
        dA_matr = np.zeros((sCOSm.nr_nodes,sCOSm.nr_nodes))
        for i in range(sCOSm.nr_nodes):
            dA_matr[i,i] = deta[i] * sCOSm.dz_soil[i]
        dB_matr = np.zeros((sCOSm.nr_nodes,sCOSm.nr_nodes))
        for i in range(sCOSm.nr_nodes-1):
            dB_matr[i,i] = -(dconduct[i]+dconduct[i+1])
            dB_matr[i,i+1] = dconduct[i+1]
        dB_matr[sCOSm.nr_nodes-1,sCOSm.nr_nodes-1] = -dconduct[sCOSm.nr_nodes-1]
        for i in range(1,sCOSm.nr_nodes):
            dB_matr[i,i-1] = dconduct[i]
        #invmatreq12 = np.linalg.inv(2*A_matr - dt*B_matr)
        #from https://math.stackexchange.com/questions/1471825/derivative-of-the-inverse-of-a-matrix
        dinvmatreq12 = np.matmul(-1*np.linalg.inv(2*A_matr - dt*B_matr),np.matmul((2*dA_matr - dt*dB_matr),np.linalg.inv(2*A_matr - dt*B_matr)))
        #matr_2_eq12 = np.matmul(2*A_matr + dt * B_matr, self.C_soilair_current)
        dmatr_2_eq12 = np.matmul((2*dA_matr + dt * dB_matr),(C_soilair_current)) + np.matmul((2*A_matr + dt * B_matr),self.dC_soilair_current) #dC_soilair_current is a vector!
        dmatr_3_eq12 = dmatr_2_eq12 + 2*dt* dsource #
        dC_soilair = np.matmul(dinvmatreq12,matr_3_eq12) + np.matmul(invmatreq12,dmatr_3_eq12) 
        for i in range(sCOSm.nr_nodes): 
            if (C_soilair[i] < 0.0): #we really need a small numerical perturbation in the gradient test for this, otherwise Conc becomes 0
                dC_soilair[i] = 0.0 #not a problem, since the adjoint is a local derivative! So no need to take into account that a perturbation can make C_soilair negative or positive, only the value of the default run (reference around where we take the derivative) counts
        dC_soilair_next = cp.deepcopy(dC_soilair) #from the assignment of the function result to self.C_soilair_next
        #now the fluxes
        #fluxes use C_soilair_current as argument, not C_soilair_next
        C_soilair = checkpoint['rsCm_C_soilair_calcJ']
        dOCS_fluxes = np.zeros(sCOSm.nr_nodes)
        for i in range(1,sCOSm.nr_nodes):
            dOCS_fluxes[i] = -1. * (dconduct[i] * (C_soilair[i] - C_soilair[i-1]) + conduct[i] * (self.dC_soilair_current[i] - self.dC_soilair_current[i-1]))
        dOCS_fluxes[0] = -1. * (dconduct[0] * (C_soilair[0] - C_air) + conduct[0] * (self.dC_soilair_current[0] - dC_air))
        dCOS_netuptake_soilsun = -1 * dOCS_fluxes[0]
        dC_soilair_current = cp.deepcopy(dC_soilair_next)
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rsCm.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rsCm:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rsCm[key]
        for key in self.Output_tl_rsCm:
            if key == returnvariable:
                returnvar = self.Output_tl_rsCm[returnvariable]
                return returnvar
    
    def tl_run_cumulus(self,model,checkpoint,returnvariable=None):
        self.Output_tl_rc = {}
        wthetav = checkpoint['rc_wthetav']
        deltaq = checkpoint['rc_deltaq']
        dz_h = checkpoint['rc_dz_h']
        wstar = checkpoint['rc_wstar']
        wqe = checkpoint['rc_wqe']
        wqM = checkpoint['rc_wqM']
        h = checkpoint['rc_h']
        deltaCO2 = checkpoint['rc_deltaCO2']
        deltaCOS = checkpoint['rc_deltaCOS']
        wCO2e = checkpoint['rc_wCO2e']
        wCO2M = checkpoint['rc_wCO2M']
        wCOSe = checkpoint['rc_wCOSe']
        wCOSM = checkpoint['rc_wCOSM']
        q = checkpoint['rc_q']
        T_h = checkpoint['rc_T_h']
        P_h = checkpoint['rc_P_h']
        ac = checkpoint['rc_ac_end']
        M = checkpoint['rc_M_end']
        qsat_variable_rc = checkpoint['rc_qsat_variable_rc_end']
        q2_h = checkpoint['rc_q2_h_middle']
        CO22_h = checkpoint['rc_CO22_h_middle']
        COS2_h = checkpoint['rc_COS2_h_middle']
        
        if(wthetav > 0):
            dq2_h_dwqe = -(self.dwqe) * deltaq   * h / (dz_h * wstar)
            dq2_h_dwqM = -(self.dwqM) * deltaq   * h / (dz_h * wstar)
            dq2_h_ddeltaq = -(wqe  + wqM  ) * h / (dz_h * wstar) * self.ddeltaq
            dq2_h_dh = -(wqe  + wqM  ) * deltaq / (dz_h * wstar) * self.dh
            dq2_h_ddz_h = -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            dq2_h_dwstar = -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            dq2_h = dq2_h_dwqe + dq2_h_dwqM + dq2_h_ddeltaq + dq2_h_dh + dq2_h_ddz_h + dq2_h_dwstar
            dCO22_h_dwCO2e = -(self.dwCO2e) * deltaCO2 * h / (dz_h * wstar)
            dCO22_h_dwCO2M = -(self.dwCO2M) * deltaCO2 * h / (dz_h * wstar)
            dCO22_h_ddeltaCO2 = -(wCO2e+ wCO2M) * h / (dz_h * wstar) * self.ddeltaCO2
            dCO22_h_dh = -(wCO2e+ wCO2M) * deltaCO2 / (dz_h * wstar) * self.dh
            dCO22_h_ddz_h = -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            dCO22_h_dwstar = -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            dCO22_h = dCO22_h_dwCO2e + dCO22_h_dwCO2M + dCO22_h_ddeltaCO2 + dCO22_h_dh + dCO22_h_ddz_h + dCO22_h_dwstar
            dCOS2_h_dwCOSe = -(self.dwCOSe) * deltaCOS * h / (dz_h * wstar)
            dCOS2_h_dwCOSM = -(self.dwCOSM) * deltaCOS * h / (dz_h * wstar)
            dCOS2_h_ddeltaCOS = -(wCOSe+ wCOSM) * h / (dz_h * wstar) * self.ddeltaCOS
            dCOS2_h_dh = -(wCOSe+ wCOSM) * deltaCOS / (dz_h * wstar) * self.dh
            dCOS2_h_ddz_h = -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            dCOS2_h_dwstar = -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            dCOS2_h = dCOS2_h_dwCOSe + dCOS2_h_dwCOSM + dCOS2_h_ddeltaCOS + dCOS2_h_dh + dCOS2_h_ddz_h + dCOS2_h_dwstar

        else:
            dq2_h = 0.
            dCO22_h = 0.
            dCOS2_h = 0.
        if q2_h <= 0.:
            dq2_h   = 0.
        if CO22_h <= 0.:
            dCO22_h = 0.
        if COS2_h <= 0.:
            dCOS2_h = 0.
        q2_h = checkpoint['rc_q2_h_end']
        CO22_h = checkpoint['rc_CO22_h_end']
        COS2_h = checkpoint['rc_COS2_h_end']
        #ac     = max(0., 0.5 + (0.36 * np.arctan(1.55 * ((q - qsat(T_h, P_h)) / q2_h**0.5))))
        if 0.5 + (0.36 * np.arctan(1.55 * ((q - qsat_variable_rc) / q2_h**0.5))) < 0:
            dac = 0
        else: #darctan(x)/dx = 1 / (1+x**2)
            dqsat_variable_rc_dT_h = dqsat_dT(T_h,P_h,self.dT_h) #rc stands for run cumulus, added since in statistics module the name qsat_variable already exists
            dqsat_variable_rc_dP_h = dqsat_dp(T_h,P_h,self.dP_h)
            dac_dq = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * self.dq
            dac_dT_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * dqsat_variable_rc_dT_h
            dac_dP_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * dqsat_variable_rc_dP_h
            dac_dq2_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * (q - qsat_variable_rc) * -0.5 * q2_h**(-1.5) * dq2_h
            dac = dac_dq + dac_dT_h + dac_dP_h + dac_dq2_h
        dM = dac * wstar + ac * self.dwstar
        dwqM = dM * q2_h**0.5 + M * 0.5 * q2_h**(-0.5) * dq2_h
        if(deltaCO2 < 0):
            dwCO2M  = dM * CO22_h**0.5 + M * 0.5 * CO22_h**(-0.5) * dCO22_h
        else:
            dwCO2M  = 0.
        if(deltaCOS < 0):
            dwCOSM  = dM * COS2_h**0.5 + M * 0.5 * COS2_h**(-0.5) * dCOS2_h
        else:
            dwCOSM  = 0.
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rc.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rc:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rc[key]
        if returnvariable is not None:
            for key in self.Output_tl_rc:
                if key == returnvariable:
                    returnvar = self.Output_tl_rc[returnvariable]
                    return returnvar
    
    def tl_run_mixed_layer(self,model,checkpoint,returnvariable=None):
        h = checkpoint['rml_h']
        ustar = checkpoint['rml_ustar']
        u = checkpoint['rml_u']
        v = checkpoint['rml_v']
        uw = checkpoint['rml_uw_end']
        vw = checkpoint['rml_vw_end']
        deltatheta = checkpoint['rml_deltatheta']
        deltathetav = checkpoint['rml_deltathetav']
        deltau = checkpoint['rml_deltau']
        deltav = checkpoint['rml_deltav']
        thetav = checkpoint['rml_thetav']
        wthetav = checkpoint['rml_wthetav']
        wthetave = checkpoint['rml_wthetave_end']
        deltaq = checkpoint['rml_deltaq']
        deltaCO2 = checkpoint['rml_deltaCO2']
        deltaCOS = checkpoint['rml_deltaCOS']
        wtheta = checkpoint['rml_wtheta']
        wthetae = checkpoint['rml_wthetae_end']
        wq = checkpoint['rml_wq']
        wqe = checkpoint['rml_wqe_end']
        wqM = checkpoint['rml_wqM']
        wCO2 = checkpoint['rml_wCO2']
        wCO2e = checkpoint['rml_wCO2e_end']
        wCO2M = checkpoint['rml_wCO2M']
        wCOS = checkpoint['rml_wCOS']
        wCOSe = checkpoint['rml_wCOSe_end']
        wCOSM = checkpoint['rml_wCOSM']
        ac = checkpoint['rml_ac']
        lcl = checkpoint['rml_lcl']
        ws = checkpoint['rml_ws_end']
        wf = checkpoint['rml_wf_end']
        M = checkpoint['rml_M']
        gammatheta = checkpoint['rml_gammatheta']
        gammatheta2 = checkpoint['rml_gammatheta2']
        htrans = checkpoint['rml_htrans']
        gammaq = checkpoint['rml_gammaq']
        gammaCO2 = checkpoint['rml_gammaCO2']
        gammaCOS = checkpoint['rml_gammaCOS']
        gammau = checkpoint['rml_gammau']
        gammav = checkpoint['rml_gammav']
        divU = checkpoint['rml_divU']
        fc = checkpoint['rml_fc']
        dFz = checkpoint['rml_dFz']
        wstar = checkpoint['rml_wstar_end']
        beta = checkpoint['rml_beta_end']
        self.Output_tl_rml = {}
        if(not model.sw_sl):
            duw_dustar = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * self.dustar * 1 / (v ** 2. / u ** 2. + 1.)
            duw_du = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * (v **2.) * -2 * u**(-3) * self.du 
            duw_dv = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * 1 / (u ** 2.) * 2 * v * self.dv
            duw = duw_dustar + duw_du + duw_dv
            self.duw = duw  #we need to use self here (LHS of equation) because self.duw occurs later in this function, depending on a switch the duw defined here or a different duw will be used. The duw defined here has therefore to be declared a self variable. Does not need adj statement
            dvw_dustar = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * self.dustar * 1 / (u ** 2. / v ** 2. + 1.)
            dvw_du = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * 1 / (v **2.) * 2 * u * self.du 
            dvw_dv = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * (u **2.) * -2 * v**(-3) * self.dv
            dvw = dvw_dustar + dvw_du + dvw_dv
            self.dvw = dvw
        dws = - h * self.ddivU - divU * self.dh    
        if(model.sw_fixft):
            if h <= htrans:
                dw_th_ft  = ws * self.dgammatheta + gammatheta * dws
            else:
                dw_th_ft  = ws * self.dgammatheta2 + gammatheta2 * dws
            dw_q_ft   = ws * self.dgammaq + gammaq * dws
            dw_CO2_ft = ws * self.dgammaCO2 + gammaCO2   * dws
            dw_COS_ft = ws * self.dgammaCOS + gammaCOS   * dws
        else:
            dw_th_ft = 0
            dw_q_ft  = 0
            dw_CO2_ft = 0
            dw_COS_ft = 0
        dwf = 1 / (model.rho * model.cp) * (1 / deltatheta * self.ddFz + dFz * (-1) * deltatheta**(-2) * self.ddeltatheta) #dFz is cloud top radiative divergence [W m-2]
        if(wthetav > 0.):
            dwstar_dh = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * (model.g * wthetav / thetav) * self.dh
            dwstar_dwthetav = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h / thetav * self.dwthetav
            dwstar_dthetav = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h * wthetav * (-1) * thetav**(-2) * self.dthetav
        else:
            dwstar_dh  = 0
            dwstar_dwthetav = 0
            dwstar_dthetav = 0
        dwstar = dwstar_dh + dwstar_dwthetav + dwstar_dthetav
        if model.sw_dyn_beta:
            dbeta = 5 * 3 * (ustar/wstar)**2. * (1 / wstar * self.dustar + ustar * -1 * wstar**-2 * dwstar)
            self.dbeta = dbeta #needed since in the statement below, dbeta either comes from outside this module (self.dbeta), or this dbeta is used, depending on a switch
        dwthetave = -1 * beta * self.dwthetav - wthetav * self.dbeta
        if(model.sw_shearwe):
            dwe_dwthetav = -1 / deltathetav * dwthetave
            dwe_dustar = 5. * thetav / (model.g * h * deltathetav) * 3 * ustar**2 * self.dustar
            dwe_dthetav = 5. * ustar ** 3. / (model.g * h * deltathetav) * self.dthetav
            dwe_dh    = 5* ustar ** 3. * thetav / model.g / deltathetav * (-1) * h**(-2) * self.dh
            dwe_ddeltathetav = (-wthetave + 5. * ustar ** 3. * thetav / (model.g * h)) * (-1) * deltathetav**(-2) * self.ddeltathetav
        else:
            dwe_dwthetav = -1/ deltathetav * dwthetave
            dwe_dustar = 0
            dwe_dthetav = 0
            dwe_dh    = 0
            dwe_ddeltathetav = -wthetave * (-1) * deltathetav**(-2) * self.ddeltathetav
        we = checkpoint['rml_we_middle']
        if(we < 0):
            dwe_dwthetav = 0
            dwe_dustar = 0
            dwe_dthetav = 0
            dwe_dh = 0.
            dwe_ddeltathetav = 0
        dwe = dwe_dwthetav + dwe_dustar + dwe_dthetav + dwe_dh + dwe_ddeltathetav
        we = checkpoint['rml_we_end']
        dwthetae = -dwe * deltatheta + -we * self.ddeltatheta
        dwqe = -dwe * deltaq + -we * self.ddeltaq
        dwCO2e = -dwe * deltaCO2 + -we * self.ddeltaCO2
        dwCOSe = -dwe * deltaCOS + -we * self.ddeltaCOS
        dhtend = dwe + dws + dwf - self.dM
        dthetatend = (self.dwtheta - dwthetae) / h + (wtheta - wthetae) * (-1) * h**(-2) * self.dh + self.dadvtheta 
        dqtend = (self.dwq - dwqe - self.dwqM) / h + (wq - wqe - wqM) * (-1) * h**(-2) * self.dh + self.dadvq
        dCO2tend = (self.dwCO2 - dwCO2e - self.dwCO2M) / h + (wCO2 - wCO2e - wCO2M) * (-1) * h**(-2) * self.dh + self.dadvCO2
        dCOStend = (self.dwCOS - dwCOSe - self.dwCOSM) / h + (wCOS - wCOSe - wCOSM) * (-1) * h**(-2) * self.dh + self.dadvCOS
        if h <= htrans:
            ddeltathetatend = (we + wf - M) * self.dgammatheta + gammatheta * (dwe + dwf - self.dM) - dthetatend + dw_th_ft
        else:
            ddeltathetatend = (we + wf - M) * self.dgammatheta2 + gammatheta2 * (dwe + dwf - self.dM) - dthetatend + dw_th_ft
        ddeltaqtend = (we + wf - M) * self.dgammaq + gammaq * (dwe + dwf - self.dM) - dqtend + dw_q_ft
        ddeltaCO2tend = (we + wf - M) * self.dgammaCO2 + gammaCO2 * (dwe + dwf - self.dM) - dCO2tend + dw_CO2_ft
        ddeltaCOStend = (we + wf - M) * self.dgammaCOS + gammaCOS * (dwe + dwf - self.dM) - dCOStend + dw_COS_ft
        if model.sw_advfp:
            ddeltathetatend += self.dadvtheta
            ddeltaqtend += self.dadvq 
            ddeltaCO2tend += self.dadvCO2
            ddeltaCOStend += self.dadvCOS
        if(model.sw_wind): 
            dutend = -self.dfc * deltav - fc * self.ddeltav + (self.duw + dwe * deltau + we * self.ddeltau) / h + (uw + we * deltau) * -1 * h**(-2) * self.dh + self.dadvu
            dvtend = self.dfc * deltau + fc * self.ddeltau + (self.dvw + dwe * deltav + we * self.ddeltav) / h + (vw + we * deltav) * -1 * h**(-2) * self.dh + self.dadvv
            ddeltautend = self.dgammau * (we + wf - M) + gammau * (dwe + dwf - self.dM) - dutend
            ddeltavtend = self.dgammav * (we + wf - M) + gammav * (dwe + dwf - self.dM) - dvtend
            if model.sw_advfp:
                ddeltautend += self.dadvu 
                ddeltavtend += self.dadvv 
        if(ac > 0 or lcl - h < 300):
            ddztend = (self.dlcl - self.dh - self.ddz_h) / 7200.
        else:
            ddztend = 0
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_rml.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_rml:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_rml[key]
        if returnvariable is not None:
            for key in self.Output_tl_rml:
                if key == returnvariable:
                    returnvar = self.Output_tl_rml[returnvariable]
                    return returnvar
    
    def tl_store(self,model,checkpoint,returnvariable=None):
        #not all variables of the forward model necessarily have to be included here, if you don't use observations of the output variables you dont really need them for the optimisation framework
        self.Output_tl_sto = {}
        fac = checkpoint['sto_fac_end']
        dout_h             = self.dh
        dout_theta         = self.dtheta
        dout_thetav        = self.dthetav
        dout_deltatheta    = self.ddeltatheta
        dout_deltathetav   = self.ddeltathetav
        dout_wtheta        = self.dwtheta
        dout_wthetav       = self.dwthetav
        dout_wthetae       = self.dwthetae
        dout_wthetave      = self.dwthetave
        dout_q             = self.dq
        dout_deltaq        = self.ddeltaq
        dout_wq            = self.dwq
        dout_wqe           = self.dwqe
        dout_wqM           = self.dwqM
        dout_qsatvar          = self.dqsatvar
        dout_e             = self.de
        dout_esatvar          = self.desatvar
        dout_CO2           = self.dCO2
        dout_deltaCO2      = self.ddeltaCO2
        dout_wCO2          = self.dwCO2 * fac
        dout_wCO2A         = self.dwCO2A * fac
        dout_wCO2R         = self.dwCO2R *fac
        dout_wCO2e         = self.dwCO2e * fac
        dout_wCO2M         = self.dwCO2M * fac
        dout_wCOS          = self.dwCOS
        if model.sw_ls:
            if model.ls_type=='ags':
                dout_wCOSP         = self.dwCOSP
                dout_wCOSS         = self.dwCOSS
        if model.ls_type=='canopy_model':
            pass #to be implemented later
        dout_COS           = self.dCOS
        dout_deltaCOS      = self.ddeltaCOS
        dout_u             = self.du
        dout_deltau        = self.ddeltau
        dout_uw            = self.duw
        dout_v             = self.dv
        dout_deltav        = self.ddeltav
        dout_vw            = self.dvw
        
        dout_T2m           = self.dT2m
        dout_q2m           = self.dq2m
        dout_u2m           = self.du2m
        dout_v2m           = self.dv2m
        dout_e2m           = self.de2m
        dout_esat2m        = self.desat2m
        if model.sw_sl:
            
            dout_thetamh       = self.dthetamh
            dout_thetamh2      = self.dthetamh2
            dout_thetamh3      = self.dthetamh3
            dout_thetamh4      = self.dthetamh4
            dout_thetamh5      = self.dthetamh5
            dout_thetamh6      = self.dthetamh6
            dout_thetamh7      = self.dthetamh7
            dout_Tmh           = self.dTmh
            dout_Tmh2          = self.dTmh2
            dout_Tmh3          = self.dTmh3
            dout_Tmh4          = self.dTmh4
            dout_Tmh5          = self.dTmh5
            dout_Tmh6          = self.dTmh6
            dout_Tmh7          = self.dTmh7
            
            dout_qmh           = self.dqmh
            dout_qmh2          = self.dqmh2
            dout_qmh3          = self.dqmh3
            dout_qmh4          = self.dqmh4
            dout_qmh5          = self.dqmh5
            dout_qmh6          = self.dqmh6
            dout_qmh7          = self.dqmh7
            dout_COSmh         = self.dCOSmh
            dout_COSmh2        = self.dCOSmh2
            dout_COSmh3        = self.dCOSmh3
            dout_CO22m         = self.dCO22m
            dout_CO2mh         = self.dCO2mh
            dout_CO2mh2        = self.dCO2mh2
            dout_CO2mh3        = self.dCO2mh3
            dout_CO2mh4        = self.dCO2mh4
            dout_COS2m         = self.dCOS2m
            dout_COSsurf       = self.dCOSsurf
            dout_CO2surf       = self.dCO2surf
            dout_Tsurf         = self.dTsurf
            dout_Cm            = self.dCm
        dout_thetasurf     = self.dthetasurf
        dout_thetavsurf    = self.dthetavsurf
        dout_qsurf         = self.dqsurf
        dout_ustar         = self.dustar
        dout_Cs            = self.dCs
        dout_L             = self.dL
        dout_Rib           = self.dRib
        dout_Swin          = self.dSwin
        dout_Swout         = self.dSwout
        dout_Lwin          = self.dLwin
        dout_Lwout         = self.dLwout
        dout_Q             = self.dQ
        dout_ra            = self.dra
        dout_rs            = self.drs
        dout_H             = self.dH
        dout_LE            = self.dLE
        dout_LEliq         = self.dLEliq
        dout_LEveg         = self.dLEveg
        dout_LEsoil        = self.dLEsoil
        dout_LEpot         = self.dLEpot
        dout_LEref         = self.dLEref
        dout_G             = self.dG
        dout_Ts            = self.dTs
        dout_zlcl          = self.dlcl
        dout_RH_h          = self.dRH_h
        dout_ac            = self.dac
        dout_M             = self.dM
        dout_dz            = self.ddz_h   
        
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_sto.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_sto:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_sto[key]
        if returnvariable is not None:
            for key in self.Output_tl_sto:
                if key == returnvariable:
                    returnvar = self.Output_tl_sto[returnvariable]
                    return returnvar
        
    def tl_integrate_land_surface(self,model,checkpoint,returnvariable=None):
        self.Output_tl_ils = {}
        dTsoil0        = self.dTsoil
        dwg0           = self.dwg
        dWl0           = self.dWl
        dTsoil    = dTsoil0  + model.dt * self.dTsoiltend
        dwg       = dwg0     + model.dt * self.dwgtend
        dWl       = dWl0     + model.dt * self.dWltend
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_ils.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_ils:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_ils[key]
        if returnvariable is not None:
            for key in self.Output_tl_ils:
                if key == returnvariable:
                    returnvar = self.Output_tl_ils[returnvariable]
                    return returnvar
            
    def tl_integrate_mixed_layer(self,model,checkpoint,returnvariable=None):
        dz_h = checkpoint['iml_dz_h_middle']
        self.Output_tl_iml = {}
        dh0 = self.dh
        dtheta0  = self.dtheta
        ddeltatheta0 = self.ddeltatheta
        dq0      = self.dq
        ddeltaq0     = self.ddeltaq
        dCO20    = self.dCO2
        dCOS0    = self.dCOS
        ddeltaCO20   = self.ddeltaCO2
        ddeltaCOS0   = self.ddeltaCOS
        du0      = self.du
        ddeltau0 = self.ddeltau
        dv0      = self.dv
        ddeltav0 = self.ddeltav
        ddz0     = self.ddz_h
        dh        = dh0      + model.dt * self.dhtend
        dtheta    = dtheta0  + model.dt * self.dthetatend
        ddeltatheta   = ddeltatheta0 + model.dt * self.ddeltathetatend
        dq        = dq0      + model.dt * self.dqtend
        ddeltaq       = ddeltaq0     + model.dt * self.ddeltaqtend
        dCO2      = dCO20    + model.dt * self.dCO2tend
        dCOS      = dCOS0    + model.dt * self.dCOStend
        ddeltaCO2     = ddeltaCO20   + model.dt * self.ddeltaCO2tend
        ddeltaCOS     = ddeltaCOS0   + model.dt * self.ddeltaCOStend
        ddz_h     = ddz0     + model.dt * self.ddztend
        dz0 = 50
        ddz0 = 0
        if(dz_h < dz0):
            ddz_h = 0
        if(model.sw_wind):
            #before here was the statement: raise Exception ('wind not implemented')
            du        = du0      + model.dt * self.dutend
            ddeltau   = ddeltau0     + model.dt * self.ddeltautend
            dv        = dv0      + model.dt * self.dvtend
            ddeltav   = ddeltav0     + model.dt * self.ddeltavtend
        the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
        for variablename in the_locals: #note that the self variables are not included
            if variablename.startswith('d'): #still includes some unnecessary stuff
                self.Output_tl_iml.update({variablename: the_locals[variablename]})
        if (self.adjointtesting or self.gradienttesting):
            for key in self.Output_tl_iml:
                if key in self.__dict__:
                    self.__dict__[key] = self.Output_tl_iml[key]
        if returnvariable is not None:
            for key in self.Output_tl_iml:
                if key == returnvariable:
                    returnvar = self.Output_tl_iml[returnvariable]
                    return returnvar
    
    def initialise_adjoint(self):
        #All variables that could be used in multiple modules, without being set to zero after use in every module should appear here
        #surface layer derivatives
        self.adq2m = 0
        self.adqmh = 0
        self.ade2m = 0
        self.adT2m = 0
        self.adthetamh = 0
        self.adthetamh2 = 0
        self.adthetamh3 = 0
        self.adthetamh4 = 0
        self.adthetamh5 = 0
        self.adthetamh6 = 0
        self.adthetamh7 = 0
        self.adTmh = 0
        self.adTmh2 = 0
        self.adTmh3 = 0
        self.adTmh4 = 0
        self.adTmh5 = 0
        self.adTmh6 = 0
        self.adTmh7 = 0
        self.adesat2m = 0
        self.adv2m_dvw = 0
        self.adv2m = 0
        self.adv2m_dustar = 0 #you can add a statement dv2m_dustar = 0 which belongs to this statement to the end of the TL, this doesnt change the model (except for grad test)
        self.adv2m_dL = 0
        self.adpsim_2_L = 0
        self.adpsim_z0m_L = 0
        self.adustar = 0
        self.advw = 0
        self.adu2m = 0
        self.adu2m_duw = 0
        self.adu2m_dustar = 0
        self.adu2m_dL = 0
        self.adzeta_dL_z0m = 0
        self.adzeta_dL_2 = 0
        self.adzeta_dL_Tmh = 0
        self.adzeta_dL_Tmh2 = 0
        self.adzeta_dL_Tmh3 = 0
        self.adzeta_dL_Tmh4 = 0
        self.adzeta_dL_Tmh5 = 0
        self.adzeta_dL_Tmh6 = 0
        self.adzeta_dL_Tmh7 = 0
        self.adzeta_dL_qmh = 0
        self.adzeta_dL_qmh2 = 0
        self.adzeta_dL_qmh3 = 0
        self.adzeta_dL_qmh4 = 0
        self.adzeta_dL_qmh5 = 0
        self.adzeta_dL_qmh6 = 0
        self.adzeta_dL_qmh7 = 0
        self.adzeta_dL_COSmh = 0
        self.adzeta_dL_COSmh2 = 0
        self.adzeta_dL_COSmh3 = 0
        self.adzeta_dL_CO2mh = 0
        self.adzeta_dL_CO2mh2 = 0
        self.adzeta_dL_CO2mh3 = 0
        self.adzeta_dL_CO2mh4 = 0
        self.aduw = 0
        self.adCOSmh_dCOSsurf = 0
        self.adCOSmh = 0
        self.adCOSmh_dwCOS = 0
        self.adCOSmh_dustar = 0
        self.adCOSmh_dL = 0
        self.adCO2mh_dz0h = 0
        self.adCOSmh2_dCOSsurf = 0
        self.adCOSmh2 = 0
        self.adCOSmh2_dwCOS = 0
        self.adCOSmh2_dustar = 0
        self.adCOSmh2_dL = 0
        self.adCOSmh3_dCOSsurf = 0
        self.adCOSmh3 = 0
        self.adCOSmh3_dwCOS = 0
        self.adCOSmh3_dustar = 0
        self.adCOSmh3_dL = 0
        self.adCO2mh_dCO2surf = 0
        self.adCO2mh = 0
        self.adCO2mh_dwCO2 = 0
        self.adCO2mh_dustar = 0
        self.adCO2mh_dL = 0
        self.adCO2mh2_dCO2surf = 0
        self.adCO2mh2 = 0
        self.adCO2mh2_dwCO2 = 0
        self.adCO2mh2_dustar = 0
        self.adCO2mh2_dL = 0
        self.adCO2mh2_dz0h = 0
        self.adCO2mh3_dCO2surf = 0
        self.adCO2mh3 = 0
        self.adCO2mh3_dwCO2 = 0
        self.adCO2mh3_dustar = 0
        self.adCO2mh3_dL = 0 
        self.adCO2mh3_dz0h = 0
        self.adCO2mh4_dCO2surf = 0
        self.adCO2mh4 = 0
        self.adCO2mh4_dwCO2 = 0
        self.adCO2mh4_dustar = 0
        self.adCO2mh4_dL = 0
        self.adCO2mh4_dz0h = 0
        self.adpsih_COSmh_L = 0
        self.adpsih_COSmh2_L = 0
        self.adpsih_COSmh3_L = 0
        self.adpsih_CO2mh_L = 0
        self.adpsih_CO2mh2_L = 0
        self.adpsih_CO2mh3_L = 0
        self.adpsih_CO2mh4_L = 0
        self.adpsih_z0h_L = 0
        self.adCOSsurf = 0
        self.adCOS2m_dCOSsurf = 0
        self.adCOS2m = 0
        self.adCOS2m_dwCOS = 0
        self.adCOS2m_dustar = 0
        self.adCOS2m_dL = 0
        self.adCO22m_dCO2surf = 0
        self.adCO22m = 0
        self.adCO22m_dwCO2 = 0
        self.adCO22m_dustar = 0
        self.adCO22m_dL = 0
        self.adpsih_2_L = 0
        self.adpsih_qmh_L = 0
        self.adpsih_qmh2_L = 0
        self.adpsih_qmh3_L = 0
        self.adpsih_qmh4_L = 0
        self.adpsih_qmh5_L = 0
        self.adpsih_qmh6_L = 0
        self.adpsih_qmh7_L = 0
        self.adpsih_Tmh_L = 0
        self.adpsih_Tmh2_L = 0
        self.adpsih_Tmh3_L = 0
        self.adpsih_Tmh4_L = 0
        self.adpsih_Tmh5_L = 0
        self.adpsih_Tmh6_L = 0
        self.adpsih_Tmh7_L = 0
        self.adq2m_dqsurf = 0
        self.adqmh_dqsurf = 0
        self.adwq = 0
        self.adq2m_dwq = 0
        self.adq2m_dustar = 0
        self.adq2m_dL = 0
        self.adqmh_dwq = 0
        self.adqmh_dustar = 0
        self.adqmh_dL = 0
        self.adqsurf = 0
        self.adqsurf_dq = 0
        self.adqsurf_dwq = 0
        self.adqsurf_dCs_start = 0
        self.adqsurf_dueff = 0
        self.adesurf = 0
        self.adT2m_dthetasurf = 0
        self.adwtheta = 0
        self.adT2m_dwtheta = 0
        self.adT2m_dustar = 0
        self.adT2m_dL = 0
        self.adthetamh_dthetasurf = 0
        self.adthetamh_dwtheta = 0
        self.adthetamh_dustar = 0
        self.adthetamh_dL = 0
        self.adthetamh_dz0h = 0
        self.adthetamh2_dthetasurf = 0
        self.adthetamh2_dwtheta = 0
        self.adthetamh2_dustar = 0
        self.adthetamh2_dL = 0
        self.adthetamh2_dz0h = 0
        self.adthetamh3_dthetasurf = 0
        self.adthetamh3_dwtheta = 0
        self.adthetamh3_dustar = 0
        self.adthetamh3_dL = 0
        self.adthetamh3_dz0h = 0
        self.adthetamh4_dthetasurf = 0
        self.adthetamh4_dwtheta = 0
        self.adthetamh4_dustar = 0
        self.adthetamh4_dL = 0
        self.adthetamh4_dz0h = 0
        self.adthetamh5_dthetasurf = 0
        self.adthetamh5_dwtheta = 0
        self.adthetamh5_dustar = 0
        self.adthetamh5_dL = 0
        self.adthetamh5_dz0h = 0
        self.adthetamh6_dthetasurf = 0
        self.adthetamh6_dwtheta = 0
        self.adthetamh6_dustar = 0
        self.adthetamh6_dL = 0
        self.adthetamh6_dz0h = 0
        self.adthetamh7_dthetasurf = 0
        self.adthetamh7_dwtheta = 0
        self.adthetamh7_dustar = 0
        self.adthetamh7_dL = 0
        self.adthetamh7_dz0h = 0
        self.adzeta_dL_z0h = 0
        self.adCm = 0
        self.adCs = 0
        self.adCs_start = 0
        self.adustar_start = 0
        self.adCs_dzsl = 0
        self.adCs_dL = 0
        self.adpsimterm_for_dCs_dL = 0
        self.adpsihterm_for_dCs_dL = 0
        self.adzeta_dL_zsl = 0
        self.adzsl = 0
        self.adpsim_term_for_dCs_dzsl = 0
        self.adpsih_term_for_dCs_dzsl = 0
        self.adCm_dzsl = 0
        self.adCm_dL = 0
        self.adpsimterm_for_Cm_zsl = 0
        self.adpsimterm_for_Cm_z0m = 0
        self.adpsim_term_for_dCm_dzsl = 0
        self.adL = 0
        self.adL_dthetav = 0
        self.adL_dustar_start = 0
        self.adL_dwthetav = 0
        self.adwthetav = 0
        self.adthetav = 0
        self.adh = 0
        self.adthetavsurf = 0
        self.adq = 0
        self.adthetasurf_dCs_start = 0
        self.adthetasurf_dwtheta = 0
        self.adthetasurf_dtheta = 0
        self.adtheta = 0
        self.adCOS = 0
        self.adCOSsurf_dwCOS = 0
        self.adCOSsurf_dCOS = 0
        self.adCOSsurf_dCs_start = 0
        self.adueff = 0
        self.adthetasurf_dueff = 0
        self.adCOSsurf_dueff = 0
        self.adthetasurf = 0
        self.adTsurf = 0
        self.adCO2surf = 0
        self.adCO2surf_dwCO2 = 0
        self.adCO2surf_dCO2 = 0
        self.adCO2surf_dCs_start = 0
        self.adCO2surf_dueff = 0
        self.adCOSmh_dCOSmeasuring_height = 0
        self.adpsih_COSmh_L_num = 0
        self.adzeta_dCOSmh = 0
        self.adCOSmeasuring_height = 0
        self.adv2m_dz0m = 0
        self.adu2m_dz0m = 0
        self.adz0m = 0
        self.adpsimterm_for_dCm_dz0m = 0
        self.adCs_dz0m = 0
        self.adCm_dz0m = 0
        self.adCOSmh3_dz0h = 0
        self.adCOSmh2_dz0h = 0
        self.adCOSmh_dz0h = 0
        self.adpsihterm_for_dCs_dz0h = 0
        self.adCOS2m_dz0h = 0
        self.adCO22m_dz0h = 0
        self.adqmh_dz0h = 0
        self.adq2m_dz0h = 0
        self.adT2m_dz0h = 0
        self.adCs_dz0h = 0
        self.adz0h = 0
        self.adqmh2 = 0
        self.adqmh2_dqsurf = 0
        self.adqmh2_dz0h = 0
        self.adqmh2_dwq = 0
        self.adqmh2_dustar = 0
        self.adqmh2_dL = 0
        self.adqmh3 = 0
        self.adqmh3_dqsurf = 0
        self.adqmh3_dz0h = 0
        self.adqmh3_dwq = 0
        self.adqmh3_dustar = 0
        self.adqmh3_dL = 0
        self.adqmh4 = 0
        self.adqmh4_dqsurf = 0
        self.adqmh4_dz0h = 0
        self.adqmh4_dwq = 0
        self.adqmh4_dustar = 0
        self.adqmh4_dL = 0
        self.adqmh5 = 0
        self.adqmh5_dqsurf = 0
        self.adqmh5_dz0h = 0
        self.adqmh5_dwq = 0
        self.adqmh5_dustar = 0
        self.adqmh5_dL = 0
        self.adqmh6 = 0
        self.adqmh6_dqsurf = 0
        self.adqmh6_dz0h = 0
        self.adqmh6_dwq = 0
        self.adqmh6_dustar = 0
        self.adqmh6_dL = 0
        self.adqmh7 = 0
        self.adqmh7_dqsurf = 0
        self.adqmh7_dz0h = 0
        self.adqmh7_dwq = 0
        self.adqmh7_dustar = 0
        self.adqmh7_dL = 0
        if self.model.sw_use_ribtol:
            self.adRib_dthetav = 0
            self.adRib_dzsl = 0
            self.adRib_dthetavsurf = 0
            self.adRib_dueff = 0
        
        #ribtol
        if self.model.sw_use_ribtol:
            self.adL_new = 0
            self.adfx = 0
            self.adfxdif = 0
            self.adfxdif_part1 = 0
            self.adfxdif_part2 = 0
            self.adLstart = 0
            self.adLend = 0
            self.adfxdif_part2_dzsl = 0
            self.adfxdif_part2_dLend = 0
            self.adfxdif_part2_dz0h = 0
            self.adfxdif_part2_dz0m = 0
            self.adpsimterm_for_dfxdif_part2_dz0m = 0
            self.adpsihterm_for_dfxdif_part2_dz0h = 0
            self.adpsihterm1_for_dfxdif_part2_dLend = 0
            self.adpsihterm2_for_dfxdif_part2_dLend = 0
            self.adpsimterm1_for_dfxdif_part2_dLend = 0
            self.adpsimterm2_for_dfxdif_part2_dLend = 0
            self.adpsihterm_for_dfxdif_part2_dzsl = 0
            self.adpsimterm_for_dfxdif_part2_dzsl = 0
            self.adzeta_dLend_z0m = 0
            self.adzeta_dLend_zsl = 0
            self.adzeta_dLend_z0h = 0
            self.adfxdif_part1_dzsl = 0
            self.adfxdif_part1_dLstart = 0
            self.adfxdif_part1_dz0h = 0
            self.adfxdif_part1_dz0m = 0
            self.adpsimterm_for_dfxdif_part1_dz0m = 0
            self.adpsihterm_for_dfxdif_part1_dz0h = 0
            self.adpsihterm1_for_dfxdif_part1_dLstart = 0
            self.adpsihterm2_for_dfxdif_part1_dLstart = 0
            self.adpsimterm1_for_dfxdif_part1_dLstart = 0
            self.adpsimterm2_for_dfxdif_part1_dLstart = 0
            self.adpsihterm_for_dfxdif_part1_dzsl = 0
            self.adpsimterm_for_dfxdif_part1_dzsl = 0
            self.adzeta_dLstart_z0m = 0
            self.adzeta_dLstart_zsl = 0
            self.adzeta_dLstart_z0h = 0
            self.adfx_dRib = 0
            self.adfx_dzsl = 0
            self.adfx_dL = 0
            self.adfx_dz0m = 0
            self.adfx_dz0h = 0
            self.adpsihterm_for_dfx_dz0h = 0
            self.adpsimterm_for_dfx_dz0m = 0
            self.adpsihterm1_for_dfx_dL = 0
            self.adpsihterm2_for_dfx_dL = 0
            self.adpsimterm1_for_dfx_dL = 0
            self.adpsimterm2_for_dfx_dL = 0
            self.adpsihterm_for_dfx_dzsl = 0
            self.adpsimterm_for_dfx_dzsl = 0
            self.adL0 = 0
        self.adRib = 0
        
        #ags
        self.adwCO2, self.adwCO2A,  self.adwCO2R, self.adrs = 0,0,0,0
        self.adalfa_sto = 0
        self.adwCOS  = 0
        self.adwCOSP = 0
        self.adwCOSS = 0
        self.adwCOSS_molm2s = 0
        self.adTsoil = 0
        self.adthetasurf = 0
        self.adwg = 0
        self.adra = 0
        self.adSwin = 0
        self.adCO2 = 0
        self.adTs = 0
        self.adgcco2 = 0
        self.adPARfract = 0
        self.adR10 = 0
        self.adE0 = 0
        
        self.adgctCOS_dthetasurf = 0
        self.adgctCOS_dTs = 0
        self.adgctCOS_dCO2 = 0
        self.adgctCOS_dSwin = 0
        self.adgctCOS_dw2 = 0
        self.adgctCOS_dalfa_sto = 0
        self.adgctCOS_de = 0
        self.adgctCOS_dcveg = 0
        self.adgctCOS_dwfc = 0
        self.adgctCOS_dwwilt = 0
        self.adgctCOS_dLAI = 0
        self.adResp_dwg = 0
        self.adResp_dTsoil = 0
        self.adAn_dthetasurf = 0
        self.adAn_dTs = 0
        self.adAn_dCO2 = 0
        self.adAn_dra = 0
        self.adAn_dSwin = 0
        self.adAn_dw2 = 0
        self.adAn_dalfa_sto = 0
        self.adfw_dwg = 0
        self.adtexp_dTsoil = 0
        self.adrsCO2_dalfa_sto = 0
        self.adrsCO2_dw2 = 0
        self.adrsCO2_dSwin = 0
        self.adrsCO2_dCO2 = 0
        self.adci_dCO2 = 0
        self.adco2abs_dCO2 = 0
        self.adco2abs_dCO2surf = 0
        self.adrsCO2_dTs = 0
        self.adci_dTs = 0
        self.adrsCO2_dthetasurf = 0
        self.adci_dthetasurf = 0
        self.adgcco2_dalfa_sto = 0
        self.adgcco2_dw2 = 0
        self.adgcco2_dSwin = 0
        self.adgcco2_dCO2 = 0
        self.adgcco2_dTs = 0
        self.adgcco2_dthetasurf = 0
        self.adpart1_dw2 = 0
        self.adpart1_dSwin = 0
        self.adpart1_dCO2 = 0
        self.adpart1_dTs = 0
        self.adpart1_dthetasurf = 0
        self.adfstr_dw2 = 0
        self.adAn_temporary_dthetasurf = 0
        self.adAn_temporary_dSwin = 0
        self.adAn_temporary_dCO2 = 0
        self.adAn_temporary_dLAI = 0
        self.addiv2_dCO2 = 0
        self.adAn_temporary_dTs = 0
        self.addiv2_dTs = 0
        self.addiv2_dthetasurf = 0
        self.addiv1_dTs = 0
        self.addiv1_de = 0
        self.adCO2comp_dthetasurf = 0
        self.adDs_dTs = 0
        self.adD0_dthetasurf = 0
        self.ada11_dthetasurf = 0
        self.adsy_dSwin = 0
        self.adAmRdark_dCO2 = 0
        self.adsy_dCO2 = 0
        self.adAmRdark_dTs = 0
        self.adsy_dTs = 0
        self.adAmRdark_dthetasurf = 0
        self.adsy_dthetasurf = 0
        self.ady1_dSwin = 0
        self.ady_dSwin = 0
        self.ady1_dCO2 = 0
        self.ady_dCO2 = 0
        self.ady1_dTs = 0
        self.ady_dTs = 0
        self.ady1_dthetasurf = 0
        self.ady_dthetasurf = 0
        self.adPAR_dSwin = 0
        self.adalphac_dCO2 = 0
        self.adalphac_dthetasurf = 0
        self.adAg = 0
        self.adpexp_dthetasurf = 0
        self.adpexp_dTs = 0
        self.adpexp_dCO2 = 0
        self.adpexp_dSwin = 0
        self.adAm_dCO2 = 0
        self.adRdark_dCO2 = 0
        self.adAm_dTs = 0
        self.adRdark_dTs = 0
        self.adAm_dthetasurf = 0
        self.adRdark_dthetasurf = 0
        self.adxdiv_dCO2 = 0
        self.adxdiv_dthetasurf = 0
        self.adSwina_dSwin = 0
        self.adaexp_dCO2 = 0
        self.adaexp_dTs = 0
        self.adAmmax_dthetasurf = 0
        self.adaexp_dthetasurf = 0
        self.adgm_dthetasurf = 0
        self.adbetaw_dw2 = 0
        self.adbetaw_dwwilt = 0
        self.adbetaw_dwfc = 0
        self.adw2 = 0
        self.adAmmax1_dthetasurf = 0
        self.adAmmax2_dthetasurf = 0
        self.adAmmax3_dthetasurf = 0
        self.adcfrac_dTs = 0
        self.adcfrac_de = 0
        self.adevap = 0
        self.adfmin_dthetasurf = 0
        self.adfmin0_dthetasurf = 0
        self.adsqterm_dthetasurf = 0
        self.adsqrtf_dthetasurf = 0
        self.adgm1_dthetasurf = 0
        self.adgm2_dthetasurf = 0
        self.adgm3_dthetasurf = 0
        self.adgctCOS = 0
        self.adAn_de = 0
        self.adAn_dcveg = 0
        self.adAn_dwfc = 0
        self.adAn_dwwilt = 0
        self.adAn_dLAI = 0
        self.adrsCO2_dLAI = 0
        self.adrsCO2_dwwilt = 0
        self.adrsCO2_dwfc = 0
        self.adrsCO2_dcveg = 0
        self.adrsCO2_de = 0
        self.adci_de = 0
        self.adgcco2_dLAI = 0
        self.adgcco2_dwwilt = 0
        self.adgcco2_dwfc = 0
        self.adgcco2_dcveg = 0
        self.adgcco2_de = 0
        self.adLAI = 0
        self.adpart1_dLAI = 0
        self.adpart1_dwwilt = 0
        self.adpart1_dwfc = 0
        self.adpart1_dcveg = 0
        self.adpart1_de = 0
        self.adfstr_dwwilt = 0
        self.adfstr_dwfc = 0
        self.adAn_temporary_dcveg = 0
        self.adAn_temporary_de = 0
        self.addiv2_de = 0
        self.adDs_de = 0
        self.adsy_dLAI = 0
        self.adsy_dcveg = 0
        self.adAmRdark_de = 0
        self.adsy_de = 0
        self.ady1_dLAI = 0
        self.ady1_dcveg = 0
        self.ady_dcveg = 0
        self.ady1_de = 0
        self.ady_de = 0
        self.adPAR_dcveg = 0
        self.adpexp_de = 0
        self.adpexp_dcveg = 0
        self.adAm_de = 0
        self.adRdark_de = 0
        self.adSwina_dcveg = 0
        self.adcveg = 0
        self.adaexp_de = 0
        self.adwwilt = 0
        self.adwfc = 0
        self.ade = 0
        self.adco2abs = 0
        self.adgctCOS_dgciCOS = 0
        self.adgciCOS = 0
        self.adrsCO2_dPARfract = 0
        self.adAn_dPARfract = 0
        self.adgcco2_dPARfract = 0
        self.adgctCOS_dPARfract = 0
        self.adpart1_dPARfract = 0
        self.adAn_temporary_dPARfract = 0
        self.adsy_dPARfract = 0
        self.ady1_dPARfract = 0
        self.ady_dPARfract = 0
        self.adPAR_dPARfract = 0
        self.adpexp_dPARfract = 0
        self.adResp_dR10 = 0
        self.adResp_dE0 = 0
        self.adtexp_dE0 = 0
        
        #run mixed layer
        self.addeltatheta = 0
        self.addeltathetav = 0
        self.addeltaq = 0
        self.addeltaCO2 = 0
        self.addeltaCOS = 0
        self.addeltau = 0
        self.addeltav = 0
        self.adM = 0
        self.adadvtheta = 0
        self.adwqM = 0
        self.adadvq = 0
        self.adwCO2M = 0
        self.adadvCO2 = 0
        self.adwCOSM = 0
        self.adadvCOS = 0
        self.adlcl = 0
        self.addztend = 0
        self.adwe = 0
        self.addeltaCOStend = 0
        self.adwf = 0
        self.adCOStend = 0
        self.adw_COS_ft = 0
        self.addeltaCO2tend = 0
        self.adCO2tend = 0
        self.adw_CO2_ft = 0
        self.addeltaqtend = 0
        self.adqtend = 0
        self.adw_q_ft = 0
        self.addeltathetatend = 0
        self.addeltautend = 0
        self.addeltavtend = 0
        self.adutend = 0
        self.advtend = 0
        self.adadvu = 0
        self.adadvv = 0
        self.adu = 0
        self.adv = 0
        self.adfc = 0
        self.adthetatend = 0
        self.adw_th_ft = 0
        self.adwCOSe = 0
        self.adwCO2e = 0
        self.adwqe = 0
        self.adwthetae = 0
        self.adhtend = 0
        self.adwe_dwthetav = 0
        self.adwe_dustar = 0
        self.adwe_dthetav = 0
        self.adwe_dh = 0
        self.adwe_ddeltathetav = 0
        self.adwthetave = 0
        self.adwstar_dh = 0
        self.adwstar_dwthetav = 0
        self.adwstar_dthetav = 0
        self.adwstar = 0
        self.adws = 0
        self.adgammatheta = 0
        self.adgammatheta2 = 0
        self.adgammaq = 0
        self.adgammaCO2 = 0
        self.adgammaCOS = 0
        self.adgammau = 0
        self.adgammav = 0
        self.addivU = 0
        self.addFz = 0
        self.adbeta = 0
        self.advw_dustar = 0
        self.advw_du = 0
        self.advw_dv = 0
        self.aduw_dustar = 0
        self.aduw_du = 0
        self.aduw_dv = 0
        
        #integrate mixed layer
        self.addz_h = 0
        self.addeltaCO20 = 0
        self.addeltaCOS0 = 0
        self.adCOS0 = 0
        self.adCO20 = 0
        self.addeltaq0 = 0
        self.adq0 = 0
        self.addeltatheta0 = 0
        self.adtheta0 = 0
        self.adh0 = 0
        self.addeltav0 = 0
        self.addeltau0 = 0
        self.adv0 = 0
        self.adu0 = 0
        
        #run radiation
        self.addoy = 0
        self.adlat = 0
        self.adlon = 0
        self.adpart1_sinlea = 0
        self.adsinlea_lon = 0
        self.adpart2_sinlea = 0
        self.adsinlea = 0
        self.adTa_dtheta = 0
        self.adTa_dh = 0
        self.adTa = 0
        self.adTr = 0
        self.adcc = 0
        self.adSwin = 0
        self.adSwout = 0
        self.adLwin = 0
        self.adLwout = 0
        self.adQ = 0
        self.adalpha = 0
        self.adsda = 0
        
        #run land surface
        self.adLE = 0
        self.adH = 0
        self.adwgtend_dLEsoil = 0
        self.adwgtend = 0
        self.adwgtend_dC1 = 0
        self.adwgtend_dC2 = 0
        self.adwgtend_dwg = 0
        self.adwgtend_dwgeq = 0
        self.adwgeq = 0
        self.adC2 = 0
        self.adC1 = 0
        self.adLEsoil = 0
        self.adwgeq_dw2 = 0
        self.adwgeq_dwsat = 0
        self.adwsat = 0
        self.adC2_dw2 = 0
        self.adC2_dwsat = 0
        self.adTsoiltend_dCG = 0
        self.adTsoiltend = 0
        self.adTsoiltend_dG = 0
        self.adTsoiltend_dTsoil = 0
        self.adTsoiltend_dT2 = 0
        self.adT2 = 0
        self.adG = 0
        self.adCG = 0
        self.adnumerator_LEref = 0
        self.adLEref = 0
        self.addenominator_LEref = 0
        self.addqsatdT_dtheta = 0
        self.adrsmin = 0
        self.adp1_numerator_LEref = 0
        self.adp2_numerator_LEref = 0
        self.adqsatvar = 0
        self.adnumerator_LEpot = 0
        self.adLEpot = 0
        self.adp1_numerator_LEpot = 0
        self.adp2_numerator_LEpot = 0
        self.adLambda = 0
        self.adLEveg = 0
        self.adLEliq = 0
        self.adWltend = 0
        self.adp1_LEsoil = 0
        self.adp2_LEveg_liq_soil = 0
        self.adrssoil = 0
        self.adp1_LEliq = 0
        self.adcliq = 0
        self.adp1_LEveg = 0
        self.adesatsurf = 0
        self.adnumerator_Ts = 0
        self.addenominator_Ts = 0
        self.adp1_denominator_Ts = 0
        self.adp2_denominator_Ts = 0
        self.adp3_denominator_Ts = 0
        self.adp4_denominator_Ts = 0
        self.adp1_numerator_Ts = 0
        self.adp2_numerator_Ts = 0
        self.adp3_numerator_Ts = 0
        self.adp4_numerator_Ts = 0
        self.adp5_numerator_Ts = 0
        self.adWl = 0
        self.adWlmx = 0
        self.adWmax = 0
        self.adf2 = 0
        self.addesatdT_dtheta = 0
        self.adesatvar = 0
        self.adwstar = 0
        self.adueff = 0
        self.adqsatsurf = 0
        self.adrssoilmin = 0
        self.adwgeq_da = 0
        self.adwgeq_dp = 0
        self.adp = 0
        self.ada = 0
        self.adC2_dC2ref = 0
        self.adC2ref = 0
        self.adC1_dC1sat = 0
        self.adC1_dwsat = 0
        self.adC1_dwg = 0
        self.adC1_db = 0
        self.adb = 0
        self.adC1sat = 0
        self.adCG_dCGsat = 0
        self.adCG_dwsat = 0
        self.adCG_dw2 = 0
        self.adCG_db = 0
        self.adCGsat = 0
        
        #integrate land surface
        #some vars needed by this have already been set in another module
        self.adWl0 = 0
        self.adwg0 = 0
        self.adTsoil0 = 0
        
        #statistics
        self.adRH_h = 0
        self.adqsat_variable = 0
        self.adqsat_variable_dT_H = 0
        self.adqsat_variable_dP_H = 0
        self.adP_h = 0
        self.adT_h = 0
        self.adlcl = 0
        self.adRHlcl = 0
        self.adqsat_variable_dp_lcl = 0
        self.adqsat_variable_dT_lcl = 0
        self.adT_lcl = 0
        self.adp_lcl = 0
        self.adlcl_new = 0
        
        #run_soil_COS_mod       
        self.adC_soilair_next = np.zeros(self.model.input.nr_nodes)
        self.adC_soilair_current = np.zeros(self.model.input.nr_nodes)
        self.adOCS_fluxes = np.zeros(self.model.input.nr_nodes)
        self.adCOS_netuptake_soilsun = 0
        self.adconduct = np.zeros(self.model.input.nr_nodes)
        self.adC_soilair = np.zeros(self.model.input.nr_nodes)
        self.adinvmatreq12 = np.zeros((self.model.input.nr_nodes,self.model.input.nr_nodes))
        self.admatr_3_eq12 = np.zeros(self.model.input.nr_nodes)
        self.admatr_2_eq12 = np.zeros(self.model.input.nr_nodes)
        self.adsource = np.zeros(self.model.input.nr_nodes)
        self.adA_matr = np.zeros((self.model.input.nr_nodes,self.model.input.nr_nodes))
        self.adB_matr = np.zeros((self.model.input.nr_nodes,self.model.input.nr_nodes))
        self.adeta = np.zeros(self.model.input.nr_nodes)
        self.adkH = np.zeros(self.model.input.nr_nodes)
        self.ads_moist = np.zeros(self.model.input.nr_nodes)
        self.ads_uptake = np.zeros(self.model.input.nr_nodes)
        self.ads_prod = np.zeros(self.model.input.nr_nodes)
        self.adD_a_0 = 0
        self.addiffus = np.zeros(self.model.input.nr_nodes)
        self.addiffus_nodes = np.zeros(self.model.input.nr_nodes)
        self.adD_a = 0
        self.addiffus_nodes_dwsat = np.zeros(self.model.input.nr_nodes)
        self.addiffus_nodes_ds_moist = np.zeros(self.model.input.nr_nodes)
        self.addiffus_nodes_db_sCOSm = np.zeros(self.model.input.nr_nodes)
        self.addiffus_nodes_dT_nodes = np.zeros(self.model.input.nr_nodes)
        self.adT_nodes = np.zeros(self.model.input.nr_nodes)
        self.adb_sCOSm = 0
        self.adVspmax = 0
        self.adQ10 = 0
        self.adktot = np.zeros(self.model.input.nr_nodes)
        self.adfCA = 0
        self.adxCA = np.zeros(self.model.input.nr_nodes)
        self.admol_rat_ocs_atm = 0
        self.adairtemp = 0
        self.adC_air = 0
        
        #run_cumulus
        self.adCOS2_h = 0
        self.adCOS2_h_dwstar = 0
        self.adCOS2_h_ddz_h = 0
        self.adCOS2_h_dh = 0
        self.adCOS2_h_ddeltaCOS = 0
        self.adCOS2_h_dwCOSM = 0
        self.adCOS2_h_dwCOSe = 0
        self.adCO22_h = 0
        self.adCO22_h_dwstar = 0
        self.adCO22_h_ddz_h = 0
        self.adCO22_h_dh = 0
        self.adCO22_h_ddeltaCO2 = 0
        self.adCO22_h_dwCO2M = 0
        self.adCO22_h_dwCO2e = 0
        self.adq2_h = 0
        self.adq2_h_dwstar = 0
        self.adq2_h_ddz_h = 0
        self.adq2_h_dh = 0
        self.adq2_h_ddeltaq = 0
        self.adq2_h_dwqM = 0
        self.adq2_h_dwqe = 0
        self.adqsat_variable_rc_dT_h = 0
        self.adqsat_variable_rc_dP_h = 0
        self.adac = 0
        self.adac_dq = 0
        self.adac_dT_h = 0
        self.adac_dP_h = 0
        self.adac_dq2_h = 0
        self.adM = 0
        self.adwqM = 0
        self.adwCO2M = 0
        self.adwCOSM = 0 
        
        #init_soil_COS_mod
        self.adC_air_init = 0
        
        #store
        self.adout_dz = 0
        self.adout_M = 0
        self.adout_ac = 0
        self.adout_RH_h = 0
        self.adout_zlcl = 0
        self.adout_Ts = 0
        self.adout_G = 0
        self.adout_LEref = 0
        self.adout_LEpot = 0
        self.adout_LEsoil = 0
        self.adout_LEveg = 0
        self.adout_LEliq = 0
        self.adout_LE = 0
        self.adout_H = 0
        self.adout_rs = 0
        self.adout_ra = 0
        self.adout_Q = 0
        self.adout_Lwout = 0
        self.adout_Lwin = 0
        self.adout_Swout = 0
        self.adout_Swin = 0
        self.adout_Cs = 0
        self.adout_ustar = 0
        self.adout_qsurf = 0
        self.adout_thetavsurf = 0
        self.adout_thetasurf = 0
        self.adout_Tsurf = 0
        self.adout_CO2surf = 0
        self.adout_COSsurf = 0
        self.adout_COS2m = 0
        self.adout_CO2mh4 = 0
        self.adout_CO2mh3 = 0
        self.adout_CO2mh2 = 0
        self.adout_CO2mh = 0
        self.adout_CO22m = 0
        self.adout_COSmh3 = 0
        self.adout_COSmh2 = 0
        self.adout_COSmh = 0
        self.adout_esat2m = 0
        self.adout_e2m = 0
        self.adout_v2m = 0
        self.adout_u2m = 0
        self.adout_qmh = 0
        self.adout_qmh2 = 0
        self.adout_qmh3 = 0
        self.adout_qmh4 = 0
        self.adout_qmh5 = 0
        self.adout_qmh6 = 0
        self.adout_qmh7 = 0
        self.adout_q2m = 0
        self.adout_thetamh = 0
        self.adout_thetamh2 = 0
        self.adout_thetamh3 = 0
        self.adout_thetamh4 = 0
        self.adout_thetamh5 = 0
        self.adout_thetamh6 = 0
        self.adout_thetamh7 = 0
        self.adout_Tmh = 0
        self.adout_Tmh2 = 0
        self.adout_Tmh3 = 0
        self.adout_Tmh4 = 0
        self.adout_Tmh5 = 0
        self.adout_Tmh6 = 0
        self.adout_Tmh7 = 0
        self.adout_T2m = 0
        self.adout_vw = 0
        self.adout_uw = 0
        self.adout_u = 0
        self.adout_v = 0
        self.adout_deltaCOS = 0
        self.adout_deltau = 0
        self.adout_deltav = 0
        self.adout_COS = 0
        self.adout_wCO2A = 0
        self.adout_wCO2R = 0
        self.adout_wCO2e = 0
        self.adout_wCO2M = 0
        self.adout_wCO2 = 0
        self.adout_deltaCO2 = 0
        self.adout_CO2 = 0
        self.adout_esatvar = 0
        self.adout_e = 0
        self.adout_qsatvar = 0
        self.adout_wqM = 0
        self.adout_wqe = 0
        self.adout_wq = 0
        self.adout_deltaq = 0
        self.adout_q = 0
        self.adout_wthetave = 0
        self.adout_wthetae = 0
        self.adout_wthetav = 0
        self.adout_wtheta = 0
        self.adout_deltathetav = 0
        self.adout_deltatheta = 0
        self.adout_thetav = 0
        self.adout_theta = 0
        self.adout_h = 0
        self.adout_wCOS = 0
        self.adout_wCOSP = 0
        self.adout_wCOSS = 0
        self.adout_Cm = 0
        self.adout_Rib = 0
        self.adout_L = 0
        
        #jarvis_stewart
        self.adrs_drsmin = 0
        self.adrs_dLAI = 0
        self.adrs_df1 = 0
        self.adrs_df3 = 0
        self.adrs_df4 = 0
        self.adf4 = 0
        self.adf3 = 0
        self.adgD = 0
        self.adf1 = 0
        self.adrs_df2js = 0
        self.adf2js_dwfc = 0
        self.adf2js_dwwilt = 0
        self.adf2js_dw2 = 0
        self.adf2js = 0
        
        #timestep (in tangent linear it is part of tl_full_model)
        self.adwtheta_input = np.zeros(self.model.tsteps) #this would crash if the model has not yet been ran
        self.adwq_input = np.zeros(self.model.tsteps)
        self.adwCO2_input = np.zeros(self.model.tsteps)
        self.adwCOS_input = np.zeros(self.model.tsteps)
        
    def adjoint(self,forcing,checkpoint,checkpoint_init,model,returnvariables=None): #the adjoint of one timestep
        for i in range(model.tsteps-1,-1,-1):
            for item in forcing[i]:#
                if self.adjointtesting: #In the adjoint test, we test way more variables than what is normally considered to be a possible forcing variable (a variable from the store module). 
                    #therefore, we need a seperate line for the case we run an adjoint test
                    self.__dict__[item] += forcing[i][item] 
                else:
                    self.__dict__['adout_'+item] += forcing[i][item] #takes the storing into account by looking at the adout variables
            #the forcing in the argument list of the adjoint parts functions is not used, the statement avove takes care of the forcings
            if(model.sw_ml):
                self.adj_integrate_mixed_layer(forcing[i],checkpoint[i],model)
            if(model.sw_ls):
                self.adj_integrate_land_surface(forcing[i],checkpoint[i],model)
            self.adj_store(forcing[i],checkpoint[i],model)
            if(model.sw_ml):
                self.adj_run_mixed_layer(forcing[i],checkpoint[i],model)
            if model.sw_cu:
                self.adj_run_cumulus(forcing[i],checkpoint[i],model)
            if(model.sw_ls):
                self.adj_run_land_surface(forcing[i],checkpoint[i],model)
            if(model.sw_sl):
                self.adj_run_surface_layer(forcing[i],checkpoint[i],model)
            if(model.sw_rad):
                self.adj_run_radiation(forcing[i],checkpoint[i],model)
            self.adj_statistics(forcing[i],checkpoint[i],model)
            if not model.sw_ls: #
                if hasattr(model.input,'wCOS_input'):
                    #statement self.dwCOS = self.dwCOS_input[i]
                    self.adwCOS_input[i] += self.adwCOS
                    self.adwCOS = 0
                if hasattr(model.input,'wCO2_input'):
                    #statement self.dwCO2 = self.dwCO2_input[i]
                    self.adwCO2_input[i] += self.adwCO2
                    self.adwCO2 = 0
                if hasattr(model.input,'wq_input'):
                    #statement self.dwq = self.dwq_input[i]
                    self.adwq_input[i] += self.adwq
                    self.adwq = 0
                if hasattr(model.input,'wtheta_input'):
                    #statement self.dwtheta = self.dwtheta_input[i]
                    self.adwtheta_input[i] += self.adwtheta
                    self.adwtheta = 0
        self.adj_init(checkpoint_init,model)            
        returnlist = []
        if returnvariables!=None:
            for item in returnvariables:
                returnlist.append(self.__dict__[item])
            return returnlist
        
    def adj_init(self,checkpoint_init,model,returnvariables=None): #this is the adjoint of the initialiation of the model run (in forwardmodel initialisation is called once in run)
        forcing={} #just a dummy
        if(model.sw_ml):
            #statement self.tl_run_mixed_layer(model,checkpoint)
            self.adj_run_mixed_layer(forcing,checkpoint_init[0],model)
        if(model.sw_cu):
            #statement self.tl_run_cumulus(model,checkpoint)
            self.adj_run_cumulus(forcing,checkpoint_init[0],model)
            #statement self.tl_run_mixed_layer(model,checkpoint)
            self.adj_run_mixed_layer(forcing,checkpoint_init[0],model)
        if(model.sw_ls):
            #statement self.tl_run_land_surface(model,checkpoint)
            self.adj_run_land_surface(forcing,checkpoint_init[0],model)
            if self.model.input.soilCOSmodeltype == 'Sun_Ogee':
                self.adj_init_soil_COS_mod(checkpoint_init[0],model)
        if(model.sw_sl):
            for i in range(model.nr_of_surf_lay_its-1,-1,-1): #model.nr_of_surf_lay_its-1 because index model.nr_of_surf_lay_its does not exist
                #statement self.tl_run_surface_layer(model,checkpoint)
                self.adj_run_surface_layer(forcing,checkpoint_init[i],model)
        if(model.sw_rad):
            #statement self.tl_run_radiation(model,checkpoint)
            self.adj_run_radiation(forcing,checkpoint_init[0],model)
        #statement self.tl_statistics(model,checkpoint)
        self.adj_statistics(forcing,checkpoint_init[0],model)
        fac = model.mair / (model.rho*model.mco2)
        if not model.sw_ls:
            if hasattr(model.input,'wCO2_input'):
                #statement self.dwCO2_input = self.dwCO2_input * fac
                self.adwCO2_input *= fac
        #statement self.dwCO2 = fac * self.dwCO2
        self.adwCO2 *= fac
        if not hasattr(model.input,'gammatheta2'):
            #statement self.dgammatheta2 = self.dgammatheta
            self.adgammatheta += self.adgammatheta2
            self.adgammatheta2 = 0
        returnlist = []
        if returnvariables!=None:
            for item in returnvariables:
                returnlist.append(self.__dict__[item])
            return returnlist
    
    def adj_init_soil_COS_mod(self,checkpoint_init,model,returnvariables=None): #this is the adjoint belonging to the initialisation of the soil_COS_mod object
        airtemp = checkpoint_init['isCm_airtemp']
        Rgas = checkpoint_init['isCm_Rgas_end']
        pressure = checkpoint_init['isCm_pressure']
        mol_rat_ocs_atm = checkpoint_init['isCm_mol_rat_ocs_atm']
        for i in range(model.soilCOSmodel.nr_nodes-1,-1,-1):
            #statement dC_soilair_current[i] = dC_air_init
            self.adC_air_init += self.adC_soilair_current[i]
            self.adC_soilair_current[i] = 0
        #statement dC_air_init = 1.e-9 * pressure / Rgas * (dmol_rat_ocs_atm / airtemp + mol_rat_ocs_atm * -1 * airtemp**(-2) * dairtemp)
        self.admol_rat_ocs_atm += 1.e-9 * pressure / Rgas / airtemp * self.adC_air_init
        self.adairtemp += 1.e-9 * pressure / Rgas * mol_rat_ocs_atm * -1 * airtemp**(-2) * self.adC_air_init
        self.adC_air_init = 0
        #statement dairtemp = self.dTsurf
        self.adTsurf += self.adairtemp
        self.adairtemp = 0
        #statement dmol_rat_ocs_atm = self.dCOSsurf 
        self.adCOSsurf += self.admol_rat_ocs_atm
        self.admol_rat_ocs_atm = 0
        
        returnlist = []
        if returnvariables!=None:
            for item in returnvariables:
                returnlist.append(self.__dict__[item])
            return returnlist
    
    def adj_integrate_mixed_layer(self,forcing,checkpoint,model,HTy_variables=None):
        dz0 = 50 #this is just a constant
        dz_h = checkpoint['iml_dz_h_middle']
        if(model.sw_wind):
            #statement ddeltav   = ddeltav0     + model.dt * self.ddeltavtend
            self.addeltav0 += self.addeltav
            self.addeltavtend += model.dt * self.addeltav
            self.addeltav = 0
            #statement dv        = dv0      + model.dt * self.dvtend
            self.adv0 += self.adv
            self.advtend += model.dt * self.adv
            self.adv = 0
            #statement ddeltau   = ddeltau0     + model.dt * self.ddeltautend
            self.addeltau0 += self.addeltau
            self.addeltautend += model.dt * self.addeltau
            self.addeltau = 0
            #statement du        = du0      + model.dt * self.dutend
            self.adu0 += self.adu
            self.adutend += model.dt * self.adu
            self.adu = 0
        if(dz_h < dz0):
            #statement ddz_h = 0
            self.addz_h = 0
        #statement ddz0 = 0
        self.addz0 = 0
        #statement ddz_h     = ddz0     + model.dt * self.ddztend
        self.addz0 += self.addz_h
        self.addztend += model.dt * self.addz_h
        self.addz_h = 0
        #statement ddeltaCOS     = ddeltaCOS0   + model.dt * self.ddeltaCOStend
        self.addeltaCOS0 += self.addeltaCOS
        self.addeltaCOStend += model.dt * self.addeltaCOS
        self.addeltaCOS = 0
        #statement ddeltaCO2     = ddeltaCO20   + model.dt * self.ddeltaCO2tend
        self.addeltaCO20 += self.addeltaCO2
        self.addeltaCO2tend += model.dt * self.addeltaCO2
        self.addeltaCO2 = 0
        #statement dCOS      = dCOS0    + model.dt * self.dCOStend
        self.adCOS0 += self.adCOS
        self.adCOStend += model.dt * self.adCOS
        self.adCOS = 0
        #statement dCO2      = dCO20    + model.dt * self.dCO2tend
        self.adCO20 += self.adCO2
        self.adCO2tend += model.dt * self.adCO2
        self.adCO2 = 0
        #statement ddeltaq       = ddeltaq0     + model.dt * self.ddeltaqtend
        self.addeltaq0 += self.addeltaq
        self.addeltaqtend += model.dt * self.addeltaq
        self.addeltaq = 0
        #statement dq        = dq0      + model.dt * self.dqtend
        self.adq0 += self.adq
        self.adqtend += model.dt * self.adq
        self.adq = 0
        #statement ddeltatheta   = ddeltatheta0 + model.dt * self.ddeltathetatend
        self.addeltatheta0 += self.addeltatheta
        self.addeltathetatend += model.dt * self.addeltatheta
        self.addeltatheta = 0
        #statement dtheta    = dtheta0  + model.dt * self.dthetatend
        self.adtheta0 += self.adtheta
        self.adthetatend += model.dt * self.adtheta
        self.adtheta = 0
        #statement dh        = dh0      + model.dt * self.dhtend
        self.adh0 += self.adh
        self.adhtend += model.dt * self.adh
        self.adh = 0
        #statement ddz0     = self.ddz_h
        self.addz_h += self.addz0
        self.addz0 = 0
        #statement ddeltav0 = self.ddeltav
        self.addeltav += self.addeltav0
        self.addeltav0 = 0
        #statement dv0      = self.dv
        self.adv += self.adv0
        self.adv0 = 0
        #statement ddeltau0 = self.ddeltau
        self.addeltau += self.addeltau0
        self.addeltau0 = 0
        #statement du0      = self.du
        self.adu += self.adu0
        self.adu0 = 0
        #statement ddeltaCOS0   = self.ddeltaCOS
        self.addeltaCOS += self.addeltaCOS0
        self.addeltaCOS0 = 0
        #statement ddeltaCO20   = self.ddeltaCO2
        self.addeltaCO2 += self.addeltaCO20
        self.addeltaCO20 = 0
        #statement dCOS0    = self.dCOS
        self.adCOS += self.adCOS0
        self.adCOS0 = 0
        #statement dCO20    = self.dCO2
        self.adCO2 += self.adCO20
        self.adCO20 = 0
        #statement ddeltaq0     = self.ddeltaq
        self.addeltaq += self.addeltaq0
        self.addeltaq0 = 0
        #statement dq0      = self.dq
        self.adq += self.adq0
        self.adq0 = 0
        #statement ddeltatheta0 = self.ddeltatheta
        self.addeltatheta += self.addeltatheta0
        self.addeltatheta0 = 0
        #statement dtheta0  = self.dtheta
        self.adtheta += self.adtheta0
        self.adtheta0 = 0
        #statement dh0 = self.dh
        self.adh += self.adh0
        self.adh0 = 0
        if self.adjointtestingint_mixed_layer:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    
    def adj_integrate_land_surface(self,forcing,checkpoint,model,HTy_variables=None):
        #statement dWl       = dWl0     + model.dt * self.dWltend
        self.adWl0 += self.adWl
        self.adWltend += model.dt * self.adWl
        self.adWl = 0
        #statement dwg       = dwg0     + model.dt * self.dwgtend
        self.adwg0 += self.adwg
        self.adwgtend += model.dt * self.adwg
        self.adwg = 0
        #statement dTsoil    = dTsoil0  + model.dt * self.dTsoiltend
        self.adTsoil0 += self.adTsoil
        self.adTsoiltend += model.dt * self.adTsoil
        self.adTsoil = 0
        #statement dWl0           = self.dWl
        self.adWl += self.adWl0
        self.adWl0 = 0
        #statement dwg0           = self.dwg
        self.adwg += self.adwg0
        self.adwg0 = 0
        #statement dTsoil0        = self.dTsoil
        self.adTsoil += self.adTsoil0
        self.adTsoil0 = 0
        
        if self.adjointtestingint_land_surface:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    def adj_store(self,forcing,checkpoint,model,HTy_variables=None):
        fac = checkpoint['sto_fac_end']
        #statement dout_dz            = self.ddz_h
        self.addz_h += self.adout_dz
        self.adout_dz = 0
        #statement dout_M             = self.dM
        self.adM += self.adout_M
        self.adout_M = 0
        #statement dout_ac            = self.dac
        self.adac += self.adout_ac
        self.adout_ac = 0
        #statement dout_RH_h          = self.dRH_h
        self.adRH_h += self.adout_RH_h
        self.adout_RH_h  = 0
        #statement dout_zlcl          = self.dlcl
        self.adlcl += self.adout_zlcl
        self.adout_zlcl = 0
        #statement dout_Ts            = self.dTs
        self.adTs += self.adout_Ts
        self.adout_Ts = 0
        #statement dout_G             = self.dG
        self.adG += self.adout_G 
        self.adout_G  = 0
        #statement dout_LEref         = self.dLEref
        self.adLEref += self.adout_LEref
        self.adout_LEref = 0
        #statement dout_LEpot         = self.dLEpot
        self.adLEpot += self.adout_LEpot 
        self.adout_LEpot = 0
        #statement dout_LEsoil        = self.dLEsoil
        self.adLEsoil += self.adout_LEsoil
        self.adout_LEsoil = 0
        #statement dout_LEveg         = self.dLEveg
        self.adLEveg += self.adout_LEveg
        self.adout_LEveg = 0
        #statement dout_LEliq         = self.dLEliq
        self.adLEliq += self.adout_LEliq 
        self.adout_LEliq = 0
        #statement dout_LE            = self.dLE
        self.adLE += self.adout_LE 
        self.adout_LE = 0
        #statement dout_H             = self.dH
        self.adH += self.adout_H   
        self.adout_H = 0
        #statement dout_rs            = self.drs
        self.adrs += self.adout_rs
        self.adout_rs = 0
        #statement dout_ra            = self.dra
        self.adra += self.adout_ra
        self.adout_ra = 0
        #statement dout_Q             = self.dQ
        self.adQ += self.adout_Q
        self.adout_Q = 0
        #statement dout_Lwout         = self.dLwout
        self.adLwout += self.adout_Lwout
        self.adout_Lwout = 0
        #statement dout_Lwin          = self.dLwin
        self.adLwin += self.adout_Lwin
        self.adout_Lwin = 0
        #statement dout_Swout         = self.dSwout
        self.adSwout += self.adout_Swout
        self.adout_Swout = 0
        #statement dout_Swin          = self.dSwin
        self.adSwin += self.adout_Swin
        self.adout_Swin = 0
        #statement dout_Rib           = self.dRib
        self.adRib += self.adout_Rib
        self.adout_Rib = 0
        #statement dout_L             = self.dL
        self.adL += self.adout_L
        self.adout_L = 0
        #statement dout_Cs            = self.dCs
        self.adCs += self.adout_Cs 
        self.adout_Cs = 0
        #statement dout_ustar         = self.dustar
        self.adustar += self.adout_ustar
        self.adout_ustar = 0
        #statement dout_qsurf         = self.dqsurf
        self.adqsurf += self.adout_qsurf 
        self.adout_qsurf = 0
        #statement dout_thetavsurf    = self.dthetavsurf
        self.adthetavsurf += self.adout_thetavsurf 
        self.adout_thetavsurf = 0
        #statement dout_thetasurf     = self.dthetasurf
        self.adthetasurf += self.adout_thetasurf
        self.adout_thetasurf = 0
        if model.sw_sl:
            #statement dout_Cm         = self.dCm
            self.adCm += self.adout_Cm
            self.adout_Cm = 0
            #statement dout_Tsurf         = self.dTsurf
            self.adTsurf += self.adout_Tsurf
            self.adout_Tsurf = 0
            #statement dout_CO2surf       = self.dCO2surf
            self.adCO2surf += self.adout_CO2surf
            self.adout_CO2surf = 0
            #statement dout_COSsurf       = self.dCOSsurf
            self.adCOSsurf += self.adout_COSsurf  
            self.adout_COSsurf = 0
            #statement dout_COS2m         = self.dCOS2m
            self.adCOS2m += self.adout_COS2m 
            self.adout_COS2m = 0
            #statement dout_CO2mh4        = self.dCO2mh4
            self.adCO2mh4 += self.adout_CO2mh4 
            self.adout_CO2mh4 = 0
            #statement dout_CO2mh3        = self.dCO2mh3
            self.adCO2mh3 += self.adout_CO2mh3 
            self.adout_CO2mh3 = 0
            #statement dout_CO2mh2        = self.dCO2mh2
            self.adCO2mh2 += self.adout_CO2mh2 
            self.adout_CO2mh2 = 0
            #statement dout_CO2mh         = self.dCO2mh
            self.adCO2mh += self.adout_CO2mh
            self.adout_CO2mh = 0
            #statement dout_CO22m         = self.dCO22m
            self.adCO22m += self.adout_CO22m
            self.adout_CO22m = 0
            #statement dout_COSmh3        = self.dCOSmh3
            self.adCOSmh3 += self.adout_COSmh3
            self.adout_COSmh3 = 0
            #statement dout_COSmh2        = self.dCOSmh2
            self.adCOSmh2 += self.adout_COSmh2  
            self.adout_COSmh2 = 0
            #statement dout_COSmh         = self.dCOSmh
            self.adCOSmh += self.adout_COSmh
            self.adout_COSmh = 0
            #statement dout_qmh7           = self.dqmh7
            self.adqmh7 += self.adout_qmh7 
            self.adout_qmh7 = 0
            #statement dout_qmh6           = self.dqmh6
            self.adqmh6 += self.adout_qmh6 
            self.adout_qmh6 = 0
            #statement dout_qmh5           = self.dqmh5
            self.adqmh5 += self.adout_qmh5 
            self.adout_qmh5 = 0
            #statement dout_qmh4           = self.dqmh4
            self.adqmh4 += self.adout_qmh4 
            self.adout_qmh4 = 0
            #statement dout_qmh3           = self.dqmh3
            self.adqmh3 += self.adout_qmh3 
            self.adout_qmh3 = 0
            #statement dout_qmh2           = self.dqmh2
            self.adqmh2 += self.adout_qmh2 
            self.adout_qmh2 = 0
            #statement dout_qmh           = self.dqmh
            self.adqmh += self.adout_qmh 
            self.adout_qmh = 0
            #statement dout_Tmh7           = self.dTmh7
            self.adTmh7 += self.adout_Tmh7  
            self.adout_Tmh7 = 0
            #statement dout_Tmh6           = self.dTmh6
            self.adTmh6 += self.adout_Tmh6  
            self.adout_Tmh6 = 0
            #statement dout_Tmh5           = self.dTmh5
            self.adTmh5 += self.adout_Tmh5  
            self.adout_Tmh5 = 0
            #statement dout_Tmh4           = self.dTmh4
            self.adTmh4 += self.adout_Tmh4  
            self.adout_Tmh4 = 0
            #statement dout_Tmh3           = self.dTmh3
            self.adTmh3 += self.adout_Tmh3  
            self.adout_Tmh3 = 0
            #statement dout_Tmh2           = self.dTmh2
            self.adTmh2 += self.adout_Tmh2  
            self.adout_Tmh2 = 0
            #statement dout_Tmh           = self.dTmh
            self.adTmh += self.adout_Tmh  
            self.adout_Tmh = 0
            #statement dout_thetamh7       = self.dthetamh7
            self.adthetamh7 += self.adout_thetamh7  
            self.adout_thetamh7 = 0
            #statement dout_thetamh6       = self.dthetamh6
            self.adthetamh6 += self.adout_thetamh6  
            self.adout_thetamh6 = 0
            #statement dout_thetamh5       = self.dthetamh5
            self.adthetamh5 += self.adout_thetamh5  
            self.adout_thetamh5 = 0
            #statement dout_thetamh4       = self.dthetamh4
            self.adthetamh4 += self.adout_thetamh4  
            self.adout_thetamh4 = 0
            #statement dout_thetamh3       = self.dthetamh3
            self.adthetamh3 += self.adout_thetamh3  
            self.adout_thetamh3 = 0
            #statement dout_thetamh2       = self.dthetamh2
            self.adthetamh2 += self.adout_thetamh2  
            self.adout_thetamh2 = 0
            #statement dout_thetamh       = self.dthetamh
            self.adthetamh += self.adout_thetamh  
            self.adout_thetamh = 0
            
        #statement dout_esat2m        = self.desat2m
        self.adesat2m += self.adout_esat2m
        self.adout_esat2m = 0
        #statement dout_e2m           = self.de2m
        self.ade2m += self.adout_e2m
        self.adout_e2m = 0
        #statement dout_v2m           = self.dv2m
        self.adv2m += self.adout_v2m  
        self.adout_v2m = 0
        #statement dout_u2m           = self.du2m
        self.adu2m += self.adout_u2m   
        self.adout_u2m = 0
        #statement dout_q2m           = self.dq2m
        self.adq2m += self.adout_q2m   
        self.adout_q2m = 0
        #statement dout_T2m           = self.dT2m
        self.adT2m += self.adout_T2m 
        self.adout_T2m = 0
        #statement dout_vw            = self.dvw
        self.advw += self.adout_vw   
        self.adout_vw = 0
        #statement dout_deltav        = self.ddeltav
        self.addeltav += self.adout_deltav
        self.adout_deltav = 0
        #statement dout_v             = self.dv
        self.adv += self.adout_v
        self.adout_v = 0
        #statement dout_uw            = self.duw
        self.aduw += self.adout_uw
        self.adout_uw = 0
        #statement dout_deltau        = self.ddeltau
        self.addeltau += self.adout_deltau
        self.adout_deltau = 0
        #statement dout_u             = self.du
        self.adu += self.adout_u
        self.adout_u = 0
        #statement dout_deltaCOS      = self.ddeltaCOS
        self.addeltaCOS += self.adout_deltaCOS 
        self.adout_deltaCOS = 0
        #statement dout_COS           = self.dCOS
        self.adCOS += self.adout_COS
        self.adout_COS = 0
        if model.sw_ls:
            if model.ls_type=='ags':
                #statement dout_wCOSS         = self.dwCOSS
                self.adwCOSS += self.adout_wCOSS
                self.adout_wCOSS = 0
                #statement dout_wCOSP         = self.dwCOSP
                self.adwCOSP += self.adout_wCOSP
                self.adout_wCOSP = 0
        #statement dout_wCOS          = self.dwCOS
        self.adwCOS += self.adout_wCOS
        self.adout_wCOS = 0
        #statement dout_wCO2M          = self.dwCO2M * fac
        self.adwCO2M += fac * self.adout_wCO2M
        self.adout_wCO2M = 0
        #statement dout_wCO2e         = self.dwCO2e * fac
        self.adwCO2e += fac * self.adout_wCO2e
        self.adout_wCO2e = 0
        #statement dout_wCO2R         = self.dwCO2R *fac
        self.adwCO2R += fac * self.adout_wCO2R
        self.adout_wCO2R = 0
        #statement dout_wCO2A         = self.dwCO2A * fac
        self.adwCO2A += fac * self.adout_wCO2A
        self.adout_wCO2A = 0
        #statement dout_wCO2          = self.dwCO2 * fac
        self.adwCO2 += fac * self.adout_wCO2
        self.adout_wCO2 = 0
        #statement dout_deltaCO2      = self.ddeltaCO2
        self.addeltaCO2 += self.adout_deltaCO2
        self.adout_deltaCO2 = 0
        #statement dout_CO2           = self.dCO2
        self.adCO2 += self.adout_CO2
        self.adout_CO2 = 0
        #statement dout_esatvar          = self.desatvar
        self.adesatvar += self.adout_esatvar 
        self.adout_esatvar = 0
        #statement dout_e             = self.de
        self.ade += self.adout_e 
        self.adout_e = 0
        #statement dout_qsatvar          = self.dqsatvar
        self.adqsatvar += self.adout_qsatvar
        self.adout_qsatvar = 0
        #statement dout_wqM           = self.dwqM
        self.adwqM += self.adout_wqM 
        self.adout_wqM = 0
        #statement dout_wqe           = self.dwqe
        self.adwqe += self.adout_wqe
        self.adout_wqe = 0
        #statement dout_wq            = self.dwq
        self.adwq += self.adout_wq 
        self.adout_wq = 0
        #statement dout_deltaq        = self.ddeltaq
        self.addeltaq += self.adout_deltaq  
        self.adout_deltaq = 0
        #statement dout_q             = self.dq
        self.adq += self.adout_q
        self.adout_q = 0
        #statement dout_wthetave      = self.dwthetave
        self.adwthetave += self.adout_wthetave
        self.adout_wthetave = 0
        #statement dout_wthetae       = self.dwthetae
        self.adwthetae += self.adout_wthetae
        self.adout_wthetae = 0
        #statement dout_wthetav       = self.dwthetav
        self.adwthetav += self.adout_wthetav
        self.adout_wthetav = 0
        #statement dout_wtheta        = self.dwtheta
        self.adwtheta += self.adout_wtheta 
        self.adout_wtheta = 0
        #statement dout_deltathetav   = self.ddeltathetav
        self.addeltathetav += self.adout_deltathetav
        self.adout_deltathetav = 0
        #statement dout_deltatheta    = self.ddeltatheta
        self.addeltatheta += self.adout_deltatheta
        self.adout_deltatheta = 0
        #statement dout_thetav        = self.dthetav
        self.adthetav += self.adout_thetav
        self.adout_thetav = 0
        #statement dout_theta         = self.dtheta
        self.adtheta += self.adout_theta
        self.adout_theta = 0
        #statement dout_h             = self.dh
        self.adh += self.adout_h 
        self.adout_h = 0
        
        if self.adjointtestingstore:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    
    def adj_run_mixed_layer(self,forcing,checkpoint,model,HTy_variables=None):
        h = checkpoint['rml_h']
        ac = checkpoint['rml_ac']
        lcl = checkpoint['rml_lcl']
        wCO2 = checkpoint['rml_wCO2']
        wCOS = checkpoint['rml_wCOS']
        wq = checkpoint['rml_wq']
        we = checkpoint['rml_we_end']
        wCO2M = checkpoint['rml_wCO2M']
        wCOSM = checkpoint['rml_wCOSM']
        wqM = checkpoint['rml_wqM']
        wCO2e = checkpoint['rml_wCO2e_end']
        wCOSe = checkpoint['rml_wCOSe_end']
        wqe = checkpoint['rml_wqe_end']
        wtheta = checkpoint['rml_wtheta']
        wthetae = checkpoint['rml_wthetae_end']
        deltatheta = checkpoint['rml_deltatheta']
        deltaq = checkpoint['rml_deltaq']
        deltaCO2 = checkpoint['rml_deltaCO2']
        deltaCOS = checkpoint['rml_deltaCOS']
        deltathetav = checkpoint['rml_deltathetav']
        deltau = checkpoint['rml_deltau']
        deltav = checkpoint['rml_deltav']
        thetav = checkpoint['rml_thetav']
        ustar = checkpoint['rml_ustar']
        u = checkpoint['rml_u']
        v = checkpoint['rml_v']
        uw = checkpoint['rml_uw_end']
        vw = checkpoint['rml_vw_end']
        wthetave = checkpoint['rml_wthetave_end']
        wthetav = checkpoint['rml_wthetav']
        ws = checkpoint['rml_ws_end']
        gammatheta = checkpoint['rml_gammatheta']
        gammatheta2 = checkpoint['rml_gammatheta2']
        htrans = checkpoint['rml_htrans']
        gammaq = checkpoint['rml_gammaq']
        gammaCO2 = checkpoint['rml_gammaCO2']
        gammaCOS = checkpoint['rml_gammaCOS']
        gammau = checkpoint['rml_gammau']
        gammav = checkpoint['rml_gammav']
        wf = checkpoint['rml_wf_end']
        M = checkpoint['rml_M']
        divU = checkpoint['rml_divU']
        fc = checkpoint['rml_fc']
        dFz = checkpoint['rml_dFz']
        wstar = checkpoint['rml_wstar_end']
        beta = checkpoint['rml_beta_end']
        if(ac > 0 or lcl - h < 300):
            #statement ddztend = (self.dlcl - self.dh - self.ddz_h) / 7200.
            self.adlcl += self.addztend/7200
            self.adh += -self.addztend/7200
            self.addz_h += -self.addztend/7200
            self.addztend = 0
        else:
            #statement ddztend = 0
            self.addztend = 0
        if(model.sw_wind): 
            if model.sw_advfp:
                #statement ddeltavtend += self.dadvv #if instead the dadvv would have been added to the statement for ddeltavtend below, it would eventually give the same result, also in the adjoint. Don't include a statement self.addeltavtend = 0 here!
                self.adadvv += self.addeltavtend
                #statement ddeltautend += self.dadvu
                self.adadvu += self.addeltautend
            #statement ddeltavtend = self.dgammav * (we + wf - M) + gammav * (dwe + dwf - self.dM) - dvtend
            self.adgammav += (we + wf - M) * self.addeltavtend
            self.adwe += gammav * self.addeltavtend
            self.adwf += gammav * self.addeltavtend
            self.adM += - gammav * self.addeltavtend
            self.advtend += -1 * self.addeltavtend 
            self.addeltavtend = 0
            #statement ddeltautend = self.dgammau * (we + wf - M) + gammau * (dwe + dwf - self.dM) - dutend
            self.adgammau += (we + wf - M) * self.addeltautend
            self.adwe += gammau * self.addeltautend
            self.adwf += gammau * self.addeltautend
            self.adM += - gammau * self.addeltautend
            self.adutend += -1 * self.addeltautend 
            self.addeltautend = 0
            #statement dvtend = self.dfc * deltau + fc * self.ddeltau + (self.dvw + dwe * deltav + we * self.ddeltav) / h + (vw + we * deltav) * -1 * h**(-2) * self.dh + self.dadvv
            self.adfc += deltau * self.advtend
            self.addeltau += fc * self.advtend
            self.advw += 1 / h * self.advtend
            self.adwe += deltav / h * self.advtend
            self.addeltav += we / h * self.advtend
            self.adh += (vw + we * deltav) * -1 * h**(-2) * self.advtend
            self.adadvv += self.advtend
            self.advtend = 0
            #statement dutend = -self.dfc * deltav - fc * self.ddeltav + (self.duw + dwe * deltau + we * self.ddeltau) / h + (uw + we * deltau) * -1 * h**(-2) * self.dh + self.dadvu
            self.adfc += - deltav * self.adutend
            self.addeltav += - fc * self.adutend
            self.aduw += 1 / h * self.adutend
            self.adwe += deltau / h * self.adutend
            self.addeltau += we / h * self.adutend
            self.adh += (uw + we * deltau) * -1 * h**(-2) * self.adutend
            self.adadvu += self.adutend
            self.adutend = 0
        if model.sw_advfp:
            #statement ddeltaCOStend += self.dadvCOS
            self.adadvCOS += self.addeltaCOStend
            #statement ddeltaCO2tend += self.dadvCO2
            self.adadvCO2 += self.addeltaCO2tend
            #statement ddeltaqtend += self.dadvq 
            self.adadvq += self.addeltaqtend
            #statement ddeltathetatend += self.dadvtheta
            self.adadvtheta += self.addeltathetatend 
        #statement ddeltaCOStend = (we + wf - M) * self.dgammaCOS + gammaCOS * (dwe + dwf - self.dM) - dCOStend + dw_COS_ft
        self.adgammaCOS += (we + wf - M) * self.addeltaCOStend
        self.adwe += gammaCOS * self.addeltaCOStend
        self.adwf += gammaCOS * self.addeltaCOStend
        self.adM += -gammaCOS * self.addeltaCOStend
        self.adCOStend += - self.addeltaCOStend
        self.adw_COS_ft += self.addeltaCOStend
        self.addeltaCOStend = 0
        #statement ddeltaCO2tend = (we + wf - M) * self.dgammaCO2 + gammaCO2 * (dwe + dwf - self.dM) - dCO2tend + dw_CO2_ft
        self.adgammaCO2 += (we + wf - M) * self.addeltaCO2tend
        self.adwe += gammaCO2 * self.addeltaCO2tend
        self.adwf += gammaCO2 * self.addeltaCO2tend
        self.adM += -gammaCO2 * self.addeltaCO2tend
        self.adCO2tend += - self.addeltaCO2tend
        self.adw_CO2_ft += self.addeltaCO2tend
        self.addeltaCO2tend = 0
        #statement ddeltaqtend = (we + wf - M) * self.dgammaq + gammaq * (dwe + dwf - self.dM) - dqtend + dw_q_ft
        self.adgammaq += (we + wf - M) * self.addeltaqtend
        self.adwe += gammaq * self.addeltaqtend
        self.adwf += gammaq * self.addeltaqtend
        self.adM += -gammaq * self.addeltaqtend
        self.adqtend += - self.addeltaqtend
        self.adw_q_ft += self.addeltaqtend
        self.addeltaqtend = 0
        if h <= htrans:
            #statement ddeltathetatend = (we + wf - M) * self.dgammatheta + gammatheta * (dwe + dwf - self.dM) - dthetatend + dw_th_ft
            self.adgammatheta += (we + wf - M) * self.addeltathetatend
            self.adwe += gammatheta * self.addeltathetatend
            self.adwf += gammatheta * self.addeltathetatend
            self.adM += -gammatheta * self.addeltathetatend
            self.adthetatend += - self.addeltathetatend
            self.adw_th_ft += self.addeltathetatend
            self.addeltathetatend = 0
        else:
            #statement ddeltathetatend = (we + wf - M) * self.dgammatheta2 + gammatheta2 * (dwe + dwf - self.dM) - dthetatend + dw_th_ft
            self.adgammatheta2 += (we + wf - M) * self.addeltathetatend
            self.adwe += gammatheta2 * self.addeltathetatend
            self.adwf += gammatheta2 * self.addeltathetatend
            self.adM += -gammatheta2 * self.addeltathetatend
            self.adthetatend += - self.addeltathetatend
            self.adw_th_ft += self.addeltathetatend
            self.addeltathetatend = 0
        #statement dCOStend = (self.dwCOS - dwCOSe - self.dwCOSM) / h + (wCOS - wCOSe - wCOSM) * (-1) * h**(-2) * self.dh + self.dadvCOS
        self.adwCOS += 1 / h * self.adCOStend
        self.adwCOSe += - 1 / h * self.adCOStend
        self.adwCOSM += - 1 / h * self.adCOStend
        self.adh += (wCOS - wCOSe - wCOSM) * (-1) * h**(-2) * self.adCOStend
        self.adadvCOS += self.adCOStend
        self.adCOStend = 0
        #statement dCO2tend = (self.dwCO2 - dwCO2e - self.dwCO2M) / h + (wCO2 - wCO2e - wCO2M) * (-1) * h**(-2) * self.dh + self.dadvCO2
        self.adwCO2 += 1 / h * self.adCO2tend
        self.adwCO2e += - 1 / h * self.adCO2tend
        self.adwCO2M += - 1 / h * self.adCO2tend
        self.adh += (wCO2 - wCO2e - wCO2M) * (-1) * h**(-2) * self.adCO2tend
        self.adadvCO2 += self.adCO2tend
        self.adCO2tend = 0
        #statement dqtend = (self.dwq - dwqe - self.dwqM) / h + (wq - wqe - wqM) * (-1) * h**(-2) * self.dh + self.dadvq
        self.adwq += 1 / h * self.adqtend
        self.adwqe += - 1 / h * self.adqtend
        self.adwqM += - 1 / h * self.adqtend
        self.adh += (wq - wqe - wqM) * (-1) * h**(-2) * self.adqtend
        self.adadvq += self.adqtend
        self.adqtend = 0
        #statement dthetatend = (self.dwtheta - dwthetae) / h + (wtheta - wthetae) * (-1) * h**(-2) * self.dh + self.dadvtheta 
        self.adwtheta += 1 / h * self.adthetatend
        self.adwthetae += - 1 / h * self.adthetatend
        self.adh += (wtheta - wthetae) * (-1) * h**(-2) * self.adthetatend
        self.adadvtheta += self.adthetatend
        self.adthetatend = 0
        #statement dhtend = dwe + dws + dwf - self.dM
        self.adwe += self.adhtend
        self.adws += self.adhtend
        self.adwf += self.adhtend
        self.adM += -self.adhtend
        self.adhtend = 0
        #statement dwCOSe = -dwe * deltaCOS + -we * self.ddeltaCOS
        self.adwe += - deltaCOS * self.adwCOSe
        self.addeltaCOS += -we * self.adwCOSe
        self.adwCOSe = 0
        #statement dwCO2e = -dwe * deltaCO2 + -we * self.ddeltaCO2
        self.adwe += - deltaCO2 * self.adwCO2e
        self.addeltaCO2 += -we * self.adwCO2e
        self.adwCO2e = 0
        #statement dwqe = -dwe * deltaq + -we * self.ddeltaq
        self.adwe += - deltaq * self.adwqe
        self.addeltaq += -we * self.adwqe
        self.adwqe = 0
        #statement dwthetae = -dwe * deltatheta + -we * self.ddeltatheta
        self.adwe += - deltatheta * self.adwthetae
        self.addeltatheta += -we * self.adwthetae
        self.adwthetae = 0
        #statement dwe = dwe_dwthetav + dwe_dustar + dwe_dthetav + dwe_dh + dwe_ddeltathetav
        self.adwe_dwthetav += self.adwe
        self.adwe_dustar += self.adwe
        self.adwe_dthetav += self.adwe
        self.adwe_dh += self.adwe
        self.adwe_ddeltathetav += self.adwe
        self.adwe = 0
        we = checkpoint['rml_we_middle']
        if(we < 0):
            #statement dwe_dwthetav = 0
            self.adwe_dwthetav = 0
            #statement dwe_dustar = 0
            self.adwe_dustar = 0
            #statement dwe_dthetav = 0
            self.adwe_dthetav = 0
            #statement dwe_dh = 0.
            self.adwe_dh = 0.
            #statement dwe_ddeltathetav = 0
            self.adwe_ddeltathetav = 0
        if(model.sw_shearwe):
            #statement dwe_dwthetav = -1 / deltathetav * dwthetave
            self.adwthetave += -1 / deltathetav * self.adwe_dwthetav
            self.adwe_dwthetav = 0
            #statement dwe_dustar = 5. * thetav / (model.g * h * deltathetav) * 3 * ustar**2 * self.dustar
            self.adustar += 5. * thetav / (model.g * h * deltathetav) * 3 * ustar**2 * self.adwe_dustar
            self.adwe_dustar = 0
            #statement dwe_dthetav = 5. * ustar ** 3. / (model.g * h * deltathetav) * self.dthetav
            self.adthetav += 5. * ustar ** 3. / (model.g * h * deltathetav) * self.adwe_dthetav
            self.adwe_dthetav = 0
            #statement dwe_dh    = 5* ustar ** 3. * thetav / model.g / deltathetav * (-1) * h**(-2) * self.dh
            self.adh += 5* ustar ** 3. * thetav / model.g / deltathetav * (-1) * h**(-2) * self.adwe_dh
            self.adwe_dh = 0
            #statement dwe_ddeltathetav = (-wthetave + 5. * ustar ** 3. * thetav / (model.g * h)) * (-1) * deltathetav**(-2) * self.ddeltathetav
            self.addeltathetav += (-wthetave + 5. * ustar ** 3. * thetav / (model.g * h)) * (-1) * deltathetav**(-2) * self.adwe_ddeltathetav
            self.adwe_ddeltathetav = 0
        else:
            #statement dwe_dwthetav = -1/ deltathetav * dwthetave
            self.adwthetave += -1/ deltathetav * self.adwe_dwthetav
            self.adwe_dwthetav = 0
            #statement dwe_dustar = 0
            self.adwe_dustar = 0
            #statement dwe_dthetav = 0
            self.adwe_dthetav = 0
            #statement dwe_dh    = 0
            self.adwe_dh    = 0
            #statement dwe_ddeltathetav = -wthetave * (-1) * deltathetav**(-2) * self.ddeltathetav
            self.addeltathetav += -wthetave * (-1) * deltathetav**(-2) * self.adwe_ddeltathetav
            self.adwe_ddeltathetav = 0    
        #statement dwthetave = -1 * beta * self.dwthetav - wthetav * self.dbeta
        self.adwthetav += -1 * beta * self.adwthetave
        self.adbeta += - wthetav * self.adwthetave
        self.adwthetave = 0
        if model.sw_dyn_beta:
            #statement dbeta = 5 * 3 * (ustar/wstar)**2. * (1 / wstar * self.dustar + ustar * -1 * wstar**-2 * dwstar)
            self.adustar += 5 * 3 * (ustar/wstar)**2. * (1 / wstar * self.adbeta)
            self.adwstar += 5 * 3 * (ustar/wstar)**2. * (ustar * -1 * wstar**-2 * self.adbeta)
            self.adbeta = 0.
        #statement dwstar = dwstar_dh + dwstar_dwthetav + dwstar_dthetav
        self.adwstar_dh += self.adwstar
        self.adwstar_dwthetav += self.adwstar
        self.adwstar_dthetav += self.adwstar
        self.adwstar = 0
        if(wthetav > 0.):
            #statement dwstar_dh = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * (model.g * wthetav / thetav) * self.dh
            self.adh += (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * (model.g * wthetav / thetav) * self.adwstar_dh
            self.adwstar_dh = 0
            #statement dwstar_dwthetav = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h / thetav * self.dwthetav
            self.adwthetav += (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h / thetav * self.adwstar_dwthetav
            self.adwstar_dwthetav = 0
            #statement dwstar_dthetav = (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h * wthetav * (-1) * thetav**(-2) * self.dthetav
            self.adthetav += (1./3.) * ((model.g * h * wthetav) / thetav)**(-2./3.) * model.g * h * wthetav * (-1) * thetav**(-2) * self.adwstar_dthetav
            self.adwstar_dthetav = 0
        else:
            #statement dwstar_dh  = 0
            self.adwstar_dh = 0
            #statement dwstar_dwthetav = 0
            self.adwstar_dwthetav = 0
            #statement dwstar_dthetav = 0
            self.adwstar_dthetav = 0
        #statement dwf = 1 / (model.rho * model.cp) * (1 / deltatheta * self.ddFz + dFz * (-1) * deltatheta**(-2) * self.ddeltatheta) 
        self.addFz += 1 / (model.rho * model.cp) * 1 / deltatheta * self.adwf
        self.addeltatheta += dFz / (model.rho * model.cp) * (-1) * deltatheta**(-2) * self.adwf
        self.adwf = 0
        if(model.sw_fixft):
            #statement dw_COS_ft = ws * self.dgammaCOS + gammaCOS * dws
            self.adgammaCOS += ws * self.adw_COS_ft
            self.adws +=  gammaCOS * self.adw_COS_ft
            self.adw_COS_ft = 0
            #statement dw_CO2_ft = ws * self.dgammaCO2 + gammaCO2 * dws
            self.adgammaCO2 += ws * self.adw_CO2_ft
            self.adws += gammaCO2 * self.adw_CO2_ft
            self.adw_CO2_ft = 0
            #statement dw_q_ft   = ws * self.dgammaq + gammaq * dws
            self.adgammaq += ws * self.adw_q_ft
            self.adws += gammaq * self.adw_q_ft
            self.adw_q_ft = 0
            if h <= htrans:
                #statement dw_th_ft  = ws * self.dgammatheta + gammatheta * dws
                self.adgammatheta += ws * self.adw_th_ft
                self.adws += gammatheta * self.adw_th_ft
                self.adw_th_ft = 0
            else:
                #statement dw_th_ft  = ws * self.dgammatheta2 + gammatheta2 * dws
                self.adgammatheta2 += ws * self.adw_th_ft
                self.adws += gammatheta2 * self.adw_th_ft
                self.adw_th_ft = 0
        else:
            #statement dw_COS_ft = 0
            self.adw_COS_ft = 0
            #statement dw_CO2_ft = 0
            self.adw_CO2_ft = 0
            #statement dw_q_ft  = 0
            self.adw_q_ft  = 0
            #statement dw_th_ft = 0
            self.adw_th_ft = 0    
        #statement dws = - h * self.ddivU - divU * self.dh
        self.addivU += - h * self.adws
        self.adh += -divU * self.adws
        self.adws = 0
        if(not model.sw_sl):
            #statement dvw = dvw_dustar + dvw_du + dvw_dv
            self.advw_dustar += self.advw
            self.advw_du += self.advw
            self.advw_dv += self.advw
            self.advw = 0
            #statement dvw_dv = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * (u **2.) * -2 * v**(-3) * self.dv
            self.adv += - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * (u **2.) * -2 * v**(-3) * self.advw_dv
            self.advw_dv = 0
            #statement dvw_du = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * 1 / (v **2.) * 2 * u * self.du
            self.adu += - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (u ** 2. / v ** 2. + 1.)**(-2) * 1 / (v **2.) * 2 * u * self.advw_du
            self.advw_du = 0
            #statement dvw_dustar = - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * self.dustar * 1 / (u ** 2. / v ** 2. + 1.)
            self.adustar += - np.sign(v) * 0.5 * (ustar ** 4. / (u ** 2. / v ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * 1 / (u ** 2. / v ** 2. + 1.) * self.advw_dustar
            self.advw_dustar = 0
            #statement duw = duw_dustar + duw_du + duw_dv
            self.aduw_dustar += self.aduw
            self.aduw_du += self.aduw
            self.aduw_dv += self.aduw
            self.aduw = 0
            #statement duw_dv = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * 1 / (u ** 2.) * 2 * v * self.dv
            self.adv += - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * 1 / (u ** 2.) * 2 * v * self.aduw_dv
            self.aduw_dv = 0
            #statement duw_du = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * (v **2.) * -2 * u**(-3) * self.du
            self.adu += - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * ustar ** 4. * -1 * (v ** 2. / u ** 2. + 1.)**(-2) * (v **2.) * -2 * u**(-3) * self.aduw_du
            self.aduw_du = 0
            #statement duw_dustar = - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * self.dustar * 1 / (v ** 2. / u ** 2. + 1.)
            self.adustar += - np.sign(u) * 0.5 * (ustar ** 4. / (v ** 2. / u ** 2. + 1.)) ** (-0.5) * 4 * ustar ** 3 * 1 / (v ** 2. / u ** 2. + 1.) * self.aduw_dustar
            self.aduw_dustar = 0
        if self.adjointtestingrun_mixed_layer:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
    
    def adj_run_cumulus(self,forcing,checkpoint,model,HTy_variables=None):
        wthetav = checkpoint['rc_wthetav']
        deltaq = checkpoint['rc_deltaq']
        dz_h = checkpoint['rc_dz_h']
        wstar = checkpoint['rc_wstar']
        wqe = checkpoint['rc_wqe']
        wqM = checkpoint['rc_wqM']
        h = checkpoint['rc_h']
        deltaCO2 = checkpoint['rc_deltaCO2']
        deltaCOS = checkpoint['rc_deltaCOS']
        wCO2e = checkpoint['rc_wCO2e']
        wCO2M = checkpoint['rc_wCO2M']
        wCOSe = checkpoint['rc_wCOSe']
        wCOSM = checkpoint['rc_wCOSM']
        q = checkpoint['rc_q']
        T_h = checkpoint['rc_T_h']
        P_h = checkpoint['rc_P_h']
        q2_h = checkpoint['rc_q2_h_end']
        ac = checkpoint['rc_ac_end']
        M = checkpoint['rc_M_end']
        CO22_h = checkpoint['rc_CO22_h_end']
        COS2_h = checkpoint['rc_COS2_h_end']
        qsat_variable_rc = checkpoint['rc_qsat_variable_rc_end']
        if(deltaCOS < 0):
            #statement dwCOSM  = dM * COS2_h**0.5 + M * 0.5 * COS2_h**(-0.5) * dCOS2_h
            self.adM += COS2_h**0.5 * self.adwCOSM
            self.adCOS2_h += M * 0.5 * COS2_h**(-0.5) * self.adwCOSM 
            self.adwCOSM = 0
        else:
            #statement dwCOSM  = 0.
            self.adwCOSM = 0
        if(deltaCO2 < 0):
            #statement dwCO2M  = dM * CO22_h**0.5 + M * 0.5 * CO22_h**(-0.5) * dCO22_h
            self.adM += CO22_h**0.5 * self.adwCO2M
            self.adCO22_h += M * 0.5 * CO22_h**(-0.5) * self.adwCO2M 
            self.adwCO2M = 0
        else:
            #statement dwCO2M  = 0.
            self.adwCO2M = 0
        #statement dwqM = dM * q2_h**0.5 + M * 0.5 * q2_h**(-0.5) * dq2_h
        self.adM += q2_h**0.5 * self.adwqM
        self.adq2_h += M * 0.5 * q2_h**(-0.5) * self.adwqM
        self.adwqM = 0
        #statement dM = dac * wstar + ac * self.dwstar
        self.adac += wstar * self.adM
        self.adwstar += ac * self.adM
        self.adM = 0
        if 0.5 + (0.36 * np.arctan(1.55 * ((q - qsat_variable_rc) / q2_h**0.5))) < 0:
            #statement dac = 0
            self.adac = 0
        else:
            #statement dac = dac_dq + dac_dT_h + dac_dP_h + dac_dq2_h
            self.adac_dq += self.adac
            self.adac_dT_h += self.adac
            self.adac_dP_h += self.adac
            self.adac_dq2_h += self.adac
            self.adac = 0
            #statement dac_dq2_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * (q - qsat_variable_rc) * -0.5 * q2_h**(-1.5) * dq2_h
            self.adq2_h += 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * (q - qsat_variable_rc) * -0.5 * q2_h**(-1.5) * self.adac_dq2_h
            self.adac_dq2_h = 0
            # (((q - qsat_variable_rc) / q2_h**0.5)) will evaluate to inf if q2_h = 0 Than the statementfor self.adq2_h will give 0 * inf which evaluates to nan
            #statement dac_dP_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * dqsat_variable_rc_dP_h
            self.adqsat_variable_rc_dP_h += 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * self.adac_dP_h
            self.adac_dP_h = 0
            #statement dac_dT_h = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * dqsat_variable_rc_dT_h
            self.adqsat_variable_rc_dT_h += 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * - 1 * self.adac_dT_h
            self.adac_dT_h = 0
            #statement dac_dq = 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * self.dq
            self.adq += 0.36 * 1 / (1 + (1.55 * ((q - qsat_variable_rc) / q2_h**0.5))**2) * 1.55 * 1 / q2_h**0.5 * self.adac_dq
            self.adac_dq = 0
            #statement dqsat_variable_rc_dP_h = dqsat_dp(T_h,P_h,self.dP_h)
            self.adP_h += dqsat_dp(T_h,P_h,self.adqsat_variable_rc_dP_h)
            self.adqsat_variable_rc_dP_h = 0
            #statement dqsat_variable_rc_dT_h = dqsat_dT(T_h,P_h,self.dT_h)
            self.adT_h += dqsat_dT(T_h,P_h,self.adqsat_variable_rc_dT_h)
            self.adqsat_variable_rc_dT_h = 0
        q2_h = checkpoint['rc_q2_h_middle']    
        CO22_h = checkpoint['rc_CO22_h_middle']
        COS2_h = checkpoint['rc_COS2_h_middle']
        if COS2_h <= 0.:
            #statement dCOS2_h = 0.
            self.adCOS2_h = 0.
        if CO22_h <= 0.:
            #statement dCO22_h = 0.
            self.adCO22_h = 0.
        if q2_h <= 0.:
            #statement dq2_h   = 0.
            self.adq2_h   = 0.
        if(wthetav > 0):
            #statement dCOS2_h = dCOS2_h_dwCOSe + dCOS2_h_dwCOSM + dCOS2_h_ddeltaCOS + dCOS2_h_dh + dCOS2_h_ddz_h + dCOS2_h_dwstar
            self.adCOS2_h_dwCOSe += self.adCOS2_h
            self.adCOS2_h_dwCOSM += self.adCOS2_h
            self.adCOS2_h_ddeltaCOS += self.adCOS2_h
            self.adCOS2_h_dh += self.adCOS2_h
            self.adCOS2_h_ddz_h += self.adCOS2_h
            self.adCOS2_h_dwstar += self.adCOS2_h
            self.adCOS2_h = 0
            #statement dCOS2_h_dwstar = -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            self.adwstar += -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * dz_h * self.adCOS2_h_dwstar
            self.adCOS2_h_dwstar = 0
            #statement dCOS2_h_ddz_h = -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            self.addz_h += -(wCOSe+ wCOSM) * deltaCOS * h * -1 * (dz_h * wstar)**-2 * wstar * self.adCOS2_h_ddz_h
            self.adCOS2_h_ddz_h = 0
            #statement dCOS2_h_dh = -(wCOSe+ wCOSM) * deltaCOS / (dz_h * wstar) * self.dh
            self.adh += -(wCOSe+ wCOSM) * deltaCOS / (dz_h * wstar) * self.adCOS2_h_dh
            self.adCOS2_h_dh = 0
            #statement dCOS2_h_ddeltaCOS = -(wCOSe+ wCOSM) * h / (dz_h * wstar) * self.ddeltaCOS
            self.addeltaCOS += -(wCOSe+ wCOSM) * h / (dz_h * wstar) * self.adCOS2_h_ddeltaCOS
            self.adCOS2_h_ddeltaCOS = 0
            #statement dCOS2_h_dwCOSM = -(self.dwCOSM) * deltaCOS * h / (dz_h * wstar)
            self.adwCOSM += -1 * deltaCOS * h / (dz_h * wstar) * self.adCOS2_h_dwCOSM
            self.adCOS2_h_dwCOSM = 0
            #statement dCOS2_h_dwCOSe = -(self.dwCOSe) * deltaCOS * h / (dz_h * wstar)
            self.adwCOSe += -1 * deltaCOS * h / (dz_h * wstar) * self.adCOS2_h_dwCOSe
            self.adCOS2_h_dwCOSe = 0          
            #statement dCO22_h = dCO22_h_dwCO2e + dCO22_h_dwCO2M + dCO22_h_ddeltaCO2 + dCO22_h_dh + dCO22_h_ddz_h + dCO22_h_dwstar
            self.adCO22_h_dwCO2e += self.adCO22_h
            self.adCO22_h_dwCO2M += self.adCO22_h
            self.adCO22_h_ddeltaCO2 += self.adCO22_h
            self.adCO22_h_dh += self.adCO22_h
            self.adCO22_h_ddz_h += self.adCO22_h
            self.adCO22_h_dwstar += self.adCO22_h
            self.adCO22_h = 0
            #statement dCO22_h_dwstar = -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            self.adwstar += -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * dz_h * self.adCO22_h_dwstar
            self.adCO22_h_dwstar = 0
            #statement dCO22_h_ddz_h = -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            self.addz_h += -(wCO2e+ wCO2M) * deltaCO2 * h * -1 * (dz_h * wstar)**-2 * wstar * self.adCO22_h_ddz_h
            self.adCO22_h_ddz_h = 0
            #statement dCO22_h_dh = -(wCO2e+ wCO2M) * deltaCO2 / (dz_h * wstar) * self.dh
            self.adh += -(wCO2e+ wCO2M) * deltaCO2 / (dz_h * wstar) * self.adCO22_h_dh
            self.adCO22_h_dh = 0
            #statement dCO22_h_ddeltaCO2 = -(wCO2e+ wCO2M) * h / (dz_h * wstar) * self.ddeltaCO2
            self.addeltaCO2 += -(wCO2e+ wCO2M) * h / (dz_h * wstar) * self.adCO22_h_ddeltaCO2
            self.adCO22_h_ddeltaCO2 = 0
            #statement dCO22_h_dwCO2M = -(self.dwCO2M) * deltaCO2 * h / (dz_h * wstar)
            self.adwCO2M += -1 * deltaCO2 * h / (dz_h * wstar) * self.adCO22_h_dwCO2M
            self.adCO22_h_dwCO2M = 0
            #statement dCO22_h_dwCO2e = -(self.dwCO2e) * deltaCO2 * h / (dz_h * wstar)
            self.adwCO2e += -1 * deltaCO2 * h / (dz_h * wstar) * self.adCO22_h_dwCO2e
            self.adCO22_h_dwCO2e = 0
            #statement dq2_h = dq2_h_dwqe + dq2_h_dwqM + dq2_h_ddeltaq + dq2_h_dh + dq2_h_ddz_h + dq2_h_dwstar
            self.adq2_h_dwqe += self.adq2_h
            self.adq2_h_dwqM += self.adq2_h
            self.adq2_h_ddeltaq += self.adq2_h
            self.adq2_h_dh  += self.adq2_h
            self.adq2_h_ddz_h += self.adq2_h
            self.adq2_h_dwstar += self.adq2_h
            self.adq2_h = 0
            #statement dq2_h_dwstar = -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * dz_h * self.dwstar
            self.adwstar += -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * dz_h * self.adq2_h_dwstar
            self.adq2_h_dwstar = 0
            #statement dq2_h_ddz_h = -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * wstar * self.ddz_h
            self.addz_h += -(wqe  + wqM  ) * deltaq   * h * -1 * (dz_h * wstar)**-2 * wstar * self.adq2_h_ddz_h
            self.adq2_h_ddz_h = 0
            #statement dq2_h_dh = -(wqe  + wqM  ) * deltaq / (dz_h * wstar) * self.dh
            self.adh += -(wqe  + wqM  ) * deltaq / (dz_h * wstar) * self.adq2_h_dh
            self.adq2_h_dh = 0
            #statement dq2_h_ddeltaq = -(wqe  + wqM  ) * h / (dz_h * wstar) * self.ddeltaq
            self.addeltaq += -(wqe  + wqM  ) * h / (dz_h * wstar) * self.adq2_h_ddeltaq
            self.adq2_h_ddeltaq = 0
            #statement dq2_h_dwqM = -(self.dwqM) * deltaq   * h / (dz_h * wstar)
            self.adwqM += -1 * deltaq   * h / (dz_h * wstar) * self.adq2_h_dwqM
            self.adq2_h_dwqM = 0
            #statement dq2_h_dwqe = -(self.dwqe) * deltaq   * h / (dz_h * wstar)
            self.adwqe += -1 * deltaq   * h / (dz_h * wstar) * self.adq2_h_dwqe
            self.adq2_h_dwqe = 0
        else:
            #statement dCOS2_h = 0.
            self.adCOS2_h = 0.
            #statement dCO22_h = 0.
            self.adCO22_h = 0.
            #statement dq2_h = 0.
            self.adq2_h = 0.
        
        if self.adjointtestingrun_cumulus:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    def adj_run_land_surface(self,forcing,checkpoint,model,HTy_variables=None):
        u = checkpoint['rls_u']
        v = checkpoint['rls_v']
        wstar = checkpoint['rls_wstar']
        Cs = checkpoint['rls_Cs']
        ueff = checkpoint['rls_ueff_end']
        theta = checkpoint['rls_theta']
        esatvar = checkpoint['rls_esatvar_end']
        wg = checkpoint['rls_wg']
        wfc = checkpoint['rls_wfc']
        wwilt = checkpoint['rls_wwilt']
        Wmax = checkpoint['rls_Wmax']
        LAI = checkpoint['rls_LAI']
        Wlmx = checkpoint['rls_Wlmx_end']
        Wl = checkpoint['rls_Wl']
        ra = checkpoint['rls_ra_end']
        cveg = checkpoint['rls_cveg']
        cliq = checkpoint['rls_cliq_end']
        dqsatdT = checkpoint['rls_dqsatdT_end']
        rssoil = checkpoint['rls_rssoil_end']
        Lambda = checkpoint['rls_Lambda']
        qsatvar = checkpoint['rls_qsatvar_end']
        q = checkpoint['rls_q']
        Tsoil = checkpoint['rls_Tsoil']
        p2_numerator_Ts = checkpoint['rls_p2_numerator_Ts_end']
        p3_numerator_Ts = checkpoint['rls_p3_numerator_Ts_end']
        p4_numerator_Ts = checkpoint['rls_p4_numerator_Ts_end']
        p5_numerator_Ts = checkpoint['rls_p5_numerator_Ts_end']
        p3_denominator_Ts = checkpoint['rls_p3_denominator_Ts_end']
        p4_denominator_Ts = checkpoint['rls_p4_denominator_Ts_end']
        numerator_Ts = checkpoint['rls_numerator_Ts_end']
        denominator_Ts = checkpoint['rls_denominator_Ts_end']
        p1_LEveg = checkpoint['rls_p1_LEveg_end']
        p2_LEveg_liq_soil = checkpoint['rls_p2_LEveg_liq_soil_end']
        Ts = checkpoint['rls_Ts_end']
        p1_LEliq = checkpoint['rls_p1_LEliq_end']
        p1_LEsoil = checkpoint['rls_p1_LEsoil_end']
        G = checkpoint['rls_G_end']
        Q = checkpoint['rls_Q']
        numerator_LEpot = checkpoint['rls_numerator_LEpot_end']
        rsmin = checkpoint['rls_rsmin']
        denominator_LEref = checkpoint['rls_denominator_LEref_end']
        numerator_LEref = checkpoint['rls_numerator_LEref_end']
        w2 = checkpoint['rls_w2']
        wsat = checkpoint['rls_wsat']
        CG = checkpoint['rls_CG_end']
        C1 = checkpoint['rls_C1_end']
        C2 = checkpoint['rls_C2_end']
        d1 = checkpoint['rls_d1_end']
        LEsoil = checkpoint['rls_LEsoil_end']
        wgeq = checkpoint['rls_wgeq_end']
        rs = checkpoint['rls_rs_end'] #rs calculated in e.g. ags
        ustar = checkpoint['rls_ustar']
        rssoilmin = checkpoint['rls_rssoilmin']
        f2 = checkpoint['rls_f2_end']
        CGsat = checkpoint['rls_CGsat']
        b = checkpoint['rls_b']
        C1sat = checkpoint['rls_C1sat']
        C2ref = checkpoint['rls_C2ref']
        a = checkpoint['rls_a']
        p = checkpoint['rls_p']
        
        #statement dwq       = dLE / (model.rho * model.Lv)
        self.adLE += 1 / (model.rho * model.Lv) * self.adwq
        self.adwq = 0
        #statement dwtheta   = dH  / (model.rho * model.cp)
        self.adH += 1 / (model.rho * model.cp) * self.adwtheta
        self.adwtheta = 0
        #statement dwgtend = dwgtend_dLEsoil + dwgtend_dC1 + dwgtend_dC2 + dwgtend_dwg +dwgtend_dwgeq
        self.adwgtend_dLEsoil += self.adwgtend
        self.adwgtend_dC1 += self.adwgtend
        self.adwgtend_dC2 += self.adwgtend
        self.adwgtend_dwg += self.adwgtend
        self.adwgtend_dwgeq += self.adwgtend
        self.adwgtend = 0
        #statement dwgtend_dwgeq = C2 / 86400. * dwgeq
        self.adwgeq += C2 / 86400. * self.adwgtend_dwgeq
        self.adwgtend_dwgeq = 0
        #statement dwgtend_dwg = - C2 / 86400. * self.dwg
        self.adwg += - C2 / 86400. * self.adwgtend_dwg
        self.adwgtend_dwg = 0
        #statement dwgtend_dC2 = - dC2 / 86400. * (wg - wgeq)
        self.adC2 += - 1 / 86400. * (wg - wgeq) * self.adwgtend_dC2
        self.adwgtend_dC2 =  0
        #statement dwgtend_dC1 = - dC1 / (model.rhow * d1) * LEsoil / model.Lv
        #note that d1 is just a constant!
        self.adC1 += - self.adwgtend_dC1 / (model.rhow * d1) * LEsoil / model.Lv
        self.adwgtend_dC1 = 0
        #statement dwgtend_dLEsoil = - C1 / (model.rhow * d1) / model.Lv * dLEsoil 
        self.adLEsoil += - C1 / (model.rhow * d1) / model.Lv * self.adwgtend_dLEsoil 
        self.adwgtend_dLEsoil = 0
        #statement dwgeq      = dwgeq_dw2 + dwgeq_dwsat + dwgeq_da + dwgeq_dp
        self.adwgeq_dw2 += self.adwgeq
        self.adwgeq_dwsat += self.adwgeq
        self.adwgeq_da += self.adwgeq
        self.adwgeq_dp += self.adwgeq
        self.adwgeq = 0
        #statement dwgeq_dp = - wsat * a * ( (1. - (w2 / wsat) ** (8. * p)) * (w2 / wsat) ** p * np.log(w2 / wsat) + (w2 / wsat) ** p * -1 * (w2 / wsat) ** (8. * p) * np.log(w2 / wsat) * 8) * self.dp
        self.adp += - wsat * a * ( (1. - (w2 / wsat) ** (8. * p)) * (w2 / wsat) ** p * np.log(w2 / wsat) + (w2 / wsat) ** p * -1 * (w2 / wsat) ** (8. * p) * np.log(w2 / wsat) * 8) * self.adwgeq_dp
        self.adwgeq_dp = 0
        #statement dwgeq_da = - wsat * (w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p)) * self.da
        self.ada += - wsat * (w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p)) * self.adwgeq_da
        self.adwgeq_da = 0
        #statement dwgeq_dwsat = -self.dwsat * a * ((w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p))) - wsat * a * (p * (w2 / wsat) ** (p - 1) * w2 * (-1) * wsat**(-2) * self.dwsat * (1. - (w2 / wsat) ** (8. * p)) + (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * w2 * (-1) * wsat**(-2) * self.dwsat)
        self.adwsat += -1 * a * ((w2 / wsat) ** p * (1. - (w2 / wsat) ** (8. * p))) * self.adwgeq_dwsat
        self.adwsat += - wsat * a * p * (w2 / wsat) ** (p - 1) * w2 * (-1) * wsat**(-2) * (1. - (w2 / wsat) ** (8. * p)) * self.adwgeq_dwsat
        self.adwsat += - wsat * a * (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * w2 * (-1) * wsat**(-2) * self.adwgeq_dwsat
        self.adwgeq_dwsat = 0
        #statement dwgeq_dw2  = self.dw2 - wsat * a * (p * (w2 / wsat) ** (p - 1) * self.dw2 / wsat * (1. - (w2 / wsat) ** (8. * p)) + (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * self.dw2 / wsat)
        self.adw2 += self.adwgeq_dw2
        self.adw2 += - wsat * a * p * (w2 / wsat) ** (p - 1) * self.adwgeq_dw2 / wsat * (1. - (w2 / wsat) ** (8. * p))
        self.adw2 += - wsat * a * (w2 / wsat) ** (p) * -1 * (8. * p) * (w2 / wsat) ** (8. * p - 1) * self.adwgeq_dw2 / wsat
        self.adwgeq_dw2 = 0
        #statement dC2        = dC2_dC2ref + dC2_dw2 + dC2_dwsat
        self.adC2_dC2ref += self.adC2
        self.adC2_dw2 += self.adC2
        self.adC2_dwsat += self.adC2
        self.adC2 = 0
        #statement dC2_dwsat  = C2ref * (w2 * (-1) * (wsat - w2)**(-2) * self.dwsat)
        self.adwsat += C2ref * w2 * (-1) * (wsat - w2)**(-2) * self.adC2_dwsat
        self.adC2_dwsat = 0
        #statement dC2_dw2    = C2ref * (self.dw2 / (wsat - w2) + w2 * (-1) * (wsat - w2)**(-2) * -self.dw2)
        self.adw2 += C2ref * 1 / (wsat - w2) * self.adC2_dw2
        self.adw2 += C2ref * w2 * (-1) * (wsat - w2)**(-2) * -self.adC2_dw2
        self.adC2_dw2 = 0
        #statement dC2_dC2ref = w2 / (wsat - w2) * self.dC2ref
        self.adC2ref += w2 / (wsat - w2) * self.adC2_dC2ref
        self.adC2_dC2ref = 0
        #statement dC1 = dC1_dC1sat + dC1_dwsat + dC1_dwg + dC1_db
        self.adC1_dC1sat += self.adC1
        self.adC1_dwsat += self.adC1
        self.adC1_dwg += self.adC1
        self.adC1_db += self.adC1
        self.adC1 = 0
        #statement dC1_db     = C1sat * (wsat / wg) ** (b / 2. + 1.) * np.log(wsat / wg) * 1 / 2. * self.db
        self.adb += C1sat * (wsat / wg) ** (b / 2. + 1.) * np.log(wsat / wg) * 1 / 2. * self.adC1_db
        self.adC1_db = 0
        #statement dC1_dwg    = C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) * wsat * (-1) * wg**(-2) * self.dwg
        self.adwg += C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) * wsat * (-1) * wg**(-2) * self.adC1_dwg
        self.adC1_dwg = 0
        #statement dC1_dwsat  = C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) * self.dwsat / wg
        self.adwsat += C1sat * (b / 2 + 1.) * (wsat / wg) ** (b / 2.) / wg * self.adC1_dwsat 
        self.adC1_dwsat = 0
        #statement dC1_dC1sat = (wsat / wg) ** (b / 2. + 1.) * self.dC1sat
        self.adC1sat += (wsat / wg) ** (b / 2. + 1.) * self.adC1_dC1sat
        self.adC1_dC1sat = 0
        #statement dTsoiltend = dTsoiltend_dCG + dTsoiltend_dG + dTsoiltend_dTsoil + dTsoiltend_dT2
        self.adTsoiltend_dCG += self.adTsoiltend
        self.adTsoiltend_dG += self.adTsoiltend
        self.adTsoiltend_dTsoil += self.adTsoiltend
        self.adTsoiltend_dT2 += self.adTsoiltend
        self.adTsoiltend = 0
        #statement dTsoiltend_dT2 = 2. * np.pi / 86400. * self.dT2
        self.adT2 += 2. * np.pi / 86400. * self.adTsoiltend_dT2
        self.adTsoiltend_dT2 = 0
        #statement dTsoiltend_dTsoil = - 2. * np.pi / 86400. * self.dTsoil
        self.adTsoil += - 2. * np.pi / 86400. * self.adTsoiltend_dTsoil
        self.adTsoiltend_dTsoil = 0
        #statement dTsoiltend_dG   = CG * dG
        self.adG += CG * self.adTsoiltend_dG
        self.adTsoiltend_dG = 0
        #statement dTsoiltend_dCG = G * dCG
        self.adCG += G * self.adTsoiltend_dCG
        self.adTsoiltend_dCG = 0
        #statement dCG = dCG_dCGsat + dCG_dwsat + dCG_dw2 + dCG_db
        self.adCG_dCGsat += self.adCG
        self.adCG_dwsat += self.adCG
        self.adCG_dw2 += self.adCG
        self.adCG_db += self.adCG
        self.adCG = 0
        #statement dCG_db = CGsat * (wsat / w2)**(b / (2. * np.log(10.))) * np.log(wsat / w2) * 1 / (2. * np.log(10.)) * self.db 
        self.adb += CGsat * (wsat / w2)**(b / (2. * np.log(10.))) * np.log(wsat / w2) * 1 / (2. * np.log(10.)) * self.adCG_db
        self.adCG_db = 0
        #statement dCG_dw2 = CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) *  wsat * (-1) * w2**(-2) * self.dw2
        self.adw2 += CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) *  wsat * (-1) * w2**(-2) * self.adCG_dw2
        self.adCG_dw2 = 0
        #statement dCG_dwsat = CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) * self.dwsat / w2
        self.adwsat += CGsat * (b / (2. * np.log(10.))) * (wsat / w2)**(b / (2. * np.log(10.)) - 1) * self.adCG_dwsat / w2
        self.adCG_dwsat = 0
        #statement dCG_dCGsat = (wsat / w2)**(b / (2. * np.log(10.))) * self.dCGsat
        self.adCGsat += (wsat / w2)**(b / (2. * np.log(10.))) * self.adCG_dCGsat
        self.adCG_dCGsat = 0
        if(self.model.ls_type != 'canopy_model'):
            #statement dLEref  = dnumerator_LEref / denominator_LEref + numerator_LEref * (-1) * denominator_LEref**(-2) * ddenominator_LEref
            self.adnumerator_LEref += 1 / denominator_LEref * self.adLEref
            self.addenominator_LEref += numerator_LEref * (-1) * denominator_LEref**(-2) * self.adLEref
            self.adLEref = 0
            #statement ddenominator_LEref = ddqsatdT_dtheta + model.cp / model.Lv * (self.drsmin / LAI / ra + rsmin / ra * (-1) * LAI**(-2) * self.dLAI + rsmin / LAI * (-1) * ra**(-2) * dra)
            self.addqsatdT_dtheta += self.addenominator_LEref
            self.adrsmin += model.cp / model.Lv * 1 / LAI / ra * self.addenominator_LEref
            self.adLAI += model.cp / model.Lv * rsmin / ra * (-1) * LAI**(-2) * self.addenominator_LEref
            self.adra += model.cp / model.Lv * rsmin / LAI * (-1) * ra**(-2) * self.addenominator_LEref
            self.addenominator_LEref = 0
            #statement dnumerator_LEref = dp1_numerator_LEref + dp2_numerator_LEref
            self.adp1_numerator_LEref += self.adnumerator_LEref
            self.adp2_numerator_LEref += self.adnumerator_LEref
            self.adnumerator_LEref = 0
            #statement dp2_numerator_LEref = model.rho * model.cp * ((qsatvar - q) * (-1) * ra**(-2) * dra + 1 / ra * (dqsatvar - self.dq))
            self.adra += model.rho * model.cp * (qsatvar - q) * (-1) * ra**(-2) * self.adp2_numerator_LEref
            self.adqsatvar += model.rho * model.cp *  1 / ra * self.adp2_numerator_LEref
            self.adq += model.rho * model.cp *  1 / ra * -self.adp2_numerator_LEref
            self.adp2_numerator_LEref = 0
            #statement dp1_numerator_LEref = ddqsatdT_dtheta * (Q - G) + dqsatdT * (self.dQ - dG)
            #dqsatdT is not a derivative here
            self.addqsatdT_dtheta += (Q - G) * self.adp1_numerator_LEref
            self.adQ += dqsatdT * self.adp1_numerator_LEref
            self.adG += -dqsatdT * self.adp1_numerator_LEref
            self.adp1_numerator_LEref = 0
            #statement dLEpot = dnumerator_LEpot / (dqsatdT + model.cp / model.Lv) + numerator_LEpot * (-1) * (dqsatdT + model.cp / model.Lv)**(-2) * ddqsatdT_dtheta
            #dqsatdT is not a derivative here
            self.adnumerator_LEpot += 1 / (dqsatdT + model.cp / model.Lv) * self.adLEpot
            self.addqsatdT_dtheta += numerator_LEpot * (-1) * (dqsatdT + model.cp / model.Lv)**(-2) * self.adLEpot
            self.adLEpot = 0
            #statement dnumerator_LEpot = dp1_numerator_LEpot + dp2_numerator_LEpot
            self.adp1_numerator_LEpot += self.adnumerator_LEpot
            self.adp2_numerator_LEpot += self.adnumerator_LEpot
            self.adnumerator_LEpot = 0
            #statement dp2_numerator_LEpot = model.rho * model.cp * ((qsatvar - q) * (-1) * ra**(-2) * dra + 1 / ra * (dqsatvar - self.dq))
            self.adra += model.rho * model.cp * (qsatvar - q) * (-1) * ra**(-2) * self.adp2_numerator_LEpot
            self.adqsatvar += model.rho * model.cp * 1 / ra * self.adp2_numerator_LEpot
            self.adq += model.rho * model.cp * 1 / ra * -self.adp2_numerator_LEpot
            self.adp2_numerator_LEpot = 0
            #statement dp1_numerator_LEpot = ddqsatdT_dtheta * (Q - G) + dqsatdT * (self.dQ - dG)
            self.addqsatdT_dtheta += (Q - G) * self.adp1_numerator_LEpot
            self.adQ += dqsatdT * self.adp1_numerator_LEpot
            self.adG += -dqsatdT * self.adp1_numerator_LEpot
            self.adp1_numerator_LEpot = 0
            #statement dG      = self.dLambda * (Ts - Tsoil) + Lambda * (dTs - self.dTsoil)
            self.adLambda += (Ts - Tsoil) * self.adG
            self.adTs += Lambda * self.adG
            self.adTsoil += Lambda * -self.adG
            self.adG = 0
            #statement dH    = model.rho * model.cp * ((Ts - theta) * (-1) * (ra)**(-2) * dra + 1 / ra * (dTs - self.dtheta))
            self.adra += model.rho * model.cp * (Ts - theta) * (-1) * (ra)**(-2) * self.adH
            self.adTs += model.rho * model.cp * 1 / ra * self.adH
            self.adtheta += model.rho * model.cp * 1 / ra * -self.adH
            self.adH = 0
            #statement dLE   = dLEsoil + dLEveg + dLEliq
            self.adLEsoil += self.adLE
            self.adLEveg += self.adLE
            self.adLEliq += self.adLE
            self.adLE = 0
            #statement dWltend = - dLEliq / (model.rhow * model.Lv)
            self.adLEliq += - 1 / (model.rhow * model.Lv) * self.adWltend
            self.adWltend = 0
            #statement dLEsoil = dp1_LEsoil * p2_LEveg_liq_soil + p1_LEsoil * dp2_LEveg_liq_soil
            self.adp1_LEsoil += p2_LEveg_liq_soil * self.adLEsoil
            self.adp2_LEveg_liq_soil += p1_LEsoil * self.adLEsoil
            self.adLEsoil = 0
            #statement dp1_LEsoil = model.rho * model.Lv * (-self.dcveg / (ra + rssoil) + (1. - cveg) * (-1) * (ra + rssoil)**(-2) * (dra + drssoil))
            self.adcveg += model.rho * model.Lv * -1 / (ra + rssoil) * self.adp1_LEsoil
            self.adra += model.rho * model.Lv * (1. - cveg) * (-1) * (ra + rssoil)**(-2) * self.adp1_LEsoil
            self.adrssoil += model.rho * model.Lv * (1. - cveg) * (-1) * (ra + rssoil)**(-2) * self.adp1_LEsoil
            self.adp1_LEsoil = 0
            #statement dLEliq = dp1_LEliq * p2_LEveg_liq_soil + p1_LEliq * dp2_LEveg_liq_soil
            self.adp1_LEliq += p2_LEveg_liq_soil * self.adLEliq
            self.adp2_LEveg_liq_soil += p1_LEliq * self.adLEliq 
            self.adLEliq = 0
            #statement dp1_LEliq = model.rho * model.Lv * (dcliq * cveg / ra + cliq / ra * self.dcveg + cliq * cveg * (-1) * ra**(-2) * dra)
            self.adcliq += model.rho * model.Lv * cveg / ra * self.adp1_LEliq
            self.adcveg += model.rho * model.Lv * cliq / ra * self.adp1_LEliq
            self.adra += model.rho * model.Lv * cliq * cveg * (-1) * ra**(-2) * self.adp1_LEliq 
            self.adp1_LEliq = 0
            #statement dLEveg = dp1_LEveg * p2_LEveg_liq_soil + p1_LEveg * dp2_LEveg_liq_soil
            self.adp1_LEveg += p2_LEveg_liq_soil * self.adLEveg
            self.adp2_LEveg_liq_soil += p1_LEveg * self.adLEveg
            self.adLEveg = 0
            #statement dp2_LEveg_liq_soil = ddqsatdT_dtheta * (Ts - theta) + dqsatdT * (dTs - self.dtheta) + dqsatvar - self.dq
            #dqsatdT is not a derivative here
            if self.manualadjointtesting:
                self.adp2_LEveg_liq_soil = self.y
            self.addqsatdT_dtheta += (Ts - theta) * self.adp2_LEveg_liq_soil
            self.adTs += dqsatdT * self.adp2_LEveg_liq_soil
            self.adtheta += dqsatdT * -self.adp2_LEveg_liq_soil
            self.adqsatvar += self.adp2_LEveg_liq_soil
            self.adq += -self.adp2_LEveg_liq_soil
            self.adp2_LEveg_liq_soil = 0
            #statement dp1_LEveg = model.rho * model.Lv * (- dcliq * cveg / (ra + rs) + (1 - cliq) / (ra + rs) * self.dcveg + (1 - cliq) * cveg * (-1) * (ra + rs)**(-2) * ((dra + drs)))
            self.adcliq += model.rho * model.Lv * cveg / (ra + rs) * -self.adp1_LEveg 
            self.adcveg += model.rho * model.Lv * (1 - cliq) / (ra + rs) * self.adp1_LEveg
            self.adra += model.rho * model.Lv * (1 - cliq) * cveg * (-1) * (ra + rs)**(-2) * self.adp1_LEveg
            self.adrs += model.rho * model.Lv * (1 - cliq) * cveg * (-1) * (ra + rs)**(-2) * self.adp1_LEveg
            self.adp1_LEveg = 0
            #statement dqsatsurf = dqsat_dT(Ts,model.Ps,dTs)
            self.adTs += dqsat_dT(Ts,model.Ps,self.adqsatsurf)
            self.adqsatsurf = 0
            #statement desatsurf      = desat(Ts,dTs)
            self.adTs += desat(Ts,self.adesatsurf)
            self.adesatsurf  = 0
            #statement dTs = dnumerator_Ts / denominator_Ts + numerator_Ts * (-1) * denominator_Ts**(-2) * ddenominator_Ts
            self.adnumerator_Ts += 1 / denominator_Ts * self.adTs
            self.addenominator_Ts += numerator_Ts * (-1) * denominator_Ts**(-2) * self.adTs
            self.adTs = 0
            #statement ddenominator_Ts = dp1_denominator_Ts + dp2_denominator_Ts + (1. - cveg) * dp3_denominator_Ts + p3_denominator_Ts * -self.dcveg + cveg * dp4_denominator_Ts + p4_denominator_Ts * self.dcveg + self.dLambda
            self.adp1_denominator_Ts += self.addenominator_Ts
            self.adp2_denominator_Ts += self.addenominator_Ts
            self.adp3_denominator_Ts += (1. - cveg) * self.addenominator_Ts
            self.adcveg += p3_denominator_Ts * -self.addenominator_Ts
            self.adp4_denominator_Ts += cveg * self.addenominator_Ts
            self.adcveg += p4_denominator_Ts * self.addenominator_Ts
            self.adLambda += self.addenominator_Ts
            self.addenominator_Ts = 0
            #statement dp4_denominator_Ts = model.rho * model.Lv * (dcliq * 1 / ra * dqsatdT + cliq * dqsatdT * (-1) * ra**(-2) * dra + cliq / ra * ddqsatdT_dtheta)
            #dqsatdT is not a derivative here
            self.adcliq += model.rho * model.Lv * 1 / ra * dqsatdT * self.adp4_denominator_Ts
            self.adra += model.rho * model.Lv * cliq * dqsatdT * (-1) * ra**(-2) * self.adp4_denominator_Ts
            self.addqsatdT_dtheta += model.rho * model.Lv * cliq / ra * self.adp4_denominator_Ts
            self.adp4_denominator_Ts = 0
            #statement dp3_denominator_Ts = model.rho * model.Lv * (dqsatdT * (-1) * (ra + rssoil)**(-2) * (dra + drssoil) + 1 / (ra + rssoil) * ddqsatdT_dtheta)
            #dqsatdT is not a derivative here
            self.adra += model.rho * model.Lv * dqsatdT * (-1) * (ra + rssoil)**(-2) * self.adp3_denominator_Ts
            self.adrssoil += model.rho * model.Lv * dqsatdT * (-1) * (ra + rssoil)**(-2) * self.adp3_denominator_Ts
            self.addqsatdT_dtheta += model.rho * model.Lv * 1 / (ra + rssoil) * self.adp3_denominator_Ts
            self.adp3_denominator_Ts = 0
            #statement dp2_denominator_Ts = self.dcveg * (1. - cliq) * model.rho * model.Lv / (ra + rs) * dqsatdT + cveg * model.rho * model.Lv / (ra + rs) * dqsatdT * -dcliq + cveg * (1. - cliq) * model.rho * model.Lv * (dqsatdT * (-1) * (ra + rs)**(-2) * (dra + drs) + 1 / (ra + rs) * ddqsatdT_dtheta)
            #dqsatdT is not a derivative here
            self.adcveg += (1. - cliq) * model.rho * model.Lv / (ra + rs) * dqsatdT * self.adp2_denominator_Ts
            self.adcliq += cveg * model.rho * model.Lv / (ra + rs) * dqsatdT * -self.adp2_denominator_Ts
            self.adra += cveg * (1. - cliq) * model.rho * model.Lv * dqsatdT * (-1) * (ra + rs)**(-2) * self.adp2_denominator_Ts
            self.adrs += cveg * (1. - cliq) * model.rho * model.Lv * dqsatdT * (-1) * (ra + rs)**(-2) * self.adp2_denominator_Ts
            self.addqsatdT_dtheta += cveg * (1. - cliq) * model.rho * model.Lv * 1 / (ra + rs) * self.adp2_denominator_Ts
            self.adp2_denominator_Ts = 0
            #statement dp1_denominator_Ts = model.rho * model.cp * (-1) * ra**(-2) * dra
            self.adra += model.rho * model.cp * (-1) * ra**(-2) * self.adp1_denominator_Ts
            self.adp1_denominator_Ts = 0
            #statement dnumerator_Ts = self.dQ + dp1_numerator_Ts + dp2_numerator_Ts * p3_numerator_Ts + p2_numerator_Ts * dp3_numerator_Ts + (1. - cveg) * dp4_numerator_Ts + p4_numerator_Ts * - self.dcveg + cveg * dp5_numerator_Ts + self.dcveg * p5_numerator_Ts + self.dLambda * Tsoil + Lambda * self.dTsoil
            self.adQ += self.adnumerator_Ts
            self.adp1_numerator_Ts += self.adnumerator_Ts
            self.adp2_numerator_Ts += p3_numerator_Ts * self.adnumerator_Ts
            self.adp3_numerator_Ts += p2_numerator_Ts * self.adnumerator_Ts
            self.adp4_numerator_Ts += (1. - cveg) * self.adnumerator_Ts
            self.adcveg += p4_numerator_Ts * -self.adnumerator_Ts
            self.adp5_numerator_Ts += cveg * self.adnumerator_Ts
            self.adcveg += p5_numerator_Ts * self.adnumerator_Ts
            self.adLambda += Tsoil * self.adnumerator_Ts
            self.adTsoil += Lambda * self.adnumerator_Ts
            self.adnumerator_Ts = 0
            #statement dp5_numerator_Ts = dcliq * model.rho * model.Lv /  ra * (dqsatdT * theta - qsatvar + q) + cliq * model.rho * model.Lv * ((dqsatdT * theta - qsatvar + q)*(-1)*ra**(-2)*dra + 1 / ra * (ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq))
            #dqsatdT is not a derivative here, dqsatvar is, also in other statements
            self.adcliq += model.rho * model.Lv /  ra * (dqsatdT * theta - qsatvar + q) * self.adp5_numerator_Ts
            self.adra += cliq * model.rho * model.Lv * (dqsatdT * theta - qsatvar + q)*(-1)*ra**(-2) * self.adp5_numerator_Ts
            self.addqsatdT_dtheta += cliq * model.rho * model.Lv * 1 / ra * theta * self.adp5_numerator_Ts
            self.adtheta += cliq * model.rho * model.Lv * 1 / ra * dqsatdT * self.adp5_numerator_Ts
            self.adqsatvar += cliq * model.rho * model.Lv * 1 / ra * -self.adp5_numerator_Ts
            self.adq += cliq * model.rho * model.Lv * 1 / ra * self.adp5_numerator_Ts
            self.adp5_numerator_Ts = 0
            #statement dp4_numerator_Ts = model.rho * model.Lv * ((dqsatdT * theta - qsatvar + q)*(-1)*(ra + rssoil)**(-2)*(dra + drssoil) + 1 / (ra + rssoil) * (ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq))
            self.adra += model.rho * model.Lv * (dqsatdT * theta - qsatvar + q)*(-1)*(ra + rssoil)**(-2)*self.adp4_numerator_Ts
            self.adrssoil += model.rho * model.Lv * (dqsatdT * theta - qsatvar + q)*(-1)*(ra + rssoil)**(-2)*self.adp4_numerator_Ts
            self.addqsatdT_dtheta += model.rho * model.Lv * 1 / (ra + rssoil) * theta * self.adp4_numerator_Ts
            self.adtheta += model.rho * model.Lv * 1 / (ra + rssoil) * dqsatdT * self.adp4_numerator_Ts
            self.adqsatvar += model.rho * model.Lv * 1 / (ra + rssoil) * -self.adp4_numerator_Ts
            self.adq += model.rho * model.Lv * 1 / (ra + rssoil) * self.adp4_numerator_Ts
            self.adp4_numerator_Ts = 0
            #statement dp3_numerator_Ts = ddqsatdT_dtheta * theta + self.dtheta * dqsatdT - dqsatvar + self.dq
            self.addqsatdT_dtheta += theta * self.adp3_numerator_Ts
            self.adtheta += dqsatdT * self.adp3_numerator_Ts
            self.adqsatvar += -self.adp3_numerator_Ts
            self.adq += self.adp3_numerator_Ts
            self.adp3_numerator_Ts = 0
            #statement dp2_numerator_Ts = self.dcveg * (1. - cliq) * model.rho * model.Lv / (ra + rs) + cveg * - dcliq * model.rho * model.Lv / (ra + rs) + cveg * (1. - cliq) * model.rho * model.Lv * (-1) * (ra + rs)**(-2) * (dra + drs)
            self.adcveg += (1. - cliq) * model.rho * model.Lv / (ra + rs) * self.adp2_numerator_Ts
            self.adcliq += cveg * -1 * model.rho * model.Lv / (ra + rs) * self.adp2_numerator_Ts
            self.adra += cveg * (1. - cliq) * model.rho * model.Lv * (-1) * (ra + rs)**(-2) * self.adp2_numerator_Ts
            self.adrs += cveg * (1. - cliq) * model.rho * model.Lv * (-1) * (ra + rs)**(-2) * self.adp2_numerator_Ts
            self.adp2_numerator_Ts = 0
            #statement dp1_numerator_Ts = model.rho * model.cp *(self.dtheta / ra + theta * (-1) * ra**(-2) * dra) 
            self.adtheta += model.rho * model.cp / ra * self.adp1_numerator_Ts
            self.adra += model.rho * model.cp * theta * (-1) * ra**(-2) * self.adp1_numerator_Ts
            self.adp1_numerator_Ts = 0
            if Wl / Wlmx <= 1: #cliq = min(1., self.Wl / Wlmx)
                #statement dcliq = self.dWl / Wlmx + Wl * (-1) * Wlmx ** (-2) * dWlmx
                self.adWl += 1 / Wlmx * self.adcliq
                self.adWlmx += Wl * (-1) * Wlmx ** (-2) * self.adcliq
                self.adcliq = 0
            else:
                #statement dcliq = 0
                self.adcliq = 0
            #statement dWlmx = self.dLAI * Wmax + LAI * self.dWmax
            self.adLAI += Wmax * self.adWlmx
            self.adWmax += LAI * self.adWlmx
            self.adWlmx = 0
        else:
            raise Exception('part not yet implemented')
        #statement drssoil = rssoilmin * df2 + f2 * self.drssoilmin
        self.adf2 += rssoilmin * self.adrssoil
        self.adrssoilmin += f2 * self.adrssoil
        self.adrssoil = 0
        if(wg > wwilt):
            #statement df2 = (self.dwfc - self.dwwilt) / (wg - wwilt) + (wfc - wwilt) * -1 * (wg - wwilt)**(-2) * (self.dwg - self.dwwilt)
            self.adwfc += 1 / (wg - wwilt) * self.adf2
            self.adwwilt += - 1 / (wg - wwilt) * self.adf2
            self.adwg += (wfc - wwilt) * -1 * (wg - wwilt)**(-2) * self.adf2
            self.adwwilt += (wfc - wwilt) * -1 * (wg - wwilt)**(-2) * -self.adf2
            self.adf2 = 0
        else:
            #statement df2        = 0
            self.adf2        = 0 
        if(self.model.ls_type == 'js'): 
            #statement drs = self.tl_jarvis_stewart(model,checkpoint,returnvariable='drs')
            self.adj_jarvis_stewart(forcing,checkpoint,model)
        elif(self.model.ls_type == 'ags'):
            #statement drs = self.tl_ags(model,checkpoint,returnvariable='drs')
            self.adj_ags(forcing,checkpoint,model)
            #self.adrs = 0 #not needed, since adrs set to zero in ags, and in the forward model ags does not return rs explicitly
        elif(self.model.ls_type == 'canopy_model'):
            raise Exception('not yet implemented')
        elif(self.ls_type == 'sib4'):
            raise Exception('not yet implemented')
        else:
            raise Exception('problem with ls switch')
        #statement de = model.Ps / 0.622 * self.dq
        self.adq += model.Ps / 0.622 * self.ade
        self.ade = 0
        #statement ddqsatdT_dtheta = 0.622 / model.Ps * ddesatdT_dtheta
        self.addesatdT_dtheta += 0.622 / model.Ps * self.addqsatdT_dtheta
        self.addqsatdT_dtheta = 0
        #statement ddesatdT_dtheta = desatvar * (17.2694 / (theta - 35.86) - 17.2694 * (theta - 273.16) / (theta - 35.86)**2.) + esatvar * ((17.2694 * -1 * (theta - 35.86)**(-2)) * self.dtheta + -1 * 17.2694 * (self.dtheta/(theta - 35.86)**2. + (theta - 273.16) * (-2) * (theta - 35.86)**(-3) * self.dtheta))
        self.adesatvar += (17.2694 / (theta - 35.86) - 17.2694 * (theta - 273.16) / (theta - 35.86)**2.) * self.addesatdT_dtheta
        self.adtheta += esatvar * (17.2694 * -1 * (theta - 35.86)**(-2)) * self.addesatdT_dtheta
        self.adtheta += esatvar * -1 * 17.2694 * self.addesatdT_dtheta/(theta - 35.86)**2.
        self.adtheta += esatvar * -1 * 17.2694 * (theta - 273.16) * (-2) * (theta - 35.86)**(-3) * self.addesatdT_dtheta
        self.addesatdT_dtheta = 0
        #statement dqsatvar    = dqsat_dT(theta,model.Ps,self.dtheta)
        self.adtheta += dqsat_dT(theta,model.Ps,self.adqsatvar)
        self.adqsatvar = 0
        #statement desatvar    = desat(theta,self.dtheta)
        self.adtheta += desat(theta,self.adesatvar)
        self.adesatvar = 0
        if(model.sw_sl):
            #statement dra = -1 * (Cs * ueff)**-2. *(self.dCs * ueff + Cs * dueff)
            self.adCs += -1 * (Cs * ueff)**-2. * ueff * self.adra
            self.adueff += -1 * (Cs * ueff)**-2. * Cs * self.adra
            self.adra = 0
        else:
            if ustar >= 1.e-3:
                #statement dra = dueff / ustar**2. + ueff * -2 * ustar**(-3) * self.dustar
                self.adueff +=  self.adra / ustar**2.
                self.adustar += ueff * -2 * ustar**(-3) * self.adra
                self.adra = 0
            else:
                #statement dra = dueff / (1.e-3)**2.
                self.adueff += self.adra / (1.e-3)**2.
                self.adra = 0
        #statement dueff = 0.5*(u ** 2. + v ** 2. + wstar**2.)**(-1/2) * (2 * u * self.du + 2 * v * self.dv + 2 * wstar * self.dwstar)
        self.adu += 0.5*(u ** 2. + v ** 2. + wstar**2.)**(-1/2) * 2 * u * self.adueff
        self.adv += 0.5*(u ** 2. + v ** 2. + wstar**2.)**(-1/2) * 2 * v * self.adueff
        self.adwstar += 0.5*(u ** 2. + v ** 2. + wstar**2.)**(-1/2) * 2 * wstar * self.adueff
        self.adueff = 0
        if self.manualadjointtesting:
            self.HTy = self.adwstar,self.adCs,self.adtheta,self.adq,self.adwfc,self.adwwilt,self.adwg,self.adLAI,self.adWmax,self.adWl,self.adcveg,self.adQ,self.adLambda,self.adTsoil,self.adrsmin,self.adwsat,self.adw2,self.adT2
        if self.adjointtestingrun_land_surface:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    def adj_run_soil_COS_mod(self,forcing,checkpoint,model,HTy_variables=None):
        sCOSm = model.soilCOSmodel #just for shorter notation
        airtemp = checkpoint['rsCm_airtemp']
        mol_rat_ocs_atm = checkpoint['rsCm_mol_rat_ocs_atm']
        T_nodes = checkpoint['rsCm_T_nodes_end']
        s_moist = checkpoint['rsCm_s_moist_end']
        Q10 = checkpoint['rsCm_Q10']
        SunTref = checkpoint['rsCm_SunTref']
        Vspmax = checkpoint['rsCm_Vspmax']
        wsat = checkpoint['rsCm_wsat']
        diffus_nodes = checkpoint['rsCm_diffus_nodes_end']
        D_a_0 = checkpoint['rsCm_D_a_0_end']
        C_air = checkpoint['rsCm_C_air_end']
        conduct = checkpoint['rsCm_conduct_end']
        dt = checkpoint['rsCm_dt']
        A_matr = checkpoint['rsCm_A_matr_end']
        B_matr = checkpoint['rsCm_B_matr_end']
        matr_3_eq12 = checkpoint['rsCm_matr_3_eq12_end']
        invmatreq12 = checkpoint['rsCm_invmatreq12_end']
        kH = checkpoint['rsCm_kH_end']
        
        C_soilair_current = checkpoint['rsCm_C_soilair_current']
        Rgas = sCOSm.Rgas
        pressure = model.Ps
        #statement dC_soilair_current = cp.deepcopy(dC_soilair_next)
        self.adC_soilair_next += cp.deepcopy(self.adC_soilair_current)
        self.adC_soilair_current[:] = 0
        #statement dCOS_netuptake_soilsun = -1 * dOCS_fluxes[0]
        self.adOCS_fluxes[0] += -1 * self.adCOS_netuptake_soilsun
        self.adCOS_netuptake_soilsun = 0
        #statement dOCS_fluxes[0] = -1. * (dconduct[0] * (C_soilair[0] - C_air) + conduct[0] * (self.dC_soilair_current[0] - dC_air))
        C_soilair = checkpoint['rsCm_C_soilair_calcJ']
        self.adconduct[0] += -1. * (C_soilair[0] - C_air) * self.adOCS_fluxes[0]
        self.adC_soilair_current[0] += -1. * conduct[0] * self.adOCS_fluxes[0]
        self.adC_air += conduct[0] * self.adOCS_fluxes[0]
        self.adOCS_fluxes[0] = 0
        for i in range(sCOSm.nr_nodes-1,0,-1):
            #statement dOCS_fluxes[i] = -1. * (dconduct[i] * (C_soilair[i] - C_soilair[i-1]) + conduct[i] * (self.dC_soilair_current[i] - self.dC_soilair_current[i-1]))
            self.adconduct[i] += -1. * (C_soilair[i] - C_soilair[i-1]) * self.adOCS_fluxes[i]
            self.adC_soilair_current[i] += -1. * conduct[i] * self.adOCS_fluxes[i]
            self.adC_soilair_current[i-1] += conduct[i] * self.adOCS_fluxes[i]
            self.adOCS_fluxes[i] = 0
        #statement dC_soilair_next = cp.deepcopy(dC_soilair)
        self.adC_soilair += self.adC_soilair_next
        self.adC_soilair_next[:] = 0
        C_soilair = checkpoint['rsCm_C_soilair_middle']
        for i in range(sCOSm.nr_nodes-1,-1,-1):
            if (C_soilair[i] < 0.0):
                #statement dC_soilair[i] = 0.0
                self.adC_soilair[i] = 0.0
        #statement dC_soilair = np.matmul(dinvmatreq12,matr_3_eq12) + np.matmul(invmatreq12,dmatr_3_eq12)
        self.adinvmatreq12 += np.matmul(np.transpose(np.array([self.adC_soilair])),np.array([matr_3_eq12])) #transpose and numpy array to make shapes fit, np.matmul(self.adC_soilair,matr_3_eq12) would return a scalar
        self.admatr_3_eq12 += np.matmul(invmatreq12,self.adC_soilair)
        self.adC_soilair[:] = 0
        #statement dmatr_3_eq12 = dmatr_2_eq12 + 2*dt* dsource
        self.admatr_2_eq12 += self.admatr_3_eq12
        self.adsource += 2*dt* self.admatr_3_eq12
        self.admatr_3_eq12[:] = 0
        #statement dmatr_2_eq12 = np.matmul((2*dA_matr + dt * dB_matr),(C_soilair_current)) + np.matmul((2*A_matr + dt * B_matr),self.dC_soilair_current)
        #vector multiplication is commutative (if you transpose properly), matrix multiplication is not
        #ORIG: self.adA_matr += np.matmul((2*self.admatr_2_eq12),(C_soilair_current))
        self.adA_matr += np.matmul(np.transpose(np.array([2*self.admatr_2_eq12])),np.array([C_soilair_current]))
        #np.matmul((2*self.admatr_2_eq12),(C_soilair_current))) gives a scalar
        self.adB_matr += np.matmul(np.transpose(np.array([dt * self.admatr_2_eq12])),np.array([C_soilair_current])) #distributivity of matrix multiplication
        #self.adC_soilair_current += np.matmul(2*A_matr + dt * B_matr,self.admatr_2_eq12)
        self.adC_soilair_current += np.matmul(np.transpose(2*A_matr + dt * B_matr),self.admatr_2_eq12) #see notes matrix stuff, but A and B are symmetric matrices, trnsposing does not matter
        self.admatr_2_eq12[:] = 0
        #statement dinvmatreq12 = np.matmul(-1*np.linalg.inv(2*A_matr - dt*B_matr),np.matmul((2*dA_matr - dt*dB_matr),np.linalg.inv(2*A_matr - dt*B_matr)))
        try:
            self.adA_matr += np.matmul(-1*np.linalg.inv(2*A_matr - dt*B_matr),np.matmul((2*self.adinvmatreq12),np.linalg.inv(2*A_matr - dt*B_matr)))
            self.adB_matr += np.matmul(-1*np.linalg.inv(2*A_matr - dt*B_matr),np.matmul((- dt*self.adinvmatreq12),np.linalg.inv(2*A_matr - dt*B_matr))) # [A(B+C)] * D = ABD + ACD, distributivity
        except (np.linalg.linalg.LinAlgError):
            print('np.linalg.linalg.LinAlgError in adjoint modelling file')
            self.adA_matr[:] = np.nan
            self.adB_matr[:] = np.nan
        self.adinvmatreq12[:] = 0
        for i in range(sCOSm.nr_nodes-1,0,-1):
            #statement dB_matr[i,i-1] = dconduct[i]
            self.adconduct[i] += self.adB_matr[i,i-1]
            self.adB_matr[i,i-1] = 0
        #statement dB_matr[sCOSm.nr_nodes-1,sCOSm.nr_nodes-1] = -dconduct[sCOSm.nr_nodes-1]
        self.adconduct[sCOSm.nr_nodes-1] += -self.adB_matr[sCOSm.nr_nodes-1,sCOSm.nr_nodes-1]
        self.adB_matr[sCOSm.nr_nodes-1,sCOSm.nr_nodes-1] = 0
        for i in range(sCOSm.nr_nodes-2,-1,-1):
            #statement dB_matr[i,i+1] = dconduct[i+1]
            self.adconduct[i+1] += self.adB_matr[i,i+1]
            self.adB_matr[i,i+1] = 0
            #statement dB_matr[i,i] = -(dconduct[i]+dconduct[i+1])
            self.adconduct[i] += -self.adB_matr[i,i]
            self.adconduct[i+1] += -self.adB_matr[i,i]
            self.adB_matr[i,i] = 0
        for i in range(sCOSm.nr_nodes-1,-1,-1):
            #statement dA_matr[i,i] = deta[i] * sCOSm.dz_soil[i]
            self.adeta[i] += sCOSm.dz_soil[i] * self.adA_matr[i,i]
            self.adA_matr[i,i] = 0
        #statement deta = dkH * s_moist + kH * ds_moist + (self.dwsat - ds_moist)
        #!!np.sum because dwsat is a scalar. If we would have written the tangent linear in a for loop we would also add something to adwsat for every node
        self.adkH += s_moist * self.adeta
        self.ads_moist += kH * self.adeta
        self.adwsat += np.sum(self.adeta)
        self.ads_moist += -self.adeta
        self.adeta[:] = 0
        #statement dsource[0] = (ds_uptake[0]+ds_prod[0])*sCOSm.dz_soil[0] + 1 / sCOSm.z_soil[0] * (dD_a_0 * C_air + D_a_0 * dC_air)
        self.ads_uptake[0] += sCOSm.dz_soil[0] * self.adsource[0]
        self.ads_prod[0] += sCOSm.dz_soil[0] * self.adsource[0]
        self.adD_a_0 += 1 / sCOSm.z_soil[0] * C_air * self.adsource[0]
        self.adC_air += 1 / sCOSm.z_soil[0] * D_a_0 * self.adsource[0]
        self.adsource[0] = 0
        #statement dD_a_0 = ddiffus[0]
        self.addiffus[0] += self.adD_a_0
        self.adD_a_0 = 0
        #statement dsource = (ds_uptake+ds_prod)*sCOSm.dz_soil
        self.ads_uptake += sCOSm.dz_soil * self.adsource
        self.ads_prod += sCOSm.dz_soil * self.adsource
        self.adsource[:] = 0
        #statement dconduct[0] = ddiffus[0] / (sCOSm.z_soil[0] - 0)
        self.addiffus[0] += 1 / (sCOSm.z_soil[0] - 0) * self.adconduct[0]
        self.adconduct[0] = 0
        for i in range(sCOSm.nr_nodes-1,0,-1):    
            #statement dconduct[i] = ddiffus[i] / (sCOSm.z_soil[i] - sCOSm.z_soil[i-1])
            self.addiffus[i] += 1 / (sCOSm.z_soil[i] - sCOSm.z_soil[i-1]) * self.adconduct[i]
            self.adconduct[i] = 0
        if sCOSm.Diffus_type == ('Sun'):
            Dm = checkpoint['rsCm_Dm_end']
            n = checkpoint['rsCm_n_end']
            b_sCOSm = checkpoint['rsCm_b_sCOSm_end']
            D_a = checkpoint['rsCm_D_a_end']
            #statement db_sCOSm = self.db_sCOSm does not need adjoint statement (since we could just have written self.db_sCOSm everywhere)
            for i in range(sCOSm.nr_nodes-1,0,-1): #0, because it should count down to (including) i=1
                #statement ddiffus[i] = (ddiffus_nodes[i] + ddiffus_nodes[i-1])/2.
                self.addiffus_nodes[i] += 1/2 * self.addiffus[i]
                self.addiffus_nodes[i-1] += 1/2 * self.addiffus[i]
                self.addiffus[i] = 0
            #statement ddiffus[0] = 2. * -1 * (1./diffus_nodes[0] + 1./D_a)**(-2) * (-1 * (diffus_nodes[0])**(-2) * ddiffus_nodes[0] + -1 * D_a**(-2) * dD_a)
            self.addiffus_nodes[0] += 2. * -1 * (1./diffus_nodes[0] + 1./D_a)**(-2) * (-1 * (diffus_nodes[0])**(-2)) * self.addiffus[0]
            self.adD_a += 2. * -1 * (1./diffus_nodes[0] + 1./D_a)**(-2) * (-1 * D_a**(-2)) * self.addiffus[0]
            self.addiffus[0] = 0
            #statement dD_a = Dm * n * (airtemp/SunTref)**(n-1) * self.dairtemp/SunTref
            self.adairtemp += Dm * n * (airtemp/SunTref)**(n-1) / SunTref * self.adD_a
            self.adD_a = 0
            #statement ddiffus_nodes = ddiffus_nodes_dwsat + ddiffus_nodes_ds_moist + ddiffus_nodes_db_sCOSm + ddiffus_nodes_dT_nodes
            self.addiffus_nodes_dwsat += self.addiffus_nodes
            self.addiffus_nodes_ds_moist += self.addiffus_nodes
            self.addiffus_nodes_db_sCOSm += self.addiffus_nodes
            self.addiffus_nodes_dT_nodes += self.addiffus_nodes
            self.addiffus_nodes[:] = 0
            #statement ddiffus_nodes_dT_nodes = Dm * (wsat - s_moist)**2 * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * n * (T_nodes / SunTref)**(n-1) * 1 / SunTref * dT_nodes
            self.adT_nodes += Dm * (wsat - s_moist)**2 * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * n * (T_nodes / SunTref)**(n-1) * 1 / SunTref * self.addiffus_nodes_dT_nodes
            self.addiffus_nodes_dT_nodes[:] = 0
            #statement ddiffus_nodes_db_sCOSm = Dm * (wsat - s_moist)**2 * np.log((wsat - s_moist)/wsat) * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * 3 * -1 * b_sCOSm**(-2) * db_sCOSm * (T_nodes/SunTref)**n
            #!!np.sum because db_sCOSm is a scalar. If we would have written the tangent linear in a for loop we would also add something to adb_sCOSm for every node
            self.adb_sCOSm += np.sum(Dm * (wsat - s_moist)**2 * np.log((wsat - s_moist)/wsat) * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * 3 * -1 * b_sCOSm**(-2) * (T_nodes/SunTref)**n * self.addiffus_nodes_db_sCOSm) 
            self.addiffus_nodes_db_sCOSm[:] = 0
            #statement ddiffus_nodes_ds_moist = Dm * (T_nodes / SunTref)**n * (2 * (wsat - s_moist) * - ds_moist * ((wsat - s_moist)/wsat)**(3./b_sCOSm) + (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * - ds_moist/wsat)
            self.ads_moist += Dm * (T_nodes / SunTref)**n * 2 * (wsat - s_moist) * ((wsat - s_moist)/wsat)**(3./b_sCOSm) * - self.addiffus_nodes_ds_moist
            self.ads_moist += Dm * (T_nodes / SunTref)**n * (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * - self.addiffus_nodes_ds_moist/wsat
            self.addiffus_nodes_ds_moist[:] = 0
            #statement ddiffus_nodes_dwsat = Dm * (T_nodes / SunTref)**n * (2 * (wsat - s_moist) * self.dwsat * ((wsat - s_moist)/wsat)**(3./b_sCOSm) + (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * (1 / wsat * self.dwsat + (wsat - s_moist) * -1 * wsat**(-2) * self.dwsat))
            self.adwsat += np.sum(Dm * (T_nodes / SunTref)**n * 2 * (wsat - s_moist) * self.addiffus_nodes_dwsat * ((wsat - s_moist)/wsat)**(3./b_sCOSm))
            self.adwsat += np.sum(Dm * (T_nodes / SunTref)**n * (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * 1 / wsat * self.addiffus_nodes_dwsat)
            self.adwsat += np.sum(Dm * (T_nodes / SunTref)**n * (wsat - s_moist)**2 * (3./b_sCOSm) * ((wsat - s_moist)/wsat)**(3./b_sCOSm - 1) * (wsat - s_moist) * -1 * wsat**(-2) * self.addiffus_nodes_dwsat)
            self.addiffus_nodes_dwsat[:] = 0
            #statement db_sCOSm = self.db_sCOSm does not need anything
        elif sCOSm.Diffus_type == ('Ogee'):
            raise Exception('Ogee diffusion not yet implemented')
        else:
            raise Exception('Error in Diffus_type switch inputdata')
        #statement ds_prod = self.dVspmax * Q10 **((T_nodes-SunTref)/10.0) + Vspmax * (((T_nodes - SunTref)/10.0) * Q10 **((T_nodes - SunTref)/10.0 - 1) * self.dQ10 + np.log(Q10) * Q10 **((T_nodes-SunTref)/10.0) * dT_nodes/10.0)
        self.adVspmax += np.sum(Q10 **((T_nodes-SunTref)/10.0) * self.ads_prod)
        self.adQ10 += np.sum(Vspmax * ((T_nodes - SunTref)/10.0) * Q10 **((T_nodes - SunTref)/10.0 - 1) * self.ads_prod)
        self.adT_nodes += Vspmax * np.log(Q10) * Q10 **((T_nodes-SunTref)/10.0) * self.ads_prod/10.0                         
        self.ads_prod[:] = 0
        if 	(sCOSm.uptakemodel == 'Sun'):
            raise Exception ('Sun uptake not implemented in TL')
        elif sCOSm.uptakemodel == 'Ogee':
            deltaHa = checkpoint['rsCm_deltaHa_end']
            deltaHd = checkpoint['rsCm_deltaHd_end']
            deltaSd = checkpoint['rsCm_deltaSd_end']
            xCA = checkpoint['rsCm_xCA_end']
            fCA = checkpoint['rsCm_fCA']
            kuncat_ref = checkpoint['rsCm_kuncat_ref_end']
            xCAref = checkpoint['rsCm_xCAref_end']
            ktot = checkpoint['rsCm_ktot_end']
            #statement ds_uptake = (s_moist * C_soilair_current) * (-dktot * kH - ktot * dkH) + (-ktot * kH) * (ds_moist * C_soilair_current + s_moist * self.dC_soilair_current)
            self.adktot += (s_moist * C_soilair_current) * - kH * self.ads_uptake
            self.adkH += (s_moist * C_soilair_current) * - ktot * self.ads_uptake
            self.ads_moist += (-ktot * kH) * C_soilair_current * self.ads_uptake
            self.adC_soilair_current += (-ktot * kH) * s_moist * self.ads_uptake
            self.ads_uptake[:] = 0
            #statement dktot = kuncat_ref/xCAref * (self.dfCA*xCA + fCA*dxCA)
            self.adfCA += np.sum(kuncat_ref/xCAref * xCA * self.adktot)
            self.adxCA += kuncat_ref/xCAref * fCA * self.adktot
            self.adktot[:] = 0
            #statement dxCA = np.exp(-deltaHa/(Rgas*T_nodes)) / (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas)) * -deltaHa/Rgas * -1 * T_nodes**(-2) * dT_nodes + np.exp(-deltaHa/(Rgas*T_nodes)) * -1 * (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas))**(-2) * np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas) * -deltaHd/Rgas* -1 * T_nodes**(-2) * dT_nodes
            self.adT_nodes += np.exp(-deltaHa/(Rgas*T_nodes)) / (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas)) * -deltaHa/Rgas * -1 * T_nodes**(-2) * self.adxCA
            self.adT_nodes += np.exp(-deltaHa/(Rgas*T_nodes)) * -1 * (1. + np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas))**(-2) * np.exp(-deltaHd/(Rgas*T_nodes) + deltaSd/Rgas) * -deltaHd/Rgas* -1 * T_nodes**(-2) * self.adxCA
            self.adxCA[:] = 0
        elif sCOSm.uptakemodel == 'newSun':
            raise Exception ('newSun uptake not implemented in TL')
        else:
            raise Exception('ERROR: Problem with uptake in switch inputdata')
        if sCOSm.kH_type == 'Sun':
            alfa = checkpoint['rsCm_alfa_end']
            beta = checkpoint['rsCm_beta_end']
            K_eq20 = checkpoint['rsCm_K_eq20_end']
            #statement dkH = (dT_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / T_nodes) + (T_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / T_nodes) * beta * K_eq20 * -1 *  T_nodes**(-2) * dT_nodes
            self.adT_nodes += 1 / K_eq20 * np.exp(alfa + beta * K_eq20 / T_nodes) * self.adkH
            self.adT_nodes += (T_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / T_nodes) * beta * K_eq20 * -1 *  T_nodes**(-2) * self.adkH
            self.adkH[:] = 0
        elif sCOSm.kH_type == 'Ogee':
            kHog = checkpoint['rsCm_kHog_end']
            #statement dkH = Rgas * 0.01 * (dkHog * T_nodes + kHog * dT_nodes)
            self.adkHog += Rgas * 0.01 * T_nodes * self.adkH
            self.adT_nodes += Rgas * 0.01 * kHog * self.adkH
            self.adkH[:] = 0
        else:
            raise Exception('ERROR: Problem in kH_type switch inputdata')
        for i in range(sCOSm.nr_nodes-1,-1,-1):
            if sCOSm.sw_soilmoisture == 'simple':
                if (sCOSm.z_soil[i] > sCOSm.layer1_2division):
                    #statement ds_moist[i] = self.dw2
                    self.adw2 += self.ads_moist[i]
                    self.ads_moist[i] = 0
                else:
                    #statement ds_moist[i] = self.dwg    
                    self.adwg += self.ads_moist[i]
                    self.ads_moist[i] = 0
            elif sCOSm.sw_soilmoisture == 'interpol':
                #statement ds_moist[i] = self.dwg + (sCOSm.z_soil[i] - 0) * (self.dw2 - self.dwg)/(1 - 0)
                self.adwg += self.ads_moist[i]
                self.adw2 += (sCOSm.z_soil[i] - 0) * self.ads_moist[i]
                self.adwg += (sCOSm.z_soil[i] - 0) * -self.ads_moist[i]
                self.ads_moist[i] = 0
            else:
                raise Exception('ERROR: Problem in soilmoisture switch inputdata')
            if sCOSm.sw_soiltemp == 'Sunpaper':
                #statement dT_nodes[i] = 0
                self.adT_nodes[i] = 0
            elif sCOSm.sw_soiltemp == 'simple':
                if (sCOSm.z_soil[i] > sCOSm.layer1_2division):
                    #statement dT_nodes[i] = self.dT2
                    self.adT2 += self.adT_nodes[i]
                    self.adT_nodes[i] = 0
                else:
                    #statement dT_nodes[i] = self.dTsoil
                    self.adTsoil += self.adT_nodes[i]
                    self.adT_nodes[i] = 0
            elif sCOSm.sw_soiltemp == 'interpol': ##y= y1 + (x-x1) * (y2-y1)/(x2-x1); y1 is Tsoil, x1 is 0
                #statement dT_nodes[i] = self.dTsoil + (sCOSm.z_soil[i] - 0) * (self.dT2 - self.dTsoil)/(1 - 0)
                self.adTsoil += self.adT_nodes[i]
                self.adT2 += (sCOSm.z_soil[i] - 0) * self.adT_nodes[i]
                self.adTsoil += (sCOSm.z_soil[i] - 0) * -self.adT_nodes[i]
                self.adT_nodes[i] = 0
            else:
                raise Exception('ERROR: Problem in soiltemp switch inputdata')
        #statement dC_air = 1.e-9 * pressure / Rgas * (self.dmol_rat_ocs_atm / airtemp + mol_rat_ocs_atm * (-1) * airtemp**(-2) * self.dairtemp)
        self.admol_rat_ocs_atm += 1.e-9 * pressure / Rgas / airtemp * self.adC_air
        self.adairtemp += 1.e-9 * pressure / Rgas * mol_rat_ocs_atm * (-1) * airtemp**(-2) * self.adC_air
        self.adC_air = 0
        
        if self.adjointtestingrun_soil_COS_mod:
            self.HTy_dict = {}
            for item in HTy_variables:
                self.HTy_dict[item] = self.__dict__[item]
    
    def adj_jarvis_stewart(self,forcing,checkpoint,model,HTy_variables=None):
        Swin = checkpoint['js_Swin']
        w2 = checkpoint['js_w2']
        wwilt = checkpoint['js_wwilt']
        wfc = checkpoint['js_wfc']
        gD = checkpoint['js_gD']
        e = checkpoint['js_e']
        esatvar = checkpoint['js_esatvar']
        theta = checkpoint['js_theta']
        LAI = checkpoint['js_LAI']
        f1 = checkpoint['js_f1_end']
        f3 = checkpoint['js_f3_end']
        f4 = checkpoint['js_f4_end']
        rsmin = checkpoint['js_rsmin']
        f2js = checkpoint['js_f2js_end']
        
        #statement drs = drs_drsmin + drs_dLAI + drs_df1 + drs_df2js + drs_df3 + drs_df4
        self.adrs_drsmin += self.adrs
        self.adrs_dLAI += self.adrs
        self.adrs_df1 += self.adrs
        self.adrs_df2js += self.adrs
        self.adrs_df3 += self.adrs
        self.adrs_df4 += self.adrs
        self.adrs = 0
        #statement drs_df4 = rsmin / LAI * f1 * f2js * f3 * df4
        self.adf4 += rsmin / LAI * f1 * f2js * f3 * self.adrs_df4
        self.adrs_df4 = 0
        #statement drs_df3 = rsmin / LAI * f1 * f2js * f4 * df3
        self.adf3 += rsmin / LAI * f1 * f2js * f4 * self.adrs_df3
        self.adrs_df3 = 0
        #statement drs_df2js = rsmin / LAI * f1 * f3 * f4 * df2js
        self.adf2js += rsmin / LAI * f1 * f3 * f4 * self.adrs_df2js
        self.adrs_df2js = 0
        #statement drs_df1 = rsmin / LAI * f2js * f3 * f4 * df1
        self.adf1 += rsmin / LAI * f2js * f3 * f4 * self.adrs_df1 
        self.adrs_df1 = 0
        #statement drs_dLAI = rsmin * f1 * f2js * f3 * f4 * -1 * LAI**(-2) * self.dLAI
        self.adLAI += rsmin * f1 * f2js * f3 * f4 * -1 * LAI**(-2) * self.adrs_dLAI
        self.adrs_dLAI = 0
        #statement drs_drsmin = self.drsmin / LAI * f1 * f2js * f3 * f4 
        self.adrsmin += 1 / LAI * f1 * f2js * f3 * f4 * self.adrs_drsmin
        self.adrs_drsmin = 0
        #statement df4 = -1 * (1. - 0.0016 * (298.0-theta)**2.)**(-2) * - 0.0016 * 2 * (298.0 - theta) * -1 * self.dtheta
        self.adtheta += -1 * (1. - 0.0016 * (298.0-theta)**2.)**(-2) * - 0.0016 * 2 * (298.0 - theta) * -1 * self.adf4
        self.adf4 = 0
        #statement df3 = -1 * (np.exp(- gD * (esatvar - e) / 100.))**-2 * np.exp(- gD * (esatvar - e) / 100.) * (-self.dgD  * (esatvar - e) / 100 - 1 / 100 * gD * self.desatvar + 1 / 100 * gD * self.de)
        self.adgD += -1 * (np.exp(- gD * (esatvar - e) / 100.))**-2 * np.exp(- gD * (esatvar - e) / 100.) * (esatvar - e) / 100 * -1 * self.adf3
        self.adesatvar += -1 * (np.exp(- gD * (esatvar - e) / 100.))**-2 * np.exp(- gD * (esatvar - e) / 100.) * -1 / 100 * gD * self.adf3
        self.ade += -1 * (np.exp(- gD * (esatvar - e) / 100.))**-2 * np.exp(- gD * (esatvar - e) / 100.) * 1 / 100 * gD * self.adf3
        self.adf3 = 0
        f2js = checkpoint['js_f2js_middle']
        if f2js < 1:
            #statement df2js = 0
            self.adf2js = 0
        if(w2 > wwilt):
            #statement df2js = df2js_dwfc + df2js_dwwilt + df2js_dw2
            self.adf2js_dwfc += self.adf2js
            self.adf2js_dwwilt += self.adf2js
            self.adf2js_dw2 += self.adf2js
            self.adf2js = 0
            #statement df2js_dw2 = (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * self.dw2
            self.adw2 += (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * self.adf2js_dw2
            self.adf2js_dw2 = 0
            #statement df2js_dwwilt = 1 / (w2 - wwilt) * -1 * self.dwwilt + (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * -1 * self.dwwilt
            self.adwwilt += 1 / (w2 - wwilt) * -1 * self.adf2js_dwwilt
            self.adwwilt += (wfc - wwilt) * -1 * (w2 - wwilt)**(-2) * -1 * self.adf2js_dwwilt
            self.adf2js_dwwilt = 0
            #statement df2js_dwfc = 1 / (w2 - wwilt) * self.dwfc
            self.adwfc += 1 / (w2 - wwilt) * self.adf2js_dwfc
            self.adf2js_dwfc = 0
        else:
            #statement df2js = 0
            self.adf2js = 0
        if(model.sw_rad):
            if (0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)) <= 1:
                #statement df1 = -1 * ((0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)))**(-2) * (0.004 * self.dSwin / (0.81 * (0.004 * Swin + 1.)) + (0.004 * Swin + 0.05) * -1 * (0.81 * (0.004 * Swin + 1.))**(-2) * 0.81 * 0.004 * self.dSwin)
                self.adSwin += -1 * ((0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)))**(-2) * 0.004 / (0.81 * (0.004 * Swin + 1.)) * self.adf1
                self.adSwin += -1 * ((0.004 * Swin + 0.05) / (0.81 * (0.004 * Swin + 1.)))**(-2) * (0.004 * Swin + 0.05) * -1 * (0.81 * (0.004 * Swin + 1.))**(-2) * 0.81 * 0.004 * self.adf1
                self.adf1 = 0
            else:
                # statement df1 = 0
                self.adf1 = 0
        else:
            # statement df1 = 0
            self.adf1 = 0
               
        if self.adjointtestingjarvis_stewart:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
    
    def adj_ags(self,forcing,checkpoint,model,HTy_variables=None):
        if(model.c3c4 == 'c3'):
            c = 0
        elif(model.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('option \"%s\" for \"c3c4\" invalid'%self.c3c4)

        #adwCO2, adwCO2A,  adwCO2R, adrs, adAg = self.y
        thetasurf, Ts, CO2, wg, Swin, ra, Tsoil = checkpoint['ags_thetasurf'],checkpoint['ags_Ts'],checkpoint['ags_CO2'],checkpoint['ags_wg'],checkpoint['ags_Swin'],checkpoint['ags_ra'],checkpoint['ags_Tsoil']      
        COS,LAI,alfa_sto = checkpoint['ags_COS'],checkpoint['ags_LAI'],checkpoint['ags_alfa_sto']
        
        texp, fw, Ds, D0, Dstar, co2abs, CO2comp, rsCO2, ci, PAR, alphac, cfrac = checkpoint['ags_texp_end'],checkpoint['ags_fw_end'],checkpoint['ags_Ds_end'],checkpoint['ags_D0_end'],checkpoint['ags_Dstar_end'],checkpoint['ags_co2abs_end'],checkpoint['ags_CO2comp_end'],checkpoint['ags_rsCO2_end'],checkpoint['ags_ci_end'],checkpoint['ags_PAR_end'],checkpoint['ags_alphac_end'],checkpoint['ags_cfrac_end']
        gm, gm1,gm2,gm3,sqrtf,sqterm,fmin0 = checkpoint['ags_gm_end'],checkpoint['ags_gm1_end'],checkpoint['ags_gm2_end'],checkpoint['ags_gm3_end'],checkpoint['ags_sqrtf_end'],checkpoint['ags_sqterm_end'],checkpoint['ags_fmin0_end']
        Ammax1,Ammax2,Ammax3,fmin = checkpoint['ags_Ammax1_end'],checkpoint['ags_Ammax2_end'],checkpoint['ags_Ammax3_end'],checkpoint['ags_fmin_end']
        pexp, xdiv, aexp, Ammax = checkpoint['ags_pexp_end'],checkpoint['ags_xdiv_end'],checkpoint['ags_aexp_end'],checkpoint['ags_Ammax_end']
        y, y1, AmRdark, sy = checkpoint['ags_y_end'],checkpoint['ags_y1_end'],checkpoint['ags_AmRdark_end'],checkpoint['ags_sy_end']
        fstr, a1, div1, div2, An_temporary, part1,gcco2, a11 = checkpoint['ags_fstr_end'],checkpoint['ags_a1_end'],checkpoint['ags_div1_end'],checkpoint['ags_div2_end'],checkpoint['ags_An_temporary_end'],checkpoint['ags_part1_end'],checkpoint['ags_gcco2_end'],checkpoint['ags_a11_end']
        gciCOS = checkpoint['ags_gciCOS_end']
        gctCOS = checkpoint['ags_gctCOS_end']
        PARfract = checkpoint['ags_PARfract']
        cveg = checkpoint['ags_cveg']
        wwilt = checkpoint['ags_wwilt']
        wfc = checkpoint['ags_wfc']
        w2 = checkpoint['ags_w2']
        R10,E0 = checkpoint['ags_R10'],checkpoint['ags_E0']
        #statement dwCOS   = dwCOSP + dwCOSS
        self.adwCOSP += self.adwCOS
        self.adwCOSS += self.adwCOS
        self.adwCOS = 0
        if(model.input.soilCOSmodeltype == 'Sun_Ogee'):
            #statement dwCOSS = dwCOSS_molm2s / model.rho * model.mair * 1.e-3 * 1.e9
            self.adwCOSS_molm2s += 1. / model.rho * model.mair * 1.e-3 * 1.e9 * self.adwCOSS
            self.adwCOSS = 0
            #statement dwCOSS_molm2s = self.tl_run_soil_COS_mod(model,checkpoint,returnvariable='dCOS_netuptake_soilsun')
            self.adCOS_netuptake_soilsun += self.adwCOSS_molm2s #THIS IS ESSENTIAL!! needed because there is a hidden statement in the tangent linear saying that dwCOSS_molm2s = dCOS_netuptake_soilsun. this occurs in the TL after calling tl_run_soil_COS_mod
            self.adwCOSS_molm2s = 0
            self.adj_run_soil_COS_mod(forcing,checkpoint,model) 
            #statement self.dairtemp = self.dTsurf
            self.adTsurf += self.adairtemp
            self.adairtemp = 0
            #statement self.dmol_rat_ocs_atm = self.dCOSsurf
            self.adCOSsurf += self.admol_rat_ocs_atm
            self.admol_rat_ocs_atm = 0
        elif model.soilCOSmodeltype == None:
            #statement dwCOSS = 0
            self.adwCOSS = 0
        if model.ags_C_mode == 'MXL': 
            #statement dwCOSP  = COS * (1 / gctCOS + ra)**(-2) * (-1 * gctCOS**(-2) * dgctCOS + self.dra) + -1 / (1 / gctCOS + ra) * self.dCOS
            self.adgctCOS += COS * (1 / gctCOS + ra)**(-2) * -1 * gctCOS**(-2) * self.adwCOSP
            self.adra += COS * (1 / gctCOS + ra)**(-2) * self.adwCOSP
            self.adCOS += -1 / (1 / gctCOS + ra) * self.adwCOSP
            self.adwCOSP = 0
        elif model.ags_C_mode == 'surf':
            COSsurf = checkpoint['ags_COSsurf']
            #statement dwCOSP  = COSsurf * (1 / gctCOS + ra)**(-2) * (-1 * gctCOS**(-2) * dgctCOS + self.dra) + -1 / (1 / gctCOS + ra) * self.dCOSsurf
            self.adgctCOS += COSsurf * (1 / gctCOS + ra)**(-2) * -1 * gctCOS**(-2) * self.adwCOSP
            self.adra += COSsurf * (1 / gctCOS + ra)**(-2) * self.adwCOSP
            self.adCOSsurf += -1 / (1 / gctCOS + ra) * self.adwCOSP
            self.adwCOSP = 0
        else:
            raise Exception('wrong ags_C_mode switch')
        # statement dwCO2   = dwCO2A + dwCO2R
        self.adwCO2A  += self.adwCO2
        self.adwCO2R  += self.adwCO2
        self.adwCO2   = 0.0
        # statement dwCO2R  = (dResp_dE0 + dResp_dTsoil + dResp_dwg + dResp_dR10)  * (model.mair / (model.rho * model.mco2))
        self.adResp_dE0     += self.adwCO2R * (model.mair / (model.rho * model.mco2))
        self.adResp_dTsoil  += self.adwCO2R * (model.mair / (model.rho * model.mco2))
        self.adResp_dwg     += self.adwCO2R * (model.mair / (model.rho * model.mco2))
        self.adResp_dR10    += self.adwCO2R * (model.mair / (model.rho * model.mco2))
        self.adwCO2R    = 0.0
        #statement dwCO2A  = (dAn_dthetasurf + dAn_dTs + dAn_de + dAn_dCO2 + dAn_dra + dAn_dPARfract + dAn_dSwin + dAn_dcveg + dAn_dw2 + dAn_dwfc + dAn_dwwilt + dAn_dalfa_sto + dAn_dLAI)  * (model.mair / (model.rho * model.mco2))
        self.adAn_dthetasurf += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_dTs += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_de += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_dCO2 += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_dra += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_dPARfract += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dSwin += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dcveg += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dw2 += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dwfc += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dwwilt += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adAn_dalfa_sto += (model.mair / (model.rho * model.mco2)) * self.adwCO2A 
        self.adAn_dLAI += (model.mair / (model.rho * model.mco2)) * self.adwCO2A
        self.adwCO2A    = 0.0
        #statement dResp_dR10    = (1. - fw) * np.exp(texp) * self.dR10
        self.adR10 += (1. - fw) * np.exp(texp) * self.adResp_dR10
        self.adResp_dR10 = 0
        #statement dResp_dwg    = -dfw_dwg*R10* np.exp(texp)
        self.adfw_dwg      += -self.adResp_dwg    *R10* np.exp(texp)
        self.adResp_dwg     = 0.0
        #statement dResp_dTsoil = dtexp_dTsoil*R10 * (1. - fw) * np.exp(texp)
        self.adtexp_dTsoil +=  self.adResp_dTsoil *R10 * (1. - fw) * np.exp(texp)       
        self.adResp_dTsoil  = 0.0
        #statement dResp_dE0 = dtexp_dE0*R10 * (1. - fw) * np.exp(texp)
        self.adtexp_dE0 += R10 * (1. - fw) * np.exp(texp) * self.adResp_dE0
        self.adResp_dE0 = 0
        #       dtexp_dTsoil = dTsoil*E0/(8.314*Tsoil**2)
        self.adTsoil   += self.adtexp_dTsoil*E0/(8.314*Tsoil**2)
        self.adtexp_dTsoil  = 0.0
        #statement dtexp_dE0 = self.dE0 / (283.15 * 8.314) * (1. - 283.15 / Tsoil)
        self.adE0 += self.adtexp_dE0 / (283.15 * 8.314) * (1. - 283.15 / Tsoil)
        self.adtexp_dE0 = 0
        # statement dfw_dwg      = -dwg* model.Cw * model.wmax / ((wg + model.wmin)**2)
        self.adwg       += -self.adfw_dwg*model.Cw * model.wmax / ((wg + model.wmin)**2)
        self.adfw_dwg = 0.0        
        #statement dAn_dLAI      = drsCO2_dLAI   * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dLAI += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dLAI
        self.adAn_dLAI = 0
        #statement dAn_dalfa_sto = -(co2abs - ci) * -1 * (ra + rsCO2)**(-2) * drsCO2_dalfa_sto
        self.adrsCO2_dalfa_sto += -(co2abs - ci) * -1 * (ra + rsCO2)**(-2) * self.adAn_dalfa_sto 
        self.adAn_dalfa_sto = 0
        #statement dAn_dwwilt      = drsCO2_dwwilt   * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dwwilt += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dwwilt
        self.adAn_dwwilt = 0
        #statement dAn_dwfc      = drsCO2_dwfc   * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dwfc += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dwfc
        self.adAn_dwfc = 0
        #statement dAn_dw2      = drsCO2_dw2   * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dw2 += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dw2 
        self.adAn_dw2 = 0.0
        #statement dAn_dcveg    = drsCO2_dcveg * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dcveg += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dcveg 
        self.adAn_dcveg = 0
        #statement dAn_dSwin    = drsCO2_dSwin * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dSwin += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dSwin 
        self.adAn_dSwin = 0
        #statement dAn_dPARfract    = drsCO2_dPARfract * (co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dPARfract += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dPARfract
        self.adAn_dPARfract = 0
        #dAn_dra      = dra * (co2abs-ci)/((ra + rsCO2)**2)
        self.adra += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dra 
        self.adAn_dra = 0
        #statement dAn_dCO2     = -(dco2abs - dci_dCO2)/ (ra + rsCO2) + drsCO2_dCO2*(co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dCO2 += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dCO2 
        self.adci_dCO2 += 1/(ra + rsCO2) * self.adAn_dCO2 
        self.adco2abs += -1/(ra + rsCO2) * self.adAn_dCO2 
        self.adAn_dCO2 = 0
        #statement dAn_de       = dci_de/ (ra + rsCO2) + drsCO2_de*(co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_de += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_de 
        self.adci_de += 1/ (ra + rsCO2) * self.adAn_de
        self.adAn_de = 0
        #statement dAn_dTs      = dci_dTs/ (ra + rsCO2) + drsCO2_dTs*(co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dTs += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dTs 
        self.adci_dTs += 1/ (ra + rsCO2) * self.adAn_dTs 
        self.adAn_dTs = 0
        #statement dAn_dthetasurf          = dci_dthetasurf/ (ra + rsCO2) + drsCO2_dthetasurf*(co2abs-ci)/((ra + rsCO2)**2)
        self.adrsCO2_dthetasurf += (co2abs-ci)/((ra + rsCO2)**2) * self.adAn_dthetasurf 
        self.adci_dthetasurf += 1 / (ra + rsCO2) * self.adAn_dthetasurf 
        self.adAn_dthetasurf = 0
        #statement drsCO2_dLAI   = - dgcco2_dLAI   / (gcco2**2)
        self.adgcco2_dLAI += - 1 / (gcco2**2) * self.adrsCO2_dLAI 
        self.adrsCO2_dLAI = 0
        #statement drsCO2_dalfa_sto   = - dgcco2_dalfa_sto   / (gcco2**2)
        self.adgcco2_dalfa_sto += - 1 / (gcco2**2) * self.adrsCO2_dalfa_sto 
        self.adrsCO2_dalfa_sto = 0
        #statement drsCO2_dwwilt   = - dgcco2_dwwilt   / (gcco2**2)
        self.adgcco2_dwwilt += - 1  / (gcco2**2) * self.adrsCO2_dwwilt 
        self.adrsCO2_dwwilt = 0
        #statement drsCO2_dwfc   = - dgcco2_dwfc   / (gcco2**2)
        self.adgcco2_dwfc += - 1  / (gcco2**2) * self.adrsCO2_dwfc
        self.adrsCO2_dwfc = 0
        #statement drsCO2_dw2   = - dgcco2_dw2   / (gcco2**2)
        self.adgcco2_dw2 += - 1  / (gcco2**2) * self.adrsCO2_dw2 
        self.adrsCO2_dw2 = 0
        #statement drsCO2_dcveg   = - dgcco2_dcveg   / (gcco2**2)
        self.adgcco2_dcveg += - 1  / (gcco2**2) * self.adrsCO2_dcveg 
        self.adrsCO2_dcveg = 0
        #statement drsCO2_dSwin = - dgcco2_dSwin / (gcco2**2)
        self.adgcco2_dSwin += - 1 / (gcco2**2) * self.adrsCO2_dSwin 
        self.adrsCO2_dSwin = 0
        #statement drsCO2_dPARfract = - dgcco2_dPARfract / (gcco2**2)
        self.adgcco2_dPARfract += - 1. / (gcco2**2) * self.adrsCO2_dPARfract
        self.adrsCO2_dPARfract = 0
        #statement drsCO2_dCO2  = - dgcco2_dCO2  / (gcco2**2)
        self.adgcco2_dCO2 += - 1  / (gcco2**2) * self.adrsCO2_dCO2 
        self.adrsCO2_dCO2= 0
        #statement drsCO2_de   = - dgcco2_de   / (gcco2**2)
        self.adgcco2_de += - 1  / (gcco2**2) * self.adrsCO2_de
        self.adrsCO2_de = 0
        #statement drsCO2_dTs   = - dgcco2_dTs   / (gcco2**2)
        self.adgcco2_dTs += - 1  / (gcco2**2) * self.adrsCO2_dTs
        self.adrsCO2_dTs = 0
        #statement drsCO2_dthetasurf       = - dgcco2_dthetasurf       / (gcco2**2)
        self.adgcco2_dthetasurf += - 1 / (gcco2**2) * self.adrsCO2_dthetasurf 
        self.adrsCO2_dthetasurf = 0
        #statement drs     = - ( dgcco2_dthetasurf + dgcco2_dTs + dgcco2_de + dgcco2_dCO2 + dgcco2_dPARfract + dgcco2_dSwin + dgcco2_dcveg + dgcco2_dw2 + dgcco2_dwfc + dgcco2_dwwilt + dgcco2_dalfa_sto + dgcco2_dLAI) / (1.6 * gcco2**2)
        self.adgcco2_dthetasurf += - 1 / (1.6 * gcco2**2) * self.adrs 
        self.adgcco2_dTs += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_de += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dCO2 += - 1 / (1.6 * gcco2**2) * self.adrs 
        self.adgcco2_dPARfract += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dSwin += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dcveg += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dw2 += - 1 / (1.6 * gcco2**2) * self.adrs 
        self.adgcco2_dwfc += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dwwilt += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dalfa_sto += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adgcco2_dLAI += - 1 / (1.6 * gcco2**2) * self.adrs
        self.adrs = 0
        #statement dgctCOS = dgctCOS_dthetasurf + dgctCOS_dTs + dgctCOS_de + dgctCOS_dCO2 + dgctCOS_dPARfract + dgctCOS_dSwin + dgctCOS_dcveg + dgctCOS_dw2 + dgctCOS_dwfc + dgctCOS_dwwilt + dgctCOS_dalfa_sto + dgctCOS_dLAI + dgctCOS_dgciCOS
        self.adgctCOS_dthetasurf += self.adgctCOS
        self.adgctCOS_dTs += self.adgctCOS
        self.adgctCOS_de += self.adgctCOS
        self.adgctCOS_dCO2 += self.adgctCOS
        self.adgctCOS_dPARfract += self.adgctCOS
        self.adgctCOS_dSwin += self.adgctCOS
        self.adgctCOS_dcveg += self.adgctCOS
        self.adgctCOS_dw2 += self.adgctCOS
        self.adgctCOS_dwfc += self.adgctCOS
        self.adgctCOS_dwwilt += self.adgctCOS
        self.adgctCOS_dalfa_sto += self.adgctCOS
        self.adgctCOS_dLAI += self.adgctCOS
        self.adgctCOS_dgciCOS += self.adgctCOS
        self.adgctCOS = 0
        #statement dgctCOS_dgciCOS = -1 * (1 / gciCOS + 1.21/gcco2)**-2 * -1 * gciCOS**-2 * self.dgciCOS
        self.adgciCOS += -1 * (1 / gciCOS + 1.21/gcco2)**-2 * -1 * gciCOS**-2 * self.adgctCOS_dgciCOS 
        self.adgctCOS_dgciCOS = 0
        #statement dgctCOS_dLAI = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dLAI
        self.adgcco2_dLAI += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dLAI
        self.adgctCOS_dLAI = 0
        #statement dgctCOS_dalfa_sto = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dalfa_sto
        self.adgcco2_dalfa_sto += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dalfa_sto
        self.adgctCOS_dalfa_sto = 0
        #statement dgctCOS_dwwilt = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dwwilt
        self.adgcco2_dwwilt += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2) * self.adgctCOS_dwwilt
        self.adgctCOS_dwwilt = 0
        #statement dgctCOS_dwfc = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dwfc
        self.adgcco2_dwfc += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2) * self.adgctCOS_dwfc
        self.adgctCOS_dwfc = 0
        #statement dgctCOS_dw2 = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dw2
        self.adgcco2_dw2 += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2) * self.adgctCOS_dw2
        self.adgctCOS_dw2 = 0
        #statement dgctCOS_dcveg = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dcveg
        self.adgcco2_dcveg += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2) * self.adgctCOS_dcveg
        self.adgctCOS_dcveg = 0
        #statement dgctCOS_dSwin = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dSwin
        self.adgcco2_dSwin += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dSwin
        self.adgctCOS_dSwin = 0
        #statement dgctCOS_dPARfract = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dPARfract
        self.adgcco2_dPARfract +=  - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2) * self.adgctCOS_dPARfract
        self.adgctCOS_dPARfract = 0
        #statement dgctCOS_dCO2 = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dCO2
        self.adgcco2_dCO2 += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dCO2
        self.adgctCOS_dCO2 = 0
        #statement dgctCOS_de = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_de
        self.adgcco2_de += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_de
        self.adgctCOS_de = 0
        #statement dgctCOS_dTs = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dTs
        self.adgcco2_dTs += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dTs
        self.adgctCOS_dTs = 0
        #statement dgctCOS_dthetasurf = - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*dgcco2_dthetasurf
        self.adgcco2_dthetasurf += - (1/gciCOS + 1.21/gcco2)**(-2) * 1.21 * -1 * gcco2**(-2)*self.adgctCOS_dthetasurf
        self.adgctCOS_dthetasurf = 0
        # statement dgcco2 = dgcco2_dthetasurf + dgcco2_dTs + dgcco2_de + dgcco2_dCO2 + dgcco2_dPARfract + dgcco2_dSwin + dgcco2_dcveg + dgcco2_dw2 + dgcco2_dwfc + dgcco2_dwwilt + dgcco2_dalfa_sto + dgcco2_dLAI
        self.adgcco2_dthetasurf += self.adgcco2
        self.adgcco2_dTs += self.adgcco2
        self.adgcco2_de += self.adgcco2
        self.adgcco2_dCO2 += self.adgcco2
        self.adgcco2_dPARfract += self.adgcco2
        self.adgcco2_dSwin += self.adgcco2
        self.adgcco2_dcveg += self.adgcco2
        self.adgcco2_dw2 += self.adgcco2
        self.adgcco2_dwfc += self.adgcco2
        self.adgcco2_dwwilt += self.adgcco2
        self.adgcco2_dalfa_sto += self.adgcco2
        self.adgcco2_dLAI += self.adgcco2
        self.adgcco2 = 0
        #statement dgcco2_dLAI = alfa_sto * (model.gmin[c] / model.nuco2q + part1) * self.dLAI + alfa_sto * LAI * dpart1_dLAI
        self.adLAI += alfa_sto * (model.gmin[c] / model.nuco2q + part1) * self.adgcco2_dLAI
        self.adpart1_dLAI += alfa_sto * LAI * self.adgcco2_dLAI
        self.adgcco2_dLAI = 0
        #statement dgcco2_dalfa_sto = dgcco2_dalfa_sto = LAI * (model.gmin[c] / model.nuco2q + part1) * self.dalfa_sto
        self.adalfa_sto += LAI * (model.gmin[c] / model.nuco2q + part1) * self.adgcco2_dalfa_sto
        self.adgcco2_dalfa_sto = 0
        #statement dgcco2_dwwilt   = alfa_sto * LAI * dpart1_dwwilt
        self.adpart1_dwwilt += alfa_sto * LAI * self.adgcco2_dwwilt 
        self.adgcco2_dwwilt = 0
        #statement dgcco2_dwfc   = alfa_sto * LAI * dpart1_dwfc
        self.adpart1_dwfc += alfa_sto * LAI * self.adgcco2_dwfc 
        self.adgcco2_dwfc = 0
        #statement dgcco2_dw2   = alfa_sto * LAI * dpart1_dw2
        self.adpart1_dw2 += alfa_sto * LAI * self.adgcco2_dw2 
        self.adgcco2_dw2 = 0
        #statement dgcco2_dcveg = alfa_sto * LAI * dpart1_dcveg
        self.adpart1_dcveg += alfa_sto * LAI * self.adgcco2_dcveg 
        self.adgcco2_dcveg = 0
        #statement dgcco2_dSwin = alfa_sto * LAI * dpart1_dSwin
        self.adpart1_dSwin += alfa_sto * LAI * self.adgcco2_dSwin
        self.adgcco2_dSwin = 0
        #statement dgcco2_dPARfract = alfa_sto * LAI * dpart1_dPARfract
        self.adpart1_dPARfract += alfa_sto * LAI * self.adgcco2_dPARfract
        self.adgcco2_dPARfract = 0
        #statement dgcco2_dCO2  = alfa_sto * LAI * dpart1_dCO2
        self.adpart1_dCO2 += alfa_sto * LAI * self.adgcco2_dCO2 
        self.adgcco2_dCO2 = 0
        #statement dgcco2_de   = alfa_sto * LAI * dpart1_de
        self.adpart1_de += alfa_sto * LAI * self.adgcco2_de
        self.adgcco2_de = 0
        #statement dgcco2_dTs   = alfa_sto * LAI * dpart1_dTs
        self.adpart1_dTs += alfa_sto * LAI * self.adgcco2_dTs
        self.adgcco2_dTs = 0
        #statement dgcco2_dthetasurf       = alfa_sto * LAI * dpart1_dthetasurf
        self.adpart1_dthetasurf += alfa_sto * LAI * self.adgcco2_dthetasurf 
        self.adgcco2_dthetasurf = 0
        #statement dpart1_dLAI  = a1 * fstr * dAn_temporary_dLAI / div2
        self.adAn_temporary_dLAI += a1 * fstr / div2 * self.adpart1_dLAI
        self.adpart1_dLAI = 0
        #statement dpart1_dwwilt  = a1 * dfstr_dwwilt * An_temporary / div2
        self.adfstr_dwwilt += a1 * An_temporary / div2 * self.adpart1_dwwilt
        self.adpart1_dwwilt = 0
        #statement dpart1_dwfc  = a1 * dfstr_dwfc * An_temporary / div2
        self.adfstr_dwfc += a1 * An_temporary / div2 * self.adpart1_dwfc
        self.adpart1_dwfc = 0
        #statement dpart1_dw2   = a1 * dfstr_dw2 * An_temporary / div2
        self.adfstr_dw2 += a1 * An_temporary / div2 * self.adpart1_dw2 
        self.adpart1_dw2 = 0
        #statement dpart1_dcveg = a1 * fstr * dAn_temporary_dcveg / div2
        self.adAn_temporary_dcveg += a1 * fstr / div2 * self.adpart1_dcveg 
        self.adpart1_dcveg = 0
        #statement dpart1_dSwin = a1 * fstr * dAn_temporary_dSwin / div2
        self.adAn_temporary_dSwin += a1 * fstr / div2 * self.adpart1_dSwin 
        self.adpart1_dSwin = 0
        #statement dpart1_dPARfract = a1 * fstr * dAn_temporary_dPARfract / div2
        self.adAn_temporary_dPARfract += a1 * fstr / div2 * self.adpart1_dPARfract
        self.adpart1_dPARfract = 0
        #statement dpart1_dCO2  = a1 * fstr * dAn_temporary_dCO2 / div2 - a1 * fstr * An * ddiv2_dCO2 / (div2**2)
        self.adAn_temporary_dCO2 += a1 * fstr / div2 * self.adpart1_dCO2 
        self.addiv2_dCO2 += - a1 * fstr * An_temporary / (div2**2) * self.adpart1_dCO2 
        self.adpart1_dCO2 = 0
        #statement dpart1_de   = a1 * fstr * dAn_temporary_de / div2 - a1 * fstr * An * ddiv2_de / (div2**2)
        self.adAn_temporary_de += a1 * fstr / div2 * self.adpart1_de
        self.addiv2_de += - a1 * fstr * An_temporary / (div2**2) * self.adpart1_de 
        self.adpart1_de = 0
        #statement dpart1_dTs   = a1 * fstr * dAn_temporary_dTs / div2 - a1 * fstr * An * ddiv2_dTs / (div2**2)
        self.adAn_temporary_dTs += a1 * fstr / div2 * self.adpart1_dTs
        self.addiv2_dTs += - a1 * fstr * An_temporary / (div2**2) * self.adpart1_dTs 
        self.adpart1_dTs = 0
        #statement dpart1_dthetasurf       = a1 * fstr * dAn_temporary_dthetasurf / div2 - a1 * fstr * An * ddiv2_dthetasurf / (div2**2)
        self.adAn_temporary_dthetasurf += a1 * fstr / div2 * self.adpart1_dthetasurf 
        self.addiv2_dthetasurf += - a1 * fstr * An_temporary * self.adpart1_dthetasurf / (div2**2)
        self.adpart1_dthetasurf = 0
        #statement ddiv2_dCO2   = dco2abs * div1
        self.adco2abs += div1 * self.addiv2_dCO2
        self.addiv2_dCO2 = 0
        #statement ddiv2_de    = (co2abs - CO2comp)*ddiv1_de
        self.addiv1_de += (co2abs - CO2comp) * self.addiv2_de
        self.addiv2_de = 0
        #statement ddiv2_dTs    = (co2abs - CO2comp)*ddiv1_dTs
        self.addiv1_dTs += (co2abs - CO2comp) * self.addiv2_dTs 
        self.addiv2_dTs = 0
        #statement ddiv2_dthetasurf =  -dCO2comp_dthetasurf*div1 
        self.adCO2comp_dthetasurf += -div1 * self.addiv2_dthetasurf  
        self.addiv2_dthetasurf = 0
        #statement ddiv1_de    =  dDs_de/Dstar
        self.adDs_de += 1/Dstar * self.addiv1_de 
        self.addiv1_de = 0
        #statement ddiv1_dTs    =  dDs_dTs/Dstar
        self.adDs_dTs += 1/Dstar * self.addiv1_dTs 
        self.addiv1_dTs = 0
        #statement ddiv1_dthetasurf =  -dDstar_dthetasurf*Ds/(Dstar**2) = 0, do not include it here to prevent rounding errors
        #statement dDstar_dthetasurf       = dD0_dthetasurf/a11 - da11_dthetasurf*D0/(a11**2)   = 0, do not include
        #statement da11_dthetasurf         =  -dfmin_dthetasurf*a1       
        self.adfmin_dthetasurf  += -self.ada11_dthetasurf*a1 
        self.ada11_dthetasurf   = 0.0
        #statement dAn_temporary_dLAI    = - dsy_dLAI * AmRdark
        self.adsy_dLAI += - self.adAn_temporary_dLAI * AmRdark 
        self.adAn_temporary_dLAI = 0
        #statement dAn_temporary_dcveg    = - dsy_dcveg * AmRdark
        self.adsy_dcveg += - self.adAn_temporary_dcveg * AmRdark 
        self.adAn_temporary_dcveg = 0
        #statement dAn_temporary_dSwin    = - dsy_dSwin * AmRdark
        self.adsy_dSwin += - self.adAn_temporary_dSwin * AmRdark 
        self.adAn_temporary_dSwin = 0
        #statement dAn_temporary_dPARfract    = - dsy_dPARfract * AmRdark
        self.adsy_dPARfract += - AmRdark * self.adAn_temporary_dPARfract
        self.adAn_temporary_dPARfract = 0
        #statement dAn_temporary_dCO2     = dAmRdark_dCO2 * (1. - sy) - dsy_dCO2 * AmRdark
        self.adAmRdark_dCO2 += (1. - sy) * self.adAn_temporary_dCO2
        self.adsy_dCO2 += - 1 * AmRdark * self.adAn_temporary_dCO2 
        self.adAn_temporary_dCO2 = 0
        #statement dAn_temporary_de      = dAmRdark_de * (1. - sy) - dsy_de * AmRdark
        self.adAmRdark_de += (1. - sy) * self.adAn_temporary_de 
        self.adsy_de += - 1 * AmRdark * self.adAn_temporary_de
        self.adAn_temporary_de = 0
        #statement dAn_temporary_dTs      = dAmRdark_dTs * (1. - sy) - dsy_dTs * AmRdark
        self.adAmRdark_dTs += (1. - sy) * self.adAn_temporary_dTs 
        self.adsy_dTs += - 1 * AmRdark * self.adAn_temporary_dTs
        self.adAn_temporary_dTs = 0
        #statement dAn_temporary_dthetasurf = dAmRdark_dthetasurf * (1. - sy) - dsy_dthetasurf * AmRdark
        self.adAmRdark_dthetasurf += (1. - sy) * self.adAn_temporary_dthetasurf 
        self.adsy_dthetasurf += -1 * AmRdark * self.adAn_temporary_dthetasurf
        self.adAn_temporary_dthetasurf = 0
        #statement dsy_dLAI    =  dy1_dLAI * self.dE1(y1) / (model.Kx[c] * LAI) + (model.E1(y1) - model.E1(y)) * -1 * (model.Kx[c] * LAI)**(-2) * model.Kx[c] * self.dLAI
        self.ady1_dLAI += self.dE1(y1) / (model.Kx[c] * LAI) * self.adsy_dLAI
        self.adLAI += (model.E1(y1) - model.E1(y)) * -1 * (model.Kx[c] * LAI)**(-2) * model.Kx[c] * self.adsy_dLAI
        self.adsy_dLAI = 0
        #statement dsy_dcveg    =  (dy1_dcveg * self.dE1(y1) - dy_dcveg * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dcveg += self.dE1(y1) /(model.Kx[c] * LAI) * self.adsy_dcveg 
        self.ady_dcveg += -1 * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_dcveg
        self.adsy_dcveg = 0
        #statement dsy_dSwin    =  (dy1_dSwin * self.dE1(y1) - dy_dSwin * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dSwin += self.dE1(y1) /(model.Kx[c] * LAI) * self.adsy_dSwin 
        self.ady_dSwin += -1 * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_dSwin
        self.adsy_dSwin = 0
        if self.manualadjointtesting:
            self.adsy_dPARfract = self.y
        #statement dsy_dPARfract    =  (dy1_dPARfract * self.dE1(y1) - dy_dPARfract * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dPARfract += self.dE1(y1) /(model.Kx[c] * LAI) * self.adsy_dPARfract
        self.ady_dPARfract += -1 * self.dE1(y) /(model.Kx[c] * LAI) * self.adsy_dPARfract
        self.adsy_dPARfract = 0
        if self.manualadjointtesting:
            self.HTy = self.ady1_dPARfract,self.ady_dPARfract
        #statement dsy_dCO2 =  (dy1_dCO2  * self.dE1(y1) - dy_dCO2  * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dCO2 += self.dE1(y1) / (model.Kx[c] * LAI) * self.adsy_dCO2 
        self.ady_dCO2 += - 1  * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_dCO2 
        self.adsy_dCO2 = 0
        #statement dsy_de      =  (dy1_de   * self.dE1(y1) - dy_de   * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_de += self.dE1(y1) / (model.Kx[c] * LAI) * self.adsy_de
        self.ady_de += - 1 * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_de
        self.adsy_de = 0
        #statement dsy_dTs      =  (dy1_dTs   * self.dE1(y1) - dy_dTs   * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dTs += self.dE1(y1) / (model.Kx[c] * LAI) * self.adsy_dTs
        self.ady_dTs += - 1 * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_dTs
        self.adsy_dTs = 0
        #statement dsy_dthetasurf =  (dy1_dthetasurf * self.dE1(y1) - dy_dthetasurf * self.dE1(y))/(model.Kx[c] * LAI)
        self.ady1_dthetasurf += self.dE1(y1) / (model.Kx[c] * LAI) * self.adsy_dthetasurf
        self.ady_dthetasurf += - 1 * self.dE1(y)/(model.Kx[c] * LAI) * self.adsy_dthetasurf
        self.adsy_dthetasurf = 0
        #statement dy1_dLAI     =  y * np.exp(-model.Kx[c] * LAI) * -model.Kx[c] * self.dLAI
        self.adLAI += y * np.exp(-model.Kx[c] * LAI) * -model.Kx[c] * self.ady1_dLAI
        self.ady1_dLAI = 0
        #statement dy1_dcveg =  dy_dcveg * np.exp(-model.Kx[c] * LAI)
        self.ady_dcveg += np.exp(-model.Kx[c] * LAI) * self.ady1_dcveg
        self.ady1_dcveg = 0
        #statement dy1_dSwin =  dy_dSwin * np.exp(-model.Kx[c] * LAI)
        self.ady_dSwin += np.exp(-model.Kx[c] * LAI) * self.ady1_dSwin
        self.ady1_dSwin = 0
        #statement dy1_dPARfract =  dy_dPARfract * np.exp(-model.Kx[c] * LAI)
        self.ady_dPARfract += np.exp(-model.Kx[c] * LAI) * self.ady1_dPARfract
        self.ady1_dPARfract = 0
        #statement dy1_dCO2     =  dy_dCO2  * np.exp(-model.Kx[c] * LAI)
        self.ady_dCO2 += np.exp(-model.Kx[c] * LAI) * self.ady1_dCO2 
        self.ady1_dCO2 = 0
        #statement dy1_de      =  dy_de   * np.exp(-model.Kx[c] * LAI)
        self.ady_de += np.exp(-model.Kx[c] * LAI) * self.ady1_de
        self.ady1_de = 0
        #statement dy1_dTs      =  dy_dTs   * np.exp(-model.Kx[c] * LAI)
        self.ady_dTs += np.exp(-model.Kx[c] * LAI) * self.ady1_dTs
        self.ady1_dTs = 0
        #statement dy1_dthetasurf          =  dy_dthetasurf * np.exp(-model.Kx[c] * LAI)
        self.ady_dthetasurf += np.exp(-model.Kx[c] * LAI) * self.ady1_dthetasurf  
        self.ady1_dthetasurf  = 0
        #statement dy_dcveg     =  alphac * model.Kx[c] * dPAR_dcveg / (AmRdark)
        self.adPAR_dcveg += alphac * model.Kx[c] * 1 / (AmRdark) * self.ady_dcveg 
        self.ady_dcveg = 0
        #statement dy_dSwin     =  alphac * model.Kx[c] * dPAR_dSwin / (AmRdark)
        self.adPAR_dSwin += alphac * model.Kx[c] * 1 / (AmRdark) * self.ady_dSwin 
        self.ady_dSwin = 0
        #statement dy_dPARfract     =  alphac * model.Kx[c] * dPAR_dPARfract / (AmRdark)
        self.adPAR_dPARfract += alphac * model.Kx[c] / (AmRdark) * self.ady_dPARfract
        self.ady_dPARfract = 0
        #statement dy_dCO2 =  model.Kx[c] * PAR * (dalphac_dCO2 / AmRdark - dAmRdark_dCO2 * alphac / (AmRdark**2))
        self.adalphac_dCO2 += model.Kx[c] * PAR / AmRdark * self.ady_dCO2
        self.adAmRdark_dCO2 += -1 * model.Kx[c] * PAR * alphac / (AmRdark**2) * self.ady_dCO2
        self.ady_dCO2 = 0
        #statement dy_de = -model.Kx[c] * PAR * dAmRdark_de * alphac / (AmRdark**2)
        self.adAmRdark_de += -model.Kx[c] * PAR * alphac / (AmRdark**2) * self.ady_de
        self.ady_de = 0
        #statement dy_dTs = -model.Kx[c] * PAR * dAmRdark_dTs * alphac / (AmRdark**2)
        self.adAmRdark_dTs += -model.Kx[c] * PAR * alphac / (AmRdark**2) * self.ady_dTs
        self.ady_dTs = 0
        #statement dy_dthetasurf=  model.Kx[c] * PAR * (dalphac_dthetasurf / AmRdark - dAmRdark_dthetasurf * alphac / (AmRdark**2))
        self.adalphac_dthetasurf += model.Kx[c] * PAR / AmRdark * self.ady_dthetasurf 
        self.adAmRdark_dthetasurf += -1 * model.Kx[c] * PAR * alphac / (AmRdark**2) * self.ady_dthetasurf
        self.ady_dthetasurf = 0
        #statement dAg = ( dAmRdark_dthetasurf + dAmRdark_dTs + dAmRdark_de + dAmRdark_dCO2  ) * (1. - np.exp(pexp)) -  AmRdark * ( dpexp_dthetasurf + dpexp_dTs + dpexp_de + dpexp_dCO2 + dpexp_dPARfract + dpexp_dSwin + dpexp_dcveg) * np.exp(pexp)
        self.adAmRdark_dthetasurf += (1. - np.exp(pexp)) * self.adAg
        self.adAmRdark_dTs += (1. - np.exp(pexp)) * self.adAg
        self.adAmRdark_de += (1. - np.exp(pexp)) * self.adAg
        self.adAmRdark_dCO2 += (1. - np.exp(pexp)) * self.adAg
        self.adpexp_dthetasurf += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_dTs += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_de += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_dCO2 += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_dPARfract += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_dSwin += - AmRdark * np.exp(pexp) * self.adAg
        self.adpexp_dcveg += - AmRdark * np.exp(pexp) * self.adAg
        self.adAg = 0
        #statement dpexp_dcveg  = -1 * alphac * dPAR_dSwin / (AmRdark)
        self.adPAR_dcveg += -1 * alphac / (AmRdark) * self.adpexp_dcveg
        self.adpexp_dcveg = 0
        #statement dpexp_dSwin  = -1 * alphac * dPAR_dSwin / (AmRdark)
        self.adPAR_dSwin += -1 * alphac / (AmRdark) * self.adpexp_dSwin
        self.adpexp_dSwin = 0
        #statement dpexp_dPARfract  = -1 * alphac * dPAR_dPARfract / (AmRdark)
        self.adPAR_dPARfract += -1 * alphac / (AmRdark) * self.adpexp_dPARfract 
        self.adpexp_dPARfract = 0
        #statement dpexp_dCO2 = -1 * dalphac_dCO2 * PAR / (AmRdark) + dAmRdark_dCO2 * alphac * PAR/(AmRdark**2)
        self.adalphac_dCO2 += -1 * PAR / (AmRdark) * self.adpexp_dCO2
        self.adAmRdark_dCO2 += alphac * PAR/(AmRdark**2) * self.adpexp_dCO2
        self.adpexp_dCO2 = 0
        #statement dpexp_de    = dAmRdark_de*alphac*PAR/(AmRdark**2)
        self.adAmRdark_de += alphac*PAR/(AmRdark**2) * self.adpexp_de
        self.adpexp_de = 0
        #statement dpexp_dTs    = dAmRdark_dTs*alphac*PAR/(AmRdark**2)
        self.adAmRdark_dTs += alphac*PAR/(AmRdark**2) * self.adpexp_dTs
        self.adpexp_dTs = 0
        #statement dpexp_dthetasurf = -1 * dalphac_dthetasurf * PAR / (AmRdark) + dAmRdark_dthetasurf * alphac * PAR/(AmRdark**2)
        self.adalphac_dthetasurf += -1 * PAR / (AmRdark) * self.adpexp_dthetasurf
        self.adAmRdark_dthetasurf += alphac * PAR/(AmRdark**2) * self.adpexp_dthetasurf
        self.adpexp_dthetasurf = 0
        #statement dalphac_dCO2 = model.alpha0[c] * ( (CO2comp-co2abs) * dxdiv_dCO2 / (xdiv**2) + dco2abs/xdiv)
        self.adxdiv_dCO2 += model.alpha0[c] *  (CO2comp-co2abs) / (xdiv**2) * self.adalphac_dCO2
        self.adco2abs += model.alpha0[c] * 1 / xdiv * self.adalphac_dCO2
        self.adalphac_dCO2 = 0
        #statement dalphac_dthetasurf = model.alpha0[c] * ( (CO2comp-co2abs) * dxdiv_dthetasurf / (xdiv**2) - dCO2comp_dthetasurf/xdiv)
        self.adxdiv_dthetasurf += model.alpha0[c] * (CO2comp-co2abs) / (xdiv**2) * self.adalphac_dthetasurf
        self.adCO2comp_dthetasurf += model.alpha0[c] * -1 / xdiv * self.adalphac_dthetasurf
        self.adalphac_dthetasurf = 0
        #statement dxdiv_dCO2 = dco2abs
        self.adco2abs += self.adxdiv_dCO2
        self.adxdiv_dCO2 = 0
        #statement dxdiv_dthetasurf      = 2.*dCO2comp_dthetasurf
        self.adCO2comp_dthetasurf += 2.* self.adxdiv_dthetasurf
        self.adxdiv_dthetasurf = 0
        if Swin  * cveg < 1e-1:
            #statement dPAR_dcveg = 0
            self.adPAR_dcveg = 0
            #statement dPAR_dSwin = 0
            self.adPAR_dSwin = 0
            #statement dPAR_dPARfract = 1e-1 * self.dPARfract
            self.aself.dPARfract += 1e-1 * self.adPAR_dPARfract
            self.adPAR_dPARfract = 0
        else:
            #statement dPAR_dcveg   = PARfract * dSwina_dcveg
            self.adSwina_dcveg += PARfract * self.adPAR_dcveg
            self.adPAR_dcveg = 0
            #statement dPAR_dSwin   = PARfract * dSwina_dSwin
            self.adSwina_dSwin += PARfract * self.adPAR_dSwin
            self.adPAR_dSwin = 0
            #statement dPAR_dPARfract = Swin * cveg * self.dPARfract
            self.adPARfract += Swin * cveg * self.adPAR_dPARfract
            self.adPAR_dPARfract = 0
        #statement dSwina_dcveg = Swin  * self.dcveg
        self.adcveg += Swin  * self.adSwina_dcveg
        self.adSwina_dcveg = 0
        #statement dSwina_dSwin = dSwin * cveg
        self.adSwin += cveg * self.adSwina_dSwin
        self.adSwina_dSwin = 0
        #statement dAmRdark_dCO2 = dAm_dCO2 + dRdark_dCO2
        self.adAm_dCO2 += self.adAmRdark_dCO2
        self.adRdark_dCO2 += self.adAmRdark_dCO2
        self.adAmRdark_dCO2 = 0.0
        #statement dAmRdark_de  = dAm_de + dRdark_de
        self.adAm_de += self.adAmRdark_de
        self.adRdark_de += self.adAmRdark_de
        self.adAmRdark_de = 0
        #statement dAmRdark_dTs  = dAm_dTs + dRdark_dTs
        self.adAm_dTs += self.adAmRdark_dTs
        self.adRdark_dTs += self.adAmRdark_dTs
        self.adAmRdark_dTs = 0
        #statement dAmRdark_dthetasurf = dAm_dthetasurf + dRdark_dthetasurf
        self.adAm_dthetasurf += self.adAmRdark_dthetasurf
        self.adRdark_dthetasurf += self.adAmRdark_dthetasurf
        self.adAmRdark_dthetasurf = 0
        #statement dRdark_dCO2  = (1. / 9.) * dAm_dCO2
        self.adAm_dCO2 += (1. / 9.) * self.adRdark_dCO2
        self.adRdark_dCO2 = 0
        #statement dRdark_de   = (1. / 9.) * dAm_de
        self.adAm_de += (1. / 9.) * self.adRdark_de
        self.adRdark_de = 0
        #statement dRdark_dTs   = (1. / 9.) * dAm_dTs
        self.adAm_dTs += (1. / 9.) * self.adRdark_dTs
        self.adRdark_dTs = 0
        #statement dRdark_dthetasurf  = (1. / 9.) * dAm_dthetasurf
        self.adAm_dthetasurf += (1. / 9.) * self.adRdark_dthetasurf
        self.adRdark_dthetasurf = 0
        #statement dAm_dCO2 = -Ammax * daexp_dCO2 * np.exp(aexp)
        self.adaexp_dCO2 += -Ammax * np.exp(aexp) * self.adAm_dCO2
        self.adAm_dCO2 = 0
        #statement dAm_de = -Ammax * daexp_de  * np.exp(aexp)
        self.adaexp_de += -Ammax * np.exp(aexp) * self.adAm_de
        self.adAm_de  = 0
        #statement dAm_dTs = -Ammax * daexp_dTs  * np.exp(aexp)
        self.adaexp_dTs += -Ammax * np.exp(aexp) * self.adAm_dTs
        self.adAm_dTs  = 0
        #statement dAm_dthetasurf = dAmmax_dthetasurf * (1. - np.exp(aexp)) - Ammax*daexp_dthetasurf*np.exp(aexp)
        self.adAmmax_dthetasurf += (1. - np.exp(aexp)) * self.adAm_dthetasurf
        self.adaexp_dthetasurf += - Ammax*np.exp(aexp) * self.adAm_dthetasurf
        self.adAm_dthetasurf = 0
        #statement daexp_dCO2    = -dci_dCO2 * gm / Ammax
        self.adci_dCO2 += -1 * gm / Ammax * self.adaexp_dCO2
        self.adaexp_dCO2 = 0
        #statement daexp_de   = -dci_de  * gm / Ammax
        self.adci_de += -1 * gm / Ammax * self.adaexp_de
        self.adaexp_de = 0
        #statement daexp_dTs   = -dci_dTs  * gm / Ammax
        self.adci_dTs += -1 * gm / Ammax * self.adaexp_dTs
        self.adaexp_dTs = 0
        #statement daexp_dthetasurf = -dgm_dthetasurf*ci/Ammax - dci_dthetasurf*gm/Ammax + dAmmax_dthetasurf*gm*ci/(Ammax**2) + 
        # dgm_dthetasurf*CO2comp/Ammax + dCO2comp_dthetasurf*gm/Ammax - dAmmax_dthetasurf*gm*CO2comp/(Ammax**2)
        self.adgm_dthetasurf += -1 * ci/Ammax * self.adaexp_dthetasurf
        self.adci_dthetasurf += -1 * gm/Ammax * self.adaexp_dthetasurf
        self.adAmmax_dthetasurf += gm*ci/(Ammax**2) * self.adaexp_dthetasurf
        self.adgm_dthetasurf += CO2comp/Ammax * self.adaexp_dthetasurf
        self.adCO2comp_dthetasurf += gm/Ammax * self.adaexp_dthetasurf
        self.adAmmax_dthetasurf += -1 *gm*CO2comp/(Ammax**2) * self.adaexp_dthetasurf
        self.adaexp_dthetasurf = 0
        if (model.c_beta == 0):
            #statement dfstr_dwwilt = dbetaw_dwwilt
            self.adbetaw_dwwilt += self.adfstr_dwwilt
            self.adfstr_dwwilt = 0
            #statement dfstr_dwfc = dbetaw_dwfc
            self.adbetaw_dwfc += self.adfstr_dwfc
            self.adfstr_dwfc = 0
            #statement dfstr_dw2  = dbetaw_dw2
            self.adbetaw_dw2 += self.adfstr_dw2
            self.adfstr_dw2  = 0
        else:
            P = checkpoint['ags_P_end']
            betaw = checkpoint['ags_betaw_end']
            #statement dfstr_dwwilt  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dwwilt
            self.adbetaw_dwwilt += 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * self.adfstr_dwwilt
            self.adfstr_dwwilt = 0
            #statement dfstr_dwfc  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dwfc
            self.adbetaw_dwfc += 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * self.adfstr_dwfc
            self.adfstr_dwfc = 0
            #statement dfstr_dw2  = 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * dbetaw_dw2
            self.adbetaw_dw2 += 1 / (1 - np.exp(-P)) * -1 * np.exp(-P * betaw) * -P * self.adfstr_dw2
            self.adfstr_dw2 = 0
        if (w2 - wwilt)/(wfc - wwilt) > 1 or (w2 - wwilt)/(wfc - wwilt) < 1e-3:
            #statement dbetaw_dwwilt = 0
            self.adbetaw_dwwilt = 0
            #statement dbetaw_dwfc = 0
            self.adbetaw_dwfc = 0
            #statement dbetaw_dw2 = 0
            self.adbetaw_dw2 = 0
        else:
            #statement dbetaw_dwwilt = -self.dwwilt / (wfc - wwilt) + (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * -self.dwwilt
            self.adwwilt += -1 / (wfc - wwilt) * self.adbetaw_dwwilt
            self.adwwilt += (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * -self.adbetaw_dwwilt
            self.adbetaw_dwwilt = 0
            #statement dbetaw_dwfc = (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * self.dwfc
            self.adwfc += (w2 - wwilt) * -1 * (wfc - wwilt)**(-2) * self.adbetaw_dwfc
            self.adbetaw_dwfc = 0
            #statement dbetaw_dw2 = dw2/(wfc - wwilt)
            self.adw2 += 1/(wfc - wwilt) * self.adbetaw_dw2
            self.adbetaw_dw2 = 0        
        #statement dAmmax_dthetasurf = dAmmax1_dthetasurf/(Ammax2*Ammax3) - Ammax1*dAmmax2_dthetasurf/(Ammax3*Ammax2**2) - Ammax1*dAmmax3_dthetasurf/(Ammax2*Ammax3**2)
        self.adAmmax1_dthetasurf += 1/(Ammax2*Ammax3) * self.adAmmax_dthetasurf
        self.adAmmax2_dthetasurf += -1*Ammax1/(Ammax3*Ammax2**2) * self.adAmmax_dthetasurf
        self.adAmmax3_dthetasurf += - Ammax1/(Ammax2*Ammax3**2) * self.adAmmax_dthetasurf
        self.adAmmax_dthetasurf = 0
        #statement dAmmax3_dthetasurf = +0.3*dthetasurf*np.exp(0.3 * (thetasurf - model.T2Am[c]))
        self.adthetasurf += 0.3*np.exp(0.3 * (thetasurf - model.T2Am[c])) * self.adAmmax3_dthetasurf
        self.adAmmax3_dthetasurf = 0
        #statement dAmmax2_dthetasurf = -0.3*dthetasurf*np.exp(0.3 * (model.T1Am[c] - thetasurf))
        self.adthetasurf += -0.3 * np.exp(0.3 * (model.T1Am[c] - thetasurf)) * self.adAmmax2_dthetasurf
        self.adAmmax2_dthetasurf = 0
        #statement dAmmax1_dthetasurf = 0.1*dthetasurf*np.log(model.Q10Am[c])* model.Ammax298[c] *  pow(model.Q10Am[c],(0.1 * (thetasurf - 298.)))
        self.adthetasurf += 0.1*np.log(model.Q10Am[c])* model.Ammax298[c] *  pow(model.Q10Am[c],(0.1 * (thetasurf - 298.))) * self.adAmmax1_dthetasurf
        self.adAmmax1_dthetasurf = 0
        #statement dci_dCO2      = cfrac * dco2abs
        self.adco2abs += cfrac * self.adci_dCO2
        self.adci_dCO2 = 0
        #statement dci_de = dcfrac_de * (co2abs - CO2comp)
        self.adcfrac_de += (co2abs - CO2comp) * self.adci_de
        self.adci_de = 0
        #statement dci_dTs = dcfrac_dTs * (co2abs - CO2comp)
        self.adcfrac_dTs += (co2abs - CO2comp) * self.adci_dTs
        self.adci_dTs = 0
        #statement dci_dthetasurf = dCO2comp_dthetasurf *(1.- cfrac)
        self.adCO2comp_dthetasurf += (1.- cfrac) * self.adci_dthetasurf
        self.adci_dthetasurf = 0
        if model.ags_C_mode == 'MXL':
            #statement dco2abs = dco2abs_dCO2
            self.adco2abs_dCO2 += self.adco2abs
            self.adco2abs = 0
            #statement dco2abs_dCO2  = dCO2* (model.mco2 / model.mair) * model.rho 
            self.adCO2 += (model.mco2 / model.mair) * model.rho * self.adco2abs_dCO2
            self.adco2abs_dCO2 = 0
        elif model.ags_C_mode == 'surf':
            #statement dco2abs = dco2abs_dCO2surf
            self.adco2abs_dCO2surf += self.adco2abs
            self.adco2abs = 0
            #statement dco2abs_dCO2surf  = self.dCO2surf * (model.mco2 / model.mair) * model.rho
            self.adCO2surf += (model.mco2 / model.mair) * model.rho * self.adco2abs_dCO2surf
            self.adco2abs_dCO2surf = 0
        else:
            raise Exception('wrong ags_C_mode switch')
        #statement dcfrac_de    = -model.f0[c]*(dDs_de/D0) + fmin*(dDs_de/D0)
        self.adDs_de += -model.f0[c] / D0 * self.adcfrac_de
        self.adDs_de += fmin/D0 * self.adcfrac_de
        self.adcfrac_de = 0
        #statement dcfrac_dTs    = -model.f0[c]*(dDs_dTs/D0) + fmin*(dDs_dTs/D0)
        self.adDs_dTs += -model.f0[c] / D0 * self.adcfrac_dTs
        self.adDs_dTs += fmin/D0 * self.adcfrac_dTs
        self.adcfrac_dTs = 0
        #statement dDs_de        = -self.de / 1000
        self.ade += -self.adDs_de / 1000
        self.adDs_de = 0
        #statement dDs_dTs  = (desat(Ts,dTs)-devap)/1000.
        self.adTs  +=  desat(Ts,self.adDs_dTs/1000.)
        self.adevap += -1/1000 * self.adDs_dTs
        self.adDs_dTs = 0
#        if self.manualadjointtesting:
#            self.adcfrac_dthetasurf = self.y
        #statement dD0_dthetasurf  = -dfmin_dthetasurf/model.ad[c]
        self.adfmin_dthetasurf += -1/model.ad[c] * self.adD0_dthetasurf
        self.adD0_dthetasurf = 0
#        if self.manualadjointtesting:
#            self.HTy = self.adfmin_dthetasurf
        #statement dfmin_dthetasurf         = -dfmin0_dthetasurf/(2.*gm) + dsqterm_dthetasurf/(2.*gm)-(-fmin0 + sqterm)*dgm_dthetasurf/(2*gm**2)
        self.adfmin0_dthetasurf += -self.adfmin_dthetasurf/(2.*gm)
        self.adsqterm_dthetasurf += 1/(2.*gm) * self.adfmin_dthetasurf
        self.adgm_dthetasurf += -(-fmin0 + sqterm)*1/(2*gm**2) * self.adfmin_dthetasurf
        self.adfmin_dthetasurf = 0
        #statement dsqterm_dthetasurf = dsqrtf_dthetasurf*0.5*pow(sqrtf,-0.5)
        self.adsqrtf_dthetasurf += 0.5*pow(sqrtf,-0.5) * self.adsqterm_dthetasurf
        self.adsqterm_dthetasurf = 0
        #statement dsqrtf_dthetasurf = 2.*fmin0*dfmin0_dthetasurf +  4*model.gmin[c]/model.nuco2q *dgm_dthetasurf
        self.adfmin0_dthetasurf += 2. * fmin0 * self.adsqrtf_dthetasurf
        self.adgm_dthetasurf += 4*model.gmin[c]/model.nuco2q * self.adsqrtf_dthetasurf
        self.adsqrtf_dthetasurf = 0
        #statement dfmin0_dthetasurf        = -1./9.*dgm_dthetasurf
        self.adgm_dthetasurf += -1./9.*self.adfmin0_dthetasurf
        self.adfmin0_dthetasurf = 0
        #statement dgm_dthetasurf  = dgm_dthetasurf / 1000.
        self.adgm_dthetasurf = self.adgm_dthetasurf/1000
        #statement dgm_dthetasurf  = dgm1_dthetasurf/(gm2*gm3) - gm1*dgm2_dthetasurf/(gm3*gm2**2) - gm1*dgm3_dthetasurf/(gm2*gm3**2)
        self.adgm1_dthetasurf += 1/(gm2*gm3) * self.adgm_dthetasurf
        self.adgm2_dthetasurf += - gm1*1/(gm3*gm2**2) * self.adgm_dthetasurf
        self.adgm3_dthetasurf += - gm1*1/(gm2*gm3**2) * self.adgm_dthetasurf
        self.adgm_dthetasurf = 0
        #statement dgm3_dthetasurf =   0.3*dthetasurf*np.exp(0.3 * (thetasurf - model.T2gm[c]))
        self.adthetasurf += 0.3*np.exp(0.3 * (thetasurf - model.T2gm[c])) * self.adgm3_dthetasurf
        self.adgm3_dthetasurf = 0
        #statement dgm2_dthetasurf =  -0.3*dthetasurf*np.exp(0.3 * (model.T1gm[c] - thetasurf))
        self.adthetasurf += -0.3*np.exp(0.3 * (model.T1gm[c] - thetasurf)) * self.adgm2_dthetasurf
        self.adgm2_dthetasurf = 0
        #statement dgm1_dthetasurf = 0.1* dthetasurf*np.log(model.Q10gm[c])* model.gm298[c] *  pow(model.Q10gm[c],(0.1 * (thetasurf-298.)))  
        self.adthetasurf += 0.1* np.log(model.Q10gm[c])* model.gm298[c] *  pow(model.Q10gm[c],(0.1 * (thetasurf-298.))) * self.adgm1_dthetasurf
        self.adgm1_dthetasurf = 0
        #statement dCO2comp_dthetasurf = model.CO2comp298[c] * model.rho * np.log(model.Q10CO2[c]) * pow(model.Q10CO2[c],(0.1 * (thetasurf - 298.))) * 0.1* dthetasurf
        self.adthetasurf += model.CO2comp298[c] * model.rho * np.log(model.Q10CO2[c]) * pow(model.Q10CO2[c],(0.1 * (thetasurf - 298.))) * 0.1 * self.adCO2comp_dthetasurf
        self.adCO2comp_dthetasurf = 0
        if self.adjointtestingags:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
                      
    def adj_run_surface_layer(self,forcing,checkpoint,model,HTy_variables=None):
        ueff = checkpoint['rsl_ueff_end']
        q = checkpoint['rsl_q']
        qsurf = checkpoint['rsl_qsurf_end']
        thetasurf = checkpoint['rsl_thetasurf_end']
        wq = checkpoint['rsl_wq']
        thetav = checkpoint['rsl_thetav']
        wthetav = checkpoint['rsl_wthetav']
        zsl = checkpoint['rsl_zsl_end']
        L = checkpoint['rsl_L_end']
        wtheta = checkpoint['rsl_wtheta']
        Cm = checkpoint['rsl_Cm_end']
        ustar = checkpoint['rsl_ustar_end']
        Cs_start = checkpoint['rsl_Cs']
        ustar_start = checkpoint['rsl_ustar']
        wCOS = checkpoint['rsl_wCOS']
        wCO2 = checkpoint['rsl_wCO2']
        T2m = checkpoint['rsl_T2m_end']
        vw = checkpoint['rsl_vw_end']
        uw = checkpoint['rsl_uw_end']
        v = checkpoint['rsl_v']
        u = checkpoint['rsl_u']
        wstar = checkpoint['rsl_wstar']
        #from old qsurf implementation
        #qsatsurf_rsl = checkpoint['rsl_qsatsurf_rsl_end']
        #cq = checkpoint['rsl_cq_end']
        #rs = checkpoint['rsl_rs']
        COSmeasuring_height = checkpoint['rsl_COSmeasuring_height']
        z0m = checkpoint['rsl_z0m']
        z0h = checkpoint['rsl_z0h']
        thetavsurf = checkpoint['rsl_thetavsurf_end']
        if model.sw_use_ribtol:
            Rib = checkpoint['rsl_Rib_middle']
        
        if model.sw_dynamicsl_border:
            if model.CO2measuring_height4 > zsl:
                #statement dCO2mh4 = self.dCO2
                self.adCO2 += self.adCO2mh4
                self.adCO2mh4 = 0
            if model.CO2measuring_height3 > zsl:
                #statement dCO2mh3 = self.dCO2
                self.adCO2 += self.adCO2mh3
                self.adCO2mh3 = 0
            if model.CO2measuring_height2 > zsl:
                #statement dCO2mh2 = self.dCO2
                self.adCO2 += self.adCO2mh2
                self.adCO2mh2 = 0
            if model.CO2measuring_height > zsl:
                #statement dCO2mh = self.dCO2
                self.adCO2 += self.adCO2mh
                self.adCO2mh = 0
            if model.COSmeasuring_height3 > zsl:
                #statement dCOSmh3 = self.dCOS
                self.adCOS += self.adCOSmh3
                self.adCOSmh3 = 0
            if model.COSmeasuring_height2 > zsl:
                #statement dCOSmh2 = self.dCOS
                self.adCOS += self.adCOSmh2
                self.adCOSmh2 = 0
            if COSmeasuring_height > zsl:
                #statement dCOSmh = self.dCOS
                self.adCOS += self.adCOSmh
                self.adCOSmh = 0
            if model.qmeasuring_height7 > zsl:
                #statement dqmh7 = self.dq
                self.adq += self.adqmh7
                self.adqmh7 = 0
            if model.qmeasuring_height6 > zsl:
                #statement dqmh6 = self.dq
                self.adq += self.adqmh6
                self.adqmh6 = 0
            if model.qmeasuring_height5 > zsl:
                #statement dqmh5 = self.dq
                self.adq += self.adqmh5
                self.adqmh5 = 0
            if model.qmeasuring_height4 > zsl:
                #statement dqmh4 = self.dq
                self.adq += self.adqmh4
                self.adqmh4 = 0
            if model.qmeasuring_height3 > zsl:
                #statement dqmh3 = self.dq
                self.adq += self.adqmh3
                self.adqmh3 = 0
            if model.qmeasuring_height2 > zsl:
                #statement dqmh2 = self.dq
                self.adq += self.adqmh2
                self.adqmh2 = 0
            if model.qmeasuring_height > zsl:
                #statement dqmh = self.dq
                self.adq += self.adqmh
                self.adqmh = 0
            if model.Tmeasuring_height7 > zsl:
                #statement dTmh7 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp) * self.adTmh7
                self.adTmh7 = 0
                #statement dthetamh7 = self.dtheta
                self.adtheta += self.adthetamh7
                self.adthetamh7 = 0
            if model.Tmeasuring_height6 > zsl:
                #statement dTmh6 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp) * self.adTmh6
                self.adTmh6 = 0
                #statement dthetamh6 = self.dtheta
                self.adtheta += self.adthetamh6
                self.adthetamh6 = 0
            if model.Tmeasuring_height5 > zsl:
                #statement dTmh5 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp) * self.adTmh5
                self.adTmh5 = 0
                #statement dthetamh5 = self.dtheta
                self.adtheta += self.adthetamh5
                self.adthetamh5 = 0
            if model.Tmeasuring_height4 > zsl:
                #statement dTmh4 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp) * self.adTmh4
                self.adTmh4 = 0
                #statement dthetamh4 = self.dtheta
                self.adtheta += self.adthetamh4
                self.adthetamh4 = 0
            if model.Tmeasuring_height3 > zsl:
                #statement dTmh3 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp) * self.adTmh3
                self.adTmh3 = 0
                #statement dthetamh3 = self.dtheta
                self.adtheta += self.adthetamh3
                self.adthetamh3 = 0
            if model.Tmeasuring_height2 > zsl:
                #statement dTmh2 = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp) * self.adTmh2
                self.adTmh2 = 0
                #statement dthetamh2 = self.dtheta
                self.adtheta += self.adthetamh2
                self.adthetamh2 = 0
            if model.Tmeasuring_height > zsl:
                #statement dTmh = self.dtheta * ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp)
                self.adtheta += ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp) * self.adTmh
                self.adTmh = 0
                #statement dthetamh = self.dtheta
                self.adtheta += self.adthetamh
                self.adthetamh = 0
        
#        if self.adjointtesting:
#            self.adT2m = self.y
        #statement de2m = 1 * model.Ps / 0.622 * dq2m
        self.adq2m = 1 * model.Ps / 0.622 * self.ade2m + self.adq2m
        self.ade2m = 0
        #statement desat2m = desat(T2m,dT2m)
        self.adT2m = desat(T2m,self.adesat2m) + self.adT2m
        self.adesat2m = 0
        #statement dv2m = dv2m_dvw + dv2m_dustar + dv2m_dz0m + dv2m_dL
        self.adv2m_dvw = self.adv2m + self.adv2m_dvw
        self.adv2m_dustar = self.adv2m + self.adv2m_dustar
        self.adv2m_dz0m += self.adv2m
        self.adv2m_dL = self.adv2m + self.adv2m_dL
        self.adv2m = 0
        #statement dv2m_dL = - vw / ustar / model.k * (- dpsim_2_L + dpsim_z0m_L)
        self.adpsim_2_L = - vw / ustar / model.k * (- self.adv2m_dL) + self.adpsim_2_L
        self.adpsim_z0m_L = - vw / ustar / model.k * (self.adv2m_dL) + self.adpsim_z0m_L
        self.adv2m_dL = 0
        #statement dv2m_dz0m = - vw / ustar / model.k * (1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        self.adz0m += - vw / ustar / model.k * 1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.adv2m_dz0m
        self.adpsimterm_for_dCm_dz0m += - vw / ustar / model.k * self.adv2m_dz0m
        self.adv2m_dz0m = 0
        #statement dv2m_dustar = - vw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * dustar 
        self.adustar = - vw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * self.adv2m_dustar + self.adustar
        self.adv2m_dustar = 0        
        #statement dv2m_dvw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * dvw
        self.advw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * self.adv2m_dvw + self.advw
        self.adv2m_dvw = 0
        #statement du2m = du2m_duw + du2m_dustar + du2m_dz0m + du2m_dL
        self.adu2m_duw = self.adu2m + self.adu2m_duw
        self.adu2m_dustar = self.adu2m + self.adu2m_dustar
        self.adu2m_dz0m += self.adu2m
        self.adu2m_dL = self.adu2m + self.adu2m_dL
        self.adu2m = 0
        #statement du2m_dL = - uw / ustar / model.k * (- dpsim_2_L + dpsim_z0m_L)
        self.adpsim_2_L = - uw / ustar / model.k * (-1) * self.adu2m_dL + self.adpsim_2_L
        self.adpsim_z0m_L = - uw / ustar / model.k * self.adu2m_dL + self.adpsim_z0m_L 
        self.adu2m_dL = 0
        #statement dpsim_z0m_L = self.dpsim(z0m / L,dzeta_dL_z0m)
        self.adzeta_dL_z0m = self.dpsim(z0m / L,self.adpsim_z0m_L) + self.adzeta_dL_z0m
        self.adpsim_z0m_L = 0
        #statement dpsim_2_L = self.dpsim(2. / L,dzeta_dL_2)
        self.adzeta_dL_2 = self.dpsim(2. / L,self.adpsim_2_L) + self.adzeta_dL_2
        self.adpsim_2_L = 0
        #statement du2m_dz0m = - uw / ustar / model.k * (1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        self.adz0m += - uw / ustar / model.k * 1 / (2. / z0m) * 2 * -1 * z0m**-2 * self.adu2m_dz0m
        self.adpsimterm_for_dCm_dz0m += - uw / ustar / model.k * self.adu2m_dz0m
        self.adu2m_dz0m = 0
        #statement du2m_dustar = - uw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - uw / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * (-1) * ustar**(-2) * self.adu2m_dustar + self.adustar
        self.adu2m_dustar = 0
        #statement du2m_duw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * duw
        self.aduw = - 1 / ustar / model.k * (np.log(2. / z0m) - model.psim(2. / L) + model.psim(z0m / L)) * self.adu2m_duw + self.aduw
        self.adu2m_duw = 0
        #statement dCO2mh4 = dCO2mh4_dCO2surf + dCO2mh4_dwCO2 + dCO2mh4_dustar + dCO2mh4_dz0h + dCO2mh4_dL
        self.adCO2mh4_dCO2surf = self.adCO2mh4 + self.adCO2mh4_dCO2surf
        self.adCO2mh4_dwCO2 = self.adCO2mh4 + self.adCO2mh4_dwCO2
        self.adCO2mh4_dustar = self.adCO2mh4 + self.adCO2mh4_dustar
        self.adCO2mh4_dz0h += self.adCO2mh4
        self.adCO2mh4_dL = self.adCO2mh4 + self.adCO2mh4_dL
        self.adCO2mh4 = 0
        #statement dCO2mh4_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh4_L + dpsih_z0h_L)
        self.adpsih_CO2mh4_L = - wCO2 / ustar / model.k * (-1) * self.adCO2mh4_dL + self.adpsih_CO2mh4_L
        self.adpsih_z0h_L = - wCO2 / ustar / model.k * self.adCO2mh4_dL + self.adpsih_z0h_L
        self.adCO2mh4_dL = 0
        #statement dpsih_CO2mh4_L = self.dpsih(model.CO2measuring_height4 / L,dzeta_dL_CO2mh4)
        self.adzeta_dL_CO2mh4 = self.dpsih(model.CO2measuring_height4 / L,self.adpsih_CO2mh4_L) + self.adzeta_dL_CO2mh4
        self.adpsih_CO2mh4_L = 0
        #statement dCO2mh4_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height4 / z0h) * model.CO2measuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCO2 / ustar / model.k * 1 / (model.CO2measuring_height4 / z0h) * model.CO2measuring_height4 * -1 * z0h**-2 * self.adCO2mh4_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCO2 / ustar / model.k * self.adCO2mh4_dz0h
        self.adCO2mh4_dz0h = 0
        #statement dCO2mh4_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCO2 / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCO2mh4_dustar + self.adustar
        self.adCO2mh4_dustar = 0
        #statement dCO2mh4_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * self.dwCO2
        self.adwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height4 / z0h) - model.psih(model.CO2measuring_height4 / L) + model.psih(z0h / L)) * self.adCO2mh4_dwCO2 + self.adwCO2
        self.adCO2mh4_dwCO2 = 0
        #statement dCO2mh4_dCO2surf = dCO2surf
        self.adCO2surf = self.adCO2mh4_dCO2surf + self.adCO2surf
        self.adCO2mh4_dCO2surf = 0
        #statement dCO2mh3 = dCO2mh3_dCO2surf + dCO2mh3_dwCO2 + dCO2mh3_dustar + dCO2mh3_dz0h + dCO2mh3_dL
        self.adCO2mh3_dCO2surf = self.adCO2mh3 + self.adCO2mh3_dCO2surf
        self.adCO2mh3_dwCO2 = self.adCO2mh3 + self.adCO2mh3_dwCO2
        self.adCO2mh3_dustar = self.adCO2mh3 + self.adCO2mh3_dustar
        self.adCO2mh3_dz0h += self.adCO2mh3
        self.adCO2mh3_dL = self.adCO2mh3 + self.adCO2mh3_dL
        self.adCO2mh3 = 0
        #statement dCO2mh3_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh3_L + dpsih_z0h_L)
        self.adpsih_CO2mh3_L = - wCO2 / ustar / model.k * (-1) * self.adCO2mh3_dL + self.adpsih_CO2mh3_L
        self.adpsih_z0h_L = - wCO2 / ustar / model.k * self.adCO2mh3_dL + self.adpsih_z0h_L
        self.adCO2mh3_dL = 0
        #statement dpsih_CO2mh3_L = self.dpsih(model.CO2measuring_height3 / L,dzeta_dL_CO2mh3)
        self.adzeta_dL_CO2mh3 = self.dpsih(model.CO2measuring_height3 / L,self.adpsih_CO2mh3_L) + self.adzeta_dL_CO2mh3
        self.adpsih_CO2mh3_L = 0
        #statement dCO2mh3_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height3 / z0h) * model.CO2measuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCO2 / ustar / model.k * 1 / (model.CO2measuring_height3 / z0h) * model.CO2measuring_height3 * -1 * z0h**-2 * self.adCO2mh3_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCO2 / ustar / model.k * self.adCO2mh3_dz0h
        self.adCO2mh3_dz0h = 0
        #statement dCO2mh3_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCO2 / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCO2mh3_dustar + self.adustar
        self.adCO2mh3_dustar = 0
        #statement dCO2mh3_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * self.dwCO2
        self.adwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height3 / z0h) - model.psih(model.CO2measuring_height3 / L) + model.psih(z0h / L)) * self.adCO2mh3_dwCO2 + self.adwCO2
        self.adCO2mh3_dwCO2 = 0
        #statement dCO2mh3_dCO2surf = dCO2surf
        self.adCO2surf = self.adCO2mh3_dCO2surf + self.adCO2surf
        self.adCO2mh3_dCO2surf = 0
        #statement dCO2mh2 = dCO2mh2_dCO2surf + dCO2mh2_dwCO2 + dCO2mh2_dustar + dCO2mh2_dz0h + dCO2mh2_dL
        self.adCO2mh2_dCO2surf = self.adCO2mh2 + self.adCO2mh2_dCO2surf
        self.adCO2mh2_dwCO2 = self.adCO2mh2 + self.adCO2mh2_dwCO2
        self.adCO2mh2_dustar = self.adCO2mh2 + self.adCO2mh2_dustar
        self.adCO2mh2_dz0h += self.adCO2mh2
        self.adCO2mh2_dL = self.adCO2mh2 + self.adCO2mh2_dL
        self.adCO2mh2 = 0
        #statement dCO2mh2_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh2_L + dpsih_z0h_L)
        self.adpsih_CO2mh2_L = - wCO2 / ustar / model.k * (-1) * self.adCO2mh2_dL + self.adpsih_CO2mh2_L
        self.adpsih_z0h_L = - wCO2 / ustar / model.k * self.adCO2mh2_dL + self.adpsih_z0h_L
        self.adCO2mh2_dL = 0
        #statement dpsih_CO2mh2_L = self.dpsih(model.CO2measuring_height2 / L,dzeta_dL_CO2mh2)
        self.adzeta_dL_CO2mh2 = self.dpsih(model.CO2measuring_height2 / L,self.adpsih_CO2mh2_L) + self.adzeta_dL_CO2mh2
        self.adpsih_CO2mh2_L = 0
        #statement dCO2mh2_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height2 / z0h) * model.CO2measuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCO2 / ustar / model.k * 1 / (model.CO2measuring_height2 / z0h) * model.CO2measuring_height2 * -1 * z0h**-2 * self.adCO2mh2_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCO2 / ustar / model.k * self.adCO2mh2_dz0h
        self.adCO2mh2_dz0h = 0
        #statement dCO2mh2_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCO2 / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCO2mh2_dustar + self.adustar
        self.adCO2mh2_dustar = 0
        #statement dCO2mh2_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * self.dwCO2
        self.adwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height2 / z0h) - model.psih(model.CO2measuring_height2 / L) + model.psih(z0h / L)) * self.adCO2mh2_dwCO2 + self.adwCO2
        self.adCO2mh2_dwCO2 = 0
        #statement dCO2mh2_dCO2surf = dCO2surf
        self.adCO2surf = self.adCO2mh2_dCO2surf + self.adCO2surf
        self.adCO2mh2_dCO2surf = 0
        #statement dCO2mh = dCO2mh_dCO2surf + dCO2mh_dwCO2 + dCO2mh_dustar + dCO2mh_dz0h + dCO2mh_dL
        self.adCO2mh_dCO2surf = self.adCO2mh + self.adCO2mh_dCO2surf
        self.adCO2mh_dwCO2 = self.adCO2mh + self.adCO2mh_dwCO2
        self.adCO2mh_dustar = self.adCO2mh + self.adCO2mh_dustar
        self.adCO2mh_dz0h += self.adCO2mh
        self.adCO2mh_dL = self.adCO2mh + self.adCO2mh_dL
        self.adCO2mh = 0
        #statement dCO2mh_dL = - wCO2 / ustar / model.k * (- dpsih_CO2mh_L + dpsih_z0h_L)
        self.adpsih_CO2mh_L = - wCO2 / ustar / model.k * (-1) * self.adCO2mh_dL + self.adpsih_CO2mh_L
        self.adpsih_z0h_L = - wCO2 / ustar / model.k * self.adCO2mh_dL + self.adpsih_z0h_L
        self.adCO2mh_dL = 0
        #statement dpsih_CO2mh_L = self.dpsih(model.CO2measuring_height / L,dzeta_dL_CO2mh)
        self.adzeta_dL_CO2mh = self.dpsih(model.CO2measuring_height / L,self.adpsih_CO2mh_L) + self.adzeta_dL_CO2mh
        self.adpsih_CO2mh_L = 0
        #statement dCO2mh_dz0h = - wCO2 / ustar / model.k * (1 / (model.CO2measuring_height / z0h) * model.CO2measuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCO2 / ustar / model.k * 1 / (model.CO2measuring_height / z0h) * model.CO2measuring_height * -1 * z0h**-2 * self.adCO2mh_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCO2 / ustar / model.k * self.adCO2mh_dz0h
        self.adCO2mh_dz0h = 0
        #statement dCO2mh_dustar = - wCO2 / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCO2 / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCO2mh_dustar + self.adustar
        self.adCO2mh_dustar = 0
        #statement dCO2mh_dwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * self.dwCO2
        self.adwCO2 = - 1 / ustar / model.k * (np.log(model.CO2measuring_height / z0h) - model.psih(model.CO2measuring_height / L) + model.psih(z0h / L)) * self.adCO2mh_dwCO2 + self.adwCO2
        self.adCO2mh_dwCO2 = 0
        #statement dCO2mh_dCO2surf = dCO2surf
        self.adCO2surf = self.adCO2mh_dCO2surf + self.adCO2surf
        self.adCO2mh_dCO2surf = 0
        #statement dCO22m = dCO22m_dCO2surf + dCO22m_dwCO2 + dCO22m_dustar + dCO22m_dz0h + dCO22m_dL
        self.adCO22m_dCO2surf += self.adCO22m
        self.adCO22m_dwCO2 += self.adCO22m
        self.adCO22m_dustar += self.adCO22m
        self.adCO22m_dz0h += self.adCO22m
        self.adCO22m_dL += self.adCO22m
        self.adCO22m = 0
        #statement dCO22m_dL = - wCO2 / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        self.adpsih_2_L += - wCO2 / ustar / model.k * -1 * self.adCO22m_dL
        self.adpsih_z0h_L += - wCO2 / ustar / model.k * self.adCO22m_dL
        self.adCO22m_dL = 0
        #statement dCO22m_dz0h = - wCO2 / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCO2 / ustar / model.k * 1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.adCO22m_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCO2 / ustar / model.k * self.adCO22m_dz0h
        self.adCO22m_dz0h = 0
        #statement dCO22m_dustar = - wCO2 / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar += - wCO2 / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCO22m_dustar
        self.adCO22m_dustar = 0
        #statement dCO22m_dwCO2 = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwCO2
        self.adwCO2 += - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.adCO22m_dwCO2
        self.adCO22m_dwCO2 = 0
        #statement dCO22m_dCO2surf = dCO2surf
        self.adCO2surf += self.adCO22m_dCO2surf
        self.adCO22m_dCO2surf = 0
        #statement dCOSmh3 = dCOSmh3_dCOSsurf + dCOSmh3_dwCOS + dCOSmh3_dustar + dCOSmh3_dz0h + dCOSmh3_dL
        self.adCOSmh3_dCOSsurf = self.adCOSmh3 + self.adCOSmh3_dCOSsurf
        self.adCOSmh3_dwCOS = self.adCOSmh3 + self.adCOSmh3_dwCOS
        self.adCOSmh3_dustar = self.adCOSmh3 + self.adCOSmh3_dustar
        self.adCOSmh3_dz0h += self.adCOSmh3
        self.adCOSmh3_dL = self.adCOSmh3 + self.adCOSmh3_dL
        self.adCOSmh3 = 0
        #statement dCOSmh3_dL = - wCOS / ustar / model.k * (- dpsih_COSmh3_L + dpsih_z0h_L)
        self.adpsih_COSmh3_L = - wCOS / ustar / model.k * (-1) * self.adCOSmh3_dL + self.adpsih_COSmh3_L
        self.adpsih_z0h_L = - wCOS / ustar / model.k * self.adCOSmh3_dL + self.adpsih_z0h_L
        self.adCOSmh3_dL = 0
        #statement dpsih_COSmh3_L = self.dpsih(model.COSmeasuring_height3 / L,dzeta_dL_COSmh3)
        self.adzeta_dL_COSmh3 = self.dpsih(model.COSmeasuring_height3 / L,self.adpsih_COSmh3_L) + self.adzeta_dL_COSmh3
        self.adpsih_COSmh3_L = 0
        #statement dCOSmh3_dz0h = - wCOS / ustar / model.k * (1 / (model.COSmeasuring_height3 / z0h) * model.COSmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCOS / ustar / model.k * 1 / (model.COSmeasuring_height3 / z0h) * model.COSmeasuring_height3 * -1 * z0h**-2 * self.adCOSmh3_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCOS / ustar / model.k * self.adCOSmh3_dz0h
        self.adCOSmh3_dz0h = 0
        #statement dCOSmh3_dustar = - wCOS / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCOS / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCOSmh3_dustar + self.adustar
        self.adCOSmh3_dustar = 0
        #statement dCOSmh3_dwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwCOS
        self.adwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height3 / z0h) - model.psih(model.COSmeasuring_height3 / L) + model.psih(z0h / L)) * self.adCOSmh3_dwCOS + self.adwCOS
        self.adCOSmh3_dwCOS = 0
        #statement dCOSmh3_dCOSsurf = dCOSsurf
        self.adCOSsurf = self.adCOSmh3_dCOSsurf + self.adCOSsurf
        self.adCOSmh3_dCOSsurf = 0
        #statement dCOSmh2 = dCOSmh2_dCOSsurf + dCOSmh2_dwCOS + dCOSmh2_dustar + dCOSmh2_dz0h + dCOSmh2_dL
        self.adCOSmh2_dCOSsurf = self.adCOSmh2 + self.adCOSmh2_dCOSsurf
        self.adCOSmh2_dwCOS = self.adCOSmh2 + self.adCOSmh2_dwCOS
        self.adCOSmh2_dustar = self.adCOSmh2 + self.adCOSmh2_dustar
        self.adCOSmh2_dz0h += self.adCOSmh2
        self.adCOSmh2_dL = self.adCOSmh2 + self.adCOSmh2_dL
        self.adCOSmh2 = 0
        #statement dCOSmh2_dL = - wCOS / ustar / model.k * (- dpsih_COSmh2_L + dpsih_z0h_L)
        self.adpsih_COSmh2_L = - wCOS / ustar / model.k * (-1) * self.adCOSmh2_dL + self.adpsih_COSmh2_L
        self.adpsih_z0h_L = - wCOS / ustar / model.k * self.adCOSmh2_dL + self.adpsih_z0h_L
        self.adCOSmh2_dL = 0
        #statement dpsih_COSmh2_L = self.dpsih(model.COSmeasuring_height2 / L,dzeta_dL_COSmh2)
        self.adzeta_dL_COSmh2 = self.dpsih(model.COSmeasuring_height2 / L,self.adpsih_COSmh2_L) + self.adzeta_dL_COSmh2
        self.adpsih_COSmh2_L = 0
        #statement dCOSmh2_dz0h = - wCOS / ustar / model.k * (1 / (model.COSmeasuring_height2 / z0h) * model.COSmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCOS / ustar / model.k * 1 / (model.COSmeasuring_height2 / z0h) * model.COSmeasuring_height2 * -1 * z0h**-2 * self.adCOSmh2_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCOS / ustar / model.k * self.adCOSmh2_dz0h                            
        self.adCOSmh2_dz0h = 0
        #statement dCOSmh2_dustar = - wCOS / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCOS / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCOSmh2_dustar + self.adustar
        self.adCOSmh2_dustar = 0
        #statement dCOSmh2_dwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwCOS
        self.adwCOS = - 1 / ustar / model.k * (np.log(model.COSmeasuring_height2 / z0h) - model.psih(model.COSmeasuring_height2 / L) + model.psih(z0h / L)) * self.adCOSmh2_dwCOS + self.adwCOS
        self.adCOSmh2_dwCOS = 0
        #statement dCOSmh2_dCOSsurf = dCOSsurf
        self.adCOSsurf = self.adCOSmh2_dCOSsurf + self.adCOSsurf
        self.adCOSmh2_dCOSsurf = 0
        #statement dCOSmh = dCOSmh_dCOSsurf + dCOSmh_dwCOS + dCOSmh_dustar + dCOSmh_dCOSmeasuring_height + dCOSmh_dz0h + dCOSmh_dL
        self.adCOSmh_dCOSsurf = self.adCOSmh + self.adCOSmh_dCOSsurf
        self.adCOSmh_dwCOS = self.adCOSmh + self.adCOSmh_dwCOS
        self.adCOSmh_dustar = self.adCOSmh + self.adCOSmh_dustar
        self.adCOSmh_dCOSmeasuring_height += self.adCOSmh
        self.adCOSmh_dz0h += self.adCOSmh
        self.adCOSmh_dL = self.adCOSmh + self.adCOSmh_dL
        self.adCOSmh = 0
        #statement dCOSmh_dL = - wCOS / ustar / model.k * (- dpsih_COSmh_L + dpsih_z0h_L)
        self.adpsih_COSmh_L = - wCOS / ustar / model.k * (-1) * self.adCOSmh_dL + self.adpsih_COSmh_L
        self.adpsih_z0h_L = - wCOS / ustar / model.k * self.adCOSmh_dL + self.adpsih_z0h_L
        self.adCOSmh_dL = 0
        #statement dpsih_COSmh_L = self.dpsih(COSmeasuring_height / L,dzeta_dL_COSmh)
        self.adzeta_dL_COSmh = self.dpsih(COSmeasuring_height / L,self.adpsih_COSmh_L) + self.adzeta_dL_COSmh
        self.adpsih_COSmh_L = 0
        #statement dCOSmh_dz0h = - wCOS / ustar / model.k * (1 / (COSmeasuring_height / z0h) * COSmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCOS / ustar / model.k * 1 / (COSmeasuring_height / z0h) * COSmeasuring_height * -1 * z0h**-2 * self.adCOSmh_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCOS / ustar / model.k * self.adCOSmh_dz0h 
        self.adCOSmh_dz0h = 0
        #statement dCOSmh_dCOSmeasuring_height = - wCOS / ustar / model.k * (1 / (COSmeasuring_height / z0h) * 1 / z0h * self.dCOSmeasuring_height - dpsih_COSmh_L_num)
        self.adCOSmeasuring_height += - wCOS / ustar / model.k * 1 / (COSmeasuring_height / z0h) * 1 / z0h * self.adCOSmh_dCOSmeasuring_height
        self.adpsih_COSmh_L_num += - wCOS / ustar / model.k * - self.adCOSmh_dCOSmeasuring_height
        self.adCOSmh_dCOSmeasuring_height = 0
        #statement dpsih_COSmh_L_num = self.dpsih(COSmeasuring_height / L,dzeta_dCOSmh)
        self.adzeta_dCOSmh += self.dpsih(COSmeasuring_height / L,self.adpsih_COSmh_L_num)
        self.adpsih_COSmh_L_num = 0
        #statement dzeta_dCOSmh = 1 / L * self.dCOSmeasuring_height
        self.adCOSmeasuring_height += 1 / L * self.adzeta_dCOSmh
        self.adzeta_dCOSmh = 0
        #statement dCOSmh_dustar = - wCOS / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCOS / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCOSmh_dustar + self.adustar
        self.adCOSmh_dustar = 0
        #statement dCOSmh_dwCOS = - 1 / ustar / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * self.dwCOS
        self.adwCOS = - 1 / ustar / model.k * (np.log(COSmeasuring_height / z0h) - model.psih(COSmeasuring_height / L) + model.psih(z0h / L)) * self.adCOSmh_dwCOS + self.adwCOS
        self.adCOSmh_dwCOS = 0
        #statement dCOSmh_dCOSsurf = dCOSsurf
        self.adCOSsurf = self.adCOSmh_dCOSsurf + self.adCOSsurf
        self.adCOSmh_dCOSsurf = 0
        #statement dCOS2m = dCOS2m_dCOSsurf + dCOS2m_dwCOS + dCOS2m_dustar + dCOS2m_dz0h + dCOS2m_dL
        self.adCOS2m_dCOSsurf = self.adCOS2m + self.adCOS2m_dCOSsurf
        self.adCOS2m_dwCOS = self.adCOS2m + self.adCOS2m_dwCOS
        self.adCOS2m_dustar = self.adCOS2m + self.adCOS2m_dustar
        self.adCOS2m_dz0h += self.adCOS2m
        self.adCOS2m_dL = self.adCOS2m + self.adCOS2m_dL
        self.adCOS2m = 0
        #statement dCOS2m_dL = - wCOS / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        self.adpsih_2_L = - wCOS / ustar / model.k * (- 1) * self.adCOS2m_dL + self.adpsih_2_L 
        self.adpsih_z0h_L = - wCOS / ustar / model.k * self.adCOS2m_dL + self.adpsih_z0h_L
        self.adCOS2m_dL = 0
        #statement dCOS2m_dz0h = - wCOS / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wCOS / ustar / model.k * 1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.adCOS2m_dz0h
        self.adpsihterm_for_dCs_dz0h += - wCOS / ustar / model.k * self.adCOS2m_dz0h
        self.adCOS2m_dz0h = 0
        #statement dCOS2m_dustar = - wCOS / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wCOS / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adCOS2m_dustar + self.adustar
        self.adCOS2m_dustar = 0
        #statement dCOS2m_dwCOS = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwCOS
        self.adwCOS = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.adCOS2m_dwCOS + self.adwCOS
        self.adCOS2m_dwCOS = 0
        #statement dCOS2m_dCOSsurf = dCOSsurf
        self.adCOSsurf = self.adCOS2m_dCOSsurf + self.adCOSsurf
        self.adCOS2m_dCOSsurf = 0
        #statement dqmh7 = dqmh7_dqsurf + dqmh7_dwq + dqmh7_dustar + dqmh7_dz0h + dqmh7_dL
        self.adqmh7_dqsurf = self.adqmh7 + self.adqmh7_dqsurf
        self.adqmh7_dwq = self.adqmh7 + self.adqmh7_dwq
        self.adqmh7_dustar = self.adqmh7 + self.adqmh7_dustar
        self.adqmh7_dz0h += self.adqmh7
        self.adqmh7_dL = self.adqmh7 + self.adqmh7_dL
        self.adqmh7 = 0
        #statement dqmh7_dL = - wq / ustar / model.k * (- dpsih_qmh7_L + dpsih_z0h_L)
        self.adpsih_qmh7_L = - wq / ustar / model.k * (-1) * self.adqmh7_dL + self.adpsih_qmh7_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh7_dL + self.adpsih_z0h_L
        self.adqmh7_dL = 0
        #statement dpsih_qmh7_L = self.dpsih(model.qmeasuring_height7 / L,dzeta_dL_qmh7)
        self.adzeta_dL_qmh7 += self.dpsih(model.qmeasuring_height7 / L,self.adpsih_qmh7_L)
        self.adpsih_qmh7_L = 0
        #statement dqmh7_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height7 / z0h) * model.qmeasuring_height7 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height7 / z0h) * model.qmeasuring_height7 * -1 * z0h**-2 * self.adqmh7_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh7_dz0h
        self.adqmh7_dz0h = 0
        #statement dqmh7_dustar = - wq / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh7_dustar + self.adustar
        self.adqmh7_dustar = 0
        #statement dqmh7_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height7 / z0h) - model.psih(model.qmeasuring_height7 / L) + model.psih(z0h / L)) * self.adqmh7_dwq + self.adwq
        self.adqmh7_dwq = 0
        #statement dqmh7_dqsurf = dqsurf
        self.adqsurf = self.adqmh7_dqsurf + self.adqsurf
        self.adqmh7_dqsurf = 0  
        #statement dqmh6 = dqmh6_dqsurf + dqmh6_dwq + dqmh6_dustar + dqmh6_dz0h + dqmh6_dL
        self.adqmh6_dqsurf = self.adqmh6 + self.adqmh6_dqsurf
        self.adqmh6_dwq = self.adqmh6 + self.adqmh6_dwq
        self.adqmh6_dustar = self.adqmh6 + self.adqmh6_dustar
        self.adqmh6_dz0h += self.adqmh6
        self.adqmh6_dL = self.adqmh6 + self.adqmh6_dL
        self.adqmh6 = 0
        #statement dqmh6_dL = - wq / ustar / model.k * (- dpsih_qmh6_L + dpsih_z0h_L)
        self.adpsih_qmh6_L = - wq / ustar / model.k * (-1) * self.adqmh6_dL + self.adpsih_qmh6_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh6_dL + self.adpsih_z0h_L
        self.adqmh6_dL = 0
        #statement dpsih_qmh6_L = self.dpsih(model.qmeasuring_height6 / L,dzeta_dL_qmh6)
        self.adzeta_dL_qmh6 += self.dpsih(model.qmeasuring_height6 / L,self.adpsih_qmh6_L)
        self.adpsih_qmh6_L = 0
        #statement dqmh6_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height2 / z0h) * model.qmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height6 / z0h) * model.qmeasuring_height6 * -1 * z0h**-2 * self.adqmh6_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh6_dz0h
        self.adqmh6_dz0h = 0
        #statement dqmh6_dustar = - wq / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh6_dustar + self.adustar
        self.adqmh6_dustar = 0
        #statement dqmh6_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height6 / z0h) - model.psih(model.qmeasuring_height6 / L) + model.psih(z0h / L)) * self.adqmh6_dwq + self.adwq
        self.adqmh6_dwq = 0
        #statement dqmh6_dqsurf = dqsurf
        self.adqsurf = self.adqmh6_dqsurf + self.adqsurf
        self.adqmh6_dqsurf = 0  
        #statement dqmh5 = dqmh5_dqsurf + dqmh5_dwq + dqmh5_dustar + dqmh5_dz0h + dqmh5_dL
        self.adqmh5_dqsurf = self.adqmh5 + self.adqmh5_dqsurf
        self.adqmh5_dwq = self.adqmh5 + self.adqmh5_dwq
        self.adqmh5_dustar = self.adqmh5 + self.adqmh5_dustar
        self.adqmh5_dz0h += self.adqmh5
        self.adqmh5_dL = self.adqmh5 + self.adqmh5_dL
        self.adqmh5 = 0
        #statement dqmh5_dL = - wq / ustar / model.k * (- dpsih_qmh5_L + dpsih_z0h_L)
        self.adpsih_qmh5_L = - wq / ustar / model.k * (-1) * self.adqmh5_dL + self.adpsih_qmh5_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh5_dL + self.adpsih_z0h_L
        self.adqmh5_dL = 0
        #statement dpsih_qmh5_L = self.dpsih(model.qmeasuring_height5 / L,dzeta_dL_qmh5)
        self.adzeta_dL_qmh5 += self.dpsih(model.qmeasuring_height5 / L,self.adpsih_qmh5_L)
        self.adpsih_qmh5_L = 0
        #statement dqmh5_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height5 / z0h) * model.qmeasuring_height5 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height5 / z0h) * model.qmeasuring_height5 * -1 * z0h**-2 * self.adqmh5_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh5_dz0h
        self.adqmh5_dz0h = 0
        #statement dqmh5_dustar = - wq / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh5_dustar + self.adustar
        self.adqmh5_dustar = 0
        #statement dqmh5_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height5 / z0h) - model.psih(model.qmeasuring_height5 / L) + model.psih(z0h / L)) * self.adqmh5_dwq + self.adwq
        self.adqmh5_dwq = 0
        #statement dqmh5_dqsurf = dqsurf
        self.adqsurf = self.adqmh5_dqsurf + self.adqsurf
        self.adqmh5_dqsurf = 0  
        #statement dqmh4 = dqmh4_dqsurf + dqmh4_dwq + dqmh4_dustar + dqmh4_dz0h + dqmh4_dL
        self.adqmh4_dqsurf = self.adqmh4 + self.adqmh4_dqsurf
        self.adqmh4_dwq = self.adqmh4 + self.adqmh4_dwq
        self.adqmh4_dustar = self.adqmh4 + self.adqmh4_dustar
        self.adqmh4_dz0h += self.adqmh4
        self.adqmh4_dL = self.adqmh4 + self.adqmh4_dL
        self.adqmh4 = 0
        #statement dqmh4_dL = - wq / ustar / model.k * (- dpsih_qmh4_L + dpsih_z0h_L)
        self.adpsih_qmh4_L = - wq / ustar / model.k * (-1) * self.adqmh4_dL + self.adpsih_qmh4_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh4_dL + self.adpsih_z0h_L
        self.adqmh4_dL = 0
        #statement dpsih_qmh4_L = self.dpsih(model.qmeasuring_height4 / L,dzeta_dL_qmh4)
        self.adzeta_dL_qmh4 += self.dpsih(model.qmeasuring_height4 / L,self.adpsih_qmh4_L)
        self.adpsih_qmh4_L = 0
        #statement dqmh4_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height4 / z0h) * model.qmeasuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height4 / z0h) * model.qmeasuring_height4 * -1 * z0h**-2 * self.adqmh4_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh4_dz0h
        self.adqmh4_dz0h = 0
        #statement dqmh4_dustar = - wq / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh4_dustar + self.adustar
        self.adqmh4_dustar = 0
        #statement dqmh4_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height4 / z0h) - model.psih(model.qmeasuring_height4 / L) + model.psih(z0h / L)) * self.adqmh4_dwq + self.adwq
        self.adqmh4_dwq = 0
        #statement dqmh4_dqsurf = dqsurf
        self.adqsurf = self.adqmh4_dqsurf + self.adqsurf
        self.adqmh4_dqsurf = 0  
        #statement dqmh3 = dqmh3_dqsurf + dqmh3_dwq + dqmh3_dustar + dqmh3_dz0h + dqmh3_dL
        self.adqmh3_dqsurf = self.adqmh3 + self.adqmh3_dqsurf
        self.adqmh3_dwq = self.adqmh3 + self.adqmh3_dwq
        self.adqmh3_dustar = self.adqmh3 + self.adqmh3_dustar
        self.adqmh3_dz0h += self.adqmh3
        self.adqmh3_dL = self.adqmh3 + self.adqmh3_dL
        self.adqmh3 = 0
        #statement dqmh3_dL = - wq / ustar / model.k * (- dpsih_qmh3_L + dpsih_z0h_L)
        self.adpsih_qmh3_L = - wq / ustar / model.k * (-1) * self.adqmh3_dL + self.adpsih_qmh3_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh3_dL + self.adpsih_z0h_L
        self.adqmh3_dL = 0
        #statement dpsih_qmh3_L = self.dpsih(model.qmeasuring_height3 / L,dzeta_dL_qmh3)
        self.adzeta_dL_qmh3 += self.dpsih(model.qmeasuring_height3 / L,self.adpsih_qmh3_L)
        self.adpsih_qmh3_L = 0
        #statement dqmh3_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height3 / z0h) * model.qmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height3 / z0h) * model.qmeasuring_height3 * -1 * z0h**-2 * self.adqmh3_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh3_dz0h
        self.adqmh3_dz0h = 0
        #statement dqmh3_dustar = - wq / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh3_dustar + self.adustar
        self.adqmh3_dustar = 0
        #statement dqmh3_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height3 / z0h) - model.psih(model.qmeasuring_height3 / L) + model.psih(z0h / L)) * self.adqmh3_dwq + self.adwq
        self.adqmh3_dwq = 0
        #statement dqmh3_dqsurf = dqsurf
        self.adqsurf = self.adqmh3_dqsurf + self.adqsurf
        self.adqmh3_dqsurf = 0  
        #statement dqmh2 = dqmh2_dqsurf + dqmh2_dwq + dqmh2_dustar + dqmh2_dz0h + dqmh2_dL
        self.adqmh2_dqsurf = self.adqmh2 + self.adqmh2_dqsurf
        self.adqmh2_dwq = self.adqmh2 + self.adqmh2_dwq
        self.adqmh2_dustar = self.adqmh2 + self.adqmh2_dustar
        self.adqmh2_dz0h += self.adqmh2
        self.adqmh2_dL = self.adqmh2 + self.adqmh2_dL
        self.adqmh2 = 0
        #statement dqmh2_dL = - wq / ustar / model.k * (- dpsih_qmh2_L + dpsih_z0h_L)
        self.adpsih_qmh2_L = - wq / ustar / model.k * (-1) * self.adqmh2_dL + self.adpsih_qmh2_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh2_dL + self.adpsih_z0h_L
        self.adqmh2_dL = 0
        #statement dpsih_qmh2_L = self.dpsih(model.qmeasuring_height2 / L,dzeta_dL_qmh2)
        self.adzeta_dL_qmh2 += self.dpsih(model.qmeasuring_height2 / L,self.adpsih_qmh2_L)
        self.adpsih_qmh2_L = 0
        #statement dqmh2_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height2 / z0h) * model.qmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height2 / z0h) * model.qmeasuring_height2 * -1 * z0h**-2 * self.adqmh2_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh2_dz0h
        self.adqmh2_dz0h = 0
        #statement dqmh2_dustar = - wq / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh2_dustar + self.adustar
        self.adqmh2_dustar = 0
        #statement dqmh2_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height2 / z0h) - model.psih(model.qmeasuring_height2 / L) + model.psih(z0h / L)) * self.adqmh2_dwq + self.adwq
        self.adqmh2_dwq = 0
        #statement dqmh2_dqsurf = dqsurf
        self.adqsurf = self.adqmh2_dqsurf + self.adqsurf
        self.adqmh2_dqsurf = 0  
        #statement dqmh = dqmh_dqsurf + dqmh_dwq + dqmh_dustar + dqmh_dz0h + dqmh_dL
        self.adqmh_dqsurf = self.adqmh + self.adqmh_dqsurf
        self.adqmh_dwq = self.adqmh + self.adqmh_dwq
        self.adqmh_dustar = self.adqmh + self.adqmh_dustar
        self.adqmh_dz0h += self.adqmh
        self.adqmh_dL = self.adqmh + self.adqmh_dL
        self.adqmh = 0
        #statement dqmh_dL = - wq / ustar / model.k * (- dpsih_qmh_L + dpsih_z0h_L)
        self.adpsih_qmh_L = - wq / ustar / model.k * (-1) * self.adqmh_dL + self.adpsih_qmh_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adqmh_dL + self.adpsih_z0h_L
        self.adqmh_dL = 0
        #statement dpsih_qmh_L = self.dpsih(model.qmeasuring_height / L,dzeta_dL_qmh)
        self.adzeta_dL_qmh += self.dpsih(model.qmeasuring_height / L,self.adpsih_qmh_L)
        self.adpsih_qmh_L = 0
        #statement dqmh_dz0h = - wq / ustar / model.k * (1 / (model.qmeasuring_height / z0h) * model.qmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (model.qmeasuring_height / z0h) * model.qmeasuring_height * -1 * z0h**-2 * self.adqmh_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adqmh_dz0h
        self.adqmh_dz0h = 0
        #statement dqmh_dustar = - wq / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adqmh_dustar + self.adustar
        self.adqmh_dustar = 0
        #statement dqmh_dwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(model.qmeasuring_height / z0h) - model.psih(model.qmeasuring_height / L) + model.psih(z0h / L)) * self.adqmh_dwq + self.adwq
        self.adqmh_dwq = 0
        #statement dqmh_dqsurf = dqsurf
        self.adqsurf = self.adqmh_dqsurf + self.adqsurf
        self.adqmh_dqsurf = 0
        #statement dq2m = dq2m_dqsurf + dq2m_dwq + dq2m_dustar + dq2m_dz0h + dq2m_dL
        self.adq2m_dqsurf = self.adq2m + self.adq2m_dqsurf
        self.adq2m_dwq = self.adq2m + self.adq2m_dwq
        self.adq2m_dustar = self.adq2m + self.adq2m_dustar
        self.adq2m_dz0h += self.adq2m
        self.adq2m_dL = self.adq2m + self.adq2m_dL
        self.adq2m = 0
        #statement dq2m_dL = - wq / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        self.adpsih_2_L = - wq / ustar / model.k * (-1) * self.adq2m_dL + self.adpsih_2_L
        self.adpsih_z0h_L = - wq / ustar / model.k * self.adq2m_dL + self.adpsih_z0h_L
        self.adq2m_dL = 0
        #statement dq2m_dz0h = - wq / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wq / ustar / model.k * 1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.adq2m_dz0h
        self.adpsihterm_for_dCs_dz0h += - wq / ustar / model.k * self.adq2m_dz0h
        self.adq2m_dz0h = 0
        #statement dq2m_dustar = - wq / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wq / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adq2m_dustar + self.adustar
        self.adq2m_dustar = 0
        #statement dq2m_dwq = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwq
        self.adwq = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.adq2m_dwq + self.adwq
        self.adq2m_dwq = 0
        #statement dq2m_dqsurf = dqsurf
        self.adqsurf = self.adq2m_dqsurf + self.adqsurf
        self.adq2m_dqsurf = 0
        #statement dTmh7 = dthetamh7 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp)
        self.adthetamh7 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height7) / 100000)**(model.Rd/model.cp) * self.adTmh7
        self.adTmh7 = 0
        #statement dthetamh7 = dthetamh7_dthetasurf + dthetamh7_dwtheta + dthetamh7_dustar + dthetamh7_dz0h + dthetamh7_dL
        self.adthetamh7_dthetasurf = self.adthetamh7 + self.adthetamh7_dthetasurf
        self.adthetamh7_dwtheta = self.adthetamh7 + self.adthetamh7_dwtheta
        self.adthetamh7_dustar = self.adthetamh7 + self.adthetamh7_dustar
        self.adthetamh7_dz0h += self.adthetamh7
        self.adthetamh7_dL = self.adthetamh7 + self.adthetamh7_dL
        self.adthetamh7 = 0
        #statement dthetamh7_dL = - wtheta / ustar / model.k * (- dpsih_Tmh7_L + dpsih_z0h_L)
        self.adpsih_Tmh7_L = - wtheta / ustar / model.k * (-1) * self.adthetamh7_dL + self.adpsih_Tmh7_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh7_dL + self.adpsih_z0h_L
        self.adthetamh7_dL = 0
        #statement dpsih_Tmh7_L = self.dpsih(model.Tmeasuring_height7 / L,dzeta_dL_Tmh7)
        self.adzeta_dL_Tmh7 = self.dpsih(model.Tmeasuring_height7 / L,self.adpsih_Tmh7_L) + self.adzeta_dL_Tmh7
        self.adpsih_Tmh7_L = 0
        #statement dthetamh7_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height7 / z0h) * model.Tmeasuring_height7 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height7 / z0h) * model.Tmeasuring_height7 * -1 * z0h**-2 * self.adthetamh7_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh7_dz0h
        self.adthetamh7_dz0h = 0
        #statement dthetamh7_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh7_dustar + self.adustar
        self.adthetamh7_dustar = 0
        #statement dthetamh7_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height7 / z0h) - model.psih(model.Tmeasuring_height7 / L) + model.psih(z0h / L)) * self.adthetamh7_dwtheta + self.adwtheta
        self.adthetamh7_dwtheta = 0
        #statement dthetamh7_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh7_dthetasurf + self.adthetasurf
        self.adthetamh7_dthetasurf = 0
        #statement dTmh6 = dthetamh6 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp)
        self.adthetamh6 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height6) / 100000)**(model.Rd/model.cp) * self.adTmh6
        self.adTmh6 = 0
        #statement dthetamh6 = dthetamh6_dthetasurf + dthetamh6_dwtheta + dthetamh6_dustar + dthetamh6_dz0h + dthetamh6_dL
        self.adthetamh6_dthetasurf = self.adthetamh6 + self.adthetamh6_dthetasurf
        self.adthetamh6_dwtheta = self.adthetamh6 + self.adthetamh6_dwtheta
        self.adthetamh6_dustar = self.adthetamh6 + self.adthetamh6_dustar
        self.adthetamh6_dz0h += self.adthetamh6
        self.adthetamh6_dL = self.adthetamh6 + self.adthetamh6_dL
        self.adthetamh6 = 0
        #statement dthetamh6_dL = - wtheta / ustar / model.k * (- dpsih_Tmh6_L + dpsih_z0h_L)
        self.adpsih_Tmh6_L = - wtheta / ustar / model.k * (-1) * self.adthetamh6_dL + self.adpsih_Tmh6_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh6_dL + self.adpsih_z0h_L
        self.adthetamh6_dL = 0
        #statement dpsih_Tmh6_L = self.dpsih(model.Tmeasuring_height6 / L,dzeta_dL_Tmh6)
        self.adzeta_dL_Tmh6 = self.dpsih(model.Tmeasuring_height6 / L,self.adpsih_Tmh6_L) + self.adzeta_dL_Tmh6
        self.adpsih_Tmh6_L = 0
        #statement dthetamh6_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height6 / z0h) * model.Tmeasuring_height6 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height6 / z0h) * model.Tmeasuring_height6 * -1 * z0h**-2 * self.adthetamh6_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh6_dz0h
        self.adthetamh6_dz0h = 0
        #statement dthetamh6_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh6_dustar + self.adustar
        self.adthetamh6_dustar = 0
        #statement dthetamh6_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height6 / z0h) - model.psih(model.Tmeasuring_height6 / L) + model.psih(z0h / L)) * self.adthetamh6_dwtheta + self.adwtheta
        self.adthetamh6_dwtheta = 0
        #statement dthetamh6_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh6_dthetasurf + self.adthetasurf
        self.adthetamh6_dthetasurf = 0
        #statement dTmh5 = dthetamh5 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp)
        self.adthetamh5 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height5) / 100000)**(model.Rd/model.cp) * self.adTmh5
        self.adTmh5 = 0
        #statement dthetamh5 = dthetamh5_dthetasurf + dthetamh5_dwtheta + dthetamh5_dustar + dthetamh5_dz0h + dthetamh5_dL
        self.adthetamh5_dthetasurf = self.adthetamh5 + self.adthetamh5_dthetasurf
        self.adthetamh5_dwtheta = self.adthetamh5 + self.adthetamh5_dwtheta
        self.adthetamh5_dustar = self.adthetamh5 + self.adthetamh5_dustar
        self.adthetamh5_dz0h += self.adthetamh5
        self.adthetamh5_dL = self.adthetamh5 + self.adthetamh5_dL
        self.adthetamh5 = 0
        #statement dthetamh5_dL = - wtheta / ustar / model.k * (- dpsih_Tmh5_L + dpsih_z0h_L)
        self.adpsih_Tmh5_L = - wtheta / ustar / model.k * (-1) * self.adthetamh5_dL + self.adpsih_Tmh5_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh5_dL + self.adpsih_z0h_L
        self.adthetamh5_dL = 0
        #statement dpsih_Tmh5_L = self.dpsih(model.Tmeasuring_height5 / L,dzeta_dL_Tmh5)
        self.adzeta_dL_Tmh5 = self.dpsih(model.Tmeasuring_height5 / L,self.adpsih_Tmh5_L) + self.adzeta_dL_Tmh5
        self.adpsih_Tmh5_L = 0
        #statement dthetamh5_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height5 / z0h) * model.Tmeasuring_height5 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height5 / z0h) * model.Tmeasuring_height5 * -1 * z0h**-2 * self.adthetamh5_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh5_dz0h
        self.adthetamh5_dz0h = 0
        #statement dthetamh5_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh5_dustar + self.adustar
        self.adthetamh5_dustar = 0
        #statement dthetamh5_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height5 / z0h) - model.psih(model.Tmeasuring_height5 / L) + model.psih(z0h / L)) * self.adthetamh5_dwtheta + self.adwtheta
        self.adthetamh5_dwtheta = 0
        #statement dthetamh5_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh5_dthetasurf + self.adthetasurf
        self.adthetamh5_dthetasurf = 0
        #statement dTmh4 = dthetamh4 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp)
        self.adthetamh4 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height4) / 100000)**(model.Rd/model.cp) * self.adTmh4
        self.adTmh4 = 0
        #statement dthetamh4 = dthetamh4_dthetasurf + dthetamh4_dwtheta + dthetamh4_dustar + dthetamh4_dz0h + dthetamh4_dL
        self.adthetamh4_dthetasurf = self.adthetamh4 + self.adthetamh4_dthetasurf
        self.adthetamh4_dwtheta = self.adthetamh4 + self.adthetamh4_dwtheta
        self.adthetamh4_dustar = self.adthetamh4 + self.adthetamh4_dustar
        self.adthetamh4_dz0h += self.adthetamh4
        self.adthetamh4_dL = self.adthetamh4 + self.adthetamh4_dL
        self.adthetamh4 = 0
        #statement dthetamh4_dL = - wtheta / ustar / model.k * (- dpsih_Tmh4_L + dpsih_z0h_L)
        self.adpsih_Tmh4_L = - wtheta / ustar / model.k * (-1) * self.adthetamh4_dL + self.adpsih_Tmh4_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh4_dL + self.adpsih_z0h_L
        self.adthetamh4_dL = 0
        #statement dpsih_Tmh4_L = self.dpsih(model.Tmeasuring_height4 / L,dzeta_dL_Tmh4)
        self.adzeta_dL_Tmh4 = self.dpsih(model.Tmeasuring_height4 / L,self.adpsih_Tmh4_L) + self.adzeta_dL_Tmh4
        self.adpsih_Tmh4_L = 0
        #statement dthetamh4_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height4 / z0h) * model.Tmeasuring_height4 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height4 / z0h) * model.Tmeasuring_height4 * -1 * z0h**-2 * self.adthetamh4_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh4_dz0h
        self.adthetamh4_dz0h = 0
        #statement dthetamh4_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh4_dustar + self.adustar
        self.adthetamh4_dustar = 0
        #statement dthetamh4_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height4 / z0h) - model.psih(model.Tmeasuring_height4 / L) + model.psih(z0h / L)) * self.adthetamh4_dwtheta + self.adwtheta
        self.adthetamh4_dwtheta = 0
        #statement dthetamh4_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh4_dthetasurf + self.adthetasurf
        self.adthetamh4_dthetasurf = 0
        #statement dTmh3 = dthetamh3 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp)
        self.adthetamh3 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height3) / 100000)**(model.Rd/model.cp) * self.adTmh3
        self.adTmh3 = 0
        #statement dthetamh3 = dthetamh3_dthetasurf + dthetamh3_dwtheta + dthetamh3_dustar + dthetamh3_dz0h + dthetamh3_dL
        self.adthetamh3_dthetasurf = self.adthetamh3 + self.adthetamh3_dthetasurf
        self.adthetamh3_dwtheta = self.adthetamh3 + self.adthetamh3_dwtheta
        self.adthetamh3_dustar = self.adthetamh3 + self.adthetamh3_dustar
        self.adthetamh3_dz0h += self.adthetamh3
        self.adthetamh3_dL = self.adthetamh3 + self.adthetamh3_dL
        self.adthetamh3 = 0
        #statement dthetamh3_dL = - wtheta / ustar / model.k * (- dpsih_Tmh3_L + dpsih_z0h_L)
        self.adpsih_Tmh3_L = - wtheta / ustar / model.k * (-1) * self.adthetamh3_dL + self.adpsih_Tmh3_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh3_dL + self.adpsih_z0h_L
        self.adthetamh3_dL = 0
        #statement dpsih_Tmh3_L = self.dpsih(model.Tmeasuring_height3 / L,dzeta_dL_Tmh3)
        self.adzeta_dL_Tmh3 = self.dpsih(model.Tmeasuring_height3 / L,self.adpsih_Tmh3_L) + self.adzeta_dL_Tmh3
        self.adpsih_Tmh3_L = 0
        #statement dthetamh3_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height3 / z0h) * model.Tmeasuring_height3 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height3 / z0h) * model.Tmeasuring_height3 * -1 * z0h**-2 * self.adthetamh3_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh3_dz0h
        self.adthetamh3_dz0h = 0
        #statement dthetamh3_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh3_dustar + self.adustar
        self.adthetamh3_dustar = 0
        #statement dthetamh3_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height3 / z0h) - model.psih(model.Tmeasuring_height3 / L) + model.psih(z0h / L)) * self.adthetamh3_dwtheta + self.adwtheta
        self.adthetamh3_dwtheta = 0
        #statement dthetamh3_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh3_dthetasurf + self.adthetasurf
        self.adthetamh3_dthetasurf = 0
        #statement dTmh2 = dthetamh2 * ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp)
        self.adthetamh2 += ((model.Ps - model.rho * model.g * model.Tmeasuring_height2) / 100000)**(model.Rd/model.cp) * self.adTmh2
        self.adTmh2 = 0
        #statement dthetamh2 = dthetamh2_dthetasurf + dthetamh2_dwtheta + dthetamh2_dustar + dthetamh2_dz0h + dthetamh2_dL
        self.adthetamh2_dthetasurf = self.adthetamh2 + self.adthetamh2_dthetasurf
        self.adthetamh2_dwtheta = self.adthetamh2 + self.adthetamh2_dwtheta
        self.adthetamh2_dustar = self.adthetamh2 + self.adthetamh2_dustar
        self.adthetamh2_dz0h += self.adthetamh2
        self.adthetamh2_dL = self.adthetamh2 + self.adthetamh2_dL
        self.adthetamh2 = 0
        #statement dthetamh2_dL = - wtheta / ustar / model.k * (- dpsih_Tmh2_L + dpsih_z0h_L)
        self.adpsih_Tmh2_L = - wtheta / ustar / model.k * (-1) * self.adthetamh2_dL + self.adpsih_Tmh2_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh2_dL + self.adpsih_z0h_L
        self.adthetamh2_dL = 0
        #statement dpsih_Tmh2_L = self.dpsih(model.Tmeasuring_height2 / L,dzeta_dL_Tmh2)
        self.adzeta_dL_Tmh2 = self.dpsih(model.Tmeasuring_height2 / L,self.adpsih_Tmh2_L) + self.adzeta_dL_Tmh2
        self.adpsih_Tmh2_L = 0
        #statement dthetamh2_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height2 / z0h) * model.Tmeasuring_height2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height2 / z0h) * model.Tmeasuring_height2 * -1 * z0h**-2 * self.adthetamh2_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh2_dz0h
        self.adthetamh2_dz0h = 0
        #statement dthetamh2_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh2_dustar + self.adustar
        self.adthetamh2_dustar = 0
        #statement dthetamh2_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height2 / z0h) - model.psih(model.Tmeasuring_height2 / L) + model.psih(z0h / L)) * self.adthetamh2_dwtheta + self.adwtheta
        self.adthetamh2_dwtheta = 0
        #statement dthetamh2_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh2_dthetasurf + self.adthetasurf
        self.adthetamh2_dthetasurf = 0        
        #statement dTmh = dthetamh * ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp)
        self.adthetamh += ((model.Ps - model.rho * model.g * model.Tmeasuring_height) / 100000)**(model.Rd/model.cp) * self.adTmh
        self.adTmh = 0
        #statement dthetamh = dthetamh_dthetasurf + dthetamh_dwtheta + dthetamh_dustar + dthetamh_dz0h + dthetamh_dL
        self.adthetamh_dthetasurf = self.adthetamh + self.adthetamh_dthetasurf
        self.adthetamh_dwtheta = self.adthetamh + self.adthetamh_dwtheta
        self.adthetamh_dustar = self.adthetamh + self.adthetamh_dustar
        self.adthetamh_dz0h += self.adthetamh
        self.adthetamh_dL = self.adthetamh + self.adthetamh_dL
        self.adthetamh = 0
        #statement dthetamh_dL = - wtheta / ustar / model.k * (- dpsih_Tmh_L + dpsih_z0h_L)
        self.adpsih_Tmh_L = - wtheta / ustar / model.k * (-1) * self.adthetamh_dL + self.adpsih_Tmh_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adthetamh_dL + self.adpsih_z0h_L
        self.adthetamh_dL = 0
        #statement dpsih_Tmh_L = self.dpsih(model.Tmeasuring_height / L,dzeta_dL_Tmh)
        self.adzeta_dL_Tmh = self.dpsih(model.Tmeasuring_height / L,self.adpsih_Tmh_L) + self.adzeta_dL_Tmh
        self.adpsih_Tmh_L = 0
        #statement dthetamh_dz0h = - wtheta / ustar / model.k * (1 / (model.Tmeasuring_height / z0h) * model.Tmeasuring_height * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (model.Tmeasuring_height / z0h) * model.Tmeasuring_height * -1 * z0h**-2 * self.adthetamh_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adthetamh_dz0h
        self.adthetamh_dz0h = 0
        #statement dthetamh_dustar = - wtheta / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adthetamh_dustar + self.adustar
        self.adthetamh_dustar = 0
        #statement dthetamh_dwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(model.Tmeasuring_height / z0h) - model.psih(model.Tmeasuring_height / L) + model.psih(z0h / L)) * self.adthetamh_dwtheta + self.adwtheta
        self.adthetamh_dwtheta = 0
        #statement dthetamh_dthetasurf = dthetasurf
        self.adthetasurf = self.adthetamh_dthetasurf + self.adthetasurf
        self.adthetamh_dthetasurf = 0
        #statement dT2m = dT2m_dthetasurf + dT2m_dwtheta + dT2m_dustar + dT2m_dz0h + dT2m_dL
        self.adT2m_dthetasurf = self.adT2m + self.adT2m_dthetasurf
        self.adT2m_dwtheta = self.adT2m + self.adT2m_dwtheta
        self.adT2m_dustar = self.adT2m + self.adT2m_dustar
        self.adT2m_dz0h += self.adT2m
        self.adT2m_dL = self.adT2m + self.adT2m_dL
        self.adT2m = 0
        #statement dT2m_dL = - wtheta / ustar / model.k * (- dpsih_2_L + dpsih_z0h_L)
        self.adpsih_2_L = - wtheta / ustar / model.k * (-1) * self.adT2m_dL + self.adpsih_2_L
        self.adpsih_z0h_L = - wtheta / ustar / model.k * self.adT2m_dL + self.adpsih_z0h_L
        self.adT2m_dL = 0
        #statement dpsih_z0h_L = self.dpsih(z0h / L,dzeta_dL_z0h)
        self.adzeta_dL_z0h = self.dpsih(z0h / L,self.adpsih_z0h_L) + self.adzeta_dL_z0h
        self.adpsih_z0h_L = 0
        #statement dpsih_2_L = self.dpsih(2. / L,dzeta_dL_2)
        self.adzeta_dL_2 = self.dpsih(2. / L,self.adpsih_2_L) + self.adzeta_dL_2
        self.adpsih_2_L = 0
        #statement dT2m_dz0h = - wtheta / ustar / model.k * (1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        self.adz0h += - wtheta / ustar / model.k * 1 / (2. / z0h) * 2 * -1 * z0h**-2 * self.adT2m_dz0h
        self.adpsihterm_for_dCs_dz0h += - wtheta / ustar / model.k * self.adT2m_dz0h
        self.adT2m_dz0h = 0
        #statement dT2m_dustar = - wtheta / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * dustar
        self.adustar = - wtheta / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * (-1) * ustar**(-2) * self.adT2m_dustar + self.adustar
        self.adT2m_dustar = 0
        #statement dT2m_dwtheta = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.dwtheta
        self.adwtheta = - 1 / ustar / model.k * (np.log(2. / z0h) - model.psih(2. / L) + model.psih(z0h / L)) * self.adT2m_dwtheta + self.adwtheta
        self.adT2m_dwtheta = 0
        #statement dT2m_dthetasurf = dthetasurf
        self.adthetasurf = self.adT2m_dthetasurf + self.adthetasurf
        self.adT2m_dthetasurf = 0
        #statement dvw    = - (dCm * ueff * v + Cm * v * dueff + Cm * ueff * self.dv)
        self.adCm = - ueff * v * self.advw + self.adCm
        self.adueff += - Cm * v * self.advw
        self.adv += - Cm * ueff * self.advw
        self.advw = 0
        #statement duw    = - (dCm * ueff * u + Cm * u * dueff + Cm * ueff * self.du)
        self.adCm = - ueff * u * self.aduw + self.adCm
        self.adueff += - Cm * u * self.aduw
        self.adu += - Cm * ueff * self.aduw
        self.aduw = 0
        #statements if model.updatevals_surf_lay:
        if model.updatevals_surf_lay:
            #statement self.dCs_start = dCs
            self.adCs = self.adCs_start + self.adCs
            self.adCs_start = 0
            #statement self.dustar_start = dustar
            self.adustar = self.adustar_start + self.adustar
            self.adustar_start = 0
        #statement dustar = 0.5*(Cm)**(-0.5) * ueff * dCm + Cm**(0.5) * dueff
        self.adCm = 0.5*(Cm)**(-0.5) * ueff * self.adustar + self.adCm
        self.adueff += Cm**(0.5) * self.adustar
        self.adustar = 0
        #statement dCs = dCs_dzsl + dCs_dz0m + dCs_dL + dCs_dz0h
        self.adCs_dzsl = self.adCs + self.adCs_dzsl
        self.adCs_dz0m += self.adCs
        self.adCs_dL = self.adCs + self.adCs_dL
        self.adCs_dz0h += self.adCs
        self.adCs = 0
        #statement dCs_dz0h = constant3_for_Cs * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dCs_dz0h)
        constant3_for_Cs = model.k**2. / (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) * -1  * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) ** (-2)
        self.adz0h += constant3_for_Cs * 1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.adCs_dz0h
        self.adpsihterm_for_dCs_dz0h += constant3_for_Cs * self.adCs_dz0h
        self.adCs_dz0h = 0
        #statement dpsihterm_for_dCs_dz0h = self.dpsih(z0h / L,1 / L * self.dz0h)
        self.adz0h += self.dpsih(z0h / L,1 / L * self.adpsihterm_for_dCs_dz0h)
        self.adpsihterm_for_dCs_dz0h = 0
        #statement dCs_dL = constant_for_Cs * ((np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))*dpsimterm_for_dCs_dL + 
        #              (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) *dpsihterm_for_dCs_dL)
        constant_for_Cs = model.k**2. *(-1) * ((np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))) ** (-2)
        self.adpsimterm_for_dCs_dL = constant_for_Cs * ((np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)))*self.adCs_dL + self.adpsimterm_for_dCs_dL
        self.adpsihterm_for_dCs_dL = constant_for_Cs * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) * self.adCs_dL + self.adpsihterm_for_dCs_dL
        self.adCs_dL = 0
        #statement dpsihterm_for_dCs_dL = (- self.dpsih(zsl / L,dzeta_dL_zsl)+ self.dpsih(z0h / L,dzeta_dL_z0h))
        self.adzeta_dL_zsl = - self.dpsih(zsl / L,self.adpsihterm_for_dCs_dL) + self.adzeta_dL_zsl
        self.adzeta_dL_z0h = self.dpsih(z0h / L,self.adpsihterm_for_dCs_dL) + self.adzeta_dL_z0h
        self.adpsihterm_for_dCs_dL = 0
        #statement dpsimterm_for_dCs_dL = (- self.dpsim(zsl / L,dzeta_dL_zsl)+self.dpsim(z0m / L,dzeta_dL_z0m))
        self.adzeta_dL_zsl = - self.dpsim(zsl / L,self.adpsimterm_for_dCs_dL) + self.adzeta_dL_zsl
        self.adzeta_dL_z0m = self.dpsim(z0m / L,self.adpsimterm_for_dCs_dL) + self.adzeta_dL_z0m
        self.adpsimterm_for_dCs_dL = 0
        #statement dCs_dz0m = constant2_for_Cs * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        constant2_for_Cs = model.k**2. *(-1) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L))**-2 / (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L))
        self.adz0m += constant2_for_Cs * (1 / (zsl / z0m) * zsl * -1 * z0m**-2) * self.adCs_dz0m
        self.adpsimterm_for_dCm_dz0m += constant2_for_Cs * self.adCs_dz0m
        self.adCs_dz0m = 0
        #statement dCs_dzsl = constant_for_Cs * ((1/(zsl / z0m) * 1/z0m * dzsl - dpsim_term_for_dCs_dzsl) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) + (1/(zsl / z0h) * 1/z0h * dzsl - dpsih_term_for_dCs_dzsl ) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) )
        self.adzsl = constant_for_Cs * (1/(zsl / z0m) * 1/z0m * self.adCs_dzsl) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) + self.adzsl
        self.adzsl += constant_for_Cs * (1/(zsl / z0h) * 1/z0h * self.adCs_dzsl) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L))
        self.adpsim_term_for_dCs_dzsl = constant_for_Cs * (- self.adCs_dzsl) * (np.log(zsl / z0h) - model.psih(zsl / L) + model.psih(z0h / L)) + self.adpsim_term_for_dCs_dzsl
        self.adpsih_term_for_dCs_dzsl = constant_for_Cs * (- self.adCs_dzsl) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) + self.adpsih_term_for_dCs_dzsl
        self.adCs_dzsl = 0
        #statement dpsih_term_for_dCs_dzsl = self.dpsih(zsl / L, 1/L*dzsl)
        self.adzsl = self.dpsih(zsl / L, 1/L*self.adpsih_term_for_dCs_dzsl) + self.adzsl
        self.adpsih_term_for_dCs_dzsl = 0
        #statement dpsim_term_for_dCs_dzsl = self.dpsim(zsl / L, 1/L*dzsl)
        self.adzsl = self.dpsim(zsl / L, 1/L*self.adpsim_term_for_dCs_dzsl) + self.adzsl
        self.adpsim_term_for_dCs_dzsl = 0
        #statement dCm = dCm_dzsl + dCm_dz0m + dCm_dL
        self.adCm_dzsl = self.adCm + self.adCm_dzsl
        self.adCm_dz0m += self.adCm
        self.adCm_dL = self.adCm + self.adCm_dL
        self.adCm = 0
        #statement dCm_dL = constant_for_Cm * (-1* dpsimterm_for_Cm_zsl + dpsimterm_for_Cm_z0m)
        constant_for_Cm = model.k**2. *(-2) * (np.log(zsl / z0m) - model.psim(zsl / L) + model.psim(z0m / L)) ** (-3)
        self.adpsimterm_for_Cm_zsl = constant_for_Cm * (-1* self.adCm_dL) + self.adpsimterm_for_Cm_zsl
        self.adpsimterm_for_Cm_z0m = constant_for_Cm * (self.adCm_dL) + self.adpsimterm_for_Cm_z0m
        self.adCm_dL = 0
        #statement dCm_dz0m = constant_for_Cm * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dCm_dz0m)
        self.adz0m += constant_for_Cm * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.adCm_dz0m)
        self.adpsimterm_for_dCm_dz0m += constant_for_Cm * self.adCm_dz0m
        self.adCm_dz0m = 0
        #statement dCm_dzsl = constant_for_Cm * (1/(zsl / z0m) * 1/z0m * dzsl - dpsim_term_for_dCm_dzsl)
        self.adzsl = constant_for_Cm * (1/(zsl / z0m) * 1/z0m * self.adCm_dzsl) + self.adzsl
        self.adpsim_term_for_dCm_dzsl = constant_for_Cm * (-1) * self.adCm_dzsl + self.adpsim_term_for_dCm_dzsl
        self.adCm_dzsl = 0
        #statement dpsimterm_for_dCm_dz0m = self.dpsim(z0m / L,1 / L * self.dz0m)
        self.adz0m += self.dpsim(z0m / L,1 / L * self.adpsimterm_for_dCm_dz0m)
        self.adpsimterm_for_dCm_dz0m = 0
        #statement dpsimterm_for_Cm_z0m = self.dpsim(z0m / L,dzeta_dL_z0m)
        self.adzeta_dL_z0m = self.dpsim(z0m / L,self.adpsimterm_for_Cm_z0m) + self.adzeta_dL_z0m
        self.adpsimterm_for_Cm_z0m  = 0
        #statement dpsimterm_for_Cm_zsl = self.dpsim(zsl / L,dzeta_dL_zsl)
        self.adzeta_dL_zsl = self.dpsim(zsl / L,self.adpsimterm_for_Cm_zsl) + self.adzeta_dL_zsl
        self.adpsimterm_for_Cm_zsl = 0
        #if self.adjointtesting:
#        self.HTy = np.array([adT2m_dtheta,adT2m_dwtheta])
        #statement dzeta_dL_CO2mh4 = self.dzeta_dL(model.CO2measuring_height4,L) * dL
        self.adL = self.dzeta_dL(model.CO2measuring_height4,L) * self.adzeta_dL_CO2mh4 + self.adL
        self.adzeta_dL_CO2mh4 = 0
        #statement dzeta_dL_CO2mh3 = self.dzeta_dL(model.CO2measuring_height3,L) * dL
        self.adL = self.dzeta_dL(model.CO2measuring_height3,L) * self.adzeta_dL_CO2mh3 + self.adL
        self.adzeta_dL_CO2mh3 = 0
        #statement dzeta_dL_CO2mh2 = self.dzeta_dL(model.CO2measuring_height2,L) * dL
        self.adL = self.dzeta_dL(model.CO2measuring_height2,L) * self.adzeta_dL_CO2mh2 + self.adL
        self.adzeta_dL_CO2mh2 = 0
        #statement dzeta_dL_CO2mh = self.dzeta_dL(model.CO2measuring_height,L) * dL
        self.adL = self.dzeta_dL(model.CO2measuring_height,L) * self.adzeta_dL_CO2mh + self.adL
        self.adzeta_dL_CO2mh = 0
        #statement dzeta_dL_COSmh3 = self.dzeta_dL(model.COSmeasuring_height3,L) * dL
        self.adL = self.dzeta_dL(model.COSmeasuring_height3,L) * self.adzeta_dL_COSmh3 + self.adL
        self.adzeta_dL_COSmh3 = 0
        #statement dzeta_dL_COSmh2 = self.dzeta_dL(model.COSmeasuring_height2,L) * dL
        self.adL = self.dzeta_dL(model.COSmeasuring_height2,L) * self.adzeta_dL_COSmh2 + self.adL
        self.adzeta_dL_COSmh2 = 0
        #statement dzeta_dL_COSmh = self.dzeta_dL(COSmeasuring_height,L) * dL
        self.adL = self.dzeta_dL(COSmeasuring_height,L) * self.adzeta_dL_COSmh + self.adL
        self.adzeta_dL_COSmh = 0
        #statement dzeta_dL_qmh7 = self.dzeta_dL(model.qmeasuring_height7,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height7,L) * self.adzeta_dL_qmh7 + self.adL
        self.adzeta_dL_qmh7 = 0
        #statement dzeta_dL_qmh6 = self.dzeta_dL(model.qmeasuring_height6,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height6,L) * self.adzeta_dL_qmh6 + self.adL
        self.adzeta_dL_qmh6 = 0
        #statement dzeta_dL_qmh5 = self.dzeta_dL(model.qmeasuring_height5,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height5,L) * self.adzeta_dL_qmh5 + self.adL
        self.adzeta_dL_qmh5 = 0
        #statement dzeta_dL_qmh4 = self.dzeta_dL(model.qmeasuring_height4,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height4,L) * self.adzeta_dL_qmh4 + self.adL
        self.adzeta_dL_qmh4 = 0
        #statement dzeta_dL_qmh3 = self.dzeta_dL(model.qmeasuring_height3,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height3,L) * self.adzeta_dL_qmh3 + self.adL
        self.adzeta_dL_qmh3 = 0
        #statement dzeta_dL_qmh2 = self.dzeta_dL(model.qmeasuring_height2,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height2,L) * self.adzeta_dL_qmh2 + self.adL
        self.adzeta_dL_qmh2 = 0
        #statement dzeta_dL_qmh = self.dzeta_dL(model.qmeasuring_height,L) * dL
        self.adL = self.dzeta_dL(model.qmeasuring_height,L) * self.adzeta_dL_qmh + self.adL
        self.adzeta_dL_qmh = 0
        #statement dzeta_dL_Tmh7 = self.dzeta_dL(model.Tmeasuring_height7,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height7,L) * self.adzeta_dL_Tmh7 + self.adL
        self.adzeta_dL_Tmh7 = 0
        #statement dzeta_dL_Tmh6 = self.dzeta_dL(model.Tmeasuring_height6,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height6,L) * self.adzeta_dL_Tmh6 + self.adL
        self.adzeta_dL_Tmh6 = 0
        #statement dzeta_dL_Tmh5 = self.dzeta_dL(model.Tmeasuring_height5,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height5,L) * self.adzeta_dL_Tmh5 + self.adL
        self.adzeta_dL_Tmh5 = 0
        #statement dzeta_dL_Tmh4 = self.dzeta_dL(model.Tmeasuring_height4,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height4,L) * self.adzeta_dL_Tmh4 + self.adL
        self.adzeta_dL_Tmh4 = 0
        #statement dzeta_dL_Tmh3 = self.dzeta_dL(model.Tmeasuring_height3,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height3,L) * self.adzeta_dL_Tmh3 + self.adL
        self.adzeta_dL_Tmh3 = 0
        #statement dzeta_dL_Tmh2 = self.dzeta_dL(model.Tmeasuring_height2,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height2,L) * self.adzeta_dL_Tmh2 + self.adL
        self.adzeta_dL_Tmh2 = 0
        #statement dzeta_dL_Tmh = self.dzeta_dL(model.Tmeasuring_height,L) * dL
        self.adL = self.dzeta_dL(model.Tmeasuring_height,L) * self.adzeta_dL_Tmh + self.adL
        self.adzeta_dL_Tmh = 0
        #statement dzeta_dL_2 = self.dzeta_dL(2.,L) * dL
        self.adL = self.dzeta_dL(2.,L) * self.adzeta_dL_2 + self.adL
        self.adzeta_dL_2 = 0
        #statement dzeta_dL_z0h = self.dzeta_dL(z0h,L) * dL
        self.adL = self.dzeta_dL(z0h,L) * self.adzeta_dL_z0h + self.adL
        self.adzeta_dL_z0h = 0
        #statement dzeta_dL_zsl = self.dzeta_dL(zsl,L) * dL
        self.adL = self.dzeta_dL(zsl,L) * self.adzeta_dL_zsl + self.adL
        self.adzeta_dL_zsl = 0
        #statement dzeta_dL_z0m = self.dzeta_dL(z0m,L) * dL
        self.adL = self.dzeta_dL(z0m,L) *  self.adzeta_dL_z0m + self.adL
        self.adzeta_dL_z0m = 0
        #statement dpsim_term_for_dCm_dzsl = self.dpsim(zsl / L, 1/L*dzsl)
        self.adzsl = self.dpsim(zsl / L, 1/L*self.adpsim_term_for_dCm_dzsl) + self.adzsl
        self.adpsim_term_for_dCm_dzsl = 0
        if model.sw_use_ribtol:
            #statement dL = self.tl_ribtol(model,checkpoint,returnvariable='dL')
            self.adj_ribtol(forcing,checkpoint,model)
            self.adL = 0
            #statement dRib = dRib_dthetav + dRib_dzsl + dRib_dthetavsurf + dRib_dueff
            self.adRib_dthetav += self.adRib
            self.adRib_dzsl += self.adRib
            self.adRib_dthetavsurf += self.adRib
            self.adRib_dueff += self.adRib
            self.adRib = 0
            if Rib > 0.2:
                #statement dRib_dueff = 0.
                self.adRib_dueff = 0
                #statement dRib_dthetavsurf = 0.
                self.adRib_dthetavsurf = 0
                #statement dRib_dzsl = 0.
                self.adRib_dzsl = 0
                #statement dRib_dthetav = 0.
                self.adRib_dthetav = 0
            #statement dRib_dueff = model.g / thetav * zsl * (thetav - thetavsurf) * -2 * ueff**-3 * dueff
            self.adueff += model.g / thetav * zsl * (thetav - thetavsurf) * -2 * ueff**-3 * self.adRib_dueff
            self.adRib_dueff = 0
            #statement dRib_dthetavsurf = model.g / thetav * zsl / ueff**2. * -1 * dthetavsurf
            self.adthetavsurf += model.g / thetav * zsl / ueff**2. * -1 * self.adRib_dthetavsurf
            self.adRib_dthetavsurf = 0
            #statement dRib_dzsl = model.g / thetav * (thetav - thetavsurf) / ueff**2. * dzsl
            self.adzsl += model.g / thetav * (thetav - thetavsurf) / ueff**2. * self.adRib_dzsl
            self.adRib_dzsl = 0
            #statement dRib_dthetav = model.g * -1 * thetav**-2 * self.dthetav * zsl * (thetav - thetavsurf) / ueff**2. + model.g / thetav * zsl / ueff**2. * self.dthetav
            self.adthetav += model.g * -1 * thetav**-2 * zsl * (thetav - thetavsurf) / ueff**2. * self.adRib_dthetav
            self.adthetav += model.g / thetav * zsl / ueff**2. * self.adRib_dthetav
            self.adRib_dthetav = 0
        else:
            #statement dL = dL_dthetav + dL_dustar_start + dL_dwthetav
            self.adL_dthetav = self.adL + self.adL_dthetav
            self.adL_dustar_start = self.adL + self.adL_dustar_start
            self.adL_dwthetav = self.adL + self.adL_dwthetav
            self.adL = 0
            #statement dL_dwthetav = thetav * ustar_start**3 /(model.k * model.g * -1) * (-1) * wthetav**(-2) * self.dwthetav
            self.adwthetav = thetav * ustar_start**3 /(model.k * model.g * -1) * (-1) * wthetav**(-2) * self.adL_dwthetav + self.adwthetav
            self.adL_dwthetav = 0
            #statement dL_dustar_start = thetav * 3*ustar_start**2 /(model.k * model.g * -1 * wthetav) * self.dustar_start
            self.adustar_start = thetav * 3*ustar_start**2 /(model.k * model.g * -1 * wthetav) * self.adL_dustar_start + self.adustar_start
            self.adL_dustar_start = 0
            #statement dL_dthetav = ustar_start**3 /(model.k * model.g * -1 * wthetav) * self.dthetav
            self.adthetav = ustar_start**3 /(model.k * model.g * -1 * wthetav) * self.adL_dthetav + self.adthetav
            self.adL_dthetav = 0
        #statement dzsl      = 0.1 * self.dh
        self.adh = 0.1 * self.adzsl + self.adh
        self.adzsl = 0
        #statement dthetavsurf = dthetasurf * (1. + 0.61 * qsurf) + 0.61 * thetasurf * dqsurf
        self.adthetasurf = (1. + 0.61 * qsurf) * self.adthetavsurf + self.adthetasurf
        self.adqsurf = 0.61 * thetasurf * self.adthetavsurf + self.adqsurf
        self.adthetavsurf = 0
        #statement desurf = dqsurf * model.Ps / 0.622
        self.adqsurf += model.Ps / 0.622 * self.adesurf
        self.adesurf = 0
        #new method for qsurf:
        #statement dqsurf = dqsurf_dq + dqsurf_dwq + dqsurf_dCs_start + dqsurf_dueff 
        self.adqsurf_dq += self.adqsurf
        self.adqsurf_dwq += self.adqsurf
        self.adqsurf_dCs_start += self.adqsurf
        self.adqsurf_dueff += self.adqsurf
        self.adqsurf = 0
        #statement dqsurf_dueff = wq / Cs_start * (-1) * ueff**(-2) * dueff
        self.adueff += wq / Cs_start * (-1) * ueff**(-2) * self.adqsurf_dueff
        self.adqsurf_dueff = 0
        #statement dqsurf_dCs_start = wq / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        self.adCs_start += wq / ueff * (-1) * Cs_start**(-2) * self.adqsurf_dCs_start
        self.adqsurf_dCs_start = 0
        #statement dqsurf_dwq = self.dwq / (Cs_start * ueff)
        self.adwq += self.adqsurf_dwq / (Cs_start * ueff)
        self.adqsurf_dwq = 0
        #statement dqsurf_dq = self.dq
        self.adq += self.adqsurf_dq
        self.adqsurf_dq = 0
        #old, wrong method of qsurf:
#        #statement dqsurf = dqsurf_dq + dqsurf_dcq + dqsurf_dqsatsurf_rsl
#        self.adqsurf_dq = self.adqsurf + self.adqsurf_dq
#        self.adqsurf_dcq = self.adqsurf + self.adqsurf_dcq
#        self.adqsurf_dqsatsurf_rsl = self.adqsurf + self.adqsurf_dqsatsurf_rsl
#        self.adqsurf = 0
#        #statement dqsurf_dqsatsurf_rsl = cq * dqsatsurf_rsl
#        self.adqsatsurf_rsl = cq * self.adqsurf_dqsatsurf_rsl + self.adqsatsurf_rsl
#        self.adqsurf_dqsatsurf_rsl = 0
#        #statement dqsurf_dcq = (-q + qsatsurf_rsl) * dcq
#        self.adcq = (-q + qsatsurf_rsl) * self.adqsurf_dcq + self.adcq
#        self.adqsurf_dcq = 0
#        #statement dqsurf_dq    = (1. - cq) * self.dq
#        self.adq = (1. - cq) * self.adqsurf_dq + self.adq
#        self.adqsurf_dq = 0
#        #statement dcq = dcq_dCs_start + dcq_dueff + dcq_drs
#        self.adcq_dCs_start = self.adcq + self.adcq_dCs_start
#        self.adcq_dueff = self.adcq + self.adcq_dueff
#        self.adcq_drs = self.adcq + self.adcq_drs
#        self.adcq = 0
#        #statement dcq_drs = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * Cs_start * self.drs
#        self.adrs = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * Cs_start * self.adcq_drs + self.adrs
#        self.adcq_drs = 0
#        #statement dcq_dueff = -1 * (1. + Cs_start * ueff * rs) ** -2. * Cs_start * rs * dueff
#        self.adueff += -1 * (1. + Cs_start * ueff * rs) ** -2. * Cs_start * rs * self.adcq_dueff
#        self.adcq_dueff = 0
#        #statement dcq_dCs_start = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * rs * self.dCs_start
#        self.adCs_start = -1 * (1. + Cs_start * ueff * rs) ** -2. * ueff * rs * self.adcq_dCs_start + self.adCs_start
#        self.adcq_dCs_start = 0
#        #statement dqsatsurf_rsl = dqsat_dT(thetasurf, model.Ps, dthetasurf)
#        self.adthetasurf = dqsat_dT(thetasurf, model.Ps, self.adqsatsurf_rsl) + self.adthetasurf
#        self.adqsatsurf_rsl = 0
        #statement dTsurf = dthetasurf * (100000/model.Ps)**(-model.Rd/model.cp)
        if self.manualadjointtesting:
            self.adTsurf = self.y
        self.adthetasurf += self.adTsurf * (100000/model.Ps)**(-model.Rd/model.cp)
        self.adTsurf = 0
        #statement dthetasurf = dthetasurf_dtheta + dthetasurf_dwtheta + dthetasurf_dCs_start + dthetasurf_dueff
        self.adthetasurf_dtheta = self.adthetasurf + self.adthetasurf_dtheta
        self.adthetasurf_dwtheta = self.adthetasurf + self.adthetasurf_dwtheta #!important to add self.adthetasurf_dwtheta
        self.adthetasurf_dCs_start = self.adthetasurf + self.adthetasurf_dCs_start
        self.adthetasurf_dueff = self.adthetasurf + self.adthetasurf_dueff
        self.adthetasurf = 0
        if self.manualadjointtesting:
            self.HTy = self.adthetasurf_dtheta
        #statement dthetasurf_dueff = wtheta / Cs_start * (-1) * ueff**(-2) * dueff
        self.adueff += wtheta / Cs_start * (-1) * ueff**(-2) * self.adthetasurf_dueff
        self.adthetasurf_dueff = 0
        #statement dthetasurf_dCs_start = wtheta / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        self.adCs_start = wtheta / ueff * (-1) * Cs_start**(-2) * self.adthetasurf_dCs_start + self.adCs_start
        self.adthetasurf_dCs_start = 0
        #statement dthetasurf_dwtheta = self.dwtheta/(Cs_start * ueff)
        self.adwtheta =  1/(Cs_start * ueff) * self.adthetasurf_dwtheta + self.adwtheta
        self.adthetasurf_dwtheta = 0
        #statement dthetasurf_dtheta = self.dtheta
        self.adtheta = self.adthetasurf_dtheta + self.adtheta
        self.adthetasurf_dtheta = 0
        #statement dCO2surf = dCO2surf_dwCO2 + dCO2surf_dCO2 + dCO2surf_dCs_start + dCO2surf_dueff 
        self.adCO2surf_dwCO2 = self.adCO2surf + self.adCO2surf_dwCO2
        self.adCO2surf_dCO2 = self.adCO2surf + self.adCO2surf_dCO2
        self.adCO2surf_dCs_start = self.adCO2surf + self.adCO2surf_dCs_start
        self.adCO2surf_dueff = self.adCO2surf + self.adCO2surf_dueff
        self.adCO2surf = 0
        #statement dCO2surf_dueff = wCO2 / Cs_start * (-1) * ueff**(-2) * dueff
        self.adueff += wCO2 / Cs_start * (-1) * ueff**(-2) * self.adCO2surf_dueff
        self.adCO2surf_dueff = 0
        #statement dCO2surf_dCs_start = wCO2 / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        self.adCs_start = wCO2 / ueff * (-1) * Cs_start**(-2) * self.adCO2surf_dCs_start + self.adCs_start
        self.adCO2surf_dCs_start = 0
        #statement dCO2surf_dCO2 = self.dCO2
        self.adCO2 = self.adCO2surf_dCO2 + self.adCO2
        self.adCO2surf_dCO2 = 0
        #statement dCO2surf_dwCO2 = dwCO2 / (Cs_start * ueff)
        self.adwCO2 = 1 / (Cs_start * ueff) * self.adCO2surf_dwCO2 + self.adwCO2
        self.adCO2surf_dwCO2 = 0
        #statement dCOSsurf = dCOSsurf_dwCOS + dCOSsurf_dCOS + dCOSsurf_dCs_start + dCOSsurf_dueff 
        self.adCOSsurf_dwCOS = self.adCOSsurf + self.adCOSsurf_dwCOS
        self.adCOSsurf_dCOS = self.adCOSsurf + self.adCOSsurf_dCOS
        self.adCOSsurf_dCs_start = self.adCOSsurf + self.adCOSsurf_dCs_start
        self.adCOSsurf_dueff = self.adCOSsurf + self.adCOSsurf_dueff
        self.adCOSsurf = 0
        #statement dCOSsurf_dueff = wCOS / Cs_start * (-1) * ueff**(-2) * dueff
        self.adueff += wCOS / Cs_start * (-1) * ueff**(-2) * self.adCOSsurf_dueff
        self.adCOSsurf_dueff = 0
        #statement dCOSsurf_dCs_start = wCOS / ueff * (-1) * Cs_start**(-2) * self.dCs_start
        self.adCs_start = wCOS / ueff * (-1) * Cs_start**(-2) * self.adCOSsurf_dCs_start + self.adCs_start
        self.adCOSsurf_dCs_start = 0
        #statement dCOSsurf_dCOS = self.dCOS
        self.adCOS = self.adCOSsurf_dCOS + self.adCOS
        self.adCOSsurf_dCOS = 0
        #statement dCOSsurf_dwCOS = dwCOS / (Cs_start * ueff)
        self.adwCOS = 1 / (Cs_start * ueff) * self.adCOSsurf_dwCOS + self.adwCOS
        self.adCOSsurf_dwCOS = 0
        if np.sqrt(u**2. + v**2. + wstar**2.) < 0.01:
            #statement dueff = 0
            self.adueff = 0
        else:
            #statement dueff = 0.5 * (u**2. + v**2. + wstar**2.)**(-0.5) * (2 * u * self.du + 2 * v * self.dv + 2 * wstar * self.dwstar)
            self.adu += 0.5 * (u**2. + v**2. + wstar**2.) **(-0.5) * 2 * u * self.adueff
            self.adv += 0.5 * (u**2. + v**2. + wstar**2.) **(-0.5) * 2 * v * self.adueff
            self.adwstar += 0.5 * (u**2. + v**2. + wstar**2.) **(-0.5) * 2 * wstar * self.adueff
            self.adueff = 0
        if self.adjointtestingrun_surf_lay:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
                    
    def adj_ribtol(self,forcing,checkpoint,model,HTy_variables=None):
        it = checkpoint['rtl_it_end']
        zsl = checkpoint['rtl_zsl']
        z0m = checkpoint['rtl_z0m']
        z0h = checkpoint['rtl_z0h']
        L = checkpoint['rtl_L_middle']
        Lstart = checkpoint['rtl_Lstart_end']
        Lend = checkpoint['rtl_Lend_end']
        fxdif_part1 = checkpoint['rtl_fxdif_part1_end']
        fxdif_part2 = checkpoint['rtl_fxdif_part2_end']
        fx = checkpoint['rtl_fx_end']
        fxdif = checkpoint['rtl_fxdif_end']
        for i in range(it-1,-1,-1):
            #statement dL = dL_new
            self.adL_new += self.adL
            self.adL = 0
            #statement dL_new       = dL - (dfx / fxdif[i] + fx[i] * -1 * fxdif[i]**-2 * dfxdif)
            self.adL += self.adL_new
            self.adfx += - 1 / fxdif[i] * self.adL_new
            self.adfxdif += - 1 * fx[i] * -1 * fxdif[i]**-2 * self.adL_new
            self.adL_new = 0
            #statement dfxdif = (dfxdif_part1 + dfxdif_part2) / (Lstart[i] - Lend[i]) + (fxdif_part1[i] + fxdif_part2[i]) * -1 * (Lstart[i] - Lend[i])**-2 * (dLstart - dLend)
            self.adfxdif_part1 += 1 / (Lstart[i] - Lend[i]) * self.adfxdif 
            self.adfxdif_part2 += 1 / (Lstart[i] - Lend[i]) * self.adfxdif 
            self.adLstart += (fxdif_part1[i] + fxdif_part2[i]) * -1 * (Lstart[i] - Lend[i])**-2 * self.adfxdif
            self.adLend += (fxdif_part1[i] + fxdif_part2[i]) * -1 * (Lstart[i] - Lend[i])**-2 * -1 * self.adfxdif
            self.adfxdif = 0
            #statement dfxdif_part2 = dfxdif_part2_dzsl + dfxdif_part2_dLend + dfxdif_part2_dz0h + dfxdif_part2_dz0m
            self.adfxdif_part2_dzsl += self.adfxdif_part2
            self.adfxdif_part2_dLend += self.adfxdif_part2
            self.adfxdif_part2_dz0h += self.adfxdif_part2
            self.adfxdif_part2_dz0m += self.adfxdif_part2
            self.adfxdif_part2 = 0
            #statement dfxdif_part2_dz0m = -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfxdif_part2_dz0m)
            self.adz0m += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2) * self.adfxdif_part2_dz0m
            self.adpsimterm_for_dfxdif_part2_dz0m += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * self.adfxdif_part2_dz0m
            self.adfxdif_part2_dz0m = 0
            #statement dfxdif_part2_dz0h = -1 * - zsl / Lend[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfxdif_part2_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adz0h += -1 * - zsl / Lend[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.adfxdif_part2_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adpsihterm_for_dfxdif_part2_dz0h += -1 * - zsl / Lend[i] * (self.adfxdif_part2_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adfxdif_part2_dz0h = 0
            #statement dfxdif_part2_dLend = -1 * (- zsl * -1 * Lend[i]**-2 * dLend * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                                  # - zsl / Lend[i] * (-dpsihterm1_for_dfxdif_part2_dLend + dpsihterm2_for_dfxdif_part2_dLend) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. +\
                                  # - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (-dpsimterm1_for_dfxdif_part2_dLend + dpsimterm2_for_dfxdif_part2_dLend))
            self.adLend += -1 * - zsl * -1 * Lend[i]**-2 * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. * self.adfxdif_part2_dLend
            self.adpsihterm1_for_dfxdif_part2_dLend += -1 * - zsl / Lend[i] * (-1 * self.adfxdif_part2_dLend) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adpsihterm2_for_dfxdif_part2_dLend += -1 * - zsl / Lend[i] * (self.adfxdif_part2_dLend) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adpsimterm1_for_dfxdif_part2_dLend += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (-1 * self.adfxdif_part2_dLend)
            self.adpsimterm2_for_dfxdif_part2_dLend += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (self.adfxdif_part2_dLend)
            self.adfxdif_part2_dLend = 0
            #statement dfxdif_part2_dzsl = -1 * (- self.dzsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                               # - zsl / Lend[i] * (1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfxdif_part2_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. + \
                               # - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfxdif_part2_dzsl))
            self.adzsl += -1 * -1 * self.adfxdif_part2_dzsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2. 
            self.adzsl += -1 * - zsl / Lend[i] * (1 / (zsl / z0h) * 1 / z0h * self.adfxdif_part2_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adpsihterm_for_dfxdif_part2_dzsl += -1 * - zsl / Lend[i] * -1 * self.adfxdif_part2_dzsl / (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**2.
            self.adzsl += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.adfxdif_part2_dzsl)
            self.adpsimterm_for_dfxdif_part2_dzsl += -1 * - zsl / Lend[i] * (np.log(zsl / z0h) - model.psih(zsl / Lend[i]) + model.psih(z0h / Lend[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lend[i]) + model.psim(z0m / Lend[i]))**-3 * (-1 * self.adfxdif_part2_dzsl)
            self.adfxdif_part2_dzsl = 0
            #statement dpsimterm_for_dfxdif_part2_dz0m = self.dpsim(z0m / Lend[i], 1 / Lend[i] * self.dz0m)
            self.adz0m += self.dpsim(z0m / Lend[i], 1 / Lend[i] * self.adpsimterm_for_dfxdif_part2_dz0m)
            self.adpsimterm_for_dfxdif_part2_dz0m = 0
            #statement dpsihterm_for_dfxdif_part2_dz0h = self.dpsih(z0h / Lend[i], 1 / Lend[i] * self.dz0h)
            self.adz0h += self.dpsih(z0h / Lend[i], 1 / Lend[i] * self.adpsihterm_for_dfxdif_part2_dz0h)
            self.adpsihterm_for_dfxdif_part2_dz0h = 0
            #statement dpsimterm2_for_dfxdif_part2_dLend = self.dpsim(z0m / Lend[i], dzeta_dLend_z0m)
            self.adzeta_dLend_z0m += self.dpsim(z0m / Lend[i], self.adpsimterm2_for_dfxdif_part2_dLend)
            self.adpsimterm2_for_dfxdif_part2_dLend = 0
            #statement dpsimterm1_for_dfxdif_part2_dLend = self.dpsim(zsl / Lend[i], dzeta_dLend_zsl)
            self.adzeta_dLend_zsl += self.dpsim(zsl / Lend[i], self.adpsimterm1_for_dfxdif_part2_dLend)
            self.adpsimterm1_for_dfxdif_part2_dLend = 0
            #statement dpsihterm2_for_dfxdif_part2_dLend = self.dpsih(z0h / Lend[i], dzeta_dLend_z0h)
            self.adzeta_dLend_z0h += self.dpsih(z0h / Lend[i], self.adpsihterm2_for_dfxdif_part2_dLend)
            self.adpsihterm2_for_dfxdif_part2_dLend = 0
            #statement dpsihterm1_for_dfxdif_part2_dLend = self.dpsih(zsl / Lend[i], dzeta_dLend_zsl)
            self.adzeta_dLend_zsl += self.dpsih(zsl / Lend[i], self.adpsihterm1_for_dfxdif_part2_dLend)
            self.adpsihterm1_for_dfxdif_part2_dLend = 0
            #statement dpsimterm_for_dfxdif_part2_dzsl = self.dpsim(zsl / Lend[i], 1 / Lend[i] * self.dzsl)
            self.adzsl += self.dpsim(zsl / Lend[i], 1 / Lend[i] * self.adpsimterm_for_dfxdif_part2_dzsl)
            self.adpsimterm_for_dfxdif_part2_dzsl = 0
            #statement dpsihterm_for_dfxdif_part2_dzsl = self.dpsih(zsl / Lend[i], 1 / Lend[i] * self.dzsl)
            self.adzsl += self.dpsih(zsl / Lend[i], 1 / Lend[i] * self.adpsihterm_for_dfxdif_part2_dzsl)
            self.adpsihterm_for_dfxdif_part2_dzsl = 0
            #statement dzeta_dLend_z0m = self.dzeta_dL(z0m,Lend[i]) * dLend
            self.adLend += self.dzeta_dL(z0m,Lend[i]) * self.adzeta_dLend_z0m
            self.adzeta_dLend_z0m = 0
            #statement dzeta_dLend_z0h = self.dzeta_dL(z0h,Lend[i]) * dLend
            self.adLend += self.dzeta_dL(z0h,Lend[i]) * self.adzeta_dLend_z0h
            self.adzeta_dLend_z0h = 0
            #statement dzeta_dLend_zsl = self.dzeta_dL(zsl,Lend[i]) * dLend
            self.adLend += self.dzeta_dL(zsl,Lend[i]) * self.adzeta_dLend_zsl
            self.adzeta_dLend_zsl = 0
            #statement dfxdif_part1 = dfxdif_part1_dzsl + dfxdif_part1_dLstart + dfxdif_part1_dz0h + dfxdif_part1_dz0m
            self.adfxdif_part1_dzsl += self.adfxdif_part1
            self.adfxdif_part1_dLstart += self.adfxdif_part1
            self.adfxdif_part1_dz0h += self.adfxdif_part1
            self.adfxdif_part1_dz0m += self.adfxdif_part1
            self.adfxdif_part1 = 0
            #statement dfxdif_part1_dz0m = - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfxdif_part1_dz0m)
            self.adz0m += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.adfxdif_part1_dz0m)
            self.adpsimterm_for_dfxdif_part1_dz0m += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (self.adfxdif_part1_dz0m)
            self.adfxdif_part1_dz0m = 0
            #statement dfxdif_part1_dz0h = - zsl / Lstart[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfxdif_part1_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adz0h += - zsl / Lstart[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.adfxdif_part1_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adpsihterm_for_dfxdif_part1_dz0h += - zsl / Lstart[i] * (self.adfxdif_part1_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adfxdif_part1_dz0h = 0
            #statement dfxdif_part1_dLstart = - zsl * -1 * Lstart[i]**-2 * dLstart * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                                  # - zsl / Lstart[i] * (-dpsihterm1_for_dfxdif_part1_dLstart + dpsihterm2_for_dfxdif_part1_dLstart) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. +\
                                  # - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (-dpsimterm1_for_dfxdif_part1_dLstart + dpsimterm2_for_dfxdif_part1_dLstart)
            self.adLstart += - zsl * -1 * Lstart[i]**-2 * self.adfxdif_part1_dLstart * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adpsihterm1_for_dfxdif_part1_dLstart  += - zsl / Lstart[i] * (-1 * self.adfxdif_part1_dLstart) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adpsihterm2_for_dfxdif_part1_dLstart += - zsl / Lstart[i] * (self.adfxdif_part1_dLstart) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adpsimterm1_for_dfxdif_part1_dLstart += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (-1 * self.adfxdif_part1_dLstart)
            self.adpsimterm2_for_dfxdif_part1_dLstart += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (self.adfxdif_part1_dLstart)
            self.adfxdif_part1_dLstart = 0
            #statement dfxdif_part1_dzsl = - self.dzsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                               # - zsl / Lstart[i] * (1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfxdif_part1_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. + \
                               # - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfxdif_part1_dzsl)
            self.adzsl += - self.adfxdif_part1_dzsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2.
            self.adzsl += - zsl / Lstart[i] * (1 / (zsl / z0h) * 1 / z0h * self.adfxdif_part1_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. 
            self.adpsihterm_for_dfxdif_part1_dzsl += - zsl / Lstart[i] * (-1 * self.adfxdif_part1_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**2. 
            self.adzsl += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (1 / (zsl / z0m) * 1 / z0m * self.adfxdif_part1_dzsl)
            self.adpsimterm_for_dfxdif_part1_dzsl += - zsl / Lstart[i] * (np.log(zsl / z0h) - model.psih(zsl / Lstart[i]) + model.psih(z0h / Lstart[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / Lstart[i]) + model.psim(z0m / Lstart[i]))**-3 * (-1 * self.adfxdif_part1_dzsl)
            self.adfxdif_part1_dzsl = 0
            #statement dpsimterm_for_dfxdif_part1_dz0m = self.dpsim(z0m / Lstart[i], 1 / Lstart[i] * self.dz0m)
            self.adz0m += self.dpsim(z0m / Lstart[i], 1 / Lstart[i] * self.adpsimterm_for_dfxdif_part1_dz0m)
            self.adpsimterm_for_dfxdif_part1_dz0m = 0
            #statement dpsihterm_for_dfxdif_part1_dz0h = self.dpsih(z0h / Lstart[i], 1 / Lstart[i] * self.dz0h)
            self.adz0h += self.dpsih(z0h / Lstart[i], 1 / Lstart[i] * self.adpsihterm_for_dfxdif_part1_dz0h)
            self.adpsihterm_for_dfxdif_part1_dz0h = 0
            #statement dpsimterm2_for_dfxdif_part1_dLstart = self.dpsim(z0m / Lstart[i], dzeta_dLstart_z0m)
            self.adzeta_dLstart_z0m += self.dpsim(z0m / Lstart[i], self.adpsimterm2_for_dfxdif_part1_dLstart)
            self.adpsimterm2_for_dfxdif_part1_dLstart = 0
            #statement dpsimterm1_for_dfxdif_part1_dLstart = self.dpsim(zsl / Lstart[i], dzeta_dLstart_zsl)
            self.adzeta_dLstart_zsl += self.dpsim(zsl / Lstart[i], self.adpsimterm1_for_dfxdif_part1_dLstart)
            self.adpsimterm1_for_dfxdif_part1_dLstart = 0
            #statement dpsihterm2_for_dfxdif_part1_dLstart = self.dpsih(z0h / Lstart[i], dzeta_dLstart_z0h)
            self.adzeta_dLstart_z0h += self.dpsih(z0h / Lstart[i], self.adpsihterm2_for_dfxdif_part1_dLstart)
            self.adpsihterm2_for_dfxdif_part1_dLstart = 0
            #statement dpsihterm1_for_dfxdif_part1_dLstart = self.dpsih(zsl / Lstart[i], dzeta_dLstart_zsl)
            self.adzeta_dLstart_zsl += self.dpsih(zsl / Lstart[i], self.adpsihterm1_for_dfxdif_part1_dLstart)
            self.adpsihterm1_for_dfxdif_part1_dLstart = 0
            #statement dpsimterm_for_dfxdif_part1_dzsl = self.dpsim(zsl / Lstart[i], 1 / Lstart[i] * self.dzsl)
            self.adzsl += self.dpsim(zsl / Lstart[i], 1 / Lstart[i] * self.adpsimterm_for_dfxdif_part1_dzsl)
            self.adpsimterm_for_dfxdif_part1_dzsl = 0
            #statement dpsihterm_for_dfxdif_part1_dzsl = self.dpsih(zsl / Lstart[i], 1 / Lstart[i] * self.dzsl)
            self.adzsl += self.dpsih(zsl / Lstart[i], 1 / Lstart[i] * self.adpsihterm_for_dfxdif_part1_dzsl)
            self.adpsihterm_for_dfxdif_part1_dzsl = 0
            #statement dzeta_dLstart_z0m = self.dzeta_dL(z0m,Lstart[i]) * dLstart
            self.adLstart += self.dzeta_dL(z0m,Lstart[i]) * self.adzeta_dLstart_z0m
            self.adzeta_dLstart_z0m = 0
            #statement dzeta_dLstart_z0h = self.dzeta_dL(z0h,Lstart[i]) * dLstart
            self.adLstart += self.dzeta_dL(z0h,Lstart[i]) * self.adzeta_dLstart_z0h
            self.adzeta_dLstart_z0h = 0
            #statement dzeta_dLstart_zsl = self.dzeta_dL(zsl,Lstart[i]) * dLstart
            self.adLstart += self.dzeta_dL(zsl,Lstart[i]) * self.adzeta_dLstart_zsl
            self.adzeta_dLstart_zsl = 0
            #statement dLend    = dL + 0.001*dL
            self.adL += 1.001 * self.adLend
            self.adLend = 0
            #statement dLstart  = dL - 0.001*dL
            self.adL += 0.999 * self.adLstart
            self.adLstart = 0
            #statement dfx = dfx_dRib + dfx_dzsl + dfx_dL + dfx_dz0m + dfx_dz0h
            self.adfx_dRib += self.adfx
            self.adfx_dzsl += self.adfx
            self.adfx_dL += self.adfx
            self.adfx_dz0m += self.adfx
            self.adfx_dz0h += self.adfx
            self.adfx = 0
            #statement dfx_dz0h = - zsl / L[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.dz0h + dpsihterm_for_dfx_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2
            self.adz0h += - zsl / L[i] * (1 / (zsl / z0h) * zsl * -1 * z0h**-2 * self.adfx_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2
            self.adpsihterm_for_dfx_dz0h += - zsl / L[i] * (self.adfx_dz0h) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2
            self.adfx_dz0h = 0
            #statement dfx_dz0m = - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.dz0m + dpsimterm_for_dfx_dz0m)
            self.adz0m += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * zsl * -1 * z0m**-2 * self.adfx_dz0m)
            self.adpsimterm_for_dfx_dz0m += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (self.adfx_dz0m)
            self.adfx_dz0m = 0
            #statement dfx_dL   = - zsl * -1 * L[i]**-2 * dL * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. +\
                      # - zsl / L[i] * ((- dpsihterm1_for_dfx_dL + dpsihterm2_for_dfx_dL) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. +
                      #  (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (- dpsimterm1_for_dfx_dL + dpsimterm2_for_dfx_dL))
            self.adL += - zsl * -1 * L[i]**-2 * self.adfx_dL * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adpsihterm1_for_dfx_dL += - zsl / L[i] * (-1 * self.adfx_dL) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adpsihterm2_for_dfx_dL += - zsl / L[i] * (self.adfx_dL) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adpsimterm1_for_dfx_dL += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (- self.adfx_dL)
            self.adpsimterm2_for_dfx_dL += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (self.adfx_dL)
            self.adfx_dL = 0
            #statement dfx_dzsl = - self.dzsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. + \
                      # - zsl / L[i] * ((1 / (zsl / z0h) * 1 / z0h * self.dzsl - dpsihterm_for_dfx_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2. + \
                      #  (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * 1 / z0m * self.dzsl - dpsimterm_for_dfx_dzsl))
            self.adzsl += - self.adfx_dzsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adzsl += - zsl / L[i] * (1 / (zsl / z0h) * 1 / z0h * self.adfx_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adpsihterm_for_dfx_dzsl += - zsl / L[i] * (-1 * self.adfx_dzsl) / (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**2.
            self.adzsl += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (1 / (zsl / z0m) * 1 / z0m * self.adfx_dzsl)
            self.adpsimterm_for_dfx_dzsl += - zsl / L[i] * (np.log(zsl / z0h) - model.psih(zsl / L[i]) + model.psih(z0h / L[i])) * -2 * (np.log(zsl / z0m) - model.psim(zsl / L[i]) + model.psim(z0m / L[i]))**-3. * (-1 * self.adfx_dzsl)
            self.adfx_dzsl = 0
            #statement dpsihterm_for_dfx_dz0h = self.dpsih(z0h / L[i],1 / L[i] * self.dz0h)
            self.adz0h += self.dpsih(z0h / L[i],1 / L[i] * self.adpsihterm_for_dfx_dz0h)
            self.adpsihterm_for_dfx_dz0h = 0
            #statement dpsimterm_for_dfx_dz0m = self.dpsim(z0m/ L[i],1 / L[i] * self.dz0m)
            self.adz0m += self.dpsim(z0m/ L[i],1 / L[i] * self.adpsimterm_for_dfx_dz0m)
            self.adpsimterm_for_dfx_dz0m = 0
            #statement dpsimterm2_for_dfx_dL = self.dpsim(z0m/ L[i],dzeta_dL_z0m)
            self.adzeta_dL_z0m += self.dpsim(z0m/ L[i],self.adpsimterm2_for_dfx_dL)
            self.adpsimterm2_for_dfx_dL = 0
            # statement dpsimterm1_for_dfx_dL = self.dpsim(zsl / L[i],dzeta_dL_zsl)
            self.adzeta_dL_zsl += self.dpsim(zsl / L[i],self.adpsimterm1_for_dfx_dL)
            self.adpsimterm1_for_dfx_dL = 0
            #statement dpsihterm2_for_dfx_dL = self.dpsih(z0h / L[i],dzeta_dL_z0h)
            self.adzeta_dL_z0h += self.dpsih(z0h / L[i],self.adpsihterm2_for_dfx_dL)
            self.adpsihterm2_for_dfx_dL = 0
            #statement dpsihterm1_for_dfx_dL = self.dpsih(zsl / L[i],dzeta_dL_zsl)
            self.adzeta_dL_zsl += self.dpsih(zsl / L[i],self.adpsihterm1_for_dfx_dL)
            self.adpsihterm1_for_dfx_dL = 0
            #statement dpsimterm_for_dfx_dzsl = self.dpsim(zsl / L[i],1 / L[i] * self.dzsl)
            self.adzsl += self.dpsim(zsl / L[i],1 / L[i] * self.adpsimterm_for_dfx_dzsl)
            self.adpsimterm_for_dfx_dzsl = 0
            #statement dpsihterm_for_dfx_dzsl = self.dpsih(zsl / L[i],1 / L[i] * self.dzsl)
            self.adzsl += self.dpsih(zsl / L[i],1 / L[i] * self.adpsihterm_for_dfx_dzsl)
            self.adpsihterm_for_dfx_dzsl = 0
            #statement dzeta_dL_z0m = self.dzeta_dL(z0m,L[i]) * dL
            self.adL += self.dzeta_dL(z0m,L[i]) * self.adzeta_dL_z0m
            self.adzeta_dL_z0m = 0
            #statement dzeta_dL_z0h = self.dzeta_dL(z0h,L[i]) * dL
            self.adL += self.dzeta_dL(z0h,L[i]) * self.adzeta_dL_z0h
            self.adzeta_dL_z0h = 0
            #statement dzeta_dL_zsl = self.dzeta_dL(zsl,L[i]) * dL
            self.adL += self.dzeta_dL(zsl,L[i]) * self.adzeta_dL_zsl
            self.adzeta_dL_zsl = 0
            #statement dfx_dRib = self.dRib
            self.adRib += self.adfx_dRib
            self.adfx_dRib = 0
            #statement dL0      = dL
            self.adL += self.adL0
            self.adL0 = 0
        #statement dL0 = 0.
        self.adL0 = 0
        #statement dL = 0
        self.adL = 0
            
        if self.adjointtestingribtol:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable

    
    def adj_run_radiation(self,forcing,checkpoint,model,HTy_variables=None):
        doy = checkpoint['rr_doy']
        lat = checkpoint['rr_lat']
        h = checkpoint['rr_h']
        theta = checkpoint['rr_theta']
        sinlea_lon = checkpoint['rr_sinlea_lon_end']
        sda = checkpoint['rr_sda_end']
        sinlea = checkpoint['rr_sinlea_end']
        Tr = checkpoint['rr_Tr_end']
        alpha = checkpoint['rr_alpha']
        Ta = checkpoint['rr_Ta_end']
        cc = checkpoint['rr_cc']
        Ts = checkpoint['rr_Ts']
        t = checkpoint['rr_t']
        lon = checkpoint['rr_lon']
        #statement dQ = dSwin - dSwout + dLwin - dLwout
        self.adSwin += self.adQ
        self.adSwout += - self.adQ
        self.adLwin += self.adQ
        self.adLwout += - self.adQ
        self.adQ = 0
        #statement dLwout = model.bolz * 4 * Ts ** 3. * self.dTs
        self.adTs += model.bolz * 4 * Ts ** 3. * self.adLwout
        self.adLwout = 0
        #statement dLwin = 0.8 * model.bolz * 4 * Ta ** 3. * dTa
        self.adTa += 0.8 * model.bolz * 4 * Ta ** 3. * self.adLwin
        self.adLwin = 0
        #statement dSwout = model.S0 * (Tr * sinlea * self.dalpha + alpha * sinlea * dTr + alpha * Tr * dsinlea)
        self.adalpha += model.S0 * Tr * sinlea * self.adSwout
        self.adTr += model.S0 * alpha * sinlea * self.adSwout
        self.adsinlea += model.S0 * alpha * Tr * self.adSwout
        self.adSwout = 0
        #statement dSwin = model.S0 * (dTr * sinlea + Tr * dsinlea)
        self.adTr += model.S0 * sinlea * self.adSwin
        self.adsinlea += model.S0 * Tr * self.adSwin
        self.adSwin = 0
        #statement dTr  = (1. - 0.4 * cc) * 0.2 * dsinlea + (0.6 + 0.2 * sinlea) * - 0.4 * self.dcc
        self.adsinlea += (1. - 0.4 * cc) * 0.2 * self.adTr
        self.adcc += (0.6 + 0.2 * sinlea) * - 0.4 * self.adTr
        self.adTr = 0
        #statement dTa = dTa_dtheta + dTa_dh
        self.adTa_dtheta += self.adTa
        self.adTa_dh += self.adTa
        self.adTa = 0
        #statement dTa_dh = theta * (model.Rd / model.cp) * ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp - 1) * -1 * 0.1 * model.rho * model.g / model.Ps * self.dh
        self.adh += theta * (model.Rd / model.cp) * ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp - 1) * -1 * 0.1 * model.rho * model.g / model.Ps * self.adTa_dh
        self.adTa_dh = 0
        #statement dTa_dtheta = self.dtheta * ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp)
        self.adtheta += ((model.Ps - 0.1 * h * model.rho * model.g) / model.Ps ) ** (model.Rd / model.cp) * self.adTa_dtheta
        self.adTa_dtheta = 0
        sinlea = checkpoint['rr_sinlea_middle']
        if sinlea < 0.0001:
        #statement dsinlea = 0
            self.adsinlea = 0
        #statement dsinlea = dpart1_sinlea - dpart2_sinlea
        self.adpart1_sinlea += self.adsinlea
        self.adpart2_sinlea += - self.adsinlea
        self.adsinlea = 0
        #statement dpart2_sinlea = np.cos(sda) * sinlea_lon * -np.sin(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.dlat + np.cos(2. * np.pi * lat / 360.) * sinlea_lon * -np.sin(sda) * dsda + np.cos(2. * np.pi * lat / 360.) * np.cos(sda) * dsinlea_lon
        self.adlat += np.cos(sda) * sinlea_lon * -np.sin(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.adpart2_sinlea
        self.adsda += np.cos(2. * np.pi * lat / 360.) * sinlea_lon * -np.sin(sda) * self.adpart2_sinlea
        self.adsinlea_lon += np.cos(2. * np.pi * lat / 360.) * np.cos(sda) * self.adpart2_sinlea
        self.adpart2_sinlea = 0
        #statement dsinlea_lon = -np.sin(2. * np.pi * (t * model.dt + model.tstart * 3600.) / 86400. + 2. * np.pi * lon / 360.) * 2. * np.pi / 360. * self.dlon
        self.adlon += -np.sin(2. * np.pi * (t * model.dt + model.tstart * 3600.) / 86400. + 2. * np.pi * lon / 360.) * 2. * np.pi / 360. * self.adsinlea_lon
        self.adsinlea_lon = 0
        #statement dpart1_sinlea = np.sin(sda) * np.cos(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.dlat + np.sin(2. * np.pi * lat / 360.) * np.cos(sda)*dsda
        self.adlat += np.sin(sda) * np.cos(2. * np.pi * lat / 360.) * 2. * np.pi / 360. * self.adpart1_sinlea
        self.adsda += np.sin(2. * np.pi * lat / 360.) * np.cos(sda) * self.adpart1_sinlea
        self.adpart1_sinlea = 0
        #statement dsda = 0.409 * -np.sin(2. * np.pi * (doy - 173.) / 365.) * 2. * np.pi / 365 * self.ddoy
        self.addoy += 0.409 * -np.sin(2. * np.pi * (doy - 173.) / 365.) * 2. * np.pi / 365 * self.adsda
        self.adsda = 0
        
        if self.adjointtestingrun_radiation:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
        
    def adj_statistics(self,forcing,checkpoint,model,HTy_variables=None):
        q = checkpoint['stat_q']
        theta = checkpoint['stat_theta']
        wq = checkpoint['stat_wq']
        deltaq = checkpoint['stat_deltaq']
        deltatheta = checkpoint['stat_deltatheta']
        qsat_variable = checkpoint['stat_qsat_variable_end']
        T_h = checkpoint['stat_T_h_end']
        P_h = checkpoint['stat_P_h_end']
        t = checkpoint['stat_t'] #note this is time!!
        it = checkpoint['stat_it_end']
        p_lcl = checkpoint['stat_p_lcl_end']
        T_lcl = checkpoint['stat_T_lcl_end']
        for iteration in range(it-1,-1,-1):
            #statement dlcl = dlcl_new
            self.adlcl_new += self.adlcl
            self.adlcl = 0
            #statement dRHlcl        = self.dq / fwdm.qsat(T_lcl[iteration], p_lcl[iteration]) + q * -1 * fwdm.qsat(T_lcl[iteration], p_lcl[iteration])**-2 * (dqsat_variable_dp_lcl + dqsat_variable_dT_lcl)
            self.adq += 1 / fwdm.qsat(T_lcl[iteration], p_lcl[iteration]) * self.adRHlcl
            self.adqsat_variable_dp_lcl += q * -1 * fwdm.qsat(T_lcl[iteration], p_lcl[iteration])**-2 * self.adRHlcl
            self.adqsat_variable_dT_lcl += q * -1 * fwdm.qsat(T_lcl[iteration], p_lcl[iteration])**-2 * self.adRHlcl
            self.adRHlcl = 0
            #statement dqsat_variable_dT_lcl = dqsat_dT(T_lcl[iteration],p_lcl[iteration],dT_lcl)
            self.adT_lcl += dqsat_dT(T_lcl[iteration],p_lcl[iteration],self.adqsat_variable_dT_lcl)
            self.adqsat_variable_dT_lcl = 0
            #statement dqsat_variable_dp_lcl = dqsat_dp(T_lcl[iteration],p_lcl[iteration],dp_lcl)
            self.adp_lcl += dqsat_dp(T_lcl[iteration],p_lcl[iteration],self.adqsat_variable_dp_lcl)
            self.adqsat_variable_dp_lcl = 0
            #statement dT_lcl       = self.dtheta - model.g/model.cp * dlcl_new
            self.adtheta += self.adT_lcl
            self.adlcl_new += - model.g/model.cp * self.adT_lcl
            self.adT_lcl = 0
            #statement dp_lcl       = - model.rho * model.g * dlcl_new
            self.adlcl_new += - model.rho * model.g * self.adp_lcl
            self.adp_lcl = 0
            #statement dlcl_new         = self.dlcl + -1*self.dRHlcl*1000.
            self.adlcl += self.adlcl_new 
            self.adRHlcl += -1 * 1000. * self.adlcl_new
            self.adlcl_new = 0
        if(t == 0):
            #statement dRHlcl = 0.
            self.adRHlcl = 0
            #statement dlcl = self.dh
            self.adh += self.adlcl
            self.adlcl = 0
        else:
            #statement dRHlcl = 0
            self.adRHlcl = 0.
        #statement dRH_h   = 1 / qsat_variable * self.dq + q * (-1) * qsat_variable**(-2) * dqsat_variable
        self.adq +=  1 / qsat_variable * self.adRH_h
        self.adqsat_variable += q * (-1) * qsat_variable**(-2) * self.adRH_h
        self.adRH_h = 0
        #statement dqsat_variable = dqsat_variable_dT_H + dqsat_variable_dP_H
        self.adqsat_variable_dT_H += self.adqsat_variable
        self.adqsat_variable_dP_H += self.adqsat_variable
        self.adqsat_variable = 0
        #statement dqsat_variable_dP_H = dqsat_dp(T_h,P_h,dP_h)
        self.adP_h += dqsat_dp(T_h,P_h,self.adqsat_variable_dP_H)
        self.adqsat_variable_dP_H = 0
        #statement dqsat_variable_dT_H = dqsat_dT(T_h,P_h,dT_h)
        self.adT_h += dqsat_dT(T_h,P_h,self.adqsat_variable_dT_H)
        self.adqsat_variable_dT_H = 0
        #statement dT_h    = self.dtheta - model.g/model.cp * self.dh
        self.adtheta += self.adT_h
        self.adh += - model.g/model.cp * self.adT_h
        self.adT_h = 0
        #statement dP_h    = - model.rho * model.g * self.dh
        self.adh += - model.rho * model.g * self.adP_h
        self.adP_h = 0
        #statement ddeltathetav  = (self.dtheta + self.ddeltatheta) * (1. + 0.61 * (q + deltaq)) + (theta + deltatheta) * (0.61 * (self.dq + self.ddeltaq)) - (self.dtheta * (1. + 0.61 * q) + theta * 0.61 * self.dq)
        self.adtheta += (1. + 0.61 * (q + deltaq)) * self.addeltathetav
        self.addeltatheta += (1. + 0.61 * (q + deltaq)) * self.addeltathetav
        self.adq += (theta + deltatheta) * 0.61 * self.addeltathetav
        self.addeltaq += (theta + deltatheta) * 0.61 * self.addeltathetav
        self.adtheta += - (1. + 0.61 * q) * self.addeltathetav
        self.adq += -theta * 0.61 * self.addeltathetav
        self.addeltathetav = 0
        #statement dwthetav  = self.dwtheta + 0.61 * (self.dtheta * wq + theta * self.dwq)
        self.adwtheta += self.adwthetav
        self.adtheta += 0.61 * wq * self.adwthetav
        self.adwq += 0.61 * theta * self.adwthetav
        self.adwthetav = 0
        #statement dthetav   = self.dtheta  + 0.61 * (self.dtheta * q + theta * self.dq)
        self.adtheta += self.adthetav
        self.adtheta += 0.61 * q * self.adthetav
        self.adq += 0.61 * theta * self.adthetav
        self.adthetav = 0
        
        if self.adjointtestingstatistics:
            self.HTy = np.zeros(len(HTy_variables))
            for i in range(len(HTy_variables)):
                try: 
                    self.HTy[i] = self.__dict__[HTy_variables[i]]
                except KeyError:
                    self.HTy[i] = locals()[HTy_variables[i]] #in case it is not a self variable
    
    def grad_test(self,inputdata,perturbationvars,outputvar,dstate,returnvariable,output_dict,printmode='absolute'):
        if not hasattr(self,'failed_grad_test_list'):
            self.failed_grad_test_list = [] #list of vars where test fails
        #outputtime is the last timestep
        self.gradienttesting = True
        inputdata_copy = cp.deepcopy(inputdata) #just to not manipulate orig model data  
        default_model = fwdm.model(inputdata_copy) 
        default_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        cpx =  default_model.cpx
        cpx_init =  default_model.cpx_init
        output_dict_model = 'vars_'+output_dict
        if outputvar in default_model.__dict__[output_dict_model]:
            default_out = default_model.__dict__[output_dict_model][outputvar] #not all output of the model is a self variable
            if ((hasattr(default_model,outputvar) or hasattr(default_model.soilCOSmodel,outputvar)) or hasattr(default_model.out,outputvar)):
                print ('WARNING: outputvar exists both as self-variable or self.out var and in dictionary')
        else:
            try:
                if (output_dict_model == 'vars_sto'):
                    default_out = default_model.out.__dict__[outputvar][-1]
                else:
                    default_out = default_model.__dict__[outputvar]
            except KeyError:
                default_out = default_model.soilCOSmodel.__dict__[outputvar] #than we are dealing with a module outside the main forwardmodel (or a typo, which will raise a new KeyError)
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_full_model(default_model,cpx,cpx_init,returnvariable=returnvariable,tl_dict='Output_tl_'+output_dict)
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-7,1e-9,1e-12]
        numderiv = {}
        for alpha in alpharange:
            inputdata_copy = cp.deepcopy(inputdata)            
            for item in perturbationvars:
                inputdata_copy.__dict__[item] += alpha
            p_model = fwdm.model(inputdata_copy)
            p_model.run(checkpoint=False,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
            if outputvar in p_model.__dict__[output_dict_model]:
                p_out = p_model.__dict__[output_dict_model][outputvar]
            elif (output_dict_model == 'vars_sto' and hasattr(p_model.out,outputvar)): #store part is special when it comes to output
                p_out = p_model.out.__dict__[outputvar][-1]
            elif hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            elif hasattr(p_model.soilCOSmodel,outputvar):
                p_out = p_model.soilCOSmodel.__dict__[outputvar]
            else:
                raise Exception('invalid outputvar: '+str(outputvar))
            numderiv[str(alpha)] = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv[str(alpha)])
            elif printmode == 'relative':
                try:
                    print(numderiv[str(alpha)]/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if output_dict_model != 'vars_rsCm': #than we only have scalar output variables
            if not ((tl_output==0 and numderiv[str(alpharange[-1])]==0) and numderiv[str(alpharange[-2])]==0):
                if ((tl_output/numderiv[str(alpharange[-1])]<0.999 or tl_output/numderiv[str(alpharange[-1])]>1.001) and (tl_output/numderiv[str(alpharange[-2])]<0.999 or tl_output/numderiv[str(alpharange[-2])]>1.001)):
                    for i in range(5):
                        print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                    self.all_tests_pass = False
                    self.failed_grad_test_list.append(str(returnvariable))
        else: #than we can have scalar output variables as well as vectors and matrices
            if (np.size(numderiv[str(alpharange[-1])])>1 and np.size(numderiv[str(alpharange[-1])])<=default_model.soilCOSmodel.nr_nodes): #than it is a vector
                thistestfails = False
                for i in range(len(numderiv[str(alpharange[-1])])):
                    if not ((tl_output[i]==0 and numderiv[str(alpharange[-1])][i]==0) and numderiv[str(alpharange[-2])][i]==0): #in this test I say it passes if it is ok for the last or one but last value in alpha
                        if ((tl_output[i]/numderiv[str(alpharange[-1])][i]<0.999 or tl_output[i]/numderiv[str(alpharange[-1])][i]>1.001) and (tl_output[i]/numderiv[str(alpharange[-2])][i]<0.999 or tl_output[i]/numderiv[str(alpharange[-2])][i]>1.001)):
                            thistestfails = True
                            self.all_tests_pass = False
                if thistestfails: #to avoid printing for every element in the array
                    for i in range(5):
                        print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                    self.failed_grad_test_list.append(str(returnvariable))
            elif np.size(numderiv[str(alpharange[-1])])>default_model.soilCOSmodel.nr_nodes: #than it is a matrix
                thistestfails = False
                for i in range(len(numderiv[str(alpharange[-1])])):
                    for j in range(len(numderiv[str(alpharange[-1])][i])):
                        if not ((tl_output[i,j]==0 and numderiv[str(alpharange[-1])][i,j]==0) and numderiv[str(alpharange[-2])][i,j]==0):
                            if ((tl_output[i,j]/numderiv[str(alpharange[-1])][i,j]<0.999 or tl_output[i,j]/numderiv[str(alpharange[-1])][i,j]>1.001) and (tl_output[i,j]/numderiv[str(alpharange[-2])][i,j]<0.999 or tl_output[i,j]/numderiv[str(alpharange[-2])][i,j]>1.001)):
                                thistestfails = True
                                self.all_tests_pass = False
                if thistestfails: #to avoid printing for every element in the array
                    for i in range(5):
                        print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                    self.failed_grad_test_list.append(str(returnvariable))
            else: #than it is a scalar
                if not ((tl_output==0 and numderiv[str(alpharange[-1])]==0) and numderiv[str(alpharange[-2])]==0):
                    if ((tl_output/numderiv[str(alpharange[-1])]<0.999 or tl_output/numderiv[str(alpharange[-1])]>1.001) and (tl_output/numderiv[str(alpharange[-2])]<0.999 or tl_output/numderiv[str(alpharange[-2])]>1.001)):
                        for i in range(5):
                            print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                        self.all_tests_pass = False
                        self.failed_grad_test_list.append(str(returnvariable))
        self.gradienttesting = False
    
    def grad_test_run_surface_layer(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.run_surface_layer()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_rsl[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_surface_layer(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.run_surface_layer()
            #p_model.run(checkpoint=False,updatevals_surf_lay=True)
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_rsl[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
            for i in range(5):
                print('GRADIENT TEST FAILURE!! '+str(returnvariable))
            self.all_tests_pass = False
    
    def grad_test_ribtol(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        #this module is a bit special since it is a submodule
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.ribtol(default_model.Rib, default_model.zsl, default_model.z0m, default_model.z0h) 
        cpx =  default_model.cpx[-1]
        default_out = default_model.vars_rtl[outputvar] #we do not check for self variables, since Ribtol does not have any
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_ribtol(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.ribtol(p_model.Rib, p_model.zsl, p_model.z0m, p_model.z0h)
            p_out = p_model.vars_rtl[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_ags(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.ags()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_ags[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_ags(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-7]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.ags()
            #p_model.run(checkpoint=False,updatevals_surf_lay=True)
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_ags[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_run_mixed_layer(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.run_mixed_layer()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_rml[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_mixed_layer(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.run_mixed_layer()
            #p_model.run(checkpoint=False,updatevals_surf_lay=True)
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_rml[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_int_mixed_layer(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.integrate_mixed_layer()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_iml[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_integrate_mixed_layer(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.integrate_mixed_layer()
            #p_model.run(checkpoint=False,updatevals_surf_lay=True)
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_iml[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_run_radiation(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.run_radiation()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_rr[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_radiation(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.run_radiation()
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_rr[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_run_land_surface(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.run_land_surface()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_rls[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_land_surface(default_model,cpx,returnvariable=returnvariable)
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.run_land_surface()
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_rls[outputvar]
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_int_land_surface(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.integrate_land_surface()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_ils[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_integrate_land_surface(default_model,cpx,returnvariable=returnvariable)
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.integrate_land_surface()
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_ils[outputvar]  
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_statistics(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.statistics()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_stat[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_statistics(default_model,cpx,returnvariable=returnvariable)
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.statistics()
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_stat[outputvar]   
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_run_cumulus(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.run_cumulus()
        cpx =  default_model.cpx[-1]
        if outputvar in default_model.vars_rc:
            default_out = default_model.vars_rc[outputvar]
            if hasattr(default_model,outputvar):
                print ('WARNING: outputvar exists both as self-variable and in dictionary')    
        else:
            default_out = default_model.__dict__[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_cumulus(default_model,cpx,returnvariable=returnvariable)
        numderiv = {}
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.run_cumulus()
            if outputvar in default_model.vars_rc:
                p_out = p_model.vars_rc[outputvar]
            elif hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                raise Exception('invalid outputvar: '+str(outputvar))   
            numderiv[str(alpha)] = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv[str(alpha)])
            elif printmode == 'relative':
                try:
                    print(numderiv[str(alpha)]/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not ((tl_output==0 and numderiv[str(alpharange[-1])]==0) and numderiv[str(alpharange[-2])]==0):
            if ((tl_output/numderiv[str(alpharange[-1])]<0.999 or tl_output/numderiv[str(alpharange[-1])]>1.001) and (tl_output/numderiv[str(alpharange[-2])]<0.999 or tl_output/numderiv[str(alpharange[-2])]>1.001)):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def grad_test_run_soil_COS_mod(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        #this one is a bit special, since the forward model code is in a seperate class
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        #in this case the perturbed vars are partly in the argument list
        default_model.soilCOSmodel.run_soil_COS_mod(default_model.Tsoil,default_model.T2,default_model.wg,default_model.w2,default_model.COSsurf,default_model.Ps,default_model.Tsurf,default_model.wsat,default_model.dt)
        cpx =  default_model.cpx[-1]
        if outputvar in default_model.vars_rsCm:
            default_out = default_model.vars_rsCm[outputvar]
            if hasattr(default_model.soilCOSmodel,outputvar):
                print ('WARNING: outputvar exists both as self-variable and in dictionary')
        else:
            default_out = default_model.soilCOSmodel.__dict__[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_run_soil_COS_mod(default_model,cpx,returnvariable=returnvariable)
        #in this test I say it passes if it is ok for the last or one but last value in alpharange
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-12] #put at least two values here
        numderiv = {}
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                if hasattr(p_model.soilCOSmodel,item):
                    p_model.soilCOSmodel.__dict__[item] += alpha
                elif hasattr(p_model,item): #for those in the argument list of the function call to run_soil_COS_mod
                    p_model.__dict__[item] += alpha
                else:
                    raise Exception('invalid item in perturbationvars list '+str(outputvar))
            p_model.soilCOSmodel.run_soil_COS_mod(p_model.Tsoil,p_model.T2,p_model.wg,p_model.w2,p_model.COSsurf,p_model.Ps,p_model.Tsurf,p_model.wsat,p_model.dt)
            if outputvar in p_model.vars_rsCm:
                p_out = p_model.vars_rsCm[outputvar]
            else:
                p_out = p_model.soilCOSmodel.__dict__[outputvar]
            numderiv[str(alpha)] = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv[str(alpha)])
            elif printmode == 'relative':
                try:
                    print(numderiv[str(alpha)]/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if (np.size(numderiv[str(alpharange[-1])])>1 and np.size(numderiv[str(alpharange[-1])])<=default_model.soilCOSmodel.nr_nodes): #than it is a vector
            thistestfails = False
            for i in range(len(numderiv[str(alpharange[-1])])):
                if not ((tl_output[i]==0 and numderiv[str(alpharange[-1])][i]==0) and numderiv[str(alpharange[-2])][i]==0): #in this test I say it passes if it is ok for the last or one but last value in alpha
                    if ((tl_output[i]/numderiv[str(alpharange[-1])][i]<0.999 or tl_output[i]/numderiv[str(alpharange[-1])][i]>1.001) and (tl_output[i]/numderiv[str(alpharange[-2])][i]<0.999 or tl_output[i]/numderiv[str(alpharange[-2])][i]>1.001)):
                        thistestfails = True
                        self.all_tests_pass = False
            if thistestfails: #to avoid printing for every element in the array
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
        elif np.size(numderiv[str(alpharange[-1])])>default_model.soilCOSmodel.nr_nodes: #than it is a matrix
            thistestfails = False
            for i in range(len(numderiv[str(alpharange[-1])])):
                for j in range(len(numderiv[str(alpharange[-1])][i])):
                    if not ((tl_output[i,j]==0 and numderiv[str(alpharange[-1])][i,j]==0) and numderiv[str(alpharange[-2])][i,j]==0):
                        if ((tl_output[i,j]/numderiv[str(alpharange[-1])][i,j]<0.999 or tl_output[i,j]/numderiv[str(alpharange[-1])][i,j]>1.001) and (tl_output[i,j]/numderiv[str(alpharange[-2])][i,j]<0.999 or tl_output[i,j]/numderiv[str(alpharange[-2])][i,j]>1.001)):
                            thistestfails = True
                            self.all_tests_pass = False
            if thistestfails: #to avoid printing for every element in the array
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
        else: #than it is a scalar
            if not ((tl_output==0 and numderiv[str(alpharange[-1])]==0) and numderiv[str(alpharange[-2])]==0):
                if ((tl_output/numderiv[str(alpharange[-1])]<0.999 or tl_output/numderiv[str(alpharange[-1])]>1.001) and (tl_output/numderiv[str(alpharange[-2])]<0.999 or tl_output/numderiv[str(alpharange[-2])]>1.001)):
                    for i in range(5):
                        print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                    self.all_tests_pass = False
        
    def grad_test_store(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        #this test is a bit special since it involves output stored in the model output class
        varssetto0list = []
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.store()
        cpx =  default_model.cpx[-1]
        if outputvar in default_model.vars_sto:
            default_out = default_model.vars_sto[outputvar]
            if hasattr(default_model.out,outputvar):
                print ('WARNING: outputvar exists both as self.out-variable and in dictionary')
        else:
            default_out = default_model.out.__dict__[outputvar][-1]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_store(default_model,cpx,returnvariable=returnvariable)
        for alpha in [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned, without the extra call to surf layer
            for item in perturbationvars:
                if not hasattr(p_model,item) or p_model.__dict__[item] is None: #only for grad test, in case e.g. surf layer not turned on
                    p_model.__dict__[item] = 0.
                    if item not in varssetto0list:
                        varssetto0list.append(item)
                p_model.__dict__[item] += alpha
            p_model.store()
            if outputvar in default_model.vars_sto:
                p_out = p_model.vars_sto[outputvar]
            elif hasattr(p_model.out,outputvar):
                p_out = p_model.out.__dict__[outputvar][-1]
            else:
                raise Exception('invalid outputvar: '+str(outputvar))
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        if len(varssetto0list) > 0:
            print('Variables set to 0: '+str(varssetto0list))
        
    def grad_test_jarvis_stewart(self,model,perturbationvars,outputvar,dstate,returnvariable,printmode='absolute'):
        dummy_model = cp.deepcopy(model) #just to not manipulate orig model
        dummy_model.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=True)
        default_model = cp.deepcopy(dummy_model)
        default_model.jarvis_stewart()
        cpx =  default_model.cpx[-1]
        if hasattr(default_model,outputvar):
            default_out = default_model.__dict__[outputvar]
        else:
            default_out = default_model.vars_js[outputvar]
        print(returnvariable+' :')
        self.initialise_tl(dstate)
        tl_output = self.tl_jarvis_stewart(default_model,cpx,returnvariable=returnvariable)
        alpharange = [1e-2,1e-4,1e-5,1e-6,1e-8,1e-9]
        for alpha in alpharange:
            p_model = cp.deepcopy(dummy_model) #here we need to use a model that has been runned
            for item in perturbationvars:
                p_model.__dict__[item] += alpha
            p_model.jarvis_stewart()
            if hasattr(p_model,outputvar):
                p_out = p_model.__dict__[outputvar]
            else:
                p_out = p_model.vars_js[outputvar]   
            numderiv = (p_out - default_out)/alpha
            if printmode == 'absolute':
                print(numderiv)
            elif printmode == 'relative':
                try:
                    print(numderiv/tl_output)
                except ZeroDivisionError:
                    print('NAN')
        if printmode == 'absolute':
            print('tl :'+str(tl_output))
        if not (tl_output==0 and numderiv==0):
            if (tl_output/numderiv<0.999 or tl_output/numderiv>1.001):
                for i in range(5):
                    print('GRADIENT TEST FAILURE!! '+str(returnvariable))
                self.all_tests_pass = False
        
    def adjoint_test(self,model,x_variables,Hx_variable,y_variable,HTy_variables,Hx_dict):
        #Note that the order of x and HTy should be identical, e.g. the first element of HTy should be the adjoint variable of the first element of HTy!!
        if not hasattr(self,'failed_adj_test_list'):
            self.failed_adj_test_list = [] #list of vars where this test fails
        self.adjointtesting = True
        testmodel = cp.deepcopy(model)
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False,save_vars_indict=False)
        cpx = testmodel.cpx
        cpx_init =  testmodel.cpx_init
        self.initialise_tl({})
        self.x_nrs = [] #This makes it applicable for variables that require to be initialised with an array, as well as those that require a scalar
        for i in range(len(x_variables)):
            randnrs = np.random.random_sample(np.size(self.__dict__[x_variables[i]]))
            if np.size(randnrs) ==1:
                randnrs = randnrs[0] #because np.random.random_sample(1) gives an array containing 1 number, we just want the number
            self.__dict__[x_variables[i]] = randnrs
            self.x_nrs.append(randnrs)
        self.tl_full_model(testmodel,cpx,cpx_init,Hx_variable=Hx_variable,tl_dict=Hx_dict)
        try:
            columnlength = len(self.Hx[0][0])
            self.y = np.zeros((model.tsteps,len(self.Hx[0]),columnlength)) #matrix for every timestep, statement above raises TypeError or IndexError if Hx not a matrix
            for i in range(model.tsteps):
                for j in range(len(self.Hx[0])): #use len() instead of numpy.size!!, np.size looks at nr of elements
                    for k in range(len(self.Hx[0][0])):
                        self.y[i,j,k] = np.random.random_sample(1)[0] #because np.random.random_sample(1) gives an array containing 1 number
        except (TypeError,IndexError):
            self.y = np.zeros((model.tsteps,np.size(self.Hx[0]))) #in case not a matrix
            for i in range(model.tsteps):
                for j in range(np.size(self.Hx[0])):
                    self.y[i,j] = np.random.random_sample(1)[0] #because np.random.random_sample(1) gives an array containing 1 number
        self.initialise_adjoint()
        #the following code was used before the time loop and the initialisation was made part of adjoint function: 
#        forcingdummy = {}
#        for i in range(testmodel.tsteps-1,-1,-1):
#            self.__dict__[y_variable] += self.y[i] #ESSENTIAL TO USE += !!!
#        self.adjoint(forcingdummy,cpx[i],testmodel) #important that forcings are zero
#        self.adj_init(cpx_init,testmodel) #the adjoint of the model initialisation part
        forcing = []
        for i in range(testmodel.tsteps):
            forcing.append ({}) #it has to be a list of dictionaries
        for i in range(testmodel.tsteps-1,-1,-1):
            forcing[i][y_variable] = self.y[i] #forcing will be added to existing y_variable
        self.adjoint(forcing,cpx,cpx_init,testmodel) #important that forcings are zero
        self.x =[] #x and HTy have equal size and structure
        self.HTy =[] #just like x, this is only at one instance in time
        for i in range(np.size(HTy_variables)):
            self.HTy.append([])
            self.x.append([])
        for i in range(len(self.x_nrs)): #do not use np.size on a list of arrays
            self.HTy[i] = self.__dict__[HTy_variables[i]]
            self.x[i] = self.x_nrs[i]
        if np.size(self.Hx[0]) > 1: #than it is not a scalar
            self.Hx = np.ndarray.flatten(np.array(self.Hx))
            self.y = np.ndarray.flatten(np.array(self.y))
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = 0
        for i in range(len(self.x)):
            dotelem = np.dot(np.array(self.x[i]),np.array(self.HTy[i]))
            dotxhty += dotelem
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-13:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
            self.failed_adj_test_list.append(str(Hx_variable))
        self.adjointtesting = False
        
    def adjoint_test_surf_lay(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_surf_lay = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_run_surface_layer(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rsl:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rsl:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rsl[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_run_surface_layer(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_surf_lay = False
        
    def adjoint_test_surf_lay_manual(self,testmodel):
        #for this function, place HTy, Hx etc manually in the code
        self.manualadjointtesting = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        rand1 = np.random.random_sample(1)[0]
        rand2 = np.random.random_sample(1)[0]
        rand3 = np.random.random_sample(1)[0]
        rand4 = np.random.random_sample(1)[0]
        self.x = np.array([rand1])
        self.initialise_tl({})
        self.tl_run_surface_layer(testmodel,checkpoint_test)
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        forcingdummy = {}
        self.adj_run_surface_layer(forcingdummy,checkpoint_test,testmodel) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference',dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy)
        if abs((dothxy-dotxhty)/dothxy) > 9.99e-16:
            for i in range(5):
                print("ADJOINT TEST FAILURE!")
            self.all_tests_pass = False
        self.manualadjointtesting = False
        
    def adjoint_test_ribtol(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingribtol = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[-1] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_ribtol(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rtl:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rtl:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rtl[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_ribtol(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingribtol = False
        
    def adjoint_test_ags(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingags = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_ags(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_ags:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_ags:
            if key == Hx_variable:
                self.Hx = self.Output_tl_ags[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_ags(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingags = False
        
    def adjoint_test_ags_manual(self,testmodel):
        #for this function, place HTy, Hx etc manually in the code
        self.manualadjointtesting = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        rand1 = np.random.random_sample(1)[0]
        rand2 = np.random.random_sample(1)[0]
        rand3 = np.random.random_sample(1)[0]
        rand4 = np.random.random_sample(1)[0]
        self.x = np.array([rand1,rand2])
        self.initialise_tl({})
        self.tl_ags(testmodel,checkpoint_test)
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        forcingdummy = {}
        self.adj_ags(forcingdummy,checkpoint_test,testmodel) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference',dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy)
        if abs((dothxy-dotxhty)/dothxy) > 9.99e-16:
            for i in range(5):
                print("ADJOINT TEST FAILURE!")
            self.all_tests_pass = False
        self.manualadjointtesting = False
              
    def adjoint_test_run_mixed_layer(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_mixed_layer = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_run_mixed_layer(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rml:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rml:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rml[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_run_mixed_layer(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_mixed_layer = False
        
    def adjoint_test_int_mixed_layer(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingint_mixed_layer = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_integrate_mixed_layer(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_iml:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_iml:
            if key == Hx_variable:
                self.Hx = self.Output_tl_iml[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_integrate_mixed_layer(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingint_mixed_layer = False
        
    def adjoint_test_run_radiation(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_radiation = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_run_radiation(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rr:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rr:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rr[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_run_radiation(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15:
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_radiation = False
        
    def adjoint_test_run_land_surface(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_land_surface = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_run_land_surface(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rls:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rls:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rls[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_run_land_surface(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 2e-14: #note that this is not zero at all.. 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_land_surface = False
        
    def adjoint_test_run_land_surface_manual(self,testmodel):
        #for this function, place HTy, Hx etc manually in the code
        self.manualadjointtesting = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(18)
        self.initialise_tl({})
        self.tl_run_land_surface(testmodel,checkpoint_test)
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        forcingdummy = {}
        self.adj_run_land_surface(forcingdummy,checkpoint_test,testmodel) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference',dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy)
        if abs((dothxy-dotxhty)/dothxy) > 9.99e-16:
            for i in range(5):
                print("ADJOINT TEST FAILURE!")
            self.all_tests_pass = False
        self.manualadjointtesting = False
        
    def adjoint_test_int_land_surface(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingint_land_surface = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_integrate_land_surface(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_ils:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_ils:
            if key == Hx_variable:
                self.Hx = self.Output_tl_ils[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_integrate_land_surface(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 2e-14: #note that this is not zero at all.. 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingint_land_surface = False
        
    def adjoint_test_statistics(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingstatistics = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[-1] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_statistics(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_stat:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_stat:
            if key == Hx_variable:
                self.Hx = self.Output_tl_stat[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_statistics(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15: 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingstatistics = False
        
    def adjoint_test_run_cumulus(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_cumulus = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0] #index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_run_cumulus(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rc:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rc:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rc[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_run_cumulus(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15: 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_cumulus = False
        
    def adjoint_test_run_soil_COS_mod(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingrun_soil_COS_mod = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0]  #index for time step, use the same for adjoint and TL
        self.initialise_tl({})
        self.x_dict = {} #create dictionary, because some variables require to be initialised with an array, some with a scalar
        for i in range(len(x_variables)):
            self.x_dict[x_variables[i]] = np.random.random_sample(np.size(self.__dict__[x_variables[i]]))
            self.__dict__[x_variables[i]] = self.x_dict[x_variables[i]] 
        self.tl_run_soil_COS_mod(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_rsCm:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_rsCm:
            if key == Hx_variable:
                self.Hx = self.Output_tl_rsCm[Hx_variable]
        if np.size(self.Hx) <= testmodel.soilCOSmodel.nr_nodes:
            self.y = np.random.random_sample(np.size(self.Hx))
        else:
            self.y = np.zeros((len(self.Hx),len(self.Hx[0]))) #so a matrix
            for i in range(len(self.Hx)):
                for j in range(len(self.Hx[0])):
                    self.y[i,j] = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = cp.deepcopy(self.y)
        forcingdummy = {}
        self.adj_run_soil_COS_mod(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        #to prevent that dothxy is a matrix product, while dotxhty is not
        if np.size(self.Hx) > testmodel.soilCOSmodel.nr_nodes:
            self.Hx = np.ndarray.flatten(self.Hx)
            self.y = np.ndarray.flatten(self.y)
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        #now x and HTy ar a mix of scalars and vectors
        self.x =[] #x and HTy have equal size and structure
        self.HTy =[]
        for item in self.x_dict:
            if np.size(self.x_dict[item]) == 1: #than it is a scalar
                self.x.append(self.x_dict[item][0]) #because self.x_dict[item] is an array containing 1 number
                self.HTy.append(self.HTy_dict['a'+item])
            else:
                for i in range(len(self.x_dict[item])):
                    self.x.append(self.x_dict[item][i])
                    self.HTy.append(self.HTy_dict['a'+item][i])
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15: 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingrun_soil_COS_mod = False
        
    def adjoint_test_store(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingstore = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[0]
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_store(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_sto:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_sto:
            if key == Hx_variable:
                self.Hx = self.Output_tl_sto[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_store(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15: 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingstore= False
        
    def adjoint_test_jarvis_stewart(self,testmodel,x_variables,Hx_variable,y_variable,HTy_variables):
        self.adjointtestingjarvis_stewart = True
        testmodel.run(checkpoint=True,updatevals_surf_lay=True,delete_at_end=False)
        checkpoint_test = testmodel.cpx[-1] # index for time step, use the same for adjoint and TL
        self.x = np.random.random_sample(len(x_variables))
        self.initialise_tl({})
        for i in range(len(self.x)):
            self.__dict__[x_variables[i]] = self.x[i]
        self.tl_jarvis_stewart(testmodel,checkpoint_test)
        if Hx_variable not in self.Output_tl_js:
            print('ERROR, HX VARIABLE NOT IN TL OUTPUT')    
        for key in self.Output_tl_js:
            if key == Hx_variable:
                self.Hx = self.Output_tl_js[Hx_variable]
        self.y = np.random.random_sample(1)[0]
        self.initialise_adjoint()
        self.__dict__[y_variable] = self.y
        forcingdummy = {}
        self.adj_jarvis_stewart(forcingdummy,checkpoint_test,testmodel,HTy_variables=HTy_variables) #important that forcings are zero
        dothxy = np.dot(np.array(self.Hx),np.array(self.y))
        dotxhty = np.dot(np.array(self.x),np.array(self.HTy))
        print('<Hx,y>,<x,Hty>, rel difference: % 10.4E % 10.4E % 10.4E'%(dothxy,dotxhty,abs(dothxy-dotxhty)/dothxy))
        if abs((dothxy-dotxhty)/dothxy) > 5e-15: 
            for i in range(5):
                print("ADJOINT TEST FAILURE! "+str(Hx_variable))
            self.all_tests_pass = False
        self.adjointtestingjarvis_stewart = False