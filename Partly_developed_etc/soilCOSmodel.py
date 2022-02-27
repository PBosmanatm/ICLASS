import numpy as np
import copy as cp
import inspect

class soil_COS_mod:

    def __init__(self,inputdata,mol_rat_ocs_atm,pressure,airtemp,mothermodel=None):
        if mothermodel != None: #mothermodel is class model calling the soil model
            self.mothermodel = mothermodel
        #pressure in Pa, mol_rat_ocs in ppb
        self.Rgas = 8.3145
        self.Km = 1.9 #mol/m3,
        self.dGcat, self.dHeq = 84.10e3, 358.9e3 #J/mol, not kJ/mol!!
        self.SunTref = 298.15
        self.nr_nodes = inputdata.nr_nodes
        self.z_soil = np.zeros(self.nr_nodes)
        self.dz_soil = np.zeros(self.nr_nodes)
        self.z_int = np.zeros(self.nr_nodes)
        self.T_nodes = np.zeros(self.nr_nodes)
        self.s_moist = np.zeros(self.nr_nodes)
        for i in range(0, self.nr_nodes):
            y = 0.2 * (self.nr_nodes-1) #in case of different nr of nodes as in Sunpaper
            self.z_soil[i] = np.exp(0.2*i-y) #location of the soil nodes
        for i in range(0, self.nr_nodes-1):
            self.z_int[i] = (self.z_soil[i]+self.z_soil[i+1])/2 #depth of the interfaces
        self.z_int[self.nr_nodes-1] = self.z_soil[self.nr_nodes-1] + (self.z_soil[self.nr_nodes-1]-self.z_soil[self.nr_nodes-2])/2
        self.dz_soil[0] = (self.z_soil[0]+self.z_soil[1])/2
        for i in range(1, self.nr_nodes-1):
            self.dz_soil[i] = (self.z_soil[i+1]-self.z_soil[i-1])/2
        self.dz_soil[self.nr_nodes-1] = self.z_soil[self.nr_nodes-1] - self.z_soil[self.nr_nodes-2]
        C_air_init = mol_rat_ocs_atm * 1.e-9 * pressure /(self.Rgas*airtemp) #in mol/m3
        self.C_soilair_current = np.zeros(self.nr_nodes)
        for i in range(self.nr_nodes):
            self.C_soilair_current[i] = C_air_init  #initialise with air concentration
        #self.C_soilair_current is checkpointed at the start of run_soil_COS_mod
        self.sw_soiltemp = inputdata.sw_soiltemp
        if self.sw_soiltemp == 'Sunpaper':
            self.z_T == inputdata.z_T
        self.time = inputdata.tstart * 3600
        self.sw_soilmoisture = inputdata.sw_soilmoisture
        self.kH_type = inputdata.kH_type
        self.Diffus_type = inputdata.Diffus_type
        if self.Diffus_type == 'Sun':
            self.b_sCOSm = inputdata.b_sCOSm
        if self.Diffus_type == 'Ogee':
            self.ta_type = inputdata.ta_type
            self.tl_type = inputdata.tl_type
            if ((self.tl_type == 'Mol03r' or self.tl_type == 'Mol03u') or (self.ta_type == 'Mol03u')):
                self.b_pore = inputdata.b_pore
        self.uptakemodel = inputdata.uptakemodel
        if 	(self.uptakemodel == 'Sun'):
            self.s_moist_opt = inputdata.s_moist_opt
            self.Teq = inputdata.Teq
            self.At = self.determine_At(self.dGcat,self.dHeq,self.Teq)
            self.Aw = self.determine_Aw()
        if self.uptakemodel == 'newSun':
            self.inc_abi_newSun = inputdata.inc_abi_newSun
            self.pH_newSun = inputdata.pH_newSun
        if self.uptakemodel == 'Ogee':
            self.fCA = inputdata.fCA
        if (self.uptakemodel == 'newSun') or (self.uptakemodel == 'Sun'):
            self.Vsumax = inputdata.Vsumax
        self.Vspmax = inputdata.Vspmax
        self.Q10 = inputdata.Q10
        if hasattr(inputdata,'layer1_2division'):
            self.layer1_2division = inputdata.layer1_2division
        self.write_soilCOS_to_f = inputdata.write_soilCOS_to_f
        if self.write_soilCOS_to_f:
            self.nr_nodes_for_filewr = inputdata.nr_nodes_for_filewr
            open('soilCOS.txt','w').write('{0:>25s}'.format('time')) #here we make the file
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format('Conc_'+str(i)))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format('flux_'+str(i)))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format('kh_'+str(i)))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format('prod_'+str(i)))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format('upt_'+str(i)))
        if self.mothermodel.checkpoint:
            self.mothermodel.cpx_init[0]['isCm_airtemp'] = airtemp
            self.mothermodel.cpx_init[0]['isCm_Rgas_end'] = self.Rgas
            self.mothermodel.cpx_init[0]['isCm_pressure'] = pressure
            self.mothermodel.cpx_init[0]['isCm_mol_rat_ocs_atm'] = mol_rat_ocs_atm
            
    def run_soil_COS_mod(self,Tsoil,T2,wg,w2,mol_rat_ocs_atm,pressure,airtemp,wsat,dt,call_from_init=False):
        self.mothermodel.vars_rsCm = {}
        if call_from_init:
            self.call_from_init = True
        if self.mothermodel.checkpoint:
            if call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_airtemp'] = airtemp
                self.mothermodel.cpx_init[0]['rsCm_mol_rat_ocs_atm'] = mol_rat_ocs_atm
                self.mothermodel.cpx_init[0]['rsCm_C_soilair_current'] = cp.deepcopy(self.C_soilair_current)
                self.mothermodel.cpx_init[0]['rsCm_Q10'] = self.Q10
                self.mothermodel.cpx_init[0]['rsCm_SunTref'] = self.SunTref
                self.mothermodel.cpx_init[0]['rsCm_Vspmax'] = self.Vspmax
                self.mothermodel.cpx_init[0]['rsCm_wsat'] = wsat
                self.mothermodel.cpx_init[0]['rsCm_dt'] = dt
                self.mothermodel.cpx_init[0]['rsCm_fCA'] = self.fCA
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_airtemp'] = airtemp
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_mol_rat_ocs_atm'] = mol_rat_ocs_atm
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_C_soilair_current'] = cp.deepcopy(self.C_soilair_current)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_Q10'] = self.Q10
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_SunTref'] = self.SunTref
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_Vspmax'] = self.Vspmax
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_wsat'] = wsat
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_dt'] = dt
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_fCA'] = self.fCA
            
        C_air = mol_rat_ocs_atm * 1.e-9 * pressure /(self.Rgas*airtemp) #gas law, n/V = p/(RT) and mol/m3 = mol_OCS/mol_air * mol_air/m3
        for i in range(0,self.nr_nodes):
            if self.sw_soiltemp == 'Sunpaper':
                omega = 2*3.14159265359 / (3600*24)
                psi = -0.5*3.14159265359 #psi from Enpei model
                T_s = 298.15 ##to be chosen, this is Enpei
                T_F=10. #to be chosen, this is Enpei
                self.T_nodes[i] = T_s + T_F * np.exp(-self.z_soil[i]/self.z_T) * np.sin(omega * self.time + psi - self.z_soil[i]/self.z_T) #eq 26 Sun
            elif self.sw_soiltemp == 'simple':
                if (self.z_soil[i] > self.layer1_2division):
                    self.T_nodes[i] = T2
                else:
                    self.T_nodes[i] = Tsoil
            elif self.sw_soiltemp == 'interpol': ##y= y1 + (x-x1) * (y2-y1)/(x2-x1); y1 is Tsoil, x1 is 0
                self.T_nodes[i] = Tsoil + (self.z_soil[i] - 0) * (T2 - Tsoil)/(1 - 0)
            else:
                raise Exception('ERROR: Problem in soiltemp switch inputdata')   
            if self.sw_soilmoisture == 'simple':
                if (self.z_soil[i] > self.layer1_2division):
                    self.s_moist[i] = w2
                else:
                    self.s_moist[i] = wg
            elif self.sw_soilmoisture == 'interpol':
                self.s_moist[i] = wg + (self.z_soil[i] - 0) * (w2 - wg)/(1 - 0)
            else:
                raise Exception('ERROR: Problem in soilmoisture switch inputdata')
        kH = self.calckH()
        s_uptake = self.soil_uptake(kH,wsat)
        s_prod = self.soil_prod()
        diffus = self.calcD(wsat,airtemp,pressure,kH)
        conduct = self.calcG(diffus)
        self.C_soilair_next = self.calcC(kH,conduct,diffus, s_uptake, s_prod, pressure,mol_rat_ocs_atm,airtemp,wsat,C_air,dt) #so conc for next timestep
        OCS_fluxes = self.calcJ(conduct,self.C_soilair_current,C_air) #Output in mol/m2/s
        COS_netuptake_soilsun = -1*OCS_fluxes[0] #uptake should be negative for mixed layer model
        if (self.write_soilCOS_to_f):
            open('soilCOS.txt','a').write('\n')
            open('soilCOS.txt','a').write('{0:>25s}'.format(str(self.time)))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format(str(self.C_soilair_current[i])))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format(str(OCS_fluxes[i])))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format(str(kH[i])))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format(str(s_prod[i])))
            for i in range(self.nr_nodes_for_filewr):
                open('soilCOS.txt','a').write('{0:>25s}'.format(str(s_uptake[i])))
        self.C_soilair_current = cp.deepcopy(self.C_soilair_next)
        self.time += dt
        
        if self.mothermodel.checkpoint:
            if call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_T_nodes_end'] = cp.deepcopy(self.T_nodes)
                self.mothermodel.cpx_init[0]['rsCm_s_moist_end'] = cp.deepcopy(self.s_moist)
                self.mothermodel.cpx_init[0]['rsCm_C_air_end'] = cp.deepcopy(C_air)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_T_nodes_end'] = cp.deepcopy(self.T_nodes)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_s_moist_end'] = cp.deepcopy(self.s_moist)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_C_air_end'] = cp.deepcopy(C_air)       
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.run_soil_COS_mod).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        self.call_from_init = False #just to make sure it's off for future calls
        return COS_netuptake_soilsun
           
    def soil_uptake(self,kH,wsat):
        if 	(self.uptakemodel == 'Sun'):
            s_uptake = self.soil_uptake_Sun(kH)
        elif self.uptakemodel == 'Ogee':
            s_uptake = self.soil_uptake_Ogee(kH)
        elif self.uptakemodel == 'newSun':
            s_uptake = self.soil_uptake_newSun(kH,wsat)
        else:
            raise Exception('ERROR: Problem with uptake in switch inputdata')
        return s_uptake
          
    def soil_uptake_Sun(self,kH):
        ft = self.At * self.T_nodes * np.exp(-self.dGcat/(self.Rgas*self.T_nodes)) / (1. + np.exp(-self.dHeq/self.Rgas * (1./self.T_nodes - 1./self.Teq))) #eq 22
        gw = self.Aw * self.s_moist/(self.s_moist_opt**2) * np.exp(-(self.s_moist**2)/(self.s_moist_opt**2))  #eq 23 
        s_uptake = -self.Vsumax * kH * self.C_soilair_current /(self.Km + kH * self.C_soilair_current) * ft * gw
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.soil_uptake_Sun).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return s_uptake
    
    def soil_uptake_Ogee(self,kH): #paper Ogee 2016
        #fCA = 111*CAconc #text below eq 11b
        #fCA = 3.e4 #From Enpei
        deltaHa = 40.e3
        deltaHd = 200.e3
        deltaSd = 660.
        pKw = 14.
        pH_ref = 4.5
        xCAref = np.exp(-deltaHa/(self.Rgas*self.SunTref)) / (1. + np.exp(-deltaHd/(self.Rgas*self.SunTref) + deltaSd/self.Rgas))
        xCA = np.exp(-deltaHa/(self.Rgas*self.T_nodes)) / (1. + np.exp(-deltaHd/(self.Rgas*self.T_nodes) + deltaSd/self.Rgas)) #eq 10a Ogee
        kuncat_ref=2.15e-5 * np.exp(-10450*(1./self.SunTref-1./298.15)) + 12.7*10**(-pKw+pH_ref) * np.exp(-6040*(1./self.SunTref-1./298.15)) #in Ogee 298 instead of 298.15
        ktot = self.fCA*kuncat_ref*xCA/xCAref #eq 11b
        s_uptake = -ktot * kH * (self.s_moist) * self.C_soilair_current #p8 Ogee, above eq 11a ;B from Ogee is kH	
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_deltaHa_end'] = deltaHa
                self.mothermodel.cpx_init[0]['rsCm_deltaHd_end'] = deltaHd
                self.mothermodel.cpx_init[0]['rsCm_deltaSd_end'] = deltaSd
                self.mothermodel.cpx_init[0]['rsCm_xCA_end'] = cp.deepcopy(xCA)     
                self.mothermodel.cpx_init[0]['rsCm_kuncat_ref_end'] = kuncat_ref
                self.mothermodel.cpx_init[0]['rsCm_xCAref_end'] = xCAref
                self.mothermodel.cpx_init[0]['rsCm_ktot_end'] = cp.deepcopy(ktot)
                self.mothermodel.cpx_init[0]['rsCm_kH_end'] = cp.deepcopy(kH)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_deltaHa_end'] = deltaHa
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_deltaHd_end'] = deltaHd
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_deltaSd_end'] = deltaSd
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_xCA_end'] = cp.deepcopy(xCA)         
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_kuncat_ref_end'] = kuncat_ref
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_xCAref_end'] = xCAref
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_ktot_end'] = cp.deepcopy(ktot)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_kH_end'] = cp.deepcopy(kH)
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.soil_uptake_Ogee).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return s_uptake
        
    def soil_uptake_newSun(self,kH,wsat):
        newSunDelta_G_a = 8.41e+4
        newSunDelta_H_d = 3.59e+5
        newSunDelta_S_d = 1.236e+3 #1.224e+3 if you calculate it as dHeq/Teq as in Sunpaper
        Km_newSun = 3.9e-2 #See Ogee paper
        s_uptake = np.zeros(self.nr_nodes)
        for i in range(0,self.nr_nodes):
            s_uptake[i] = - self.Vsumax * kH[i] * self.C_soilair_current[i] / (Km_newSun + kH[i] * self.C_soilair_current[i]) * \
            self.f_enzyme_temp_dependence(self.T_nodes[i], newSunDelta_G_a, newSunDelta_H_d, newSunDelta_S_d) / \
            self.f_enzyme_temp_dependence(self.SunTref, newSunDelta_G_a, newSunDelta_H_d, newSunDelta_S_d) * \
            self.s_moist[i] / wsat #kH added to correct error of Sun
            if (self.inc_abi_newSun  == True):
                s_uptake[i] = s_uptake[i] - self.hydrolysis_cos(self.T_nodes[i], self.pH_newSun) * kH[i] * self.C_soilair_current[i]
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.soil_uptake_newSun).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return s_uptake
    
    def f_enzyme_temp_dependence(self,temp, newSunDelta_G_a, newSunDelta_H_d, newSunDelta_S_d): #From Sun's own implementation in SIB4
        res = temp * np.exp(-newSunDelta_G_a / (self.Rgas* temp)) / (1 + np.exp((newSunDelta_S_d - newSunDelta_H_d / temp) / self.Rgas))
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.f_enzyme_temp_dependence).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return res
        
    def hydrolysis_cos(self,temp, pH):
        params = [2.11834513803e-5,1.04183722377e+4,1.416881179e+1,6.46911889197e+3]
        c_OH = 10 ** (pH - 14)  # OH concentration [mol L^-1]
        delta_Tinv = 1 / temp - 1 / self.SunTref
        k_hyd = params[0] * np.exp(-params[1] * delta_Tinv) + \
        params[2] * np.exp(-params[3] * delta_Tinv) * c_OH
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.hydrolysis_cos).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return k_hyd
    
    def determine_At(self,dGcat,dHeq,Teq): #we need to find the max value of the function f(T) to determine At for eq 22
        x=np.zeros(100000)
        y=np.zeros(100000)
        x[0] = 250. #we test temps from 250 to 320K (see also excel graph)
        for i in range(1,100000):
            x[i] = x[i-1] + 70./100000
            y[i] = x[i] * np.exp(-dGcat/(self.Rgas*x[i])) / (1 + np.exp(-dHeq/self.Rgas * (1/x[i] - 1/Teq)))
        At = 1/np.max(y)
        return At
    
    def determine_Aw(self): #we need to find the max value of the function f(T) to determine At for eq 22
        x=np.zeros(100000)
        y=np.zeros(100000)
        x[0] = 0. #we test moisture from 0 to 1
        for i in range(1,100000):
            x[i] = x[i-1] + 1./100000
            y[i] = x[i]/(self.s_moist_opt**2) * np.exp(-(x[i]**2)/(self.s_moist_opt**2))
        Aw = 1/np.max(y)
        return Aw
    
    def calckH(self):
#        Mw = 18.e-3 #Mw in kg/mol
#        p_ref = 100000 
        if self.kH_type == 'Sun':
            alfa = -20.
            beta = 4050.32 #for eq 20 Sun, see fig 2
            #K_eq20 = Mw * p_ref /(1000 * Rgas) #1000 is rho water; K_eq20 in Kelvin
            K_eq20 =1. #probably an error in the paper, the K is included in the alfa and beta values, see also fig2 in paper
            kH = (self.T_nodes / K_eq20) * np.exp(alfa + beta * K_eq20 / self.T_nodes)
            if self.mothermodel.checkpoint:
                if self.call_from_init:
                    self.mothermodel.cpx_init[0]['rsCm_alfa_end'] = alfa
                    self.mothermodel.cpx_init[0]['rsCm_beta_end'] = beta
                    self.mothermodel.cpx_init[0]['rsCm_K_eq20_end'] = K_eq20
                else:
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_alfa_end'] = alfa
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_beta_end'] = beta
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_K_eq20_end'] = K_eq20
        elif self.kH_type == 'Ogee':
            kHog=0.021*np.exp(24900/self.Rgas*(1/self.T_nodes-1/298.15)) #mol m-3 Pa-1#p4 Ogee
            kH = kHog*self.Rgas*self.T_nodes*0.01 #0.01 due to error in paper
            if self.mothermodel.checkpoint:
                if self.call_from_init:
                    self.mothermodel.cpx_init[0]['rsCm_kHog_end'] = cp.deepcopy(kHog)
                else:
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_kHog_end'] = cp.deepcopy(kHog)
        else:
            raise Exception('ERROR: Problem in kH_type switch inputdata')
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self'): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return kH
    
    def soil_prod(self):
        s_prod = self.Vspmax * self.Q10 **((self.T_nodes-self.SunTref)/10.0) #rewritten eq 24
        #Q10 = 1.9 in eq 24, we use it from inputdata
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self'): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return s_prod
        
    def calcC(self,kH,conduct,diffus,s_uptake, s_prod, pressure,mol_rat_ocs_atm,airtemp,wsat,C_air,dt): #see eq 21,22 and 23
        #mol_rat_ocs_atm is in ppb, pressure in hPa
        source = (s_uptake+s_prod)*self.dz_soil
        D_a_0 = diffus[0]
        source[0] = (s_uptake[0]+s_prod[0])*self.dz_soil[0] + D_a_0 / self.z_soil[0] * C_air
        eta = kH * self.s_moist + (wsat - self.s_moist)
        A_matr = np.zeros((self.nr_nodes,self.nr_nodes))
        for i in range(self.nr_nodes):
            A_matr[i,i] = eta[i] * self.dz_soil[i]
        B_matr = np.zeros((self.nr_nodes,self.nr_nodes))
        for i in range(self.nr_nodes-1):
            B_matr[i,i] = -(conduct[i]+conduct[i+1])
            B_matr[i,i+1] = conduct[i+1]
        B_matr[self.nr_nodes-1,self.nr_nodes-1] = -conduct[self.nr_nodes-1]
        for i in range(1,self.nr_nodes):
            B_matr[i,i-1] = conduct[i]
        try:
            invmatreq12 = np.linalg.inv(2*A_matr - dt*B_matr)
        except np.linalg.linalg.LinAlgError:
            invmatreq12 = np.zeros((self.nr_nodes,self.nr_nodes))
            invmatreq12[:] = np.nan
        matr_2_eq12 = np.matmul(2*A_matr + dt * B_matr, self.C_soilair_current)
        matr_3_eq12 = matr_2_eq12 + 2*dt* source #
        C_soilair = np.matmul(invmatreq12,matr_3_eq12)
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_C_soilair_middle'] = cp.deepcopy(C_soilair) #ESSENTIAL TO COPY!!!
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_C_soilair_middle'] = cp.deepcopy(C_soilair) #ESSENTIAL TO COPY!!!
        for i in range(self.nr_nodes):
            if (C_soilair[i] < 0.0): #avoid negative conc:
                C_soilair[i] = 0.0
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_D_a_0_end'] = cp.deepcopy(D_a_0)
                self.mothermodel.cpx_init[0]['rsCm_A_matr_end'] = cp.deepcopy(A_matr)
                self.mothermodel.cpx_init[0]['rsCm_B_matr_end'] = cp.deepcopy(B_matr)
                self.mothermodel.cpx_init[0]['rsCm_matr_2_eq12_end'] = cp.deepcopy(matr_2_eq12)
                self.mothermodel.cpx_init[0]['rsCm_matr_3_eq12_end'] = cp.deepcopy(matr_3_eq12)
                self.mothermodel.cpx_init[0]['rsCm_invmatreq12_end'] = cp.deepcopy(invmatreq12)
                self.mothermodel.cpx_init[0]['rsCm_source_end'] = cp.deepcopy(source)
                self.mothermodel.cpx_init[0]['rsCm_C_soilair_end'] = cp.deepcopy(C_soilair)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_D_a_0_end'] = cp.deepcopy(D_a_0)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_A_matr_end'] = cp.deepcopy(A_matr)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_B_matr_end'] = cp.deepcopy(B_matr)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_matr_2_eq12_end'] = cp.deepcopy(matr_2_eq12)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_matr_3_eq12_end'] = cp.deepcopy(matr_3_eq12)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_invmatreq12_end'] = cp.deepcopy(invmatreq12)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_source_end'] = cp.deepcopy(source)
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_C_soilair_end'] = cp.deepcopy(C_soilair)
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.calcC).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return C_soilair
    
    def calcD(self,wsat,airtemp,pressure,kH):#see eq 21,22 and 23. kH and pressure only needed for Ogee
        diffus = np.zeros(self.nr_nodes)
        if self.Diffus_type == ('Sun'):
            Dm = 1.337e-5
            n = 1.5
            #b_sCOSm = 5.3 #obtained from model Enpei, 4.9 in Wingate et al (see Sunpaper for reference), range of values in Clapp and Hornberger
            b_sCOSm = self.b_sCOSm
            diffus_nodes = Dm * (wsat - self.s_moist)**2 * ((wsat - self.s_moist)/wsat)**(3./b_sCOSm) * (self.T_nodes/self.SunTref)**n
            D_a = Dm * (airtemp/self.SunTref)**n #diffusivity of air,check
            diffus[0] = 2. / (1./diffus_nodes[0] + 1./D_a) #eq8
            for i in range(1,self.nr_nodes): #diffusion at upper edge of grid cell
                diffus[i] = (diffus_nodes[i] + diffus_nodes[i-1])/2. #text below eq6
            if self.mothermodel.checkpoint:
                if self.call_from_init:
                    self.mothermodel.cpx_init[0]['rsCm_Dm_end'] = cp.deepcopy(Dm)
                    self.mothermodel.cpx_init[0]['rsCm_n_end'] = n
                    self.mothermodel.cpx_init[0]['rsCm_D_a_end'] = D_a
                    self.mothermodel.cpx_init[0]['rsCm_b_sCOSm_end'] = b_sCOSm
                else:
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_Dm_end'] = Dm
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_n_end'] = n
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_D_a_end'] = D_a
                    self.mothermodel.cpx[self.mothermodel.t]['rsCm_b_sCOSm_end'] = b_sCOSm
        elif self.Diffus_type == ('Ogee'):
            D0a = 1.27e-5*(self.T_nodes/self.SunTref)**1.5 * (101325 / pressure) #pressure in Pa, eq p5 Ogee, not fully sure if T0 is indeed 25 as assumed here
            D0lT0=1.94e-9/((298.15/216-1)**2) #using the info at the bottom of p5 and top p6
            D0l = D0lT0*(self.T_nodes/216-1)**2 #eq p 5 Ogee
            if self.ta_type == 'Pen40':
                ta = 0.66 #table 1 Ogee
            elif self.ta_type == 'Mol03r':
                ta = ((wsat-self.s_moist)**(3./2))/wsat #p5 Ogee, what Enpei uses
            elif self.ta_type == 'Mol03u':
                ta = ((wsat-self.s_moist)**(1.+3./self.b_pore))/(wsat**(3./self.b_pore))
            elif self.ta_type == 'Deepa11':
                ta = (0.2*((wsat-self.s_moist)/wsat)**(2.0)+0.004)/wsat
            else:
                raise Exception ('Error in ta_type switch inputdata')
            if self.tl_type == 'Pen40':
                tl = 0.66
            elif self.tl_type == 'Enpei':
                tl = (self.s_moist**(3./2))/wsat #THIS IS WRONG##
            elif (self.tl_type == 'Mol03r' or self.tl_type == 'Mol03u'):
                tl = (self.s_moist**(self.b_pore/3.0)) / (wsat**(self.b_pore/3.0-1))
            else:
                raise Exception ('Error in tl_type switch inputdata')
            Deffa = D0a*ta*(wsat-self.s_moist)
            Deffl = D0l*tl*self.s_moist #Error in Enpei
            diffus_nodes = Deffa + Deffl * kH #eq 15 Ogee, with no advection
            diffus[0] = 2. / (1./diffus_nodes[0] + 1./D0a[0])
            for i in range(1,self.nr_nodes): #diffusion at upper edge of grid cell
                diffus[i] = (diffus_nodes[i] + diffus_nodes[i-1])/2. #text below eq6 Sun
        else:
            raise Exception('Error in Diffus_type switch inputdata')
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_diffus_nodes_end'] = cp.deepcopy(diffus_nodes)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_diffus_nodes_end'] = cp.deepcopy(diffus_nodes)
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.calcD).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return diffus
    
    def calcG(self,diffus): #see eq 21,22 and 23
        conduct = np.zeros(self.nr_nodes)
        for i in range(1,self.nr_nodes):
            conduct[i] = diffus[i] / (self.z_soil[i] - self.z_soil[i-1])
        conduct[0] = diffus[0] / (self.z_soil[0] - 0)
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_conduct_end'] = cp.deepcopy(conduct)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_conduct_end'] = cp.deepcopy(conduct)
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.calcG).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return conduct
    def calcJ(self,conduct,C_soilair,C_air): #see eq 6 and 7
        if self.mothermodel.checkpoint:
            if self.call_from_init:
                self.mothermodel.cpx_init[0]['rsCm_C_soilair_calcJ'] = cp.deepcopy(C_soilair)
            else:
                self.mothermodel.cpx[self.mothermodel.t]['rsCm_C_soilair_calcJ'] = cp.deepcopy(C_soilair)
        OCS_fluxes = np.zeros(self.nr_nodes)
        for i in range(1,self.nr_nodes):
           OCS_fluxes[i] = -1. * conduct[i] * (C_soilair[i] - C_soilair[i-1])
        OCS_fluxes[0] = -1. * conduct[0] * (C_soilair[0] - C_air) #negative is emission here
        if self.mothermodel.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.calcJ).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                    self.mothermodel.vars_rsCm.update({variablename: the_locals[variablename]})
        return OCS_fluxes
