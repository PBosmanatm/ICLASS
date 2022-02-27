import numpy as np
import copy as cp
import sys
import time
import inspect


#for the canopy model we take the same system as for the soil model (see Sun et al. 2015)
#the nodes now start from near the soil surface upward towards the top of the canopy.

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86)) #Tetens formula in Pa, see intro atm

def qsat(T,p):
    return 0.622 * esat(T) / p

class canopy_mod:

    def __init__(self,inputdata=None,mothermodel=None):
        self.stomblock = 0.
        self.diffrb_COS = 1.56 #Stimler et al 2010
        self.diffrb_CO2 = 1.37 #1.2 see line 2007 messy_emdep_xtsurf.f90 in Laurens' model, 1.37 from Jacobs 1994
        self.avo = 6.022045E23 # Avogadro constant [1/mol]
        self.rhow = 1000. #density of water
        self.use_ags_leaf = True #no alternative implemented
        if mothermodel != None: #mothermodel is class model calling the canopy model
            self.mothermodel = mothermodel
            self.nveglay = inputdata.nr_nodes_veg
            self.rho = mothermodel.rho
            self.hc = inputdata.hc
            self.mair = mothermodel.mair
            self.mcos = mothermodel.mcos
            self.mh2o = 18.
            self.z_veglay = inputdata.z_veglay
            self.gliCOS = inputdata.gliCOS
            self.lad_veglay = inputdata.lad_veglay
            self.cp_const = mothermodel.cp
            self.Lv = mothermodel.Lv
            if self.use_ags_leaf :
                self.c3c4 = mothermodel.c3c4
                self.CO2comp298 = mothermodel.CO2comp298 
                self.Q10CO2 = mothermodel.Q10CO2
                self.gm298 = mothermodel.gm298
                self.Q10gm = mothermodel.Q10gm
                self.T1gm = mothermodel.T1gm
                self.T2gm = mothermodel.T2gm
                self.gmin = mothermodel.gmin
                self.nuco2q = mothermodel.nuco2q
                self.f0 = mothermodel.f0
                self.ad = mothermodel.ad
                self.mco2 = mothermodel.mco2
                self.Ammax298 = mothermodel.Ammax298
                self.Q10Am = mothermodel.Q10Am
                self.T1Am = mothermodel.T1Am
                self.T2Am = mothermodel.T2Am
                self.wwilt = mothermodel.wwilt
                self.wfc = mothermodel.wfc
                self.c_beta = mothermodel.c_beta
                self.alpha0 = mothermodel.alpha0
                self.Kx = mothermodel.Kx
                self.alfa_sto = inputdata.alfa_sto
            self.rws_COS_leaf = inputdata.rws_COS_leaf #wet skin resistance (resistance for water layer)
            self.rws_CO2_leaf = inputdata.rws_CO2_leaf
            self.C_CO2_veglayer_current = np.array(inputdata.C_CO2_veglayer_init_ppm) * 1.e-6 / self.mair * self.rho * 1000 # conversion mumol mol-1 (ppm) to mol_CO2 m-3 : mumol_CO2 / mol_air * mol_CO2/mumol_co2 * mol_air/g_air * kg_air / m3 * g_air/kg_air
            self.C_COS_veglayer_current = np.array(inputdata.C_COS_veglayer_init_ppb) * 1.e-9 / self.mair * self.rho * 1000# conversion nmol mol-1 (ppb) to mol_COS m-3 
            self.C_H2O_veglayer_current = np.array(inputdata.C_H2O_veglayer_init_pct) * 1.e-2 / self.mair * self.rho * 1000# conversion cmol mol-1 (%) to mol_H2O m-3 
            self.calcWindspTop = False #if False, taken from CLASS
            if hasattr(inputdata,'calcWindspTop'): #There should not be a line self.calcWindspTop = None to class model_input in forwardmodel.py!!
                self.calcWindspTop = inputdata.calcWindspTop
            self.lad_dependence_u = True #to make the extinction of the exponential wind profile dependent on leaf area density
            if hasattr(inputdata,'lad_dependence_u'):
                self.lad_dependence_u = inputdata.lad_dependence_u
            if hasattr(inputdata,'ladi'):
                self.ladi = inputdata.ladi #leaf angle distribution index, see eq C3 Launianen et al, only used when using sun shaded radiation model
            else:
                self.ladi = 1.0
            self.frac_green = 1.0
            if hasattr(inputdata,'frac_green'):
                self.frac_green = inputdata.frac_green #The fraction of the absorbed radiation that is absorbed by green leaves, the rest of the fraction are branches, dead leaves etc. 
            self.clumpf = 1.0 #see Launianen 2011, Stenberg 1998, it is a factor to account for grouping of needles around shoots.
            if hasattr(inputdata,'clumpf'):
                self.clumpf = inputdata.clumpf
            self.U_mode = inputdata.U_mode
            self.K_scale = 1.0 #a scaling factor for all the K-values (exchange coefficients)
            if hasattr(inputdata,'K_scale'):
                self.K_scale = inputdata.K_scale
            self.K_mode = inputdata.K_mode
            if self.K_mode == 'prescribed':
                self.K = np.zeros(self.nveglay+1) #also 1 exchange coeff between veg and surf layer
                for lay in range(1,self.nveglay+1):
                    self.K[lay] = inputdata.__dict__['Kveg'+lay] #inputdata for K should be in the form Kveg1 = .., Kveg1=..., ....
            self.U_ext_scale = 1.0 #a scaling factor for all the alpha-values in the wind formula (wind extiction coefficients)
            if hasattr(inputdata,'U_ext_scale'):
                self.U_ext_scale = inputdata.U_ext_scale
            self.calc_sun_shad = inputdata.calc_sun_shad #This is a switch to determine wether to calculate sunlit and shaded leaves differently
            if self.calc_sun_shad:
                self.Kdif = inputdata.Kdif #extinction ceofficient diffuse radiation
                self.prescr_fPARdif = inputdata.prescr_fPARdif #fraction of diffuse PAR calculated (False) or prescribed (true)
            self.incl_H2O = True #include H2O, or just COS and CO2
            if hasattr(inputdata,'incl_H2O'):
                self.incl_H2O = inputdata.incl_H2O
            if self.incl_H2O:
                self.Wl = inputdata.Wl
                self.Wmax = inputdata.Wmax
                self.wetfveg = np.zeros(self.nveglay) #wet fraction of vegetation
                self.Ts_mode = 'CLASS_en_bal'
                if hasattr(inputdata,'Ts_mode'):
                    self.Ts_mode = inputdata.Ts_mode
        else:
            #if runned stand alone, adapt values here
            self.nveglay = 6
            self.rho = 1.2
            self.hc = 16
            self.mair = 28.97
            self.mcos = 12. + 16. + 32.07
            self.z_veglay = np.linspace(1,self.hc-0.25,self.nveglay)
            self.gliCOS = 0.2 /(self.rho*1000) * self.mair #m s-1
            self.lad_veglay = np.array([0.4,0.8,1.0,1.0,1.4,0.9])
            self.tod = 8*3600.
            self.wetfveg = np.zeros(self.nveglay)  #wet fraction of vegetation
            self.calc_sun_shad = False
            if self.calc_sun_shad:
                self.lat = 51.
                self.lon = 0.
                self.doy = 173.
                self.ladi = 1.
                self.Kdif = 0.7
                self.fPARdif = 0.3
            if self.use_ags_leaf :
                self.c3c4 = 'c3'
                self.CO2comp298 =  [68.5,    4.3    ]
                self.Q10CO2 = [1.5,     1.5    ]
                self.gm298 = [7.0,     17.5   ]
                self.Q10gm = [2.0,     2.0    ]
                self.T1gm = [278.,    286.   ]
                self.T2gm = [301.,    309.   ]
                self.gmin = [0.25e-3, 0.25e-3]
                self.nuco2q = 1.6
                self.f0 = [0.89,    0.85   ]
                self.ad = [0.07,    0.15   ]
                self.mco2 = 44.
                self.Ammax298 = [2.2,     1.7    ]
                self.Q10Am = [2.0,     2.0    ]
                self.T1Am = [281.,    286.   ]
                self.T2Am = [311.,    311.   ]
                self.wwilt = 0.171
                self.wfc = 0.323
                self.c_beta = 0
                self.alpha0 = [0.017,   0.014  ]
                self.Kx = [0.7,     0.7    ]
                self.alfa_sto = 1.0
            self.rws_COS_leaf = 1.e5
            self.rws_CO2_leaf = 1.e5
            self.C_CO2_veglayer_current = np.array([400,400,400,400,400,400]) * 1.e-6 / self.mair * self.rho * 1000 # conversion mumol mol-1 (ppm) to mol_CO2 m-3 : mumol_CO2 / mol_air * mol_CO2/mumol_co2 * mol_air/g_air * kg_air / m3 * g_air/kg_air
            self.C_COS_veglayer_current = np.array([0.500,0.500,0.500,0.500,0.500,0.500]) * 1.e-9 / self.mair * self.rho * 1000# conversion nmol mol-1 (ppb) to mol_COS m-3 
            self.C_H2O_veglayer_current = np.array([1.0,1.0,1.0,1.0,1.0,1.0]) * 1.e-2 / self.mair * self.rho * 1000# conversion cmol mol-1 (%) to mol_H2O m-3 
            self.calcWindspTop = False
            self.lad_dependence_u = True
            self.K_scale = 1.0 #a scaling factor for all the K-values (exchange coefficients)
            self.U_mode = inputdata.U_mode
            self.U_ext_scale = 1.0 #a scaling factor for all the alpha-values in the wind formula (wind extiction coefficients)
            self.K_mode = 'Launianen'
            if self.K_mode == 'prescribed':
                self.K = np.array([0.05,0.05,0.05,0.05,0.05,0.05,0.05])
        self.dz_veglay = np.zeros(self.nveglay) #the size of the gridcell
        self.dz_veglay[0] = (self.z_veglay[0]+self.z_veglay[1])/2 #z = 0 at the soil surface
        for i in range(1, self.nveglay-1):
            self.dz_veglay[i] = (self.z_veglay[i+1]-self.z_veglay[i-1])/2
        self.dz_veglay[self.nveglay-1] = (self.hc - self.z_veglay[self.nveglay-1]) + (self.z_veglay[self.nveglay-1]-self.z_veglay[self.nveglay-2]) / 2. #
        self.z_int_veg = np.zeros(self.nveglay+1) #depth of the interfaces, self.z_int_veg[i] is at the lower edge of grid cell i, one interface added at the top of canopy
        self.z_int_veg[0] = 0.
        for i in range(1, self.nveglay):
            self.z_int_veg[i] = self.z_veglay[i] - (self.z_veglay[i]-self.z_veglay[i-1])/2 
        self.z_int_veg[self.nveglay] = self.hc
        if self.hc < 0.20:
            raise Exception('Cannot use canopy model for such low canopies')
        self.LAI_veglay = np.zeros(self.nveglay)
        for lay in range(self.nveglay):
            self.LAI_veglay[lay] = self.lad_veglay[lay] * self.dz_veglay[lay]
        self.lai_total = np.sum(self.LAI_veglay)
        if self.lai_total < 2.0:
            print('reflection from soil neglected, although LAI < 2 !!') #Xabi suppl p4
        
    
    def calcPAR(self,Swin,layer_ind): #layer_ind is the index of the layer we are looking at
        if(self.c3c4 == 'c3'):
            c = 0
        elif(self.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('option \"%s\" for \"c3c4\" invalid'%self.c3c4)
        PAR_top = 0.5 * Swin
        LAI_above = 0 #the LAI that is above a certain height
        for lay in range(self.nveglay-1,layer_ind,-1): #layer_ind not part of this range
            LAI_above += self.dz_veglay[lay] * self.lad_veglay[lay]
        if layer_ind == self.nveglay-1: #the range above will be empty in that case
            LAI_above += (self.hc - self.z_veglay[layer_ind]) * self.lad_veglay[layer_ind] #upper part of current layer also contributes to LAI above
        else:    
            LAI_above += (self.z_veglay[layer_ind+1] - self.z_veglay[layer_ind]) / 2. * self.lad_veglay[layer_ind] #upper part of current layer also contributes to LAI above
        PAR = PAR_top * np.exp(-self.Kx[c] * LAI_above)
        return PAR
    
    def calcPARsun_shad(self,Swin,layer_ind,fPARdif=None): #layer_ind is the index of the layer we are looking at
        #p 7 book Hikosaka is very informative for general extinction modelling principle
        if hasattr(self,'mothermodel'):
            sea = np.arcsin(self.mothermodel.sinlea) #solar elevation angle
            sza = np.pi / 2. - sea #solar zenith angle, CLASS calculates the sine of the elevation angle, which is 90° (pi/2 rad) minus the solar zenith angle 
        else:
            sda    = 0.409 * np.cos(2. * np.pi * (self.doy - 173.) / 365.)
            sinlea = np.sin(2. * np.pi * self.lat / 360.) * np.sin(sda) - np.cos(2. * np.pi * self.lat / 360.) * np.cos(sda) * np.cos(2. * np.pi * (self.tod) / 86400. + 2. * np.pi * self.lon / 360.)
            sea = np.arcsin(sinlea) #solar elevation angle
            sza = np.pi / 2. - sea
        sca_coeff = 0.2 #0.2 from p51 and p 38 photosynthesis book. This includes reflection!!
        #Kdir = (self.ladi**2 + (np.tan(sza)))**0.5 / (self.ladi + 1.774 * (self.ladi + 1.182)**-0.733) #Launianen 2011, this is the extinction of direct radiation, when scattered and transformed into diffuse radiation it is considered lost for the direct beam (we know since Launinanen calculates the fraction of sunlit leaves with this). 
        Kdir = 0.5 / np.sin(sea) #eq 7 Spitters 1986, an alternative for the equation above. Note that Kdir would equal 1 if the sun was at 90 deg elev angle and the leaves horizontal (see p7 and p 8 book Hikosaka, neglect branches) 
        Kdirsca = Kdir * np.sqrt(1 - sca_coeff) #Xabi suppl eq 3 and Spitter 1986 eq 4, consistent with eq C6 Launianen et al. This is the rate with which direct radiation goes extinct, but now slower than the rate above, since scattering of direct radiation into diffuse is now kept.
        #self.Kdif = 0.8 * np.sqrt(1 - sca_coeff) #Spitters 1986 eq 7,
        LAI_above = 0 #the LAI that is above a certain height
        for lay in range(self.nveglay-1,layer_ind,-1): #layer_ind not part of this range
            LAI_above += self.dz_veglay[lay] * self.lad_veglay[lay] / self.frac_green#divide by frac_green to also include extinction by dead leaves, branches etc. This assumes that the lad that is given in the input file is only the green LAI
        if layer_ind == self.nveglay-1: #the range above will be empty in that case
            LAI_above += (self.hc - self.z_veglay[layer_ind]) * self.lad_veglay[layer_ind] / self.frac_green#upper part of current layer also contributes to LAI above
        else:    
            LAI_above += (self.z_veglay[layer_ind+1] - self.z_veglay[layer_ind]) / 2. * self.lad_veglay[layer_ind] / self.frac_green#upper part of current layer also contributes to LAI above
        tau_dir = np.exp(-Kdir * LAI_above * self.clumpf) #This is the fraction of sunlit leaves (p51 Hikosaka, Launianen et al 2011 p16, Xabi suppl p 5)
        tau_dif = np.exp(-self.Kdif * LAI_above * self.clumpf)#see Launianen 2011, Stenberg 1998: clumpf is a factor to account for grouping of needles around shoots. eq is exp part of eq3 Spitters 1986
        tau_sca = np.exp(-Kdirsca * LAI_above * self.clumpf)#this is the exponent term from eq 4 Spitters 1986, it is the fraction of direct + secondary diffuse radiation that is left at a level
#        leaf_abs = 0.8
#        tau_sca = np.exp(-np.sqrt(leaf_abs) * Kdir * LAI_above * self.clumpf) eq Launianen 2011
        if not self.prescr_fPARdif: #Paper Weiss & Norman 1985, implementation partly from MLC-CHEM (messy_emdep_xtsurf.f90, line 2263)
            OT = 35./(1224.*np.cos(sza)**2.+1)**.5 #from MLC-CHEM, different from paper Weiss and Norman
            #OT = 1. / np.cos(sza) #original Weiss and Norman paper eq 2
            RDV = 600 * np.exp(-0.185 * OT ) * np.cos(sza) # P/P0 term neglected, close to 1
            #RdV = 0.4 * (600 - RDV) * np.cos(sza) #original eq 3 Weiss and Norman 1985
            RdV = 0.4 * (600 * np.cos(sza) - RDV) #What I think it should be, with a correction to eq 3 of the paper of Weiss and Norman 1985
            RV = RdV + RDV #
            w_infr = 1320 * 10**(-1.1950 + 0.4459 * np.log10(OT) - 0.0345 * (np.log10(OT))**2) #different from MLC-CHEM
            #w_infr= 1320*.077*(2.*OT)**0.3 #MLC-CHEM
            RDN = (720*np.exp(-0.06 * OT) - w_infr) * np.cos(sza) # 
            #RdN = 0.6 * (720 - RDN - w_infr) * np.cos(sza) #original eq 5 Weiss and Norman 1985
            RdN = 0.6 * (720 * np.cos(sza) - RDN - w_infr * np.cos(sza)) #What I think it should be, with a correction to eq 5 of the paper of Weiss and Norman 1985
            RN = RdN + RDN
            RATIO = Swin/(RV+RN)
            if RATIO > 0.9: #text p5 paper, we do not account for the 0.88 (C), since eq 12 of the paper is not used
                RATIO = 0.9
            self.fPARdif = 1 - (RDV / RV * (1 - ((0.9 - RATIO) / 7.0)**(2/3))) #eq 11 and 1 - direct fraction is diffuse fraction. 7.0 instead of 0.7 in MLC-CHEM!!
        else:
            self.fPARdif = fPARdif
        PAR_dir_top = 0.5 * Swin * (1 - self.fPARdif) #W/m2 horizontal surface, 50% as in ClASS
        PAR_dif_top = 0.5 * Swin * self.fPARdif
        #PAR_sun = (1 - sca_coeff) * PAR_dir_top * tau_dir + sca_coeff * tau_sca * PAR_dir_top + (1 - sca_coeff) * PAR_dif_top * tau_dif + sca_coeff * PAR_dif_top * tau_sca
        #PAR_sha = (1 - sca_coeff) * PAR_dif_top * tau_dif + sca_coeff * tau_sca * PAR_dir_top + sca_coeff * PAR_dif_top * tau_sca
        refl_dif = (1 - np.sqrt(1 - sca_coeff)) / (1 + np.sqrt(1 - sca_coeff)) #reflection coefficient for diffuse radiation, Xabi supplement eq 6 , photosynth book eq 1.21
        #Note that the reflection coefficients are not those of a leaf, but they are valid at each level of the canopy (see p11(44 in pdf) photosynthesis book). reflection coefficient is defined as upward flux/downward flux. 
        #Keep in mind that this is something different as the fraction of the downward flux that is reflected upward locally, which explains why the amount of leaf are at a certain level is not in the equation, although this matters a lot for how much radiation is reflected upward locally!
        refl_dir = refl_dif * 2 / (1 + 1.6 * np.sin(sea)) #eq 1 Spitters 1986
        PAR_sha = PAR_dif_top * tau_dif + (PAR_dir_top * tau_sca - PAR_dir_top * tau_dir) #W/m2 leaf in the shadow (or m2 surface area, does not matter in this case), see Xabi suppl eq 21 and 14, the second term is the diffuse radiation that originates from the scattering of the direct beam
        PAR_sun = PAR_dir_top + PAR_sha #W/m2 ,keep in mind that this is per square meter of horizontally oriented sunlit leaf, direct radiation is the same at the bottom as in the top of the canopy
        #the next eq is eq 20 Xabi suppl and Spitters 86 eq 12 (except for frac_green and clumpf), see notes canopy model.  
        PAR_dirdir_abs = self.frac_green * Kdir * self.clumpf * (1 - sca_coeff) * tau_dir * PAR_dir_top * 1 / tau_dir # The direct PAR that is absorbed per m2 of sunlit photosynthesizing leaves, excluding direct radiation that is scattered and turned into diffuse. 1 / tau_dir is fraction of sunlit leaves. See notes canopy model, similar to eq 1.33 photosynthesis book. frac_green since not all absorption by photosynthesizing leaves
        PAR_dif_abs = self.frac_green * (1 - refl_dif) * PAR_dif_top * self.Kdif * self.clumpf * tau_dif#absorbtion of primary diffuse radiation per m2 photosynthesizing leaf (W/m2_leaf, not per m2 surface area, since we take the derivative to LAI_above). Same principle as above, except that we need to look at the change in net flux with LAI now (and per m2 of leaf instead of sunlit).
        # the above does not include the direct tradiation that is scattered into diffuse. approx eq 10 Spitters 1986
        PAR_dir_abs = self.frac_green * (1 - refl_dir) * PAR_dir_top * Kdirsca * self.clumpf * tau_sca #W / m2_leaf, eq 11 Spitters 1986, except for clumpf, frac_green
        PAR_dif2_abs = PAR_dir_abs - PAR_dirdir_abs * tau_dir #times tau_dir, since PAR_dirdir_abs was per m2 of sunlit leaf area, tau_dir has units m2_sunlit leaf / m2_leaf (see notes 2 canopy model). together with next eq, forms eq 21 Xabi suppl, and eq 13 Spitters 1986.
        PAR_sha_abs = PAR_dif_abs + PAR_dif2_abs #per square meter of shaded leaf (or per m2 leaf in general, since the sunlit will also absorb this)
        PAR_sun_abs = PAR_sha_abs + PAR_dirdir_abs #per square meter of sunlit leaf. eq 14 Spitters 1986
        return PAR_sun_abs,PAR_sha_abs,PAR_sun,PAR_sha,tau_dir
    
    def calcU_HF07(self): #paper Harmann and Finnigan 2007
        U_veg = np.zeros(self.nveglay)
        mix_l = 2 * self.beta_U**3 * self.Lc #eq 6
        U_top_of_can = self.ustar / self.beta_U #eq 5
        for lay in range(self.nveglay):
            U_veg[lay] = U_top_of_can * np.exp(self.beta_U * self.z_veglay[lay] / mix_l) #eq 3
    
    def calcU_Cionco(self,windsp_surf_lay,lai_total,z0m=None,disp=None,ustar=None): #line 4300 messy_emdep_xtsurf.f90 Laurens' model
        #note that this function calculates wind speed, not just zonal wind speed
        U_veg = np.zeros(self.nveglay)
        if lai_total > 1.e-10:
            ptype=2
            if self.hc < 2.5:#than we assume it is not a forest
                ptype = 4 #forest (ptype = 2) or not forest (ptype = 4)
            beta = 1.-(ptype-1.)*0.25
            if (self.hc > 10.):
                alpha = lai_total
                if (alpha > 4.0):
                    alpha = 4.0
            else:
                alpha = 0.65*lai_total
                if (alpha > 3.0):
                    alpha = 3.0
            if self.calcWindspTop:
                d0 = self.hc*(0.05+0.5*lai_total**0.20 + (ptype-1.)*0.05)
                if (lai_total < 1.):
                    z0 = self.hc*0.1
                else:
                    z0 = self.hc*(0.23 - 0.1*lai_total**0.25 - (ptype-1.)*0.015)
                if (disp > 0.):
                    d0=disp
                if (z0m > 0.):
                    z0=z0m
                U_top_of_can = 2.5*ustar*np.log((self.hc - d0 + z0) / z0)
            else:
                U_top_of_can = windsp_surf_lay
            if self.lad_dependence_u:
                mean_lad = np.mean(self.lad_veglay)
                #approximating the same function as Laurens has, but with a height(lad) dependent alpha seems not so easy, especially given the rather comlex function in the exp brackets. Therefore we assume a simpler function, du/d(hc - z) = -alfa, i.e. u = u0 * exp(-alfa * (hc - z)). 
                #Than, if we set e.g. u0 = 10 and alfa = 2 and define level zero to be at hc - z = 0, level 4 at hc -z = 3 and level 5 at hc -z = 7, than we get: u4 = 10exp(-2*3); u5 = 10exp(-2*7), and thus u5 = u4 * exp (-2*4) , which is equal to 
                #u4 * exp(- alfa * [(hc - z)_5 - (hc - z)_4]) = u4 * exp(- alfa * [z_4 - z_5]) since hc(canopy height) is a constant. Also in writing u5 = u4 * exp(- alfa * [(hc - z)_5 - (hc - z)_4]) we assumed alpha independent of lad. 
                #Generalising, u_i = u_i-1 * exp(- alfa * [z_i - z_i-1]) where we should keep in mind that in this notation z_i is lower in the canopy than z_i-1
                #Now we scale alpha with a lad dependent factor (we can see alpha as a height dependent alpha), and keeping in mind that z is positive upward, the equation becomes: (the upper layer is a bit special)
#                U_veg[-1] = U_top_of_can * np.exp(- alpha * self.lad_veglay[-1] / mean_lad * (self.z_veglay[-1] - self.hc))
#                for lay in range(self.nveglay-2,-1,-1):
#                    U_veg[lay] = U_veg[lay+1] * np.exp(- alpha * (self.lad_veglay[lay]+self.lad_veglay[lay+1]) / 2 / mean_lad * (self.z_veglay[lay] - self.z_veglay[lay+1]))
                #Note that this isn't really an analytical solution anymore, since due to the lad dependent scaling factor, u5 = u0 * exp(-alfa*(hc - z_5)) is not true anymore. 
                #And taking the alpha terms together as we did is something that is strictly not allowed if alpha is really height dependent...
                #We can instead of the above also use the equation from Laurens:
                #u5 = u0 * np.exp(-alpha*((1. - z_5 / hc)**beta)); u4 = u0 * np.exp(-alpha*((1. - z_4 / hc)**beta))
                #than u5 = u4 * np.exp(-alpha*((1. - z_5 / hc)**beta) + alpha*((1. - z_4 / hc)**beta))
                #-> u5 = u4 * np.exp(-alpha*((1. - z_5 / hc)**beta - ((1. - z_4 / hc)**beta))) where z_5 is at a lower height than z_4
                #now scaling alpha with a lad dependent factor (and a general scaling factor U_ext_scale):
                U_veg[-1] = U_top_of_can * np.exp(-alpha * self.U_ext_scale * (self.lad_veglay[-1] / mean_lad) * ((1. - self.z_veglay[-1] / self.hc)**beta))
                for lay in range(self.nveglay-2,-1,-1):
                    U_veg[lay] = U_veg[lay+1] * np.exp(-alpha * self.U_ext_scale * ((self.lad_veglay[lay]+self.lad_veglay[lay+1]) / 2 / mean_lad) * ((1. - self.z_veglay[lay] / self.hc)**beta - ((1. - self.z_veglay[lay+1] / self.hc)**beta)))
                #If instead we would not take the alpha terms together, we get u5 = u0 * exp(-alpha * (1 - z_5/hc)^beta) and u4 = u0 * exp(-alpha * (1 - z_4/hc)^beta). Thus, u5 = u4 * exp(-alpha * (1 - z_5/hc)^beta + alpha * (1 - z_4/hc)^beta)
                #Now, we get complicated things: 
#                U_veg[-1] = U_top_of_can * np.exp(-alpha*(self.lad_veglay[-1]/mean_lad) * (1. - self.z_veglay[-1] / self.hc)**beta)
#                U_veg[-2] = U_veg[-1] * np.exp(-alpha*((self.lad_veglay[-2]+self.lad_veglay[-1]) / 2 / mean_lad) * (1. - self.z_veglay[-2] / (self.hc))**beta + alpha*(self.lad_veglay[-1] / mean_lad) * (1. - self.z_veglay[-1] / (self.hc))**beta)
#                for lay in range(self.nveglay-3,-1,-1):
#                    U_veg[lay] = U_veg[lay+1] * np.exp(-alpha*((self.lad_veglay[lay]+self.lad_veglay[lay+1]) / 2 / mean_lad) * (1. - self.z_veglay[lay] / (self.hc))**beta + alpha*((self.lad_veglay[lay+1]+self.lad_veglay[lay+2]) / 2 / mean_lad) * (1. - self.z_veglay[lay+1] / (self.hc))**beta)
     
            else:
                for lay in range(0,self.nveglay):
                    #small difference with Laurens model for lowest vegetation layer: term between brackets of exponent becomes -alpha in my case, -alpha*((1 - 1/nveglay)**beta) in Laurens' case
                    #note that in Laurens model i=1 means the highest vegetation layer, we let define layer indexes to increase with height
                    U_veg[lay] = U_top_of_can*np.exp(-alpha * self.U_ext_scale *((1. - self.z_veglay[lay] / self.hc)**beta)) #if z == top of canopy, we get np.exp(0)
                    #working with layers instead of height: U_veg[lay] = U_top_of_can*np.exp(-alpha*((1. - lay / (self.nveglay-1))**beta)) #if lay == self.nveglay-1 (top of canopy), we get np.exp(0)
        else:
            raise Exception('Cannot calculate u for such low LAI')
        return U_veg

    def ags_leaf(self,thetasurf,Ts,e,C_co2,w2,rb_co2,Swin,layer_ind,fPARdif=None):
        #first lines only if not run stand alone
        if hasattr(self,'mothermodel'): #note that self.mothermodel only assigned if mothermodel is not None
            if self.mothermodel.checkpoint:
                if self.call_from_init:
                    self.mothermodel.cpx_init[0]['rcm_thetasurf'] = thetasurf
                else:
                    self.mothermodel.cpx[self.mothermodel.t]['rcm_thetasurf'] = thetasurf
        # Select index for plant type
        if(self.c3c4 == 'c3'):
            c = 0
        elif(self.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('option \"%s\" for \"c3c4\" invalid'%self.c3c4)
        
        # calculate CO2 compensation concentration
        CO2comp       = self.CO2comp298[c] * self.rho * pow(self.Q10CO2[c],(0.1 * (thetasurf - 298.)))  

        # calculate mesophyll conductance
        gm1 =  self.gm298[c] *  pow(self.Q10gm[c],(0.1 * (thetasurf-298.))) 
        gm2 =  1. + np.exp(0.3 * (self.T1gm[c] - thetasurf))
        gm3 =  1. + np.exp(0.3 * (thetasurf - self.T2gm[c]))
        gm            = gm1 / ( gm2 * gm3)
        gm            = gm / 1000. # conversion from mm s-1 to m s-1
  
        # calculate CO2 concentration inside the leaf (ci)
        fmin0         = self.gmin[c] / self.nuco2q - 1. / 9. * gm
        sqrtf         = pow(fmin0,2.) + 4 * self.gmin[c]/self.nuco2q * gm
        sqterm        = pow(sqrtf,0.5)
        
        fmin          = (-fmin0 + sqterm) / (2. * gm) #Minimum value of cfrac, calculated differently as Jacobs 1994 eq 3.24, this is eq A9 from Ronda et al 2001.
        #fmin = (self.gmin[c] / self.nuco2q) / (self.gmin[c] / self.nuco2q + gm) #eq 3.24 Jacobs 1994
  
        Ds            = (esat(Ts) - e) / 1000. # kPa #If this gets very negative we have a problem...
        if Ds < 0:
            Ds = 0
            print('Ds set to zero')
        D0            = (self.f0[c] - fmin) / self.ad[c] #eq 13 Ronda et al 2001, value of Ds at which the stomata close

        cfrac         = self.f0[c] * (1. - (Ds / D0)) + fmin * (Ds / D0) #eq 3.25 Jacobs 1994
        #cfrac = self.f0[c] - self.ad[c] * Ds#eq 2.11 Steeneveld 2002
        co2abs = C_co2 * self.mco2 * 1.e3 #conversion mol/m3 to mg/m3
        ci = cfrac * (co2abs - CO2comp) + CO2comp #eq 3.21 Jacobs, cfrac is f
        self.ci_co2[layer_ind] = ci
#        print('C_co2')
#        print(C_co2)
  
        # calculate maximal gross primary production in high light conditions (Ag)
        Ammax1        = self.Ammax298[c] *  pow(self.Q10Am[c],(0.1 * (thetasurf - 298.))) 
        Ammax2        = 1. + np.exp(0.3 * (self.T1Am[c] - thetasurf))
        Ammax3        = 1. + np.exp(0.3 * (thetasurf - self.T2Am[c]))
        Ammax         = Ammax1 / ( Ammax2 * Ammax3)

        # calculate effect of soil moisture stress on gross assimilation rate
        betaw         = max(1e-3, min(1.,(w2 - self.wwilt)/(self.wfc - self.wwilt)))
  
        # calculate stress function
        if (self.c_beta == 0):
            fstr = betaw;
        else:
            # Following Combe et al (2016)
            if (self.c_beta < 0.25):
                P = 6.4 * self.c_beta
            elif (self.c_beta < 0.50):
                P = 7.6 * self.c_beta - 0.3
            else:
                P = 2**(3.66 * self.c_beta + 0.34) - 1
            fstr = (1. - np.exp(-P * betaw)) / (1 - np.exp(-P))
  
        # calculate gross assimilation rate (Am)
        aexp          = -(gm * (ci - CO2comp) / Ammax) #=gm/Ammax * (co2comp - ci)
        Am           = Ammax * (1. - np.exp(aexp)) #eq 3.13 Jacobs 1994
        Rdark        = (1. / 9.) * Am #as in Jacobs 1994
        AmRdark      = Am + Rdark
        xdiv         = co2abs + 2.*CO2comp
        # calculate  light use efficiency
        alphac       = self.alpha0[c] * (co2abs - CO2comp) / (xdiv) #eq 3.10 Jacobs 1994
        
        #the following are used later, but are not PAR dependent
        a1           = 1. / (1. - self.f0[c])
        a11          = a1 * (self.f0[c] - fmin)
        Dstar        = D0 / (a11)
        gcutco2_leaf = self.gmin[c] / self.nuco2q #cuticular conductance
        rcutco2_leaf = 1 / gcutco2_leaf

        if self.calc_sun_shad:
            PAR_sun_abs,PAR_sha_abs,PAR_sun,PAR_sha,fract_sun = self.calcPARsun_shad(Swin,layer_ind,fPARdif)
            self.PAR_sun_abs[layer_ind] = PAR_sun_abs
            self.PAR_sha_abs[layer_ind] = PAR_sha_abs
            self.PAR_sun[layer_ind] = PAR_sun
            self.PAR_sha[layer_ind] = PAR_sha
            self.PAR[layer_ind] = PAR_sun * fract_sun + PAR_sha * (1 - fract_sun) #Note that this is for a horizontal surface
            # calculate gross primary productivity for sunlit
            pexp_sun         = -1 * alphac * PAR_sun_abs / (AmRdark)  
            Ag_sun           = fstr * (AmRdark) * (1 - np.exp(pexp_sun)) #positive means uptake here. #eq 3.12 from Jacobs 1994, but no fstr there yet, and Rd at the end of formula not included here
            gsco2_leaf_sun = a1 * Ag_sun / ((co2abs - CO2comp) * (1 + Ds / Dstar)) #
            gsco2_leaf_sun = self.alfa_sto * gsco2_leaf_sun #scaling the conductance
            An_leaf_dry_sun    = -(co2abs - ci) / (rb_co2 + 1 / (2 * gcutco2_leaf + gsco2_leaf_sun)) #mgCO2 / m2 /s; cuticular and stomatal resistance in parallel
            An_leaf_wet_sun    = -(co2abs - ci) / (rb_co2 + 1 / (2 / (rcutco2_leaf + self.rws_CO2_leaf) + gsco2_leaf_sun * (1 - self.stomblock) + 1 / (self.rws_CO2_leaf + 1 / gsco2_leaf_sun) * self.stomblock)) #similar to line 828 Laurens model (messy_emdep_xtsurf.f90)
#            print('agsun')
#            print(Ag_sun)
            # now for shaded
            pexp_sha         = -1 * alphac * PAR_sha_abs / (AmRdark)
            Ag_sha           = fstr * (Am + Rdark) * (1 - np.exp(pexp_sha)) #positive means uptake here
            gsco2_leaf_sha = a1 * Ag_sha / ((co2abs - CO2comp) * (1 + Ds / Dstar)) #
            gsco2_leaf_sha = self.alfa_sto * gsco2_leaf_sha #scaling the conductance
            An_leaf_dry_sha    = -(co2abs - ci) / (rb_co2 + 1 / (2 * gcutco2_leaf + gsco2_leaf_sha)) #mgCO2 / m2 /s; cuticular and stomatal resistance in parallel
            An_leaf_wet_sha    = -(co2abs - ci) / (rb_co2 + 1 / (2 / (rcutco2_leaf + self.rws_CO2_leaf) + gsco2_leaf_sha * (1 - self.stomblock) + 1 / (self.rws_CO2_leaf + 1 / gsco2_leaf_sha) * self.stomblock)) #similar to line 828 Laurens model (messy_emdep_xtsurf.f90)
#            self.aapsun[layer_ind] = pexp_sun
#            self.aapsha[layer_ind] = pexp_sha
        else:
            PAR          = self.calcPAR(Swin,layer_ind)
            self.PAR[layer_ind] = PAR
            pexp         = -1 * alphac * PAR / (AmRdark)
            # calculate gross primary productivity
            Ag           = fstr * (Am + Rdark) * (1 - np.exp(pexp)) #positive means uptake here, eq 3.12 Jacobs except Rd missing
    
            gsco2_leaf = a1 * Ag / ((co2abs - CO2comp) * (1 + Ds / Dstar)) #
            gsco2_leaf = self.alfa_sto * gsco2_leaf #scaling the conductance
            An_leaf_dry    = -(co2abs - ci) / (rb_co2 + 1 / (2 * gcutco2_leaf + gsco2_leaf)) #mgCO2 / m2 /s; cuticular and stomatal resistance in parallel
    #        time.sleep(0.15)
            An_leaf_wet    = -(co2abs - ci) / (rb_co2 + 1 / (2 / (rcutco2_leaf + self.rws_CO2_leaf) + gsco2_leaf * (1 - self.stomblock) + 1 / (self.rws_CO2_leaf + 1 / gsco2_leaf) * self.stomblock)) #similar to line 828 Laurens model (messy_emdep_xtsurf.f90)
        
        
        if hasattr(self,'mothermodel'):
            if self.mothermodel.checkpoint:
                if self.call_from_init:
                    self.cpx_init[0]['rcm_gm_end'] = gm
                else:
                    self.cpx[self.mothermodel.t]['rcm_gm_end'] = gm
            if self.mothermodel.save_vars_indict:
                the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
                for variablename in the_locals: #note that the self variables are not included
                    if str(variablename) != 'self':
                        self.vars_rcm.update({variablename: the_locals[variablename]})
        if self.calc_sun_shad:
            return gsco2_leaf_sun,An_leaf_dry_sun,An_leaf_wet_sun,gsco2_leaf_sha,An_leaf_dry_sha,An_leaf_wet_sha,fract_sun,Ds
        else:
            return gsco2_leaf,An_leaf_dry,An_leaf_wet,Ds
        
    def H2O_exchange_leaf(self,rs_H2O_leaf,rb_H2O_leaf,C_H2O,surf_press,Ts):
        H2Osat_leaf = esat(Ts) / surf_press * self.rho / (self.mair/1000) #Pa to mol_h2o/m3; e/p is in mol_h2o/mol_air, mol_h2o/mol_air * mol_air/kg_air * kg_air/m3 = mol_h2o/m3
        H2O_exchange = (H2Osat_leaf - C_H2O) / (rs_H2O_leaf + rb_H2O_leaf)
        return H2O_exchange #mol/m2_leaf/s
    
    def H2O_liq_evap(self,ra,C_H2O,surf_press,Ts):
        #The use of Ts is actually not fully correct, the air in which it evaporates will have a different temp
        H2Osat_Ts = esat(Ts) / surf_press * self.rho / (self.mair/1000) #Pa to mol_h2o/m3; e/p is in mol_h2o/mol_air, mol_h2o/mol_air * mol_air/kg_air * kg_air/m3 = mol_h2o/m3
        H2O_exchange = (H2Osat_Ts - C_H2O) / (ra)
        return H2O_exchange #mol/m2_leaf/s
    
    def H2O_liq_cont(self,Wl0,H2Oliqflux,dt_can): #the liquid water content of the leaves, expressed in a water layer (m), just as in CLASS
        Wltend   = - H2Oliqflux * self.mh2o/1000 / self.rhow #m/s, as in CLASS: mol/m2/s * kg/mol * m3/kg
        Wl       = Wl0     + dt_can * Wltend
        if Wl < 0.:
            Wl = 0.
        return Wl
    
    def H2O_soil_evap(self,ra,rsoil,C_H2O,surf_press,Ts):
        #The use of Ts is actually not fully correct, the air in which it evaporates will have a different temp
        H2Osat_Ts = esat(Ts) / surf_press * self.rho / (self.mair/1000) #Pa to mol_h2o/m3; e/p is in mol_h2o/mol_air, mol_h2o/mol_air * mol_air/kg_air * kg_air/m3 = mol_h2o/m3
        H2O_exchange = (H2Osat_Ts - C_H2O) / (ra + rsoil)
        return H2O_exchange #mol/m2/s
        
    def calc_evapotrans(self,U_veg,surf_press,rs_H2O_leaf,Ts = None):
        self.H2Oplantflux = np.zeros(self.nveglay) #canopy scale
        self.H2Oliqflux = np.zeros(self.nveglay) # liquid water evaporation for every layer
        self.rbveg_H2O = np.zeros(self.nveglay)   
        if self.Ts_mode == 'CLASS_en_bal': #Energy balance CLASS, calculated below
            esatvar    = esat(self.mothermodel.theta)
            desatdT      = esatvar * (17.2694 / (self.mothermodel.theta - 35.86) - 17.2694 * (self.mothermodel.theta - 273.16) / (self.mothermodel.theta - 35.86)**2.)
            parts_LEveg_numerator_Ts = np.zeros(self.nveglay) #parts, since there is one part for every vegetation layer. Note that this has to be consistent with the H2Oplantflux defined below
            parts_LEveg_denominator_Ts = np.zeros(self.nveglay)
            parts_LEliq_numerator_Ts = np.zeros(self.nveglay)
            parts_LEliq_denominator_Ts = np.zeros(self.nveglay)
            for lay in range(self.nveglay):
                self.rbveg_H2O[lay]= 1.0 *180 * (0.07/np.max([1.e-10,U_veg[lay]]))**0.5
                #the full LEveg of the layer equals: self.LAI_veglay[lay] * self.mh2o / 1000 * self.Lv / (self.rbveg_H2O[lay] + self.rs_H2O_leaf[lay]) * (self.C_H2O_sat - self.C_H2O_veglayer_current + (self.desatdT * (Ts - self.mothermodel.theta) / surf_press * self.rho / (self.mair/1000)) ) #
                #The above in W/m2_ground: m2_leaf/m2_ground * g_h2o / mol_h2o * kg_h2o / g_h2o * J / kg_h2o * m / s * mol_h2o/m3 (and m / s is m3 / m2_leaf as we are dealing with vegetation transpiration)
                self.C_H2O_sat_theta = esatvar / surf_press * self.rho / (self.mair/1000)
                parts_LEveg_numerator_Ts[lay] = self.mh2o / 1000 * self.Lv * self.LAI_veglay[lay] * (1 - self.wetfveg[lay]) / (self.rbveg_H2O[lay] + self.rs_H2O_leaf[lay]) * ((desatdT * self.mothermodel.theta / surf_press * self.rho / (self.mair/1000))  - self.C_H2O_sat_theta + self.C_H2O_veglayer_current[lay]) #W/m2, * self.mh2o / 1000 * self.Lv to convert from molar flux to energy flux
                parts_LEveg_denominator_Ts[lay] = self.mh2o / 1000 * self.Lv * self.LAI_veglay[lay] * (1 - self.wetfveg[lay]) / (self.rbveg_H2O[lay] + self.rs_H2O_leaf[lay]) * (desatdT / surf_press * self.rho / (self.mair/1000) ) #W/(m2 / K)
                parts_LEliq_numerator_Ts[lay] = self.mh2o / 1000 * self.Lv * self.LAI_veglay[lay] * self.wetfveg[lay] / self.mothermodel.ra * (desatdT * self.mothermodel.theta / surf_press * self.rho / (self.mair/1000) - self.C_H2O_sat_theta + self.C_H2O_veglayer_current[lay])
                parts_LEliq_denominator_Ts[lay] = self.mh2o / 1000 * self.Lv * self.LAI_veglay[lay] * self.wetfveg[lay] / self.mothermodel.ra * desatdT / surf_press * self.rho / (self.mair/1000)
            p1_numerator_Ts = self.mothermodel.Q + self.rho * self.cp_const / self.mothermodel.ra * self.mothermodel.theta #it would actually make more sense to use thetasurf(but than you need different resistance as well, now it uses ra based on Cs, which is calculated for the top of the surf layer), but CLASS uses theta. This part stems from H in the energy balance
            pLEveg_numerator_Ts = np.sum(parts_LEveg_numerator_Ts)
            pLEsoil_numerator_Ts = self.mh2o / 1000 * self.Lv / (self.mothermodel.ra + self.mothermodel.rssoil) * (desatdT * self.mothermodel.theta / surf_press * self.rho / (self.mair/1000) - self.C_H2O_sat_theta + self.C_H2O_veglayer_current[0])
            pLEliq_numerator_Ts = np.sum(parts_LEliq_numerator_Ts)
            numerator_Ts = p1_numerator_Ts + pLEveg_numerator_Ts + pLEsoil_numerator_Ts \
                + pLEliq_numerator_Ts + self.mothermodel.Lambda * self.mothermodel.Tsoil
            p1_denominator_Ts = self.rho * self.cp_const / self.mothermodel.ra 
            pLEveg_denominator_Ts = np.sum(parts_LEveg_denominator_Ts)
            pLEsoil_denominator_Ts = self.mh2o / 1000 * self.Lv / (self.mothermodel.ra + self.mothermodel.rssoil) * desatdT / surf_press * self.rho / (self.mair/1000)
            pLEliq_denominator_Ts = np.sum(parts_LEliq_denominator_Ts)
            denominator_Ts = p1_denominator_Ts + pLEveg_denominator_Ts + pLEsoil_denominator_Ts + pLEliq_denominator_Ts + self.mothermodel.Lambda
            self.Ts   = numerator_Ts / denominator_Ts
#            print('Q')
#            print(self.mothermodel.Q)
#            print(p1_numerator_Ts)
#            print(pLEveg_numerator_Ts)
#            print(pLEsoil_numerator_Ts)
#            print(pLEliq_numerator_Ts)
#            print(denominator_Ts)
        elif self.Ts_mode == 'input':
            self.Ts = Ts
        else:
            raise Exception('invalid Ts_mode')
        for lay in range(self.nveglay):
            self.H2Oplantflux[lay] = self.LAI_veglay[lay] * (1 - self.wetfveg[lay]) * self.H2O_exchange_leaf(rs_H2O_leaf[lay],self.rbveg_H2O[lay],self.C_H2O_veglayer_current[lay],surf_press,self.Ts)
            self.H2Oliqflux[lay] = self.LAI_veglay[lay] * self.wetfveg[lay] * self.H2O_liq_evap(self.mothermodel.ra,self.C_H2O_veglayer_current[lay],surf_press,self.Ts)
        soilflux_H2O = self.H2O_soil_evap(self.mothermodel.ra,self.mothermodel.rssoil,self.C_H2O_veglayer_current[0],surf_press,self.Ts) 
        return self.H2Oplantflux,soilflux_H2O,self.H2Oliqflux #fluxes in mol/m2/s
    
    def run_canopy_mod(self,thetasurf,Ts,w2,Swin,soilflux_COS,C_surf_layer_COS_ppb,soilflux_CO2,C_surf_layer_CO2_ppm,C_surf_layer_H2O_pct,windsp_surf_lay,surf_press,dt_can,e=None,ra_veg=None,fPARdif=None,call_from_init=False):
        #input in SI units unless specified otherwise. soilflux in mol/m2/s, C_surf_layer_COS_ppb in ppb, C_surf_layer_CO2_ppm in ppm,C_surf_layer_H2O_pct in %
        if hasattr(self,'mothermodel'):
            self.mothermodel.vars_rcm = {} #rcm stands for run canopy model
        if call_from_init:
            self.call_from_init = True
        if self.incl_H2O:
            for lay in range(self.nveglay):
                Wlmx = self.LAI_veglay[lay] * self.Wmax
                self.wetfveg[lay] = min(1., self.Wl[lay] / Wlmx) #fraction of veg that is wet, similar system as CLASS
        if self.U_mode == 'Cionco':
            self.U_veg = self.calcU_Cionco(windsp_surf_lay,self.lai_total) #Cionco 1972
        elif self.U_mode == 'HF07':
            self.U_veg = self.calcU_HF07() #Harmann and Finnigan 2007
        if self.K_mode=='int_resistance':
            self.K = self.calcK_ra(self.U_veg,ra_veg,windsp_surf_lay)
        elif self.K_mode=='Launianen':
            self.K = self.calcK_Lau(self.U_veg,windsp_surf_lay)
        G_turb = self.calcTurbConduct(self.K)
        if self.incl_H2O:
            self.e_veglayer_current = self.C_H2O_veglayer_current / self.rho * (self.mair/1000) * surf_press # mol_h2o / m3 to Pa: mol_h2o / m3 * m3 / kg_air * g_air / mol_air * kg_air / g_air gives mol_h2o/mol_air, times pressure in Pa gives Pa
        else:
            self.e_veglayer_current = np.zeros(self.nveglay)
            for lay in range(self.nveglay):
                self.e_veglayer_current[lay] = e
        self.veg_exchange_COS,self.veg_exchange_CO2 = self.calcVegExchange(self.U_veg,thetasurf,Ts,self.e_veglayer_current,w2,Swin,fPARdif)
        summed_veg_exchange_COS = np.sum(self.veg_exchange_COS)
        summed_veg_exchange_CO2 = np.sum(self.veg_exchange_CO2)
        C_surf_layer_COS = C_surf_layer_COS_ppb * 1.e-9 / self.mair * self.rho * 1000# conversion nmol mol-1 (ppb) to mol_COS m-3
        C_surf_layer_CO2 = C_surf_layer_CO2_ppm * 1.e-6 / self.mair * self.rho * 1000 # conversion mumol mol-1 (ppm) to mol_CO2 m-3 : mumol_CO2 / mol_air * mol_CO2/mumol_co2 * mol_air/g_air * kg_air / m3 * g_air/kg_air
        self.C_COS_veglayer_next = self.calcC(G_turb,self.veg_exchange_COS,soilflux_COS,C_surf_layer_COS,self.C_COS_veglayer_current,dt_can)
        self.C_CO2_veglayer_next = self.calcC(G_turb,self.veg_exchange_CO2,soilflux_CO2,C_surf_layer_CO2,self.C_CO2_veglayer_current,dt_can)
        self.turbCOSfluxes = self.calcJ(G_turb,soilflux_COS,self.C_COS_veglayer_current,C_surf_layer_COS)
        self.turbCO2fluxes = self.calcJ(G_turb,soilflux_CO2,self.C_CO2_veglayer_current,C_surf_layer_CO2)
        #time.sleep(0.4)
#        print(self.turbCOSfluxes)
        COSfluxVegSurfLay = self.turbCOSfluxes[-1] #positive if net flux is upward
        CO2fluxVegSurfLay = self.turbCO2fluxes[-1] #positive if net flux is upward
        #Besides the fluxes above, the other turbulent COS and CO2 fluxes are purely diagnostic
        self.C_COS_veglayer_old = cp.deepcopy(self.C_COS_veglayer_current)
        self.C_CO2_veglayer_old = cp.deepcopy(self.C_CO2_veglayer_current)
        self.C_COS_veglayer_current = cp.deepcopy(self.C_COS_veglayer_next)
        self.C_CO2_veglayer_current = cp.deepcopy(self.C_CO2_veglayer_next)
        if self.incl_H2O:
            self.rs_H2O_leaf = self.rs_CO2_leaf / 1.6 #as in CLASS
            if self.Ts_mode != 'CLASS_en_bal':
                self.veg_exchange_H2O,self.soilflux_H2O,self.H2Oliqflux = self.calc_evapotrans(self.U_veg,surf_press,self.rs_H2O_leaf) #in mol/m2/s
            else:
                self.veg_exchange_H2O,self.soilflux_H2O,self.H2Oliqflux = self.calc_evapotrans(self.U_veg,surf_press,self.rs_H2O_leaf,Ts) #in mol/m2/s
            Wl0 = cp.deepcopy(self.Wl)
            for lay in range(self.nveglay):
                self.Wl[lay] = self.H2O_liq_cont(Wl0[lay],self.H2Oliqflux[lay],dt_can)
            summed_veg_exchange_H2O = np.sum(self.veg_exchange_H2O)
            C_surf_layer_H2O = C_surf_layer_H2O_pct * 1.e-2 / self.mair * self.rho * 1000 # conversion cmol mol-1 (%) to mol_H2O m-3 : cmol_H2O / mol_air * mol_H2O/cmol_H2O * mol_air/g_air * kg_air / m3 * g_air/kg_air
            self.C_H2O_veglayer_next = self.calcC(G_turb,self.veg_exchange_H2O,self.soilflux_H2O,C_surf_layer_H2O,self.C_H2O_veglayer_current,dt_can,extra_source = self.H2Oliqflux)
            self.turbH2Ofluxes = self.calcJ(G_turb,self.soilflux_H2O,self.C_H2O_veglayer_current,C_surf_layer_H2O)
            H2OfluxVegSurfLay = self.turbH2Ofluxes[-1] #positive if net flux is upward
            self.C_H2O_veglayer_old = cp.deepcopy(self.C_H2O_veglayer_current)
            self.C_H2O_veglayer_current = cp.deepcopy(self.C_H2O_veglayer_next)
        if not hasattr(self,'mothermodel'): #when runned stand alone, keep track of the time of day
            self.tod += dt_can
            if self.tod >= 24 * 3600: #if end of day
                self.tod = self.tod - 24 * 3600
        if self.incl_H2O:
            return COSfluxVegSurfLay,CO2fluxVegSurfLay,H2OfluxVegSurfLay,summed_veg_exchange_COS,summed_veg_exchange_CO2,summed_veg_exchange_H2O,self.C_COS_veglayer_old,self.C_CO2_veglayer_old,self.C_H2O_veglayer_old,self.Ts,self.soilflux_H2O,self.H2Oliqflux,self.rs_H2O_leaf #note that we return the old concentrations to be consistent with the 'store' function in CLASS
        else:
            return COSfluxVegSurfLay,CO2fluxVegSurfLay,summed_veg_exchange_COS,summed_veg_exchange_CO2,self.C_COS_veglayer_old,self.C_CO2_veglayer_old #note that we return the old concentrations to be consistent with the 'store' function in CLASS

    def calcVegExchange(self,U_veg,thetasurf,Ts,e,w2,Swin,fPARdif=None):
        if self.calc_sun_shad:
            self.gsco2_leaf_sun = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_dry_sun = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_wet_sun = np.zeros(self.nveglay) #leaf scale
            self.gsco2_leaf_sha = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_dry_sha = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_wet_sha = np.zeros(self.nveglay) #leaf scale
            self.fract_sun = np.zeros(self.nveglay)
            self.rs_COS_leaf_sun = np.zeros(self.nveglay)
            self.rs_CO2_leaf_sun = np.zeros(self.nveglay)
            self.rs_COS_leaf_sha = np.zeros(self.nveglay)
            self.rs_CO2_leaf_sha = np.zeros(self.nveglay)
            self.PAR_sun = np.zeros(self.nveglay)
            self.PAR_sha = np.zeros(self.nveglay)
            self.PAR_sun_abs = np.zeros(self.nveglay)
            self.PAR_sha_abs = np.zeros(self.nveglay)
        else:
            self.gsco2_leaf = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_dry = np.zeros(self.nveglay) #leaf scale
            self.An_leaf_wet = np.zeros(self.nveglay) #leaf scale
        self.rbveg_COS = np.zeros(self.nveglay) #leaf scale
        self.rbveg_CO2 = np.zeros(self.nveglay) #leaf scale
        self.COSplantflux = np.zeros(self.nveglay) #canopy scale
        self.CO2plantflux = np.zeros(self.nveglay) #canopy scale
        self.rs_COS_leaf = np.zeros(self.nveglay) #leaf scale
        self.rs_CO2_leaf = np.zeros(self.nveglay) #leaf scale
        self.rtot_veg_COS = np.zeros(self.nveglay) #canopy scale (canopy of one layer)
        self.PAR = np.zeros(self.nveglay)
        self.ci_co2 = np.zeros(self.nveglay)
        self.Ds_veg = np.zeros(self.nveglay)
        if(self.c3c4 == 'c3'):
            c = 0
        elif(self.c3c4 == 'c4'):
            c = 1
        #rahcan=np.max(1.,14.*self.LAI_veglay[0]*self.hc/ ustveg) #not for every layer individually
        for lay in range(self.nveglay):
            self.rbveg_CO2[lay]= self.diffrb_CO2 *180 * (0.07/np.max([1.e-10,U_veg[lay]]))**0.5 #Laurens model line 817 messy_empdep_xtsurf.f90, see also eq 3.6 Jacobs 1994
            if self.use_ags_leaf:
                #ags_leaf calculates flux in mgCO2/m2/s, because I copied this largely from CLASS
                if self.calc_sun_shad:
                    self.gsco2_leaf_sun[lay],self.An_leaf_dry_sun[lay],self.An_leaf_wet_sun[lay],self.gsco2_leaf_sha[lay],self.An_leaf_dry_sha[lay],self.An_leaf_wet_sha[lay],self.fract_sun[lay],self.Ds_veg[lay] = self.ags_leaf(thetasurf,Ts,e[lay],self.C_CO2_veglayer_current[lay],w2,self.rbveg_CO2[lay],Swin,lay,fPARdif)
                    #the lines below scales CO2 uptake from the leaf scale to the canopy scale (for one layer), taking the amount of dry and wet vegetation into account, and the fraction of sunlit and shaded leaves
                    self.An_massflux_sun = self.fract_sun[lay] * self.An_leaf_dry_sun[lay] * (1. - self.wetfveg[lay]) * self.LAI_veglay[lay] + self.fract_sun[lay] * self.An_leaf_wet_sun[lay] * self.wetfveg[lay] * self.LAI_veglay[lay] #mgCO2 / m2 /s
                    self.An_massflux_sha = (1 - self.fract_sun[lay]) * self.An_leaf_dry_sha[lay] * (1. - self.wetfveg[lay]) * self.LAI_veglay[lay] + (1 - self.fract_sun[lay]) * self.An_leaf_wet_sha[lay] * self.wetfveg[lay] * self.LAI_veglay[lay] #mgCO2 / m2 /s
                    self.An_massflux = self.An_massflux_sun + self.An_massflux_sha
                else:
                    self.gsco2_leaf[lay],self.An_leaf_dry[lay],self.An_leaf_wet[lay],self.Ds_veg[lay] = self.ags_leaf(thetasurf,Ts,e[lay],self.C_CO2_veglayer_current[lay],w2,self.rbveg_CO2[lay],Swin,lay)
                    #the line below scales CO2 uptake from the leaf scale to the canopy scale, taking the amount of dry and wet vegetation into account
                    self.An_massflux = self.An_leaf_dry[lay] * (1. - self.wetfveg[lay]) * self.LAI_veglay[lay] + self.An_leaf_wet[lay] * self.wetfveg[lay] * self.LAI_veglay[lay] #mgCO2 / m2 /s
                self.CO2plantflux[lay] = self.An_massflux * 0.001 / self.mco2 #from mgCO2 / m2 /s to molCO2 / m2 /s
            else:
                raise Exception('Alternative for ags_leaf not implemented')
            self.rint_COS_leaf = 1 / self.gliCOS #also leaf scale
            self.gcutCOS_leaf = self.gmin[c] / 2.01 #Seibt et al. 2010
            self.rbveg_COS[lay]= self.diffrb_COS *180 * (0.07/np.max([1.e-10,U_veg[lay]]))**0.5
            if self.calc_sun_shad:
                self.rs_CO2_leaf_sun[lay] = 1 / self.gsco2_leaf_sun[lay] #for dry vegetation, and on the leaf scale 
                self.rs_CO2_leaf_sha[lay] = 1 / self.gsco2_leaf_sha[lay] #for dry vegetation, and on the leaf scale
                self.rs_CO2_leaf[lay] = self.fract_sun[lay] * self.rs_CO2_leaf_sun[lay] + (1 - self.fract_sun[lay]) * self.rs_CO2_leaf_sha[lay]#this one is used only for the water calculations
                self.rs_COS_leaf_sun[lay] = 1.21 * self.rs_CO2_leaf_sun[lay] #Seibt et al 2010
                self.rs_COS_leaf_sha[lay] = 1.21 * self.rs_CO2_leaf_sha[lay] #Seibt et al 2010
                self.rdryveg_COS_leaf_sun = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf_sun[lay] + self.rint_COS_leaf) + 2 * self.gcutCOS_leaf)
                self.rdryveg_COS_leaf_sha = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf_sha[lay] + self.rint_COS_leaf) + 2 * self.gcutCOS_leaf)
                self.rwetveg_COS_leaf_sun = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf_sun[lay] + self.rint_COS_leaf) * (1 - self.stomblock) + 2 * 1 / (1 / (self.gcutCOS_leaf) + self.rws_COS_leaf) + 1 / (self.rws_COS_leaf + self.rs_COS_leaf_sun[lay] + self.rint_COS_leaf) * self.stomblock)
                self.rwetveg_COS_leaf_sha = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf_sha[lay] + self.rint_COS_leaf) * (1 - self.stomblock) + 2 * 1 / (1 / (self.gcutCOS_leaf) + self.rws_COS_leaf) + 1 / (self.rws_COS_leaf + self.rs_COS_leaf_sha[lay] + self.rint_COS_leaf) * self.stomblock)
                #important to calculate the leaf scale resistances for the four options (dry-sun, dry-sha, wet-sun, wet-sha) seperately. You cannot take the four types in parallel and add a boundary layer resistance in series to that, that is a different resistance scheme than when you calculte the leaf scale total resistances for all four types separately, and upscale afterwards with the correct fractions.
                self.rtot_veg_COS[lay] = 1./((1. - self.wetfveg[lay]) * self.LAI_veglay[lay] *(self.fract_sun[lay] * 1./(self.rdryveg_COS_leaf_sun) + (1 - self.fract_sun[lay]) * 1./(self.rdryveg_COS_leaf_sha)) + self.wetfveg[lay] * self.LAI_veglay[lay] * (self.fract_sun[lay] * 1./(self.rwetveg_COS_leaf_sun) + (1 - self.fract_sun[lay]) * 1./(self.rwetveg_COS_leaf_sha)))
            else:
                self.rs_CO2_leaf[lay] = 1 / self.gsco2_leaf[lay] #for dry vegetation, and on the leaf scale
                self.rs_COS_leaf[lay] = 1.21 * self.rs_CO2_leaf[lay] #Seibt et al 2010
                self.rdryveg_COS_leaf = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf[lay] + self.rint_COS_leaf) + 2 * self.gcutCOS_leaf) #factor 2 since two sides on a leaf, but stomata often on just one side. 
                #no internal resistance after cuticular, since we assume concentration to be zero after COS has passed the cuticle (or is included in cuticular resistance)
                self.rwetveg_COS_leaf = self.rbveg_COS[lay] + 1 / (1 / (self.rs_COS_leaf[lay] + self.rint_COS_leaf) * (1 - self.stomblock) + 2 * 1 / (1 / (self.gcutCOS_leaf) + self.rws_COS_leaf) + 1 / (self.rws_COS_leaf + self.rs_COS_leaf[lay] + self.rint_COS_leaf) * self.stomblock)
                #check with Laurens wether line above makes sense: the idea is that we first pass the boundary layer resist, than we have parallel resistance paths through either the cuticle or the stomata. Depending on stomblock variable, the path through the stomata involves
                #3 or two resistances. if stomblock == 0, the stomata are open and the pathway through the stomata is the same as in the dry case. if stomblock is 1, there is an extra resistance of the water layer blocking the stomata added in series.
                #note that parallel conductances can be added
                self.rtot_veg_COS[lay] = 1./((1. - self.wetfveg[lay]) * self.LAI_veglay[lay] *(1./(self.rdryveg_COS_leaf))+ self.wetfveg[lay] * self.LAI_veglay[lay] * (1./(self.rwetveg_COS_leaf))) #canopy scale, where canopy is the canopy of one layer here
            self.COSplantflux[lay] = - self.C_COS_veglayer_current[lay] / self.rtot_veg_COS[lay] #mol/m2/s, minus since uptake should be negative
        return self.COSplantflux,self.CO2plantflux
    
    def calcK_ra(self,U_veg,ra_veg,windsp_surf_lay):
        K = np.zeros(self.nveglay+1) #incoming flux from soil is prescribed 
        for lay in range(1,self.nveglay): #K at interface at bottom of every layer, but at interface closest to the ground we do not calculate it since flux prescribed, and we need one extra at the interface between highest veg layer and surface layer
            K[lay] = self.K_scale * (self.z_veglay[lay] - self.z_veglay[lay-1]) / ra_veg * (U_veg[lay] + U_veg[lay-1]) / 2 / windsp_surf_lay #height needs to be checked
        K[-1] = self.K_scale * (self.hc - self.z_veglay[-1]) / ra_veg * U_veg[-1] / windsp_surf_lay #since 1/ra_veg is K/deltax , but integrated over dx. so ra_veg is 1/K integrated over dx
        return K
    
    def calcK_Lau(self,U_veg,windsp_surf_lay): #from paper Launianen et al 2011
        k = 0.4 #von Karman constant
        d = 0.7 * self.hc 
        K = np.zeros(self.nveglay+1) #incoming flux from soil is prescribed
        l = np.zeros(self.nveglay+1) #mixing length
        u_deriv = np.zeros(self.nveglay+1) #the lowest one will remain zero, we don't need a K at the soil-veg interface
        alf_prim = k * (1 - d / self.hc)
        for i in range(1,self.nveglay): #K at interface at bottom of every layer, but at interface closest to the ground we do not calculate it since flux prescribed, and we need one extra at the interface between highest veg layer and surface layer
            if self.z_int_veg[i] < alf_prim * self.hc / k:
                l[i] = k * self.z_int_veg[i]
            elif (self.z_int_veg[i] < self.hc and self.z_int_veg[i] >= alf_prim * self.hc / k):
                l[i] = alf_prim * self.hc
#            elif self.z_int_veg[i] >= self.hc: # > hc will not happen in the current implementation
#                l[i] = k * (self.z_int_veg[i] - d)
            u_deriv [i] = (U_veg [i] - U_veg [i-1]) / (self.z_veglay[i] - self.z_veglay[i-1])
            K[i] = self.K_scale * l[i]**2 * np.abs(u_deriv [i])
        l[-1] = k * (self.hc - d) #the case where z >= h in Launianen, At the top interface z = hc
        u_deriv [-1] = (windsp_surf_lay - U_veg [-1]) / (self.hc - self.z_veglay[-1])
        K[-1] = self.K_scale * l[-1]**2 * np.abs(u_deriv [-1])
        #This is actually the K for momentum, K_scale can translate it to scalar K
        return K
    
    def calcTurbConduct(self,K): #see p 3 paper Sun 2015
        turb_conduct = np.zeros(self.nveglay +1) #one more than in Sun 2015, since we also calculate the flux towards the surface layer
        for i in range(1,self.nveglay):
            turb_conduct[i] = K[i] / (self.z_veglay[i] - self.z_veglay[i-1])
        #turb_conduct[0] = K[0] / (self.z_veglay[0] - 0); but we dont calculate it since soilflux is prescribed
        turb_conduct[-1] = K[-1] / (self.hc - self.z_veglay[-1])
        return turb_conduct # m s-1
        
    def calcC(self,turb_conduct,veg_exchange,soilflux,C_surf_layer,C_veglayer_current,dt,extra_source = 0):
        source_initial = cp.deepcopy(veg_exchange) #times self.dz_veglay is not needed in contrast to Sun, since uptake in Sun is in mol/m3/s
        source = source_initial + extra_source
        source [0] += soilflux #This is different from Sun, since we have a direct input from the soil, rather calculating the flux here based on the soil-canopy gradient. The soil model already does this.
        source [-1] += turb_conduct[-1] * C_surf_layer #see notes where source term is written, conductance is 1 / resistance
        A_matr = np.zeros((self.nveglay,self.nveglay))
        for i in range(self.nveglay):
            A_matr[i,i] = self.dz_veglay[i] #It is not the dz between two nodes, but the deltaz of the cells (see notes canopy model)!! Note that in the discretized version of eq 1 Sun, the inner deltaz is between nodes (conc diff) 
            #and the outer deltaz is the size of the cell (difference incoming and outgoing flux)
        B_matr = np.zeros((self.nveglay,self.nveglay))
        B_matr[0,0] = - turb_conduct[1] #e.g. turb_conduct[5] is turb_conduct at bottom of cell with index 5
        B_matr[0,1] = turb_conduct[1]
        for i in range(1,self.nveglay-1):
            B_matr[i,i] = -(turb_conduct[i]+turb_conduct[i+1])
            B_matr[i,i+1] = turb_conduct[i+1]
        B_matr[self.nveglay-1,self.nveglay-1] = -(turb_conduct[self.nveglay-1] + turb_conduct[self.nveglay])
        for i in range(1,self.nveglay):
            B_matr[i,i-1] = turb_conduct[i]
        invmatreq12 = np.linalg.inv(2*A_matr - dt*B_matr)
        matr_2_eq12 = np.matmul(2*A_matr + dt * B_matr, C_veglayer_current)
        matr_3_eq12 = matr_2_eq12 + 2*dt* source #
        C_veglayer = np.matmul(invmatreq12,matr_3_eq12)
        for lay in range(len(C_veglayer)):
            if C_veglayer[lay] < 0:
                C_veglayer[lay] = 0.
        return C_veglayer

    def calcJ(self,turb_conduct,soilflux,C_veglayer,C_surf_layer):
        veglayerfluxes = np.zeros(self.nveglay+1) #One flux more than in paper of Sun 2015!
        for i in range(1,self.nveglay):
           veglayerfluxes[i] = -1. * turb_conduct[i] * (C_veglayer[i] - C_veglayer[i-1])
        veglayerfluxes[0] = soilflux
        veglayerfluxes[-1] = -1. * turb_conduct[-1] * (C_surf_layer - C_veglayer[-1])
        if hasattr(self,'mothermodel'):
            if self.mothermodel.save_vars_indict:
                the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
                for variablename in the_locals: #note that the self variables are not included
                    if (str(variablename) != 'self' and str(variablename) not in inspect.getfullargspec(self.calcJ).args): #inspect.getfullargspec(self.calcJ).args gives argument list of function, those should not be updated
                        self.mothermodel.vars_rcm.update({variablename: the_locals[variablename]})
        return veglayerfluxes
