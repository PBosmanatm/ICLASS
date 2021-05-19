# -*- coding: utf-8 -*-
# 
# CLASS
# Copyright (c) 2010-2015 Meteorology and Air Quality section, Wageningen University and Research centre
# Copyright (c) 2011-2015 Jordi Vila-Guerau de Arellano
# Copyright (c) 2011-2015 Chiel van Heerwaarden
# Copyright (c) 2011-2015 Bart van Stratum
# Copyright (c) 2011-2015 Kees van den Dries
# 
# This file is part of CLASS
# 
# CLASS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# CLASS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with CLASS.  If not, see <http://www.gnu.org/licenses/>.
#

import copy as cp
import numpy as np
import sys
import os.path
if os.path.isfile('soilCOSmodel.py'): #with if statement, so you do not need the file soilCOSmodel.py if it is not used
    import soilCOSmodel as sCOSm
if os.path.isfile('canopy_model.py'):
    import canopy_model as canm
if os.path.isfile('pho_sib4.py'):
    import pho_sib4 as sib4
#import ribtol

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86))

def qsat(T,p):
    return 0.622 * esat(T) / p

class model:
    def __init__(self, model_input):
        # initialize the different components of the model
        self.input = cp.deepcopy(model_input)
        self.nr_of_surf_lay_its = 10 #nr of surface layer iterations in init function
        
  
    def run(self,checkpoint=False,updatevals_surf_lay=True,delete_at_end=True,save_vars_indict=False):
        # initialize model variables
        self.updatevals_surf_lay = updatevals_surf_lay #init needs it, it is a switch to update variables in the surface layer function (self.Cs and self.ustar) or not
        self.save_vars_indict = False #save model variables in dictionary, only needed when doing gradient test in inverse modelling
        if checkpoint: #checkpointing is saving values of variables (needed for the adjoint)
            self.checkpoint = True
            self.cpx_init = [] #a separate set just for init
            for t in range((self.nr_of_surf_lay_its)):
                self.cpx_init += [{}]
        else:
            self.checkpoint = False
        self.init()
        if checkpoint:
            self.cpx = [] #I cannot do this before running init, since self.tsteps is calculated there
            for t in range((self.tsteps)):
                self.cpx += [{}]
        if save_vars_indict:
            self.save_vars_indict = True #best to do this after init, otherwise time is wasted in saving variables
        # time integrate model 
        for self.t in range(self.tsteps):
          
            # time integrate components
            self.timestep()
  
        # delete unnecessary variables from memory
        if delete_at_end==True:
            self.exitmodel()
        
    def init(self):
        # assign variables from input data
        # initialize constants
        self.Lv         = 2.5e6                 # heat of vaporization [J kg-1]
        self.cp         = 1005.                 # specific heat of dry air [J kg-1 K-1]
        self.rho        = 1.2                   # density of air [kg m-3]
        self.k          = 0.4                   # Von Karman constant [-]
        self.g          = 9.81                  # gravity acceleration [m s-2]
        self.Rd         = 287.                  # gas constant for dry air [J kg-1 K-1]
        self.Rv         = 461.5                 # gas constant for moist air [J kg-1 K-1]
        self.bolz       = 5.67e-8               # Bolzman constant [-]
        self.rhow       = 1000.                 # density of water [kg m-3]
        self.S0         = 1368.                 # solar constant [W m-2]
        
        # Read switches
        self.sw_ml      = self.input.sw_ml      # mixed-layer model switch
        self.sw_shearwe = self.input.sw_shearwe # shear growth ABL switch
        self.sw_fixft   = self.input.sw_fixft   # Fix the free-troposphere switch
        self.sw_wind    = self.input.sw_wind    # prognostic wind switch
        self.sw_sl      = self.input.sw_sl      # surface layer switch
        self.sw_rad     = self.input.sw_rad     # radiation switch
        self.sw_ls      = self.input.sw_ls      # land surface switch
        self.ls_type    = self.input.ls_type    # land surface paramaterization (js or ags)
        if self.ls_type == 'canopy_model':
            if not os.path.isfile('canopy_model.py'):
                raise Exception('canopy_model selected, but cannot find canopy_model.py')
        elif self.ls_type == 'sib4':
            if not os.path.isfile('pho_sib4.py'):
                raise Exception('sib4 ls_type selected, but cannot find pho_sib4.py')
        self.sw_cu      = self.input.sw_cu      # cumulus parameterization switch
        self.soilCOSmodel = None                #an instance of the soilCOSmodel
        self.soilCOSmodeltype = self.input.soilCOSmodeltype # soil COS model switch
        if self.soilCOSmodeltype == 'Sun_Ogee':
            if not os.path.isfile('soilCOSmodel.py'):
                raise Exception('Sun_Ogee model selected, but cannot find soilCOSmodel.py')
        self.sw_dynamicsl_border = False
        if hasattr(self.input,'sw_dynamicsl_border'):
            self.sw_dynamicsl_border = self.input.sw_dynamicsl_border
        self.sw_use_ribtol = False #use ribtol, the more complex way of surface layer calculations
        if hasattr(self.input,'sw_use_ribtol'):
            self.sw_use_ribtol = self.input.sw_use_ribtol
#        self.use_rsl = False #use roughness sublayer
#        if hasattr(self.input,'use_rsl'):
#            self.use_rsl = self.input.use_rsl
        self.sw_advfp = False #switch for prescribed advection to take place over full profile (also in Free troposphere), only in ML if FALSE
        if hasattr(self.input,'sw_advfp'):
            self.sw_advfp = self.input.sw_advfp
        self.sw_printwarnings = True #print or hide warnings
        if hasattr(self.input,'sw_printwarnings'):
            self.sw_printwarnings = self.input.sw_printwarnings
        self.sw_dyn_beta = False
        if hasattr(self.input,'sw_dyn_beta'):
            self.sw_dyn_beta = self.input.sw_dyn_beta

        # A-Gs constants and settings
        # Plant type:       -C3-     -C4-
        if hasattr(self.input,'CO2comp298'):
            self.CO2comp298 = self.input.CO2comp298
        else:
            self.CO2comp298 =  [68.5,    4.3    ]   # CO2 compensation concentration [mg m-3]
        if hasattr(self.input,'Q10CO2'):
            self.Q10CO2 = self.input.Q10CO2
        else:
            self.Q10CO2     =  [1.5,     1.5    ]   # function parameter to calculate CO2 compensation concentration [-]
        if hasattr(self.input,'gm298'):
            self.gm298 = self.input.gm298
        else:
            self.gm298      =  [7.0,     17.5   ]   # mesophyill conductance at 298 K [mm s-1]
        if hasattr(self.input,'Ammax298'):
            self.Ammax298 = self.input.Ammax298
        else:
            self.Ammax298   =  [2.2,     1.7    ]   # CO2 maximal primary productivity [mg m-2 s-1]
        if hasattr(self.input,'Q10gm'):
            self.Q10gm = self.input.Q10gm
        else:
            self.Q10gm      =  [2.0,     2.0    ]   # function parameter to calculate mesophyll conductance [-]
        if hasattr(self.input,'T1gm'):
            self.T1gm = self.input.T1gm
        else:
            self.T1gm       =  [278.,    286.   ]   # reference temperature to calculate mesophyll conductance gm [K]
        if hasattr(self.input,'T2gm'):
            self.T2gm = self.input.T2gm
        else:
            self.T2gm       =  [301.,    309.   ]   # reference temperature to calculate mesophyll conductance gm [K]
        if hasattr(self.input,'Q10Am'):
            self.Q10Am = self.input.Q10Am
        else:
            self.Q10Am      =  [2.0,     2.0    ]   # function parameter to calculate maximal primary profuctivity Ammax
        if hasattr(self.input,'T1Am'):
            self.T1Am = self.input.T1Am
        else:
            self.T1Am       =  [281.,    286.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
        if hasattr(self.input,'T2Am'):
            self.T2Am = self.input.T2Am
        else:
            self.T2Am       =  [311.,    311.   ]   # reference temperature to calculate maximal primary profuctivity Ammax [K]
        if hasattr(self.input,'f0'):
            self.f0 = self.input.f0
        else:
            self.f0         =  [0.89,    0.85   ]   # maximum value Cfrac [-]
        if hasattr(self.input,'ad'):
            self.ad = self.input.ad
        else:
            self.ad         =  [0.07,    0.15   ]   # regression coefficient to calculate Cfrac [kPa-1]
        if hasattr(self.input,'alpha0'):
            self.alpha0 = self.input.alpha0
        else:
            self.alpha0     =  [0.017,   0.014  ]   # initial low light conditions [mg J-1]
        if hasattr(self.input,'Kx'):
            self.Kx = self.input.Kx
        else:
            self.Kx         =  [0.7,     0.7    ]   # extinction coefficient PAR [-]
        if hasattr(self.input,'gmin'):
            self.gmin = self.input.gmin
        else:
            self.gmin       =  [0.25e-3, 0.25e-3]   # cuticular (minimum) conductance [mm s-1]
        if self.sw_ls:
            if self.ls_type == 'ags':
                if hasattr(self.input,'PARfract'):
                    self.PARfract = self.input.PARfract
                else:
                    self.PARfract = 0.5 #fraction of incoming shortwave radiation that is PAR (at the vegetation)

        self.mco2       =  44.;                 # molecular weight CO2 [g mol -1]
        self.mcos       =  12. + 16. + 32.07;   # molecular weight COS [g mol -1]
        self.mair       =  28.9;                # molecular weight air [g mol -1]
        self.mh2o       =  18                   # molecular weight water [g mol -1]
        self.nuco2q     =  1.6;                 # ratio molecular viscosity water to carbon dioxide

        self.Cw         =  0.0016;              # constant water stress correction (eq. 13 Jacobs et al. 2007) [-]
        self.wmax       =  0.55;                # upper reference value soil water [-]
        self.wmin       =  0.005;               # lower reference value soil water [-]
        if hasattr(self.input,'R10'):
            self.R10 = self.input.R10
        else:
            self.R10        =  0.23;                # respiration at 10 C [mg CO2 m-2 s-1]
        self.E0         =  53.3e3;              # activation energy [53.3 kJ kmol-1]

  
        # initialize mixed-layer
        self.h          = self.input.h          # initial ABL height [m]
        self.Ps         = self.input.Ps         # surface pressure [Pa]
        self.divU       = self.input.divU       # horizontal large-scale divergence of wind [s-1]
        self.ws         = None                  # large-scale vertical velocity [m s-1]
        self.wf         = None                  # mixed-layer growth due to radiative divergence [m s-1]
        self.fc         = self.input.fc         # coriolis parameter [s-1]
        self.we         = -1.                   # entrainment velocity [m s-1]
       
         # Temperature 
        self.theta      = self.input.theta      # initial mixed-layer potential temperature [K]
        self.deltatheta = self.input.deltatheta     # initial temperature jump at h [K]
        self.gammatheta = self.input.gammatheta # free atmosphere potential temperature lapse rate [K m-1]
        if hasattr(self.input,'gammatheta2'):
            self.gammatheta2 = self.input.gammatheta2
            self.htrans = self.input.htrans #above this height, use gammatheta2, otherwise gammatheta
        else:
            self.gammatheta2 = self.gammatheta #if gammatheta2 not given, take equal to gammatheta
            self.htrans = 1000000. #the value does not matter, we just need a value for it
        self.advtheta   = self.input.advtheta   # advection of heat [K s-1]
        if not self.sw_dyn_beta:
            self.beta       = self.input.beta       # entrainment ratio for virtual heat [-]
        self.wtheta     = self.input.wtheta     # surface kinematic heat flux [K m s-1]
        self.wthetae    = None                  # entrainment kinematic heat flux [K m s-1]
 
        self.wstar      = 0.                    # convective velocity scale [m s-1]
 
        # 2m diagnostic variables 
        self.T2m        = None                  # 2m temperature [K]
        self.q2m        = None                  # 2m specific humidity [kg kg-1]
        self.e2m        = None                  # 2m vapor pressure [Pa]
        self.esat2m     = None                  # 2m saturated vapor pressure [Pa]
        self.u2m        = None                  # 2m u-wind [m s-1]
        self.v2m        = None                  # 2m v-wind [m s-1]
 
        # Surface variables 
        self.thetasurf  = self.input.theta      # surface potential temperature [K]
        self.thetavsurf = None                  # surface virtual potential temperature [K]
        self.qsurf      = None                  # surface specific humidity [g kg-1]

        # Mixed-layer top variables
        self.P_h        = None                  # Mixed-layer top pressure [pa]
        self.T_h        = None                  # Mixed-layer top absolute temperature [K]
        self.q2_h       = None                  # Mixed-layer top specific humidity variance [kg2 kg-2]
        self.CO22_h     = None                  # Mixed-layer top CO2 variance [ppm2]
        self.RH_h       = None                  # Mixed-layer top relavtive humidity [-]
        self.dz_h       = None                  # Transition layer thickness [-]
        self.lcl        = None                  # Lifting condensation level [m]

        # Virtual temperatures and fluxes
        self.thetav     = None                  # initial mixed-layer potential temperature [K]
        self.deltathetav= None                  # initial virtual temperature jump at h [K]
        self.wthetav    = None                  # surface kinematic virtual heat flux [K m s-1]
        self.wthetave   = None                  # entrainment kinematic virtual heat flux [K m s-1]
       
        # Moisture 
        self.q          = self.input.q          # initial mixed-layer specific humidity [kg kg-1]
        self.deltaq     = self.input.deltaq         # initial specific humidity jump at h [kg kg-1]
        self.gammaq     = self.input.gammaq     # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.advq       = self.input.advq       # advection of moisture [kg kg-1 s-1]
        self.wq         = self.input.wq         # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wqe        = None                  # entrainment moisture flux [kg kg-1 m s-1]
        self.wqM        = None                  # moisture cumulus mass flux [kg kg-1 m s-1]
  
        self.qsatvar       = None                  # mixed-layer saturated specific humidity [kg kg-1]
        self.esatvar       = None                  # mixed-layer saturated vapor pressure [Pa]
        self.e          = None                  # mixed-layer vapor pressure [Pa]
        self.qsatsurf   = None                  # surface saturated specific humidity [g kg-1]
        self.dqsatdT    = None                  # slope saturated specific humidity curve [g kg-1 K-1]
        
        # Wind 
        self.u          = self.input.u          # initial mixed-layer u-wind speed [m s-1]
        self.deltau     = self.input.deltau         # initial u-wind jump at h [m s-1]
        self.gammau     = self.input.gammau     # free atmosphere u-wind speed lapse rate [s-1]
        self.advu       = self.input.advu       # advection of u-wind [m s-2]
        
        self.v          = self.input.v          # initial mixed-layer u-wind speed [m s-1]
        self.deltav     = self.input.deltav     # initial u-wind jump at h [m s-1]
        self.gammav     = self.input.gammav     # free atmosphere v-wind speed lapse rate [s-1]
        self.advv       = self.input.advv       # advection of v-wind [m s-2]
 
        # Tendencies 
        self.htend          = None                  # tendency of CBL [m s-1]
        self.thetatend      = None                  # tendency of mixed-layer potential temperature [K s-1]
        self.deltathetatend = None                  # tendency of potential temperature jump at h [K s-1]
        self.qtend          = None                  # tendency of mixed-layer specific humidity [kg kg-1 s-1]
        self.deltaqtend     = None                  # tendency of specific humidity jump at h [kg kg-1 s-1]
        self.CO2tend        = None                  # tendency of CO2 humidity [ppm]
        self.COStend        = None                  # tendency of COS [ppb]
        self.deltaCO2tend   = None                  # tendency of CO2 jump at h [ppm s-1]
        self.deltaCOStend   = None                  # tendency of CO2 jump at h [ppb s-1]
        self.utend          = None                  # tendency of u-wind [m s-1 s-1]
        self.deltautend     = None                  # tendency of u-wind jump at h [m s-1 s-1]
        self.vtend          = None                  # tendency of v-wind [m s-1 s-1]
        self.deltavtend     = None                  # tendency of v-wind jump at h [m s-1 s-1]
        self.dztend         = None                  # tendency of transition layer thickness [m s-1]
  
        # initialize surface layer
        self.ustar      = self.input.ustar      # surface friction velocity [m s-1]
        self.uw         = None                  # surface momentum flux in u-direction [m2 s-2]
        self.vw         = None                  # surface momentum flux in v-direction [m2 s-2]
        self.z0m        = self.input.z0m        # roughness length for momentum [m]
        self.z0h        = self.input.z0h        # roughness length for scalars [m]
        if hasattr(self.input,'Cs'):
            self.Cs         = self.input.Cs         # drag coefficient for scalars [-]
        else:
            self.Cs         = 1e12                  # drag coefficient for scalars [-]
        #Cm is calculated before it used, no need to specify it here, even though this was done in the original CLASS from 2019
        self.L          = None                  # Obukhov length [m]
        self.Rib        = None                  # bulk Richardson number [-]
        self.ra         = None                  # aerodynamic resistance [s m-1]
        self.sw_useWilson = False
        if hasattr(self.input,'sw_useWilson'):
            self.sw_useWilson  = self.input.sw_useWilson  #switch to use Wilson or Businger Dyer for flux gradient relationships
        self.sw_model_stable_con = True
        if hasattr(self.input,'sw_model_stable_con'):
            self.sw_model_stable_con = self.input.sw_model_stable_con #switch to use Businger Dyer or nothing for flux gradient relationships in stable conditions
  
        # initialize radiation
        self.lat        = self.input.lat        # latitude [deg]
        self.lon        = self.input.lon        # longitude [deg]
        self.doy        = self.input.doy        # day of the year [-]
        self.tstart     = self.input.tstart     # time of the day [-]
        self.cc         = self.input.cc         # cloud cover fraction [-]
        self.Swin       = None                  # incoming short wave radiation [W m-2]
        self.Swout      = None                  # outgoing short wave radiation [W m-2]
        self.Lwin       = None                  # incoming long wave radiation [W m-2]
        self.Lwout      = None                  # outgoing long wave radiation [W m-2]
        self.sinlea     = None
        self.Q          = self.input.Q          # net radiation [W m-2], this value is not used if sw_rad == True. But if sw_rad == False and sw_ls == True, the model will crash as Swin is None...
        self.dFz        = self.input.dFz        # cloud top radiative divergence [W m-2] 
  
        # initialize land surface
        self.wg         = self.input.wg         # volumetric water content top soil layer [m3 m-3]
        self.w2         = self.input.w2         # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil      = self.input.Tsoil      # temperature top soil layer [K]
        self.T2         = self.input.T2         # temperature deeper soil layer [K]
                           
        self.a          = self.input.a          # Clapp and Hornberger retention curve parameter a [-]
        self.b          = self.input.b          # Clapp and Hornberger retention curve parameter b [-]
        self.p          = self.input.p          # Clapp and Hornberger retention curve parameter p [-]
        self.CGsat      = self.input.CGsat      # saturated soil conductivity for heat
                           
        self.wsat       = self.input.wsat       # saturated volumetric water content ECMWF config [-]
        self.wfc        = self.input.wfc        # volumetric water content field capacity [-]
        self.wwilt      = self.input.wwilt      # volumetric water content wilting point [-]
                           
        self.C1sat      = self.input.C1sat      
        self.C2ref      = self.input.C2ref      

        self.c_beta     = self.input.c_beta     # Curvature plant water-stress factor (0..1) [-]
        
        self.LAI        = self.input.LAI        # leaf area index [-]
        self.gD         = self.input.gD         # correction factor transpiration for VPD [-]
        self.rsmin      = self.input.rsmin      # minimum resistance transpiration [s m-1]
        self.rssoilmin  = self.input.rssoilmin  # minimum resistance soil evaporation [s m-1]
        self.alpha      = self.input.alpha      # surface albedo [-]
  
        self.rs         = 1.e6                  # resistance transpiration [s m-1]
        self.rssoil     = 1.e6                  # resistance soil [s m-1]
                           
        self.Ts         = self.input.Ts         # surface temperature [K]
                           
        self.cveg       = self.input.cveg       # vegetation fraction [-]
        self.Wmax       = self.input.Wmax       # thickness of water layer on wet vegetation [m]
        if self.ls_type != 'canopy_model':
            self.Wl         = self.input.Wl         # equivalent water layer depth for wet vegetation [m]
        self.cliq       = None                  # wet fraction [-]
                          
        self.Lambda     = self.input.Lambda     # thermal diffusivity skin layer [-]
  
        self.Tsoiltend  = None                  # soil temperature tendency [K s-1]
        self.wgtend     = None                  # soil moisture tendency [m3 m-3 s-1]
        self.Wltend     = None                  # equivalent liquid water tendency [m s-1]
  
        self.H          = None                  # sensible heat flux [W m-2]
        self.LE         = None                  # evapotranspiration [W m-2]
        self.LEliq      = None                  # open water evaporation [W m-2]
        self.LEveg      = None                  # transpiration [W m-2]
        self.LEsoil     = None                  # soil evaporation [W m-2]
        self.LEpot      = None                  # potential evaporation [W m-2]
        self.LEref      = None                  # reference evaporation using rs = rsmin / LAI [W m-2]
        self.G          = None                  # ground heat flux [W m-2]

        # initialize A-Gs surface scheme
        self.c3c4       = self.input.c3c4       # plant type ('c3' or 'c4')
        if hasattr(self.input,'ags_C_mode'): #only if using the normal a-gs, not if using the canopy model
            self.ags_C_mode = self.input.ags_C_mode # which mixing ratios to use in ags, 'surf' or 'MXL'
            if self.ags_C_mode == 'surf' and self.sw_sl == False:
                raise Exception('When ags_C_mode set to surf, turn on the surface layer')
        else:
            self.ags_C_mode = 'MXL'
 
        # initialize cumulus parameterization
        self.sw_cu      = self.input.sw_cu      # Cumulus parameterization switch
        self.dz_h       = self.input.dz_h       # Transition layer thickness [m]
        self.ac         = 0.                    # Cloud core fraction [-]
        self.M          = 0.                    # Cloud core mass flux [m s-1] 
        self.wqM        = 0.                    # Cloud core moisture flux [kg kg-1 m s-1] 
  
        # initialize time variables
        self.tsteps = int(np.floor(self.input.runtime / self.input.dt))
        self.dt     = self.input.dt
        self.t      = 0
        
        # CO2,COS and canopy
        #fac = self.mair / (self.rho*self.mco2)  # Conversion factor mgC m-2 s-1 to ppm m s-1
        self.CO2        = self.input.CO2        # initial mixed-layer CO2 [ppm]
        self.COS        = self.input.COS        # initial mixed-layer COS [ppb]
        if hasattr(self.input,'alfa_sto'):
            self.alfa_sto = self.input.alfa_sto #dimensionless coefficient for multiplying stomatal conductance with 
        else:
            self.alfa_sto = 1.0
        self.COSmeasuring_height = 10. #assume 10 if not given
        if hasattr(self.input,'COSmeasuring_height'):
            self.COSmeasuring_height = self.input.COSmeasuring_height        # height COS mixing rat measurements [m]   
        if self.COSmeasuring_height < self.z0h:
            raise Exception('measuring height below z0h')
        self.COSmeasuring_height2 = 10.
        if hasattr(self.input,'COSmeasuring_height2'):
            self.COSmeasuring_height2 = self.input.COSmeasuring_height2        # height COS mixing rat measurements [m], in case of a second set of obs
        if self.COSmeasuring_height2 < self.z0h:
            raise Exception('measuring height below z0h')
        self.COSmeasuring_height3 = 10.
        if hasattr(self.input,'COSmeasuring_height3'):
            self.COSmeasuring_height3 = self.input.COSmeasuring_height3        # height COS mixing rat measurements [m]  , in case of a third set of obs 
        if self.COSmeasuring_height3 < self.z0h:
            raise Exception('measuring height below z0h')
        self.CO2measuring_height = 10.
        if hasattr(self.input,'CO2measuring_height'):
            self.CO2measuring_height = self.input.CO2measuring_height        # height CO2 mixing rat measurements [m]
        if self.CO2measuring_height < self.z0h:
            raise Exception('measuring height below z0h')
        self.CO2measuring_height2 = 10.
        if hasattr(self.input,'CO2measuring_height2'):
            self.CO2measuring_height2 = self.input.CO2measuring_height2        # height CO2 mixing rat measurements [m]   , in case of a second set of obs
        if self.CO2measuring_height2 < self.z0h:
            raise Exception('measuring height below z0h')
        self.CO2measuring_height3 = 10.
        if hasattr(self.input,'CO2measuring_height3'):
            self.CO2measuring_height3 = self.input.CO2measuring_height3        # height CO2 mixing rat measurements [m], in case of a third set of obs
        if self.CO2measuring_height3 < self.z0h:
            raise Exception('measuring height below z0h')
        self.CO2measuring_height4 = 10.
        if hasattr(self.input,'CO2measuring_height4'):
            self.CO2measuring_height4 = self.input.CO2measuring_height4        # height CO2 mixing rat measurements [m], in case of a fourth set of obs
        if self.CO2measuring_height4 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height = 10.
        if hasattr(self.input,'Tmeasuring_height'):
            self.Tmeasuring_height = self.input.Tmeasuring_height        # height temperature measurements [m]
        if self.Tmeasuring_height < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height2 = 10.
        if hasattr(self.input,'Tmeasuring_height2'):
            self.Tmeasuring_height2 = self.input.Tmeasuring_height2        # height temperature measurements, in case of a 2nd set of obs [m]
        if self.Tmeasuring_height2 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height3 = 10.
        if hasattr(self.input,'Tmeasuring_height3'):
            self.Tmeasuring_height3 = self.input.Tmeasuring_height3        # height temperature measurements, in case of a 3th set of obs [m]
        if self.Tmeasuring_height3 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height4 = 10.
        if hasattr(self.input,'Tmeasuring_height4'):
            self.Tmeasuring_height4 = self.input.Tmeasuring_height4        # height temperature measurements, in case of a 4th set of obs [m]
        if self.Tmeasuring_height4 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height5 = 10.
        if hasattr(self.input,'Tmeasuring_height5'):
            self.Tmeasuring_height5 = self.input.Tmeasuring_height5       # height temperature measurements, in case of a 5th set of obs [m]
        if self.Tmeasuring_height5 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height6 = 10.
        if hasattr(self.input,'Tmeasuring_height6'):
            self.Tmeasuring_height6 = self.input.Tmeasuring_height6        # height temperature measurements, in case of a 6th set of obs [m]
        if self.Tmeasuring_height6 < self.z0h:
            raise Exception('measuring height below z0h')
        self.Tmeasuring_height7 = 10.
        if hasattr(self.input,'Tmeasuring_height7'):
            self.Tmeasuring_height7 = self.input.Tmeasuring_height7        # height temperature measurements, in case of a 7th set of obs [m]
        if self.Tmeasuring_height7 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height = 10.
        if hasattr(self.input,'qmeasuring_height'):
            self.qmeasuring_height = self.input.qmeasuring_height        # height spec. hum. measurements [m]
        if self.qmeasuring_height < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height2 = 10.
        if hasattr(self.input,'qmeasuring_height2'):
            self.qmeasuring_height2 = self.input.qmeasuring_height2        # height spec. hum. measurements [m], in case of a second set of obs
        if self.qmeasuring_height2 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height3 = 10.
        if hasattr(self.input,'qmeasuring_height3'):
            self.qmeasuring_height3 = self.input.qmeasuring_height3        # height spec. hum. measurements [m], in case of a 3th set of obs
        if self.qmeasuring_height3 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height4 = 10.
        if hasattr(self.input,'qmeasuring_height4'):
            self.qmeasuring_height4 = self.input.qmeasuring_height4        # height spec. hum. measurements [m], in case of a 4th set of obs
        if self.qmeasuring_height4 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height5 = 10.
        if hasattr(self.input,'qmeasuring_height5'):
            self.qmeasuring_height5 = self.input.qmeasuring_height5        # height spec. hum. measurements [m], in case of a 5th set of obs
        if self.qmeasuring_height5 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height6 = 10.
        if hasattr(self.input,'qmeasuring_height6'):
            self.qmeasuring_height6 = self.input.qmeasuring_height6        # height spec. hum. measurements [m], in case of a 6th set of obs
        if self.qmeasuring_height6 < self.z0h:
            raise Exception('measuring height below z0h')
        self.qmeasuring_height7 = 10.
        if hasattr(self.input,'qmeasuring_height7'):
            self.qmeasuring_height7 = self.input.qmeasuring_height7        # height spec. hum. measurements [m], in case of a 7th set of obs
        if self.qmeasuring_height7 < self.z0h:
            raise Exception('measuring height below z0h')
        
        self.deltaCO2   = self.input.deltaCO2       # initial CO2 jump at h [ppm]
        self.deltaCOS   = self.input.deltaCOS       # initial COS jump at h [ppb]
        self.gammaCO2   = self.input.gammaCO2   # free atmosphere CO2 lapse rate [ppm m-1]
        self.gammaCOS   = self.input.gammaCOS   # free atmosphere CO2 lapse rate [ppb m-1]
        self.advCO2     = self.input.advCO2     # advection of CO2 [ppm s-1]
        self.advCOS     = self.input.advCOS     # advection of COS [ppb s-1]
        self.wCO2       = self.input.wCO2       # surface kinematic CO2 flux [ppm m s-1]
        self.wCOS       = self.input.wCOS # surface kinematic COS flux [ppb m s-1] #used in surf layer module before calculated in ags (via call to land surface)
        self.wCO2A      = 0                     # surface assimulation CO2 flux [ppm m s-1]
        self.wCO2R      = 0                     # surface respiration CO2 flux [ppm m s-1]
        self.wCO2e      = None                  # entrainment CO2 flux [ppm m s-1]
        self.wCO2M      = 0                     # CO2 mass flux [ppm m s-1]
        self.wCOSM      = 0                     # COS mass flux [ppb m s-1]
        if hasattr(self.input,'gciCOS'):
            self.gciCOS     = self.input.gciCOS     # COS canopy scale internal conductance m/s
        if self.ls_type == 'canopy_model':
            self.incl_H2Ocan = True
            if hasattr(self.input,'incl_H2O'):
                self.incl_H2Ocan = self.input.incl_H2Ocan
            if hasattr(self.input,'ra_veg'):
                self.ra_veg     = self.input.ra_veg #resistance of vegetation for canopy model s m-1, only for K_mode='int_resistance'
            else:
                self.ra_veg = None
            if hasattr(self.input,'dt_can'):
                self.dt_can     = self.input.dt_can #resistance of vegetation for canopy model s m-1, only for K_mode='int_resistance'
                if self.dt/self.dt_can != np.floor(self.dt/self.dt_can):
                    raise Exception('dt should be a multiple of dt_can')
            else:
                self.dt_can = self.dt
            self.calc_sun_shad = self.input.calc_sun_shad
            if self.calc_sun_shad:
                self.prescr_fPARdif = self.input.prescr_fPARdif
                if self.prescr_fPARdif:
                    self.fPARdif = self.input.fPARdif #fraction of diffuse PAR, either a fixed number or an array the size of the number of timesteps
        
        if not self.sw_ls: #allow for prescribing time-dependent fluxes if land surface not used. For all the scalars, an initial or constant flux is already read in before. 
            #Make sure that the prescribed initial value of the flux (e.g. wtheta) is identical to the first value of the flux read in here (e.g. first value of wtheta_input) !!
            if hasattr(self.input,'wtheta_input'):
                self.wtheta_input = self.input.wtheta_input
                if len(self.wtheta_input) != self.tsteps: #check wether dimensions are ok
                    raise Exception('Wrong length of wtheta_input')
            if hasattr(self.input,'wq_input'):
                self.wq_input = self.input.wq_input
                if len(self.wq_input) != self.tsteps:
                    raise Exception('Wrong length of wq_input')
            if hasattr(self.input,'wCO2_input'):
                self.wCO2_input = self.input.wCO2_input
                if len(self.wCO2_input) != self.tsteps:
                    raise Exception('Wrong length of wCO2_input')
            if hasattr(self.input,'wCOS_input'):
                self.wCOS_input = self.input.wCOS_input
                if len(self.wCOS_input) != self.tsteps:
                    raise Exception('Wrong length of wCOS_input')
                
        # Some sanity checks for valid input
        if (self.c_beta is None): 
            self.c_beta = 0                     # Zero curvature; linear response
        assert(self.c_beta >= 0 or self.c_beta <= 1)

        # initialize output
        self.out = model_output(self,self.tsteps)
 
        self.statistics(call_from_init=True)
  
        # calculate initial diagnostic variables
        if(self.sw_rad):
            self.run_radiation(call_from_init=True)
 
        if(self.sw_sl):
            for i in range(self.nr_of_surf_lay_its): 
                self.run_surface_layer(call_from_init=True,iterationnumber=i)
  
        if(self.sw_ls):
            self.run_land_surface(call_from_init=True)

        if(self.sw_cu):
            self.run_mixed_layer(call_from_init=True)
            self.run_cumulus(call_from_init=True)
        
        if(self.sw_ml):
            self.run_mixed_layer(call_from_init=True)

    def timestep(self):
        if not self.sw_ls: #only if land surface not on, otherwise fluxes are calculated
            if hasattr(self,'wtheta_input'):
                self.wtheta = self.wtheta_input[self.t]
            if hasattr(self,'wq_input'):
                self.wq = self.wq_input[self.t]
            if hasattr(self,'wCO2_input'):
                self.wCO2 = self.wCO2_input[self.t]
            if hasattr(self,'wCOS_input'):
                self.wCOS = self.wCOS_input[self.t]
        self.statistics()

        # run radiation model
        if(self.sw_rad):
            self.run_radiation()
  
        # run surface layer model
        if(self.sw_sl):
            self.run_surface_layer()
        
        # run land surface model
        if(self.sw_ls):
            self.run_land_surface()
 
        # run cumulus parameterization
        if(self.sw_cu):
            self.run_cumulus()
   
        # run mixed-layer model
        if(self.sw_ml):
            self.run_mixed_layer()
 
        # store output before time integration
        self.store()
  
        # time integrate land surface model
        if(self.sw_ls):
            self.integrate_land_surface()
  
        # time integrate mixed-layer model
        if(self.sw_ml):
            self.integrate_mixed_layer()
  
    def statistics(self,call_from_init=False):
        self.vars_stat= {}
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['stat_q']          = self.q
                self.cpx_init[0]['stat_theta']      = self.theta
                self.cpx_init[0]['stat_wq']         = self.wq 
                self.cpx_init[0]['stat_deltatheta'] = self.deltatheta 
                self.cpx_init[0]['stat_deltaq']     = self.deltaq
                self.cpx_init[0]['stat_t']          = self.t
                self.cpx_init[0]['stat_p_lcl_end']  = [] #this is special, for the while loop
                self.cpx_init[0]['stat_T_lcl_end']  = []
            else:
                self.cpx[self.t]['stat_q']          = self.q
                self.cpx[self.t]['stat_theta']      = self.theta
                self.cpx[self.t]['stat_wq']         = self.wq 
                self.cpx[self.t]['stat_deltatheta'] = self.deltatheta 
                self.cpx[self.t]['stat_deltaq']     = self.deltaq
                self.cpx[self.t]['stat_t']          = self.t
                self.cpx[self.t]['stat_p_lcl_end']  = []
                self.cpx[self.t]['stat_T_lcl_end']  = []
        # Calculate virtual temperatures 
        self.thetav   = self.theta  + 0.61 * self.theta * self.q
        self.wthetav  = self.wtheta + 0.61 * self.theta * self.wq
        self.deltathetav  = (self.theta + self.deltatheta) * (1. + 0.61 * (self.q + self.deltaq)) - self.theta * (1. + 0.61 * self.q)

        # Mixed-layer top properties
        self.P_h    = self.Ps - self.rho * self.g * self.h
        self.T_h    = self.theta - self.g/self.cp * self.h

        #self.P_h    = self.Ps / np.exp((self.g * self.h)/(self.Rd * self.theta))
        #self.T_h    = self.theta / (self.Ps / self.P_h)**(self.Rd/self.cp)
        qsat_variable = qsat(self.T_h, self.P_h)
        self.RH_h   = self.q / qsat_variable

        # Find lifting condensation level iteratively
        if(self.t == 0):
            self.lcl = self.h
            RHlcl = 0.5
        else:
            RHlcl = 0.9998 

        itmax = 50 #Peter Bosman: I have increased this to from 30 to 50
        it = 0
        while(((RHlcl <= 0.9999) or (RHlcl >= 1.0001)) and it<itmax):
            self.lcl    += (1.-RHlcl)*1000.
            p_lcl        = self.Ps - self.rho * self.g * self.lcl
            T_lcl        = self.theta - self.g/self.cp * self.lcl
            RHlcl        = self.q / qsat(T_lcl, p_lcl)
            it          += 1
            if self.checkpoint:
                if call_from_init:
                    self.cpx_init[0]['stat_p_lcl_end'] += [p_lcl]
                    self.cpx_init[0]['stat_T_lcl_end'] += [T_lcl]
                else:
                    self.cpx[self.t]['stat_p_lcl_end'] += [p_lcl]
                    self.cpx[self.t]['stat_T_lcl_end'] += [T_lcl]

        if(it == itmax):
            if self.sw_printwarnings:
                print("LCL calculation not converged!!")
                print("RHlcl = %f, zlcl=%f"%(RHlcl, self.lcl))
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['stat_qsat_variable_end'] = qsat_variable
                self.cpx_init[0]['stat_T_h_end'] = self.T_h
                self.cpx_init[0]['stat_P_h_end'] = self.P_h
                self.cpx_init[0]['stat_it_end'] = it
            else:
                self.cpx[self.t]['stat_qsat_variable_end'] = qsat_variable
                self.cpx[self.t]['stat_T_h_end'] = self.T_h
                self.cpx[self.t]['stat_P_h_end'] = self.P_h
                self.cpx[self.t]['stat_it_end'] = it
            
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_stat.update({variablename: the_locals[variablename]})

    def run_cumulus(self,call_from_init=False):
        self.vars_rc= {}
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rc_wthetav']    = self.wthetav #subscript rc from run_cumulus
                self.cpx_init[0]['rc_deltaq']    = self.deltaq
                self.cpx_init[0]['rc_dz_h']    = self.dz_h
                self.cpx_init[0]['rc_wstar']    = self.wstar
                self.cpx_init[0]['rc_wqe']    = self.wqe
                self.cpx_init[0]['rc_wqM']    = self.wqM
                self.cpx_init[0]['rc_h']    = self.h
                self.cpx_init[0]['rc_deltaCO2']    = self.deltaCO2
                self.cpx_init[0]['rc_deltaCOS']    = self.deltaCOS
                self.cpx_init[0]['rc_wCO2e']    = self.wCO2e
                self.cpx_init[0]['rc_wCO2M']    = self.wCO2M
                self.cpx_init[0]['rc_wCOSe']    = self.wCOSe
                self.cpx_init[0]['rc_wCOSM']    = self.wCOSM
                self.cpx_init[0]['rc_q']    = self.q
                self.cpx_init[0]['rc_T_h']    = self.T_h
                self.cpx_init[0]['rc_P_h']    = self.P_h
            else:
                self.cpx[self.t]['rc_wthetav']    = self.wthetav #subscript rc from run_cumulus
                self.cpx[self.t]['rc_deltaq']    = self.deltaq
                self.cpx[self.t]['rc_dz_h']    = self.dz_h
                self.cpx[self.t]['rc_wstar']    = self.wstar
                self.cpx[self.t]['rc_wqe']    = self.wqe
                self.cpx[self.t]['rc_wqM']    = self.wqM
                self.cpx[self.t]['rc_h']    = self.h
                self.cpx[self.t]['rc_deltaCO2']    = self.deltaCO2
                self.cpx[self.t]['rc_deltaCOS']    = self.deltaCOS
                self.cpx[self.t]['rc_wCO2e']    = self.wCO2e
                self.cpx[self.t]['rc_wCO2M']    = self.wCO2M
                self.cpx[self.t]['rc_wCOSe']    = self.wCOSe
                self.cpx[self.t]['rc_wCOSM']    = self.wCOSM
                self.cpx[self.t]['rc_q']    = self.q
                self.cpx[self.t]['rc_T_h']    = self.T_h
                self.cpx[self.t]['rc_P_h']    = self.P_h
        # Calculate mixed-layer top relative humidity variance (Neggers et. al 2006/7)
        if(self.wthetav > 0):
            self.q2_h   = -(self.wqe  + self.wqM  ) * self.deltaq   * self.h / (self.dz_h * self.wstar)
            self.CO22_h = -(self.wCO2e+ self.wCO2M) * self.deltaCO2 * self.h / (self.dz_h * self.wstar)
            self.COS2_h = -(self.wCOSe+ self.wCOSM) * self.deltaCOS * self.h / (self.dz_h * self.wstar)
        else:
            self.q2_h   = 0. 
            self.CO22_h = 0.
            self.COS2_h = 0.
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rc_q2_h_middle']    = self.q2_h
                self.cpx_init[0]['rc_CO22_h_middle']    = self.CO22_h
                self.cpx_init[0]['rc_COS2_h_middle']    = self.COS2_h
            else:
                self.cpx[self.t]['rc_q2_h_middle']    = self.q2_h
                self.cpx[self.t]['rc_CO22_h_middle']    = self.CO22_h
                self.cpx[self.t]['rc_COS2_h_middle']    = self.COS2_h
        if self.q2_h <= 0.:
            self.q2_h   = 1.e-200 #This I (Peter Bosman) added to prevent problems with values becoming infinity, see below, that gives problems to adjoint.
        if self.CO22_h <= 0.:
            self.CO22_h   = 1.e-200 #This I (Peter Bosman) added to prevent problems with the derivative of self.wCO2M below, derivative not defined when self.CO22_h = 0
        if self.COS2_h <= 0.:
            self.COS2_h   = 1.e-200 #This I (Peter Bosman) added
        # calculate cloud core fraction (ac), mass flux (M) and moisture flux (wqM)
        qsat_variable_rc = qsat(self.T_h, self.P_h)
        self.ac     = max(0., 0.5 + (0.36 * np.arctan(1.55 * ((self.q - qsat_variable_rc) / self.q2_h**0.5))))
        #if self.q2_h   == 0., ((self.q - qsat_variable_rc) / self.q2_h**0.5) goes to infinity, but arctan (inf) exists (=pi/2). But problems for adjoint...
        self.M      = self.ac * self.wstar
        #note that if q - qsat > 0 and the variance of q is small, ac will be large and thus M large -> BL might shrink
        self.wqM    = self.M * self.q2_h**0.5

        # Only calculate CO2 mass-flux if mixed-layer top jump is negative
        if(self.deltaCO2 < 0):
            self.wCO2M  = self.M * self.CO22_h**0.5
        else:
            self.wCO2M  = 0.
        if(self.deltaCOS < 0):
            self.wCOSM  = self.M * self.COS2_h**0.5
        else:
            self.wCOSM  = 0.
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rc_q2_h_end']    = self.q2_h
                self.cpx_init[0]['rc_ac_end']    = self.ac
                self.cpx_init[0]['rc_M_end']    = self.M
                self.cpx_init[0]['rc_CO22_h_end']    = self.CO22_h
                self.cpx_init[0]['rc_COS2_h_end']    = self.COS2_h
                self.cpx_init[0]['rc_qsat_variable_rc_end'] = qsat_variable_rc
            else:
                self.cpx[self.t]['rc_q2_h_end']    = self.q2_h
                self.cpx[self.t]['rc_ac_end']    = self.ac
                self.cpx[self.t]['rc_M_end']    = self.M
                self.cpx[self.t]['rc_CO22_h_end']    = self.CO22_h
                self.cpx[self.t]['rc_COS2_h_end']    = self.COS2_h
                self.cpx[self.t]['rc_qsat_variable_rc_end'] = qsat_variable_rc
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rc.update({variablename: the_locals[variablename]})

    def run_mixed_layer(self,call_from_init=False):
        self.vars_rml= {}
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rml_h']    = self.h #subscript rml from run_mixed_layer
                self.cpx_init[0]['rml_ustar']    = self.ustar
                self.cpx_init[0]['rml_u']    = self.u
                self.cpx_init[0]['rml_v']    = self.v
                self.cpx_init[0]['rml_deltatheta']    = self.deltatheta
                self.cpx_init[0]['rml_deltathetav']    = self.deltathetav
                self.cpx_init[0]['rml_thetav']    = self.thetav
                self.cpx_init[0]['rml_wthetav']    = self.wthetav
                self.cpx_init[0]['rml_deltaq']    = self.deltaq
                self.cpx_init[0]['rml_deltaCO2']    = self.deltaCO2
                self.cpx_init[0]['rml_deltaCOS']    = self.deltaCOS
                self.cpx_init[0]['rml_deltau']    = self.deltau
                self.cpx_init[0]['rml_deltav']    = self.deltav
                self.cpx_init[0]['rml_wtheta']    = self.wtheta
                self.cpx_init[0]['rml_wq']    = self.wq
                self.cpx_init[0]['rml_wqM']    = self.wqM
                self.cpx_init[0]['rml_wCO2']    = self.wCO2
                self.cpx_init[0]['rml_wCO2M']    = self.wCO2M
                self.cpx_init[0]['rml_wCOS']    = self.wCOS
                self.cpx_init[0]['rml_wCOSM']    = self.wCOSM
                self.cpx_init[0]['rml_ac']    = self.ac
                self.cpx_init[0]['rml_lcl']    = self.lcl
                self.cpx_init[0]['rml_gammatheta']    = self.gammatheta
                self.cpx_init[0]['rml_gammatheta2']    = self.gammatheta2
                self.cpx_init[0]['rml_htrans']    = self.htrans
                self.cpx_init[0]['rml_gammaq']    = self.gammaq
                self.cpx_init[0]['rml_gammaCO2']    = self.gammaCO2
                self.cpx_init[0]['rml_gammaCOS']    = self.gammaCOS
                self.cpx_init[0]['rml_gammau']    = self.gammau
                self.cpx_init[0]['rml_gammav']    = self.gammav
                self.cpx_init[0]['rml_M']    = self.M
                self.cpx_init[0]['rml_divU']    = self.divU
                self.cpx_init[0]['rml_fc']    = self.fc
                self.cpx_init[0]['rml_dFz']    = self.dFz
            else:
                self.cpx[self.t]['rml_h']    = self.h #subscript rml from run_mixed_layer
                self.cpx[self.t]['rml_ustar']    = self.ustar
                self.cpx[self.t]['rml_u']    = self.u
                self.cpx[self.t]['rml_v']    = self.v
                self.cpx[self.t]['rml_deltatheta']    = self.deltatheta
                self.cpx[self.t]['rml_deltathetav']    = self.deltathetav
                self.cpx[self.t]['rml_thetav']    = self.thetav
                self.cpx[self.t]['rml_wthetav']    = self.wthetav
                self.cpx[self.t]['rml_deltaq']    = self.deltaq
                self.cpx[self.t]['rml_deltaCO2']    = self.deltaCO2
                self.cpx[self.t]['rml_deltaCOS']    = self.deltaCOS
                self.cpx[self.t]['rml_deltau']    = self.deltau
                self.cpx[self.t]['rml_deltav']    = self.deltav
                self.cpx[self.t]['rml_wtheta']    = self.wtheta
                self.cpx[self.t]['rml_wq']    = self.wq
                self.cpx[self.t]['rml_wqM']    = self.wqM
                self.cpx[self.t]['rml_wCO2']    = self.wCO2
                self.cpx[self.t]['rml_wCO2M']    = self.wCO2M
                self.cpx[self.t]['rml_wCOS']    = self.wCOS
                self.cpx[self.t]['rml_wCOSM']    = self.wCOSM
                self.cpx[self.t]['rml_ac']    = self.ac
                self.cpx[self.t]['rml_lcl']    = self.lcl
                self.cpx[self.t]['rml_gammatheta']    = self.gammatheta
                self.cpx[self.t]['rml_gammatheta2']    = self.gammatheta2
                self.cpx[self.t]['rml_htrans']    = self.htrans
                self.cpx[self.t]['rml_gammaq']    = self.gammaq
                self.cpx[self.t]['rml_gammaCO2']    = self.gammaCO2
                self.cpx[self.t]['rml_gammaCOS']    = self.gammaCOS
                self.cpx[self.t]['rml_gammau']    = self.gammau
                self.cpx[self.t]['rml_gammav']    = self.gammav
                self.cpx[self.t]['rml_M']    = self.M
                self.cpx[self.t]['rml_divU']    = self.divU
                self.cpx[self.t]['rml_fc']    = self.fc
                self.cpx[self.t]['rml_dFz']    = self.dFz
        if(not self.sw_sl):
            # decompose ustar along the wind components
            self.uw = - np.sign(self.u) * (self.ustar ** 4. / (self.v ** 2. / self.u ** 2. + 1.)) ** (0.5)
            self.vw = - np.sign(self.v) * (self.ustar ** 4. / (self.u ** 2. / self.v ** 2. + 1.)) ** (0.5)
      
        # calculate large-scale vertical velocity (subsidence)
        self.ws = -self.divU * self.h
      
        # calculate compensation to fix the free troposphere in case of subsidence 
        if(self.sw_fixft):
            if self.h <= self.htrans:
                w_th_ft  = self.gammatheta * self.ws
            else:
                w_th_ft  = self.gammatheta2 * self.ws
            w_q_ft   = self.gammaq     * self.ws
            w_CO2_ft = self.gammaCO2   * self.ws
            w_COS_ft = self.gammaCOS   * self.ws
        else:
            w_th_ft  = 0.
            w_q_ft   = 0.
            w_CO2_ft = 0. 
            w_COS_ft = 0. 
      
        # calculate mixed-layer growth due to cloud top radiative divergence
        self.wf = self.dFz / (self.rho * self.cp * self.deltatheta)
       
        # calculate convective velocity scale w* 
        if(self.wthetav > 0.):
            self.wstar = ((self.g * self.h * self.wthetav) / self.thetav)**(1./3.)
        else:
            self.wstar  = 1e-6;
      
        # Virtual heat entrainment flux 
        if self.sw_dyn_beta:
            self.beta = 0.2 + 5 * (self.ustar/self.wstar)**3. #see p11 of supplementary material of Vila et al 2012 (Nature paper)
        self.wthetave    = -self.beta * self.wthetav
        
        # compute mixed-layer tendencies
        if(self.sw_shearwe):
            self.we    = (-self.wthetave + 5. * self.ustar ** 3. * self.thetav / (self.g * self.h)) / self.deltathetav
        else:
            self.we    = -self.wthetave / self.deltathetav
#            time.sleep(0.1)
#            print('deltathetav')
#            print(self.deltathetav)
#            print('wthetave')
#            print(self.wthetave)
#            print('we')
#            print(self.we)

        # Don't allow boundary layer shrinking if wtheta < 0 
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rml_we_middle']    = self.we
            else:
                self.cpx[self.t]['rml_we_middle']    = self.we
        if(self.we < 0):
            self.we = 0.
        # Calculate entrainment fluxes
        self.wthetae     = -self.we * self.deltatheta
        self.wqe         = -self.we * self.deltaq
        self.wCO2e       = -self.we * self.deltaCO2
        self.wCOSe       = -self.we * self.deltaCOS
  
        self.htend       = self.we + self.ws + self.wf - self.M
       
        self.thetatend   = (self.wtheta - self.wthetae             ) / self.h + self.advtheta 
        self.qtend       = (self.wq     - self.wqe     - self.wqM  ) / self.h + self.advq
        self.CO2tend     = (self.wCO2   - self.wCO2e   - self.wCO2M) / self.h + self.advCO2
        self.COStend     = (self.wCOS   - self.wCOSe   - self.wCOSM) / self.h + self.advCOS
        
        if self.h <= self.htrans:
            self.deltathetatend  = self.gammatheta * (self.we + self.wf - self.M) - self.thetatend + w_th_ft
        else:
            self.deltathetatend  = self.gammatheta2 * (self.we + self.wf - self.M) - self.thetatend + w_th_ft
        self.deltaqtend      = self.gammaq     * (self.we + self.wf - self.M) - self.qtend     + w_q_ft #first term is the change in the value of q just above the BL, the second term is the change in the value of q in the ML
        self.deltaCO2tend    = self.gammaCO2   * (self.we + self.wf - self.M) - self.CO2tend   + w_CO2_ft
        self.deltaCOStend    = self.gammaCOS   * (self.we + self.wf - self.M) - self.COStend   + w_COS_ft
        
        if self.sw_advfp:
            self.deltathetatend += self.advtheta #this way advection cancels out for the jump tendencies, since advection is also added to the mixed layer tendencies 
            #(assumption is advection equal at all heights)
            self.deltaqtend += self.advq
            self.deltaCO2tend += self.advCO2
            self.deltaCOStend += self.advCOS
        
        # assume u + du = ug, so ug - u = du
        if(self.sw_wind):
            self.utend       = -self.fc * self.deltav + (self.uw + self.we * self.deltau)  / self.h + self.advu
            self.vtend       =  self.fc * self.deltau + (self.vw + self.we * self.deltav)  / self.h + self.advv
  
            self.deltautend      = self.gammau * (self.we + self.wf - self.M) - self.utend
            self.deltavtend      = self.gammav * (self.we + self.wf - self.M) - self.vtend
            if self.sw_advfp:
                self.deltautend += self.advu
                self.deltavtend += self.advv
        # tendency of the transition layer thickness
        if(self.ac > 0 or self.lcl - self.h < 300):
            self.dztend = ((self.lcl - self.h)-self.dz_h) / 7200.
        else:
            self.dztend = 0.
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rml_we_end']    = self.we
                self.cpx_init[0]['rml_uw_end']    = self.uw
                self.cpx_init[0]['rml_vw_end']    = self.vw
                self.cpx_init[0]['rml_wthetave_end']    = self.wthetave
                self.cpx_init[0]['rml_wthetae_end']    = self.wthetae
                self.cpx_init[0]['rml_wqe_end']    = self.wqe
                self.cpx_init[0]['rml_wCO2e_end']    = self.wCO2e
                self.cpx_init[0]['rml_wCOSe_end']    = self.wCOSe
                self.cpx_init[0]['rml_ws_end']    = self.ws
                self.cpx_init[0]['rml_wf_end']    = self.wf
                self.cpx_init[0]['rml_wstar_end']    = self.wstar
                self.cpx_init[0]['rml_beta_end']    = self.beta
            else:
                self.cpx[self.t]['rml_we_end']    = self.we
                self.cpx[self.t]['rml_uw_end']    = self.uw
                self.cpx[self.t]['rml_vw_end']    = self.vw
                self.cpx[self.t]['rml_wthetave_end']    = self.wthetave
                self.cpx[self.t]['rml_wthetae_end']    = self.wthetae
                self.cpx[self.t]['rml_wqe_end']    = self.wqe
                self.cpx[self.t]['rml_wCO2e_end']    = self.wCO2e
                self.cpx[self.t]['rml_wCOSe_end']    = self.wCOSe
                self.cpx[self.t]['rml_ws_end']    = self.ws
                self.cpx[self.t]['rml_wf_end']    = self.wf
                self.cpx[self.t]['rml_wstar_end']    = self.wstar
                self.cpx[self.t]['rml_beta_end']    = self.beta #it depends on a switch wether beta is calculated in this module, or obtained from input, 
                #than the use of end in the name is somewhat inconsistent, but not important...
            
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rml.update({variablename: the_locals[variablename]})
                #self.vars_rml.update({variablename: the_locals[variablename]})

               
    def integrate_mixed_layer(self):
        self.vars_iml= {}
        # set values previous time step
        h0      = self.h
        
        theta0  = self.theta
        deltatheta0 = self.deltatheta
        q0      = self.q
        deltaq0     = self.deltaq
        CO20    = self.CO2
        COS0    = self.COS
        deltaCO20   = self.deltaCO2
        deltaCOS0   = self.deltaCOS
        
        u0      = self.u
        deltau0 = self.deltau
        v0      = self.v
        deltav0 = self.deltav

        dz0     = self.dz_h
  
        # integrate mixed-layer equations
        self.h        = h0      + self.dt * self.htend
        self.theta    = theta0  + self.dt * self.thetatend
        self.deltatheta   = deltatheta0 + self.dt * self.deltathetatend
        self.q        = q0      + self.dt * self.qtend
        self.deltaq       = deltaq0     + self.dt * self.deltaqtend
        self.CO2      = CO20    + self.dt * self.CO2tend
        self.COS      = COS0    + self.dt * self.COStend
        self.deltaCO2     = deltaCO20   + self.dt * self.deltaCO2tend
        self.deltaCOS     = deltaCOS0   + self.dt * self.deltaCOStend
        self.dz_h     = dz0     + self.dt * self.dztend

        # Limit dz to minimal value
        dz0 = 50
        if self.checkpoint:    
            self.cpx[self.t]['iml_dz_h_middle']    = self.dz_h
        if(self.dz_h < dz0):
            self.dz_h = dz0 
  
        if(self.sw_wind):
            self.u        = u0      + self.dt * self.utend
            self.deltau   = deltau0     + self.dt * self.deltautend
            self.v        = v0      + self.dt * self.vtend
            self.deltav   = deltav0     + self.dt * self.deltavtend
            
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_iml.update({variablename: the_locals[variablename]})
 
    def run_radiation(self,call_from_init=False):
        self.vars_rr = {} #a dictionary of all defined variables for testing purposes
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rr_doy']    = self.doy #subscript rr from run_radiation
                self.cpx_init[0]['rr_lat']    = self.lat
                self.cpx_init[0]['rr_h']    = self.h
                self.cpx_init[0]['rr_theta']    = self.theta
                self.cpx_init[0]['rr_cc']    = self.cc
                self.cpx_init[0]['rr_alpha']    = self.alpha
                self.cpx_init[0]['rr_Ts']    = self.Ts
            else:
                self.cpx[self.t]['rr_doy']    = self.doy #subscript rr from run_radiation
                self.cpx[self.t]['rr_lat']    = self.lat
                self.cpx[self.t]['rr_h']    = self.h
                self.cpx[self.t]['rr_theta']    = self.theta
                self.cpx[self.t]['rr_cc']    = self.cc
                self.cpx[self.t]['rr_alpha']    = self.alpha
                self.cpx[self.t]['rr_Ts']    = self.Ts
        sda    = 0.409 * np.cos(2. * np.pi * (self.doy - 173.) / 365.)
        part1_sinlea = np.sin(2. * np.pi * self.lat / 360.) * np.sin(sda)
        sinlea_constant = np.cos(2. * np.pi * (self.t * self.dt + self.tstart * 3600.) / 86400. + 2. * np.pi * self.lon / 360.)
        part2_sinlea = np.cos(2. * np.pi * self.lat / 360.) * np.cos(sda) * sinlea_constant        
        sinlea = part1_sinlea - part2_sinlea
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rr_sinlea_middle']    = sinlea
            else:
                self.cpx[self.t]['rr_sinlea_middle']    = sinlea
        if sinlea < 0.0001:
            if self.sw_printwarnings:
                print('Warning, very low solar angle!!')
        sinlea = max(sinlea, 0.0001)
        self.sinlea = sinlea #might be needed by canopy model as well
        Ta  = self.theta * ((self.Ps - 0.1 * self.h * self.rho * self.g) / self.Ps ) ** (self.Rd / self.cp)
  
        Tr  = (0.6 + 0.2 * sinlea) * (1. - 0.4 * self.cc)
  
        self.Swin  = self.S0 * Tr * sinlea
        self.Swout = self.alpha * self.S0 * Tr * sinlea
        self.Lwin  = 0.8 * self.bolz * Ta ** 4.
        self.Lwout = self.bolz * self.Ts ** 4.
          
        self.Q     = self.Swin - self.Swout + self.Lwin - self.Lwout
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rr_sda_end']    = sda
                self.cpx_init[0]['rr_sinlea_constant_end']    = sinlea_constant
                self.cpx_init[0]['rr_sinlea_end']    = sinlea
                self.cpx_init[0]['rr_Tr_end']    = Tr
                self.cpx_init[0]['rr_Ta_end']    = Ta
            else:
                self.cpx[self.t]['rr_sda_end']    = sda
                self.cpx[self.t]['rr_sinlea_constant_end']    = sinlea_constant
                self.cpx[self.t]['rr_sinlea_end']    = sinlea
                self.cpx[self.t]['rr_Tr_end']    = Tr
                self.cpx[self.t]['rr_Ta_end']    = Ta
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rr.update({variablename: the_locals[variablename]})
        
  
    def run_surface_layer(self,iterationnumber=0,call_from_init=False):
        self.vars_rsl = {} #a dictionary of all defined variables for testing purposes
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[iterationnumber]['rsl_Cs']    = self.Cs 
                self.cpx_init[iterationnumber]['rsl_ustar']    = self.ustar
                self.cpx_init[iterationnumber]['rsl_q']    = self.q
                self.cpx_init[iterationnumber]['rsl_wq']    = self.wq
                self.cpx_init[iterationnumber]['rsl_wCOS']    = self.wCOS
                self.cpx_init[iterationnumber]['rsl_wCO2']    = self.wCO2
                self.cpx_init[iterationnumber]['rsl_theta']    = self.theta
                self.cpx_init[iterationnumber]['rsl_h']    = self.h
                self.cpx_init[iterationnumber]['rsl_thetav']    = self.thetav
                self.cpx_init[iterationnumber]['rsl_wthetav']    = self.wthetav
                self.cpx_init[iterationnumber]['rsl_wtheta']    = self.wtheta
                self.cpx_init[iterationnumber]['rsl_rs']    = self.rs
                self.cpx_init[iterationnumber]['rsl_u']    = self.u
                self.cpx_init[iterationnumber]['rsl_v']    = self.v
                self.cpx_init[iterationnumber]['rsl_wstar']    = self.wstar
                self.cpx_init[iterationnumber]['rsl_COSmeasuring_height']    = self.COSmeasuring_height
                self.cpx_init[iterationnumber]['rsl_z0m']    = self.z0m
                self.cpx_init[iterationnumber]['rsl_z0h']    = self.z0h
            else:
                self.cpx[self.t]['rsl_Cs']    = self.Cs #subscript rsl from run_surface_layer
                self.cpx[self.t]['rsl_ustar']    = self.ustar
                self.cpx[self.t]['rsl_q']    = self.q
                self.cpx[self.t]['rsl_wq']    = self.wq
                self.cpx[self.t]['rsl_wCOS']    = self.wCOS
                self.cpx[self.t]['rsl_wCO2']    = self.wCO2
                self.cpx[self.t]['rsl_theta']    = self.theta
                self.cpx[self.t]['rsl_h']    = self.h
                self.cpx[self.t]['rsl_thetav']    = self.thetav
                self.cpx[self.t]['rsl_wthetav']    = self.wthetav
                self.cpx[self.t]['rsl_wtheta']    = self.wtheta
                self.cpx[self.t]['rsl_rs']    = self.rs
                self.cpx[self.t]['rsl_u']    = self.u
                self.cpx[self.t]['rsl_v']    = self.v
                self.cpx[self.t]['rsl_wstar']    = self.wstar
                self.cpx[self.t]['rsl_COSmeasuring_height']    = self.COSmeasuring_height
                self.cpx[self.t]['rsl_z0m']    = self.z0m
                self.cpx[self.t]['rsl_z0h']    = self.z0h
        
        ueff           = max(0.01, np.sqrt(self.u**2. + self.v**2. + self.wstar**2.))#length of wind vector (m s-1)
        if ueff < 0.01:
            if self.sw_printwarnings:
                print('ueff < 0.01!')
        self.COSsurf = self.COS + self.wCOS / (self.Cs * ueff) #3.45,3.43 rewritten using COS (wCOS = (COSsurf - COS)/ra rewritten) and self.ra = (self.Cs * ueff)**-1., see land surface module 
#        print('forwm cossurf')
#        print(self.COSsurf )
#        print('forwm cosML')
#        print(self.COS )
        self.CO2surf = self.CO2 + self.wCO2 / (self.Cs * ueff) #3.45,3.43 rewritten using CO2
        self.thetasurf = self.theta + self.wtheta / (self.Cs * ueff) #3.45,3.43 rewritten
        self.Tsurf = self.thetasurf * (100000/self.Ps)**(-self.Rd/self.cp) #convert pot temp to temp
#   Below the original, problematic way of calculating qsurf. see Notes 3 
#        qsatsurf_rsl       = qsat(self.thetasurf, self.Ps) #qsatsurf also calculated in land_surface module!! rsl means run surface layer
#        cq             = (1. + self.Cs * ueff * self.rs) ** -1.
#        self.qsurf     = (1. - cq) * self.q + cq * qsatsurf_rsl
        self.qsurf     = self.q + self.wq / (self.Cs * ueff)#3.45,3.43 rewritten
        self.esurf = self.qsurf * self.Ps / 0.622
        self.thetavsurf = self.thetasurf * (1. + 0.61 * self.qsurf) #0.61 is 1-Rw/Rd. eq 59 microhh reference paper
        
        self.zsl       = 0.1 * self.h #surface layer 10% of boundary layer
        if self.sw_use_ribtol:
            self.Rib  = self.g / self.thetav * self.zsl * (self.thetav - self.thetavsurf) / ueff**2.
            if self.checkpoint:
                if call_from_init:
                    self.cpx_init[iterationnumber]['rsl_Rib_middle']    = self.Rib
                else:
                    self.cpx[self.t]['rsl_Rib_middle']    = self.Rib
            self.Rib  = min(self.Rib, 0.2)
            self.L     = self.ribtol(self.Rib, self.zsl, self.z0m, self.z0h, iterationnumber,call_from_init)
        else: #more simple
            self.L     = self.thetav * self.ustar**3 /(self.k * self.g * -1 * self.wthetav)
            if self.wthetav ==0 :
                if self.sw_printwarnings:
                    print('zero virt temp flux')
        self.Cm   = self.k**2. / (np.log(self.zsl / self.z0m) - self.psim(self.zsl / self.L) + self.psim(self.z0m / self.L)) ** 2. #eq 3.46 AVSI, height zu is taken at top of surface layer
        Cs   = self.k**2. / (np.log(self.zsl / self.z0m) - self.psim(self.zsl / self.L) + self.psim(self.z0m / self.L)) / (np.log(self.zsl / self.z0h) - self.psih(self.zsl / self.L) + self.psih(self.z0h / self.L)) #eq 3.46 AVSI, height ztheta is taken at top of surface layer
        ustar = np.sqrt(self.Cm) * ueff #3.45
        if self.updatevals_surf_lay:#Peter Bosman: this switch and the se of a self and a non-self variable was added by me
            self.Cs   = Cs
            self.ustar = ustar
        
        self.uw    = - self.Cm * ueff * self.u #3.45 with 3.17 (mind that u is total horizontal wind in 3.17): -u*w = tau_x/rho = Cm*ueff*u (x momentum transported down) ? 
        #or can be seen by looking at 3.43 using ram = 1 /(ueff * Cm), see notes next to eq 3.45, so: uw = tau_x/rho = - (u(zu) - u(surf))/ram = - Cm * ueff * self.u, given that u(surf) = 0
        self.vw    = - self.Cm * ueff * self.v
     
        # diagnostic meteorological variables
        #never calculate a variable at a height lower than its roughness length, e.g. never calculate qmh at a height lower than z0h, or a wind variable below a height z0m. thetasurf is the value of theta at height z0h
        self.T2m    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(2. / self.z0h) - self.psih(2. / self.L) + self.psih(self.z0h / self.L)) #3.42 with 3.17
        self.thetamh    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height / self.z0h) - self.psih(self.Tmeasuring_height / self.L) + self.psih(self.z0h / self.L))
        self.thetamh2    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height2 / self.z0h) - self.psih(self.Tmeasuring_height2 / self.L) + self.psih(self.z0h / self.L))
        self.thetamh3    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height3 / self.z0h) - self.psih(self.Tmeasuring_height3 / self.L) + self.psih(self.z0h / self.L))
        self.thetamh4    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height4 / self.z0h) - self.psih(self.Tmeasuring_height4 / self.L) + self.psih(self.z0h / self.L))
        self.thetamh5    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height5 / self.z0h) - self.psih(self.Tmeasuring_height5 / self.L) + self.psih(self.z0h / self.L))
        self.thetamh6    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height6 / self.z0h) - self.psih(self.Tmeasuring_height6 / self.L) + self.psih(self.z0h / self.L))
        self.thetamh7    = self.thetasurf - self.wtheta / ustar / self.k * (np.log(self.Tmeasuring_height7 / self.z0h) - self.psih(self.Tmeasuring_height7 / self.L) + self.psih(self.z0h / self.L))
        self.Tmh = self.thetamh * ((self.Ps - self.rho * self.g * self.Tmeasuring_height) / 100000)**(self.Rd/self.cp)
        self.Tmh2 = self.thetamh2 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height2) / 100000)**(self.Rd/self.cp)
        self.Tmh3 = self.thetamh3 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height3) / 100000)**(self.Rd/self.cp)
        self.Tmh4 = self.thetamh4 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height4) / 100000)**(self.Rd/self.cp)
        self.Tmh5 = self.thetamh5 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height5) / 100000)**(self.Rd/self.cp)
        self.Tmh6 = self.thetamh6 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height6) / 100000)**(self.Rd/self.cp)
        self.Tmh7 = self.thetamh7 * ((self.Ps - self.rho * self.g * self.Tmeasuring_height7) / 100000)**(self.Rd/self.cp)
        self.q2m    = self.qsurf     - self.wq     / ustar / self.k * (np.log(2. / self.z0h) - self.psih(2. / self.L) + self.psih(self.z0h / self.L))
        self.qmh    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height / self.z0h) - self.psih(self.qmeasuring_height / self.L) + self.psih(self.z0h / self.L))
        self.qmh2    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height2 / self.z0h) - self.psih(self.qmeasuring_height2 / self.L) + self.psih(self.z0h / self.L))
        self.qmh3    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height3 / self.z0h) - self.psih(self.qmeasuring_height3 / self.L) + self.psih(self.z0h / self.L))
        self.qmh4    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height4 / self.z0h) - self.psih(self.qmeasuring_height4 / self.L) + self.psih(self.z0h / self.L))
        self.qmh5    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height5 / self.z0h) - self.psih(self.qmeasuring_height5 / self.L) + self.psih(self.z0h / self.L))
        self.qmh6    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height6 / self.z0h) - self.psih(self.qmeasuring_height6 / self.L) + self.psih(self.z0h / self.L))
        self.qmh7    = self.qsurf     - self.wq     / ustar / self.k * (np.log(self.qmeasuring_height7 / self.z0h) - self.psih(self.qmeasuring_height7 / self.L) + self.psih(self.z0h / self.L))
        self.COS2m =      self.COSsurf - self.wCOS / ustar / self.k * (np.log(2. / self.z0h) - self.psih(2. / self.L) + self.psih(self.z0h / self.L))
        self.COSmh =     self.COSsurf - self.wCOS / ustar / self.k * (np.log(self.COSmeasuring_height / self.z0h) - self.psih(self.COSmeasuring_height / self.L) + self.psih(self.z0h / self.L))
        self.COSmh2 =     self.COSsurf - self.wCOS / ustar / self.k * (np.log(self.COSmeasuring_height2 / self.z0h) - self.psih(self.COSmeasuring_height2 / self.L) + self.psih(self.z0h / self.L))
        self.COSmh3 =     self.COSsurf - self.wCOS / ustar / self.k * (np.log(self.COSmeasuring_height3 / self.z0h) - self.psih(self.COSmeasuring_height3 / self.L) + self.psih(self.z0h / self.L))
        self.CO22m =      self.CO2surf - self.wCO2 / ustar / self.k * (np.log(2. / self.z0h) - self.psih(2. / self.L) + self.psih(self.z0h / self.L))
        self.CO2mh =     self.CO2surf - self.wCO2 / ustar / self.k * (np.log(self.CO2measuring_height / self.z0h) - self.psih(self.CO2measuring_height / self.L) + self.psih(self.z0h / self.L))
        self.CO2mh2 =     self.CO2surf - self.wCO2 / ustar / self.k * (np.log(self.CO2measuring_height2 / self.z0h) - self.psih(self.CO2measuring_height2 / self.L) + self.psih(self.z0h / self.L))
        self.CO2mh3 =     self.CO2surf - self.wCO2 / ustar / self.k * (np.log(self.CO2measuring_height3 / self.z0h) - self.psih(self.CO2measuring_height3 / self.L) + self.psih(self.z0h / self.L)) #CO2 mixing ratio at measuring height 3
        self.CO2mh4 =     self.CO2surf - self.wCO2 / ustar / self.k * (np.log(self.CO2measuring_height4 / self.z0h) - self.psih(self.CO2measuring_height4 / self.L) + self.psih(self.z0h / self.L)) #CO2 mixing ratio at measuring height 4
        self.u2m    = - self.uw     / ustar / self.k * (np.log(2. / self.z0m) - self.psim(2. / self.L) + self.psim(self.z0m / self.L)) #3.42 with 3.17 (-uw = ustar**2 but u is total wind in AVSI...)
        self.v2m    = - self.vw     / ustar / self.k * (np.log(2. / self.z0m) - self.psim(2. / self.L) + self.psim(self.z0m / self.L)) #vstar?? ustar cannot be equal to both uw and vw!!
        self.esat2m = 0.611e3 * np.exp(17.2694 * (self.T2m - 273.16) / (self.T2m - 35.86)) #intro atm
        self.e2m    = self.q2m * self.Ps / 0.622 #intro atm
        
        
        #print(self.T2m,self.u2m)
        if self.sw_dynamicsl_border: #use mixed layer values if height is above the surface layer
            if self.Tmeasuring_height > self.zsl:
                self.thetamh = self.theta
                self.Tmh = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height2 > self.zsl:
                self.thetamh2 = self.theta
                self.Tmh2 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height2) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height3 > self.zsl:
                self.thetamh3 = self.theta
                self.Tmh3 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height3) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height4 > self.zsl:
                self.thetamh4 = self.theta
                self.Tmh4 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height4) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height5 > self.zsl:
                self.thetamh5 = self.theta
                self.Tmh5 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height5) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height6 > self.zsl:
                self.thetamh6 = self.theta
                self.Tmh6 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height6) / 100000)**(self.Rd/self.cp)
            if self.Tmeasuring_height7 > self.zsl:
                self.thetamh7 = self.theta
                self.Tmh7 = self.theta * ((self.Ps - self.rho * self.g * self.Tmeasuring_height7) / 100000)**(self.Rd/self.cp)     
            if self.qmeasuring_height > self.zsl:
                self.qmh = self.q
            if self.qmeasuring_height2 > self.zsl:
                self.qmh2 = self.q
            if self.qmeasuring_height3 > self.zsl:
                self.qmh3 = self.q
            if self.qmeasuring_height4 > self.zsl:
                self.qmh4 = self.q
            if self.qmeasuring_height5 > self.zsl:
                self.qmh5 = self.q
            if self.qmeasuring_height6 > self.zsl:
                self.qmh6 = self.q
            if self.qmeasuring_height7 > self.zsl:
                self.qmh7 = self.q
            if self.COSmeasuring_height > self.zsl:
                self.COSmh = self.COS
            if self.COSmeasuring_height2 > self.zsl:
                self.COSmh2 = self.COS
            if self.COSmeasuring_height3 > self.zsl:
                self.COSmh3 = self.COS
            if self.CO2measuring_height > self.zsl:
                self.CO2mh = self.CO2
            if self.CO2measuring_height2 > self.zsl:
                self.CO2mh2 = self.CO2
            if self.CO2measuring_height3 > self.zsl:
                self.CO2mh3 = self.CO2
            if self.CO2measuring_height4 > self.zsl:
                self.CO2mh4 = self.CO2
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[iterationnumber]['rsl_ueff_end'] = ueff
                self.cpx_init[iterationnumber]['rsl_qsurf_end']    = self.qsurf
                self.cpx_init[iterationnumber]['rsl_thetasurf_end'] = self.thetasurf
                self.cpx_init[iterationnumber]['rsl_zsl_end']    = self.zsl
                self.cpx_init[iterationnumber]['rsl_L_end']    = self.L
                self.cpx_init[iterationnumber]['rsl_Cm_end']    = self.Cm
                self.cpx_init[iterationnumber]['rsl_Cs_end']    = Cs
                self.cpx_init[iterationnumber]['rsl_ustar_end']    = ustar
                self.cpx_init[iterationnumber]['rsl_uw_end']    = self.uw
                self.cpx_init[iterationnumber]['rsl_vw_end']    = self.vw
                self.cpx_init[iterationnumber]['rsl_T2m_end']    = self.T2m
                self.cpx_init[iterationnumber]['rsl_thetavsurf_end']    = self.thetavsurf
            else:
                self.cpx[self.t]['rsl_ueff_end'] = ueff
                self.cpx[self.t]['rsl_qsurf_end']    = self.qsurf
                self.cpx[self.t]['rsl_thetasurf_end'] = self.thetasurf
                self.cpx[self.t]['rsl_zsl_end']    = self.zsl
                self.cpx[self.t]['rsl_L_end']    = self.L
                self.cpx[self.t]['rsl_Cm_end']    = self.Cm
                self.cpx[self.t]['rsl_Cs_end']    = Cs
                self.cpx[self.t]['rsl_ustar_end']    = ustar
                self.cpx[self.t]['rsl_uw_end']    = self.uw
                self.cpx[self.t]['rsl_vw_end']    = self.vw
                self.cpx[self.t]['rsl_T2m_end']    = self.T2m
                self.cpx[self.t]['rsl_thetavsurf_end']    = self.thetavsurf
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rsl.update({variablename: the_locals[variablename]})
    
    def ribtol(self, Rib, zsl, z0m, z0h, iterationnumber=0,call_from_init=False):
        #this module returns L, it's more of a submodule
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[iterationnumber]['rtl_zsl'] = zsl
                self.cpx_init[iterationnumber]['rtl_z0m'] = z0m
                self.cpx_init[iterationnumber]['rtl_z0h'] = z0h
                self.cpx_init[iterationnumber]['rtl_Lstart_end'] = []
                self.cpx_init[iterationnumber]['rtl_Lend_end'] = []
                self.cpx_init[iterationnumber]['rtl_fxdif_part1_end'] = []
                self.cpx_init[iterationnumber]['rtl_fxdif_part2_end'] = []
                self.cpx_init[iterationnumber]['rtl_fx_end'] = []
                self.cpx_init[iterationnumber]['rtl_fxdif_end'] = []
                self.cpx_init[iterationnumber]['rtl_L_middle'] = []
            else:
                self.cpx[self.t]['rtl_zsl'] = zsl
                self.cpx[self.t]['rtl_z0m'] = z0m
                self.cpx[self.t]['rtl_z0h'] = z0h
                self.cpx[self.t]['rtl_Lstart_end'] = []
                self.cpx[self.t]['rtl_Lend_end'] = []
                self.cpx[self.t]['rtl_fxdif_part1_end'] = []
                self.cpx[self.t]['rtl_fxdif_part2_end'] = []
                self.cpx[self.t]['rtl_fx_end'] = []
                self.cpx[self.t]['rtl_fxdif_end'] = []
                self.cpx[self.t]['rtl_L_middle'] = []
        
        self.vars_rtl = {} #a dictionary of all defined variables for testing purposes
        if(Rib > 0.):
            L    = 1.
            L0   = 2.
        else:
            L  = -1.
            L0 = -2.
        it = 0
        while (abs(L - L0) > 0.001):
            if self.checkpoint:
                if call_from_init:
                    self.cpx_init[iterationnumber]['rtl_L_middle'] += [L] #middle since it is redefined later in the loop
                else:
                    self.cpx[self.t]['rtl_L_middle'] += [L]
            L0      = L
            fx      = Rib - zsl / L * (np.log(zsl / z0h) - self.psih(zsl / L) + self.psih(z0h / L)) / (np.log(zsl / z0m) - self.psim(zsl / L) + self.psim(z0m / L))**2.
            Lstart  = L - 0.001*L
            Lend    = L + 0.001*L
#            fxdif   = ( (- zsl / Lstart * (np.log(zsl / z0h) - self.psih(zsl / Lstart) + self.psih(z0h / Lstart)) / \
#                                          (np.log(zsl / z0m) - self.psim(zsl / Lstart) + self.psim(z0m / Lstart))**2.) \
#                      - (-zsl /  Lend   * (np.log(zsl / z0h) - self.psih(zsl / Lend  ) + self.psih(z0h / Lend  )) / \
#                                          (np.log(zsl / z0m) - self.psim(zsl / Lend  ) + self.psim(z0m / Lend  ))**2.) ) / (Lstart - Lend)
            fxdif_part1 = (- zsl / Lstart * (np.log(zsl / z0h) - self.psih(zsl / Lstart) + self.psih(z0h / Lstart)) / \
                                          (np.log(zsl / z0m) - self.psim(zsl / Lstart) + self.psim(z0m / Lstart))**2.) 
            fxdif_part2 = - (-zsl /  Lend   * (np.log(zsl / z0h) - self.psih(zsl / Lend  ) + self.psih(z0h / Lend  )) / \
                                          (np.log(zsl / z0m) - self.psim(zsl / Lend  ) + self.psim(z0m / Lend  ))**2.)
            fxdif = (fxdif_part1 + fxdif_part2) / (Lstart - Lend)
            L       = L - fx / fxdif
            if self.checkpoint:
                if call_from_init:
                    self.cpx_init[iterationnumber]['rtl_Lstart_end'] += [Lstart]
                    self.cpx_init[iterationnumber]['rtl_Lend_end'] += [Lend]
                    self.cpx_init[iterationnumber]['rtl_fxdif_part1_end'] += [fxdif_part1]
                    self.cpx_init[iterationnumber]['rtl_fxdif_part2_end'] += [fxdif_part2]
                    self.cpx_init[iterationnumber]['rtl_fx_end'] += [fx]
                    self.cpx_init[iterationnumber]['rtl_fxdif_end'] += [fxdif]
                else:
                    self.cpx[self.t]['rtl_Lstart_end'] += [Lstart]
                    self.cpx[self.t]['rtl_Lend_end'] += [Lend]
                    self.cpx[self.t]['rtl_fxdif_part1_end'] += [fxdif_part1]
                    self.cpx[self.t]['rtl_fxdif_part2_end'] += [fxdif_part2]
                    self.cpx[self.t]['rtl_fx_end'] += [fx]
                    self.cpx[self.t]['rtl_fxdif_end'] += [fxdif]
            it += 1
            if(abs(L) > 1e15):
                break
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[iterationnumber]['rtl_it_end'] = it #rsl since it will only be called from run surface layer
            else:
                self.cpx[self.t]['rtl_it_end'] = it
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rtl.update({variablename: the_locals[variablename]})
        return L
    
    def psim(self, zeta):
        if(zeta <= 0):
            if self.sw_useWilson:
                x     = (1. + 3.6 * (-1*zeta) ** (2./3.)) ** (-0.5)
                psim = 3. * np.log( (1. + 1. / x) / 2.)
            else: #businger-Dyer
                x     = (1. - 16. * zeta)**(0.25)
                psim  = 3.14159265 / 2. - 2. * np.arctan(x) + np.log((1. + x)**2. * (1. + x**2.) / 8.)
                #Wilson eq7, avsi eq 3.30 (log (ab) = log(a)+log(b)) and 2log(a) = log(a**2)
        else:
            if self.sw_model_stable_con:
                psim  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - zeta - (10./3.) / 0.35
            else:
                psim = np.nan
                if self.sw_printwarnings:
                    print("stable conditions cannot be modelled,returning nan")
        return psim
      
    def psih(self, zeta):
        if(zeta <= 0):
            if self.sw_useWilson:
                x     = (1. + 7.9 * (-1*zeta) ** (2./3.)) ** (-0.5)
                psih  = 3. * np.log( (1. + 1. / x) / 2.)
            else:
                x     = (1. - 16. * zeta)**(0.25)
                psih  = 2. * np.log( (1. + x*x) / 2.)
        else:
            if self.sw_model_stable_con:
                psih  = -2./3. * (zeta - 5./0.35) * np.exp(-0.35 * zeta) - (1. + (2./3.) * zeta) ** (1.5) - (10./3.) / 0.35 + 1.
                #eq 32 paper Beljaars and Holtslag 1990
            else:
                psih = np.nan 
                if self.sw_printwarnings:
                    print("stable conditions cannot be modelled,returning nan")
        return psih
 
    def jarvis_stewart(self,call_from_init=False):
        # calculate surface resistances using Jarvis-Stewart model
        self.vars_js = {} #a dictionary of all defined variables for testing purposes 
        if self.checkpoint: 
            if call_from_init:
                self.cpx_init[0]['js_Swin'] = self.Swin
                self.cpx_init[0]['js_w2'] = self.w2
                self.cpx_init[0]['js_wwilt'] = self.wwilt
                self.cpx_init[0]['js_wfc'] = self.wfc
                self.cpx_init[0]['js_gD'] = self.gD
                self.cpx_init[0]['js_e'] = self.e
                self.cpx_init[0]['js_esatvar'] = self.esatvar
                self.cpx_init[0]['js_theta'] = self.theta
                self.cpx_init[0]['js_LAI'] = self.LAI
                self.cpx_init[0]['js_rsmin'] = self.rsmin
            else:
                self.cpx[self.t]['js_Swin'] = self.Swin
                self.cpx[self.t]['js_w2'] = self.w2
                self.cpx[self.t]['js_wwilt'] = self.wwilt
                self.cpx[self.t]['js_wfc'] = self.wfc
                self.cpx[self.t]['js_gD'] = self.gD
                self.cpx[self.t]['js_e'] = self.e
                self.cpx[self.t]['js_esatvar'] = self.esatvar
                self.cpx[self.t]['js_theta'] = self.theta
                self.cpx[self.t]['js_LAI'] = self.LAI
                self.cpx[self.t]['js_rsmin'] = self.rsmin       
        if(self.sw_rad):
            f1 = 1. / min(1.,((0.004 * self.Swin + 0.05) / (0.81 * (0.004 * self.Swin + 1.))))
        else:
            f1 = 1.
  
        if(self.w2 > self.wwilt):# and self.w2 <= self.wfc):
            f2js = (self.wfc - self.wwilt) / (self.w2 - self.wwilt)  #•Peter Bosman: I have changed variable f2 into f2js, to prevent two vars with the same name
        else:
            f2js = 1.e8
 
        # Limit f2js in case w2 > wfc, where f2js < 1
        if self.checkpoint: 
            if call_from_init:
                self.cpx_init[0]['js_f2js_middle'] = f2js
            else:
                self.cpx[self.t]['js_f2js_middle'] = f2js
        f2js = max(f2js, 1.);
 
        f3 = 1. / np.exp(- self.gD * (self.esatvar - self.e) / 100.) #Peter Bosman: I have adapted self.esat into self.esatvar
        f4 = 1./ (1. - 0.0016 * (298.0-self.theta)**2.)
  
        self.rs = self.rsmin / self.LAI * f1 * f2js * f3 * f4
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['js_f1_end'] = f1
                self.cpx_init[0]['js_f2js_end'] = f2js
                self.cpx_init[0]['js_f3_end'] = f3
                self.cpx_init[0]['js_f4_end'] = f4 
            else:
                self.cpx[self.t]['js_f1_end'] = f1
                self.cpx[self.t]['js_f2js_end'] = f2js
                self.cpx[self.t]['js_f3_end'] = f3
                self.cpx[self.t]['js_f4_end'] = f4 
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_js.update({variablename: the_locals[variablename]})

    def factorial(self,k):
        factorial = 1
        for n in range(2,k+1):
            factorial = factorial * float(n)
        return factorial;

    def E1(self,x):
        E1sum = 0
        for k in range(1,100):
            E1sum += pow((-1.),(k + 0.0)) * pow(x,(k + 0.0)) / ((k + 0.0) * self.factorial(k))
        return -0.57721566490153286060 - np.log(x) - E1sum
 
    def ags(self,call_from_init=False): #For more info on where the equations come from, check a-gs in the canopy model
        self.vars_ags = {} #a dictionary of all defined variables for testing purposes
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['ags_COS'] = self.COS
                if self.ags_C_mode == 'surf':
                    self.cpx_init[0]['ags_COSsurf'] = self.COSsurf
                self.cpx_init[0]['ags_cveg'] = self.cveg
                self.cpx_init[0]['ags_thetasurf'] = self.thetasurf
                self.cpx_init[0]['ags_Ts'] = self.Ts
                self.cpx_init[0]['ags_CO2'] = self.CO2
                self.cpx_init[0]['ags_wg'] = self.wg
                self.cpx_init[0]['ags_Swin'] = self.Swin
                self.cpx_init[0]['ags_ra'] = self.ra
                self.cpx_init[0]['ags_Tsoil'] = self.Tsoil
                self.cpx_init[0]['ags_alfa_sto'] = self.alfa_sto
                self.cpx_init[0]['ags_LAI'] = self.LAI
                self.cpx_init[0]['ags_PARfract'] = self.PARfract
                self.cpx_init[0]['ags_w2'] = self.w2
                self.cpx_init[0]['ags_wfc'] = self.wfc
                self.cpx_init[0]['ags_wwilt'] = self.wwilt
                self.cpx_init[0]['ags_R10'] = self.R10
            else:
                self.cpx[self.t]['ags_COS'] = self.COS
                if self.ags_C_mode == 'surf':
                    self.cpx[self.t]['ags_COSsurf'] = self.COSsurf
                self.cpx[self.t]['ags_cveg'] = self.cveg
                self.cpx[self.t]['ags_thetasurf'] = self.thetasurf
                self.cpx[self.t]['ags_Ts'] = self.Ts
                self.cpx[self.t]['ags_CO2'] = self.CO2
                self.cpx[self.t]['ags_wg'] = self.wg
                self.cpx[self.t]['ags_Swin'] = self.Swin
                self.cpx[self.t]['ags_ra'] = self.ra
                self.cpx[self.t]['ags_Tsoil'] = self.Tsoil
                self.cpx[self.t]['ags_alfa_sto'] = self.alfa_sto
                self.cpx[self.t]['ags_LAI'] = self.LAI
                self.cpx[self.t]['ags_PARfract'] = self.PARfract
                self.cpx[self.t]['ags_w2'] = self.w2
                self.cpx[self.t]['ags_wfc'] = self.wfc
                self.cpx[self.t]['ags_wwilt'] = self.wwilt
                self.cpx[self.t]['ags_R10'] = self.R10
        # Select index for plant type
        if(self.c3c4 == 'c3'):
            c = 0
        elif(self.c3c4 == 'c4'):
            c = 1
        else:
            sys.exit('option \"%s\" for \"c3c4\" invalid'%self.c3c4)

        # calculate CO2 compensation concentration
        CO2comp       = self.CO2comp298[c] * self.rho * pow(self.Q10CO2[c],(0.1 * (self.thetasurf - 298.)))  

        # calculate mesophyll conductance
        gm1 =  self.gm298[c] *  pow(self.Q10gm[c],(0.1 * (self.thetasurf-298.))) 
        gm2 =  1. + np.exp(0.3 * (self.T1gm[c] - self.thetasurf))
        gm3 =  1. + np.exp(0.3 * (self.thetasurf - self.T2gm[c]))
        gm            = gm1 / ( gm2 * gm3)
        gm            = gm / 1000. # conversion from mm s-1 to m s-1
  
        # calculate CO2 concentration inside the leaf (ci)
        fmin0         = self.gmin[c] / self.nuco2q - 1. / 9. * gm
        sqrtf         = pow(fmin0,2.) + 4 * self.gmin[c]/self.nuco2q * gm
        sqterm        = pow(sqrtf,0.5)
        
        fmin          = -fmin0 + sqterm / (2. * gm)
  
        Ds            = (esat(self.Ts) - self.e) / 1000. # kPa
        D0            = (self.f0[c] - fmin) / self.ad[c]
  
        cfrac         = self.f0[c] * (1. - (Ds / D0)) + fmin * (Ds / D0)
        if self.ags_C_mode == 'MXL':
            co2abs        = self.CO2 * (self.mco2 / self.mair) * self.rho # conversion mumol mol-1 (ppm) to mgCO2/m3
        elif self.ags_C_mode == 'surf':
            co2abs        = self.CO2surf * (self.mco2 / self.mair) * self.rho
        else:
            raise Exception('wrong ags_C_mode switch')
        ci            = cfrac * (co2abs - CO2comp) + CO2comp
  
        # calculate maximal gross primary production in high light conditions (Ag)
        Ammax1        = self.Ammax298[c] *  pow(self.Q10Am[c],(0.1 * (self.thetasurf - 298.))) 
        Ammax2        = 1. + np.exp(0.3 * (self.T1Am[c] - self.thetasurf))
        Ammax3        = 1. + np.exp(0.3 * (self.thetasurf - self.T2Am[c]))
        Ammax         = Ammax1 / ( Ammax2 * Ammax3)

        # calculate effect of soil moisture stress on gross assimilation rate
        betaw         = max(1e-3, min(1.,(self.w2 - self.wwilt)/(self.wfc - self.wwilt)))
  
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
            if self.checkpoint:
                if call_from_init:
                    self.cpx_init[0]['ags_P_end'] = P
                    self.cpx_init[0]['ags_betaw_end'] = betaw
                else:
                    self.cpx[self.t]['ags_P_end'] = P
                    self.cpx[self.t]['ags_betaw_end'] = betaw
            fstr = (1. - np.exp(-P * betaw)) / (1 - np.exp(-P))
  
        # calculate gross assimilation rate (Am)
        aexp          = -gm*ci/Ammax + gm*CO2comp/Ammax
        Am           = Ammax * (1. - np.exp(aexp))
        Rdark        = (1. / 9.) * Am
        AmRdark      = Am + Rdark
        PAR          = self.PARfract * max(1e-1,self.Swin * self.cveg)
        xdiv         = co2abs + 2.*CO2comp
        # calculate  light use efficiency
        alphac       = self.alpha0[c] * (co2abs - CO2comp) / (xdiv)

        # calculate gross primary productivity
        pexp         = -1 * alphac * PAR / (AmRdark) #factor -1 corrected here
        Ag           = (Am + Rdark) * (1 - np.exp(pexp))

        # 1.- calculate upscaling from leaf to canopy: net flow CO2 into the plant (An)
        y            =  alphac * self.Kx[c] * PAR / (AmRdark)
        y1           = y * np.exp(-self.Kx[c] * self.LAI)
        sy           = (self.E1(y1) - self.E1(y)) / (self.Kx[c] * self.LAI)
        An_temporary           = (AmRdark) * (1. - sy  )
        # 2.- calculate upscaling from leaf to canopy: CO2 conductance at canopy level
        a1           = 1. / (1. - self.f0[c])
        a11          = a1 * (self.f0[c] - fmin)
        Dstar        = D0 / (a11)
        div1         = 1. + Ds / Dstar
        div2         = (co2abs - CO2comp) * (div1)
        part1        = a1 * fstr * An_temporary / (div2)
        gcco2        = self.alfa_sto * self.LAI * (self.gmin[c] / self.nuco2q + part1) #alfa_sto is a scaling factor for the stomatal conductance

#        c stands for canopy, l for leaf scale
#        glco2        = self.gmin[c] / self.nuco2q + a1 * fstr * Ag / ((co2abs - CO2comp) * (1. + Ds / Dstar))
#        glsCOS       = glco2/1.21 #stomatal conductance at leaf level for COS
#        gliCOS        = 0.2 /(self.rho*1000) * self.mair #m s−1, 0.2 mol_air m−2 s−1 from Seibt et al. The units of the eq: m s−1 = mol_air m−2 s−1*m3/g_air* g_air/mol_air
#        gltcos        = 1/(1/glsCOS + 1/gliCOS) #m s−1, see appendix C book Jordi and Seibt et al.
#        ideally, I would integrate the above suantity over the canopy. But it leads to a complicated integral. So I approximate:
        gctCOS = 1/(1/self.gciCOS + 1.21/gcco2) #1.21 from Seibt et al, but now for canopy stomatal conductance
        # calculate surface resistance for moisture and carbon dioxide
            
        self.rs      = 1. / (1.6 * gcco2)
        rsCO2        = 1. / gcco2
  
        # calculate net flux of CO2 into the plant (An)
        An           = -(co2abs - ci) / (self.ra + rsCO2)
  
        # CO2 soil surface flux
        fw           = self.Cw * self.wmax / (self.wg + self.wmin)
        texp = self.E0 / (283.15 * 8.314) * (1. - 283.15 / self.Tsoil)
        Resp         = self.R10 * (1. - fw) * np.exp(texp)
  
        # CO2 flux
        self.wCO2A   = An   * (self.mair / (self.rho * self.mco2))
        self.wCO2R   = Resp * (self.mair / (self.rho * self.mco2))
        self.wCO2    = self.wCO2A + self.wCO2R
        if self.ags_C_mode == 'MXL':
            self.wCOSP   = - 1 / (1 / gctCOS + self.ra) * self.COS #plant flux COS in ppb m s-1
        elif self.ags_C_mode == 'surf':
            self.wCOSP   = - 1 / (1 / gctCOS + self.ra) * self.COSsurf #plant flux COS
        else:
            raise Exception('wrong ags_C_mode switch')
        if(self.soilCOSmodeltype == 'Sun_Ogee'): 
            if call_from_init:
                self.soilCOSmodel = sCOSm.soil_COS_mod(self.input,self.COSsurf,self.Ps,self.Tsurf,mothermodel=self)
                self.wCOSS_molm2s = self.soilCOSmodel.run_soil_COS_mod(self.Tsoil,self.T2,self.wg,self.w2,self.COSsurf,self.Ps,self.Tsurf,self.wsat,self.dt,call_from_init=True) #in mol /m2 /s
            else:
                self.wCOSS_molm2s = self.soilCOSmodel.run_soil_COS_mod(self.Tsoil,self.T2,self.wg,self.w2,self.COSsurf,self.Ps,self.Tsurf,self.wsat,self.dt)
            self.wCOSS = self.wCOSS_molm2s / self.rho * self.mair * 1.e-3 * 1.e9 #ppb m/s; mol_COS/(m2*s) * m3_air / kg_air * g_air / mol_air * kg_air / g_air * nmol_COS / mol_COS 
        elif self.soilCOSmodeltype == None:
            self.wCOSS   = 0.0 #
            self.soilCOSmodel = None
        else:
            raise Exception('problem with soilCOSmodeltype')
        self.wCOS    = self.wCOSP + self.wCOSS
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['ags_gm_end'] = gm
                self.cpx_init[0]['ags_gm1_end'] = gm1
                self.cpx_init[0]['ags_gm2_end'] = gm2
                self.cpx_init[0]['ags_gm3_end'] = gm3
                self.cpx_init[0]['ags_sqrtf_end'] = sqrtf
                self.cpx_init[0]['ags_sqterm_end'] = sqterm
                self.cpx_init[0]['ags_fmin0_end'] = fmin0
                self.cpx_init[0]['ags_CO2comp_end'] = CO2comp
                self.cpx_init[0]['ags_Ammax1_end'] = Ammax1
                self.cpx_init[0]['ags_Ammax2_end'] = Ammax2
                self.cpx_init[0]['ags_Ammax3_end'] = Ammax3
                self.cpx_init[0]['ags_fmin_end'] = fmin
                self.cpx_init[0]['ags_Ds_end'] = Ds
                self.cpx_init[0]['ags_D0_end'] = D0
                self.cpx_init[0]['ags_co2abs_end'] = co2abs
                self.cpx_init[0]['ags_ci_end'] = ci
                self.cpx_init[0]['ags_cfrac_end'] =cfrac
                self.cpx_init[0]['ags_pexp_end'] = pexp
                self.cpx_init[0]['ags_xdiv_end'] = xdiv
                self.cpx_init[0]['ags_aexp_end'] = aexp
                self.cpx_init[0]['ags_Ammax_end'] = Ammax
                self.cpx_init[0]['ags_PAR_end'] = PAR
                self.cpx_init[0]['ags_alphac_end'] = alphac
                self.cpx_init[0]['ags_AmRdark_end'] = AmRdark
                self.cpx_init[0]['ags_y_end'] = y
                self.cpx_init[0]['ags_y1_end'] = y1
                self.cpx_init[0]['ags_sy_end'] = sy
                self.cpx_init[0]['ags_fstr_end'] = fstr
                self.cpx_init[0]['ags_a1_end'] = a1
                self.cpx_init[0]['ags_div1_end'] = div1
                self.cpx_init[0]['ags_div2_end'] = div2
                self.cpx_init[0]['ags_An_temporary_end'] = An_temporary
                self.cpx_init[0]['ags_part1_end'] = part1
                self.cpx_init[0]['ags_gcco2_end'] = gcco2
                self.cpx_init[0]['ags_a11_end'] = a11
                self.cpx_init[0]['ags_Dstar_end'] = Dstar
                self.cpx_init[0]['ags_gciCOS_end'] = self.gciCOS
                self.cpx_init[0]['ags_gctCOS_end'] = gctCOS
                self.cpx_init[0]['ags_texp_end'] = texp
                self.cpx_init[0]['ags_fw_end'] = fw
                self.cpx_init[0]['ags_rsCO2_end'] = rsCO2
            else:
                self.cpx[self.t]['ags_gm_end'] = gm
                self.cpx[self.t]['ags_gm1_end'] = gm1
                self.cpx[self.t]['ags_gm2_end'] = gm2
                self.cpx[self.t]['ags_gm3_end'] = gm3
                self.cpx[self.t]['ags_sqrtf_end'] = sqrtf
                self.cpx[self.t]['ags_sqterm_end'] = sqterm
                self.cpx[self.t]['ags_fmin0_end'] = fmin0
                self.cpx[self.t]['ags_CO2comp_end'] = CO2comp
                self.cpx[self.t]['ags_Ammax1_end'] = Ammax1
                self.cpx[self.t]['ags_Ammax2_end'] = Ammax2
                self.cpx[self.t]['ags_Ammax3_end'] = Ammax3
                self.cpx[self.t]['ags_fmin_end'] = fmin
                self.cpx[self.t]['ags_Ds_end'] = Ds
                self.cpx[self.t]['ags_D0_end'] = D0
                self.cpx[self.t]['ags_co2abs_end'] = co2abs
                self.cpx[self.t]['ags_ci_end'] = ci
                self.cpx[self.t]['ags_cfrac_end'] =cfrac
                self.cpx[self.t]['ags_pexp_end'] = pexp
                self.cpx[self.t]['ags_xdiv_end'] = xdiv
                self.cpx[self.t]['ags_aexp_end'] = aexp
                self.cpx[self.t]['ags_Ammax_end'] = Ammax
                self.cpx[self.t]['ags_PAR_end'] = PAR
                self.cpx[self.t]['ags_alphac_end'] = alphac
                self.cpx[self.t]['ags_AmRdark_end'] = AmRdark
                self.cpx[self.t]['ags_y_end'] = y
                self.cpx[self.t]['ags_y1_end'] = y1
                self.cpx[self.t]['ags_sy_end'] = sy
                self.cpx[self.t]['ags_fstr_end'] = fstr
                self.cpx[self.t]['ags_a1_end'] = a1
                self.cpx[self.t]['ags_div1_end'] = div1
                self.cpx[self.t]['ags_div2_end'] = div2
                self.cpx[self.t]['ags_An_temporary_end'] = An_temporary
                self.cpx[self.t]['ags_part1_end'] = part1
                self.cpx[self.t]['ags_gcco2_end'] = gcco2
                self.cpx[self.t]['ags_a11_end'] = a11
                self.cpx[self.t]['ags_Dstar_end'] = Dstar
                self.cpx[self.t]['ags_gciCOS_end'] = self.gciCOS
                self.cpx[self.t]['ags_gctCOS_end'] = gctCOS
                self.cpx[self.t]['ags_texp_end'] = texp
                self.cpx[self.t]['ags_fw_end'] = fw
                self.cpx[self.t]['ags_rsCO2_end'] = rsCO2
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_ags.update({variablename: the_locals[variablename]})

    def run_land_surface(self,call_from_init=False):
        self.vars_rls= {} #a dictionary of all defined variables for testing purposes
        if self.checkpoint: 
            if call_from_init:
                self.cpx_init[0]['rls_u'] = self.u
                self.cpx_init[0]['rls_v'] = self.v
                self.cpx_init[0]['rls_wstar'] = self.wstar
                self.cpx_init[0]['rls_Cs'] = self.Cs
                self.cpx_init[0]['rls_theta'] = self.theta
                self.cpx_init[0]['rls_wg'] = self.wg
                self.cpx_init[0]['rls_wfc'] = self.wfc
                self.cpx_init[0]['rls_wwilt'] = self.wwilt
                self.cpx_init[0]['rls_Wmax'] = self.Wmax
                self.cpx_init[0]['rls_LAI'] = self.LAI
                self.cpx_init[0]['rls_Wl'] = self.Wl
                self.cpx_init[0]['rls_cveg'] = self.cveg
                self.cpx_init[0]['rls_Lambda'] = self.Lambda
                self.cpx_init[0]['rls_q'] = self.q
                self.cpx_init[0]['rls_Tsoil'] = self.Tsoil
                self.cpx_init[0]['rls_Q'] = self.Q
                self.cpx_init[0]['rls_rsmin'] = self.rsmin
                self.cpx_init[0]['rls_w2'] = self.w2
                self.cpx_init[0]['rls_wsat'] = self.wsat
                self.cpx_init[0]['rls_ustar'] = self.ustar
                self.cpx_init[0]['rls_rssoilmin'] = self.rssoilmin
                self.cpx_init[0]['rls_CGsat'] = self.CGsat
                self.cpx_init[0]['rls_b'] = self.b
                self.cpx_init[0]['rls_C1sat'] = self.C1sat
                self.cpx_init[0]['rls_C2ref'] = self.C2ref
                self.cpx_init[0]['rls_a'] = self.a
                self.cpx_init[0]['rls_p'] = self.p
            else:
                self.cpx[self.t]['rls_u'] = self.u
                self.cpx[self.t]['rls_v'] = self.v
                self.cpx[self.t]['rls_wstar'] = self.wstar
                self.cpx[self.t]['rls_Cs'] = self.Cs
                self.cpx[self.t]['rls_theta'] = self.theta
                self.cpx[self.t]['rls_wg'] = self.wg
                self.cpx[self.t]['rls_wfc'] = self.wfc
                self.cpx[self.t]['rls_wwilt'] = self.wwilt
                self.cpx[self.t]['rls_Wmax'] = self.Wmax
                self.cpx[self.t]['rls_LAI'] = self.LAI
                self.cpx[self.t]['rls_Wl'] = self.Wl
                self.cpx[self.t]['rls_cveg'] = self.cveg
                self.cpx[self.t]['rls_Lambda'] = self.Lambda
                self.cpx[self.t]['rls_q'] = self.q
                self.cpx[self.t]['rls_Tsoil'] = self.Tsoil
                self.cpx[self.t]['rls_Q'] = self.Q
                self.cpx[self.t]['rls_rsmin'] = self.rsmin
                self.cpx[self.t]['rls_w2'] = self.w2
                self.cpx[self.t]['rls_wsat'] = self.wsat
                self.cpx[self.t]['rls_ustar'] = self.ustar
                self.cpx[self.t]['rls_rssoilmin'] = self.rssoilmin
                self.cpx[self.t]['rls_CGsat'] = self.CGsat
                self.cpx[self.t]['rls_b'] = self.b
                self.cpx[self.t]['rls_C1sat'] = self.C1sat
                self.cpx[self.t]['rls_C2ref'] = self.C2ref
                self.cpx[self.t]['rls_a'] = self.a
                self.cpx[self.t]['rls_p'] = self.p
        # compute ra
        ueff = np.sqrt(self.u ** 2. + self.v ** 2. + self.wstar**2.)

        if(self.sw_sl):
          self.ra = (self.Cs * ueff)**-1. #see notes next to eq 3.45 AVSI
        else:
          self.ra = ueff / max(1.e-3, self.ustar)**2.

        # first calculate essential thermodynamic variables
        self.esatvar    = esat(self.theta) #var added to prevent name of function and of variable to be the same
        self.qsatvar    = qsat(self.theta, self.Ps)
        desatdT      = self.esatvar * (17.2694 / (self.theta - 35.86) - 17.2694 * (self.theta - 273.16) / (self.theta - 35.86)**2.)
        self.dqsatdT = 0.622 * desatdT / self.Ps
        self.e       = self.q * self.Ps / 0.622

        if(self.ls_type == 'js'): 
            self.jarvis_stewart(call_from_init) 
        elif(self.ls_type == 'ags'):
            self.ags(call_from_init)
        elif(self.ls_type == 'canopy_model'):
            #first respiration flux as we don't run ags that normally calculates it:
            fw           = self.Cw * self.wmax / (self.wg + self.wmin)
            texp = self.E0 / (283.15 * 8.314) * (1. - 283.15 / self.Tsoil)
            Resp         = self.R10 * (1. - fw) * np.exp(texp) #mg CO2 m-2 s-1
            self.wCO2R   = Resp * (self.mair / (self.rho * self.mco2))
            Resp_mol_m2s = Resp / self.mco2 * 0.001 #mol/m2/s
            #than soil COS flux
            if(self.soilCOSmodeltype == 'Sun_Ogee'):
                if call_from_init:
                    #the following line is needed since the canopy model has not been runned yet
                    COS_lowest_veglay = np.array(self.input.C_COS_veglayer_init_ppb)[0]# in ppb. 
                    self.soilCOSmodel = sCOSm.soil_COS_mod(self.input,COS_lowest_veglay,self.Ps,self.Tsurf,mothermodel=self) #COS_lowest_veglay should be in ppb
                    self.wCOSS_molm2s = self.soilCOSmodel.run_soil_COS_mod(self.Tsoil,self.T2,self.wg,self.w2,COS_lowest_veglay,self.Ps,self.Tsurf,self.wsat,self.dt,call_from_init=True) #in mol /m2 /s
                    self.wCOSS = self.wCOSS_molm2s * 1.e9 / self.rho * self.mair * 0.001 #mol_COS m-2s-1 to ppb ms-1; mol_COS m-2s-1 * nmol_COS / mol_COS * m3 / kg_air * g_air/mol_air * kg_air/g_air = ppb ms-1
                else:
                    COS_lowest_veglay = self.canopy_model.C_COS_veglayer_current[0] * 1.e9 * self.mair / self.rho * 0.001# conversion mol_COS m-3 to nmol mol-1 (ppb): mol_COS m-3 * nmol_COS/mol_COS * g_air/mol_air * m3/kg_air * kg_air / g_air
                    self.wCOSS_molm2s = self.soilCOSmodel.run_soil_COS_mod(self.Tsoil,self.T2,self.wg,self.w2,COS_lowest_veglay,self.Ps,self.Tsurf,self.wsat,self.dt)
                    self.wCOSS = self.wCOSS_molm2s * 1.e9 / self.rho * self.mair * 0.001 #mol_COS m-2s-1 to ppb ms-1; mol_COS m-2s-1 * nmol_COS / mol_COS * m3 / kg_air * g_air/mol_air * kg_air/g_air = ppb ms-1
            elif self.soilCOSmodeltype == None:
                self.wCOSS_molm2s = 0.
                self.wCOSS = 0.
            #now the actual canopy model
            if self.calc_sun_shad:
                if self.prescr_fPARdif:
                    if len(self.fPARdif) == 1:
                        fPARdif = self.fPARdif
                    elif len(self.fPARdif) == self.tsteps:
                        fPARdif = self.fPARdif[self.t]
                    else:
                        raise Exception('Wrong length of FPARdif input')
                else:
                    fPARdif = None
            else:
                fPARdif = None
            if call_from_init:
                self.canopy_model = canm.canopy_mod(inputdata=self.input,mothermodel=self)
                for i in range(int(self.dt/self.dt_can)): #int will only change the data type, dt should be a multiple of dt_can, this is checked in the inititalisation     
                #use COS and CO2 instead of COSsurf and CO2surf, otherwise you get numerical instability, since wCOS determines wether COSsurf will be larger or smaller than COS, you can end up with negative COSsurf etc.
                    if self.incl_H2Ocan:
                        COSfluxVegSurfLay,CO2fluxVegSurfLay,H2OfluxVegSurfLay,self.wCOSP_molm2s,self.wCO2A_molm2s,wH2OP_molm2s,self.C_COS_veglayer,self.C_CO2_veglayer,self.C_H2O_veglayer,canopymodelTs,soilflux_H2O,liqwat_flux,rs_H2Ocanmod = self.canopy_model.run_canopy_mod(self.thetasurf,self.Ts,self.w2,self.Swin,self.wCOSS_molm2s,self.COS,Resp_mol_m2s,self.CO2,self.e/self.Ps*100,np.sqrt(self.u2m**2 + self.v2m**2),self.Ps,self.dt_can,e=None,ra_veg=self.ra_veg,fPARdif=fPARdif)
                        self.C_H2O_veglayer_pct = cp.deepcopy(self.C_H2O_veglayer) * 1.e2 * self.mair / self.rho * 0.001
                    else:
                        COSfluxVegSurfLay,CO2fluxVegSurfLay,self.wCOSP_molm2s,self.wCO2A_molm2s,self.C_COS_veglayer,self.C_CO2_veglayer = self.canopy_model.run_canopy_mod(self.thetasurf,self.Ts,self.w2,self.Swin,self.wCOSS_molm2s,self.COS,Resp_mol_m2s,self.CO2,np.sqrt(self.u2m**2 + self.v2m**2),self.dt_can,e=self.e,ra_veg=self.ra_veg,fPARdif=fPARdif)
                self.wCOSP = self.wCOSP_molm2s * 1.e9 / self.rho * self.mair * 0.001 #ppb m s-1
                self.C_COS_veglayer_ppb = cp.deepcopy(self.C_COS_veglayer) * 1.e9 * self.mair / self.rho * 0.001# conversion mol_COS m-3 to nmol mol-1 (ppb)  
                self.C_CO2_veglayer_ppm = cp.deepcopy(self.C_CO2_veglayer) * 1.e6 * self.mair / self.rho * 0.001
                self.wCO2A = self.wCO2A_molm2s * 1.e6 / self.rho * self.mair * 0.001 #ppm m s-1 
            else:
                for i in range(int(self.dt/self.dt_can)):
                    if self.incl_H2Ocan:
                        COSfluxVegSurfLay,CO2fluxVegSurfLay,H2OfluxVegSurfLay,self.wCOSP_molm2s,self.wCO2A_molm2s,wH2OP_molm2s,self.C_COS_veglayer,self.C_CO2_veglayer,self.C_H2O_veglayer,canopymodelTs,soilflux_H2O,liqwat_flux,rs_H2Ocanmod = self.canopy_model.run_canopy_mod(self.thetasurf,self.Ts,self.w2,self.Swin,self.wCOSS_molm2s,self.COS,Resp_mol_m2s,self.CO2,self.e/self.Ps*100,np.sqrt(self.u2m**2 + self.v2m**2),self.Ps,self.dt_can,e=None,ra_veg=self.ra_veg,fPARdif=fPARdif)
                        self.C_H2O_veglayer_pct = cp.deepcopy(self.C_H2O_veglayer) * 1.e2 * self.mair / self.rho * 0.001
                    else:
                        COSfluxVegSurfLay,CO2fluxVegSurfLay,self.wCOSP_molm2s,self.wCO2A_molm2s,self.C_COS_veglayer,self.C_CO2_veglayer = self.canopy_model.run_canopy_mod(self.thetasurf,self.Ts,self.w2,self.Swin,self.wCOSS_molm2s,self.COS,Resp_mol_m2s,self.CO2,np.sqrt(self.u2m**2 + self.v2m**2),self.dt_can,e=self.e,ra_veg=self.ra_veg,fPARdif=fPARdif)
                self.wCOSP = self.wCOSP_molm2s * 1.e9 / self.rho * self.mair * 0.001 #ppb m s-1
                self.C_COS_veglayer_ppb = cp.deepcopy(self.C_COS_veglayer) * 1.e9 * self.mair / self.rho * 0.001# conversion mol_COS m-3 to nmol mol-1 (ppb)  
                self.C_CO2_veglayer_ppm = cp.deepcopy(self.C_CO2_veglayer) * 1.e6 * self.mair / self.rho * 0.001
                self.wCO2A = self.wCO2A_molm2s * 1.e6 / self.rho * self.mair * 0.001 #ppm m s-1
            #self.C_COS_veglayer in mol/m3/s
            self.wCOS = COSfluxVegSurfLay / self.rho * self.mair * 1.e-3 * 1.e9 #ppb m/s; mol_COS/(m2*s) * m3_air / kg_air * g_air / mol_air * kg_air / g_air * nmol_COS / mol_COS 
            self.wCO2  = CO2fluxVegSurfLay / self.rho * self.mair * 1.e-3 * 1.e6 #ppm m/s; mol_CO2/(m2*s) * m3_air / kg_air * g_air / mol_air * kg_air / g_air * mumol_CO2 / mol_CO2 
                      
        elif(self.ls_type == 'sib4'):
            #first respiration flux as we don't run ags that normally calculates it:
            fw           = self.Cw * self.wmax / (self.wg + self.wmin)
            texp = self.E0 / (283.15 * 8.314) * (1. - 283.15 / self.Tsoil)
            Resp         = self.R10 * (1. - fw) * np.exp(texp) #mg CO2 m-2 s-1
            self.wCO2R   = Resp * (self.mair / (self.rho * self.mco2))
            if call_from_init:
                self.sib4model = sib4.pho_sib4(dtsib=self.dt,CLASSPs=self.Ps,CLASSc3c4=self.c3c4,CLASSLAI=self.LAI,CLASSCO2=self.CO2,CLASStheta=self.theta,CLASSq=self.q,CLASSe=self.e,inputdata=self.input,mothermodel=self)
                #now COS, 0 for now
                self.wCOSS = 0.0
                self.wCOSP = 0.0
                self.wCOS = 0.0
                self.wCO2A = 0.0 #just for now
                resp_leaf = 0.
                resp_auto = 0.
                resp_het = Resp /1000 / self.mco2#heterotrophic respiration (mol C/m2/s)
                CLASSrhsurf = self.esurf/esat(self.Tsurf)
                CLASSWIND = np.sqrt(self.u**2 + self.v**2)
                self.G = 0. #not calculated yet
                for i in range(30):
                    self.sib4model.run_pho_sib4(resp_leaf, resp_auto, resp_het,CLASSrhsurf,self.lon,self.lat,self.cc,self.doy,self.Tsoil,self.CO2,self.theta,self.G,self.Swin,self.sinlea,CLASSWIND,self.Lwin,self.q)
                    self.wCO2_molm2s = self.sib4model.assimn
                    self.wCO2 = self.wCO2_molm2s / self.rho * self.mair * 1.e-3 * 1.e6 #ppm m/s; mol_CO2/(m2*s) * m3_air / kg_air * g_air / mol_air * kg_air / g_air * mumol_CO2 / mol_CO2 
            else:
                self.wCOSS = 0.0
                self.wCOSP = 0.0
                self.wCOS = 0.0
                self.wCO2A = 0.0 #just for now
                resp_leaf = 0.
                resp_auto = 0.
                resp_het = Resp /1000 / self.mco2#heterotrophic respiration (mol C/m2/s)
                CLASSrhsurf = self.esurf/esat(self.Tsurf)
                CLASSWIND = np.sqrt(self.u**2 + self.v**2)
                self.sib4model.run_pho_sib4(resp_leaf, resp_auto, resp_het,CLASSrhsurf,self.lon,self.lat,self.cc,self.doy,self.Tsoil,self.CO2,self.theta,self.G,self.Swin,self.sinlea,CLASSWIND,self.Lwin,self.q)
                self.wCO2_molm2s = self.sib4model.assimn
                self.wCO2 = self.wCO2_molm2s / self.rho * self.mair * 1.e-3 * 1.e6 #ppm m/s; mol_CO2/(m2*s) * m3_air / kg_air * g_air / mol_air * kg_air / g_air * mumol_CO2 / mol_CO2 
        else:
            sys.exit('option \"%s\" for \"ls_type\" invalid'%self.ls_type)

        # recompute f2 using wg instead of w2
        if(self.wg > self.wwilt):# and self.w2 <= self.wfc):
          f2          = (self.wfc - self.wwilt) / (self.wg - self.wwilt)
        else:
          f2        = 1.e8
        self.rssoil = self.rssoilmin * f2 
        if(self.ls_type != 'canopy_model'):
            Wlmx = self.LAI * self.Wmax
            self.cliq = min(1., self.Wl / Wlmx)
            # calculate skin temperature implictly see eq 9.17 Book Jordi, calculates H as resiudal of energy balance. But mind that in eq9.17, Ts is also present in right hand side equation. see notes 2 canopy model
            #below is fully written out:
    #        self.Ts   = (self.Q  + self.rho * self.cp / self.ra * self.theta \
    #            + self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs    ) * (self.dqsatdT * self.theta - self.qsatvar + self.q) \
    #           + (1. - self.cveg)             * self.rho * self.Lv / (self.ra + self.rssoil) * (self.dqsatdT * self.theta - self.qsatvar + self.q) \
    #           + self.cveg * self.cliq        * self.rho * self.Lv /  self.ra                * (self.dqsatdT * self.theta - self.qsatvar + self.q) + self.Lambda * self.Tsoil) \
    #           / (self.rho * self.cp / self.ra + self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs) * self.dqsatdT \
    #           + (1. - self.cveg) * self.rho * self.Lv / (self.ra + self.rssoil) * self.dqsatdT + self.cveg * self.cliq * self.rho * self.Lv / self.ra * self.dqsatdT + self.Lambda)
            # now split for simplifying the derivatives:
            p1_numerator_Ts = self.rho * self.cp / self.ra * self.theta
            p2_numerator_Ts = self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs    )
            p3_numerator_Ts = self.dqsatdT * self.theta - self.qsatvar + self.q
            p4_numerator_Ts = self.rho * self.Lv / (self.ra + self.rssoil) * (self.dqsatdT * self.theta - self.qsatvar + self.q)
            p5_numerator_Ts = self.cliq * self.rho * self.Lv /  self.ra * (self.dqsatdT * self.theta - self.qsatvar + self.q)
            numerator_Ts = self.Q  + p1_numerator_Ts + p2_numerator_Ts * p3_numerator_Ts + (1. - self.cveg) * p4_numerator_Ts \
                + self.cveg * p5_numerator_Ts + self.Lambda * self.Tsoil
            p1_denominator_Ts = self.rho * self.cp / self.ra 
            p2_denominator_Ts = self.cveg * (1. - self.cliq) * self.rho * self.Lv / (self.ra + self.rs) * self.dqsatdT
            p3_denominator_Ts = self.rho * self.Lv / (self.ra + self.rssoil) * self.dqsatdT
            p4_denominator_Ts = self.cliq * self.rho * self.Lv / self.ra * self.dqsatdT
            denominator_Ts = p1_denominator_Ts + p2_denominator_Ts + (1. - self.cveg) * p3_denominator_Ts + self.cveg * p4_denominator_Ts + self.Lambda
            self.Ts   = numerator_Ts / denominator_Ts
    #        print('Ts and theta')
    #        print(self.Ts)
    #        print(self.theta)
    
            esatsurf      = esat(self.Ts)
            self.qsatsurf = qsat(self.Ts, self.Ps)
    
            p1_LEveg = (1. - self.cliq) * self.cveg * self.rho * self.Lv / (self.ra + self.rs)
            #the next part is common to LEveg, lEliq and LEsoil
            p2_LEveg_liq_soil = self.dqsatdT * (self.Ts - self.theta) + self.qsatvar - self.q
            self.LEveg  = p1_LEveg * p2_LEveg_liq_soil
            p1_LEliq = self.cliq * self.cveg * self.rho * self.Lv / self.ra
            self.LEliq  = p1_LEliq * p2_LEveg_liq_soil
            p1_LEsoil = (1. - self.cveg) * self.rho * self.Lv / (self.ra + self.rssoil)
            self.LEsoil = p1_LEsoil * p2_LEveg_liq_soil
            
            self.Wltend      = - self.LEliq / (self.rhow * self.Lv)
  
            self.LE     = self.LEsoil + self.LEveg + self.LEliq
            self.H      = self.rho * self.cp / self.ra * (self.Ts - self.theta)
            self.G      = self.Lambda * (self.Ts - self.Tsoil)
            # LEpot written out fully:
            # self.LEpot  = (self.dqsatdT * (self.Q - self.G) + self.rho * self.cp / self.ra * (self.qsatvar - self.q)) / (self.dqsatdT + self.cp / self.Lv)
            p1_numerator_LEpot = self.dqsatdT * (self.Q - self.G)
            p2_numerator_LEpot = self.rho * self.cp / self.ra * (self.qsatvar - self.q)
            numerator_LEpot = (p1_numerator_LEpot + p2_numerator_LEpot)
            self.LEpot  = numerator_LEpot / (self.dqsatdT + self.cp / self.Lv)
            # LEref written out fully:
            # self.LEref  = (self.dqsatdT * (self.Q - self.G) + self.rho * self.cp / self.ra * (self.qsatvar - self.q)) / (self.dqsatdT + self.cp / self.Lv * (1. + self.rsmin / self.LAI / self.ra))
            p1_numerator_LEref = self.dqsatdT * (self.Q - self.G)
            p2_numerator_LEref = self.rho * self.cp / self.ra * (self.qsatvar - self.q)
            numerator_LEref = p1_numerator_LEref + p2_numerator_LEref
            denominator_LEref = (self.dqsatdT + self.cp / self.Lv * (1. + self.rsmin / self.LAI / self.ra))
            self.LEref  = numerator_LEref / denominator_LEref
        else:
            #self.rs = rs_H2Ocanmod[-1] / self.canopy_model.LAI_veglay[-1] #only used for qsurf in surface layer
            self.Ts = canopymodelTs
            self.LEliq  = np.sum(liqwat_flux) * self.mh2o / 1000 * self.Lv
            self.LEsoil = soilflux_H2O * self.mh2o / 1000 * self.Lv
            self.LEveg = wH2OP_molm2s * self.mh2o / 1000 * self.Lv
            
            #Note that the energy balance still closes for the system CLASS + canopy model(except for the error due to the linearization of esat(T)), but for CLASS alone Q is no longer = H + LE + G, since LE in CLASS is the evapotranspiration energy flux into the mixed layer, not into the system.
            #Since the LE calculated in CLASS end up in the mixed layer, we do not calculate it as LEveg + LEsoil + LEliq. If you would use the LE of the canopy model(LEveg + LEsoil + LEliq), Q is still H + LE + G. But using that would be inconsistent, than the same water ends up in both the canopy as the mixed layer
            self.LE = H2OfluxVegSurfLay * self.mh2o / 1000 * self.Lv
            self.LEref  = None #just because we do not calculate
            self.LEpot  = None #just because we do not calculate
            self.G      = self.Lambda * (self.Ts - self.Tsoil)
#            print('Ts - theta')
#            print(self.Ts - self.theta)
#            print('ra')
#            print(self.ra)
            self.H      = self.rho * self.cp / self.ra * (self.Ts - self.theta)
            self.enbalerr = self.Q - self.G - self.LEliq - self.LEsoil - self.LEveg - self.H
#            print('Q')
#            print(self.Q)
#            print('G')
#            print(self.G)
#            print('H')
#            print(self.H)
#            print('LE')
#            print(self.LEliq+ self.LEsoil + self.LEveg)
#            print('enbal')
#            print(self.Q - self.G - self.LEliq - self.LEsoil - self.LEveg - self.H)
        
        CG          = self.CGsat * (self.wsat / self.w2)**(self.b / (2. * np.log(10.)))
  
        self.Tsoiltend   = CG * self.G - 2. * np.pi / 86400. * (self.Tsoil - self.T2)
   
        d1          = 0.1
        C1          = self.C1sat * (self.wsat / self.wg) ** (self.b / 2. + 1.)
        C2          = self.C2ref * (self.w2 / (self.wsat - self.w2) )
        wgeq        = self.w2 - self.wsat * self.a * ( (self.w2 / self.wsat) ** self.p * (1. - (self.w2 / self.wsat) ** (8. * self.p)) )
        self.wgtend = - C1 / (self.rhow * d1) * self.LEsoil / self.Lv - C2 / 86400. * (self.wg - wgeq)
  
        # calculate kinematic heat fluxes
        self.wtheta   = self.H  / (self.rho * self.cp)
        self.wq       = self.LE / (self.rho * self.Lv)
        
        if self.checkpoint:
            if call_from_init:
                self.cpx_init[0]['rls_ueff_end'] = ueff
                self.cpx_init[0]['rls_esatvar_end'] = self.esatvar
                self.cpx_init[0]['rls_Wlmx_end'] = Wlmx
                self.cpx_init[0]['rls_ra_end'] = self.ra
                self.cpx_init[0]['rls_cliq_end'] = self.cliq
                self.cpx_init[0]['rls_dqsatdT_end'] = self.dqsatdT
                self.cpx_init[0]['rls_rssoil_end'] = self.rssoil
                self.cpx_init[0]['rls_qsatvar_end'] = self.qsatvar
                self.cpx_init[0]['rls_p2_numerator_Ts_end'] = p2_numerator_Ts
                self.cpx_init[0]['rls_p3_numerator_Ts_end'] = p3_numerator_Ts
                self.cpx_init[0]['rls_p4_numerator_Ts_end'] = p4_numerator_Ts
                self.cpx_init[0]['rls_p5_numerator_Ts_end'] = p5_numerator_Ts
                self.cpx_init[0]['rls_p3_denominator_Ts_end'] = p3_denominator_Ts
                self.cpx_init[0]['rls_p4_denominator_Ts_end'] = p4_denominator_Ts
                self.cpx_init[0]['rls_numerator_Ts_end'] = numerator_Ts
                self.cpx_init[0]['rls_denominator_Ts_end'] = denominator_Ts
                self.cpx_init[0]['rls_p1_LEveg_end'] = p1_LEveg
                self.cpx_init[0]['rls_p2_LEveg_liq_soil_end'] = p2_LEveg_liq_soil
                self.cpx_init[0]['rls_Ts_end'] = self.Ts
                self.cpx_init[0]['rls_p1_LEliq_end'] = p1_LEliq
                self.cpx_init[0]['rls_p1_LEsoil_end'] = p1_LEsoil
                self.cpx_init[0]['rls_G_end'] = self.G
                self.cpx_init[0]['rls_numerator_LEpot_end'] = numerator_LEpot
                self.cpx_init[0]['rls_denominator_LEref_end'] = denominator_LEref
                self.cpx_init[0]['rls_numerator_LEref_end'] = numerator_LEref
                self.cpx_init[0]['rls_CG_end'] = CG
                self.cpx_init[0]['rls_C1_end'] = C1
                self.cpx_init[0]['rls_C2_end'] = C2
                self.cpx_init[0]['rls_d1_end'] = d1
                self.cpx_init[0]['rls_LEsoil_end'] = self.LEsoil
                self.cpx_init[0]['rls_wgeq_end'] = wgeq
                self.cpx_init[0]['rls_rs_end'] = self.rs
                self.cpx_init[0]['rls_f2_end'] = f2
            else:
                self.cpx[self.t]['rls_ueff_end'] = ueff
                self.cpx[self.t]['rls_esatvar_end'] = self.esatvar
                self.cpx[self.t]['rls_Wlmx_end'] = Wlmx
                self.cpx[self.t]['rls_ra_end'] = self.ra
                self.cpx[self.t]['rls_cliq_end'] = self.cliq
                self.cpx[self.t]['rls_dqsatdT_end'] = self.dqsatdT
                self.cpx[self.t]['rls_rssoil_end'] = self.rssoil
                self.cpx[self.t]['rls_qsatvar_end'] = self.qsatvar
                self.cpx[self.t]['rls_p2_numerator_Ts_end'] = p2_numerator_Ts
                self.cpx[self.t]['rls_p3_numerator_Ts_end'] = p3_numerator_Ts
                self.cpx[self.t]['rls_p4_numerator_Ts_end'] = p4_numerator_Ts
                self.cpx[self.t]['rls_p5_numerator_Ts_end'] = p5_numerator_Ts
                self.cpx[self.t]['rls_p3_denominator_Ts_end'] = p3_denominator_Ts
                self.cpx[self.t]['rls_p4_denominator_Ts_end'] = p4_denominator_Ts
                self.cpx[self.t]['rls_numerator_Ts_end'] = numerator_Ts
                self.cpx[self.t]['rls_denominator_Ts_end'] = denominator_Ts
                self.cpx[self.t]['rls_p1_LEveg_end'] = p1_LEveg
                self.cpx[self.t]['rls_p2_LEveg_liq_soil_end'] = p2_LEveg_liq_soil
                self.cpx[self.t]['rls_Ts_end'] = self.Ts
                self.cpx[self.t]['rls_p1_LEliq_end'] = p1_LEliq
                self.cpx[self.t]['rls_p1_LEsoil_end'] = p1_LEsoil
                self.cpx[self.t]['rls_G_end'] = self.G
                self.cpx[self.t]['rls_numerator_LEpot_end'] = numerator_LEpot
                self.cpx[self.t]['rls_denominator_LEref_end'] = denominator_LEref
                self.cpx[self.t]['rls_numerator_LEref_end'] = numerator_LEref
                self.cpx[self.t]['rls_CG_end'] = CG
                self.cpx[self.t]['rls_C1_end'] = C1
                self.cpx[self.t]['rls_C2_end'] = C2
                self.cpx[self.t]['rls_d1_end'] = d1
                self.cpx[self.t]['rls_LEsoil_end'] = self.LEsoil
                self.cpx[self.t]['rls_wgeq_end'] = wgeq
                self.cpx[self.t]['rls_rs_end'] = self.rs
                self.cpx[self.t]['rls_f2_end'] = f2
        
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_rls.update({variablename: the_locals[variablename]})
 
    def integrate_land_surface(self):
        self.vars_ils= {}
        # integrate soil equations
        Tsoil0        = self.Tsoil
        wg0           = self.wg
  
        self.Tsoil    = Tsoil0  + self.dt * self.Tsoiltend
        self.wg       = wg0     + self.dt * self.wgtend
        if self.ls_type != 'canopy_model':
            Wl0           = self.Wl
            self.Wl       = Wl0     + self.dt * self.Wltend
        
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_ils.update({variablename: the_locals[variablename]})
  
    # store model output
    def store(self):
        self.vars_sto= {}
        t                      = self.t
        self.out.t[t]          = t * self.dt / 3600. + self.tstart
        self.out.h[t]          = self.h
        
        self.out.theta[t]      = self.theta
        self.out.thetav[t]     = self.thetav
        self.out.deltatheta[t] = self.deltatheta
        self.out.deltathetav[t]= self.deltathetav
        self.out.wtheta[t]     = self.wtheta
        self.out.wthetav[t]    = self.wthetav
        self.out.wthetae[t]    = self.wthetae
        self.out.wthetave[t]   = self.wthetave
        
        self.out.q[t]          = self.q
        self.out.deltaq[t]     = self.deltaq
        self.out.wq[t]         = self.wq
        self.out.wqe[t]        = self.wqe
        self.out.wqM[t]        = self.wqM
      
        self.out.qsatvar[t]       = self.qsatvar
        self.out.e[t]          = self.e
        self.out.esatvar[t]       = self.esatvar
      
        fac = (self.rho*self.mco2)/self.mair
        self.out.CO2[t]        = self.CO2
        self.out.deltaCO2[t]   = self.deltaCO2
        self.out.wCO2[t]       = self.wCO2  * fac
        self.out.wCO2A[t]      = self.wCO2A * fac
        self.out.wCO2R[t]      = self.wCO2R * fac
        self.out.wCO2e[t]      = self.wCO2e * fac
        self.out.wCO2M[t]      = self.wCO2M * fac
        self.out.wCOS[t]       = self.wCOS
        if self.sw_ls:
            if self.ls_type=='ags':
                self.out.wCOSP[t]      = self.wCOSP
                self.out.wCOSS[t]      = self.wCOSS
        if self.ls_type=='canopy_model':
            self.out.wCOSS_molm2s[t]      = self.wCOSS_molm2s
            self.out.C_COS_veglayer_ppb[t]      = self.C_COS_veglayer_ppb
            self.out.C_CO2_veglayer_ppm[t]      = self.C_CO2_veglayer_ppm
            self.out.C_H2O_veglayer_pct[t]      = self.C_H2O_veglayer_pct
            if self.input.calc_sun_shad == False:
                self.out.gsco2_leaf[t] = self.canopy_model.gsco2_leaf
            else:
                self.out.gsco2_leaf_sun[t] = self.canopy_model.gsco2_leaf_sun
                self.out.gsco2_leaf_sha[t] = self.canopy_model.gsco2_leaf_sha
                self.out.PAR_sun_abs[t] = self.canopy_model.PAR_sun_abs
                self.out.PAR_sha_abs[t] = self.canopy_model.PAR_sha_abs
#                self.out.aapsun[t] = self.canopy_model.aapsun
#                self.out.aapsha[t] = self.canopy_model.aapsha
            self.out.rbveg_CO2[t] = self.canopy_model.rbveg_CO2
            self.out.ci_co2[t] = self.canopy_model.ci_co2
            self.out.CO2plantflux[t] = self.canopy_model.CO2plantflux
            self.out.COSplantflux[t] = self.canopy_model.COSplantflux
            self.out.cliq[t] = self.canopy_model.wetfveg
            self.out.fPARdif[t] = self.canopy_model.fPARdif
            self.out.enbalerr[t] = self.enbalerr
            self.out.U_veg[t] = self.canopy_model.U_veg
            self.out.Ds_veg[t] = self.canopy_model.Ds_veg
        
        self.out.COS[t]        = self.COS
        self.out.deltaCOS[t]   = self.deltaCOS

        self.out.u[t]          = self.u
        self.out.deltau[t]     = self.deltau
        self.out.uw[t]         = self.uw
        
        self.out.v[t]          = self.v
        self.out.deltav[t]     = self.deltav
        self.out.vw[t]         = self.vw
        
        self.out.T2m[t]        = self.T2m
        self.out.q2m[t]        = self.q2m
        self.out.u2m[t]        = self.u2m
        self.out.v2m[t]        = self.v2m
        self.out.e2m[t]        = self.e2m
        self.out.esat2m[t]     = self.esat2m
        if self.sw_sl:
            self.out.thetamh[t]    = self.thetamh
            self.out.thetamh2[t]   = self.thetamh2
            self.out.thetamh3[t]   = self.thetamh3
            self.out.thetamh4[t]   = self.thetamh4
            self.out.thetamh5[t]   = self.thetamh5
            self.out.thetamh6[t]   = self.thetamh6
            self.out.thetamh7[t]   = self.thetamh7
            self.out.Tmh[t]        = self.Tmh
            self.out.Tmh2[t]       = self.Tmh2
            self.out.Tmh3[t]       = self.Tmh3
            self.out.Tmh4[t]       = self.Tmh4
            self.out.Tmh5[t]       = self.Tmh5
            self.out.Tmh6[t]       = self.Tmh6
            self.out.Tmh7[t]       = self.Tmh7
            self.out.qmh[t]        = self.qmh
            self.out.qmh2[t]       = self.qmh2
            self.out.qmh3[t]       = self.qmh3
            self.out.qmh4[t]       = self.qmh4
            self.out.qmh5[t]       = self.qmh5
            self.out.qmh6[t]       = self.qmh6
            self.out.qmh7[t]       = self.qmh7
            self.out.COSmh[t]      = self.COSmh
            self.out.COSmh2[t]     = self.COSmh2
            self.out.COSmh3[t]     = self.COSmh3
            self.out.CO22m[t]      = self.CO22m
            self.out.CO2mh[t]      = self.CO2mh
            self.out.CO2mh2[t]     = self.CO2mh2
            self.out.CO2mh3[t]     = self.CO2mh3
            self.out.CO2mh4[t]     = self.CO2mh4
            self.out.COS2m[t]      = self.COS2m
            self.out.COSsurf[t]    = self.COSsurf
            self.out.CO2surf[t]    = self.CO2surf
            self.out.Tsurf[t]      = self.Tsurf
            self.out.Cm[t]         = self.Cm
        self.out.thetasurf[t]  = self.thetasurf
        self.out.thetavsurf[t] = self.thetavsurf
        self.out.qsurf[t]      = self.qsurf
        self.out.ustar[t]      = self.ustar
        self.out.Cs[t]         = self.Cs
        self.out.L[t]          = self.L
        self.out.Rib[t]        = self.Rib
  
        self.out.Swin[t]       = self.Swin
        self.out.Swout[t]      = self.Swout
        self.out.Lwin[t]       = self.Lwin
        self.out.Lwout[t]      = self.Lwout
        self.out.Q[t]          = self.Q
        self.out.sinlea[t]     = self.sinlea
  
        self.out.ra[t]         = self.ra
        self.out.rs[t]         = self.rs
        self.out.H[t]          = self.H
        self.out.LE[t]         = self.LE
        self.out.LEliq[t]      = self.LEliq
        self.out.LEveg[t]      = self.LEveg
        self.out.LEsoil[t]     = self.LEsoil
        self.out.LEpot[t]      = self.LEpot
        self.out.LEref[t]      = self.LEref
        self.out.G[t]          = self.G
        self.out.Ts[t]         = self.Ts

        self.out.zlcl[t]       = self.lcl
        self.out.RH_h[t]       = self.RH_h

        self.out.ac[t]         = self.ac
        self.out.M[t]          = self.M
        self.out.dz[t]         = self.dz_h
        
        if self.checkpoint:
            self.cpx[self.t]['sto_fac_end'] = fac
        if self.save_vars_indict:
            the_locals = cp.deepcopy(locals()) #to prevent error 'dictionary changed size during iteration'
            for variablename in the_locals: #note that the self variables are not included
                if str(variablename) != 'self':
                    self.vars_sto.update({variablename: the_locals[variablename]})
  
    # delete class variables to facilitate analysis in ipython
    def exitmodel(self):
        #this list is not complete
        del(self.Lv)
        del(self.cp)
        del(self.rho)
        del(self.k)
        del(self.g)
        del(self.Rd)
        del(self.Rv)
        del(self.bolz)
        del(self.S0)
        del(self.rhow)
  
        del(self.t)
        del(self.dt)
        del(self.tsteps)
         
        del(self.h)          
        del(self.Ps)        
        del(self.fc)        
        del(self.ws)
        del(self.we)
        
        del(self.theta)
        del(self.deltatheta)
        del(self.gammatheta)
        del(self.gammatheta2)
        del(self.advtheta)
        del(self.beta)
        del(self.wtheta)
    
        del(self.T2m)
        del(self.q2m)
        del(self.e2m)
        del(self.esat2m)
        del(self.u2m)
        del(self.v2m)
        
        del(self.thetasurf)
        del(self.qsatsurf)
        del(self.thetav)
        del(self.deltathetav)
        del(self.thetavsurf)
        del(self.qsurf)
        del(self.wthetav)
        
        del(self.q)
        del(self.qsatvar)
        del(self.dqsatdT)
        del(self.e)
        del(self.esatvar)
        del(self.deltaq)
        del(self.gammaq)
        del(self.advq)
        del(self.wq)
        
        del(self.u)
        del(self.deltau)
        del(self.gammau)
        del(self.advu)
        
        del(self.v)
        del(self.deltav)
        del(self.gammav)
        del(self.advv)
  
        del(self.htend)
        del(self.thetatend)
        del(self.deltathetatend)
        del(self.qtend)
        del(self.deltaqtend)
        del(self.utend)
        del(self.deltautend)
        del(self.vtend)
        del(self.deltavtend)
     
        del(self.Tsoiltend) 
        del(self.wgtend)  
        del(self.Wltend) 
  
        del(self.ustar)
        del(self.uw)
        del(self.vw)
        del(self.z0m)
        del(self.z0h) 
        if self.sw_sl:
            del(self.Cm)            
        del(self.Cs)
        del(self.L)
        del(self.Rib)
        del(self.ra)
  
        del(self.lat)
        del(self.lon)
        del(self.doy)
        del(self.tstart)
   
        del(self.Swin)
        del(self.Swout)
        del(self.Lwin)
        del(self.Lwout)
        del(self.cc)
  
        del(self.wg)
        del(self.w2)
        del(self.cveg)
        del(self.cliq)
        del(self.Tsoil)
        del(self.T2)
        del(self.a)
        del(self.b)
        del(self.p)
        del(self.CGsat)
  
        del(self.wsat)
        del(self.wfc)
        del(self.wwilt)
  
        del(self.C1sat)
        del(self.C2ref)
  
        del(self.LAI)
        del(self.rs)
        del(self.rssoil)
        del(self.rsmin)
        del(self.rssoilmin)
        del(self.alpha)
        del(self.gD)
  
        del(self.Ts)
  
        del(self.Wmax)
        del(self.Wl)
  
        del(self.Lambda)
        
        del(self.Q)
        del(self.sinlea)
        del(self.H)
        del(self.LE)
        del(self.LEliq)
        del(self.LEveg)
        del(self.LEsoil)
        del(self.LEpot)
        del(self.LEref)
        del(self.G)
  
        del(self.sw_ls)
        del(self.sw_rad)
        del(self.sw_sl)
        del(self.sw_wind)
        del(self.sw_shearwe)

# class for storing mixed-layer model output data
class model_output:
    def __init__(self,model, tsteps):
        self.t          = np.zeros(tsteps)    # time [s]

        # mixed-layer variables
        self.h          = np.zeros(tsteps)    # ABL height [m]
        
        self.theta      = np.zeros(tsteps)    # initial mixed-layer potential temperature [K]
        self.thetav     = np.zeros(tsteps)    # initial mixed-layer virtual potential temperature [K]
        self.deltatheta = np.zeros(tsteps)    # initial potential temperature jump at h [K]
        self.deltathetav= np.zeros(tsteps)    # initial virtual potential temperature jump at h [K]
        self.wtheta     = np.zeros(tsteps)    # surface kinematic heat flux [K m s-1]
        self.wthetav    = np.zeros(tsteps)    # surface kinematic virtual heat flux [K m s-1]
        self.wthetae    = np.zeros(tsteps)    # entrainment kinematic heat flux [K m s-1]
        self.wthetave   = np.zeros(tsteps)    # entrainment kinematic virtual heat flux [K m s-1]
        
        self.q          = np.zeros(tsteps)    # mixed-layer specific humidity [kg kg-1]
        self.deltaq         = np.zeros(tsteps)    # initial specific humidity jump at h [kg kg-1]
        self.wq         = np.zeros(tsteps)    # surface kinematic moisture flux [kg kg-1 m s-1]
        self.wqe        = np.zeros(tsteps)    # entrainment kinematic moisture flux [kg kg-1 m s-1]
        self.wqM        = np.zeros(tsteps)    # cumulus mass-flux kinematic moisture flux [kg kg-1 m s-1]

        self.qsatvar       = np.zeros(tsteps)    # mixed-layer saturated specific humidity [kg kg-1]
        self.e          = np.zeros(tsteps)    # mixed-layer vapor pressure [Pa]
        self.esatvar       = np.zeros(tsteps)    # mixed-layer saturated vapor pressure [Pa]

        self.CO2        = np.zeros(tsteps)    # mixed-layer CO2 [ppm]
        self.deltaCO2   = np.zeros(tsteps)    # initial CO2 jump at h [ppm]
        self.wCO2       = np.zeros(tsteps)    # surface total CO2 flux [mgCO2 m-2 s-1]
        self.wCO2A      = np.zeros(tsteps)    # surface assimilation CO2 flux [mgCO2 m-2 s-1]
        self.wCO2R      = np.zeros(tsteps)    # surface respiration CO2 flux [mgCO2 m-2 s-1]
        self.wCO2e      = np.zeros(tsteps)    # entrainment CO2 flux [mgCO2 m-2 s-1]
        self.wCO2M      = np.zeros(tsteps)    # CO2 mass flux [mgCO2 m-2 s-1]
        
        self.COS        = np.zeros(tsteps)    # mixed-layer COS [ppb]
        self.deltaCOS   = np.zeros(tsteps)    # mixed-layer COS jump at h [ppb]
        self.wCOS       = np.zeros(tsteps)    # COS surface flux ppb m s-1
        self.wCOSP      = np.zeros(tsteps)    # COS surface flux plant ppb m s-1
        self.wCOSS      = np.zeros(tsteps)    # COS surface flux soil ppb m s-1
        self.wCOSS_molm2s      = np.zeros(tsteps)    # COS surface flux soil mol m-2 s-1
        
        
        #canopy model 
        if model.input.ls_type == 'canopy_model':
            self.C_COS_veglayer_ppb = np.zeros((tsteps,model.input.nr_nodes_veg)) # mixing ratio COS in veglayers [ppb]
            self.C_CO2_veglayer_ppm = np.zeros((tsteps,model.input.nr_nodes_veg)) # mixing ratio CO2 in veglayers [ppm]
            self.C_H2O_veglayer_pct = np.zeros((tsteps,model.input.nr_nodes_veg)) #mixing ratio water in veglayers [%]
            self.gsco2_leaf = np.zeros((tsteps,model.input.nr_nodes_veg)) #stomatal conductance on the leaf scale calculated by canopy model(m s-1)
            self.ci_co2 = np.zeros((tsteps,model.input.nr_nodes_veg)) #internal co2 concentration calculated by canopy model(mg CO2 m-3)
            self.CO2plantflux = np.zeros((tsteps,model.input.nr_nodes_veg)) #co2 plant uptake calculated by canopy model(mg CO2 m-3)
            self.COSplantflux = np.zeros((tsteps,model.input.nr_nodes_veg)) #COS plant uptake  calculated by canopy model(mg CO2 m-3)
            self.rbveg_CO2 = np.zeros((tsteps,model.input.nr_nodes_veg)) #leaf boundary layer resistance (s m-1)
            self.gsco2_leaf_sun = np.zeros((tsteps,model.input.nr_nodes_veg)) #stomatal conductance at leaf scale of sunlit leaves (m s-1)
            self.gsco2_leaf_sha = np.zeros((tsteps,model.input.nr_nodes_veg)) #stomatal conductance at leaf scale of shaded leaves (m s-1)
            self.PAR_sun_abs = np.zeros((tsteps,model.input.nr_nodes_veg)) #PAR absorbed by sunlit leaves (W / m2 leaf)
            self.PAR_sha_abs = np.zeros((tsteps,model.input.nr_nodes_veg)) #PAR absorbed by shaded leaves (W / m2 leaf)
            self.cliq = np.zeros((tsteps,model.input.nr_nodes_veg)) #wet fraction of vegetation
            self.enbalerr = np.zeros(tsteps) #energy balance closure error
            self.fPARdif = np.zeros(tsteps) #fraction of PAR that is diffuse
            self.U_veg = np.zeros((tsteps,model.input.nr_nodes_veg)) #wind speed canopy model
            self.Ds_veg = np.zeros((tsteps,model.input.nr_nodes_veg)) #vapour pressure deficit (kPa) at the surface of the vegetation in the layers of the canopy model
        
        self.u          = np.zeros(tsteps)    # initial mixed-layer u-wind speed [m s-1]
        self.deltau     = np.zeros(tsteps)    # initial u-wind jump at h [m s-1]
        self.uw         = np.zeros(tsteps)    # surface momentum flux u [m2 s-2]
        
        self.v          = np.zeros(tsteps)    # initial mixed-layer u-wind speed [m s-1]
        self.deltav         = np.zeros(tsteps)    # initial u-wind jump at h [m s-1]
        self.vw         = np.zeros(tsteps)    # surface momentum flux v [m2 s-2]

        # diagnostic meteorological variables
        self.T2m        = np.zeros(tsteps)    # 2m temperature [K]
        self.q2m        = np.zeros(tsteps)    # 2m specific humidity [kg kg-1]
        self.u2m        = np.zeros(tsteps)    # 2m u-wind [m s-1]    
        self.v2m        = np.zeros(tsteps)    # 2m v-wind [m s-1]    
        self.e2m        = np.zeros(tsteps)    # 2m vapor pressure [Pa]
        self.esat2m     = np.zeros(tsteps)    # 2m saturated vapor pressure [Pa]

        
        if model.sw_sl:
            self.thetamh    = np.zeros(tsteps)    # pot temperature at measuring height [K]
            self.thetamh2    = np.zeros(tsteps)    # pot temperature at measuring height2 [K]
            self.thetamh3    = np.zeros(tsteps)    # pot temperature at measuring height3 [K]
            self.thetamh4    = np.zeros(tsteps)    # pot temperature at measuring height4 [K]
            self.thetamh5    = np.zeros(tsteps)    # pot temperature at measuring height5 [K]
            self.thetamh6    = np.zeros(tsteps)    # pot temperature at measuring height6 [K]
            self.thetamh7    = np.zeros(tsteps)    # pot temperature at measuring height7 [K]
            self.Tmh        = np.zeros(tsteps)    # temperature at measuring height [K]
            self.Tmh2        = np.zeros(tsteps)    # temperature at measuring height2 [K]
            self.Tmh3       = np.zeros(tsteps)    # temperature at measuring height3 [K]
            self.Tmh4       = np.zeros(tsteps)    # temperature at measuring height4 [K]
            self.Tmh5       = np.zeros(tsteps)    # temperature at measuring height5 [K]
            self.Tmh6       = np.zeros(tsteps)    # temperature at measuring height6 [K]
            self.Tmh7       = np.zeros(tsteps)    # temperature at measuring height7 [K]
            self.COSmh      = np.zeros(tsteps)    # COS at measuring height [ppb]
            self.COSmh2     = np.zeros(tsteps)    # COS at measuring height2 [ppb]
            self.COSmh3     = np.zeros(tsteps)    # COS at measuring height3 [ppb]
            self.CO2mh      = np.zeros(tsteps)    # CO2 at measuring height [ppb]
            self.CO2mh2     = np.zeros(tsteps)    # CO2 at measuring height2 [ppb]
            self.CO2mh3     = np.zeros(tsteps)    # CO2 at measuring height3 [ppb]
            self.CO2mh4     = np.zeros(tsteps)    # CO2 at measuring height4 [ppb]
            self.COS2m      = np.zeros(tsteps)    # COS at 2m height [ppb]
            self.CO22m      = np.zeros(tsteps)    # CO2 at 2m height [ppm]
            self.COSsurf    = np.zeros(tsteps)    # COS at the surface [ppb]
            self.CO2surf    = np.zeros(tsteps)    # CO2 at the surface [ppm]
            self.Tsurf      = np.zeros(tsteps)    # Temp at the surface [K]
            self.qmh        = np.zeros(tsteps)    # specific humidity at measuring height [kg kg-1]
            self.qmh2       = np.zeros(tsteps)    # specific humidity at measuring height 2 [kg kg-1]
            self.qmh3       = np.zeros(tsteps)    # specific humidity at measuring height 3 [kg kg-1]
            self.qmh4       = np.zeros(tsteps)    # specific humidity at measuring height 4 [kg kg-1]
            self.qmh5       = np.zeros(tsteps)    # specific humidity at measuring height 5 [kg kg-1]
            self.qmh6       = np.zeros(tsteps)    # specific humidity at measuring height 6 [kg kg-1]
            self.qmh7       = np.zeros(tsteps)    # specific humidity at measuring height 7 [kg kg-1]
            self.Cm         = np.zeros(tsteps)    # drag coefficient for momentum []
                
            
        # surface-layer variables
        self.thetasurf  = np.zeros(tsteps)    # surface potential temperature [K]
        self.thetavsurf = np.zeros(tsteps)    # surface virtual potential temperature [K]
        self.qsurf      = np.zeros(tsteps)    # surface specific humidity [kg kg-1]
        self.ustar      = np.zeros(tsteps)    # surface friction velocity [m s-1]
        self.z0m        = np.zeros(tsteps)    # roughness length for momentum [m]
        self.z0h        = np.zeros(tsteps)    # roughness length for scalars [m]
        self.Cs         = np.zeros(tsteps)    # drag coefficient for scalars []
        self.L          = np.zeros(tsteps)    # Obukhov length [m]
        self.Rib        = np.zeros(tsteps)    # bulk Richardson number [-]

        # radiation variables
        self.Swin       = np.zeros(tsteps)    # incoming short wave radiation [W m-2]
        self.Swout      = np.zeros(tsteps)    # outgoing short wave radiation [W m-2]
        self.Lwin       = np.zeros(tsteps)    # incoming long wave radiation [W m-2]
        self.Lwout      = np.zeros(tsteps)    # outgoing long wave radiation [W m-2]
        self.Q          = np.zeros(tsteps)    # net radiation [W m-2]
        self.sinlea     = np.zeros(tsteps)    #sine of solar elevation angle

        # land surface variables
        self.ra         = np.zeros(tsteps)    # aerodynamic resistance [s m-1]
        self.rs         = np.zeros(tsteps)    # surface resistance [s m-1]
        self.H          = np.zeros(tsteps)    # sensible heat flux [W m-2]
        self.LE         = np.zeros(tsteps)    # evapotranspiration [W m-2]
        self.LEliq      = np.zeros(tsteps)    # open water evaporation [W m-2]
        self.LEveg      = np.zeros(tsteps)    # transpiration [W m-2]
        self.LEsoil     = np.zeros(tsteps)    # soil evaporation [W m-2]
        self.LEpot      = np.zeros(tsteps)    # potential evaporation [W m-2]
        self.LEref      = np.zeros(tsteps)    # reference evaporation at rs = rsmin / LAI [W m-2]
        self.G          = np.zeros(tsteps)    # ground heat flux [W m-2]
        self.Ts         = np.zeros(tsteps)    # Skin temperature [K]

        # Mixed-layer top variables
        self.zlcl       = np.zeros(tsteps)    # lifting condensation level [m]
        self.RH_h       = np.zeros(tsteps)    # mixed-layer top relative humidity [-]

        # cumulus variables
        self.ac         = np.zeros(tsteps)    # cloud core fraction [-]
        self.M          = np.zeros(tsteps)    # cloud core mass flux [m s-1]
        self.dz         = np.zeros(tsteps)    # transition layer thickness [m]

# class for storing mixed-layer model input data
class model_input:
    def __init__(self):
        # general model variables
        self.runtime    = None  # duration of model run [s]
        self.dt         = None  # time step [s]

        # mixed-layer variables
        self.sw_ml      = None  # mixed-layer model switch
        self.sw_shearwe = None  # Shear growth ABL switch
        self.sw_fixft   = None  # Fix the free-troposphere switch
        self.h          = None  # initial ABL height [m]
        self.Ps         = None  # surface pressure [Pa]
        self.divU       = None  # horizontal large-scale divergence of wind [s-1]
        self.fc         = None  # Coriolis parameter [s-1]
        
        self.theta      = None  # initial mixed-layer potential temperature [K]
        self.deltatheta = None  # initial temperature jump at h [K]
        self.gammatheta = None  # free atmosphere potential temperature lapse rate [K m-1]
        self.advtheta   = None  # advection of heat [K s-1]
        self.beta       = None  # entrainment ratio for virtual heat [-]
        self.wtheta     = None  # surface kinematic heat flux [K m s-1]
        
        self.q          = None  # initial mixed-layer specific humidity [kg kg-1]
        self.deltaq     = None  # initial specific humidity jump at h [kg kg-1]
        self.gammaq     = None  # free atmosphere specific humidity lapse rate [kg kg-1 m-1]
        self.advq       = None  # advection of moisture [kg kg-1 s-1]
        self.wq         = None  # surface kinematic moisture flux [kg kg-1 m s-1]

        self.CO2        = None  # initial mixed-layer potential temperature [K]
        self.deltaCO2   = None  # initial temperature jump at h [K]
        self.gammaCO2   = None  # free atmosphere potential temperature lapse rate [K m-1]
        self.gammaCOS   = None
        self.advCO2     = None  # advection of heat [K s-1]
        self.wCO2       = None  # surface kinematic heat flux [K m s-1]
        self.wCOS       = None  # surface kinematic COS flux [ppb m s-1]
        
        self.sw_wind    = None  # prognostic wind switch
        self.u          = None  # initial mixed-layer u-wind speed [m s-1]
        self.deltau     = None  # initial u-wind jump at h [m s-1]
        self.gammau     = None  # free atmosphere u-wind speed lapse rate [s-1]
        self.advu       = None  # advection of u-wind [m s-2]

        self.v          = None  # initial mixed-layer u-wind speed [m s-1]
        self.deltav     = None  # initial u-wind jump at h [m s-1]
        self.gammav     = None  # free atmosphere v-wind speed lapse rate [s-1]
        self.advv       = None  # advection of v-wind [m s-2]

        # surface layer variables
        self.sw_sl      = None  # surface layer switch
        self.ustar      = None  # surface friction velocity [m s-1]
        self.z0m        = None  # roughness length for momentum [m]
        self.z0h        = None  # roughness length for scalars [m]
        self.Cs         = None  # drag coefficient for scalars [-]
        self.Rib        = None  # bulk Richardson number [-]

        # radiation parameters
        self.sw_rad     = None  # radiation switch
        self.lat        = None  # latitude [deg]
        self.lon        = None  # longitude [deg]
        self.doy        = None  # day of the year [-]
        self.tstart     = None  # time of the day [h UTC]
        self.cc         = None  # cloud cover fraction [-]
        self.Q          = None  # net radiation [W m-2] 
        self.dFz        = None  # cloud top radiative divergence [W m-2] 

        # land surface parameters
        self.sw_ls      = None  # land surface switch
        self.ls_type    = None  # land-surface parameterization ('js' for Jarvis-Stewart or 'ags' for A-Gs)
        self.wg         = None  # volumetric water content top soil layer [m3 m-3]
        self.w2         = None  # volumetric water content deeper soil layer [m3 m-3]
        self.Tsoil      = None  # temperature top soil layer [K]
        self.T2         = None  # temperature deeper soil layer [K]
        
        self.a          = None  # Clapp and Hornberger retention curve parameter a
        self.b          = None  # Clapp and Hornberger retention curve parameter b
        self.p          = None  # Clapp and Hornberger retention curve parameter p 
        self.CGsat      = None  # saturated soil conductivity for heat
        
        self.wsat       = None  # saturated volumetric water content ECMWF config [-]
        self.wfc        = None  # volumetric water content field capacity [-]
        self.wwilt      = None  # volumetric water content wilting point [-]
        
        self.C1sat      = None 
        self.C2ref      = None

        self.c_beta     = None  # Curvatur plant water-stress factor (0..1) [-]
        
        self.LAI        = None  # leaf area index [-]
        self.gD         = None  # correction factor transpiration for VPD [-]
        self.rsmin      = None  # minimum resistance transpiration [s m-1]
        self.rssoilmin  = None  # minimum resistance soil evaporation [s m-1]
        self.alpha      = None  # surface albedo [-]
        
        self.Ts         = None  # initial surface temperature [K]
        
        self.cveg       = None  # vegetation fraction [-]
        self.Wmax       = None  # thickness of water layer on wet vegetation [m]
        self.Wl         = None  # equivalent water layer depth for wet vegetation [m]
        
        self.Lambda     = None  # thermal diffusivity skin layer [-]

        # A-Gs parameters
        self.c3c4       = None  # Plant type ('c3' or 'c4')

        # Cumulus parameters
        self.sw_cu      = None  # Cumulus parameterization switch
        self.dz_h       = None  # Transition layer thickness [m]