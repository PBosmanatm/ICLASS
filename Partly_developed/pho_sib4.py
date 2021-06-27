import numpy as np
from netCDF4 import Dataset
import copy as cp

#for the canopy model we take the same system as for the soil model (see Sun et al. 2015)
#the nodes now start from near the soil surface upward towards the top of the canopy.

def esat(T):
    return 0.611e3 * np.exp(17.2694 * (T - 273.16) / (T - 35.86)) #Tetens formula, see intro atm

class pho_sib4:

    def __init__(self,dtsib,CLASSPs,CLASSc3c4,CLASSLAI,CLASSCO2,CLASStheta,CLASSq,CLASSe,inputdata=None,mothermodel=None):
        #CLASSCO2 in ppm, CLASSPs in Pa
        ###################################
        ##### initialise dictionaries######
        ###################################
        self.aerovar = {}
        self.gprogt = {}
        self.gdiagt = {}
        self.gdiagt['bps'] = np.zeros(3)
        self.vegt ={}
        self.physcont = {}
        self.co2t = {}
        self.fluxt = {}
        self.hydrost = {}
        self.cast = {}
        self.radt = {}
        self.sscolt = {}
        self.sscolt['td'] = np.zeros(3)
        self.soilt = {}
        if mothermodel != None: #mothermodel is class model calling the canopy model
            self.mothermodel = mothermodel
            self.pressure = CLASSPs #Pa
            ###################################
            #####constants for phosib.f90######
            ###################################
            self.p0_sfc = 1.0e+05 #module_pparams.f90
            self.dtsib = dtsib
            self.tice   = 273.15 #freezing temp of water, #module_pparams.f90
            self.par_conv     = 4.6E-6#module pparams.f90, PAR conversion factor (umol quanta/J)
            self.minrad=1.e-6 #module_phosib.f90
            self.toatosfc = 0.8 #module_phosib.f90
            self.vmax_tref=298. #module_phosib.f90
            self.vmax_q10=2.1#module_phosib.f90
            self.minassim = 1.e-7 #module_phosib.f90
            self.minassimdiff = 10.e-6 #module_phosib.f90
            self.rstar = 8.3143,       #universal gas constant (m3 Pa/mol/K),, #module_pparams.f90
            self.zln2 = 6.9314718e-1 #see module_phosib.f90
            self.ghalf = 1.0257068e1
            self.dttin = 3.6e3
            self.dmin = 6.0e1
            #depending on input:
            c3c4 = CLASSc3c4
            if c3c4 == 'c3': #module_param.f90
                self.physcont['c4flag'] = 0
            elif c3c4 == 'c4':
                self.physcont['c4flag'] = 1
            self.physcont['atheta'] = 0.98 #sib_phys.dat
            self.physcont['btheta'] = 0.95 #sib_phys.dat
            self.physcont['effcon'] = 0.080 #sib_phys.dat
            self.physcont['wssp'] = 0.1 #sib_phys.dat
            self.physcont['tran'] = np.zeros((2+1,2+1))#! leaf transmittance (-), (1,1) - shortwave, green plants, (1,2) - shortwave, brown plants,  (2,1) - longwave, green plants, (2,2) - longwave, brown plants (module_param.f90)
            self.physcont['tran'][1,1] = 0.050 #sib_phys.dat
            self.physcont['tran'][1,2] = 0.001 #sib_phys.dat
            self.physcont['tran'][2,1] = 0.1 #sib_phys.dat
            self.physcont['tran'][2,2] = 0.001 #sib_phys.dat
            self.physcont['ref'] = np.zeros((2+1,2+1))#leaf reflectance, same categories as above
            self.physcont['ref'][1,1] = 0.080
            self.physcont['ref'][1,2] = 0.160
            self.physcont['ref'][2,1] = 0.35
            self.physcont['ref'][2,2] = 0.410
            self.physcont['slti'] = 0.95
            self.physcont['hlti'] = 270.0
            self.physcont['shti'] = 0.3
            self.physcont['hhti'] = 317 #sib_phys.dat
            self.physcont['sfti'] = 0.35
            self.physcont['hfti'] = 267.0
            self.physcont['binter'] = 0.010 #Conductance-photosynthesis intercept (mol m^-2 sec^-1), sib_phys.dat
            self.physcont['z1'] = 8.50 #sib_phys.dat
            self.physcont['z2'] = 17.00 #sib_phys.dat
            self.physcont['gradm'] = 9.0 #sib_phys.dat
            self.physcont['gmeso'] = 4000 #sib_phys.dat
            co2_casd_min = 4.0 #CAS depth minimum for CO2 (m) module_phosib.f90
            self.co2t['casd'] = max(co2_casd_min, self.physcont['z2'])
            ######################################
            #####various constants################
            ######################################
            self.tcbot = 253.15 #module_oparams.f90
            self.po2m   = 20900.0 #mixed layer O2 concentration module pparams.f90
            self.oms_q10=1.8    #Export-limited assimilation, module_phosib.f90
            self.amagatwv     = 44.032476    #water-vapor adjusted amagat (mol/m3), module_pparams.f90
            self.delta        = 0.608        #molecular_weight_air/molecular_weight_water - 1, module pparams.f90
            self.vkrmn        = 0.35                # Von Karmann's constant (unitless), module pparams.f90
            self.cstabl = 10.0 #module_oparams.f90  
            self.bstabl =  8.0 #module_oparams.f90 
            self.grav         = 9.81#earth gravity (m/s^2) #module pparams.f90
            self.bunstablM = 10.0 #Constants for surface flux functions, module oparams.f90
            self.bunstablT = 15.0
            self.cunstablM = 75.0
            self.cunstablT = 75.0
            self.perih = 1.7963 # TOA solar parameter #module oparams.f90
            self.eccn        = 0.016715 #eccentricity #module pparams.f90
            self.numic = int(6) #module_phosib.f90
            self.spec_heat_cp = 1005.0 #module_pparams.f90 #specific heat at constant pressure (J/kg/K)
            self.twmin = 173.16    #lowest allowed temperature boundary for water (K) #module pparams.f90
            self.twmax  = 373.16    #highest allowed temperature boundary for water (K) #module pparams.f90
            self.tsref  = 373.15   #reference temperature (K) #module pparams.f90
            self.psref  = 1013.246 #reference pressure (hPa) #module pparams.f90
            self.lvap  = 2.25e+06       #latent heat of vaporization (J/kg) #module pparams.f90
            self.rv      = 4.61e+02     #gas constant for water vapor #module pparams.f90
            self.molc_h2o     = 55.56           #number of moles per liter of water (mol/l) #module pparams.f90
            self.rhfac_exp_crop = 0.7 #Humidity stress crop exponent (-), module_phosib.f90
            self.rhfac_astart = 0.6 #Humidity stress curvature start (-), module_phosib.f90
            self.rhfac_exp = 2.2 #Humidity stress curvature (-), module_phosib.f90
            self.rhfac_nforest = 0.7 #Humidity stress min for needle forests (-), module_phosib.f90
            self.rhfac_tundra = 0.6     # Humidity stress min for tundra (-), module_phosib.f90
            self.gas_const_r  = 287.0              # gas constant for dry air (J/kg/K),module_pparams.f90
            self.rgfac        = 100.0/self.gas_const_r   #module_pparams.f90
            self.stefan       = 5.67e-08     #stefan boltzmann constant (W/m^2/K^4) #module_pparams.f90
            self.denh2o = 1000.0 #density of water (kg/m^3) #module_pparams.f90
            self.gwlim  = 10.00 #ground water storage limit (kg/m2) #module_pparams.f90
            self.h2ohc  = 4.186*1000.0*1000.0   #water heat capacity (J/deg/m3)#module_pparams.f90
            self.denice = 917.0 #density of ice (kg/m^3) #module_pparams.f90
            self.leafhc = 4.186*1000.0*0.2 #leaf heat capacity  (J/deg/m3)
            self.cv     = 1952.0 #specific heat of water vapor at constant pressure (J/deg/kg)
            self.kappa  = self.gas_const_r/self.spec_heat_cp #constant for pot temp conversion
            self.solar_const = 1367.0 #solar constant (W/m2), module pparams.f90
            self.canlai_min = 0.01 #Minimum LAI to use canopy water storage, module_oparams.f90
            snomel  = 3.705185e+08 #latent heat of fusion of ice (J/m3) 
            self.snofac = self.lvap/(self.lvap + snomel * 1.e-3) #ratio from Sellers (1986)
            #depending on input and simulation:
            self.physcont['ChiL'] = 0.01 #sib_phys.dat
            self.physcont['laisat'] = 5.5 #sib_phys.dat
            self.physcont['fparsat'] = 0.98#sib_phys.dat
            self.vegt['lai'] = CLASSLAI
            self.vegt['lait'] = cp.deepcopy(self.vegt['lai']) #this is not in the sib4 code!!)
            self.iseforest = True #for phostress.f90 line 131, I assume it means evergreen forest
            self.iscrop = False
            self.isgrass = False
            self.isnforest = True #needleleaf forest
            self.vmax0 = 0.1E-4#read_pgdd.f90
            secs_per_day = 86400
            steps_per_day = secs_per_day / dtsib
            self.wt_daily = 1./(steps_per_day)
            self.dtisib = 1.0/dtsib #time_init.f90, kinds.f90
            self.hydrost['satfrac'] = 0. #total fraction of water saturation in soil column (-)
            self.pft_c3a = 13 #♪info_pft.dat
            self.ztemp = 100. #temperature measurement height (m), module_io.f90
            self.zwind = 100. #wind measurement height (m), module_io.f90
            self.pnum = 2 #plant functional type number
            self.cast['tc'] = CLASStheta * (self.pressure/100000)**(self.gas_const_r/self.spec_heat_cp) + 1.0 #restart_read.f90
            self.cast['tcas'] = CLASStheta * (self.pressure/100000)**(self.gas_const_r/self.spec_heat_cp) #in Sib4 they might actually use 2m temp instead
            self.hydrost['wetfracc'] = 0. #assume dry canopy
            self.cast['shcas'] = CLASSq#see restart_read.f90
            ##################################################
            #initialisations of variables, also for phosib.f90
            if hasattr(inputdata,'co2t'):
                if 'pco2cas' in inputdata.co2t:
                    self.co2t['pco2cas'] = inputdata.co2t['pco2cas']
                else:
                    self.co2t['pco2cas'] = CLASSCO2 * 1.e-6 * self.pressure# CAS CO2 partial pressure (Pa)
                if 'pco2c' in inputdata.co2t:
                    self.co2t['pco2c'] = inputdata.co2t['pco2c']
                else:
                    self.co2t['pco2c'] = 400e-6 * self.pressure
                if 'pco2i' in inputdata.co2t:
                    self.co2t['pco2i'] = inputdata.co2t['pco2i']
                else:
                    self.co2t['pco2i'] = 400e-6 * self.pressure
                if 'pco2s' in inputdata.co2t:
                    self.co2t['pco2s'] = inputdata.co2t['pco2s']
                else:
                    self.co2t['pco2s'] = 400e-6 * self.pressure
                if 'rst' in inputdata.co2t:
                    self.co2t['rst'] = inputdata.co2t['rst'] #prognostic stomatal resistance, s/m
                else:
                    self.co2t['rst'] = 5.e6 #module_phosib.f90,phonveg.f90
            else:
                raise Exception('provide co2t inputdata!')
            

            self.pawfzw = inputdata.pawfzw #Soil-layer depth-weighted PAW fraction (kg/m2), PAW is Plant Available Water, see module_sib.f90
            self.tawfrw = inputdata.tawfrw  #Root-weighted TAW in soil column (kg/m2), TAW is total available water
            self.cast['eacas'] = CLASSe / 100. #hPa
            self.soref_vis = inputdata.soref_vis
            self.soref_nir = inputdata.soref_nir
            self.clayfrac = inputdata.clayfrac
            self.sandfrac = inputdata.sandfrac
            self.vegt['green'] = inputdata.vegt['green'] #green fraction of LAI (-)
        else:
            #if runned stand alone, adapt values here
            self.c3c4 = 'c3'
    
    
    def setup_soilt(self,clayfrac, sandfrac,soref_vis, soref_nir,fc_min, wp_min,soilt):
        soilt['clayfrac'] = clayfrac
        soilt['sandfrac'] = sandfrac
        soilt['soref_vis'] = soref_vis
        soilt['soref_nir'] = soref_nir
        sfrac = sandfrac*100.
        soilt['poros'] = 0.489-0.00126*sfrac
    
    def gmuder(self,lat, doy, ChiL,CLASScc): #mapper.f90
        
        cloud=CLASScc #0.5 in Fortran SIb4
        # Calculate solar declination in radians
        pi180 = np.pi / 180 #module_pparams.f90
        dec=pi180*23.5*np.sin(1.72e-2*(doy-80))
    
        # Calculate sine and cosine of solar declination
        sin_dec=np.sin(dec)                                                         
        cos_dec=np.cos(dec)
        topint=0.
        botint=0.
        for itime in range(1, 48+1):
        # Calculate time from zero Greenwhich Mean Time (GMT)
            
            gtime=0.5*itime 
    
            # Calculate cosine of hour angle of Grenwhich Meridion (GM)
            coshr=np.cos(-np.pi+gtime/24.*2.*np.pi)
    
            # Calculate cosine of the Sun angle (mu)
            #     longitude=GM=0 degrees, latitude=Lat
            mu=np.sin(lat*pi180)*sin_dec+np.cos(lat*pi180)*cos_dec*coshr
    
            # Ensure the cosine of Sun angle is positive, but not zero
            #     e.g., daylight, sun angle<=89.4 degrees (about start disc set/rise)
            mu=max(0.01,mu) 
    
            # It looks like this code calculates the direct and difracted PAR based
            # on the solar constant and a cloud fraction of 0.5 at the top and
            # bottom of the atmosphere.  During night, mu=0.01, a constant.  These
            # calculations do not match the definition of G(mu)/mu described in 
            # Bonan (1996) and Sellers (1985).
            tor    = 0.7**(1./mu)
            swdown = 1375.*mu*(tor+0.271-0.294*tor)
            difrat = 0.0604/(mu-0.0223)+0.0683
            difrat = max(difrat,0.)
            difrat = min(difrat,1.)
            difrat = difrat+(1.-difrat)*cloud
            vnrat  = (580.-cloud*464.)/((580.-cloud*499.) + (580.-cloud*464.))
            pardir = (1.-difrat)*vnrat*swdown
            pardif = difrat*vnrat*swdown
            topint = topint+pardir*mu+pardif*0.5
            botint = botint+pardir+pardif
        fb=topint/botint
        chiv=ChiL                                                               
        if (abs(chiv) <= 0.01):
            chiv=0.01
        aa=0.5-0.633*chiv-0.33*chiv*chiv
        bb=0.877*(1.-2.*aa)
        gmudmu=(aa+bb*fb)/fb
        return gmudmu
         
    def raddrv(self,swdown,sunang):#driver_interp.f90
        rad_c5 = 1160.#module oparams.f90
        rad_c4 = 963.
        rad_c1 = 580.
        rad_c2 = 464.
        rad_c3 = 499.
        localcosz = max( 0.001,  sunang) #cos(sza) defined as sunangle in the f90 file, this can be derived from mapper.f90, from the call to raddrv in driver_interp.f90 and p33 technical description
        
        stemp = swdown
        stemp = max(stemp,0.01)
    
        cloud = (rad_c5 * localcosz - stemp) / (rad_c4 * localcosz)                   
        cloud = max(cloud,0.)                                                
        cloud = min(cloud,1.)                                                  
    
        difrat = 0.0604 / (sunang-0.0223 + 1.0e-10 ) + 0.0683
        if (difrat < 0. ):
            difrat = 0.
        if (difrat > 1. ):
            difrat = 1.
        difrat = difrat + ( 1. - difrat ) * cloud
    
        vnrat = ( rad_c1 - cloud*rad_c2 ) / ( ( rad_c1 - cloud*rad_c3 ) + ( rad_c1 - cloud*rad_c2 ) )
    
        radvbc = (1.-difrat)*vnrat*stemp
        radvdc = difrat*vnrat*stemp
        radnbc = (1.-difrat)*(1.-vnrat)*stemp
        radndc = difrat*(1.-vnrat)*stemp
        return radvbc,radvdc,radnbc,radndc
    
    def phostress(self,pnum, physcont, ps, etc, tc, eacas,rb,ecmass,td1,td2,pawfzw, tawfrw,tcmin,co2t):
        co2t['rstfac'] = np.zeros(4)
        #...leaf humidity stress
        h2oi   = etc / ps
        h2oa   = eacas / ps
    
        tprcor = self.tice*ps*self.p0_sfc
        ecmol = self.molc_h2o * ecmass * self.dtisib 
        h2os = h2oa + ecmol / (0.5/rb * self.amagatwv*tprcor/tc)
    
        h2os  = min( h2os, h2oi )
        h2os  = max( h2os, 1.0e-7)
        h2osrh = h2os / h2oi
    
        #...soft landing: add curvature at low 
        #...relative humidities to 
        #...lessen positive feedback
        if (self.iscrop):
            h2osrh = h2osrh ** self.rhfac_exp_crop
        else:
           if (h2osrh < self.rhfac_astart):
                h2osrh = h2osrh + (self.rhfac_astart - h2osrh) ** self.rhfac_exp

        #...set stress factor
        co2t['rstfac'][1] = h2osrh
    
        #...set minimum value to relative humidity
        #...stress in larch forests due to hypothesis
        #...needles and deep roots
        #...counterbalances extreme stress
        if (self.isnforest):
             co2t['rstfac'][1] = max(h2osrh, self.rhfac_nforest)
    
        #...set minimum value to relative humidity
        #...stress in grass tundra due to hypothesis
        #...extra moisture from melting permafrost 
        #...counterbalances extreme stress
        if (pnum == self.pft_c3a): 
             co2t['rstfac'][1] = max(h2osrh, self.rhfac_tundra)
            
            
        if ((self.iseforest) or (self.iscrop)):
            lawf = pawfzw
        else:
            lawf = tawfrw
        co2t['rstfac'][2] = max(0.1, min(1.0, ((1+physcont['wssp'])*lawf) / (self.physcont['wssp']+lawf)))
        templ = 0.98 + np.exp(physcont['slti'] * (physcont['hlti'] - tc))
        temph = 0.98 + np.exp(physcont['shti'] * (tc - physcont['hhti']))
    
    #frost stress factor, we ignore this
#        if (tc < tcmin):
#            tcmin = tc
#    
#        tcmin = max(tcmin, self.tcbot) #bottom-stop tcmin at -20C
#        if (tc > self.tice):  #frost recovery at 2C/day
#            tcmin = tcmin + ((4.0*self.dtsib)/86400.0)
#        tempf = 1. + np.exp(self.physcont_sfti * (self.physcont_hfti - tcmin))  
        tempf = 1.0 #we ignore frost stress
    
        #...overall temperature scaling factor
        co2t['rstfac'][3] = min(1.0, 1./(temph*templ*tempf))  
        
    def read_aero(self):
        data_aero = Dataset('sib_aero.nc')
        self.laigrid=data_aero.variables['laigrid'][:] #dimesnion ngrid
        self.fvcovergrid = data_aero.variables['fvcovergrid'][:] #dimesnion ngrid
        self.ngrid = len(self.fvcovergrid)
        aero_zo_orig = data_aero.variables['aero_zo'][:]
        aero_zp_orig = data_aero.variables['aero_zp'][:]
        aero_rbc_orig = data_aero.variables['aero_rbc'][:]
        aero_rdc_orig = data_aero.variables['aero_rdc'][:]
        aero_rbc = np.zeros((15,self.ngrid,self.ngrid))
        aero_rdc = np.zeros((15,self.ngrid,self.ngrid))
        aero_zo = np.zeros((15,self.ngrid,self.ngrid))
        aero_zp = np.zeros((15,self.ngrid,self.ngrid))
        #the following is very important, since in Fortran dimensions are read in the pther way round as in Python!!!!!!
        #https://stackoverflow.com/questions/47085101/netcdf-startcount-exceeds-dimension-bound
        for i in  range(15):
            for j in range(self.ngrid):
                for k in range(self.ngrid):
                  aero_rbc[i,j,k] = aero_rbc_orig[k,j,i]
                  aero_rdc[i,j,k] = aero_rdc_orig[k,j,i]
                  aero_zo[i,j,k] = aero_zo_orig[k,j,i]
                  aero_zp[i,j,k] = aero_zp_orig[k,j,i]
        
        self.aerovar['zo'] = aero_zo
        self.aerovar['zp_disp'] = aero_zp
        self.aerovar['rbc'] = aero_rbc
        self.aerovar['rdc'] = aero_rdc
        
    
    def aerointerpolate(self,gref, glon, glat, pnum, pref, lai, fvcover):
        high_lai = True
        dfvcover = self.fvcovergrid[2] - self.fvcovergrid[1]
        dlai = self.laigrid[2] - self.laigrid[1]
    
        #Assign input LAI to local variables and make sure 
        #they lie within the limits of the interpolation tables.
        locfvcover = max(fvcover, 0.01)
        loclai = max(lai, 0.02)
    
        #Determine the nearest array location for the desired LAI
        i = int(loclai / dlai) + 1
        if (i+1 > self.ngrid):
           if (high_lai):
              i = self.ngrid-1
           else:
              raise Exception('LAI higher than interpolation tables: ')

        laiwt1 = (loclai - self.laigrid[i]) / dlai
        laiwt2 = (self.laigrid[i+1] - loclai) / dlai
    
        #Determine the nearest array location for the desired fVCover
        j = int(locfvcover / dfvcover) + 1
    
        if (j+1 > self.ngrid):
           if (high_lai):
              j = self.ngrid-1
           else:
              raise Exception('fVCover higher than interpolation tables: ')

        fvcovwt1 = (locfvcover - self.fvcovergrid[j]) / dfvcover
        fvcovwt2 = (self.fvcovergrid[j+1] - locfvcover) / dfvcover
        #pnum is plant functional type number, 
         # Linearly interpolate RbC, RdC, roughness length, zero plane displacement
         #indices below are adapted to match Python indexing system
        rbc = (laiwt2*self.aerovar['rbc'][pnum-1,i-1,j-1] + laiwt1*self.aerovar['rbc'][pnum-1,i,j-1])*0.5 + \
               (fvcovwt2*self.aerovar['rbc'][pnum-1,i-1,j-1] + fvcovwt1*self.aerovar['rbc'][pnum-1,i-1,j])*0.5
        rdc = (laiwt2*self.aerovar['rdc'][pnum-1,i-1,j-1] + laiwt1*self.aerovar['rdc'][pnum-1,i,j-1])*0.5 + \
               (fvcovwt2*self.aerovar['rdc'][pnum-1,i-1,j-1] + fvcovwt1*self.aerovar['rdc'][pnum-1,i-1,j])*0.5
        z0  = (laiwt2*self.aerovar['zo'][pnum-1,i-1,j-1]  + laiwt1*self.aerovar['zo'][pnum-1,i,j-1])*0.5 + \
               (fvcovwt2*self.aerovar['zo'][pnum-1,i-1,j-1] + fvcovwt1*self.aerovar['zo'][pnum-1,i-1,j])*0.5
        zp_disp = (laiwt2*self.aerovar['zp_disp'][pnum-1,i-1,j-1] + laiwt1*self.aerovar['zp_disp'][pnum-1,i,j-1])*0.5 + \
                   (fvcovwt2*self.aerovar['zp_disp'][pnum-1,i-1,j-1] + fvcovwt1*self.aerovar['zp_disp'][pnum-1,i-1,j])*0.5
        return z0, zp_disp, rbc, rdc
        
    def veg_update(self,doy,gref, glon, glat,pnum,pref,iscrop,isgrass,physcont,poollu, snow_cvfc, node_z,poollt,vegt,CLASScc):
          
        
        
        vegt['fpar'] = (1.0 - np.exp(max(min(vegt['lai'], max(physcont['laisat'], 0.001)),0.0) * \
        np.log(max(1.0-physcont['fparsat'],0.001)) / max(physcont['laisat'], 0.001))) #absorbed fraction of PAR (-)
        vegt['vcover'] = (1.0 - np.exp(max(min(vegt['lait'], max(physcont['laisat'], 0.001)),0.0) * \
        np.log(max(1.0-physcont['fparsat'],0.001)) / max(physcont['laisat'], 0.001)))
        vegt['z0d'], vegt['zp_dispd'], vegt['cc1'], vegt['cc2'] = self.aerointerpolate(gref, glon, glat, pnum, pref, vegt['lai'], vegt['vcover'])
        vegt['gmudmu'] = self.gmuder(glat, doy, physcont['ChiL'],CLASScc) #time mean projected leaf area normal to Sun, mapper.f90
        
        vegt['zpd_adj'] = vegt['zp_dispd'] + (physcont['z2'] - vegt['zp_dispd']) * snow_cvfc
        vegt['z0'] = max(0.1, vegt['z0d'] / (physcont['z2'] - vegt['zp_dispd']) * (physcont['z2'] - vegt['zpd_adj']))

        vegt['zztemp'] = physcont['z2'] - vegt['zpd_adj'] + self.ztemp
        vegt['zzwind'] = physcont['z2'] - vegt['zpd_adj'] + self.zwind
        
    def flux_vmf(self,zzwind,zztemp,z0,ros, spdm, sh, thm, sha, tha):
        zrib = zzwind **2 / zztemp              
        wgm   = sha - sh
        thgm  = tha  - thm
        thvgm = thgm + tha * self.delta * wgm
        z1z0u = zzwind/z0
        z1z0urt = np.sqrt( z1z0u )
        z1z0u = np.log(z1z0u)
        z1z0t = zzwind/z0
        z1z0trt = np.sqrt( z1z0t )
        z1z0t = np.log(z1z0t)
        #Neutral surface transfers for momentum CUN and for heat/moisture CTN:
        cun = self.vkrmn*self.vkrmn / (z1z0u*z1z0u )   #neutral Cm & Ct
        ctn = self.vkrmn*self.vkrmn / (z1z0t*z1z0t )
    
        #PATCH-when 1/cun is calculated, the square root is taken.
        cuni = z1z0u / self.vkrmn
    
                                                                               
        #   SURFACE TO AIR DIFFERENCE OF POTENTIAL TEMPERATURE.            
        #   RIB IS THE BULK RICHARDSON NUMBER, between reference
        #   height and surface.
    
        temv = tha * spdm * spdm   
        temv = max(0.000001,temv)
        rib = -thvgm * self.grav * zrib / temv 
    
        #   The stability functions for momentum and heat/moisture fluxes as
        #   derived from the surface-similarity theory by Luis (1079, 1982), and
        #   revised by Holtslag and Boville(1993), and by Beljaars and Holtslag 
        #   (1991).
        if (rib >= 0.0):                                           
    
            # THE STABLE CASE. RIB IS USED WITH AN UPPER LIMIT              
    
            rib   = min(rib, 0.5)                   
            fmomn = (1. + self.cstabl * rib * (1.+ self.bstabl * rib))
            fmomn = 1. / fmomn
            fmomn = max(0.0001,fmomn)
            fheat = fmomn
    
        else:                                  
    
            #  THE UNSTABLE CASE.    
    
            ribtemp = np.abs(rib)
            ribtemp = np.sqrt( ribtemp )
            dm      = 1. + self.cunstablM * cun * z1z0urt * ribtemp
            dh      = 1. + self.cunstablT * ctn * z1z0trt * ribtemp
            fmomn   = 1. - (self.bunstablM * rib ) / dm
            fheat   = 1. - (self.bunstablT * rib ) / dh    
    
        #   surface-air transfer coefficients for momentum CU, for heat and 
        #   moisture CT. The CUI and CTI are inversion of CU and CT respectively.
    
        cu = cun * fmomn 
        ct = ctn * fheat
    
        #   Ustar and ventlation mass flux: note that the ustar and ventlation 
        #   are calculated differently from the Deardoff's methods due to their
        #   differences in define the CU and CT.
    
        ustar  = spdm * spdm * cu 
        ustar  = np.sqrt( ustar ) 
        ventmf = ros * ct * spdm  
        return ustar, cuni, cu,ventmf
    
    def hydro_sets(self,soilt, hydrost, sscolt,CLASSrhsurf):
        hydrost['rhsoil'] = CLASSrhsurf
        
    def hydro_canopy(self,gref, glon, glat, pref,chil, tm, tcas,lai, vcover,cuprt, lsprt, tc,hydrost, sscolt):
        hydrost['wetfracc'] = 0. #assume dry canopy
        
    def flux_vrbrd(self,pnum, z2, nsl,rst, td, cast, vegt, gprogt, gdiagt,fluxt, hydrost):
        fluxt['ustar'],cuni,fluxt['cu'],fluxt['ventmf'] = self.flux_vmf(vegt['zzwind'], vegt['zztemp'], vegt['z0'],gdiagt['ros'], gprogt['spdm'], gprogt['sh'], self.gdiagt['thm'],cast['shcas'], cast['thcas'])
    #...aerodynamic resistance
        fluxt['ra'] = gdiagt['ros'] / fluxt['ventmf'] 
        fluxt['drag'] = gdiagt['ros'] * fluxt['cu'] * fluxt['ustar']
        temv = (self.physcont['z2'] - vegt['zp_dispd']) / vegt['z0']
        temv = max(temv,1.00)
        temv = np.log(temv) 
        u2     = gprogt['spdm'] / (cuni * self.vkrmn)
        u2 = max(1.0, u2 * temv)
        fluxt['rbc'],fluxt['rdc'],fluxt['rb'],fluxt['rd'] = self.flux_rbrd(z2, u2, cast['tc'], cast['tcas'], td, 0, vegt['lai'], vegt['cc1'], vegt['cc2'])    
        fluxt['rc']  = rst + fluxt['rb'] + fluxt['rb']
        if (self.isnforest):
            epct = min(0.8, max(0.2, hydrost['satfrac']*2. - 0.4))
        else:
            epct = 1.0
        self.gect = epct * (1. - hydrost['wetfracc']) / fluxt['rc']
        geci = 0. #assume no wet fraction of vegetation
        self.coc = self.gect + geci
        hydrost['ecmass'] = (self.etc - cast['eacas']) * self.coc * gdiagt['ros']  * 0.622e0 /gprogt['ps'] * self.dtsib
        
        
    def flux_rbrd(self,z2, u2, tc, ta, td, snow_cvfc, lai, ccc1, ccc2):
        #this is only called in flux_vrbrd
        #snow_cvfc = 0. #ignore snow
        rbc = ccc1 / (1. - snow_cvfc)
        rdc = ccc2 * (1. - snow_cvfc)
        temdif  = max( 0.01,tc - ta)
        fac     = lai / 890.* (temdif * 20.0)**0.25
        rb = 1.0 / ( (np.sqrt(u2) / rbc) + fac )
        temdif = max( 0.1, td - ta )
        fih = np.sqrt( 1.+9.* self.grav * temdif * z2 / (td*u2*u2) )
        rd  = rdc / (u2 * fih)
        return rbc,rdc,rb,rd
    
    def driver_interp(self,indx, lon, lat, gdiagt, gprogt,CLASSSwin,CLASSsinlea,CLASSq,CLASStheta):
        sea = np.arcsin(CLASSsinlea) #solar elevation angle
        sza = np.pi / 2. - sea #solar zenith angle
        if (np.cos(sza) > 0.0):
            coslon = np.cos( np.pi/180 * lon)
            dbarod = 1.0 + self.eccn * (coslon - self.perih)
            gdiagt['toa_solar'] = self.solar_const * dbarod * dbarod * np.cos(sza)
        else:
            gdiagt['toa_solar'] = 0.0
        gdiagt['toa_radvbc'],gdiagt['toa_radvdc'],nothing,nothing2 = self.raddrv(gdiagt['toa_solar'],np.cos(sza))
        gprogt['sw_dwn'] = CLASSSwin
        gdiagt['radvbc'],gdiagt['radvdc'],gdiagt['radnbc'],gdiagt['radndc'] = self.raddrv(gprogt['sw_dwn'],np.cos(sza))
        self.gprogt['tm'] = CLASStheta * (self.pressure/100000)**(self.gas_const_r/self.spec_heat_cp)
        self.gprogt['sh'] = CLASSq#mixed layer water vapor mixing ratio (kg/kg), although I guess they mean spec hum
        gprogt['ps'] = self.pressure/100. #hPa!!!!!!!
        gdiagt['bps'][1] = (0.001*gprogt['ps'])**self.kappa
        gdiagt['em'] = gprogt['sh'] * gprogt['ps'] / (0.622 + gprogt['sh'])
        # calculate reference level air density
        gdiagt['ros'] = self.rgfac * gprogt['ps'] / gprogt['tm']
        # calculate psycrometric constant
        gdiagt['psy'] = self.spec_heat_cp / self.lvap * gprogt['ps'] / 0.622
        
    def dtess_eau(self,len, pl, tl):
        tl0 = cp.deepcopy(tl)
        esw = np.zeros(len+1)
        dtesw = np.zeros(len+1)
        ess = np.zeros(len+1)
        dtess = np.zeros(len+1)
        for i in range(1,len+1): 

           tl0[i]    = max(self.twmin,tl0[i])
           tl0[i]    = min(self.twmax,tl0[i])
           tstl      = self.tsref / tl0[i]
           e1        = 11.344*(1.0 - tl0[i]/self.tsref)
           e2        = -3.49149*(tstl - 1.0)
           f1        = -7.90298*(tstl - 1.0)
           f2        = 5.02808*np.log10(tstl)
           f3        = -1.3816*(10.0**e1-1.0)/10000000.0
           f4        = 8.1328*(10.0**e2-1.0)/1000.0
           f5        = np.log10(self.psref)
           f         = f1 + f2 + f3 + f4 + f5
        
           esw[i]    = (10.0**f)*1.e+02
           esw[i]    = min(esw[i],pl[i]*0.9)
           dtesw[i]  = self.lvap*esw[i]/(self.rv*tl0[i]*tl0[i])
        
           ess[i]    = esw[i]
           dtess[i]  = dtesw[i]
        return ess, dtess
    
    def local_set(self,ps, tc, capacc_liq, capacc_snow, capacg, snow_gmass, nsl, www_liq, www_ice, td):
        ppl =np.zeros(1+1)
        ttl =np.zeros(1+1)
        ppl[1] = ps*100.0
        ttl[1] = tc
        esst,dtesst = self.dtess_eau(1,ppl,ttl)
        self.etc = esst[1]/100.0
        self.getc = dtesst[1]/100.0
        
    
    def radfac(self,chil, ref, tran, z1, z2, cosz, soilt, sscolt, vegt,hydrost, radt):
        soref = np.zeros(3)
        soref[1] = soilt['soref_vis']
        soref[2] = soilt['soref_nir']
    
        #...calculate minimum cosz
        fff = max(0.01746,cosz)
    
        #...calculate snow vertical veg coverage (0-1)
        #.......the 5 right now is used to represent the average 
        #.......ratio of snow depth to water content 
        hydrost['snow_cvfc'] = 0
    
        #...calculate snow vertical ground coverage (0-1)
        hydrost['snow_gvfc'] = 0.
    
        #...calculate saturation capacity depth (kg/m2)
        hydrost['satcapc'] = vegt['lait'] * 0.0001 * (1.0 - hydrost['snow_cvfc']) * self.denh2o
        hydrost['satcapg'] = self.gwlim
    
        #...calculate melting fraction
        #we ignore snow
        fmelt = 1.0
    
        #-----------------------------------------------------------------------
        radt['radfacg'] = np.zeros((2+1,2+1))
        radt['radfacc'] = np.zeros((2+1,2+1))
        albedog = np.zeros((2+1,2+1)) #mind the index difference fortran and python, we will have some unused zeros
        albedoc = np.zeros((2+1,2+1))
        tranc1 = np.zeros(2+1)
        tranc2 = np.zeros(2+1)
        tranc3 = np.zeros(2+1)
        #...loop over 1=shortwave, 2=nir radiation
        for iwave in range( 1, 2+1):
            scov =  min(0.5, hydrost['wetfracc'])
            reff1 = ( 1. - scov ) * ref[iwave,1] + scov * ( 1.2 - iwave * 0.4 ) * fmelt
            reff2 = ( 1. - scov ) * ref[iwave,2] + scov * ( 1.2 - iwave * 0.4 ) * fmelt
            tran1 = tran[iwave,1] * ( 1. - scov ) + scov * ( 1.- ( 1.2 - iwave * 0.4 ) * fmelt )* tran[iwave,1]
            tran2 = tran[iwave,2] * ( 1. - scov ) + scov * ( 1.- ( 1.2 - iwave * 0.4 ) * fmelt ) * 0.9 * tran[iwave,2]
    
            #...calculate average scattering coefficient, leaf projection, 
            #.....and other coefficients
            scat = vegt['green']*(tran1 + reff1) + (1.- vegt['green']) * (tran2 + reff2)
            chiv = chil
    
            if ( abs(chiv) <= 0.01 ):
                chiv = 0.01
            aa = 0.5 - 0.633 * chiv - 0.33 * chiv * chiv
            bb = 0.877 * ( 1. - 2. * aa )
    
            proj = aa + bb * fff
            extkb = ( aa + bb * fff ) / fff
            zmew = 1. / bb * ( 1. - aa / bb * np.log ( ( aa + bb ) / aa ) )
            acss = scat / 2. * proj / ( proj + fff * bb )
            acss = acss * ( 1. - fff * aa / ( proj + fff * bb ) * np.log ( ( proj  +   fff * bb + fff * aa ) / ( fff * aa ) ) )
    
            upscat = vegt['green'] * tran1 + ( 1.- vegt['green'] ) * tran2
            upscat = 0.5 * ( scat + ( scat - 2. * upscat ) * (( 1. - chiv ) / 2. ) ** 2 )
            betao = ( 1. + zmew * extkb ) / ( scat * zmew * extkb ) * acss
            print('scat')
            print(scat)
    
            #...calculate intermediate variables
            be = 1. - scat + upscat
            ce = upscat
            bot = ( zmew * extkb ) ** 2 + ( ce**2 - be**2 )
    
            if ( abs(bot) <= 1.e-10):
                scat = scat* 0.98
                be = 1. - scat + upscat
                bot = ( zmew * extkb ) ** 2 + ( ce**2 - be**2 )
    
            de = scat * zmew * extkb * betao
            fe = scat * zmew * extkb * ( 1. - betao )
            hh1 = -de * be + zmew * de * extkb - ce * fe
            hh4 = -be * fe - zmew * fe * extkb - ce * de
    
            psi = np.sqrt(be**2 - ce**2)/zmew
    
            zat = vegt['lait']/vegt['vcover']*(1. - hydrost['snow_cvfc'])
    
            power1 = min( psi*zat, 50.0 )
            power2 = min( extkb*zat, 50.0 )
            epsi = np.exp( - power1 )
            ek = np.exp ( - power2 )
    
            #...calculate ground albedos
            #......currently the diffuse and direct are the same
            #......making ge=1
            
            albedog[iwave,:] = soref[iwave]*(1.-hydrost['snow_gvfc']) + ( 1.2-iwave*0.4 )* fmelt * hydrost['snow_gvfc']
            ge = 1.0
    
            #...calculate canopy diffuse albedo
            f1 = be - ce / albedog[iwave,2]
            zp = zmew * psi
    
            den = ( be + zp ) * ( f1 - zp ) / epsi - ( be - zp ) * ( f1 + zp ) * epsi
            hh7 = ce * ( f1 - zp ) / epsi / den
            hh8 = -ce * ( f1 + zp ) * epsi / den
            f1 = be - ce * albedog[iwave,2]
            den = ( f1 + zp ) / epsi - ( f1 - zp ) * epsi
    
            hh9 = ( f1 + zp ) / epsi / den
            hh10 = - ( f1 - zp ) * epsi / den
            tranc2[iwave] = hh9 * epsi + hh10 / epsi
            albedoc[iwave,2] =  hh7 + hh8
    
            #...calculate canopy transmittance and direct albedo
            f1 = be - ce / albedog[iwave,2]
            zmk = zmew * extkb
    
            den = ( be + zp ) * ( f1 - zp ) / epsi - ( be - zp ) * ( f1 + zp ) * epsi
            hh2 = ( de - hh1 / bot * ( be + zmk ) ) * ( f1 - zp ) / epsi - \
                ( be - zp ) * ( de - ce*ge - hh1 / bot  * ( f1 + zmk ) ) * ek
            hh2 = hh2 / den
            hh3 = ( be + zp ) * (de - ce * ge - hh1 / bot * ( f1 + zmk )) * ek -         \
                ( de - hh1 / bot * ( be + zmk ) ) *  ( f1 + zp ) * epsi
            hh3 = hh3 / den
            f1 = be - ce * albedog[iwave,2]
            den = ( f1 + zp ) / epsi - ( f1 - zp ) * epsi
            hh5 = - hh4 / bot * ( f1 + zp ) / epsi - ( fe + ce*ge*albedog[iwave,2] + \
                hh4 / bot * ( zmk - f1 ) ) * ek
            hh5 = hh5 / den
            hh6 =   hh4 / bot * ( f1 - zp ) * epsi + ( fe + ce * ge * albedog[iwave,2] +     \
                hh4 / bot*( zmk - f1 ) ) * ek
            hh6 = hh6 / den
            tranc1[iwave] = ek
            tranc3[iwave] = hh4 / bot * ek + hh5 * epsi + hh6 / epsi
            albedoc[iwave,1] = hh1 / bot + hh2 + hh3
    
            #...calculate radfac, which multiplies incoming short wave fluxes
            #......to give absorption of radiation by canopy and ground
            radt['radfacg'][iwave,1] = (1. - vegt['vcover']) * (1. - albedog[iwave,1]) +  \
                vegt['vcover'] * (tranc1[iwave]*( 1.-albedog[iwave,1]) + tranc3[iwave]*(1.-albedog[iwave,2]))
    
            radt['radfacg'][iwave,2] = (1. - vegt['vcover']) * (1. - albedog[iwave,2]) + \
                vegt['vcover'] * (tranc2[iwave] * (1.-albedog[iwave,2]))
    
            radt['radfacc'][iwave,1] = vegt['vcover'] * ( ( 1.-albedoc[iwave,1] )  \
                     - tranc1[iwave] * (1.-albedog[iwave,1]) - tranc3[iwave] * (1.-albedog[iwave,2]) )
    
            radt['radfacc'][iwave,2] = vegt['vcover'] * ( ( 1.-albedoc[iwave,2] ) - tranc2[iwave] * ( 1.-albedog[iwave,2] ) )
        #some code here left out    ,unused
        radt['tsfc'] = min(self.tice,sscolt['td'][sscolt['nsl']+1]) * hydrost['snow_gvfc'] + sscolt['td'][1]*(1.-hydrost['snow_gvfc'])
        radt['effgc'] = min(1.0, 0.4*vegt['lait'])
        
    def radabs(self,dlwbot, gdiagt,radt):
        radc3c = 0.
        radc3g = 0.
        radn = np.zeros((2+1,2+1))
        radn[1,1] = gdiagt['radvbc']
        radn[1,2] = gdiagt['radvdc']
        radn[2,1] = gdiagt['radnbc']
        radn[2,2] = gdiagt['radndc']

        for iwave in range(1,2+1):
           for irad in range(1,2+1):
              radc3c = radc3c + radt['radfacc'][iwave,irad] * radn[iwave,irad]
              radc3g = radc3g + radt['radfacg'][iwave,irad] * radn[iwave,irad]
    
    
        #...absorb downwelling radiation 
        radt['radc3c'] = radc3c + dlwbot * radt['effgc']
        radt['radc3g'] = radc3g + dlwbot * (1.-radt['effgc'])
    
    def radnet(self,nsl, tsfc, tc, radt):
        fac1 = radt['effgc']

        #...calculate temperatures
        tg = tsfc
    
        tc4 = tc**4
        tg4 = tg**4

    
        #...derivatives
        dtc4 = 4*self.stefan * tc**3
        dtg4 = 4*self.stefan * tg**3
    
        #...canopy leaves thermal radiation contributions
        closs =  2 * fac1 * self.stefan * tc4
        #assume no snow
        closs = closs - fac1 * self.stefan * tg4
        
    
        #...ground thermal radiation contributions
        gloss =  self.stefan * tg4 - fac1 * self.stefan * tc4
    
        #...net radiation terms
        radt['radtc'] = radt['radc3c'] - closs
        radt['radtg'] = radt['radc3g'] - gloss
    
    def delef(self,psy, ros, em, ea, soilrh,rd, ra,CLASSG):
        cpdpsy = self.spec_heat_cp / psy
        ec  = (self.etc - ea) * self.coc * ros  * cpdpsy * self.dtsib
        eg  = CLASSG * self.dtsib #soil heat flux, J/m2
        #disregard snow
        #es  = (ets - ea) / rd * ros * cpdpsy/self.snofac * self.dtsib

        fws = (ea  - em) / ra * ros * cpdpsy * self.dtsib
        return ec,eg,fws
    
    def delhf(self,tm, bps, ros, ta, nsl, tsfc,tc, ra, rb, rd):
        rai = 1.0 / ra
        rbi = 1.0 / rb  
        rdi = 1.0 / rd              
    
        #    these are the current time step fluxes in J/m2
        cp = self.spec_heat_cp #see near top of delhf.f90
        hc   = cp * ros * (tc - ta) * rbi * self.dtsib
        print('ros')
        print(ros)
        print('hc')
        print(hc)
        if (nsl == 0): #no snow case
            hg   = cp * ros * (tsfc - ta) * rdi * self.dtsib 
            hs   = 0.0
        else:
            hg = 0.0
            hs   = cp * ros * (tsfc - ta) * rdi * self.dtsib 
    
        fss  = cp * ros * (ta - tm) * rai * self.dtsib
    
        #    now we do the partial derivatives
        #    these are done assuming the fluxes in W/m2        
        #    for canopy leaves sensible heat flux: W/(m2 * K)
        # 
        hcdtc =   cp * ros * rbi
        hcdta = - hcdtc
        #
        #    for ground and snow sensible heat fluxes: W/(m2 * K)
        #
        hgdtg =   cp * ros * rdi  
        hsdts =   hgdtg
        hgdta = - hgdtg
        hsdta = - hgdtg
        #
        #    for the canopy air space (CAS) sensible heat flux: W/(m2 * K)
        #
        hadta =   cp * ros * rai
        hadth = - hadta/bps[1]
        
        return fss, hc,hg,hs
        
    def sibslv(self,cast, fluxt, radt, sscolt):
        bvec = np.zeros(6)
        bvec[3]    =  fluxt['hc'] * self.dtisib - fluxt['fss'] * self.dtisib + fluxt['hg'] * self.dtisib
        bvec[4]    =  fluxt['ec'] * self.dtisib - fluxt['fws'] * self.dtisib + fluxt['eg'] * self.dtisib
        #no snow in our implementation
        bvec[5]    =  radt['radtc'] - fluxt['hc'] * self.dtisib - fluxt['ec'] * self.dtisib
        dta = bvec[3]
        dea = bvec[4]
        dtc = bvec[5]
        print('dtc')
        print(dtc)
        print('radtc')
        print(radt['radtc'])
        print('hc')
        print(fluxt['hc'])
        print('ec')
        print(fluxt['ec'])
        return dta,dtc,dea
    
    def phosort(self,numic, ic, gamma, range_, eyy, pco2y):
        if( ic < 4 ):
            pco2y[1] = gamma + 0.5*range_
            pco2y[2] = gamma + range_*( 0.5 - 0.3*1.0*np.sign(eyy[1]) ) #fortran sign: SIGN(A,B) returns the value of A with the sign of B. 
            pco2y[3] = pco2y[1]- (pco2y[1]-pco2y[2])/(eyy[1]-eyy[2]+1.e-10)*eyy[1]
            pmin = min( pco2y[1], pco2y[2] )
            emin = min(   eyy[1],   eyy[2] )
            if ( emin > 0. and pco2y[3] > pmin ):
                pco2y[3] = gamma
        else:
    
            n = ic - 1
            bitx = abs(eyy[n]) > 0.1
            if(not bitx):
                pco2y[ic] = pco2y[n]
            if(bitx):
                for j in range(2, n + 1):
                    a = eyy[j]
                    b = pco2y[j]
#                    do i = j-1,1,-1
#                    if(eyy(i) <= a ) go to 100
#                    eyy(i+1) = eyy(i)
#                    pco2y(i+1) = pco2y(i)
#                    enddo ! i loop
#                    i = 0
#                    100        continue
                    for i in range(j-1,1-1,-1): #this is tricky: we write 1-1, since in contrast to Fortran, Python does not include the end number of the range. It is -1 instead of +1 since the loop counts downward
                        if not (eyy[i] <= a ):
                            eyy[i+1] = eyy[i]
                            pco2y[i+1] = pco2y[i]
                    if not (eyy[i] <= a ):
                        i = 0 
                    eyy[i+1] = a
                    pco2y[i+1] = b
    
    #-----------------------------------------------------------------------
    
            if(bitx):
                pco2b = 0.
                IS    = 1
    
            for ix in range (1, n+1):
                if(bitx):
                    if( eyy[ix] < 0. ) :
                        pco2b = pco2y[ix]
                        IS = ix
    
            if(bitx):
                i1 = IS-1
                i1 = max(1, i1)
                i1 = min(n-2, i1)
                i2 = i1 + 1
                i3 = i1 + 2
                isp   = IS + 1
                isp = min( isp, n )
                IS = isp - 1
                eyyisp = eyy[isp]
                eyyis = eyy[IS]
                eyyi1 = eyy[i1]
                eyyi2 = eyy[i2]
                eyyi3 = eyy[i3]
                pco2yisp = pco2y[isp]
                pco2yis = pco2y[IS]
                pco2yi1 = pco2y[i1]
                pco2yi2 = pco2y[i2]
                pco2yi3 = pco2y[i3]
    
            if(bitx):
    
                #...Patch to check for zero in the denominator...
                if(eyyis != eyyisp):
                    pco2yl=pco2yis - (pco2yis-pco2yisp) / (eyyis-eyyisp)*eyyis
                else:
                    pco2yl = pco2yis * 1.01
    
                #   METHOD USING A QUADRATIC FIT
    
                ac1 = eyyi1*eyyi1 - eyyi2*eyyi2
                ac2 = eyyi2*eyyi2 - eyyi3*eyyi3
                bc1 = eyyi1 - eyyi2
                bc2 = eyyi2 - eyyi3
                cc1 = pco2yi1 - pco2yi2
                cc2 = pco2yi2 - pco2yi3
    
                #...Patch to prevent zero in denominator...
                if(bc1*ac2-ac1*bc2 != 0.0 and ac1 != 0.0):
                    bterm = (cc1*ac2-cc2*ac1)/(bc1*ac2-ac1*bc2)
                    aterm = (cc1-bc1*bterm)/ac1
                    cterm = pco2yi2-aterm*eyyi2*eyyi2-bterm*eyyi2
                    pco2yq= cterm
                    pco2yq= max( pco2yq, pco2b )
                    pco2y[ic] = ( pco2yl+pco2yq)/2.0
                else:
                    pco2y[ic] = pco2y[ic] * 1.01

    #
    # make sure pco2 does not fall below compensation point
        pco2y[ic] = max(pco2y[ic],gamma+0.01)
        return eyy, pco2y
    
    def cas_update(self,bps1, psy, ros,casd, capacc_liq, capacc_snow, lai,cast):
        #...canopy heat capacity
        cast['hcapc'] = lai*self.leafhc + (capacc_snow/self.denice + capacc_liq/self.denh2o) * self.h2ohc
    
        #...canopy potential temperature and water vapor
        cast['thcas'] = cast['tcas'] / bps1
    
        #...canopy storage inertia terms
        cast['hcapcas'] = ros * self.spec_heat_cp * casd
        cast['vcapcas'] = ros * self.cv * casd / psy

        
    def addinc(self,gref, pftref,lonsib, latsib,gdiagt, gprogt,cast, fluxt, sscolt):
        cast['eacas'] = cast['eacas'] + self.dea
        cast['tcas'] = cast['tcas'] + self.dta
        cast['tc']   = cast['tc'] + self.dtc
        if ( gdiagt['em'] <= 0.):
            gdiagt['em']=1.e-3
        if ( cast['eacas'] <= 0.):
            cast['eacas'] = gdiagt['em']
        fluxt['fss'] = gdiagt['ros'] * self.spec_heat_cp * (cast['tcas'] - gprogt['tm']) / fluxt['ra']
        fluxt['fws'] = (cast['eacas'] - gdiagt['em']) / fluxt['ra'] * self.spec_heat_cp * gdiagt['ros'] / gdiagt['psy']
    
        # Recalculate mixing ratios
        gprogt['sh']  = 0.622 / (gprogt['ps'] / gdiagt['em'] - 1.)
        cast['shcas'] = 0.622 / (gprogt['ps'] / cast['eacas'] - 1.)
        
    def flux_update(self, psy, ros, radtc, radtg, radts, poros, paw_lay, cast, fluxt, hydrost, sscolt):
        lvapi = 1.0/self.lvap
        cpdpsy = self.spec_heat_cp/psy
        ecpot = (self.etc + self.getc*self.dtc) - cast['eacas']
        fluxt['eci'] = 0. #neglect interception
        rsnow = 0#neglect snow
        facks = 1. + rsnow * (self.snofac-1.)
        fluxt['ect'] = ecpot * self.gect * ros * cpdpsy * self.dtsib
        hydrost['ecmass'] = (fluxt['eci'] + fluxt['ect']) * facks * lvapi
        ecidif = 0 #assume interception to be zero
        fluxt['hc'] = (cast['tc'] - cast['tcas']) / fluxt['rb'] * ros * self.spec_heat_cp * self.dtsib + ecidif
        print('hcfluxupd')
        print(fluxt['hc'])
        egidif = 0 #assume interception by surface zero
        fluxt['hg'] = (sscolt['td'][1] - cast['tcas']) / fluxt['rd'] * ros * self.spec_heat_cp * self.dtsib + egidif
    
    def phocycalc(self,toa_par, par, aparkk,vmax, vmaxss, atheta, btheta, gamma,rrkk, omss, c3, c4, pco2i):
        assimpot = vmax *(pco2i-gamma)/(pco2i + rrkk)*c3 + vmax * c4
        omc = vmaxss *(pco2i-gamma)/(pco2i + rrkk)*c3 + vmaxss * c4
    
        #...light assimilation
        lightpot = toa_par*(pco2i-gamma)/(pco2i+2.*gamma)*c3 + toa_par*c4
        ome = par*(pco2i-gamma)/(pco2i+2.*gamma)*c3 + par * c4
    
        #...transfer assimilation
        oms  = omss * c3 + omss*pco2i * c4
    
        #...combined assimilation
        sqrtin= max( 0.0, ( (ome+omc)**2 - 4.*atheta*ome*omc ) )
        omp  = ( ( ome+omc ) - np.sqrt( sqrtin ) ) / ( 2.*atheta )
        sqrtin= max( 0.0, ( (omp+oms)**2 - 4.*btheta*omp*oms ) ) 
        assim = (( ( oms+omp ) - np.sqrt( sqrtin ) ) / ( 2.*btheta ))*aparkk
        return omc, ome, oms,assim
    
    def sibtype_setup(self,CLASSdoy,CLASSlon,CLASSlat,CLASScc,CLASSrhsurf):
        self.setup_soilt(self.clayfrac,self.sandfrac,self.soref_vis,self.soref_nir,9999, 9999, self.soilt) #call to sibtype_setup, which calls setup_soilt in Fortran
        self.hydro_sets(self.soilt, self.hydrost, self.sscolt,CLASSrhsurf)
        self.veg_update(CLASSdoy,999, CLASSlon,CLASSlat, self.pnum, 9999, self.iscrop, self.isgrass,self.physcont, 9999, self.hydrost['snow_cvfc'], 9999,9999, self.vegt,CLASScc)
    
    def sib_main(self,gref, lonsib, latsib,pref, pnum, physcont,gprogt, gdiagt, sibgl,resp_leaf, resp_auto, resp_het,CLASSrhsurf,CLASScc,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin):
        self.local_set(gprogt['ps'], self.cast['tc'],self.hydrost['capacc_liq'], self.hydrost['capacc_snow'], self.hydrost['capacg'], self.hydrost['snow_gmass'], self.sscolt['nsl'], self.sscolt['www_liq'], self.sscolt['www_ice'], self.sscolt['td'])
        self.hydro_sets(self.soilt, self.hydrost, self.sscolt,CLASSrhsurf)
        self.radfac(physcont['ChiL'], physcont['ref'],physcont['tran'], physcont['z1'], physcont['z2'], gdiagt['cosz'], self.soilt, self.sscolt, self.vegt,self.hydrost,self.radt)
        self.radabs(gprogt['dlwbot'], gdiagt, self.radt)
        self.radnet(self.sscolt['nsl'], self.radt['tsfc'],self.cast['tc'],self.radt)
        self.cas_update(gdiagt['bps'][1], gdiagt['psy'],gdiagt['ros'], self.co2t['casd'],self.hydrost['capacc_liq'],self.hydrost['capacc_snow'],self.vegt['lai'], self.cast)
        nsl = self.sscolt['nsl']#as in sib_main.f90
        self.flux_vrbrd(self.pnum, physcont['z2'], nsl,self.co2t['rst'], self.sscolt['td'][nsl+1], self.cast, self.vegt, gprogt, gdiagt,self.fluxt, self.hydrost)
        #self.etc calculated in local_set
        self.phostress(self.pnum, physcont, gprogt['ps'], self.etc, self.cast['tc'], self.cast['eacas'],self.fluxt['rb'],self.hydrost['ecmass'],self.sscolt['td'][1], self.sscolt['td'][2],self.pawfzw, self.tawfrw,9999,self.co2t)#phostress.f90, calculate phototosynthesis stress reductions
        if self.vegt['lai'] > self.canlai_min:
            self.phosib(physcont, gprogt, gdiagt, self.cast['tc'], self.cast['tcas'], self.fluxt['ra'],self.fluxt['rb'],resp_leaf, resp_auto, resp_het, self.vegt, self.co2t)
        else:
            raise Exception('LAI too small!!!')
        self.fluxt['ec'],self.fluxt['eg'],self.fluxt['fws'] = self.delef(gdiagt['psy'], gdiagt['ros'], gdiagt['em'],self.cast['eacas'], self.hydrost['rhsoil'],self.fluxt['rd'],self.fluxt['ra'],CLASSG)
        self.fluxt['fss'], self.fluxt['hc'], self.fluxt['hg'], self.fluxt['hs'] = self.delhf(gprogt['tm'], gdiagt['bps'], gdiagt['ros'], self.cast['tcas'], self.sscolt['nsl'], self.radt['tsfc'],self.cast['tc'],self.fluxt['ra'], self.fluxt['rb'],self.fluxt['rd'])
        self.dta,self.dtc,self.dea = self.sibslv(self.cast, self.fluxt, self.radt, self.sscolt)    
        self.addinc(9999, 9999,lonsib,latsib, gdiagt, gprogt, self.cast, self.fluxt, self.sscolt)  
        self.flux_update(gdiagt['psy'], gdiagt['ros'], self.radt['radtc'],self.radt['radtg'], 9999, self.soilt['poros'],9999, self.cast, self.fluxt,self.hydrost, self.sscolt)
        if self.vegt['lai'] > self.canlai_min:
            self.hydro_canopy(9999, lonsib, latsib,9999,physcont['ChiL'], gprogt['tm'],self.cast['tcas'],self.vegt['lait'],self.vegt['vcover'], gprogt['cuprt'], gprogt['lsprt'],self.cast['tc'], self.hydrost, self.sscolt)
        else:
            raise Exception('LAI too small!!!')
            
    def sib_control(self,gnum, gref,lonsib, latsib, daynew, daylmax, doy, sibg,resp_leaf, resp_auto, resp_het,CLASSrhsurf,CLASScc,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin):
        self.veg_update(doy,9999, lonsib, latsib,self.pnum,9999, self.iscrop, self.isgrass,self.physcont, 9999, self.hydrost['snow_cvfc'], 9999,9999, self.vegt,CLASScc)
        self.sib_main(9999, lonsib, latsib,9999, self.pnum, self.physcont,self.gprogt, self.gdiagt, 9999,resp_leaf, resp_auto, resp_het,CLASSrhsurf,CLASScc,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin)
    
    def run_pho_sib4(self,resp_leaf, resp_auto, resp_het,CLASSrhsurf,CLASSlon,CLASSlat,CLASScc,CLASSdoy,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin,CLASSq):
        #this function is the simplified and adapted main program of sib4, SiBDRV.f90
        self.read_aero()
        self.calc_inputvaria(CLASSrhsurf,CLASSlon,CLASSlat,CLASScc,CLASSdoy,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin)
        self.sibtype_setup(CLASSdoy,CLASSlon,CLASSlat,CLASScc,CLASSrhsurf)
        self.driver_interp(9999, CLASSlon, CLASSlat, self.gdiagt, self.gprogt,CLASSSwin,CLASSsinlea,CLASSq,CLASStheta)
        self.sib_control(9999, 9999,CLASSlon, CLASSlat,9999, 9999, CLASSdoy, 9999,resp_leaf, resp_auto, resp_het,CLASSrhsurf,CLASScc,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSWIND,CLASSLwin)
        
    def calc_inputvaria(self,CLASSrhsurf,CLASSlon,CLASSlat,CLASScc,CLASSdoy,CLASSTsoil,CLASSCO2,CLASStheta,CLASSG,CLASSSwin,CLASSsinlea,CLASSwind,CLASSLwin):
        self.hydrost['snow_cvfc'] = 0 #ignore snow
        self.gprogt['spdm'] = CLASSwind #wind speed (mixed layer?)
        self.gdiagt['thm'] = CLASStheta
        self.vegt['vmax'] = self.vmax0#sibtype_setup.f90 and phen_update.f90 (call in sibcontrol.f90)
        self.gprogt['pco2m'] = CLASSCO2 * 1e-6 * self.pressure #Mixed Layer (background) CO2 partial pressure (Pa), see module_sib.f90
        self.gprogt['dlwbot'] = CLASSLwin
        self.hydrost['capacc_liq'],self.hydrost['capacc_snow'] = 0,0 # prognostic canopy surface liquid (kg/m2),prognostic canopy surface snow (kg/m2), assumed zero
        self.hydrost['capacg'] = 0. #prognostic ground surface liquid (kg/m2)
        self.hydrost['snow_gmass'] = 0. # mass of snow on ground (kg/m2)
        self.sscolt['www_liq'] = 0. #soil/snow liquid water (kg/m2)
        self.sscolt['www_ice'] = 0. #soil/snow ice (kg/m2)
        sea = np.arcsin(CLASSsinlea) #solar elevation angle
        sza = np.pi / 2. - sea #solar zenith angle
        self.gdiagt['cosz'] = np.cos(sza)
        self.sscolt['nsl'] = int(0) #assume no snow
        self.sscolt['td'][1] = CLASSTsoil
        self.sscolt['td'][2] = CLASSTsoil
        self.gprogt['cuprt'], self.gprogt['lsprt'] = 0,0 #assume no preciîtation
                
    def phosib(self,physcont, gprogt, gdiagt, tc, tcas, ra, rb,resp_leaf, resp_auto, resp_het, vegt, co2t):
        #snow removed  in the model
        #input in SI units unless specified otherwise. soilflux in mol/m2/s, C_surf_layer_COS_ppb in ppb, C_surf_layer_CO2_ppm in ppm
        if hasattr(self,'mothermodel'):
            self.mothermodel.vars_rcm = {} #rcm stands for run canopy model
        
        #phosib.f90 file
        c3 = 1.0 - self.physcont['c4flag']
        c4 = self.physcont['c4flag']
        atheta = self.physcont['atheta']
        btheta = self.physcont['btheta']
        tprcor = self.tice*self.pressure/self.p0_sfc
        pco2cas = self.co2t['pco2cas'] #co2t given as argument to phosib.f90, phosib.f90 called in sib_main.f90 with argument sibgl_co2t, where sibgl is an argument of sib_main, sib_main.f90 called 
        #in sibcontrol.f90 with argument sibg_l(l) where sibg is an argument of sibcontrol.f90, sibcontrol.f90 in turn called in SiBDRV.f90 with argument  sib_g(i), where sib is loaded from module_sib.f90, module_sib.f90 declares the variable pco2cas 
        #updated later in phosib.f90
        pco2c = self.co2t['pco2c'] #as above
        pco2i = self.co2t['pco2i']
        pco2s = self.co2t['pco2s']
        co2cas = pco2cas / self.pressure #pco2cas is canopy air space co2 pressure
        co2m = gprogt['pco2m']  / self.pressure
        co2cap = co2t['casd'] * self.pressure/self.rstar/tcas
        resp_cas = resp_auto + resp_het
        pdamp = np.exp (-1.0 * self.zln2*(self.dtsib*self.dmin)/(self.dttin*self.ghalf))
        qdamp = 1.0 - pdamp
        scatc = physcont['tran'][1,1] +  physcont['ref'][1,1]
        vegt['park'] = np.sqrt(1.-scatc) * vegt['gmudmu'] #heeerrrreee
        co2t['aparkk'] = vegt['fpar'] / vegt['park']
        co2t['par'] = self.par_conv * (gdiagt['radvbc']+gdiagt['radvdc'])
        if (gdiagt['radvbc'] < self.minrad):
            pfd=0.0
            toa_pfd = 0.0
        else:
            pfd = self.par_conv * vegt['gmudmu'] * (gdiagt['radvbc'] + gdiagt['radvdc'])
            toa_pfd = self.toatosfc * self.par_conv * vegt['gmudmu'] * (gdiagt['toa_radvbc'] + gdiagt['toa_radvdc'])
        co2t['apar'] = pfd * physcont['effcon'] * (1. - scatc)
        toa_apar = toa_pfd * physcont['effcon'] * (1. - scatc)
        co2t['nspar'] = pfd * (1. - scatc)
        qt = 0.1 * (tc - self.vmax_tref)
        vmaxts = vegt['vmax'] * self.vmax_q10**qt
        co2t['vmaxss'] = vmaxts * co2t['rstfac'][2] * co2t['rstfac'][3] 
        zkc     = 30. * 2.1**qt
        zko     = 30000. * 1.2**qt
        spfy    = 2600. * 0.75**qt
        co2t['gamma']   = 0.5 * self.po2m/spfy * c3
        rrkk = zkc*( 1. + self.po2m/zko ) * c3 + vegt['vmax']/5.* ( self.oms_q10**qt) * c4 
        print('tc')
        print(tc)
        omss = ( vegt['vmax'] / 2.0 ) * ( self.oms_q10**qt ) /co2t['rstfac'][2] * c3 + rrkk * co2t['rstfac'][2] * c4
        co2t['soilfrz'] = 1.0 #neglect soil freezing stuff
        bintc = physcont['binter'] * vegt['lai'] * co2t['rstfac'][2] * co2t['soilfrz'] #
        gsh2o  = 1.0/co2t['rst'] * self.amagatwv*tprcor/tc #co2t_rst is prognostic stomatal resistance (s/m), see module_sib.f90
        gbh2o  = 0.5/rb * self.amagatwv*tprcor/tc
        gah2o  = 1.0/ra * self.amagatwv*tprcor/gprogt['tm']
        gxco2 = physcont['gmeso'] * vegt['vmax'] * co2t['aparkk'] * co2t['rstfac'][2]
        range_  = gprogt['pco2m'] * ( 1. - 1.6/physcont['gradm'] ) - co2t['gamma'] #added underscore to variable name, as range is a specail word in Python
        eyy = np.zeros(self.numic+1)
        pco2y = np.zeros(self.numic+1)
        assimc = np.zeros(self.numic+1)
        assime = np.zeros(self.numic+1)
        assims = np.zeros(self.numic+1)
        assimy = np.zeros(self.numic+1)
        icconv=1
        for ic in range(1,self.numic+1):
            eyy, pco2y = self.phosort(self.numic, ic, co2t['gamma'], range_, eyy, pco2y)
            assimc[ic], assime[ic], assims[ic],assimy[ic] = self.phocycalc(toa_apar, co2t['apar'], co2t['aparkk'], vegt['vmax'], co2t['vmaxss'],
                 atheta, btheta, co2t['gamma'],rrkk, omss, c3, c4, pco2y[ic])
            co2cas = (co2cas + (self.dtsib/co2cap) * (resp_cas - assimy[ic] + co2m*gah2o) )  / (1+self.dtsib*gah2o/ co2cap ) 
            pco2cas = co2cas * self.pressure
            assimn = assimy[ic] - resp_leaf
            pco2s = pco2cas - (1.4/gbh2o * assimn * self.pressure)
            pco2i = pco2s - (1.6/gsh2o * assimn * self.pressure)
            pco2c = pco2i - c3 * assimn * self.pressure * 1.0/gxco2
            eyy[ic] = pco2y[ic] - pco2c
            if (ic>=2):
                ic1 = ic-1
                if (np.abs(eyy[ic1])>=0.1):
                    icconv = ic
                else:
                    eyy[ic] = eyy[ic1]
                    pco2y[ic] = pco2y[ic1]
        assim_omc = assimc[icconv]
        assim_ome = assime[icconv]
        assim_oms = assims[icconv]
        self.assimn = assimy[icconv] - resp_auto
        print('aap')
        print(self.assimn)
        co2t['pco2c'] = pco2y[icconv]
        co2t['pco2i'] = co2t['pco2c'] + (c3 * self.assimn/gxco2*self.pressure)
        co2t['pco2s'] = co2t['pco2i'] + (1.6 * self.assimn/gsh2o*self.pressure)
        co2t['assim'] = max(0.0,assimy[icconv])
        #the line below is commented out, since we do not need it for the coupling with CLASS
        #co2t['assimd'] = co2t['assimd']*(1. - self.wt_daily) + co2t['assim']*self.wt_daily
#        co2t['clim_assim'] = (1.-self.wt_clim)*co2t['clim_assim'] + self.wt_clim*co2t['assim'] not needed, for coupling with CLASS
        assimpot_omc = vegt['vmax']*c4 +vegt['vmax']*(co2t['pco2i']-co2t['gamma'])/(co2t['pco2i']+rrkk)*c3
        assimpot_ome = toa_apar*c4 + toa_apar*(co2t['pco2i']-co2t['gamma'])/(co2t['pco2i']+2.*co2t['gamma'])*c3
        assimpot_oms = omss*(c3 + co2t['pco2i']*c4)
        assimfac = np.zeros(5) #module_phosib.f90, but index 0 will be kept zero
        if (assimpot_omc > self.minassim):
            assimdiff = max(0.,assimpot_omc - assim_omc)
            assimfac[1] = min(1., max(0., (assimdiff / assimpot_omc)))
        assimfac[1] = 1. - assimfac[1]
        assimdiff = max(0.,assimpot_ome - assim_ome)
        if (assimdiff > self.minassimdiff):
            assimfac[2] = min(1., max(0., (assimdiff / assimpot_ome)))
        assimfac[2] = 1. - assimfac[2]
        if (assimpot_oms > self.minassim):
            assimdiff = max(0.,assimpot_oms - assim_oms)
            assimfac[3] = min(1., max(0., (assimdiff / assimpot_oms)))
        assimfac[3] = 1. - assimfac[3]
        assimfac[4] = assimfac[1] * assimfac[2] * assimfac[3]
        co2s = max(co2t['pco2s'],gprogt['pco2m']*0.05) / self.pressure
        gsh2onew = (physcont['gradm'] *  max(1.0e-12,co2t['assim']) * co2t['rstfac'][1] * co2t['soilfrz'] / co2s) + bintc 
        drst = co2t['rst'] * qdamp * ((gsh2o-gsh2onew)/(pdamp*gsh2o+qdamp*gsh2onew))
        co2t['rst'] = co2t['rst'] + drst
        bintc = bintc * tc / ( self.amagatwv * tprcor)
        co2t['rst']=min( 1./bintc, co2t['rst'] )
        co2t['cflux'] = gsh2onew*(co2cas-co2m)
        co2t['pco2cas'] = co2cas * self.pressure
        assimnp = self.assimn / co2t['aparkk']
        antemp = max(0.0,assimnp)
        pco2ipot = self.pressure * (co2s-(1.6 * assimnp / ((physcont['gradm'] * antemp / co2s) + bintc)))
        omcpot = vegt['vmax']*2.1**qt*((pco2ipot-co2t['gamma']) /(pco2ipot+rrkk)*c3 + c4)    
        omepot = co2t['apar']*((pco2ipot-co2t['gamma']) / (pco2ipot+2.*co2t['gamma'])*c3 + c4) 
        omspot = (vegt['vmax'] / 2.)*(1.8**qt)*c3 + rrkk*pco2ipot*c4
        sqrtin = max(0.0,((omepot+omcpot)**2 - 4. * atheta * omepot * omcpot))
        omppot = ((omepot+omcpot)-np.sqrt(sqrtin)) / (2.*atheta)
        sqrtin = max(0.0,((omppot+omspot)**2 -4.*btheta*omppot*omspot))
        if (omppot < 1.0E-14 ):
            omppot = 0.0
        co2t['assimpot'] = ((omspot + omppot)-np.sqrt(sqrtin)) / (2.*btheta)
        co2t['assimpot'] = max(co2t['assimpot'], co2t['assim'])
        
    