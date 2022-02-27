# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:21:12 2020

@author: Bosman Peter
"""

import numpy as np
import canopy_model as canm
import matplotlib.pyplot as plt

thetasurf,Ts,e,w2,Swin,soilflux_COS = 298,290,0.01*100000,0.35,500,0
C_surf_layer_COS_ppb,soilflux_CO2,C_surf_layer_CO2_ppm,ra_top,windsp_surf_lay = 0.5,0,400,15,3
dt = 1.0
canopy_model = canm.canopy_mod()
nstep = 36000

COSfluxVegSurfLay = np.zeros(nstep)
CO2fluxVegSurfLay = np.zeros(nstep)
summed_veg_exchange_COS = np.zeros(nstep)
summed_veg_exchange_CO2 = np.zeros(nstep)
C_COS_veglayer_old = np.zeros((nstep,canopy_model.nveglay))
C_CO2_veglayer_old = np.zeros((nstep,canopy_model.nveglay))
turbCOSfluxes = np.zeros((nstep,canopy_model.nveglay+1))
turbCO2fluxes = np.zeros((nstep,canopy_model.nveglay+1))
veg_exchange_COS = np.zeros((nstep,canopy_model.nveglay))
veg_exchange_CO2 = np.zeros((nstep,canopy_model.nveglay))
rbveg_COS = np.zeros((nstep,canopy_model.nveglay))
rs_COS_leaf = np.zeros((nstep,canopy_model.nveglay))
for t in range (0,nstep):
    COSfluxVegSurfLay[t],CO2fluxVegSurfLay[t],summed_veg_exchange_COS[t],summed_veg_exchange_CO2[t],C_COS_veglayer_old[t],C_CO2_veglayer_old[t] = canopy_model.run_canopy_mod(thetasurf,Ts,e,w2,Swin,soilflux_COS,C_surf_layer_COS_ppb,soilflux_CO2,C_surf_layer_CO2_ppm,ra_top,windsp_surf_lay,dt)
    turbCOSfluxes[t] = canopy_model.turbCOSfluxes
    turbCO2fluxes[t] = canopy_model.turbCO2fluxes
    veg_exchange_COS[t] = canopy_model.veg_exchange_COS
    veg_exchange_CO2[t] = canopy_model.veg_exchange_CO2
    rbveg_COS[t] = canopy_model.rbveg_COS
    rs_COS_leaf[t] = canopy_model.rs_COS_leaf
    
plt.rc('font', size=16)
plt.figure()
plt.plot(range(nstep),summed_veg_exchange_COS)
plt.xlabel('timestep')
plt.ylabel('COS_plant uptake')

plt.rc('font', size=16)
plt.figure()
plt.plot(range(nstep),summed_veg_exchange_CO2)
plt.xlabel('timestep')
plt.ylabel('CO2_plant uptake')

plt.figure()
plt.plot(range(nstep),C_COS_veglayer_old[:,0] * 1.e9 * 28.96 / 1.2 * 0.001)
plt.xlabel('timestep')
plt.ylabel('COS_conc lowest veg layer (ppb)')

plt.figure()
plt.plot(range(nstep),C_COS_veglayer_old[:,-1] * 1.e9 * 28.96 / 1.2 * 0.001)
plt.xlabel('timestep')
plt.ylabel('COS_conc highest veg layer (ppb)')

plt.figure()
plt.plot(range(nstep),C_CO2_veglayer_old[:,0] * 1.e6 * 28.96 / 1.2 * 0.001)
plt.xlabel('timestep')
plt.ylabel('CO2_conc lowest veg layer (ppm)')

plt.figure()
plt.plot(range(nstep),C_CO2_veglayer_old[:,-1] * 1.e6 * 28.96 / 1.2 * 0.001)
plt.xlabel('timestep')
plt.ylabel('CO2_conc highest veg layer (ppm)')

plt.figure()
plt.plot(C_COS_veglayer_old[-1,:] * 1.e9 * 28.96 / 1.2 * 0.001,canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('COS_conc at t = ' + str(nstep) +'(ppb)')

plt.figure()
plt.plot(C_CO2_veglayer_old[-1,:] * 1.e6 * 28.96 / 1.2 * 0.001,canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('CO2_conc at t = ' + str(nstep) +'(ppm)')

plt.figure()
plt.plot(range(nstep),COSfluxVegSurfLay)
plt.xlabel('timestep')
plt.ylabel('COS flux to surf layer ' +'(mol m-2 s-1)')

plt.figure()
plt.plot(range(nstep),CO2fluxVegSurfLay)
plt.xlabel('timestep')
plt.ylabel('CO2 flux to surf layer ' +'(mol m-2 s-1)')

plt.rc('font', size=16)
plt.figure()
plt.plot(canopy_model.U_veg,canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('u (m s-1)')

plt.rc('font', size=16)
plt.figure()
plt.plot(canopy_model.PAR,canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('PAR (W m-2)')

plt.figure()
C_surf_layer_COS = 1.e-9 / canopy_model.mair * canopy_model.rho * 1000 * C_surf_layer_COS_ppb
vd = np.zeros((nstep,canopy_model.nveglay))
for t in range(nstep):
    for i in range(canopy_model.nveglay):
        vd[t,i] = -veg_exchange_COS[t,i]/C_COS_veglayer_old[t,i]
plt.plot(vd[-1],canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('vd_COS (m s-1), abs value flux')

plt.figure()
cond = np.zeros((nstep,canopy_model.nveglay))
for t in range(nstep):
    for i in range(canopy_model.nveglay):
        cond[t,i] = 1 / (rbveg_COS[t,i]/canopy_model.LAI_veglay[i] + 1 / (1 / (rs_COS_leaf[t,i]/canopy_model.LAI_veglay[i] + canopy_model.rint_COS/canopy_model.LAI_veglay[i]) + 2 * canopy_model.gcutCOS_leaf*canopy_model.LAI_veglay[i]))
plt.plot(cond[-1],canopy_model.z_veglay)
plt.ylabel('height')
plt.xlabel('canopy scale conduct (m s-1)')

plt.figure()
C_surf_layer_COS = 1.e-9 / canopy_model.mair * canopy_model.rho * 1000 * C_surf_layer_COS_ppb
vd = np.zeros((nstep,canopy_model.nveglay+1))
for t in range(nstep):
    for i in range(canopy_model.nveglay):
        vd[t,i] = turbCOSfluxes[t,i]/C_COS_veglayer_old[t,i]
    vd[t,-1] = turbCOSfluxes[t,-1]/C_surf_layer_COS
plt.plot(vd[-1],canopy_model.z_int_veg)
plt.ylabel('height')
plt.xlabel('vd_COS (m s-1) based on turbflux')
