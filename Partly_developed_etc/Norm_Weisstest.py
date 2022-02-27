# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:24:38 2020

@author: Bosman Peter
"""
import numpy as np
import matplotlib.pyplot as plt

#input range is a range of angles, claculate Swin based on this, using the eq from CLASS
S0 = 1368.                 # solar constant [W m-2]
szarange = np.linspace(5,80,100)
szarange = szarange * 2 * np.pi / 360.
Swinrange = np.zeros(len(szarange))
for i in range(len(szarange)):
    #cc = 0.0 #cloud cover
    cc = np.random.random(1)[0]
    sinlea = np.sin(np.pi/2. - szarange[i])
    Tr  = (0.6 + 0.2 * sinlea) * (1. - 0.4 * cc)
    Swinrange[i]  = S0 * Tr * sinlea

RATIO = np.zeros(len(Swinrange))
fPARdir = np.zeros(len(Swinrange))

#OT = 35./(1224.*np.cos(sza)**2.+1)**.5 #from MLC-CHEM, different from paper Weiss and Norman
for i in range(len(szarange)):
    Swin = Swinrange[i]
    sza = szarange[i]
    OT = 1. / np.cos(sza)
#    OT2 = 35./(1224.*np.cos(sza)**2.+1)**.5 #from MLC-CHEM
    RDV = 600 * np.exp(-0.185 * OT ) * np.cos(sza) #cos(sza) in paper, although works better without. P/P0 term neglected, close to 1
    #RdV = 0.4 * (600 - RDV) * np.cos(sza) #orig
    RdV = 0.4 * (600 * np.cos(sza) - RDV) #What I think
    RV = RdV + RDV #first four lines as in MLC-CHEM
    w_infr = 1320 * 10**(-1.1950 + 0.4459 * np.log10(OT) - 0.0345 * (np.log10(OT))**2) #different from MLC-CHEM
#    w_infr2= 1320*.077*(2.*OT)**0.3 #MLC-CHEM
#    print('winfrdiff')
#    print((w_infr2-w_infr)/w_infr)
#    print('OTdiff')
#    print((OT2-OT)/OT)
    RDN = (720*np.exp(-0.06 * OT) - w_infr) * np.cos(sza) # np.cos(sza) in paper, although works better without
    #RdN = 0.6 * (720 - RDN - w_infr) * np.cos(sza) #orig
    RdN = 0.6 * (720 * np.cos(sza) - RDN - w_infr * np.cos(sza)) #What I think
    RN = RdN + RDN
    RATIO[i] = Swin/(RV+RN)
    if RATIO[i] > 0.9: #text p5 paper
        RATIO[i] = 0.9
#    if RATIO[i] > 0.88: #text p5 paper
#        RATIO[i] = 0.88
    fPARdif = 1 - (RDV / RV * (1 - ((0.9 - RATIO[i]) / 0.7)**(2/3))) #eq 11 and 1 - direct fraction is diffuse fraction. 7.0 instead of 0.7 in MLC-CHEM!!
    fPARdir[i] = 1 - fPARdif

plt.figure()
plt.axis([0, 1, 0, 1])
plt.plot(RATIO,fPARdir,label='')
plt.xlabel('RATIO')
plt.ylabel('PAR beam frac')

#we can also choose RATIO, to reproduce the line in fig 2
#we will plot for several sza (one line for each sza)
plt.figure()
plt.axis([0, 1, 0, 1])
plt.xlabel('RATIO')
plt.ylabel('PAR beam frac')
szarange = np.linspace(5,80,4)
szarange = szarange * 2 * np.pi / 360.
RATIO = np.linspace(0.1,0.99,25)
for i in range(len(szarange)):
    sza = szarange[i]
    OT = 1. / np.cos(sza)
    RDV = 600 * np.exp(-0.185 * OT ) * np.cos(sza) #cos(sza) in paper. P/P0 term neglected, close to 1
    RdV = 0.4 * (600 - RDV) * np.cos(sza) #orig
    #RdV = 0.4 * (600 * np.cos(sza) - RDV) #What I think
    RV = RdV + RDV #first four lines as in MLC-CHEM
    fPARdir = np.zeros(len(RATIO))
    for j in range(len(RATIO)):
        fPARdif = 1 - (RDV / RV * (1 - ((0.9 - RATIO[j]) / 0.7)**(2/3))) #eq 11 and 1 - direct fraction is diffuse fraction. 7.0 instead of 0.7 in MLC-CHEM!!
        fPARdir[j] = 1 - fPARdif
    plt.plot(RATIO,fPARdir,label='sza ' + str(round(sza,2)))  
plt.legend()
