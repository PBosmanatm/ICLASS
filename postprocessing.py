# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:52:00 2019

@author: Bosman Peter
"""
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
import seaborn as sb
import os
import matplotlib.style as style
import scipy.interpolate as interp
import glob
style.use('classic')
plt.rc('font', size=11) #plot font size
figformat = 'png' #the format in which you want figure output, e.g. 'png'
nr_bins = 10
nr_bins_int = 200 #nr of bins after interpolation
interp_pdf = False
remove_prev = True

MultiWordHeaders = ['minimum costf', 'chi squared']
for i in range(len(MultiWordHeaders)):
    MultiWordHeaders[i] = ''.join(MultiWordHeaders[i].split()) #removes the spaces in between

disp_units_par = {}
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
disp_units_par['alfa_sto'] = '-'
disp_units_par['alpha'] = '-'
disp_units_par['EnBalDiffObsHFrac'] = '-'
disp_units_par['wg'] = '-'

if remove_prev:
    filelist_list = []
    filelist_list += [glob.glob('pp_pdf_posterior*')] #add Everything starting with 'pdf_posterior' to the list
    filelist_list += [glob.glob('pp_pdf2d_posterior*')]
    filelist_list += [glob.glob('pp_correls*')]
    for filelist in filelist_list:
        for filename in filelist:
            if os.path.isfile(filename): #in case file occurs in two filelists in filelist_list, two attempts to remove would give error
                os.remove(filename)

ensemble = False
with open('Optstatsfile.txt','r') as StatsFile:
    for index, line in enumerate(StatsFile):
        if 'optimal state without ensemble:' in line:
            state = StatsFile.readline().split() #readline reads the next line
            opt_state = StatsFile.readline().split()
            post_cor_matr = np.zeros((len(state),len(state)))
        elif 'optimal state with ensemble' in line:
            opt_state = StatsFile.readline().split()
            ensemble = True
        elif 'index member with best state:' in line:
            opt_sim_nr = int(StatsFile.readline().split()[-1]) #so go to next line, and take first part
        elif 'estim post state corr matrix:' in line:
            StatsFile.readline()
            for i in range(len(state)):
                line_to_use = StatsFile.readline().split()[1:]
                for j in range(len(state)):
                    post_cor_matr[i,j] = line_to_use[j]
                StatsFile.readline()
            post_cor_matr = np.round(post_cor_matr,2)
        elif 'optimised ensemble members:' in line:
            ensemble = []
            headers = StatsFile.readline().split()
            columncounter = 0
            StateStartIndFound = False
            SuccesColumn = False#wether the ensemble has a column 'successful'
            for wordind in range(len(headers)):
                if headers[wordind] in state and StateStartIndFound == False:
                    StateStartInd = columncounter
                    StateStartIndFound = True
                if headers[wordind] == 'successful':
                    SuccesColumnInd = columncounter
                    SuccesColumn = True
                if wordind != range(len(headers))[-1]: #cause than wordind+1 as gives an IndexError
                    if not (headers[wordind]+headers[wordind+1] in MultiWordHeaders):
                        columncounter += 1
            membernr = 0
            line_to_use = StatsFile.readline().split()
            memberdict = {}
            i = StateStartInd
            for item in state:
                memberdict[item] = float(line_to_use[i])
                i += 1
            if SuccesColumn:
                BooleanString = line_to_use[SuccesColumnInd]
                if BooleanString == 'True':
                    memberdict['successful'] = True
                else:
                    memberdict['successful'] = False   
            ensemble = np.append(ensemble,memberdict)
            continueread = True
            while continueread:
                line_to_use = StatsFile.readline().split()
                memberdict = {}
                try:
                    if float(line_to_use[0]) - membernr != 1:
                        continueread = False
                except (IndexError,ValueError) as e:
                    continueread = False
                if continueread:
                    i = StateStartInd
                    for item in state:
                        memberdict[item] = float(line_to_use[i])
                        i += 1
                    if SuccesColumn:
                        BooleanString = line_to_use[SuccesColumnInd]
                        if BooleanString == 'True':
                            memberdict['successful'] = True
                        else:
                            memberdict['successful'] = False                       
                    ensemble = np.append(ensemble,memberdict)
                    membernr += 1
            
for item in state:
    if item not in disp_units_par:
        disp_units_par[item] = ''  

if ensemble:
    mean_state_post = np.zeros(len(state))
    success_ens = np.array([x['successful'] for x in ensemble[0:]],dtype=bool)
    succes_state_ens = np.zeros(len(state),dtype=list)

    if np.sum(success_ens[1:]) > 1:
        for i in range(len(state)):
            seq = np.array([x[state[i]] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            succes_state_ens[i] = np.array([seq[x] for x in range(len(seq)) if success_ens[1:][x]])
            mean_state_post[i] = np.mean(succes_state_ens[i]) #np.nanmean not necessary since we filter already for successful optimisations
        nbins = [nr_bins,nr_bins]
        for i in range(len(state)):
            for j in range(len(state)):
                if j > i: #no need to have a pdf of both e.g. element 2 combined with 4, and 4 combined with 2
                    n, binsx, binsy = np.histogram2d(succes_state_ens[i],succes_state_ens[j], nbins, density=1)
                    x_1dpdf = np.zeros(np.size(binsx)-1)
                    y_1dpdf = np.zeros(np.size(binsy)-1)
                    for k in range(len(x_1dpdf)):
                        x_1dpdf[k] = 0.5*(binsx[k]+binsx[k+1])
                    for k in range(len(y_1dpdf)):
                        y_1dpdf[k] = 0.5*(binsy[k]+binsy[k+1])
                    fig = plt.figure()
                    if interp_pdf:
                        x_1dpdf_int = np.linspace(np.min(x_1dpdf),np.max(x_1dpdf),nr_bins_int)
                        y_1dpdf_int = np.linspace(np.min(y_1dpdf),np.max(y_1dpdf),nr_bins_int)
                        interpfunc = interp.interp2d(x_1dpdf,y_1dpdf,n)
                        n_int = interpfunc(x_1dpdf_int,y_1dpdf_int)
                    if interp_pdf:
                        plot = plt.contourf(x_1dpdf_int,y_1dpdf_int,n_int,levels = nr_bins_int)
                    else:
                        plot = plt.contourf(x_1dpdf,y_1dpdf,n,levels = nr_bins)
                    cbar = plt.colorbar(plot)
                    cbar.set_label('density (-)', rotation=270, labelpad=20)
                    plt.xlabel(state[i] + ' ('+ disp_units_par[state[i]] +')')
                    plt.ylabel(state[j] + ' ('+ disp_units_par[state[j]] +')')
                    if not os.path.exists('pdf2d_posterior_'+state[i]+'_'+state[j]+'.'+figformat): 
                        plt.savefig('pp_pdf2d_posterior_'+state[i]+'_'+state[j]+'.'+figformat, format=figformat)
                    else:
                        itemname = state[i]+'_'+state[j]
                        while os.path.exists('pdf2d_posterior_'+itemname+'.'+figformat):
                            itemname += '_'
                        plt.savefig('pp_pdf2d_posterior_'+itemname+'.'+figformat, format=figformat) 
    
    plt.figure()
    sb.set(rc={'figure.figsize':(11,11)}) 
    sb.set(font_scale=1.1)           
    plot = sb.heatmap(post_cor_matr,annot=True,xticklabels=state,yticklabels = state, cmap="RdBu_r",cbar_kws={'label': 'Correlation (-)'}, linewidths=2.0,annot_kws={"size": 11 }) 
    plot.tick_params(labelsize=11)
    plt.ylim((len(state), 0))
    plt.subplots_adjust(left=0.21, right=0.92, top=0.93, bottom=0.25,wspace=0.1)
    plt.savefig('pp_correls.'+figformat)           

