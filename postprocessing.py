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
import pickle
style.use('classic')

##############################
####### settings #############
##############################
nr_bins = 37 #for 1d-pdf
nr_bins2d = nr_bins
interp_pdf = False
if interp_pdf:
    nr_bins_int = 200 #nr of bins after interpolation
remove_prev = True
plot_obsfit = False
plot_1d_pdfs = False
plot_2d_pdfs = False
plot_pdf_panels = True
plot_colored_corr_matr = True
plot_co2profiles = False
plot_manual_fitpanels = False
plot_auto_fitpanel = False
plot_enbal_panel = True
plotfontsize = 12 #plot font size, except for legend
legendsize = plotfontsize - 1
figformat = 'eps'#the format in which you want figure output, e.g. 'png'
load_stored_objects = True
if load_stored_objects:
    storefolder_objects = 'pickle_objects' 
##############################
####### end settings #########
##############################
MultiWordHeaders = ['minimum costf', 'chi squared']
if load_stored_objects:
    if storefolder_objects not in os.listdir():
        raise Exception('Unexisting folder specified for storefolder_objects')
for i in range(len(MultiWordHeaders)):
    MultiWordHeaders[i] = ''.join(MultiWordHeaders[i].split()) #removes the spaces in between
plt.rc('font', size=plotfontsize) #plot font size

disp_units_par = {}
disp_units = {}

if remove_prev:
    filelist_list = []
#    filelist_list += [glob.glob('pp_pdf_posterior*')] 
    filelist_list += [glob.glob('pp_*')] #add Everything starting with 'pp_' to the list
    for filelist in filelist_list:
        for filename in filelist:
            if os.path.isfile(filename): #in case file occurs in two filelists in filelist_list, two attempts to remove would give error
                os.remove(filename)

if load_stored_objects:
    if 'priormodel.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/priormodel.pkl', 'rb') as input:
            priormodel = pickle.load(input)
    if 'priorinput.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/priorinput.pkl', 'rb') as input:
            priorinput = pickle.load(input)
    if 'obsvarlist.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/obsvarlist.pkl', 'rb') as input:
            obsvarlist = pickle.load(input)
    if 'disp_units.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/disp_units.pkl', 'rb') as input:
            disp_units = pickle.load(input)
    if 'disp_units_par.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/disp_units_par.pkl', 'rb') as input:
            disp_units_par = pickle.load(input)
    if 'optim.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/optim.pkl', 'rb') as input:
            optim = pickle.load(input)
    if 'obs_times.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/obs_times.pkl', 'rb') as input:
            obs_times = pickle.load(input)
    if 'measurement_error.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/measurement_error.pkl', 'rb') as input:
            measurement_error = pickle.load(input)
    if 'display_names.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/display_names.pkl', 'rb') as input:
            display_names = pickle.load(input)
    if 'optimalinput.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/optimalinput.pkl', 'rb') as input:
            optimalinput = pickle.load(input)
    if 'optimalinput_onsp.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/optimalinput_onsp.pkl', 'rb') as input:
            optimalinput_onsp = pickle.load(input)
    if 'optimalmodel.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/optimalmodel.pkl', 'rb') as input:
            optimalmodel = pickle.load(input)
    if 'optimalmodel_onsp.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/optimalmodel_onsp.pkl', 'rb') as input:
            optimalmodel_onsp = pickle.load(input)
    if 'PertData_mems.pkl' in os.listdir(storefolder_objects):
        with open(storefolder_objects+'/PertData_mems.pkl', 'rb') as input:
            PertData_mems = pickle.load(input)
        
##############################
#### units for plots #########
##############################
# e.g. disp_units_par['theta'] = 'K' for parameter theta
# or disp_units['theta'] = 'K' for observations of theta

##############################
#### end units for plots #####
##############################

use_ensemble = False
with open('Optstatsfile.txt','r') as StatsFile:
    for index, line in enumerate(StatsFile):
        if 'optimal state without ensemble:' in line:
            state = StatsFile.readline().split() #readline reads the next line
            opt_state = StatsFile.readline().split()
            post_cor_matr = np.zeros((len(state),len(state)))
        elif 'optimal state with ensemble' in line:
            opt_state = StatsFile.readline().split()
            use_ensemble = True
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
            headers_line = StatsFile.readline()
            if 'perturbed non-state params:' in headers_line:
                pert_non_state_param = True
            else:
                pert_non_state_param = False
            headers = headers_line.split()
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
        elif 'prior ensemble members:' in line:
            ensemble_p = []
            headers = StatsFile.readline().split()
            columncounter_p = 0
            StateStartIndFound = False
            for wordind in range(len(headers)):
                if headers[wordind] in state and StateStartIndFound == False:
                    StateStartInd_p = columncounter_p
                    StateStartIndFound = True
                if wordind != range(len(headers))[-1]: #cause than wordind+1 as gives an IndexError
                    if not (headers[wordind]+headers[wordind+1] in MultiWordHeaders):
                        columncounter_p += 1
            membernr_p = 0
            line_to_use = StatsFile.readline().split()
            memberdict_p = {}
            i = StateStartInd_p
            for item in state:
                memberdict_p[item] = float(line_to_use[i])
                i += 1
            if ensemble[membernr_p]['successful'] == 'True':
                memberdict_p['successful'] = True
            else:
                memberdict_p['successful'] = False   
            ensemble_p = np.append(ensemble_p,memberdict_p)
            continueread = True
            while continueread:
                line_to_use = StatsFile.readline().split()
                memberdict_p = {}
                try:
                    if float(line_to_use[0]) - membernr_p != 1:
                        continueread = False
                except (IndexError,ValueError) as e:
                    continueread = False
                if continueread:
                    i = StateStartInd_p
                    for item in state:
                        memberdict_p[item] = float(line_to_use[i])
                        i += 1
                    if ensemble[membernr_p]['successful'] == 'True':
                        memberdict_p['successful'] = True
                    else:
                        memberdict_p['successful'] = False                       
                    ensemble_p = np.append(ensemble_p,memberdict_p)
                    membernr_p += 1
                    
            
for item in state:
    if item not in disp_units_par:
        disp_units_par[item] = ''  

if use_ensemble:    
    mean_state_post = np.zeros(len(state))
    success_ens = np.array([x['successful'] for x in ensemble[0:]],dtype=bool)
    succes_state_ens = np.zeros(len(state),dtype=list)

    if np.sum(success_ens[1:]) > 1:
        for i in range(len(state)):
            seq = np.array([x[state[i]] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            succes_state_ens[i] = np.array([seq[x] for x in range(len(seq)) if success_ens[1:][x]])
            mean_state_post[i] = np.mean(succes_state_ens[i]) #np.nanmean not necessary since we filter already for successful optimisations
            if plot_1d_pdfs:
                nbins = np.linspace(np.min(succes_state_ens[i]), np.max(succes_state_ens[i]), nr_bins + 1)
                n, bins = np.histogram(succes_state_ens[i], nbins, density=1)
                pdfx = np.zeros(n.size)
                pdfy = np.zeros(n.size)
                for k in range(n.size):
                    pdfx[k] = 0.5*(bins[k]+bins[k+1])
                    pdfy[k] = n[k]
                fig = plt.figure()
                plt.plot(pdfx,pdfy, linestyle='-', linewidth=2,color='red',label='post')
                mean_state_prior = np.zeros(len(state))
                seq_p = np.array([x[state[i]] for x in ensemble_p[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
                seq_suc_p = np.array([seq_p[x] for x in range(len(seq_p)) if success_ens[1:][x]])
                mean_state_prior[i] = np.mean(seq_suc_p)
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
                plt.xlabel(state[i] + ' ('+ disp_units_par[state[i]] +')')
                plt.ylabel('Probability density (-)')  
                plt.subplots_adjust(left=0.15, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
                plt.legend(loc=0, frameon=True,prop={'size':legendsize}) 
                if not ('pp_pdf_posterior_'+state[i]+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system
                    plt.savefig('pp_pdf_posterior_'+state[i]+'.'+figformat, format=figformat)
                else:
                    itemname = state[i] + '_'
                    while ('pp_pdf_posterior_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: 
                        itemname += '_'
                    plt.savefig('pp_pdf_posterior_'+itemname+'.'+figformat, format=figformat)
        if plot_pdf_panels:
            plt.rc('font', size=22)
            plotvars = ['R10','gammaq']
            fig, ax = plt.subplots(1,2,figsize=(24,8))
            seq = np.array([x[plotvars[0]] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            succes_state_ens = np.array([seq[x] for x in range(len(seq)) if success_ens[1:][x]])
            mean_state_post = np.mean(succes_state_ens) #np.nanmean not necessary since we filter already for successful optimisations
            nbins = np.linspace(np.min(succes_state_ens), np.max(succes_state_ens), nr_bins + 1)
            n, bins = np.histogram(succes_state_ens, nbins, density=1)
            pdfx = np.zeros(n.size)
            pdfy = np.zeros(n.size)
            for k in range(n.size):
                pdfx[k] = 0.5*(bins[k]+bins[k+1])
                pdfy[k] = n[k]
            ax[0].plot(pdfx,pdfy, linestyle='-', linewidth=2,color='red',label='post')
            seq_p = np.array([x[plotvars[0]] for x in ensemble_p[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            seq_suc_p = np.array([seq_p[x] for x in range(len(seq_p)) if success_ens[1:][x]])
            mean_state_prior = np.mean(seq_suc_p)
            nbins_p = np.linspace(np.min(seq_suc_p), np.max(seq_suc_p), nr_bins + 1)
            n_p, bins_p = np.histogram(seq_suc_p, nbins_p, density=1)
            pdfx = np.zeros(n_p.size)
            pdfy = np.zeros(n_p.size)
            for k in range(n_p.size):
                pdfx[k] = 0.5*(bins_p[k]+bins_p[k+1])
                pdfy[k] = n_p[k]
            ax[0].plot(pdfx,pdfy, linestyle='dashed', linewidth=2,color='gold',label='prior')
            ax[0].axvline(mean_state_post, linestyle='-',linewidth=2,color='red',label = 'mean post')
            ax[0].axvline(mean_state_prior, linestyle='dashed',linewidth=2,color='gold',label = 'mean prior')
            ax[0].set_xlabel(plotvars[0] + ' ('+ disp_units_par[plotvars[0]] +')')
            ax[0].set_ylabel('Probability density (-)')  
            
            seq = np.array([x[plotvars[1]] for x in ensemble[1:]]) #iterate over the dictionaries,gives array. We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            succes_state_ens = np.array([seq[x] for x in range(len(seq)) if success_ens[1:][x]])
            mean_state_post = np.mean(succes_state_ens) #np.nanmean not necessary since we filter already for successful optimisations
            nbins = np.linspace(np.min(succes_state_ens), np.max(succes_state_ens), nr_bins + 1)
            n, bins = np.histogram(succes_state_ens, nbins, density=1)
            pdfx = np.zeros(n.size)
            pdfy = np.zeros(n.size)
            for k in range(n.size):
                pdfx[k] = 0.5*(bins[k]+bins[k+1])
                pdfy[k] = n[k]
            ax[1].plot(pdfx,pdfy, linestyle='-', linewidth=2,color='red',label='post')
            seq_p = np.array([x[plotvars[1]] for x in ensemble_p[1:]]) #iterate over the dictionaries,gives array . We exclude the first optimisation, since it biases the sampling as we choose it ourselves.
            seq_suc_p = np.array([seq_p[x] for x in range(len(seq_p)) if success_ens[1:][x]])
            mean_state_prior = np.mean(seq_suc_p)
            nbins_p = np.linspace(np.min(seq_suc_p), np.max(seq_suc_p), nr_bins + 1)
            n_p, bins_p = np.histogram(seq_suc_p, nbins_p, density=1)
            pdfx = np.zeros(n_p.size)
            pdfy = np.zeros(n_p.size)
            for k in range(n_p.size):
                pdfx[k] = 0.5*(bins_p[k]+bins_p[k+1])
                pdfy[k] = n_p[k]
            ax[1].ticklabel_format(axis="both", style="sci", scilimits=(0,0))
            ax[1].plot(pdfx,pdfy, linestyle='-', linewidth=2,color='gold',label='prior')
            ax[1].axvline(mean_state_post, linestyle='dashed',linewidth=2,color='red',label = 'mean post')
            ax[1].axvline(mean_state_prior, linestyle='dashed',linewidth=2,color='gold',label = 'mean prior')
            ax[1].set_xlabel(plotvars[1] + ' ('+ disp_units_par[plotvars[1]] +')')
            #ax[1].set_ylabel('Probability density (-)')             
            ax[1].legend(loc=0, frameon=True,prop={'size':21}) 
            ax[0].annotate('(a)',
            xy=(0.00, 1.082), xytext=(0,0),
            xycoords=('axes fraction', 'axes fraction'),
            textcoords='offset points',
            size=20, fontweight='bold', ha='left', va='top')
            ax[1].annotate('(b)',
            xy=(0.00, 1.082), xytext=(0,0),
            xycoords=('axes fraction', 'axes fraction'),
            textcoords='offset points',
            size=20, fontweight='bold',ha='left', va='top')
            plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.10,wspace=0.1)
            plt.savefig('pp_pdfpanel_posterior.'+figformat, format=figformat)
            plt.rc('font', size=plotfontsize) #plot font size
        if plot_2d_pdfs:
            nbins = [nr_bins2d,nr_bins2d]
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
                            plot = plt.contourf(x_1dpdf,y_1dpdf,n,levels = nr_bins2d)
                        cbar = plt.colorbar(plot)
                        cbar.set_label('density (-)', rotation=270, labelpad=20)
                        plt.xlabel(state[i] + ' ('+ disp_units_par[state[i]] +')')
                        plt.ylabel(state[j] + ' ('+ disp_units_par[state[j]] +')')
                        if not ('pdf2d_posterior_'+state[i]+'_'+state[j]+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: 
                            plt.savefig('pp_pdf2d_posterior_'+state[i]+'_'+state[j]+'.'+figformat, format=figformat)
                        else:
                            itemname = state[i]+'_'+state[j]
                            while ('pdf2d_posterior_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]:
                                itemname += '_'
                            plt.savefig('pp_pdf2d_posterior_'+itemname+'.'+figformat, format=figformat) 
        if plot_colored_corr_matr:    
            plt.figure()
            sb.set(rc={'figure.figsize':(11,11)}) 
            sb.set(font_scale=1.05)           
            plot = sb.heatmap(post_cor_matr,annot=True,xticklabels=state,yticklabels = state, cmap="RdBu_r",cbar_kws={'label': 'Correlation (-)'}, linewidths=0.7,annot_kws={"size": 10.2 }) 
            plot.tick_params(labelsize=11)
            plt.ylim((len(state), 0))
            plt.subplots_adjust(left=0.21, right=0.92, top=0.93, bottom=0.25,wspace=0.1)
            plt.savefig('pp_correls.'+figformat)  
            #Now reset the plot params:
            plt.rcParams.update(plt.rcParamsDefault)
            style.use('classic')
            plt.rc('font', size=plotfontsize) #plot font size
    
if plot_obsfit:
    for i in range(len(obsvarlist)):
        unsca = 1 #a scale for plotting the obs with different units
        if (disp_units[obsvarlist[i]] == 'g/kg' or disp_units[obsvarlist[i]] == 'g kg$^{-1}$') and (obsvarlist[i] == 'q' or obsvarlist[i].startswith('qmh')): #q can be plotted differently for clarity
            unsca = 1000
        fig = plt.figure()
        plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]],yerr=unsca*optim.__dict__['error_obs_'+obsvarlist[i]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
        plt.errorbar(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]],yerr=unsca*measurement_error[obsvarlist[i]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
        plt.plot(priormodel.out.t,unsca*priormodel.out.__dict__[obsvarlist[i]], ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
        plt.plot(priormodel.out.t,unsca*optimalmodel.out.__dict__[obsvarlist[i]], linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
        if use_ensemble:
            if pert_non_state_param and opt_sim_nr != 0:
                plt.plot(priormodel.out.t,unsca*optimalmodel_onsp.out.__dict__[obsvarlist[i]], linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
        plt.plot(obs_times[obsvarlist[i]]/3600,unsca*optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='*',color = 'black',ms=10, label = 'obs')
        if 'obs_sca_cf_'+obsvarlist[i] in state: #plot the obs scaled with the scaling factors (if applicable)
            plt.plot(obs_times[obsvarlist[i]]/3600,optimalinput.__dict__['obs_sca_cf_'+obsvarlist[i]]*unsca*optim.__dict__['obs_'+obsvarlist[i]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
        plt.ylabel(display_names[obsvarlist[i]] +' ('+ disp_units[obsvarlist[i]] + ')')
        plt.xlabel('time (h)')
        plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
        plt.legend(prop={'size':legendsize},loc=0)
        if not ('pp_fig_fit_'+obsvarlist[i]+'.'+figformat).lower() in [x.lower() for x in os.listdir()]: #os.path.exists can be case-sensitive, depending on operating system
            plt.savefig('pp_fig_fit_'+obsvarlist[i]+'.'+figformat, format=figformat)
        else:
            itemname = obsvarlist[i] + '_'
            while ('pp_fig_fit_'+itemname+'.'+figformat).lower() in [x.lower() for x in os.listdir()]:
                itemname += '_'
            plt.savefig('pp_fig_fit_'+itemname+'.'+figformat, format=figformat)#Windows cannnot have a file 'fig_fit_h' and 'fig_fit_H' in the same folder. The while loop can also handle e.g. the combination of variables Abc, ABC and abc         

    if 'FracH' in state:
        if 'H' in obsvarlist:
            enbal_corr_H = optim.obs_H + optimalinput.FracH * optim.EnBalDiffObs_atHtimes
            fig = plt.figure()
            plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=optim.__dict__['error_obs_H'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            plt.errorbar(obs_times['H']/3600,enbal_corr_H,yerr=measurement_error['H'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
            plt.plot(priormodel.out.t,priormodel.out.H, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
            plt.plot(optimalmodel.out.t,optimalmodel.out.H, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
            if use_ensemble:
                if pert_non_state_param and opt_sim_nr != 0:
                    plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.H, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
            plt.plot(obs_times['H']/3600,optim.__dict__['obs_'+'H'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
            plt.plot(obs_times['H']/3600,enbal_corr_H, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
            plt.ylabel('H (' + disp_units['H']+')')
            plt.xlabel('time (h)')
            plt.legend(prop={'size':legendsize},loc=0)
            plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
            plt.savefig('pp_fig_fit_enbalcorrH.'+figformat, format=figformat)
        if 'LE' in obsvarlist:        
            enbal_corr_LE = optim.obs_LE + (1 - optimalinput.FracH) * optim.EnBalDiffObs_atLEtimes
            fig = plt.figure()
            plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=optim.__dict__['error_obs_LE'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            plt.errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=measurement_error['LE'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
            plt.plot(priormodel.out.t,priormodel.out.LE, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
            plt.plot(optimalmodel.out.t,optimalmodel.out.LE, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
            if use_ensemble:
                if pert_non_state_param and opt_sim_nr != 0:
                    plt.plot(optimalmodel.out.t,optimalmodel_onsp.out.LE, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
            plt.plot(obs_times['LE']/3600,optim.__dict__['obs_'+'LE'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
            plt.plot(obs_times['LE']/3600,enbal_corr_LE, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
            plt.ylabel('LE (' + disp_units['LE']+')')
            plt.xlabel('time (h)')
            plt.legend(prop={'size':legendsize},loc=0)
            plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
            plt.savefig('pp_fig_fit_enbalcorrLE.'+figformat, format=figformat)


if plot_co2profiles:
    #The following code can be used to plot profiles of CO2 (adapt depending on the optimisation performed)
    profileheights = np.array([priorinput.CO2measuring_height4,priorinput.CO2measuring_height3,priorinput.CO2measuring_height2,priorinput.CO2measuring_height])    
    colorlist = ['red','gold','green','blue','orange','pink']
    markerlist = ['x','v','s','p']
    
    fig = plt.figure()
    i = 0
    for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
        color = colorlist[i]
        plt.plot(priormodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'pmod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
        plt.plot(priormodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
        plt.plot(priormodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
        plt.plot(priormodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
        i += 1
    plt.ylabel('height (m)')
    plt.xlabel('CO2 mixing ratio ('+disp_units['CO2mh']+')')
    plt.ylim([np.min(profileheights)-0.01*(np.max(profileheights)-np.min(profileheights)),np.max(profileheights)+0.01*(np.max(profileheights)-np.min(profileheights))]) 
    i = 0
    for ti in range(0,len(obs_times['CO2mh']),2):
        marker = markerlist[i]
        color = colorlist[i]
        plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
        plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
        plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
        plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
        i += 1
    plt.legend(fontsize=8,loc=0)  
    plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    plt.savefig('pp_fig_'+'CO2'+'_profile_prior.'+figformat, format=figformat)
    
    fig = plt.figure()
    i = 0
    for ti in range(int(30*60/priorinput.dt),priormodel.tsteps,120):
        color = colorlist[i]
        plt.plot(optimalmodel.out.__dict__['CO2mh'][ti],profileheights[3], linestyle=' ', marker='o',color=color,label = 'mod t='+str((priorinput.tstart*3600+ti*priorinput.dt)/3600))
        plt.plot(optimalmodel.out.__dict__['CO2mh2'][ti],profileheights[2], linestyle=' ', marker='o',color=color)
        plt.plot(optimalmodel.out.__dict__['CO2mh3'][ti],profileheights[1], linestyle=' ', marker='o',color=color)
        plt.plot(optimalmodel.out.__dict__['CO2mh4'][ti],profileheights[0], linestyle=' ', marker='o',color=color)
        i += 1
    plt.ylabel('height (m)')
    plt.xlabel('CO2 mixing ratio ('+disp_units['CO2mh']+')') 
    plt.ylim([np.min(profileheights)-0.01*(np.max(profileheights)-np.min(profileheights)),np.max(profileheights)+0.01*(np.max(profileheights)-np.min(profileheights))]) 
    i = 0
    for ti in range(0,len(obs_times['CO2mh']),2):
        marker = markerlist[i]
        color = colorlist[i]
        plt.plot(optim.obs_CO2mh[ti],profileheights[3], linestyle=' ', marker=marker,color=color,label = 'obs t='+str((obs_times['CO2mh'][ti])/3600))
        plt.plot(optim.obs_CO2mh2[ti],profileheights[2], linestyle=' ', marker=marker,color=color)
        plt.plot(optim.obs_CO2mh3[ti],profileheights[1], linestyle=' ', marker=marker,color=color)
        plt.plot(optim.obs_CO2mh4[ti],profileheights[0], linestyle=' ', marker=marker,color=color)
        i += 1
    plt.legend(fontsize=8,loc=0)  
    plt.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.15,wspace=0.1)
    plt.savefig('pp_fig_'+'CO2'+'_profile.'+figformat, format=figformat)

if plot_manual_fitpanels:
    plotvars = ['h','qmh'] 
    unsca = np.ones(len(plotvars)) #a scale for plotting the obs with different units
    for i in range(len(plotvars)):
        if (disp_units[plotvars[i]] == 'g/kg' or disp_units[plotvars[i]] == 'g kg$^{-1}$') and (plotvars[i] == 'q' or plotvars[i].startswith('qmh')): #q can be plotted differently for clarity
            unsca[i] = 1000
    plt.rc('font', size=22)
    fig, ax = plt.subplots(1,2,figsize=(24,8))
    ax[0].errorbar(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]],yerr=unsca[0]*optim.__dict__['error_obs_'+plotvars[0]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[0].errorbar(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]],yerr=unsca[0]*measurement_error[plotvars[0]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[0].plot(priormodel.out.t,unsca[0]*priormodel.out.__dict__[plotvars[0]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[0].plot(optimalmodel.out.t,unsca[0]*optimalmodel.out.__dict__[plotvars[0]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[0].plot(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs')
    if 'obs_sca_cf_'+plotvars[0] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[0].plot(obs_times[plotvars[0]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[0]]*unsca[0]*optim.__dict__['obs_'+plotvars[0]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[0].set_ylabel(display_names[plotvars[0]] +' ('+ disp_units[plotvars[0]] + ')')
    ax[0].set_xlabel('time (h)')
    ax[1].errorbar(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]],yerr=unsca[1]*optim.__dict__['error_obs_'+plotvars[1]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[1].errorbar(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]],yerr=unsca[1]*measurement_error[plotvars[1]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[1].plot(priormodel.out.t,unsca[1]*priormodel.out.__dict__[plotvars[1]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[1].plot(optimalmodel.out.t,unsca[1]*optimalmodel.out.__dict__[plotvars[1]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[1].plot(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]], linestyle=' ', marker='*',color = 'black',ms=10, label = 'obs')
    if 'obs_sca_cf_'+plotvars[1] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[1].plot(obs_times[plotvars[1]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[1]]*unsca[1]*optim.__dict__['obs_'+plotvars[1]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[1].set_ylabel(display_names[plotvars[1]] +' ('+ disp_units[plotvars[1]] + ')')
    ax[1].set_xlabel('time (h)')
    ax[1].legend(loc=0, frameon=False,prop={'size':21})
    ax[0].annotate('(a)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=20, fontweight='bold', ha='left', va='top')
    ax[1].annotate('(b)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=20, fontweight='bold',ha='left', va='top')
    plt.savefig('pp_fig_fitpanelsimple.eps', format='eps')
    
    plt.rc('font', size=17)       
    plotvars = ['h','qmh','wCO2','Tmh']  #first the var for 0,0 than 0,1 than 1,0 than 1,1
    unsca = np.ones(len(plotvars)) #a scale for plotting the obs with different units
    for i in range(len(plotvars)):
        if (disp_units[plotvars[i]] == 'g/kg' or disp_units[plotvars[i]] == 'g kg$^{-1}$') and (plotvars[i] == 'q' or plotvars[i].startswith('qmh')): #q can be plotted differently for clarity
            unsca[i] = 1000
    fig, ax = plt.subplots(2,2,figsize=(16,12))   
    ax[0,0].errorbar(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]],yerr=unsca[0]*optim.__dict__['error_obs_'+plotvars[0]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[0,0].errorbar(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]],yerr=unsca[0]*measurement_error[plotvars[0]],ecolor='black',fmt='None',label = '$\sigma_{I}$')     
    ax[0,0].plot(priormodel.out.t,unsca[0]*priormodel.out.__dict__[plotvars[0]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[0,0].plot(optimalmodel.out.t,unsca[0]*optimalmodel.out.__dict__[plotvars[0]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[0,0].plot(obs_times[plotvars[0]]/3600,unsca[0]*optim.__dict__['obs_'+plotvars[0]], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs')
    if 'obs_sca_cf_'+plotvars[0] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[0,0].plot(obs_times[plotvars[0]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[0]]*unsca[0]*optim.__dict__['obs_'+plotvars[0]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[0,0].set_ylabel(display_names[plotvars[0]] +' ('+ disp_units[plotvars[0]] + ')')
    ax[0,0].set_xlabel('time (h)')
    ax[0,1].errorbar(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]],yerr=unsca[1]*optim.__dict__['error_obs_'+plotvars[1]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[0,1].errorbar(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]],yerr=unsca[1]*measurement_error[plotvars[1]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[0,1].plot(priormodel.out.t,unsca[1]*priormodel.out.__dict__[plotvars[1]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[0,1].plot(optimalmodel.out.t,unsca[1]*optimalmodel.out.__dict__[plotvars[1]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[0,1].plot(obs_times[plotvars[1]]/3600,unsca[1]*optim.__dict__['obs_'+plotvars[1]], linestyle=' ', marker='*',color = 'black',ms=10, label = 'obs')
    if 'obs_sca_cf_'+plotvars[1] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[0,1].plot(obs_times[plotvars[1]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[1]]*unsca[1]*optim.__dict__['obs_'+plotvars[1]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[0,1].set_ylabel(display_names[plotvars[1]] +' ('+ disp_units[plotvars[1]] + ')')
    ax[0,1].set_xlabel('time (h)')
    
    ax[1,0].errorbar(obs_times[plotvars[2]]/3600,unsca[2]*optim.__dict__['obs_'+plotvars[2]],yerr=unsca[2]*optim.__dict__['error_obs_'+plotvars[2]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[1,0].errorbar(obs_times[plotvars[2]]/3600,unsca[2]*optim.__dict__['obs_'+plotvars[2]],yerr=unsca[2]*measurement_error[plotvars[2]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[1,0].plot(priormodel.out.t,unsca[2]*priormodel.out.__dict__[plotvars[2]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[1,0].plot(optimalmodel.out.t,unsca[2]*optimalmodel.out.__dict__[plotvars[2]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[1,0].plot(obs_times[plotvars[2]]/3600,unsca[2]*optim.__dict__['obs_'+plotvars[2]], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs')
    if 'obs_sca_cf_'+plotvars[2] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[1,0].plot(obs_times[plotvars[2]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[2]]*unsca[2]*optim.__dict__['obs_'+plotvars[2]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[1,0].set_ylabel(display_names[plotvars[2]] +' ('+ disp_units[plotvars[2]] + ')')
    ax[1,0].set_xlabel('time (h)')
    ax[1,1].errorbar(obs_times[plotvars[3]]/3600,unsca[3]*optim.__dict__['obs_'+plotvars[3]],yerr=unsca[3]*optim.__dict__['error_obs_'+plotvars[3]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[1,1].errorbar(obs_times[plotvars[3]]/3600,unsca[3]*optim.__dict__['obs_'+plotvars[3]],yerr=unsca[3]*measurement_error[plotvars[3]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[1,1].plot(priormodel.out.t,unsca[3]*priormodel.out.__dict__[plotvars[3]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
    ax[1,1].plot(optimalmodel.out.t,unsca[3]*optimalmodel.out.__dict__[plotvars[3]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
    ax[1,1].plot(obs_times[plotvars[3]]/3600,unsca[3]*optim.__dict__['obs_'+plotvars[3]], linestyle=' ', marker='*',color = 'black',ms=10, label = 'obs')
    if 'obs_sca_cf_'+plotvars[3] in state: #plot the obs scaled with the scaling factors (if applicable)
        ax[1,1].plot(obs_times[plotvars[3]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[3]]*unsca[3]*optim.__dict__['obs_'+plotvars[3]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
    ax[1,1].set_ylabel(display_names[plotvars[3]] +' ('+ disp_units[plotvars[3]] + ')')
    ax[1,1].set_xlabel('time (h)')
    ax[1,1].legend(loc=0, frameon=False,prop={'size':17})
    
    ax[0,0].annotate('(a)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=16, fontweight='bold', ha='left', va='top')
    ax[0,1].annotate('(b)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=16, fontweight='bold',ha='left', va='top')
    ax[1,0].annotate('(c)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=16, fontweight='bold',ha='left', va='top')
    ax[1,1].annotate('(d)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=16, fontweight='bold',ha='left', va='top')
    plt.savefig('pp_fig_fitpanel.eps', format='eps')
    plt.rc('font', size=plotfontsize) #reset plot font size
    
if plot_enbal_panel:
    plt.rc('font', size=19)
    fig, ax = plt.subplots(1,2,figsize=(24,8))
    enbal_corr_H = optim.obs_H + optimalinput.FracH * optim.EnBalDiffObs_atHtimes
    ax[0].errorbar(obs_times['H']/3600,enbal_corr_H,yerr=optim.__dict__['error_obs_H'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[0].errorbar(obs_times['H']/3600,enbal_corr_H,yerr=measurement_error['H'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[0].plot(priormodel.out.t,priormodel.out.H, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
    ax[0].plot(optimalmodel.out.t,optimalmodel.out.H, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
    if use_ensemble:
        if pert_non_state_param and opt_sim_nr != 0:
            ax[0].plot(optimalmodel.out.t,optimalmodel_onsp.out.H, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
    ax[0].plot(obs_times['H']/3600,optim.__dict__['obs_'+'H'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
    ax[0].plot(obs_times['H']/3600,enbal_corr_H, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
    ax[0].set_ylabel('H (' + disp_units['H']+')')
    ax[0].set_xlabel('time (h)')   
    enbal_corr_LE = optim.obs_LE + (1 - optimalinput.FracH) * optim.EnBalDiffObs_atLEtimes
    ax[1].errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=optim.__dict__['error_obs_LE'],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
    ax[1].errorbar(obs_times['LE']/3600,enbal_corr_LE,yerr=measurement_error['LE'],ecolor='black',fmt='None',label = '$\sigma_{I}$')
    ax[1].plot(priormodel.out.t,priormodel.out.LE, ls='dashed', marker='None',color='gold',linewidth = 2.0,label = 'prior')
    ax[1].plot(optimalmodel.out.t,optimalmodel.out.LE, linestyle='-', marker='None',color='red',linewidth = 2.0,label = 'post')
    if use_ensemble:
        if pert_non_state_param and opt_sim_nr != 0:
            ax[1].plot(optimalmodel.out.t,optimalmodel_onsp.out.LE, linestyle='dashdot', marker='None',color='magenta',linewidth = 2.0,label = 'post onsp')
    ax[1].plot(obs_times['LE']/3600,optim.__dict__['obs_'+'LE'], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs ori')
    ax[1].plot(obs_times['LE']/3600,enbal_corr_LE, linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs cor')
    ax[1].set_ylabel('LE (' + disp_units['LE']+')')
    ax[1].set_xlabel('time (h)')
    ax[1].legend(prop={'size':18},loc=0)
    plt.subplots_adjust(left=0.10, right=0.94, top=0.94, bottom=0.15,wspace=0.1)
    ax[0].annotate('(a)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=20, fontweight='bold', ha='left', va='top')
    ax[1].annotate('(b)',
                xy=(0.00, 1.07), xytext=(0,0),
                xycoords=('axes fraction', 'axes fraction'),
                textcoords='offset points',
                size=20, fontweight='bold',ha='left', va='top')
    plt.savefig('pp_fig_enbalpanel.eps', format='eps')
    plt.rc('font', size=plotfontsize) #reset plot font size

if plot_auto_fitpanel:    
    #Below is a more automatised panel plot:
    plt.rc('font', size=17)
    plotvars = ['h','qmh','wCO2','Tmh']  #first the var for 0,0 than 0,1 than 1,0 than 1,1
    annotatelist = ['a','b','c','d']
    unsca = np.ones(len(plotvars)) #a scale for plotting the obs with different units
    for i in range(len(plotvars)):
        if (disp_units[plotvars[i]] == 'g/kg' or disp_units[plotvars[i]] == 'g kg$^{-1}$') and (plotvars[i] == 'q' or plotvars[i].startswith('qmh')): #q can be plotted differently for clarity
            unsca[i] = 1000
    nr_rows,nr_cols = 2,2
    fig, ax = plt.subplots(nr_rows,nr_cols,figsize=(16,12))
    k = 0
    for i in range(nr_rows):
        for j in range(nr_cols):
            ax[i,j].errorbar(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]],yerr=unsca[k]*optim.__dict__['error_obs_'+plotvars[k]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            ax[i,j].errorbar(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]],yerr=unsca[k]*measurement_error[plotvars[k]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
            ax[i,j].plot(priormodel.out.t,unsca[k]*priormodel.out.__dict__[plotvars[k]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
            ax[i,j].plot(optimalmodel.out.t,unsca[k]*optimalmodel.out.__dict__[plotvars[k]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
            ax[i,j].plot(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs')
            if 'obs_sca_cf_'+plotvars[k] in state: #plot the obs scaled with the scaling factors (if applicable)
                ax[i,j].plot(obs_times[plotvars[k]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[k]]*unsca[k]*optim.__dict__['obs_'+plotvars[k]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
            ax[i,j].set_ylabel(display_names[plotvars[k]] +' ('+ disp_units[plotvars[k]] + ')')
            ax[i,j].set_xlabel('time (h)')
            ax[i,j].annotate(annotatelist[k],
            xy=(0.00, 1.07), xytext=(0,0),
            xycoords=('axes fraction', 'axes fraction'),
            textcoords='offset points',
            size=16, fontweight='bold', ha='left', va='top')
            k += 1
    ax[1,1].legend(loc=0, frameon=False,prop={'size':19})
    plt.savefig('pp_fig_fitpanel_auto.eps', format='eps')
    
    #And another one
    plt.rc('font', size=17)
    plotvars = ['Tmh2','Tmh7','CO2mh','CO2mh2']  #first the var for 0,0 than 0,1 than 1,0 than 1,1
    annotatelist = ['(a)','(b)','(c)','(d)']
    unsca = np.ones(len(plotvars)) #a scale for plotting the obs with different units
    for i in range(len(plotvars)):
        if (disp_units[plotvars[i]] == 'g/kg' or disp_units[plotvars[i]] == 'g kg$^{-1}$') and (plotvars[i] == 'q' or plotvars[i].startswith('qmh')): #q can be plotted differently for clarity
            unsca[i] = 1000
    nr_rows,nr_cols = 2,2
    fig, ax = plt.subplots(nr_rows,nr_cols,figsize=(16,12))
    k = 0
    for i in range(nr_rows):
        for j in range(nr_cols):
            ax[i,j].errorbar(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]],yerr=unsca[k]*optim.__dict__['error_obs_'+plotvars[k]],ecolor='lightgray',fmt='None',label = '$\sigma_{O}$', elinewidth=2,capsize = 0)
            ax[i,j].errorbar(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]],yerr=unsca[k]*measurement_error[plotvars[k]],ecolor='black',fmt='None',label = '$\sigma_{I}$')
            ax[i,j].plot(priormodel.out.t,unsca[k]*priormodel.out.__dict__[plotvars[k]], ls='dashed', marker='None',color='gold',linewidth = 4.0,label = 'prior',dashes = (4,4))
            ax[i,j].plot(optimalmodel.out.t,unsca[k]*optimalmodel.out.__dict__[plotvars[k]], linestyle='-', marker='None',color='red',linewidth = 4.0,label = 'post')
            ax[i,j].plot(obs_times[plotvars[k]]/3600,unsca[k]*optim.__dict__['obs_'+plotvars[k]], linestyle=' ', marker='*',color = 'black',ms=10,label = 'obs')
            if 'obs_sca_cf_'+plotvars[k] in state: #plot the obs scaled with the scaling factors (if applicable)
                ax[i,j].plot(obs_times[plotvars[k]]/3600,optimalinput.__dict__['obs_sca_cf_'+plotvars[k]]*unsca[k]*optim.__dict__['obs_'+plotvars[k]], linestyle=' ', marker='o',color = 'red',ms=10,label = 'obs sca')
            ax[i,j].set_ylabel(display_names[plotvars[k]] +' ('+ disp_units[plotvars[k]] + ')')
            ax[i,j].set_xlabel('time (h)')
            ax[i,j].annotate(annotatelist[k],
            xy=(0.00, 1.07), xytext=(0,0),
            xycoords=('axes fraction', 'axes fraction'),
            textcoords='offset points',
            size=16, fontweight='bold', ha='left', va='top')
            k += 1
    ax[1,1].legend(loc=0, frameon=False,prop={'size':19})
    plt.savefig('pp_fig_fitpanel_auto2.eps', format='eps')
    
    
    