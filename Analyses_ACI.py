#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 08:18:49 2025

@author: devigne
"""

import pandas as pd
import xarray as xr
import os
import dask
import matplotlib.pyplot as plt
import glob
import numpy as np
from dask import delayed
# Logging configuration
import logging
logging.basicConfig(level=logging.DEBUG)
from dask.distributed import Client, LocalCluster
import multiprocessing


# Function to get list of files matching a pattern
def get_files(directory, pattern):
    return glob.glob(os.path.join(directory, pattern))

if __name__=="__main__":
    multiprocessing.set_start_method('spawn')
    # Configure the Dask LocalCluster
    cluster = LocalCluster(n_workers=1, threads_per_worker=1, memory_limit='128GB')  # Adjust memory per worker
    client = Client(cluster)
        
    """
    def fire_events(data_set, region, param_1, param_2):
        data_set['date'] = pd.to_datetime(data_set['date'])
        df_filtered = data_set[data_set['Zone'] == region]
        #df_filtered = df_filtered[df_filtered['date'].dt.month.isin([1, 2, 12])]
        threshold = df_filtered[param_1].quantile(0.75)
        df_filtered['event'] = ma.masked_where((df_filtered[param_1].values<threshold), df_filtered[param_2])
        df_filtered['climatology'] = ma.masked_where((df_filtered[param_1].values>=threshold), df_filtered[param_2])
        return df_filtered
    """
    
    # Base directory for data
    tropomi_dir_base = '/LARGE14/devigne/Données_TROPOMI_MODIS'
    year = 2020
    months = ['01']
    
    # Define the path and file pattern
    data_files = [file for month in months for file in get_files(f'{tropomi_dir_base}/{year}/', f'resampled_data_tropomodis_{year}-{month}-*.nc')]
    
    # Initialize lists for storing DataFrames
    dataframes_aerosol = []
    modis_aus_list = []
    print(data_files)
    # Process each NetCDF file
    for filename in data_files:
        try:
            # Open the NetCDF file with Dask chunks
            ds = xr.open_dataset(
                filename,
                engine='netcdf4',
                chunks='auto',  # Laissez xarray détecter les chunks
                drop_variables=['dist_aac_abl_dif', 'dist_aac_bbl_dif', 'dist_clr_abl_dif', 
                                'dist_clr_bbl_dif', 'ai_aac_abl_dif', 'ai_aac_bbl_dif', 
                                'ai_clr_abl_dif', 'ai_clr_bbl_dif', 'af_aac_abl_dif', 
                                'af_aac_bbl_dif', 'af_clr_abl_dif', 'af_clr_bbl_dif', 
                                'cf', 'ctt', 'blh', 'alh', 'cth']
            )
            ds = ds.sel(latitude=slice(-60, 0), longitude=slice(-180, -90))
        
            
            # Persist dataset in memory
            #ds = ds.persist()
    
            # Drop 'index' variable if it exists
            if 'index' in ds.variables:
                ds = ds.drop_vars('index')
            
            ds = ds.to_dataframe()
            #ds = ds[ds['Zone'].isin([10,11])]
            ds['LWP'] = ds['LWP']*1e4
            # Compute the mask explicitly before applying it
            modis_aus = ds[
                (ds['ai_clr_bbl_abs'] <= 0.05) |
                (ds['ai_aac_abl_abs'] <= 0.05) |
                (ds['ai_aac_bbl_abs'] <= 0.05) |
                (ds['ai_aac_abl_abs'] == np.nan) | 
                (ds['ai_aac_bbl_abs'] == np.nan) | 
                (ds['ai_clr_bbl_abs'] == np.nan) 
            ]
    
    
            
            dataframes_aerosol.append(ds.reset_index())
            modis_aus_list.append(modis_aus.reset_index())
    
        except Exception as e:
            print(f"Could not process file {filename}: {e}")
    
    # Concatenate all DataFrames along axis 0
    final_df_aerosol = pd.concat(dataframes_aerosol, axis=0, ignore_index=True) if dataframes_aerosol else None
    final_modis_aus = pd.concat(modis_aus_list, axis=0, ignore_index=True) if modis_aus_list else None
    
    # Set 'date' as the index column if it exists
    if final_df_aerosol is not None and 'date' in final_df_aerosol.columns:
        final_df_aerosol['date'] = pd.to_datetime(final_df_aerosol['date'])
        final_df_aerosol = final_df_aerosol.set_index('date')
    
    if final_modis_aus is not None and 'date' in final_modis_aus.columns:
        final_modis_aus['date'] = pd.to_datetime(final_modis_aus['date'])
        final_modis_aus = final_modis_aus.set_index('date')
    
    # Display the final DataFrame structures
    if final_df_aerosol is not None:
        print("Final Aerosol DataFrame:")
        print(final_df_aerosol.head())
    else:
        print("No data processed for final_df_aerosol.")
    
    if final_modis_aus is not None:
        print("Final MODIS AUS DataFrame:")
        print(final_modis_aus.head())
    else:
        print("No data processed for final_modis_aus.")
        
    """
    ###BOXPLOTS###
    """
    
    aus_cloudy_abs_data2 = final_df_aerosol
    modis_aus = final_modis_aus
    del final_df_aerosol, final_modis_aus
    import matplotlib.lines as mlines
    # Filtrage des lignes non-NaN pour chaque colonne d'intérêt
    df_abc = aus_cloudy_abs_data2[~(aus_cloudy_abs_data2['ai_aac_bbl_abs'].isna())]
    
    df_clear = aus_cloudy_abs_data2[~(aus_cloudy_abs_data2['ai_clr_bbl_abs'].isna())]
    
    aus_cloudy_abs_data2 = aus_cloudy_abs_data2[~(aus_cloudy_abs_data2['ai_aac_abl_abs'].isna())]
    #modis_aus = modis_aus[non_nan]
    q1 = np.nanquantile(aus_cloudy_abs_data2['ai_aac_abl_abs'], 0.10)
    q3 = np.nanquantile(aus_cloudy_abs_data2['ai_aac_abl_abs'], 0.90)
    
    
    q1_2 = np.nanquantile(df_abc['ai_aac_bbl_abs'], 0.10)
    q3_2 = np.nanquantile(df_abc['ai_aac_bbl_abs'], 0.90)
    
    q1_clr = np.nanquantile(df_clear['ai_clr_bbl_abs'], 0.10)
    q3_clr = np.nanquantile(df_clear['ai_clr_bbl_abs'], 0.90)
    
    print(df_abc, df_clear)
    print(q1, q3, q1_2, q3_2, q1_clr, q3_clr)
    print(df_abc['LWP'])
    
    #CER Histograms
    
    def func(x):
        if x <= q1:
            return 'LP'
        elif (x>q1 and x< q3):
            return 'MP'
        else:
            return 'HP'
    def func2(x):
        if x <= q1_2:
            return 'LP'
        elif (x>q1_2 and x< q3_2):
            return 'MP'
        else:
            return 'HP'
    
    import numpy.ma as ma
    
    import numpy as np
    
    import seaborn as sns
    
    
    
    # Paramètres nuageux
    parameters = ['cot', 'cer', 'Nd', 'LWP']
    titles = [r'$COT_{3.7µm}$', r'$CER_{3.7µm} (µm)$', r'$Nd$ (cm$^{-3}$)', r'$LWP$ (g/m$^2$)']
    positions = [1, 2.5, 3.5, 4.5, 6, 7, 8]
    
    # Initialisation de la figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Couleurs
    colors = ['r', 'b']
    colors_hp = dict(color=colors[0])
    colors_lp = dict(color=colors[1])
    
    
    # Boucle pour chaque paramètre
    for i, (param, title) in enumerate(zip(parameters, titles)):
        ax = axes[i]
    
        aus_cloudy_abs_data2['HP'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values<q3), aus_cloudy_abs_data2[param])
        aus_cloudy_abs_data2['LP'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values>q1), aus_cloudy_abs_data2[param])
        df_abc['HP'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values<q3_2), df_abc[param])
        df_abc['LP'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values>q1_2), df_abc[param])
        df_clear['HP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param])
        df_clear['LP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param])
    
        
        count_aac_bbl_hp, count_aac_abl_hp, count_clear_bbl_hp = df_abc['HP'].count(), aus_cloudy_abs_data2['HP'].count(), df_clear['HP'].count()
        count_aac_bbl_lp, count_aac_abl_lp, count_clear_bbl_lp = df_abc['LP'].count(), aus_cloudy_abs_data2['LP'].count(), df_clear['LP'].count()
    
        red_line = mlines.Line2D([], [], color='red', linestyle='-', label='HP - AAC (AI>%.2f ; N=%.0f); AAC_BBL (AI>%.2f ; N=%.0f); CLR_BBL (AI>%.2f ; N=%.0f)'%(q3, count_aac_abl_hp, q3_2, count_aac_bbl_hp, q3_clr, count_clear_bbl_hp) )
        green_line = mlines.Line2D([], [], color='blue', linestyle='-', label='LP - AAC (AI<%.2f ; N=%.0f); AAC_BBL (AI<%.2f ; N=%.0f); CLR_BBL (AI<%.2f ; N=%.0f)'%(q1, count_aac_abl_lp, q1_2, count_aac_bbl_lp, q1_clr, count_clear_bbl_lp) )
    
        # Ajout des distributions
        sns.violinplot(data=[modis_aus[param]], positions=[1], color='k', ax=ax, width=0.4)
        sns.violinplot(data=[aus_cloudy_abs_data2['HP']], positions=[6], color=colors[0], ax=ax, width=0.4)
        sns.violinplot(data=[aus_cloudy_abs_data2['LP']], positions=[2.5], color=colors[1], ax=ax, width=0.4)
        sns.violinplot(data=[df_abc['HP']], positions=[7], color=colors[0], ax=ax, width=0.4)
        sns.violinplot(data=[df_abc['LP']], positions=[3.5], color=colors[1], ax=ax, width=0.4)
        sns.violinplot(data=[df_clear['HP']], positions=[8], color=colors[0], ax=ax, width=0.4)
        sns.violinplot(data=[df_clear['LP']], positions=[4.5], color=colors[1], ax=ax, width=0.4)
    
        # Lignes médianes et moyennes globales
        ax.axhline(np.nanmedian(modis_aus[param]), color='k', linestyle='--', label='Médiane globale')
        ax.axhline(np.nanmean(modis_aus[param]), color='g', linestyle='--', label='Moyenne globale')
    
        # Réglages de l'axe
        ax.set_ylabel(title, fontsize=13)
        ax.set_xticks(positions)
        ax.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize=8)
    
        # Légende spécifique pour HP/LP
        ax.legend(handles=[
            mlines.Line2D([], [], color='r', label='HP'),
            mlines.Line2D([], [], color='b', label='LP'),
        ], loc='upper right')
    
    fig.subplots_adjust(bottom=0.7)  # Adjust the top to make space for the legend
    
    # Add the custom handles to the legend at the top
    fig.legend(handles=[red_line, green_line],
               fontsize=12, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.05), fancybox=True)
    
    fig.suptitle('Southeastern Pacific (Jan 2020)', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the title
    
    plt.show()
    fig.savefig('/home/devigne/Wildfires_Cases/SEPAC_DJF_Violinplot.pdf', bbox_inches='tight')
    plt.close()
    # Ajustement global
    plt.tight_layout()
    plt.show()



#%%
"""
    from scipy.stats import norm
    import numpy.ma as ma
    # Assuming `fire_events` and `final_df` are defined and available
    # Assuming `regions` and `param` are also defined
    regions = ['Peruvian', 'Namibian', 'Australian', 'Californian', 'Canarian', 'China', 'North Atlantic', 'Northeast Pacific', 'Northwest Pacific', 'Southeast Pacific', 'South Atlantic', 'South Indian Ocean', 'Galapagos', 'Chinese Stratus', 'Amazon', 'Equatorial Africa', 'North America', 'India', 'Europe']
    index = range(10, 13)
    
    
    
    param_2 = 'cer'
    param_1 = 'ai_aac_abl_abs'
    for zone in index:
        data_set = fire_events(final_df_aerosol, zone, f'{param_1}', f'{param_2}')
        
        # Ensure only finite values are used
        event_data = data_set['event'].compressed()
        climatology_data = data_set['climatology'].compressed()
        print(event_data.type)
        exit()
        # Fit normal distribution to the finite data
        mu, std = norm.fit(event_data)
        mu2, std2 = norm.fit(climatology_data)
        
        # Generate x values for PDF
        x1 = np.linspace(np.min(event_data), np.max(event_data), int((np.max(event_data) - np.min(event_data)) / 0.5))
        x2 = np.linspace(np.min(climatology_data), np.max(climatology_data), int((np.max(climatology_data) - np.min(climatology_data)) / 0.5))
        
        # Calculate PDF
        p1 = norm.pdf(x1, mu, std)
        p2 = norm.pdf(x2, mu2, std2)
        
        # Plot the results
        plt.plot(x1, p1, 'r', linewidth=2)
        plt.plot(x2, p2, 'b', linewidth=2)
        plt.hist(data_set['event'].values, bins=int((np.nanmax(data_set['event'].values) - np.nanmin(data_set['event'].values))/0.5), alpha=0.4, color='red', density=True)
        plt.hist(data_set['climatology'].values, bins=int((np.nanmax(data_set['climatology'].values) - np.nanmin(data_set['climatology'].values))/0.5), alpha=0.4, color='blue', density=True)
        plt.xlim(np.nanmin(data_set['climatology'].values), np.nanmax(data_set['event'].values))
        plt.title(f'{param_2} distribution DJF 2020 \n{zone} as a function of AI condition (AAC ABL)')
        plt.xlabel(r'$CER_{3.7µm}$ (µm)')
        plt.ylabel('Density')
        plt.legend(['Particular events', 'Background'])
        plt.savefig(f'/home/devigne/Wildfires_Cases/Histogramme_{param_2}_{regions[zone]}_AAC_DJF_2020.pdf')
    
        plt.show()

"""


#%%
"""

param = 'cot'
param2 = 'cer'
param3 = 'Nd'
param4 = 'LWP'
aus_cloudy_abs_data2['HP'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values<q3), aus_cloudy_abs_data2[param])
aus_cloudy_abs_data2['LP'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values>q1), aus_cloudy_abs_data2[param])
df_abc['HP'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values<q3_2), df_abc[param])
df_abc['LP'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values>q1_2), df_abc[param])
df_clear['HP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param])
df_clear['LP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param])



aus_cloudy_abs_data2['HP1'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values<q3), aus_cloudy_abs_data2[param2])
aus_cloudy_abs_data2['LP1'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values>q1), aus_cloudy_abs_data2[param2])
df_abc['HP1'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values<q3_2), df_abc[param2])
df_abc['LP1'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values>q1_2), df_abc[param2])
df_clear['HP1'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param2])
df_clear['LP1'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param2])

aus_cloudy_abs_data2['HP2'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values<q3), aus_cloudy_abs_data2[param3])
aus_cloudy_abs_data2['LP2'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values>q1), aus_cloudy_abs_data2[param3])
df_abc['HP2'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values<q3_2), df_abc[param3])
df_abc['LP2'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values>q1_2), df_abc[param3])
df_clear['HP2'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param3])
df_clear['LP2'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param3])

aus_cloudy_abs_data2['HP3'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values<q3), aus_cloudy_abs_data2[param4])
aus_cloudy_abs_data2['LP3'] = ma.masked_where((aus_cloudy_abs_data2['ai_aac_abl_abs'].values>q1), aus_cloudy_abs_data2[param4])
df_abc['HP3'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values<q3_2), df_abc[param4])
df_abc['LP3'] = ma.masked_where((df_abc['ai_aac_bbl_abs'].values>q1_2), df_abc[param4])
df_clear['HP3'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param4])
df_clear['LP3'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param4])

# Set species names as labels for the boxplot

fig, ((axes1, axes), (axes2, axes3)) = plt.subplots(nrows = 2, ncols = 2, figsize = (12,10))
# Set the colors for each distribution
colors = ['r', 'b']
colors_setosa = dict(color=colors[0])
colors_versicolor = dict(color=colors[1])

# We want to apply different properties to each species, so we're going to plot one boxplot
# for each species and set their properties individually
# positions: position of the boxplot in the plot area
# medianprops: dictionary of properties applied to median line
# whiskerprops: dictionary of properties applied to the whiskers
# capprops: dictionary of properties applied to the caps on the whiskers
# flierprops: dictionary of properties applied to outliers
#df_data1.boxplot(['Non AA TROPOMI'], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes1, widths = 0.4)

###COT###
modis_aus.violinplot([param], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['HP'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['LP'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_abc.violinplot(['HP'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_abc.violinplot(['LP'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_clear.violinplot(['HP'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_clear.violinplot(['LP'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)

#df_data1.violinplot(['MP'], positions=[5.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
#df_data1.violinplot(['LP'], positions=[6], boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)


axes1.set_ylabel(r'$COT_{3.7µm} $', fontsize = 13)
axes1.hlines(y= np.nanmedian(modis_aus[param]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes1.hlines(y= np.nanmean(modis_aus[param]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes1.set_title(r'$COT_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes1.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes1.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)
# Create custom legend handles
red_line = mlines.Line2D([], [], color='red', linestyle='-', label='HP - AAC (AI>%.2f ; N=%.0f); AAC_BBL (AI>%.2f ; N=%.0f); CLR_BBL (AI>%.2f ; N=%.0f)'%(q3, count_aac_abl_hp, q3_2, count_aac_bbl_hp, q3_clr, count_clear_bbl_hp) )
green_line = mlines.Line2D([], [], color='blue', linestyle='-', label='LP - AAC (AI<%.2f ; N=%.0f); AAC_BBL (AI<%.2f ; N=%.0f); CLR_BBL (AI<%.2f ; N=%.0f)'%(q1, count_aac_abl_lp, q1_2, count_aac_bbl_lp, q1_clr, count_clear_bbl_lp) )


# Set the colors for each distribution
colors = ['r', 'b']
colors_setosa = dict(color=colors[0])
colors_versicolor = dict(color=colors[1])

# We want to apply different properties to each species, so we're going to plot one boxplot
# for each species and set their properties individually
# positions: position of the boxplot in the plot area
# medianprops: dictionary of properties applied to median line
# whiskerprops: dictionary of properties applied to the whiskers
# capprops: dictionary of properties applied to the caps on the whiskers
# flierprops: dictionary of properties applied to outliers

#df_data2.boxplot(['Non AA TROPOMI'], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes, widths = 0.4)

###CER###
modis_aus.violinplot([param2], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['HP1'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['LP1'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_abc.violinplot(['HP1'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_abc.violinplot(['LP1'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_clear.violinplot(['HP1'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_clear.violinplot(['LP1'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)


axes.set_ylabel(r'$CER_{3.7µm} (µm) $', fontsize = 13)
axes.hlines(y= np.nanmedian(modis_aus[param2]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes.hlines(y= np.nanmean(modis_aus[param2]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes.set_title(r'$CER_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)

####ND####
modis_aus.violinplot([param3], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['HP2'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['LP2'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_abc.violinplot(['HP2'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_abc.violinplot(['LP2'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_clear.violinplot(['HP2'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_clear.violinplot(['LP2'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)


axes2.set_ylabel(r'$Nd_{3.7µm} (cm^{-3})$', fontsize = 13)
axes2.hlines(y= np.nanmedian(modis_aus[param3]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes2.hlines(y= np.nanmean(modis_aus[param3]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes.set_title(r'$CER_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes2.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes2.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)

####LWP####
modis_aus.violinplot([param4], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['HP3'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
aus_cloudy_abs_data2.violinplot(['LP3'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_abc.violinplot(['HP3'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_abc.violinplot(['LP3'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_clear.violinplot(['HP3'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_clear.violinplot(['LP3'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)


axes3.set_ylabel(r'$LWP_{3.7µm} (g.cm^{-2})$', fontsize = 13)
axes3.hlines(y= np.nanmedian(modis_aus[param4]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes3.hlines(y= np.nanmean(modis_aus[param4]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes.set_title(r'$CER_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes3.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes3.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)


fig.subplots_adjust(bottom=0.7)  # Adjust the top to make space for the legend

# Add the custom handles to the legend at the top
fig.legend(handles=[red_line, green_line],
           fontsize=12, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -0.05), fancybox=True)

fig.suptitle('SEA (JAS) 2019', fontsize=16)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the title

plt.show()
fig.savefig('/home/devigne/Wildfires_Cases/SEPAC_DJF_Violinplot.pdf', bbox_inches='tight')
plt.close()
"""

