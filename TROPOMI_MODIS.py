#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:58:38 2024

@author: devigne
"""

"""
import xarray as xr
import numpy as np
from datetime import datetime
import os
from BLH_colocalize import blh, convert_utc_to_local
import warnings
from multiprocessing import Pool, cpu_count

# Define paths to the files
path = '/LARGE14/devigne/AI_2023'
path_2 = '/LARGE14/devigne/ALH_2023'
path_3 = '/LARGE14/devigne/CTH_2023'
path_4 = '/LARGE14/devigne/CF_2023'
path_modis = '/home/devigne/MODIS_DATA_L2'

# Creation of the different TROPOMI lists of netCDF files
def get_file_list(directory, extension=".nc"):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)])

list_ai = get_file_list(path)
list_lh = get_file_list(path_2)
list_cloud = get_file_list(path_3)
list_cf = get_file_list(path_4)
list_modis = get_file_list(path_modis)

# Function to extract dates from netCDF files
def extract_dates(file_list, start, end):
    dates = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y%m%d')
        dates.append(date_obj)
    return dates

# Find common dates among lists
def find_common_dates(*date_lists):
    common_dates = set(date_lists[0])
    for dates in date_lists[1:]:
        common_dates.intersection_update(dates)
    return common_dates

# Filter files based on common dates
def filter_files(file_list, dates_set, start, end):
    filtered_files = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y%m%d')
        if date_obj in dates_set:
            filtered_files.append(file_path)
    return filtered_files

# Resample the Data resolution of netCDF file
def get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file):
    # Define the date of the file
    d = str(ai_file)[47:55]
    date_obj = datetime.strptime(d, '%Y%m%d')
    date = date_obj.strftime('%Y-%m-%d')
    ds_blh = blh('blh', date)

    # Load datasets
    with xr.open_dataset(ai_file) as original_ds, \
         xr.open_dataset(alh_file) as alh_ds, \
         xr.open_dataset(cloud_file) as cloud_ds, \
         xr.open_dataset(cf_file) as cf_ds, \
         xr.open_dataset(modis_file) as modis_ds:

        alh1 = alh_ds['aerosol_height'].isel(time=0).values * 1000
        cth1 = cloud_ds['cloud_top_height'].isel(time=0).values
        uvai = original_ds['absorbing_aerosol_index'].isel(time=0).values
        CF = cf_ds['cloud_fraction'].isel(time=0).values
        COT = modis_ds['Cloud_Optical_Thickness_37'].values
        CER = modis_ds['Cloud_Effective_Radius_37'].values
        CTT = modis_ds['cloud_top_temperature_1km'].values

        if alh1 is None or ds_blh is None:
            dist = None
            flag_alh = None
        else:
            dist = alh1 - ds_blh
            flag_alh = alh1 > ds_blh
            
        if cth1 is None or ds_blh is None:
            flag_cth = None
            print("One of the values is None, dist set to None")
        else:
            flag_cth = cth1 > ds_blh

        flag_CF = CF > 0.01
        flag_abs = uvai > 0

        nan_uvai = np.isnan(uvai)
        nan_cf = np.isnan(CF)
        flag_cf = np.where(nan_cf, False, flag_CF)

        flag_acc_abl = (flag_cf) & (~flag_cth) & (flag_alh)
        flag_acc_bbl = (flag_cf) & (~flag_cth) & (~flag_alh)
        flag_acc_und = (flag_cf) & (flag_cth) & (flag_alh)
        flag_clr_abl = (~flag_cf) & (flag_alh)
        flag_clr_bbl = (~flag_cf) & (~flag_alh)
        
        flag_acc_abl_abs = (flag_acc_abl) & (flag_abs)
        flag_acc_bbl_abs = (flag_acc_bbl) & (flag_abs)
        flag_acc_und_abs = (flag_acc_und) & (flag_abs)
        flag_clr_abl_abs = (flag_clr_abl) & (flag_abs)
        flag_clr_bbl_abs = (flag_clr_bbl) & (flag_abs)
        
        flag_acc_abl_dif = (flag_acc_abl) & (~flag_abs)
        flag_acc_bbl_dif = (flag_acc_bbl) & (~flag_abs)
        flag_acc_und_dif = (flag_acc_und) & (~flag_abs)
        flag_clr_abl_dif = (flag_clr_abl) & (~flag_abs)
        flag_clr_bbl_dif = (flag_clr_bbl) & (~flag_abs)

        # Aerosol fraction
        af_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_acc_und_abs, flag_clr_abl_abs, flag_clr_bbl_abs, 
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_acc_und_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        af_variables = ['af_aac_abl_abs', 'af_aac_bbl_abs', 'af_aac_und_abs', 'af_clr_abl_abs', 'af_clr_bbl_abs',
                        'af_aac_abl_dif', 'af_aac_bbl_dif', 'af_aac_und_dif', 'af_clr_abl_dif', 'af_clr_bbl_dif']

        af_results = {var: np.where(flag, 1, 0) for var, flag in zip(af_variables, af_flags)}
        af_results = {var: np.where(nan_uvai, np.nan, data) for var, data in af_results.items()}

        # AI
        ai_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_acc_und_abs, flag_clr_abl_abs, flag_clr_bbl_abs,
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_acc_und_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        ai_variables = ['ai_aac_abl_abs', 'ai_aac_bbl_abs', 'ai_aac_und_abs', 'ai_clr_abl_abs', 'ai_clr_bbl_abs',
                        'ai_aac_abl_dif', 'ai_aac_bbl_dif', 'ai_aac_und_dif', 'ai_clr_abl_dif', 'ai_clr_bbl_dif']

        ai_results = {var: np.where(flag & (~nan_uvai), uvai, np.nan) for var, flag in zip(ai_variables, ai_flags)}

        # ALH - BLH = dist
        dist_flags = ai_flags  # Same flags as for AI
        dist_variables = ['dist_aac_abl_abs', 'dist_aac_bbl_abs', 'dist_aac_und_abs', 'dist_clr_abl_abs', 'dist_clr_bbl_abs',
                          'dist_aac_abl_dif', 'dist_aac_bbl_dif', 'dist_aac_und_dif', 'dist_clr_abl_dif', 'dist_clr_bbl_dif']

        dist_results = {var: np.where(flag & (~nan_uvai), dist, np.nan) for var, flag in zip(dist_variables, dist_flags)}

        variables = {
            'blh': ds_blh,
            'alh': alh1,
            'cth': cth1,
            'cf': CF,
            'ai': uvai,
            'ctt': CTT,
            'cot': COT,
            'cer': CER,
        }
        variables.update(af_results)
        variables.update(ai_results)
        variables.update(dist_results)


        # Define the target dimensions
        new_rows = 8192
        new_cols = 16384

        # Create a new dataset to store the resampled variables
        resampled_ds = xr.Dataset(coords={'latitude': np.linspace(-90, 90, new_rows),
                                           'longitude': np.linspace(-180, 180, new_cols)})

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            for var_name, data in variables.items():
                resampled_ds[var_name] = (('latitude', 'longitude'), data)

        resampled_ds.to_netcdf(f'/home/devigne/resampled_data_tropomodis_{date}.nc')

# Extract dates
dates1 = extract_dates(list_ai, 47, 55)
dates2 = extract_dates(list_lh, 48, 56)
dates3 = extract_dates(list_cloud, 61, 69)
dates_cf = extract_dates(list_cf, 58, 66)

common_date = find_common_dates(dates1, dates2, dates3, dates_cf)
comparison_date = datetime(2023, 5, 14)
end_date = datetime(2023, 5, 16)
filtered_dates = {date for date in common_date if comparison_date < date < end_date}

ai_files = filter_files(list_ai, filtered_dates, 47, 55)
alh_files = filter_files(list_lh, filtered_dates, 48, 56)
cloud_files = filter_files(list_cloud, filtered_dates, 61, 69)
cf_files = filter_files(list_cf, filtered_dates, 58, 66)

for ai_file, alh_file, cloud_file, cf_file, modis_file in zip(ai_files, alh_files, cloud_files, cf_files, list_modis):
    try:
        get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file)
    except Exception as e:
        print(f"Error processing files: {ai_file}, {alh_file}, {cloud_file}, {cf_file}, {modis_file}")
        print(f"Error: {e}")
"""

#%%

import argparse
import xarray as xr
import numpy as np
from datetime import datetime
import os
from BLH_colocalize import blh, convert_utc_to_local
import warnings
from multiprocessing import Pool, cpu_count
import dask.array as da
from netCDF4 import Dataset
#import pandas as pd
import gc

# Fonction pour récupérer les arguments en ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Traitement des fichiers MODIS")
    parser.add_argument('--ai-dir', type=str, required=True, help="Répertoire des fichiers AI")
    parser.add_argument('--lh-dir', type=str, required=True, help="Répertoire des fichiers ALH")
    parser.add_argument('--cloud-dir', type=str, required=True, help="Répertoire des fichiers Cloud")
    parser.add_argument('--cf-dir', type=str, required=True, help="Répertoire des fichiers Cloud Fraction")
    parser.add_argument('--modis-dir', type=str, required=True, help="Répertoire des fichiers MODIS")
    parser.add_argument('--eac4-dir', type=str, required=True, help="Répertoire des fichiers EAC4")
    return parser.parse_args()


# Création des listes de fichiers NetCDF
def get_file_list(directory, extension=".nc"):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)])


# Fonction pour extraire les dates des fichiers NetCDF
def extract_dates(file_list, start, end):
    dates = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y%m%d')
        dates.append(date_obj)
    return dates

# Fonction pour extraire les dates des fichiers NetCDF
def extract_dates_2(file_list, start, end):
    dates = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y_%m_%d')
        
        dates.append(date_obj)
    return dates

def extract_dates_3(file_list, start, end):
    dates = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y-%m-%d')
        dates.append(date_obj)
    return dates

# Trouver les dates communes parmi les listes
def find_common_dates(*date_lists):
    common_dates = set(date_lists[0])
    for dates in date_lists[1:]:
        common_dates.intersection_update(dates)
    return common_dates


# Filtrer les fichiers selon les dates communes
def filter_files(file_list, dates_set, start, end):
    filtered_files = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y%m%d')
        if date_obj in dates_set:
            filtered_files.append(file_path)
    return filtered_files

# Filtrer les fichiers selon les dates communes
def filter_files_2(file_list, dates_set, start, end):
    filtered_files = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y_%m_%d')
        
        if date_obj in dates_set:
            filtered_files.append(file_path)
    return filtered_files

# Filtrer les fichiers selon les dates communes
def filter_files_3(file_list, dates_set, start, end):
    filtered_files = []
    for file_path in file_list:
        file_name = file_path[start:end]
        date_obj = datetime.strptime(file_name, '%Y-%m-%d')
        
        if date_obj in dates_set:
            filtered_files.append(file_path)
    return filtered_files

# Fonction pour effectuer le traitement sur les fichiers
def get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file, eac4_file, lsm_file):
    # Définir la date du fichier
    d = str(ai_file)[47:55]
    date_obj = datetime.strptime(d, '%Y%m%d')
    date = date_obj.strftime('%Y-%m-%d')
    ds_blh = blh('blh', date)
    #print(ai_file, alh_file, cloud_file, cf_file, modis_file, eac4_file)
    chunk = 10
    with xr.open_dataset(ai_file) as original_ds, \
         xr.open_dataset(alh_file) as alh_ds, \
         xr.open_dataset(cloud_file) as cloud_ds, \
         xr.open_dataset(cf_file) as cf_ds, \
         xr.open_dataset(modis_file) as modis_ds, \
         xr.open_dataset(eac4_file) as eac4_ds:
       
        # Variables
        alh1 = alh_ds['aerosol_height'].isel(time=0).fillna(0).astype(np.float32).values * 1000
        
        cth1 = cloud_ds['cloud_top_height'].isel(time=0).fillna(0).astype(np.float32).values * 1000
        #print(alh1, cth1, np.nanmax(alh1), np.nanmax(cth1))
        
        uvai = original_ds['absorbing_aerosol_index'].isel(time=0).fillna(0).astype(np.float32).values
        CF   = cf_ds['cloud_fraction'].isel(time=0).fillna(0).astype(np.float32).values
        COT  = modis_ds['Cloud_Optical_Thickness_37'].compute()
        CER  = modis_ds['Cloud_Effective_Radius_37'].compute()
        CTT  = modis_ds['cloud_top_temperature_1km'].compute()
        Nd   = modis_ds['Nd_37'].compute()
        LWP  = (5/9)*COT*(CER*1e-4)
        N_dus = eac4_ds['N_dus'].astype(np.float32).compute()
        N_dum = eac4_ds['N_dum'].astype(np.float32).compute()
        N_dul = eac4_ds['N_dul'].astype(np.float32).compute()
        N_bchphil = eac4_ds['N_bchphil'].astype(np.float32).compute()
        N_bchphob = eac4_ds['N_bchphob'].astype(np.float32).compute()
        
        # Initialize blh_alh with alh1 and handle NaNs
        blh_alh = alh1.copy()
        blh_alh[np.isnan(alh1)] = np.nan
        blh_alh[np.isnan(ds_blh)] = np.nan
        
        # Calculate distance
        blh_alh[~np.isnan(blh_alh)] = alh1[~np.isnan(blh_alh)] - ds_blh[~np.isnan(blh_alh)]
        dist = blh_alh.copy()
        
        # Create flag_alh based on conditions
        flag_alh = np.full_like(blh_alh, False, dtype=bool)
        flag_alh[(~np.isnan(blh_alh)) & (alh1 > ds_blh)] = True
        
        # Initialize blh_cth with cth1 and handle None values
        blh_cth = cth1.copy()
        blh_cth[cth1 == None] = None
        
        # Create flag_cth based on conditions
        flag_cth = np.full_like(blh_cth, False, dtype=bool)
        flag_cth[(blh_cth != None) & (cth1 > ds_blh)] = True
        
        # Create other flags
        flag_cf = CF > 0.01
        flag_abs = uvai > 0
        
        # Print results
        nan_uvai = np.isnan(uvai)
        
        
        # Nettoyage des NaN et conversion explicite en booléens si nécessaire
        flag_cf = np.where(np.isnan(CF), False, CF > 0.01) 
        flag_abs = flag_abs.astype(bool)  
        flag_cth = flag_cth.astype(bool) 
        flag_alh = flag_alh.astype(bool)  
        
        # Calcul des drapeaux pour les différentes catégories
        flag_acc_abl = flag_cf & np.logical_not(flag_cth) & flag_alh
        flag_acc_bbl = flag_cf & np.logical_not(flag_cth) & np.logical_not(flag_alh)
        
        flag_clr_abl = np.logical_not(flag_cf) & flag_alh
        flag_clr_bbl = np.logical_not(flag_cf) & np.logical_not(flag_alh)
        
        # Sous-catégories avec flag_abs
        flag_acc_abl_abs = flag_acc_abl & flag_abs
        flag_acc_bbl_abs = flag_acc_bbl & flag_abs
        
        flag_clr_abl_abs = flag_clr_abl & flag_abs
        flag_clr_bbl_abs = flag_clr_bbl & flag_abs
        
        # Sous-catégories sans flag_abs
        flag_acc_abl_dif = flag_acc_abl & np.logical_not(flag_abs)
        flag_acc_bbl_dif = flag_acc_bbl & np.logical_not(flag_abs)
        
        flag_clr_abl_dif = flag_clr_abl & np.logical_not(flag_abs)
        flag_clr_bbl_dif = flag_clr_bbl & np.logical_not(flag_abs)

       
        # Aerosol fraction
        af_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_clr_abl_abs, flag_clr_bbl_abs, 
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        af_variables = ['af_aac_abl_abs', 'af_aac_bbl_abs', 'af_clr_abl_abs', 'af_clr_bbl_abs',
                        'af_aac_abl_dif', 'af_aac_bbl_dif', 'af_clr_abl_dif', 'af_clr_bbl_dif']

        af_results = {var: da.where(flag, 1, 0) for var, flag in zip(af_variables, af_flags)}
        af_results = {var: da.where(nan_uvai, np.nan, data).astype(np.float32) for var, data in af_results.items()}

        # AI
        ai_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_clr_abl_abs, flag_clr_bbl_abs,
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        ai_variables = ['ai_aac_abl_abs', 'ai_aac_bbl_abs', 'ai_clr_abl_abs', 'ai_clr_bbl_abs',
                        'ai_aac_abl_dif', 'ai_aac_bbl_dif', 'ai_clr_abl_dif', 'ai_clr_bbl_dif']

        ai_results = {var: da.where(flag & (~nan_uvai), uvai, np.nan).astype(np.float32) for var, flag in zip(ai_variables, ai_flags)}

        # ALH - BLH = dist
        dist_flags = ai_flags  # Same flags as for AI
        dist_variables = ['dist_aac_abl_abs', 'dist_aac_bbl_abs', 'dist_clr_abl_abs', 'dist_clr_bbl_abs',
                          'dist_aac_abl_dif', 'dist_aac_bbl_dif', 'dist_clr_abl_dif', 'dist_clr_bbl_dif']

        dist_results = {var: da.where(flag & (~nan_uvai), dist, np.nan).astype(np.float32) for var, flag in zip(dist_variables, dist_flags)}
        
        variables = {
            'blh': ds_blh,
            'alh': alh1,
            'cth': cth1,
            'cf': CF,
            'ai': uvai,
            'ctt': CTT,
            'cot': COT,
            'cer': CER,
            'Nd' : Nd,
            'LWP': LWP,
            'N_dus': N_dus,
            'N_dum': N_dum,
            'N_dul': N_dul,
            'N_bchphil': N_bchphil,
            'N_bchphob': N_bchphob
        }
        variables.update(af_results)
        variables.update(ai_results)
        variables.update(dist_results)
        
        # Chargement et interpolation
        print("Chargement et interpolation du LSM...")
        lsm_interpolated = load_and_interpolate_lsm(lsm_file)
        print("Interpolation terminée.")
        
        # Recalage des longitudes
        print("Recalage des longitudes...")
        lsm_shifted = shift_longitude_and_data(lsm_interpolated)
        print("Recalage terminé.")
        
        # Création du dataset
        print("Création du Dataset...")
        df_lsm = create_lsm_dataset(lsm_shifted)
        print("Dataset créé.")
        
        # Filtrage des données
        print("Filtrage des données...")
        df_lsm_filtered = filter_data(df_lsm)
        print("Filtrage terminé.")

        print('coucou6')
        # Add the zones to your variables dictionary
        variables['Zone'] = df_lsm_filtered['Zone'].values

        name = list(variables.keys())
        print(name)
        longnames = {'blh': 'Boundary Layer Height',
                     'alh': 'Areosol Layer Height',
                     'cth': 'Cloud Top Height',
                     'cf':  'Cloud Fraction',
                     'ai':  'Aerosol Index',
                     'ctt': 'Cloud Top Temperature',
                     'cot': 'Cloud Optical Thickness at 3.7um',
                     'cer': 'Cloud Effective Radius at 3.7um',
                     'Nd' : 'Cloud top droplet number concentration at 3.7um',
                     'LWP': 'Cloud Liquid Water Path at 3.7um',
                     'N_dus': 'Small Dust Number Concentration (0.03-0.55um)',
                     'N_dum': 'Medium Dust Number Concentration (0.55-0.9um)',
                     'N_dul': 'Large Dust Number Concentration (0.9-20um)',
                     'N_bchphil': 'Hydrophilic Black Carbon Number Concentration',
                     'N_bchphob': 'Hydrophobic Black Carbon Number Concentration',
                     'af_aac_abl_abs': 'Aerosol Fraction for Absorbing Aerosols Above Clouds and Above Boundary Layer',
                     'af_aac_bbl_abs': 'Aerosol Fraction for Absorbing Aerosols Above Clouds and Below Boundary Layer',
                     'af_clr_abl_abs': 'Aerosol Fraction for Absorbing Aerosols in Clear Sky and Above Boundary Layer',
                     'af_clr_bbl_abs': 'Aerosol Fraction for Absorbing Aerosols in Clear Sky and Below Boundary Layer',
                     'af_aac_abl_dif': 'Aerosol Fraction for Diffusing Aerosols Above Clouds and Above Boundary Layer',
                     'af_aac_bbl_dif': 'Aerosol Fraction for Diffusing Aerosols Above Clouds and Below Boundary Layer',
                     'af_clr_abl_dif': 'Aerosol Fraction for Diffusing Aerosols in Clear Sky and Above Boundary Layer',
                     'af_clr_bbl_dif': 'Aerosol Fraction for Diffusing Aerosols in Clear Sky and Below Boundary Layer',
                     'ai_aac_abl_abs': 'Aerosol Index for Absorbing Aerosols Above Clouds and Above Boundary Layer',
                     'ai_aac_bbl_abs': 'Aerosol Index for Absorbing Aerosols Above Clouds and Below Boundary Layer',
                     'ai_clr_abl_abs': 'Aerosol Index for Absorbing Aerosols in Clear Sky and Above Boundary Layer',
                     'ai_clr_bbl_abs': 'Aerosol Index for Absorbing Aerosols in Clear Sky and Below Boundary Layer',
                     'ai_aac_abl_dif': 'Aerosol Index for Diffusing Aerosols Above Clouds and Above Boundary Layer',
                     'ai_aac_bbl_dif': 'Aerosol Index for Diffusing Aerosols Above Clouds and Below Boundary Layer',
                     'ai_clr_abl_dif': 'Aerosol Index for Diffusing Aerosols in Clear Sky and Above Boundary Layer',
                     'ai_clr_bbl_dif': 'Aerosol Index for Diffusing Aerosols in Clear Sky and Below Boundary Layer',
                     'dist_aac_abl_abs': 'Aerosol-BL distance for Absorbing Aerosols Above Clouds and Above Boundary Layer',
                     'dist_aac_bbl_abs': 'Aerosol-BL distance for Absorbing Aerosols Above Clouds and Below Boundary Layer',
                     'dist_clr_abl_abs': 'Aerosol-BL distance for Absorbing Aerosols in Clear Sky and Above Boundary Layer',
                     'dist_clr_bbl_abs': 'Aerosol-BL distance for Absorbing Aerosols in Clear Sky and Below Boundary Layer',
                     'dist_aac_abl_dif': 'Aerosol-BL distance for Diffusing Aerosols Above Clouds and Above Boundary Layer',
                     'dist_aac_bbl_dif': 'Aerosol-BL distance for Diffusing Aerosols Above Clouds and Below Boundary Layer',
                     'dist_clr_abl_dif': 'Aerosol-BL distance for Diffusing Aerosols in Clear Sky and Above Boundary Layer',
                     'dist_clr_bbl_dif': 'Aerosol-BL distance for Diffusing Aerosols in Clear Sky and Below Boundary Layer',
                     'Zone': 'Zone of a specific oceanic or land region'
                     
                     }
        units = {'blh': 'm',
                 'alh': 'm',
                 'cth': 'm',
                 'cf':  '1',
                 'ai':  '1',
                 'ctt': 'K',
                 'cot': '1',
                 'cer': 'um',
                 'Nd': 'cm-3',
                 'LWP': 'g.cm-2',
                 'N_dus': 'm-2',
                 'N_dum': 'm-2',
                 'N_dul': 'm-2',
                 'N_bchphil': 'm-2',
                 'N_bchphob': 'm-2',
                 'af_aac_abl_abs': '1',
                 'af_aac_bbl_abs': '1',
                 'af_clr_abl_abs': '1',
                 'af_clr_bbl_abs': '1',
                 'af_aac_abl_dif': '1',
                 'af_aac_bbl_dif': '1',
                 'af_clr_abl_dif': '1',
                 'af_clr_bbl_dif': '1',
                 'ai_aac_abl_abs': '1',
                 'ai_aac_bbl_abs': '1',
                 'ai_clr_abl_abs': '1',
                 'ai_clr_bbl_abs': '1',
                 'ai_aac_abl_dif': '1',
                 'ai_aac_bbl_dif': '1',
                 'ai_clr_abl_dif': '1',
                 'ai_clr_bbl_dif': '1',
                 'dist_aac_abl_abs': 'm',
                 'dist_aac_bbl_abs': 'm',
                 'dist_clr_abl_abs': 'm',
                 'dist_clr_bbl_abs': 'm',
                 'dist_aac_abl_dif': 'm',
                 'dist_aac_bbl_dif': 'm',
                 'dist_clr_abl_dif': 'm',
                 'dist_clr_bbl_dif': 'm',
                 'Zone': 'index'    
                 }
        
        # Créer un nouveau dataset pour stocker les variables resamplées
        new_rows = 8192
        new_cols = 16384

        # Création d'un nouveau dataset NetCDF avec les dimensions cibles
        # Define the target file path
        output_path = f'/LARGE14/devigne/Données_TROPOMI_MODIS/2020/resampled_data_tropomodis_{date}.nc'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create the new NetCDF file
        with Dataset(output_path, 'w', format='NETCDF4') as nc:
            
            # Define dimensions
            nc.createDimension('latitude', new_rows)
            nc.createDimension('longitude', new_cols)
        
            # Define coordinate variables
            latitudes = nc.createVariable('latitude', np.float32, ('latitude',))
            longitudes = nc.createVariable('longitude', np.float32, ('longitude',))
        
            # Assign values to coordinate variables
            latitudes[:] = np.linspace(-90, 90, new_rows)
            longitudes[:] = np.linspace(-180, 180, new_cols)
        
            # Assign attributes to coordinate variables
            latitudes.units = "degrees_north"
            latitudes.long_name = "Latitude"
            longitudes.units = "degrees_east"
            longitudes.long_name = "Longitude"
        
            # Add data variables
            for var_name, data in variables.items():
                
                # Define variable
                var = nc.createVariable(
                    var_name, np.float32, ('latitude', 'longitude'), fill_value=-9999, zlib=True, complevel=8
                )
                
                # Assign data
                var[:, :] = np.nan_to_num(data, True, -9999).astype(np.float32)
        
                # Assign attributes
                #var._FillValue = -9999
                var.units = f"{units[var_name]}"  # Update with appropriate units if available
                var.long_name = f"{longnames[var_name]}"  # Update with a descriptive name if available
                var.title = 'Aerosols-Clouds properties calculated using different Aerosol-Cloud scenarios'
                var.institution = 'Atmospherical Optics Laboratory, Lille, France'
                var.source = 'MODIS Collection 6/6.1 06L2, 03L2 and daily products, TROPOMI L3 daily products, CAMS ERA5 and EAC4 reanalysis data'
                var.contact = "Elise Devigne (elise.devigne@univ-lille.fr)"
                gc.enable()
        print(f"NetCDF file successfully created at {output_path}")

def load_and_interpolate_lsm(file, lat_size=8192, lon_size=16384):
    """
    Charge un fichier LSM et l'interpole sur une grille régulière.
    """
    try:
        lsm = xr.open_dataset(file, engine='netcdf4')
    except ValueError:
        lsm = xr.open_dataset(file, engine='scipy')

    # Convertir en tableau si nécessaire
    lsm = lsm.to_array().squeeze()

    # Si "time" est une dimension, prendre la moyenne
    if 'time' in lsm.dims:
        lsm = lsm.mean(dim='time')

    # Définir les nouvelles coordonnées de grille
    new_lat = np.linspace(-90, 90, lat_size)
    new_lon = np.linspace(-180, 180, lon_size)  # Conversion explicite sur [-180, 180]

    # Interpolation sur la nouvelle grille
    try:
        lsm_interpolated = lsm.interp(latitude=new_lat, longitude=new_lon, method='nearest')
    except ValueError as e:
        print(f"Erreur lors de l'interpolation : {e}")
        raise

    return xr.Dataset({'LSM': lsm_interpolated}, coords={'latitude': new_lat, 'longitude': new_lon})


def shift_longitude_and_data(dataset):
    """
    Recalage des longitudes de [0, 360) à [-180, 180] et ajustement des données.
    """
    dataarray = dataset.to_array().squeeze()
    lon = dataarray['longitude'].values
    lat = dataarray['latitude'].values

    # Vérification de la plage des longitudes
    if lon.max() > 180:
        # Conversion des longitudes sur [-180, 180)
        lon = np.where(lon > 180, lon - 360, lon)
        sorted_indices = np.argsort(lon)
        lon = lon[sorted_indices]

        # Réordonner les données pour correspondre aux nouvelles longitudes
        new_data = dataarray.values
        reordered_data = new_data[:, sorted_indices]  # Ajustement explicite sur les dimensions

        # Création d'un nouveau DataArray avec les coordonnées corrigées
        return xr.DataArray(reordered_data, coords=[lat, lon], dims=['latitude', 'longitude'])
    else:
        print("Les longitudes sont déjà dans la plage [-180, 180].")
        return dataarray




def create_lsm_dataset(lsm_shifted):
    """
    Crée un Dataset xarray à partir d'un LSM recalé.
    """
    return xr.Dataset({
        'LSM': (['latitude', 'longitude'], lsm_shifted.values),
    },
    coords={
        'latitude': lsm_shifted['latitude'].values,
        'longitude': lsm_shifted['longitude'].values
    })

regions = {
    'Peruvian': {'lat_min': -30, 'lat_max': 0, 'lon_min': -115, 'lon_max': -65, 'type': 'Ocean', 'id': 1},
    'Namibian': {'lat_min': -30, 'lat_max': 0, 'lon_min': -20, 'lon_max': 20, 'type': 'Ocean', 'id': 2},
    'Australian': {'lat_min': -35, 'lat_max': -15, 'lon_min': 55, 'lon_max': 120, 'type': 'Ocean', 'id': 3},
    'Californian': {'lat_min': 10, 'lat_max': 40, 'lon_min': -150, 'lon_max': -110, 'type': 'Ocean', 'id': 4},
    'Canarian': {'lat_min': 10, 'lat_max': 40, 'lon_min': -40, 'lon_max': -5, 'type': 'Ocean', 'id': 5},
    'China': {'lat_min': 10, 'lat_max': 40, 'lon_min': 100, 'lon_max': 160, 'type': 'Ocean', 'id': 6},
    'North Atlantic': {'lat_min': 40, 'lat_max': 70, 'lon_min': -60, 'lon_max': 0, 'type': 'Ocean', 'id': 7},
    'Northeast Pacific': {'lat_min': 40, 'lat_max': 70, 'lon_min': -180, 'lon_max': -120, 'type': 'Ocean', 'id': 8},
    'Northwest Pacific': {'lat_min': 40, 'lat_max': 70, 'lon_min': 120, 'lon_max': 180, 'type': 'Ocean', 'id': 9},
    'Southeast Pacific': {'lat_min': -70, 'lat_max': -30, 'lon_min': -180, 'lon_max': -70, 'type': 'Ocean', 'id': 10},
    'South Atlantic': {'lat_min': -70, 'lat_max': -30, 'lon_min': -70, 'lon_max': 60, 'type': 'Ocean', 'id': 11},
    'South Indian Ocean': {'lat_min': -70, 'lat_max': -35, 'lon_min': 60, 'lon_max': 180, 'type': 'Ocean', 'id': 12},
    'Galapagos': {'lat_min': 0, 'lat_max': 10, 'lon_min': -120, 'lon_max': -70, 'type': 'Ocean', 'id': 13},
    'Chinese Stratus': {'lat_min': 10, 'lat_max': 40, 'lon_min': 100, 'lon_max': 130, 'type': 'Land', 'id': 14},
    'Amazon': {'lat_min': -15, 'lat_max': 10, 'lon_min': -80, 'lon_max': -30, 'type': 'Land', 'id': 15},
    'Equatorial Africa': {'lat_min': -15, 'lat_max': 15, 'lon_min': -20, 'lon_max': 20, 'type': 'Land', 'id': 16},
    'North America': {'lat_min': 30, 'lat_max': 45, 'lon_min': -100, 'lon_max': -75, 'type': 'Land', 'id': 17},
    'India': {'lat_min': 10, 'lat_max': 30, 'lon_min': 65, 'lon_max': 90, 'type': 'Land', 'id': 18},
    'Europe': {'lat_min': 25, 'lat_max': 45, 'lon_min': 0, 'lon_max': 50, 'type': 'Land', 'id': 19},    
}

def filter_data(df):
    df['Zone'] = np.nan  # Initialise la variable Zone avec des NaN

    for region_name, region in regions.items():
        # Créez un masque pour la région
        mask = (
            (df['latitude'] >= region['lat_min']) & (df['latitude'] <= region['lat_max']) &
            (df['longitude'] >= region['lon_min']) & (df['longitude'] <= region['lon_max'])
        )
        print(f"Traitement de la région : {region_name}")
        print("Pixels dans la région avant filtrage :", mask.sum())

        # Appliquez le masque pour assigner la zone
        df['Zone'] = df['Zone'].where(~mask, region['id'])

        # Conditions supplémentaires pour les types de région
        if region['type'] == 'Ocean':
            ocean_mask = mask & (df['LSM'] >= 0.4)
            df['Zone'] = df['Zone'].where(~ocean_mask, np.nan)
        elif region['type'] == 'Land':
            land_mask = mask & (df['LSM'] < 0.4)
            df['Zone'] = df['Zone'].where(~land_mask, np.nan)

    return df




lsm_file = '/home/devigne/land_sea_mask'
def process_files(ai_file, alh_file, cloud_file, cf_file, modis_file, eac4_file):
    try:
        # Votre fonction de traitement ici
        get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file, eac4_file, lsm_file)
    except Exception as e:
        print(f"Erreur lors du traitement des fichiers {ai_file}, {alh_file}, {cloud_file}, {cf_file}, {modis_file}: {e}")


    
    
if __name__=="__main__":
    # Récupération des arguments
    args = parse_args()
    
    # Répertoires définis par les arguments en ligne de commande
    path = args.ai_dir
    path_2 = args.lh_dir
    path_3 = args.cloud_dir
    path_4 = args.cf_dir
    path_modis = args.modis_dir
    path_eac4 = args.eac4_dir
    
    list_ai = get_file_list(path)
    list_lh = get_file_list(path_2)
    list_cloud = get_file_list(path_3)
    list_cf = get_file_list(path_4)
    list_modis = get_file_list(path_modis)
    list_eac4 = get_file_list(path_eac4)
    
    
    
    # Extraire les dates des fichiers
    dates1 = extract_dates(list_ai, 47, 55)
    dates2 = extract_dates(list_lh, 48, 56)
    dates3 = extract_dates(list_cloud, 61, 69)
    dates_cf = extract_dates(list_cf, 58, 66)
    date_modis = extract_dates_2(list_modis, 47, 57)
    date_eac4 = extract_dates_3(list_eac4, 52, 62)
    
    common_date = find_common_dates(dates1, dates2, dates3, dates_cf, date_modis, date_eac4)
    comparison_date = datetime(2019, 12, 31)
    end_date = datetime(2020, 2, 1)
    filtered_dates = {date for date in common_date if comparison_date < date < end_date}

    
    # Filtrer les fichiers selon les dates filtrées
    ai_files = filter_files(list_ai, filtered_dates, 47, 55)
    alh_files = filter_files(list_lh, filtered_dates, 48, 56)
    cloud_files = filter_files(list_cloud, filtered_dates, 61, 69)
    cf_files = filter_files(list_cf, filtered_dates, 58, 66)
    modis_files = filter_files_2(list_modis, filtered_dates, 47, 57)
    eac4_files = filter_files_3(list_eac4, filtered_dates, 52, 62)
    # Créez un pool de processus pour paralléliser
    with Pool(4) as pool:
        pool.starmap(process_files, zip(ai_files, alh_files, cloud_files, cf_files, modis_files, eac4_files))

