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


# Fonction pour récupérer les arguments en ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Traitement des fichiers MODIS")
    parser.add_argument('--ai-dir', type=str, required=True, help="Répertoire des fichiers AI")
    parser.add_argument('--lh-dir', type=str, required=True, help="Répertoire des fichiers ALH")
    parser.add_argument('--cloud-dir', type=str, required=True, help="Répertoire des fichiers Cloud")
    parser.add_argument('--cf-dir', type=str, required=True, help="Répertoire des fichiers Cloud Fraction")
    parser.add_argument('--modis-dir', type=str, required=True, help="Répertoire des fichiers MODIS")
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

# Fonction pour effectuer le traitement sur les fichiers
def get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file):
    # Définir la date du fichier
    d = str(ai_file)[47:55]
    date_obj = datetime.strptime(d, '%Y%m%d')
    date = date_obj.strftime('%Y-%m-%d')
    ds_blh = blh('blh', date)
    chunk = 10
    with xr.open_dataset(ai_file) as original_ds, \
         xr.open_dataset(alh_file) as alh_ds, \
         xr.open_dataset(cloud_file) as cloud_ds, \
         xr.open_dataset(cf_file) as cf_ds, \
         xr.open_dataset(modis_file) as modis_ds:
        print("coucou")
        # Variables
        alh1 = alh_ds['aerosol_height'].isel(time=0).fillna(0).astype(np.float32).values * 1000
        cth1 = cloud_ds['cloud_top_height'].isel(time=0).fillna(0).astype(np.float32).values
        uvai = original_ds['absorbing_aerosol_index'].isel(time=0).fillna(0).astype(np.float32).values
        CF = cf_ds['cloud_fraction'].isel(time=0).fillna(0).astype(np.float32).values
        COT  = modis_ds['Cloud_Optical_Thickness_37'].compute()
        CER  = modis_ds['Cloud_Effective_Radius_37'].compute()
        CTT  = modis_ds['cloud_top_temperature_1km'].compute()
        print(f"Dimensions de alh1: {alh1.shape}, min: {np.min(alh1)}, max: {np.max(alh1)}")
        print(f"Dimensions de cth1: {cth1.shape}, min: {np.min(cth1)}, max: {np.max(cth1)}")
        print(f"Dimensions de uvai: {uvai.shape}, min: {np.min(uvai)}, max: {np.max(uvai)}")
        print(f"Dimensions de CF: {CF.shape}, min: {np.min(CF)}, max: {np.max(CF)}")
        print(f"Dimensions de COT: {COT.shape}, min: {np.min(COT)}, max: {np.max(COT)}")
        print(f"Dimensions de CER: {CER.shape}, min: {np.min(CER)}, max: {np.max(CER)}")
        print(f"Dimensions de CTT: {CTT.shape}, min: {np.min(CTT)}, max: {np.max(CTT)}")

        blh_alh = alh1
        blh_alh[alh1==np.nan] = np.nan
        blh_alh[ds_blh==np.nan] = np.nan
        print('coucou2')
        print(blh_alh.shape, ds_blh.shape)
        
        blh_alh[blh_alh != np.nan] = alh1[blh_alh != np.nan] - ds_blh[blh_alh != np.nan]
        dist = blh_alh
        print('coucou3')
        blh_alh[(blh_alh != np.nan) & (alh1 > ds_blh)] = True
        blh_alh[(blh_alh != np.nan) & (blh_alh != True)] = False
        flag_alh = blh_alh
       
        blh_cth = cth1
        blh_cth[cth1==None] = None
        blh_cth[(blh_cth != None) & (cth1 > ds_blh)] = True
        blh_cth[(blh_cth != None) & (blh_cth != True)] = False
        flag_cth = blh_cth
        
        flag_cf = CF > 0.01
        flag_abs = uvai > 0

        nan_uvai = np.isnan(uvai)
        
        print('coucou4')
        # Nettoyage des NaN et conversion explicite en booléens si nécessaire
        flag_cf = np.where(np.isnan(CF), False, CF > 0.01)  # Vérifie CF et remplace les NaN par False
        flag_abs = flag_abs.astype(bool)  # Assure que flag_abs est de type bool
        flag_cth = flag_cth.astype(bool)  # Assure que flag_cth est de type bool
        flag_alh = flag_alh.astype(bool)  # Assure que flag_alh est de type bool
        
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

        print('coucou6')
        # Aerosol fraction
        af_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_clr_abl_abs, flag_clr_bbl_abs, 
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        af_variables = ['af_aac_abl_abs', 'af_aac_bbl_abs', 'af_clr_abl_abs', 'af_clr_bbl_abs',
                        'af_aac_abl_dif', 'af_aac_bbl_dif', 'af_clr_abl_dif', 'af_clr_bbl_dif']

        af_results = {var: da.where(flag, 1, 0) for var, flag in zip(af_variables, af_flags)}
        af_results = {var: da.where(nan_uvai, np.nan, data) for var, data in af_results.items()}

        # AI
        ai_flags = [flag_acc_abl_abs, flag_acc_bbl_abs, flag_clr_abl_abs, flag_clr_bbl_abs,
                    flag_acc_abl_dif, flag_acc_bbl_dif, flag_clr_abl_dif, flag_clr_bbl_dif]
        ai_variables = ['ai_aac_abl_abs', 'ai_aac_bbl_abs', 'ai_clr_abl_abs', 'ai_clr_bbl_abs',
                        'ai_aac_abl_dif', 'ai_aac_bbl_dif', 'ai_clr_abl_dif', 'ai_clr_bbl_dif']

        ai_results = {var: da.where(flag & (~nan_uvai), uvai, np.nan) for var, flag in zip(ai_variables, ai_flags)}

        # ALH - BLH = dist
        dist_flags = ai_flags  # Same flags as for AI
        dist_variables = ['dist_aac_abl_abs', 'dist_aac_bbl_abs', 'dist_clr_abl_abs', 'dist_clr_bbl_abs',
                          'dist_aac_abl_dif', 'dist_aac_bbl_dif', 'dist_clr_abl_dif', 'dist_clr_bbl_dif']

        dist_results = {var: da.where(flag & (~nan_uvai), dist, np.nan) for var, flag in zip(dist_variables, dist_flags)}

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
        
        # Créer un nouveau dataset pour stocker les variables resamplées
        new_rows = 8192
        new_cols = 16384

        # Création d'un nouveau dataset NetCDF avec les dimensions cibles
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', message='Mean of empty slice')
            resampled_ds = xr.Dataset(coords={
                'latitude': da.linspace(-90, 90, new_rows),
                'longitude': da.linspace(-180, 180, new_cols)
            })
            # Add data to the new dataset
            for var_name, data in variables.items():
                resampled_ds[var_name] = xr.DataArray(data, coords={'latitude':da.linspace(-90, 90, new_rows) , 'longitude':da.linspace(-180, 180, new_cols) }, dims=['latitude', 'longitude'])
                resampled_ds[var_name] = resampled_ds[var_name].fillna(-9999)
                resampled_ds[var_name].attrs["_FillValue"] = -9999
        # Calculer et sauvegarder le résultat
        resampled_ds = resampled_ds.compute()
        compression = {"zlib": True, "complevel": 4}
        for var_name in resampled_ds.data_vars:
            resampled_ds[var_name].encoding.update(compression)
        resampled_ds.to_netcdf(f'/LARGE14/devigne/Données_TROPOMI_MODIS/2019/resampled_data_tropomodis_{date}.nc')


def process_files(ai_file, alh_file, cloud_file, cf_file, modis_file):
    try:
        # Votre fonction de traitement ici
        get_nc(ai_file, alh_file, cloud_file, cf_file, modis_file)
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
    
    list_ai = get_file_list(path)
    list_lh = get_file_list(path_2)
    list_cloud = get_file_list(path_3)
    list_cf = get_file_list(path_4)
    list_modis = get_file_list(path_modis)
    
    # Extraire les dates des fichiers
    dates1 = extract_dates(list_ai, 47, 55)
    dates2 = extract_dates(list_lh, 48, 56)
    dates3 = extract_dates(list_cloud, 61, 69)
    dates_cf = extract_dates(list_cf, 58, 66)
    date_modis = extract_dates_2(list_modis, 47, 57)
    print(dates_cf, date_modis)
    common_date = find_common_dates(dates1, dates2, dates3, dates_cf, date_modis)
    comparison_date = datetime(2019, 11, 28)
    end_date = datetime(2019, 11, 30)
    filtered_dates = {date for date in common_date if comparison_date < date < end_date}

    print(len(filtered_dates))
    # Filtrer les fichiers selon les dates filtrées
    ai_files = filter_files(list_ai, filtered_dates, 47, 55)
    alh_files = filter_files(list_lh, filtered_dates, 48, 56)
    cloud_files = filter_files(list_cloud, filtered_dates, 61, 69)
    cf_files = filter_files(list_cf, filtered_dates, 58, 66)
    modis_files = filter_files_2(list_modis, filtered_dates, 47, 57)
    
    # Créez un pool de processus pour paralléliser
    with Pool(4) as pool:
        pool.starmap(process_files, zip(ai_files, alh_files, cloud_files, cf_files, modis_files))

