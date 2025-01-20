#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:07:43 2024

@author: devigne
"""

import argparse

# Fonction pour parser les arguments de la ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Traitement des fichiers MODIS")
    parser.add_argument('--modis-dir', type=str, required=True, help="Répertoire des fichiers MODIS")
    parser.add_argument('--geo-dir', type=str, required=True, help="Répertoire des fichiers géographiques MODIS")
    parser.add_argument('--output-file', type=str, required=True, help="Fichier de sortie NetCDF")
    return parser.parse_args()

# Parser les arguments de la ligne de commande
args = parse_args()

# Assigner les valeurs des arguments à des variables
modis_dir = args.modis_dir
geo_dir = args.geo_dir
output_file = args.output_file

# Le reste de votre code suit ici sans modification majeure
import os
import numpy as np
import gc
from pyhdf.SD import SD, SDC
from pyresample import kd_tree, geometry
from collections import defaultdict
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor
from numba import jit
import logging
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Bandes spectrales
bands = ['', '_16', '_37']

# Variables et constantes
var_list_modis = [
    'Cloud_Phase_Optical_Properties', 'cloud_top_temperature_1km',
    'cloud_top_height_1km', 'Cloud_Optical_Thickness_37',
    'Cloud_Effective_Radius_37', 'Cloud_Optical_Thickness',
    'Cloud_Effective_Radius', 'Cloud_Optical_Thickness_16',
    'Cloud_Effective_Radius_16'
]
list_geo = ['Latitude', 'Longitude', 'SolarZenith', 'SensorZenith']
resolution = (180 / 8192, 180 / 8192)

# JIT avec numba pour un filtrage rapide
@jit(nopython=True, parallel=True)
def apply_filters_exclusion_fast(lat, lon, solar_zenith, sensor_zenith, cloud_temp, cloud_phase, cot, cer):
    combined_mask = (solar_zenith < 65) & (sensor_zenith < 55) & (cloud_temp > 268)
    combined_mask &= (cloud_phase == 2) | (cloud_phase == 4)
    combined_mask &= cot >= 4
    combined_mask &= cer >= 4
    return combined_mask

def load_hdf(filename, var_list):
    try:
        hdf = SD(filename, SDC.READ)
    except Exception as e:
        logger.error(f"Erreur de lecture du fichier {filename}: {e}")
        return {}

    data_dict = {}
    for var in var_list:
        try:
            sds_obj = hdf.select(var)
            data = sds_obj.get().astype(np.float32)
            attributes = sds_obj.attributes()
            fill_value = attributes.get('_FillValue', None)
            if fill_value is not None:
                data[data == fill_value] = np.nan
            if 'scale_factor' in attributes and 'add_offset' in attributes:
                data = (data - attributes['add_offset']) * attributes['scale_factor']
            elif 'scale_factor' in attributes:
                data *= attributes['scale_factor']
            elif 'add_offset' in attributes:
                data -= attributes['add_offset']
            data_dict[var] = data
        except Exception as e:
            logger.warning(f"Erreur de chargement {var} dans {filename}: {e}")
            data_dict[var] = None
    return data_dict

def resample_data(lat, lon, data, resolution=(180 / 8192, 180 / 8192)):
    lat_res, lon_res = resolution
    lat_bins = np.arange(-90, 90 + lat_res, lat_res)
    lon_bins = np.arange(-180, 180 + lon_res, lon_res)
    lat_center = (lat_bins[:-1] + lat_bins[1:]) / 2
    lon_center = (lon_bins[:-1] + lon_bins[1:]) / 2
    lon_grid, lat_grid = np.meshgrid(lon_center, lat_center)

    source_geo = geometry.SwathDefinition(lons=lon, lats=lat)
    target_geo = geometry.GridDefinition(lons=lon_grid, lats=lat_grid)

    resampled_data = kd_tree.resample_nearest(
        source_geo, data, target_geo, radius_of_influence=20000, fill_value=np.nan
    )
    return lat_center, lon_center, resampled_data

def process_file_pair(file_pair):
    modis_path, geo_path = file_pair
    logger.debug(f"Traitement des fichiers : {modis_path} et {geo_path}")

    var_cloud = load_hdf(modis_path, var_list_modis)
    var_geo = load_hdf(geo_path, list_geo)

    if not var_cloud or not var_geo:
        return None

    try:
        lat = var_geo['Latitude']
        lon = var_geo['Longitude']
        if lat is None or lon is None or lat.shape != lon.shape:
            logger.error("Données géographiques manquantes ou invalides. Ignorer la paire.")
            return None

        combined_mask = apply_filters_exclusion_fast(
            lat, lon,
            var_geo['SolarZenith'], var_geo['SensorZenith'],
            var_cloud['cloud_top_temperature_1km'],
            var_cloud['Cloud_Phase_Optical_Properties'],
            var_cloud['Cloud_Optical_Thickness'], 
            var_cloud['Cloud_Effective_Radius']
        )
        if np.sum(combined_mask) == 0:
            return None
    except KeyError as e:
        logger.error(f"Clé manquante {e}. Ignorer la paire.")
        return None

    results = {}
    valid_idx = np.where(combined_mask)  # Pré-calculate valid indices
    for var, data in {**var_cloud, **var_geo}.items():
        if data is not None and data.shape == lat.shape:
            try:
                masked_data = data[valid_idx]
                lat_masked, lon_masked = lat[valid_idx], lon[valid_idx]
                _, _, resampled_data = resample_data(lat_masked, lon_masked, masked_data)
                results[var] = resampled_data
            except Exception as e:
                logger.warning(f"Erreur de traitement pour la variable {var}: {e}")
    
    del var_cloud, var_geo, combined_mask  # Libère la mémoire immédiatement après utilisation
    gc.collect()
    return results

def save_to_netcdf(file_pairs, output_file, lat_center, lon_center):
    global_data = np.zeros((len(lat_center), len(lon_center), len(var_list_modis) + len(list_geo)), dtype=np.float16)
    global_count = np.zeros((len(lat_center), len(lon_center), len(var_list_modis) + len(list_geo)), dtype=np.int16)
    
    var_names = list(var_list_modis) + list(list_geo)

    with ProcessPoolExecutor(max_workers=4) as executor:
        for result in tqdm(executor.map(process_file_pair, file_pairs), total=len(file_pairs)):
            if result:
                for i, (var, resampled_data) in enumerate(result.items()):
                    valid_mask = np.isfinite(resampled_data)
                    global_data[..., i][valid_mask] += resampled_data[valid_mask]
                    global_count[..., i][valid_mask] += 1

    with Dataset(output_file, 'w', format='NETCDF4') as nc:
        nc.createDimension('latitude', len(lat_center))
        nc.createDimension('longitude', len(lon_center))

        latitudes = nc.createVariable('latitude', 'f4', ('latitude',))
        latitudes[:] = lat_center.astype(np.float32)  # NetCDF préfère float32
    
        longitudes = nc.createVariable('longitude', 'f4', ('longitude',))
        longitudes[:] = lon_center.astype(np.float32)
    
        for i, var in enumerate(var_names):
            avg_data = np.zeros_like(global_data[..., i], dtype=np.float16)
            valid = global_count[..., i] > 0
            avg_data[valid] = global_data[..., i][valid] / global_count[..., i][valid]
            avg_data[~valid] = np.nan
    
            data_var = nc.createVariable(var, 'f4', ('latitude', 'longitude'), zlib=True, complevel=4)
            data_var[:] = avg_data.astype(np.float32)  # Convertir en float32 pour NetCDF

        for band in bands:
            if f'Cloud_Optical_Thickness{band}' in nc.variables and f'Cloud_Effective_Radius{band}' in nc.variables:
                cot = nc.variables[f'Cloud_Optical_Thickness{band}'][:]
                cer = nc.variables[f'Cloud_Effective_Radius{band}'][:]
        
                # Filtrer les valeurs NaN pour éviter les erreurs
                valid_mask = np.isfinite(cot) & np.isfinite(cer)
                nd = np.full_like(cot, np.nan)
                nd[valid_mask] = 1.37e-11 * (cot[valid_mask]**0.5) * ((1e-6 * cer[valid_mask])**-2.5)
        
                # Créer la variable NetCDF pour Nd_band
                nd_var = nc.createVariable(f'Nd{band}', 'f4', ('latitude', 'longitude'), zlib=True, complevel=4)
                nd_var[:] = nd

# Utilisation des chemins passés par les arguments
modis_files = sorted(f for f in os.listdir(modis_dir) if f.endswith('.hdf'))
geo_files = sorted(f for f in os.listdir(geo_dir) if f.endswith('.hdf'))
file_pairs = [(os.path.join(modis_dir, mf), os.path.join(geo_dir, gf)) for mf, gf in zip(modis_files, geo_files)]

lat_res, lon_res = resolution
lat_bins = np.arange(-90, 90 + lat_res, lat_res)
lon_bins = np.arange(-180, 180 + lon_res, lon_res)
lat_center = (lat_bins[:-1] + lat_bins[1:]) / 2
lon_center = (lon_bins[:-1] + lon_bins[1:]) / 2

gc.enable()  # Activer la collecte automatique
save_to_netcdf(file_pairs, output_file, lat_center, lon_center)
logger.info("Traitement terminé.")