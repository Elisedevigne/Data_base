#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:02:38 2024

@author: devigne
"""

import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import argparse
import pandas as pd
import os  # To help with path management
import glob  # To handle multiple files
#import xesmf as xe
from tqdm import tqdm
import logging
import gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main function
def process_file(infile, outdir):
    # Open the dataset
    ratio = xr.open_dataset(infile)
    for var_name in ratio.data_vars:
        ratio[var_name] = ratio[var_name].astype(np.float32)
    
    date_eac4 = pd.to_datetime(ratio['time'].values[0]).strftime('%Y-%m-%d')
    
    # Define the output file path based on date_eac4
    outfile = os.path.join(outdir, f"2D_Naer_dataset_{date_eac4}.nc")    
    # Step 1: Averaging over the time dimension (dim 0)
    ds_mean = ratio.mean(dim='time')

    # Step 2: Shift longitude from [0, 360] to [-180, 180]
    old_lon = ds_mean['longitude'].values  
    shifted_lon = np.where(old_lon > 180, old_lon - 360, old_lon)  
    ds_mean['longitude'] = xr.where(ds_mean['longitude'] > 180, ds_mean['longitude'] - 360, ds_mean['longitude'])
    sorted_indices = np.argsort(ds_mean['longitude'].values)
    shifted_lon_sorted = shifted_lon[sorted_indices]
    
    ds_mean_sorted = ds_mean.isel(longitude=sorted_indices)  # Réorganise explicitement les données
    ds_mean_sorted = ds_mean_sorted.sortby('latitude')
    print(ds_mean_sorted)
   
    # Step 3: Define target latitude and longitude for regridding
    target_lat = np.linspace(-90, 90, 8192)  
    target_lon = np.linspace(-180, 180, 16384)  

    # Regrid function
    def regrid_2d(var, old_lat, old_lon, new_lat, new_lon):
        interpolator = RegularGridInterpolator((old_lat, old_lon), var, method = 'nearest',bounds_error=False, fill_value=None)
        new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)
        regridded_var = interpolator((new_lat_grid, new_lon_grid))
        return regridded_var

    
    pressure_lvl = ds_mean_sorted['level'].values
    regridded_temp= ds_mean_sorted['t'].values  # Température après interpolation

    
    pressure_lvl = ds_mean_sorted['level'].values
    r_bc = 0.0118e-6
    r_dust = 0.29e-6
    rho_bc = 1000
    rho_dust = 2610
    sigma = 2
    radius = [r_dust, r_dust, r_dust, r_bc, r_bc]
    rho = [rho_dust, rho_dust, rho_dust, rho_bc, rho_bc]

    def calculate_new_product(var, r0, sigma_g, rho_p, plev, temp):
        M_air = 29*(10**(-3))
        R = 8.314
        rho_air = (plev*M_air)/(R*temp)
        rho_aer = rho_air*var
        beta = np.exp(1.5*(np.log(sigma_g))**2)
        mp = (4/3)*np.pi*rho_p*(r0*beta)**3
        Na = rho_aer/mp
        return Na
    list_var = ['aermr04', 'aermr05', 'aermr06', 'aermr09', 'aermr10']
    new_products = []
    for var_index, var_name in enumerate(list_var):
        new_product = [] 
        for level in range(25):
            var_at_level = ds_mean_sorted[var_name].values[level, :, :]
            calculated_var = calculate_new_product(var_at_level, radius[var_index], sigma, rho[var_index], pressure_lvl[level], regridded_temp[level, :, :])
            new_product.append(calculated_var.astype(np.float32))
        new_product_array = np.stack(new_product, axis=0)
        new_products.append(new_product_array)

    new_var_dataset = xr.Dataset(
        {
            'N_dus': (['level', 'latitude', 'longitude'], new_products[0]),
            'N_dum': (['level', 'latitude', 'longitude'], new_products[1]),
            'N_dul': (['level', 'latitude', 'longitude'], new_products[2]),
            'N_bchphil': (['level', 'latitude', 'longitude'], new_products[3]),
            'N_bchphob': (['level', 'latitude', 'longitude'], new_products[4]),
        },
        coords={
            'level': ds_mean_sorted['level'].values,
            'latitude': ds_mean_sorted['latitude'].values,
            'longitude': ds_mean_sorted['longitude'].values
        }
    )

    pressure_levels = new_var_dataset['level'].values
    pressure_thickness = np.diff(np.append(pressure_levels, pressure_levels[-1]))

    new_2D_products = {}
    for var_name in new_var_dataset.data_vars:
        var_data = new_var_dataset[var_name].values
        integrated_var = np.average(var_data, axis=0, weights=pressure_thickness)
        integrated_var = regrid_2d(integrated_var, ds_mean_sorted['latitude'].values ,ds_mean_sorted['longitude'].values, target_lat, target_lon)
        new_2D_products[var_name] = (['latitude', 'longitude'], integrated_var)

    integrated_2D_N = xr.Dataset(
        new_2D_products,
        coords={'latitude': target_lat, 'longitude': target_lon}
    )
    print(integrated_2D_N)
    # Save the final result
    comp = dict(zlib=True, complevel=6)
    encoding = {var: comp for var in integrated_2D_N.data_vars}
    integrated_2D_N.to_netcdf(outfile, encoding=encoding)
   

# Main function for multiple files
def main(indir, outdir):
    # List all NetCDF files in the input directory
    files = glob.glob(os.path.join(indir, "*.nc"))
    
    # Process each file
    for infile in tqdm(files, total=len(files)):
        process_file(infile, outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process multiple NetCDF files for aerosols.')
    parser.add_argument('--indir', type=str, required=True, help='Directory with input NetCDF files')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save the output NetCDF files')
    #parser.add_argument('--output-file', type=str, required=True, help="Fichier de sortie NetCDF")
    
    args = parser.parse_args()
    #output_file = args.output_file
    gc.enable()  # Activer la collecte automatique

    # Run the main function for multiple files
    main(args.indir, args.outdir)
    logger.info("Traitement terminé.")
#%%

"""
### PLOTTING TESTS ###


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

NC_file = xr.open_dataset('/Users/devigne/Documents/THESE_2023_2026/2D_Naer_dataset_2020-07-01.nc')
dust_nc = NC_file['N_bchphil'].values
# Plot 1
fig = plt.figure(figsize=(12, 10))
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 181, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(360, 180)

x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, dust_nc, cmap='Reds', vmin = 0)
my_map1.colorbar(cs1, label=r'Large Dust Number Concentration ($m^{-2}$)')
#my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
plt.title('Dust Number Concentration 2020-07-01')
plt.plot()

"""

