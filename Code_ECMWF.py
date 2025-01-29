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
# Main function
def process_file(infile, outdir):
    # Open the dataset
    ratio = xr.open_dataset(infile)
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

    ds_mean_sorted = ds_mean.sortby('longitude') 

    # Step 3: Define target latitude and longitude for regridding
    target_lat = np.linspace(-90, 90, 8192)  
    target_lon = np.linspace(-180, 180, 16384)  

    # Regrid function
    def regrid_2d(var, old_lat, old_lon, new_lat, new_lon):
        interpolator = RegularGridInterpolator((old_lat, old_lon), var, bounds_error=False, fill_value=None)
        new_lon_grid, new_lat_grid = np.meshgrid(new_lon, new_lat)
        regridded_var = interpolator((new_lat_grid, new_lon_grid))
        return regridded_var

    variables = []
    old_lat = ds_mean_sorted['latitude'].values  
    old_lon = shifted_lon_sorted  

    for var_name in ds_mean_sorted.data_vars:
        var = ds_mean_sorted[var_name].values
        regridded_var = np.empty((25, 8192, 16384))  

        for i in range(25):
            regridded_var[i, :, :] = regrid_2d(var[i, :, :], old_lat, old_lon, target_lat, target_lon)

        variables.append(regridded_var)

    final_data = np.stack(variables, axis=0)

    temperature_dataset = ds_mean_sorted['t'].values
    regridded_temp = np.empty((25, 8192, 16384))
    for i in range(25):
        regridded_temp[i, :, :] = regrid_2d(temperature_dataset[i, :, :], old_lat, old_lon, target_lat, target_lon)

    new_products = []
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

    for var_index in range(5):
        new_product = []  
        for level in range(25):
            var_at_level = final_data[var_index, level, :, :]  
            calculated_var = calculate_new_product(var_at_level, radius[var_index], sigma, rho[var_index], pressure_lvl[level], regridded_temp[level, :, :])
            new_product.append(calculated_var)
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
            'latitude': target_lat,
            'longitude': target_lon
        }
    )

    pressure_levels = new_var_dataset['level'].values
    pressure_thickness = np.diff(np.append(pressure_levels, pressure_levels[-1]))

    new_2D_products = {}
    for var_name in new_var_dataset.data_vars:
        var_data = new_var_dataset[var_name].values
        integrated_var = np.average(var_data, axis=0, weights=pressure_thickness)
        new_2D_products[var_name] = (['latitude', 'longitude'], integrated_var)

    integrated_2D_N = xr.Dataset(
        new_2D_products,
        coords={'latitude': new_var_dataset['latitude'].values, 'longitude': new_var_dataset['longitude'].values}
    )

    # Save the final result
    integrated_2D_N.to_netcdf(outfile)
   

# Main function for multiple files
def main(indir, outdir):
    # List all NetCDF files in the input directory
    files = glob.glob(os.path.join(indir, "*.nc"))
    
    # Process each file
    for infile in files:
        process_file(infile, outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process multiple NetCDF files for aerosols.')
    parser.add_argument('--indir', type=str, required=True, help='Directory with input NetCDF files')
    parser.add_argument('--outdir', type=str, required=True, help='Directory to save the output NetCDF files')
    
    args = parser.parse_args()

    # Run the main function for multiple files
    main(args.indir, args.outdir)
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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:55:08 2024

@author: devigne
"""
"""
import numpy as np
import xarray as xr
import pandas as pd
import os
import fnmatch
from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
# Set environment variable for PROJ_LIB
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"

# Define years to process
years = range(2023, 2024)

# Define directory paths
modis_dir_base = '/LARGE/MODIS/MYD08_D3.061'
tropomi_dir_base = '/home/devigne/AI_resample_'
nd_dir_base = '/LARGE9/MODIS/erg10'
era5_base = '/home/devigne/land_sea_mask'
eac4_base = '/home/devigne/Files_NumConc_2D/'
# Function to get files for a year
def get_files(directory, pattern):
    return sorted([os.path.join(directory, fname) for fname in os.listdir(directory) if fnmatch.fnmatch(fname, pattern)])

# Function to get dates from filenames
def get_dates_from_filenames(filenames, date_format, date_position):
    dates = []
    for fname in filenames:
        date_str = os.path.basename(fname).split('_')[date_position]
        dates.append(datetime.strptime(date_str, date_format))
    return dates

# Get MODIS, TROPOMI, and ND files and dates
modis_files, tropomi_files, nd_files, eac4_files = [], [], [], []
for year in years:
    modis_files.extend(get_files(f'{modis_dir_base}/{year}/', 'MYD08_D3.A*.hdf'))
    tropomi_files.extend(get_files(f'{tropomi_dir_base}{year}/', f'resampled_data_tropomi_new_{year}-*.nc'))
    nd_files.extend(get_files(f'{nd_dir_base}/2023_new/', f'modis_nd.{year}.*.nc'))
    eac4_files.extend(get_files(f'{eac4_base}/{year}/', f'2D_Naer_dataset_{year}-*.nc'))
    


date_modis = []
for filename in modis_files:
    basename = os.path.basename(filename)
    # Extract the date substring
    date_substring = basename.split('.')[1]
    # Extract year and day of year from the substring
    year = int(date_substring[1:5])
    day_of_year = int(date_substring[5:8])
    # Convert day of year to a specific date
    date = datetime(year=year, month=1, day=1) + timedelta(day_of_year - 1)
    # Format the date
    formatted_date = date.strftime("%Y-%m-%d")
    date_m = datetime.strptime(str(formatted_date), '%Y-%m-%d')
    date_modis.append(date_m)

date_tropo = []
for file in tropomi_files:
    basename = os.path.basename(file)
    # Split the basename to get the part containing the date
    date_part = basename.split('_')[-1]
    # Extract the date
    date_str = date_part.split('.')[0]
    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_tropo.append(date)

date_eac4 = []
for file in eac4_files:
    basename = os.path.basename(file)
    # Split the basename to get the part containing the date
    date_part = basename.split('_')[-1]
    # Extract the date
    date_str = date_part.split('.')[0]
    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_eac4.append(date)

date_nd = []
for filename in nd_files:
    basename = os.path.basename(filename)
    # Extract the date substring
    year = int(basename[9:13])
    day_of_year = int(basename[14:17])
    # Convert day of year to a specific date
    date = datetime(year=year, month=1, day=1) + timedelta(day_of_year - 1)
    # Format the date
    formatted_date = date.strftime("%Y-%m-%d")
    date_n = datetime.strptime(str(formatted_date), '%Y-%m-%d')
    date_nd.append(date_n)   

# Find common dates
common_dates = sorted(set(date_modis) & set(date_tropo) & set(date_nd) & set(date_eac4))

# Function to load HDF data
def load_hdf(filename, scale_factor, fillvalue, variable_list, date):
    hdf = SD(filename, SDC.READ)
    data_dict = {'date': [date] * 64800}
    for var_name in variable_list:
        sds = hdf.select(var_name)
        var = np.array(sds) * scale_factor
        data_dict[var_name] = np.flip(var, axis=0).flatten('F')
        data_dict[var_name] = ma.masked_equal(data_dict[var_name], fillvalue * scale_factor)
    sds2 = hdf.select('Cloud_Top_Temperature_Nadir_Mean')
    var2 = np.array(sds2)
    var2 = np.flip((var2 + 15000)*0.01, axis = 0)
    data_dict['Cloud_Top_Temperature_Nadir_Mean'] = var2.flatten('F')
    data_dict['Cloud_Top_Temperature_Nadir_Mean'] = ma.masked_equal(data_dict['Cloud_Top_Temperature_Nadir_Mean'], fillvalue*scale_factor)
    sds3 = hdf.select('Cloud_Top_Height_Mean')
    var3 = np.array(sds3)
    data_dict['Cloud_Top_Height_Mean'] = np.flip(var3, axis=0).flatten('F')
    data_dict['Cloud_Top_Height_Mean'] = ma.masked_equal(data_dict['Cloud_Top_Height_Mean'], fillvalue)
    lat = hdf.select('YDim')
    lat1 = np.flip(lat, axis = 0)
    lon = hdf.select('XDim')
    lat_modis = np.tile(lat1,360)
    lon_modis = np.repeat(list(lon), 180)
    data_dict['latitude'] = lat_modis
    data_dict['longitude'] = lon_modis
    return pd.DataFrame(data_dict)

def open_nd(nc_file, var_list, date):
    # Open netCDF file
    data = xr.open_dataset(nc_file,  decode_times=False)
    
    # Extract dimensions
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.linspace(-179.5, 179.5, 360)
    
    # Initialize an empty dictionary to store data
    data_dict = {'date': [date] * 64800, 'latitude': [], 'longitude': []}
    
    # Iterate over variables in the netCDF file
    for var_name in var_list:
        values = data[var_name].isel(time=0).values
        # Flatten the data
        tab = np.flipud(np.array(values).T)
        #tab = np.flip(tab, axis = 0)
        flat_values = tab.flatten('F')
        data_dict[var_name] = flat_values
    
    # Repeat latitude and longitude values for each variable
    for lon_val in lon:
        for lat_val in lat:
            data_dict['latitude'].append(lat_val)
            data_dict['longitude'].append(lon_val)
    
    # Construct pandas dataframe
    df = pd.DataFrame(data_dict)
    
    return df

def open_netCDF_to_dataframe(nc_file, var_list, date):
    # Open netCDF file
    data = xr.open_dataset(nc_file)
    
    # Extract dimensions
    lat = data['latitude'].values
    lon = data['longitude'].values
    
    # Initialize an empty dictionary to store data
    data_dict = {'date': [date] * 64800, 'latitude': [], 'longitude': []}
    
    # Iterate over variables in the netCDF file
    for var_name in var_list:
        values = data[var_name].values
        # Flatten the data
        tab = np.array(values)
        #tab = np.flip(tab, axis = 0)
        flat_values = tab.flatten('F')
        data_dict[var_name] = flat_values
    
    # Repeat latitude and longitude values for each variable
    for lon_val in lon:
        for lat_val in lat:
            data_dict['latitude'].append(lat_val)
            data_dict['longitude'].append(lon_val)
    
    # Construct pandas dataframe
    df = pd.DataFrame(data_dict)
    
    return df

def load_Land_Fraction(filename, fillvalue, date):
    hdf = SD(filename, SDC.READ)
    data_dict = {'date': [date] * 64800}
    sds1 = hdf.select('Land_Fraction_Day')
    var1 = np.array(sds1) * 0.0001
    data_dict['Land_Fraction_Day'] = np.flip(var1, axis=0).flatten('F')
    data_dict['Land_Fraction_Day'] = np.where(data_dict['Land_Fraction_Day'] == fillvalue * 0.0001, np.nan, data_dict['Land_Fraction_Day'])
    lat = hdf.select('YDim')
    lat1 = np.flip(lat, axis=0)
    lon = hdf.select('XDim')
    lat_modis = np.tile(lat1, 360)
    lon_modis = np.repeat(list(lon), 180)
    data_dict['latitude'] = lat_modis
    data_dict['longitude'] = lon_modis
    return pd.DataFrame(data_dict)

def process_land_fraction(modis_files, date_modis, fillvalue=-9999):
    LF = pd.DataFrame()
    for i in range(len(modis_files)):
        date_str = date_modis[i].strftime('%Y-%m-%d')
        daily_LF = load_Land_Fraction(modis_files[i], -9999, date_str)
        LF = pd.concat([LF, daily_LF], axis=0)
    
    # Remove rows with NaN values
    LF = LF.dropna(subset=['Land_Fraction_Day'])

    # Group by latitude and longitude and calculate the mean Land Fraction
    LF_avg = LF.groupby(['latitude', 'longitude'])['Land_Fraction_Day'].mean().reset_index()
    LF_pivot = LF_avg.pivot(index='latitude', columns='longitude', values='Land_Fraction_Day')

    return LF_avg, LF_pivot
LF_avg, LF_pivot = process_land_fraction(modis_files, date_modis)

lat = np.linspace(-89.5, 89.5, 180)
lon = np.linspace(-179.5, 179.5, 360)

file = '/home/devigne/land_sea_mask'
try:
    lsm = xr.open_dataset(file, engine='netcdf4')
except ValueError:
    lsm = xr.open_dataset(file, engine='scipy')

# Convert to DataArray and squeeze out single-dimensional coordinates
lsm = lsm.to_array().squeeze()

# Take the mean if there is a time dimension
if 'time' in lsm.dims:
    LSM = lsm.mean(dim='time')
else:
    LSM = lsm

# Coarsen data
LSM_coarsened = LSM.coarsen(latitude=4, longitude=4, boundary='trim').mean()

# Extract longitude and latitude
longitude = LSM_coarsened['longitude'].values
latitude = LSM_coarsened['latitude'].values

# Print current longitude range
print(f"Original Longitude range: {longitude.min()} to {longitude.max()}")

# Shift longitude and data
def shift_longitude_and_data(dataarray):
    lon = dataarray['longitude'].values
    lat = dataarray['latitude'].values

    # Shift longitudes from [0, 360) to [-180, 180)
    if lon.max() > 180:
        lon = np.where(lon > 180, lon - 360, lon)
        lon = np.sort(lon)

        # Shift data to match new longitude order
        new_data = dataarray.values
        reordered_data = np.roll(new_data, shift=int(len(lon)/2), axis=1)
        
        # Create a new DataArray with shifted coordinates
        return xr.DataArray(reordered_data, coords=[lat, lon], dims=['latitude', 'longitude'])
    else:
        return dataarray

LSM_shifted = np.flip(shift_longitude_and_data(LSM_coarsened), axis = 0)

# Flatten the data for DataFrame creation
data = {
    'latitude': np.repeat(LSM_shifted['latitude'].values, len(LSM_shifted['longitude'])),
    'longitude': np.tile(LSM_shifted['longitude'].values, len(LSM_shifted['latitude'])),
    'LSM': LSM_shifted.values.flatten(order='F')
}


# Create the DataFrame
df_lsm = pd.DataFrame(data)

# Define regions for filtering
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

# Function to filter data based on regions and land/ocean type
def filter_data(df):
    df['Zone'] = np.nan
    for region_name, region in regions.items():
        region_mask = (df['latitude'] >= region['lat_min']) & (df['latitude'] <= region['lat_max']) & \
                      (df['longitude'] >= region['lon_min']) & (df['longitude'] <= region['lon_max'])
        df.loc[region_mask, 'Zone'] = region['id']
        if region['type'] == 'Ocean':
            df.loc[region_mask & (df['LSM'] >= 0.4), 'Zone'] = np.nan
        elif region['type'] == 'Land':
            df.loc[region_mask & (df['LSM'] <= 0.4), 'Zone'] = np.nan
    return df


# Process each common date
for common_date in common_dates:
    date_str = common_date.strftime('%Y-%m-%d')
    modis_file = modis_files[date_modis.index(common_date)]
    tropomi_file = tropomi_files[date_tropo.index(common_date)]
    nd_file = nd_files[date_nd.index(common_date)]
    eac4_file = eac4_files[date_eac4.index(common_date)]
    # Load MODIS data
    modis_vars = ['Cloud_Optical_Thickness_Liquid_Mean', 'Cloud_Optical_Thickness_16_Liquid_Mean', 'Cloud_Optical_Thickness_37_Liquid_Mean',
                  'Cloud_Effective_Radius_Liquid_Mean', 'Cloud_Effective_Radius_16_Liquid_Mean', 'Cloud_Effective_Radius_37_Liquid_Mean',
                  'Cloud_Water_Path_37_Liquid_Mean', 'Cloud_Water_Path_16_Liquid_Mean', 'Cloud_Water_Path_Liquid_Mean']
    modis_data = load_hdf(modis_file, 0.01, -9999, modis_vars, date_str)
    modis_data['LSM'] = df_lsm['LSM']
    modis_data = filter_data(modis_data)
    

    
    # Load TROPOMI data
    tropomi_vars = ['ai', 'af_acc_abl_abs', 'ai_acc_abl_abs', 'af_acc_bbl_abs', 'ai_acc_bbl_abs', 'af_clr_bbl_abs', 
                    'dist_acc_abl_abs', 'dist_acc_bbl_abs', 'blh', 'ai_clr_bbl_abs', 'dist_clr_bbl_abs', 'cth', 'cf']
    tropomi_data = open_netCDF_to_dataframe(tropomi_file, tropomi_vars, date_str)
    
    eac4_vars = ['N_dus', 'N_dum', 'N_dul', 'N_bchphil', 'N_bchphob']
    eac4_data = open_netCDF_to_dataframe(eac4_file, eac4_vars, date_str)
    
    # Load ND data
    var_nd = ['Nd_G18', 'Nd_G18_16', 'Nd_G18_37']
    nd_data = open_nd(nd_file, var_nd, date_str)
    modis_data['date'] = pd.to_datetime(modis_data['date'])
    tropomi_data['date'] = pd.to_datetime(tropomi_data['date'])
    nd_data['date'] = pd.to_datetime(nd_data['date'])
    eac4_data['date'] = pd.to_datetime(eac4_data['date'])
    # Merge all dataframes
    
    merged_df = pd.merge(modis_data, tropomi_data, on=['date', 'latitude', 'longitude'], how='inner')
    merged_df = pd.merge(merged_df, nd_data, on=['date', 'latitude', 'longitude'], how='inner')
    merged_df = pd.merge(merged_df, eac4_data, on=['date', 'latitude', 'longitude'], how='inner')
    # Filter data based on regions
    
    
    ds = xr.Dataset()

    # Add each variable to the Dataset with metadata
    for column in merged_df.columns:
        if column not in ['date', 'latitude', 'longitude']:
            ds[column] = xr.DataArray(
                merged_df[column].values,
                dims=("obs"),
                attrs={
                    'name': column,
                    'fill_value': np.nan  # Set the fill value (can also use another number if preferred)
                }
            )
    
    # Add coordinates as DataArrays
    ds['date'] = xr.DataArray(merged_df['date'].values, dims="obs")
    ds['latitude'] = xr.DataArray(merged_df['latitude'].values, dims="obs")
    ds['longitude'] = xr.DataArray(merged_df['longitude'].values, dims="obs")
    
    # Define your output file path
    output_file = f'/home/devigne/ACI_Data_Files/absorbing_cloud_data_{date_str}.nc'

    # Save as NetCDF
    ds.to_netcdf(output_file)
    print(f'Saved filtered data to {output_file}')

colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#d2f53c', '#fabebe',
    '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000080'
]
mat_test = np.array(merged_df['Zone']).reshape((180, 360), order='F')
import matplotlib as mpl
cmapa = mpl.colors.ListedColormap(colors)

# Plot 1
fig = plt.figure(figsize=(12, 10))
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 181, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(360, 180)

x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, mat_test, cmap=cmapa, vmin = 1, vmax = 19)
my_map1.colorbar(cs1, label=r'Zone')
#my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
plt.title('Zones')
plt.plot()
plt.savefig('/home/devigne/Zone_map.pdf')

"""
