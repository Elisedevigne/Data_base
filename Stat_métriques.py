#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:40:17 2024

@author: devigne
"""
import xarray as xr
import pandas as pd
import os
import numpy as np
import fnmatch
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
# Function to get the list of files matching the pattern in a directory
def get_files(directory, pattern):
    return sorted([os.path.join(directory, fname) for fname in os.listdir(directory) if fnmatch.fnmatch(fname, pattern)])

def make_stat(df_data, region, var_to_mean, freq):
    # Filter the dataframe for the given region
    df_data['date'] = pd.to_datetime(df_data['date'])
    df_filtered = df_data[df_data['region'] == region]
    df_filtered = df_filtered[df_filtered['date'].dt.month.isin([1, 2])]
    df_filtered = df_filtered[df_filtered['date'].dt.year.isin([2020])]
    # Group by week and calculate the mean of the specified variable
    df_mean = df_filtered.groupby(pd.Grouper(key='date', axis=0,  
                      freq=freq))[var_to_mean].median()     
    return df_mean

# Get the list of NetCDF files
data_files = get_files('/Users/devigne/Documents/THESE_2023_2026/Metrics/', 'region_differences_*.nc')

# Initialize an empty list to store DataFrames
dataframes = []

# Process each NetCDF file
for filename in data_files:
    # Open the NetCDF file, specifying the engine correctly
    ds = xr.open_dataset(filename, engine='netcdf4')
    
    # Drop the 'index' variable if it exists
    if 'index' in ds.variables:
        ds = ds.drop_vars('index')
    
    # Convert the dataset to a DataFrame
    df = ds.to_dataframe().reset_index()
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames along axis 0
final_df = pd.concat(dataframes, axis=0)

# Display the concatenated DataFrame
print(final_df.head())
for var in ds.variables:
    print(var)

#%%

###Tendency over the period 2019-2023###

regions = ['Peruvian', 'Namibian', 'Australian', 'Californian', 'Canarian', 'China', 'North Atlantic', 'Northeast Pacific', 'Northwest Pacific', 'Southeast Pacific', 'South Atlantic', 'South Indian Ocean', 'Galapagos', 'Chinese Stratus', 'Amazon', 'Equatorial Africa', 'North America', 'India', 'Europe']
param = 'Cloud_Optical_Thickness_37_Liquid_Mean'
frequence = 'D'
for zone in regions:
    df_mean_ai = make_stat(final_df, zone, f'median_hp_aac_{param}', frequence) 
    df_mean_ai_hp = make_stat(final_df, zone, f'median_hp_bbl_{param}', frequence)
    df_mean_ai_lp = make_stat(final_df, zone, f'median_hp_clr_{param}', frequence)
    
    df_mean_ai2 = make_stat(final_df, zone, f'median_lp_aac_{param}', frequence)
    df_mean_ai_hp2 = make_stat(final_df, zone, f'median_lp_bbl_{param}', frequence) 
    df_mean_ai_lp2 = make_stat(final_df, zone, f'median_lp_clr_{param}', frequence)
    
    df_nonaer = make_stat(final_df, zone, f'median_non_aer_{param}', frequence)
    fig, ax = plt.subplots(figsize=(12, 8))
    #ax.plot(df_mean_ai.index.values, df_mean_ai.values, label='COT_AAC_ABL_HP - Non_Aer', color = 'red', ls = '--')
    #ax.plot(df_mean_ai_hp.index.values, df_mean_ai_hp.values, label='Median COT_AAC_BBL_HP', color = 'red', ls = '-.')
    #ax.plot(df_mean_ai_lp.index.values, df_mean_ai_lp.values, label='Median COT_CLR_BBL_HP', color = 'red', ls = ':')
    
    #ax.plot(df_mean_ai2.index.values, df_mean_ai2.values, label='COT_AAC_ABL_LP - Non_Aer', color = 'blue', ls = '--')
    #ax.plot(df_mean_ai_hp2.index.values, df_mean_ai_hp2.values, label='Median COT_AAC_BBL_LP', color = 'blue', ls = '-.')
    #ax.plot(df_mean_ai_lp2.index.values, df_mean_ai_lp2.values, label='Median COT_CLR_BBL_LP', color = 'blue', ls = ':')
    
    ax.plot(df_nonaer.index.values, df_nonaer.values, label=f'{param} Non Aer', color = 'black', ls = '-')

    # Fill the area between the lower and upper bounds
    ax.fill_between(df_mean_ai.index.values, df_mean_ai2.values, df_mean_ai.values,
                    color='green', alpha=0.3, label='Confidence Interval')
    
    # Set labels and title
    ax.set(xlabel="Date", ylabel=r"$COT_{3.7µm}$ ",
           title=f"Daily COT (AAC compared to Non aer case) \n{zone}")
    
    # Format the x axis
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    # Rotate the x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.savefig(f'/Users/devigne/Documents/THESE_2023_2026/Wildfires_Cases/{zone}/DJF_2021_{param}_AAC_vs_Nonaer.pdf')
    plt.show()
    
#%%%
    
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy.ma as ma
# Assuming `fire_events` and `final_df` are defined and available
# Assuming `regions` and `param` are also defined
def fire_events(data_set, region, param_1, param_2):
    data_set['date'] = pd.to_datetime(data_set['date'])
    df_filtered = data_set[data_set['region'] == region]
    #df_filtered = df_filtered[df_filtered['date'].dt.month.isin([1, 2, 12])]
    threshold = df_filtered[param_1].quantile(0.75)
    df_filtered['event'] = ma.masked_where((df_filtered[param_1].values<threshold), df_filtered[param_2])
    df_filtered['climatology'] = ma.masked_where((df_filtered[param_1].values>=threshold), df_filtered[param_2])
    return df_filtered

param = 'Cloud_Effective_Radius_37_Liquid_Mean'

for zone in regions:
    data_set = fire_events(final_df, zone, 'AI_hp_CLR', f'median_hp_clr_{param}')
    
    # Ensure only finite values are used
    event_data = data_set['event'][np.isfinite(data_set['event'])]
    climatology_data = data_set['climatology'][np.isfinite(data_set['climatology'])]
    
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
    plt.title(f'{param} distribution DJF 2020 \n{zone} as a function of AI condition (CLR BBL)')
    plt.xlabel(r'$CER_{3.7µm}$ (µm)')
    plt.ylabel('Density')
    plt.legend(['Particular events', 'Background'])
    plt.savefig(f'/Users/devigne/Documents/THESE_2023_2026/Wildfires_Cases/{zone}/Histogramme_{param}_{zone}_CLR_DJF_2020.pdf')

    plt.show()
#%%
import numpy as np
import pandas as pd
import xarray as xr
"""
from mpl_toolkits.basemap import Basemap
# Open the dataset and convert to DataArray
file = '/Users/devigne/Documents/THESE_2023_2026/land_sea_mask'

lsm = xr.open_dataset(file)
longitude = lsm['longitude']
if longitude.max() > 180:
    # Assume longitudes are in [0, 360) and convert to [-180, 180)
    longitude = np.where(longitude <180 , longitude - 360, longitude)
    lsm = lsm.assign_coords(longitude=("longitude", longitude))

#lsm['longitude'] = xr.where(lsm['longitude'] > 180, lsm['longitude'] - 360, lsm['longitude'])
lsm = lsm.coarsen(latitude=4, longitude=4, boundary='trim').mean()
lsm = lsm.to_array()  # Convert to DataArray

# Take the mean across time dimension if it exists
if 'time' in lsm.dims:
    LSM = lsm.mean(dim='time').sel(variable='lsm').values.squeeze()
else:
    LSM = lsm

# Coarsen the data to a resolution of (180, 360)

LSM = np.flip(LSM, axis = 0)
print(LSM)

LSM = np.array(LSM)

# Plot 1
fig = plt.figure(figsize=(12, 10))
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 91, 30), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 181, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(360, 180)

x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, LSM, cmap='rainbow', vmin = 0, vmax = 1)
my_map1.colorbar(cs1, label=r'Zone')
#my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
plt.title('Zones')
plt.plot()

"""
from datetime import datetime

# Variables
var_cams = ['aermssbchphil', 'aermssbchphob', 'aermssdus', 'aermssdum', 'aermssdul']

# Dummy get_files function (you should replace this with your own logic)
data_cams = get_files('/Users/devigne/Documents/THESE_2023_2026/2020/', 'cams-global-reanalysis-eac4_*.nc')

date_cams = []
df_cams = []

for file in data_cams:
    basename = os.path.basename(file)
    # Extract the date from filename
    date_part = basename.split('_')[-1]
    date_str = date_part.split('.')[0]

    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_cams.append(date)

    dt_cams = xr.open_dataset(file)
    
    # Adjust longitudes to [-180, 180) if necessary
    lon_cams = dt_cams['longitude']
    if lon_cams.max() > 180:
        lon_cams = np.where(lon_cams > 180, lon_cams - 360, lon_cams)
        dt_cams = dt_cams.assign_coords(longitude=lon_cams)

    # Define new lat and lon grid
    new_lat = np.linspace(-89.5, 89.5, 180)
    new_lon = np.linspace(-179.5, 179.5, 360)

    interpolated_ds = {'latitude': np.tile(new_lat, 360), 'longitude': np.repeat(list(new_lon), 180)}
    
    for var in var_cams:
        # Select variable and average across time (assuming 'valid_time' is time)
        variable_data = dt_cams[var].mean(dim='valid_time')
        
        # Interpolate to new grid
        interpolated_data = variable_data.interp(latitude=new_lat, longitude=new_lon, method='cubic')
        
        # Flatten in Fortran order
        interpolated_ds[var] = interpolated_data.values.flatten(order='F')
    
    # Append interpolated dataset to list
    df_cams.append(pd.DataFrame(interpolated_ds))

# Concatenate all DataFrames along axis 0
final_cams = pd.concat(df_cams, axis=0)
final_cams.set_axis(np.repeat(list(date_cams), 64800), axis='index')   


#%%
import pandas as pd
import xarray as xr
import os
import numpy as np

# Function to get list of files matching a pattern
def get_files(directory, pattern):
    import glob
    return glob.glob(os.path.join(directory, pattern))

# Define the path and file pattern
data_files = get_files('/Users/devigne/Documents/THESE_2023_2026/ACI_Data_Files/', 'absorbing_cloud_data_2023*.nc')
dataframes_aerosol = []

# Process each NetCDF file
for filename in data_files:
    try:
        # Open the NetCDF file
        ds = xr.open_dataset(filename, engine='netcdf4')
        
        # Drop 'index' variable if it exists
        if 'index' in ds.variables:
            ds = ds.drop_vars('index')
        
        # Convert the dataset to a DataFrame
        df = ds.to_dataframe().reset_index()
        
        # Append to the list of DataFrames
        dataframes_aerosol.append(df)
    except Exception as e:
        print(f"Could not process file {filename}: {e}")

# Concatenate all DataFrames along axis 0
final_df_aerosol = pd.concat(dataframes_aerosol, axis=0, ignore_index=True)

# Set 'date' as the index column if it exists
if 'date' in final_df_aerosol.columns:
    final_df_aerosol['date'] = pd.to_datetime(final_df_aerosol['date'])
    final_df_aerosol = final_df_aerosol.set_index('date')

# Display the final DataFrame structure
print(final_df_aerosol.head())

# Function to calculate mean and median
def calculate_means_medians(dataframe, columns):
    return np.nanmean(dataframe[columns]), np.nanmedian(dataframe[columns])

regions = ['Peruvian', 'Namibian', 'Australian', 'Californian', 'Canarian', 'China', 'North Atlantic', 'Northeast Pacific', 'Northwest Pacific', 'Southeast Pacific', 'South Atlantic', 'South Indian Ocean', 'Galapagos', 'Chinese Stratus', 'Amazon', 'Equatorial Africa', 'North America', 'India', 'Europe']

# Process each DataFrame in dataframes_aerosol
data_aci = []
for aus_cloudy_abs_data2 in dataframes_aerosol:
    # Convert index to datetime if necessary
    if 'date' in aus_cloudy_abs_data2.columns:
        aus_cloudy_abs_data2['date'] = pd.to_datetime(aus_cloudy_abs_data2['date'])
        aus_cloudy_abs_data2 = aus_cloudy_abs_data2.set_index('date')
        
    # Ensure there is data before accessing index
    if not aus_cloudy_abs_data2.empty:
        date = aus_cloudy_abs_data2.index.values[0]
        
        # Calculate new columns
        aus_cloudy_abs_data2['diff_CER'] = (aus_cloudy_abs_data2['Cloud_Effective_Radius_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Effective_Radius_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Effective_Radius_16_Liquid_Mean']
        aus_cloudy_abs_data2['diff_COT'] = (aus_cloudy_abs_data2['Cloud_Optical_Thickness_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Optical_Thickness_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Optical_Thickness_16_Liquid_Mean']
        aus_cloudy_abs_data2['diff_Nd'] = (aus_cloudy_abs_data2['Nd_G18_16'] - aus_cloudy_abs_data2['Nd_G18_37'])/aus_cloudy_abs_data2['Nd_G18_16']

        # Adjust Cloud Water Path values and mask negative values
        for col in ['Cloud_Water_Path_37_Liquid_Mean', 'Cloud_Water_Path_16_Liquid_Mean']:
            aus_cloudy_abs_data2[col] *= 100
            aus_cloudy_abs_data2[col] = aus_cloudy_abs_data2[col].mask(aus_cloudy_abs_data2[col] < 0)

        # Calculate the difference in Cloud Water Path
        aus_cloudy_abs_data2['diff_LWP'] = (aus_cloudy_abs_data2['Cloud_Water_Path_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Water_Path_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Water_Path_16_Liquid_Mean']

        # Filter data based on aerosol index and cloud top temperature
        modis_aus = aus_cloudy_abs_data2[
            (aus_cloudy_abs_data2['ai_clr_bbl_abs'] <= 0.05) |
            (aus_cloudy_abs_data2['ai_acc_abl_abs'] <= 0.05) |
            (aus_cloudy_abs_data2['ai_acc_bbl_abs'] <= 0.05) |
            aus_cloudy_abs_data2[['ai_acc_abl_abs', 'ai_clr_bbl_abs', 'ai_acc_bbl_abs']].isna().any(axis=1)
        ]
        #aus_cloudy_abs_data2.loc[aus_cloudy_abs_data2['Cloud_Top_Temperature_Nadir_Mean'] < 273] = np.nan

        data_aci.append(aus_cloudy_abs_data2)

# Concatenate all DataFrames along axis 0
final_df_aci = pd.concat(data_aci, axis=0)
print(final_df_aci.head())


#%%

df_étude = final_df_aci[final_df_aci.index.month.isin([1, 2, 3])]
#df_étude = df_étude[df_étude.index.year.isin([2020])]
###CORRELATIONS###
import numpy as np
import matplotlib.pyplot as plt

param1 = 'Cloud_Effective_Radius_37_Liquid_Mean'
param3 = 'N_bchphob'
param2 = 'ai_acc_bbl_abs'

non_nan = ~np.isnan(df_étude[param2])
df_étude = df_étude[non_nan]
df_étude = df_étude[(df_étude[param3]>= np.nanquantile(df_étude[param3], 0.65))]
# Moving average function
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

for i in range(1, 20):
    df_étude_zone = df_étude.loc[df_étude['Zone'] == i]
    zone = regions[i - 1]
    
    if df_étude_zone.empty:
        print(f"No data for Zone {i}, skipping...")
        continue
    
    tab_uvai = np.linspace(
        np.nanmin(df_étude_zone[param2]), 
        np.nanmax(df_étude_zone[param2]), 
        int((np.nanmax(df_étude_zone[param2]) - np.nanmin(df_étude_zone[param2])) / 0.1)
    )
    
    mean_diff = np.zeros(len(tab_uvai) - 1)
    mean_q1_COT = np.zeros(len(tab_uvai) - 1)
    mean_q3_COT = np.zeros(len(tab_uvai) - 1)
    
    for j in range(len(tab_uvai) - 1):
        array_bin = df_étude_zone[
            (df_étude_zone[param2] >= tab_uvai[j]) & 
            (df_étude_zone[param2] < tab_uvai[j + 1])
        ][param1]
        
        if len(array_bin) > 20:
            mean_diff[j] = np.nanmedian(array_bin)
            mean_q1_COT[j] = np.nanquantile(array_bin, 0.25)
            mean_q3_COT[j] = np.nanquantile(array_bin, 0.75)
        else:
            mean_diff[j] = np.nan
            mean_q1_COT[j] = np.nan
            mean_q3_COT[j] = np.nan
    
    # Plotting
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(tab_uvai[:-1], moving_average(mean_diff, 1), 'r')
    ax.fill_between(
        tab_uvai[:-1], 
        moving_average(mean_q1_COT, 1), 
        moving_average(mean_q3_COT, 1), 
        color='r', alpha=0.3
    )
    
    plt.xlabel(r'AI (AAC-ABL))')
    plt.ylabel(r'$CER_{3.7 µm}$')
    plt.legend(['median', 'percentiles'])
    plt.title(f'CER as a function of AI for high BC concentration ({zone} JFM 2023)')
    plt.show()

modis_l2 = xr.open_dataset('/Users/devigne/Documents/THESE_2023_2026/modis_data_2023_05_15.nc')
modis_l2 = modis_l2.to_dataframe()
COT = np.array(modis_l2['Cloud_Optical_Thickness_37']).reshape((8192,16384))
CER = np.array(modis_l2['Cloud_Effective_Radius_37']).reshape((8192,16384))
Nd = (1.37*10e-5)*(COT**(0.5))*((CER*10e-6)**(-2.5))*(10**(-4))
from mpl_toolkits.basemap import Basemap
fig = plt.figure(figsize=(12, 12))
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 90, 30), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 180, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(16384, 8192)

x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, Nd, cmap='coolwarm', vmax = 1100)
my_map1.colorbar(cs1, label=r'Nd $(cm^{-3})$')
#my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
#plt.title(f'{var} Domain 2 {date_hdf[i]}')

#plt.savefig('/Users/devigne/Documents/THESE_2023_2026/COT_test_2023_05_15.pdf')
plt.show()

nd_L3 = xr.open_dataset('/Users/devigne/Documents/THESE_2023_2026/modis_nd.2023.135.A.v1.nc',  decode_times=False)
nd_L3 = nd_L3.to_dataframe()
Nd_37 = np.array(nd_L3['Nd_G18_37']).reshape((360,360))
Nd_37 = np.flip(Nd_37, axis = 0)
fig = plt.figure(figsize=(12, 12))
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 90, 30), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 180, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(360,360)

x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, Nd_37, cmap='coolwarm', vmax = 500)
my_map1.colorbar(cs1, label=r'Nd $(cm^{-3})$')
#my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
#plt.title(f'{var} Domain 2 {date_hdf[i]}')

#plt.savefig('/Users/devigne/Documents/THESE_2023_2026/COT_test_2023_05_15.pdf')
plt.show()
#%%
###BOXPLOTS###
import matplotlib.lines as mlines

non_nan_indices = ~np.isnan(aus_cloudy_abs_data2['ai_acc_bbl_abs'])
df_abc = aus_cloudy_abs_data2[non_nan_indices]

non_nan = ~np.isnan(aus_cloudy_abs_data2['ai_acc_abl_abs'])
aus_cloudy_abs_data2 = aus_cloudy_abs_data2[non_nan]

non_nan_clear = ~np.isnan(aus_cloudy_abs_data2['ai_clr_bbl_abs'])
df_clear = aus_cloudy_abs_data2[non_nan_clear]

#modis_aus = modis_aus[non_nan]
q1 = np.nanquantile(aus_cloudy_abs_data2['ai_acc_abl_abs'], 0.10)
q3 = np.nanquantile(aus_cloudy_abs_data2['ai_acc_abl_abs'], 0.90)


q1_2 = np.nanquantile(aus_cloudy_abs_data2['ai_acc_bbl_abs'], 0.10)
q3_2 = np.nanquantile(aus_cloudy_abs_data2['ai_acc_bbl_abs'], 0.90)

q1_clr = np.nanquantile(aus_cloudy_abs_data2['ai_clr_bbl_abs'], 0.10)
q3_clr = np.nanquantile(aus_cloudy_abs_data2['ai_clr_bbl_abs'], 0.90)

print(q1, q3, q1_2, q3_2, q1_clr, q3_clr)
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

param = 'Cloud_Optical_Thickness_37_Liquid_Mean'
param2 = 'Cloud_Effective_Radius_37_Liquid_Mean'
param3 = 'Nd_G18_37'
param4 = 'Cloud_Water_Path_37_Liquid_Mean'
aus_cloudy_abs_data2['HP'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values<q3), aus_cloudy_abs_data2[param])
aus_cloudy_abs_data2['LP'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values>q1), aus_cloudy_abs_data2[param])
df_abc['HP'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values<q3_2), df_abc[param])
df_abc['LP'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values>q1_2), df_abc[param])
df_clear['HP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param])
df_clear['LP'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param])

count_aac_bbl_hp, count_aac_abl_hp, count_clear_bbl_hp = df_abc['HP'].count(), aus_cloudy_abs_data2['HP'].count(), df_clear['HP'].count()
count_aac_bbl_lp, count_aac_abl_lp, count_clear_bbl_lp = df_abc['LP'].count(), aus_cloudy_abs_data2['LP'].count(), df_clear['LP'].count()


aus_cloudy_abs_data2['HP1'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values<q3), aus_cloudy_abs_data2[param2])
aus_cloudy_abs_data2['LP1'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values>q1), aus_cloudy_abs_data2[param2])
df_abc['HP1'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values<q3_2), df_abc[param2])
df_abc['LP1'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values>q1_2), df_abc[param2])
df_clear['HP1'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param2])
df_clear['LP1'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param2])

aus_cloudy_abs_data2['HP2'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values<q3), aus_cloudy_abs_data2[param3])
aus_cloudy_abs_data2['LP2'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values>q1), aus_cloudy_abs_data2[param3])
df_abc['HP2'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values<q3_2), df_abc[param3])
df_abc['LP2'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values>q1_2), df_abc[param3])
df_clear['HP2'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values<q3_clr), df_clear[param3])
df_clear['LP2'] = ma.masked_where((df_clear['ai_clr_bbl_abs'].values>q1_clr), df_clear[param3])

aus_cloudy_abs_data2['HP3'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values<q3), aus_cloudy_abs_data2[param4])
aus_cloudy_abs_data2['LP3'] = ma.masked_where((aus_cloudy_abs_data2['ai_acc_abl_abs'].values>q1), aus_cloudy_abs_data2[param4])
df_abc['HP3'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values<q3_2), df_abc[param4])
df_abc['LP3'] = ma.masked_where((df_abc['ai_acc_bbl_abs'].values>q1_2), df_abc[param4])
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
modis_aus.boxplot([param], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['HP'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['LP'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_abc.boxplot(['HP'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_abc.boxplot(['LP'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_clear.boxplot(['HP'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
df_clear.boxplot(['LP'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)

#df_data1.boxplot(['MP'], positions=[5.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)
#df_data1.boxplot(['LP'], positions=[6], boxprops=colors_virginica, medianprops=colors_virginica, whiskerprops=colors_virginica, capprops=colors_virginica, showfliers = False, showmeans = True, ax = axes1, widths = 0.4)


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
modis_aus.boxplot([param2], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['HP1'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['LP1'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_abc.boxplot(['HP1'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_abc.boxplot(['LP1'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_clear.boxplot(['HP1'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes, widths = 0.4)
df_clear.boxplot(['LP1'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes, widths = 0.4)


axes.set_ylabel(r'$CER_{3.7µm} (µm) $', fontsize = 13)
axes.hlines(y= np.nanmedian(modis_aus[param2]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes.hlines(y= np.nanmean(modis_aus[param2]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes.set_title(r'$CER_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)

####ND####
modis_aus.boxplot([param3], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['HP2'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['LP2'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_abc.boxplot(['HP2'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_abc.boxplot(['LP2'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_clear.boxplot(['HP2'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)
df_clear.boxplot(['LP2'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes2, widths = 0.4)


axes2.set_ylabel(r'$Nd_{3.7µm} (cm^{-3})$', fontsize = 13)
axes2.hlines(y= np.nanmedian(modis_aus[param3]),xmin = 0.5, xmax = 8.5 , color = 'k', linestyles='--')
axes2.hlines(y= np.nanmean(modis_aus[param3]),xmin = 0.5, xmax = 8.5 , color = 'g', linestyles='--')
#axes1.set_ylim([0, 30])
#axes.set_title(r'$CER_{MODIS}$ (NAme ASO 2020)', fontsize = 11)
axes2.set_xticks([1, 2.5, 3.5, 4.5, 6, 7, 8])
axes2.set_xticklabels(['No Aer', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL', 'AAC_ABL', 'AAC_BBL', 'CLR_BBL'], fontsize = 8)

####LWP####
modis_aus.boxplot([param4], positions=[1], boxprops=dict(color = 'k'), medianprops=dict(color = 'k'), whiskerprops=dict(color = 'k'), capprops=dict(color = 'k'), showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['HP3'], positions=[6], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
aus_cloudy_abs_data2.boxplot(['LP3'], positions=[2.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_abc.boxplot(['HP3'], positions=[7], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_abc.boxplot(['LP3'], positions=[3.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_clear.boxplot(['HP3'], positions=[8], boxprops=colors_setosa, medianprops=colors_setosa, whiskerprops=colors_setosa, capprops=colors_setosa, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)
df_clear.boxplot(['LP3'], positions=[4.5], boxprops=colors_versicolor, medianprops=colors_versicolor, whiskerprops=colors_versicolor, capprops=colors_versicolor, showfliers = False, showmeans = True, ax = axes3, widths = 0.4)


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

fig.suptitle('Southeast Pacific SON 2019-2023', fontsize=16)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make space for the title

plt.show()
fig.savefig('/Users/devigne/Documents/THESE_2023_2026/Wildfires_Cases/SEPAC_SON_10_90.pdf', bbox_inches='tight')
plt.close()

#%%

###########################
#####CAMPAGNE AERO_HDF#####
###########################
from mpl_toolkits.mplot3d import Axes3D


# Open the dataset
météo = xr.open_dataset('/Users/devigne/Documents/THESE_2023_2026/données_AeroHdF.nc')
data_10_07 = météo.sel(time = ('2023-07-10')).mean(dim = 'time')
data_12_07 = météo.sel(time = ('2023-07-12')).mean(dim = 'time')
data_18_07 = météo.sel(time = ('2023-07-18')).mean(dim = 'time')
data_21_07 = météo.sel(time = ('2023-07-21')).mean(dim = 'time')
data_26_07 = météo.sel(time = ('2023-07-26')).mean(dim = 'time')
del météo

#temperature = xr.open_dataset('/users/devigne/Documents/THESE_2023_2026/temp.nc')
# Step 1: Averaging over the time dimension (dim 0)
ds_hdf_list = [data_10_07, data_12_07, data_18_07, data_21_07, data_26_07]

new_time = ['2023-07-10', '2023-07-12', '2023-07-18', '2023-07-21', '2023-07-26']

# Concatenate along the new time dimension
ds_hdf_mean = xr.concat(ds_hdf_list, dim=xr.DataArray(new_time, dims='time', name='time'))
print(ds_hdf_mean.data_vars)
# Verify the concatenated dataset

def change_lon(ds_hdf_mean):
    # Step 2: Shift longitude from [0, 360] to [-180, 180]
    old_lon = ds_hdf_mean['lon'].values  
    shifted_lon = np.where(old_lon > 180, old_lon - 360, old_lon)  # Shift longitudes
    ds_hdf_mean['lon'] = xr.where(ds_hdf_mean['lon'] > 180, ds_hdf_mean['lon'] - 360, ds_hdf_mean['lon'])
    # Sort the shifted longitudes and data accordingly
    sorted_indices = np.argsort(ds_hdf_mean['lon'].values)
    shifted_lon_sorted = shifted_lon[sorted_indices]
    
    # Sort the dataset along the longitude dimension to match the new longitudes
    ds_hdf_mean_sorted = ds_hdf_mean.sortby('lon') 
    return ds_hdf_mean_sorted

new_dataset = []
# Assuming the datasets are named as in your previous example
datasets = [data_10_07, data_12_07, data_18_07, data_21_07, data_26_07]
for dataset in datasets:
    new_dataset.append(change_lon(dataset))
# Define the target latitudes and longitudes for regridding
# Replace target_lat and target_lon with the appropriate arrays
# Example: target_lat = np.linspace(-90, 90, 180), target_lon = np.linspace(-180, 180, 360)

all_regridded_data = []

for dataset in new_dataset:
    variables = []
    
    old_lat = dataset['lat'].values  # Adjust based on the actual latitude name
    old_lon = dataset['lon'].values  # Assuming 'lon' is the name of the longitude dimension
    
    # Loop over each variable in the dataset
    for var_name in dataset.data_vars:
        var = dataset[var_name].values  # Get the variable's data
        regridded_var = np.empty((25, 180, 360))  # Assuming 25 pressure levels and target grid of 180x360
        
        # Regrid each pressure level individually
        for i in range(25):
            regridded_var[i, :, :] = regrid_2d(var[i, :, :], old_lat, old_lon, target_lat, target_lon)
        
        variables.append(regridded_var)
    
    # Stack all variables for this particular day
    day_regridded_data = np.stack(variables, axis=0)  # Shape: (number_of_vars, 25, 180, 360)
    
    # Store regridded data for this day
    all_regridded_data.append(day_regridded_data)

# `all_regridded_data` now contains the regridded data for each day

target_lat = np.linspace(-90, 90, 180)  # 180 latitude points from -90 to 90
target_lon = np.linspace(-180, 180, 360)  # 360 longitude points from -180 to 180

date_hdf = ['2023-07-10', '2023-07-12', '2023-07-18', '2023-07-21', '2023-07-26']
dataset_aerohdf = []
for i in range(5):
    # Step 5: Create a new xarray dataset with all 5 new products using the regridded lat and lon
    new_var_hdf = xr.Dataset(
        {
            'Temperature': (['plev', 'lat', 'lon'], all_regridded_data[i][0,:,:,:]),  # Vertical temperature profile (K)
            'Vertical Velocity': (['plev', 'lat', 'lon'], all_regridded_data[i][1,:,:,:]),  # Vertical velocity profile (m/s)
            'Relative Humidity': (['plev', 'lat', 'lon'], all_regridded_data[i][2,:,:,:]),  # Relative Humidity vertical profile (%)
            
        },
        coords={
            'plev': ds_mean_sorted['plev'].values,  
            'lat': target_lat,  # Use the new regridded latitude coordinates
            'lon': target_lon   # Use the new regridded longitude coordinates
        }
    )
    new_var_hdf.to_netcdf(f'/Users/devigne/Documents/THESE_2023_2026/données_météorologiques_Aero_HdF_{date_hdf[i]}.nc')
    dataset_aerohdf.append(new_var_hdf)
    

for i in range(5):    
    ds = dataset_aerohdf[i]
    ds = ds.sel(plev=slice(None, None, -1))
    
    # Choose a subset of latitudes and longitudes to avoid overcrowding
    lat_min, lat_max = 37, 44  # Focus on a tropical band
    lon_min, lon_max = -2, 12  # Longitude range
    
    # Extract the indices for the selected latitude and longitude range
    lat_indices = np.where((ds['lat'].values >= lat_min) & (ds['lat'].values <= lat_max))[0]
    lon_indices = np.where((ds['lon'].values >= lon_min) & (ds['lon'].values <= lon_max))[0]
    
    # Subset temperature data for these latitudes and longitudes
    temperature_subset = ds['Temperature'][:, lat_indices, lon_indices].values  # Shape: (plev, lat, lon)
    plev = ds['plev'].values  # Pressure levels
    lat_subset = ds['lat'][lat_indices].values
    lon_subset = ds['lon'][lon_indices].values
    
    # Create a 3D meshgrid for lon, lat, and plev
    plev_grid, lat_grid, lon_grid = np.meshgrid(plev, lat_subset, lon_subset, indexing='ij')
    
    # Flatten the arrays for scatter plotting
    lon_flat = lon_grid
    lat_flat = lat_grid
    plev_flat = plev_grid
    temp_flat = temperature_subset
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with temperature values mapped to colors
    scatter = ax.scatter(lon_flat, lat_flat, plev_flat, c=temp_flat, cmap='coolwarm', marker='o')
    
    # Add color bar and axis labels
    fig.colorbar(scatter, ax=ax, label='Temperature (K)')
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_zlabel('Pressure (Pa)')
    ax.set_title(f'Temperature Profile {date_hdf[i]} Mediterranean Sea')
    
    # Invert the z-axis to show pressure decreasing with height
    ax.invert_zaxis()
    
    plt.show()
    plt.savefig(f'/Users/devigne/Documents/THESE_2023_2026/Temp_{date_hdf[i]}_Med.pdf')



for i in range(5):    
    ds = dataset_aerohdf[i]
    ds = ds.sel(plev=slice(None, None, -1))
    
    # Choose a subset of latitudes and longitudes to avoid overcrowding
    lat_min, lat_max = 30, 65  # Focus on a tropical band
    lon_min, lon_max = -25, 28  # Longitude range
    
    # Extract the indices for the selected latitude and longitude range
    lat_indices = np.where((ds['lat'].values >= lat_min) & (ds['lat'].values <= lat_max))[0]
    lon_indices = np.where((ds['lon'].values >= lon_min) & (ds['lon'].values <= lon_max))[0]
    
    # Subset temperature data for these latitudes and longitudes
    lat_subset = ds['lat'][lat_indices].values
    lon_subset = ds['lon'][lon_indices].values
    
    # Check temperature at specific pressure levels
    pressure_level_1 = 100000  # Surface pressure (Pa)
    pressure_level_2 = 95000  # Mid-troposphere (Pa)
    
    temp_level_1 = ds['Vertical Velocity'].sel(plev=pressure_level_1, method='nearest')
    temp_level_2 = ds['Vertical Velocity'].sel(plev=pressure_level_2, method='nearest')
    
    # Extract temperature values for the subset
    temperature_subset_1 = temp_level_1[lat_indices, lon_indices].values  # Shape: (lat, lon)
    temperature_subset_2 = temp_level_2[lat_indices, lon_indices].values  # Shape: (lat, lon)

    # Plot the temperature maps at these pressure levels
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    
    # Plot for the first pressure level
    im1 = axs[0].imshow(temperature_subset_1, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower', cmap='coolwarm')
    axs[0].set_title(f'Vertical Velocity at {pressure_level_1} Pa on {date_hdf[i]}')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axs[0])

    # Plot for the second pressure level
    im2 = axs[1].imshow(temperature_subset_2, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower', cmap='coolwarm')
    axs[1].set_title(f'Vertical Velocity at {pressure_level_2} Pa on {date_hdf[i]}')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axs[1])
    
    plt.tight_layout()
    plt.savefig(f'/Users/devigne/Documents/THESE_2023_2026/W_surface_{date_hdf[i]}_D2.pdf')
    plt.show()
    
# Select a single latitude and longitude
lat_sel = 10  # Adjust as needed
lon_sel = 0  # Adjust as needed

# Get the temperature profile at this location
temp_profile = ds['Temperature'].sel(lat=lat_sel, lon=lon_sel).values

# Plot the temperature profile vs. pressure
plt.plot(temp_profile, ds['plev'].values)
plt.gca().invert_yaxis()  # Invert the y-axis for pressure
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (Pa)')
plt.title(f'Temperature Profile at lat={lat_sel}°, lon={lon_sel}°')
plt.show()


#%%
from mpl_toolkits.basemap import Basemap
#############################
#Aero_HdF aerosols Databasis#
#############################

data_files_hdf = ['/Users/devigne/Documents/THESE_2023_2026/Aero_hdf_Data_Files/aero_hdf_2023-07-10.csv', '/Users/devigne/Documents/THESE_2023_2026/Aero_hdf_Data_Files/aero_hdf_2023-07-12.csv', '/Users/devigne/Documents/THESE_2023_2026/Aero_hdf_Data_Files/aero_hdf_2023-07-18.csv', '/Users/devigne/Documents/THESE_2023_2026/Aero_hdf_Data_Files/aero_hdf_2023-07-21.csv', '/Users/devigne/Documents/THESE_2023_2026/Aero_hdf_Data_Files/aero_hdf_2023-07-26.csv']

data_files_hdf
data_list_aero_hdf = [pd.read_csv(file, index_col='date') for file in data_files_hdf]



data_aero_hdf = []
# Main processing loop
for df in data_list_aero_hdf:
    aus_cloudy_abs_data2 = df
    date = aus_cloudy_abs_data2.index.values[0]
    # Convert index to datetime
    aus_cloudy_abs_data2.index = pd.to_datetime(aus_cloudy_abs_data2.index)
    
    # Calculate new columns
    aus_cloudy_abs_data2['diff_CER'] = (aus_cloudy_abs_data2['Cloud_Effective_Radius_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Effective_Radius_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Effective_Radius_16_Liquid_Mean']
    aus_cloudy_abs_data2['diff_COT'] = (aus_cloudy_abs_data2['Cloud_Optical_Thickness_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Optical_Thickness_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Optical_Thickness_16_Liquid_Mean']
    aus_cloudy_abs_data2['diff_Nd'] = (aus_cloudy_abs_data2['Nd_G18_16'] - aus_cloudy_abs_data2['Nd_G18_37'])/aus_cloudy_abs_data2['Nd_G18_16']

    # Adjust Cloud Water Path values and mask negative values
    for col in ['Cloud_Water_Path_37_Liquid_Mean', 'Cloud_Water_Path_16_Liquid_Mean']:
        aus_cloudy_abs_data2[col] *= 100
        aus_cloudy_abs_data2[col] = aus_cloudy_abs_data2[col].mask(aus_cloudy_abs_data2[col] < 0)

    # Calculate the difference in Cloud Water Path
    aus_cloudy_abs_data2['diff_LWP'] = (aus_cloudy_abs_data2['Cloud_Water_Path_16_Liquid_Mean'] - aus_cloudy_abs_data2['Cloud_Water_Path_37_Liquid_Mean'])/aus_cloudy_abs_data2['Cloud_Water_Path_16_Liquid_Mean']

    # Filter data based on aerosol index and cloud top temperature
    modis_aus = aus_cloudy_abs_data2[
        (aus_cloudy_abs_data2['ai_clr_bbl_abs'] <= 0.05) |
        (aus_cloudy_abs_data2['ai_acc_abl_abs'] <= 0.05) |
        (aus_cloudy_abs_data2['ai_acc_bbl_abs'] <= 0.05) |
        aus_cloudy_abs_data2[['ai_acc_abl_abs', 'ai_clr_bbl_abs', 'ai_acc_bbl_abs']].isna().any(axis=1)
    ]
    
    data_aero_hdf.append(aus_cloudy_abs_data2)

# Concatenate all DataFrames along axis 0
final_df_aero_hdf = pd.concat(data_aero_hdf, axis=0)

date_hdf = ['2023-07-10', '2023-07-12', '2023-07-18', '2023-07-21', '2023-07-26']
list_var = ['ai_acc_abl_abs', 'ai_acc_bbl_abs', 'ai_clr_bbl_abs']
list_days = [10, 12, 18, 21, 26]

# Set map boundaries
lon_min, lon_max = -25, 28  # Longitude range
lat_min, lat_max = 30, 65  # Latitude range
x = 53
y = 35
for i in range(5):
    for var in list_var:
        # Example data (replace with your own dataframe)
        df = pd.DataFrame(final_df_aero_hdf[final_df_aero_hdf.index.day.isin([list_days[i]])])
        
        df = df[(df['latitude']>=lat_min) & (df['latitude']<=lat_max) & (df['longitude']>=lon_min) & (df['longitude']<=lon_max)]
        AI = df[var].values
        AI = AI.reshape((53,35)).T
        # Plot 1
        fig = plt.figure(figsize=(12, 12))
        my_map1 = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max)
        my_map1.drawcoastlines(linewidth=0.5)
        my_map1.drawstates()
        my_map1.drawparallels(np.arange(lat_min, lat_max, 15), labels=[True, False, False, True])
        my_map1.drawmeridians(np.arange(lon_min, lon_max, 15), labels=[True, False, False, True])
        longitude1, latitude1 = my_map1.makegrid(53, 35)
        
        x, y = my_map1(longitude1, latitude1)
        cs1 = my_map1.pcolormesh(x, y, AI, cmap='coolwarm', vmin = 0, vmax = 4.2)
        my_map1.colorbar(cs1, label=f'{var}')
        #my_map1.set_ticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
        plt.title(f'{var} Domain 2 {date_hdf[i]}')
        plt.savefig(f'/Users/devigne/Documents/THESE_2023_2026/Études_Aero_HdF/{var}_{date_hdf[i]}_D2.pdf')
        plt.show()


