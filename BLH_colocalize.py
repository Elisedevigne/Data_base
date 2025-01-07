#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:00:01 2024

@author: devigne
"""
import numpy as np
import xarray as xr
#from scipy.interpolate import interp2d, RectBivariateSpline
#import netCDF4
#import pandas as pd
import os
#import fnmatch
#import glob
#from pyhdf.SD import SD, SDC
from scipy.interpolate import RegularGridInterpolator

#os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from datetime import datetime, timedelta
import pandas as pd
def convert_utc_to_local(utc_time, longitude):
    # Calculate the time difference in hours
    time_difference = longitude / 15  # 15 degrees of longitude corresponds to 1 hour difference in time
    # Adjust the UTC time
    local_time = utc_time + timedelta(hours=time_difference)
    return local_time

## Test for one dayy in January 2020##



def blh(var, date_file):
    path = '/LARGE/ECMWF/EA/AN/OPER/0.25x0.25/%s/ea_%s%s%s_BLH.nc4'%(date_file[0:4], date_file[0:4], date_file[5:7], date_file[8:10])
    day_1 = datetime.strptime(date_file , '%Y-%m-%d') + timedelta(days=1)
    day = datetime.strptime(date_file , '%Y-%m-%d')
    path2 = '/LARGE/ECMWF/EA/AN/OPER/0.25x0.25/%s/ea_%s%s%s_BLH.nc4' % (
        str(day_1.year), 
        str(day_1.year), 
        str(day_1.strftime('%m')),  # Get the month as a zero-padded string
        str(day_1.strftime('%d'))   # Get the day as a zero-padded string
    )
    stop_datetime = datetime(day_1.year, day_1.month, day_1.day, 2,0,0)
    start_datetime = datetime(day.year, day.month, day.day, 2,0,0)
    print("Trying to process", path)
    try:
        ds_blh1 = xr.open_dataset(path)
        ds_blh2 = xr.open_dataset(path2)
        ds_blh = xr.concat([ds_blh1, ds_blh2], dim='time')
        # Convert values greater than 180° to the range -180° to 180°
        ds_blh['longitude'] = xr.where(ds_blh['longitude'] > 180, ds_blh['longitude'] - 360, ds_blh['longitude'])
        
        # Extract latitude and longitude coordinates
        # Define the new grid with higher resolution
        # Define new latitude and longitude coordinates with higher resolution
        # Define the new latitude and longitude coordinates with higher resolution
        new_lat = np.linspace(ds_blh['latitude'].min(), ds_blh['latitude'].max(), 8192)
        new_lon = np.linspace(ds_blh['longitude'].min(), ds_blh['longitude'].max(), 16384)
        # Create an empty DataArray to store the interpolated values
        interpolated_data = np.zeros((ds_blh.sizes['time'], 8192, 16384))
        
        # Interpolate the dataset onto the new grid for each time step
        for i in range(ds_blh.sizes['time']):
            interpolated_data[i,:,:] = np.array(ds_blh['blh'][i].interp(latitude=new_lat, longitude=new_lon, method = 'nearest'))
            print('interpolation indice : ',i)
        # Create a new xarray dataset with the interpolated data
        interpolated_ds = xr.Dataset({'blh': (('time', 'latitude', 'longitude'), interpolated_data)},
                                     coords={'time': ds_blh['time'], 'latitude': new_lat, 'longitude': new_lon})
        # After interpolation, create a new DataArray with the interpolated values
        longitudes = [-180, -165, -150, -135, -120, -105, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]  # Example longitudes
        sliced_arrays = []
        
        for time_step in interpolated_ds['time']:
            
            # Extract the datetime value for the current time step
            timestamp = time_step.values.astype('datetime64[s]').astype(int)
            time_step_dt = datetime.utcfromtimestamp(timestamp)
            if time_step_dt >= start_datetime and time_step_dt < stop_datetime:
            # Perform actions for the current time step
                
            # Convert UTC time to local time for each longitude
                local_times = {}
                for longitude in longitudes:
                    
                    local_time = convert_utc_to_local(time_step_dt, longitude)
                    local_times[longitude] = local_time
                # Define the specific daytime
                specific_date = datetime.strptime(date_file, '%Y-%m-%d')
                # Define the UTC times corresponding to the desired local time range (13:00 to 14:00, exclusive) for the specific day
                utc_time_13 = datetime(specific_date.year, specific_date.month, specific_date.day, 13, 0, 0)
                utc_time_14 = datetime(specific_date.year, specific_date.month, specific_date.day, 14, 0, 0)
                # Find the longitude closest to the desired local times
                selected_longitude_13 = np.nan
                selected_longitude_14 = np.nan
                min_diff_13 = float('inf')
                min_diff_14 = float('inf')
                
                for longitude, timestamp in local_times.items():
                    
                    diff_13 = abs((timestamp - utc_time_13).total_seconds())
                    diff_14 = abs((timestamp - utc_time_14).total_seconds())
                    if diff_13 < min_diff_13:
                        min_diff_13 = diff_13
                        selected_longitude_13 = longitude
                    if diff_14 < min_diff_14:
                        min_diff_14 = diff_14
                        selected_longitude_14 = longitude    
                print(selected_longitude_13, selected_longitude_14)
                # Filter the dataset to keep only the part within the desired local time range (13:00 to 14:00)
                # Filter the dataset to keep only the part within the desired local time range (13:00 to 14:00)
                filtered_ds = interpolated_ds.sel(time=time_step)
                filtered_ds = filtered_ds.sel(longitude=slice(selected_longitude_13, selected_longitude_14))
                # Append the array values
                sliced_arrays.append(np.flip(filtered_ds['blh'].values, axis = 1))
            # Append the array values
        reordered_reconstructed_array = np.concatenate(sliced_arrays, axis=1)
        # Return the reordered reconstructed array
        return np.flip(reordered_reconstructed_array, axis = 1)
   
    except FileNotFoundError:
        # Handle the case where one or both files do not exist
        if not os.path.exists(path):
            print(f"File not found for date: {date_file}")
        if not os.path.exists(path2):
            print(f"File not found for date: {day_1.strftime('%Y-%m-%d')}")

        return None  # or perform alternative handling as needed

    except Exception as e:
        # Handle any other exceptions that may occur
        print(f"Error occurred while opening dataset: {e}")
        return None  # or perform alternative handling as needed
"""
date_file = '2020-01-01'
new_blh = blh('blh', date_file)

data_blh = new_blh
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
my_map1 = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
my_map1.drawcoastlines(linewidth=0.5)
my_map1.drawstates()
my_map1.drawparallels(np.arange(-90, 91, 45), labels=[True, False, False, True])
my_map1.drawmeridians(np.arange(-180, 181, 45), labels=[True, False, False, True])
longitude1, latitude1 = my_map1.makegrid(16384, 8192)
x, y = my_map1(longitude1, latitude1)
cs1 = my_map1.pcolormesh(x, y, data_blh, cmap="Reds")
my_map1.colorbar(cs1, label='BLH (m)')
plt.title('BLH')
plt.tight_layout()
# Save or show plot
plt.savefig('/home/devigne/January_blh_tropomi.png')

"""

