# TempestD_python_read_example.py
# Michael Cheeseman
# 06/29/2022
#
# Got information on Tempest H5 files here: (https://tempest.colostate.edu/TEMPEST-D_Level_1_Data_Description_05sep19.pdf)
# Additional information on satellite mission here at these links:
# (https://directory.eoportal.org/web/eoportal/satellite-missions/t/tempest-d)
# (https://digitalcommons.usu.edu/cgi/viewcontent.cgi?referer=&httpsredir=1&article=3609&context=smallsat)
# (https://www.jpl.nasa.gov/news/an-inside-look-at-hurricane-dorian-from-a-mini-satellite)
#
# File description:
# The TEMPEST‐D data are calibrated, geolocated and output into an HDF5 file once per day. The data are
# stored in a structure called scan. The “scan” structure indicates that the data are organized in a 2‐D
# array where each row is a complete scan. The 2‐D variables are stored as Nscan x Nbeam, where Nscan
# is the number of scans in the file and Nbeam is the number of samples cross track within each scan.
# Nscan will be variable depending on the amount of data in the file, and Nbeam is typically 133. The
# antenna temperature and brightness temperature data are stored as a 3‐D array in which the last
# dimension corresponds to the 5 radiometer channels. The spacecraft position and attitude data are
# each stored as 1‐D arrays of length Nscan.
# ------------------------------------------------------------------------------------------------

# ---------------------------------------
# Uploads
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import h5py
from netCDF4 import num2date
import datetime as dt
plt.close('all')

# ---------------------------------------
# Input/Output Directories and User Variables
out_fp = './figures_TempestD/'

# ---------------------------------------
# Open Tempest L1 data
f = h5py.File('/home/madison/Python/NOAA_Lapenta_Research/TEMPEST_Data/TEMPEST_L1_pub_20190802T163128_20190802T235920_v2.0.h5','r')
#print(list(f.keys()))
scan_dat = f.get('scan') # data is in structure called "scan"
# Print the contents of file
for fkey in scan_dat.keys():
    # Print out information in the file
    print('')
    print('------------------------------')
    print(fkey)
    print(scan_dat[fkey].attrs['Description'].decode("utf-8"))
    #print(scan_dat[fkey].attrs['Units'].decode("utf-8"))
    print(scan_dat[fkey].shape)
# Open Scan data (NOTE: CH1=181GHz,CH2=178GHz,CH3=174GHz,CH4=164GHz,CH5=87GHz)
chans = np.array([181,178,174,164,87])
tb = scan_dat['TB'][:] # Calibrated Brightness Temp [5 radiometer channels x Nscan x Nbeam]
ta = scan_dat['TB'][:] # Calibrated Antenna Temp [5 radiometer channels x Nscan x Nbeam]
temp_time_utc = scan_dat['UTCtime'][:] # UTC time of each sample [Nscan x Nbeam]
temp_time_utc_units = 'seconds since 1‐1‐2000 00:00:00' # NOTE: According to documentation?
sclat = scan_dat['SClat'][0,:] # Sub-spacecraft latitude
sclon = scan_dat['SClon'][0,:] # Sub-spacecraft longitude
scalt = scan_dat['SCalt'][0,:] # Sub-spacecraft altitude
blat = scan_dat['blat'][:] # Boresight latitude at radiomete sample rate
blon = scan_dat['blon'][:] # Boresight lonitude at radiomete sample rate
scanang = scan_dat['scanang'][:] # Scan angle from encoder
f.close()

# ---------------------------------------
# Replace nans with -999 and then mask
tb[np.isnan(tb)==True] = -999
tb = np.ma.masked_values(tb,tb.min())

##########################################################################################
# ---------------------------------------
# Plot scan of data from TEMPEST-D with boresight
#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.basemap import shiftgrid
from cartopy import config
import cartopy.crs as ccrs
#for i in range(5):
for i in range(1):
    # Plot using CARTOPY
    pl.figure(figsize=(5,4))
    x,y = blon.ravel(),blat.ravel()
    z = tb[i,:,:].ravel()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_title('%i GHz Channel'%(chans[i]),fontsize=14)
    ax.coastlines()
    sc=plt.scatter(x,y,c=z,s=.01,vmin=200,vmax=300,
                 transform=ccrs.PlateCarree())
    cbar=pl.colorbar(sc,orientation='horizontal',shrink=0.9)
    cbar.set_label(label='Brightness Temp (K)',fontsize=12)
    outname = 'TempestD-%i-scan-TB'%chans[i]
    pl.tight_layout()
    #pl.savefig(out_fp+outname,bbox_to_inches='tight')
    #pl.close('all')

# ---------------------------------------
# Plot histograms of data from various radiometers
test_dat = tb.reshape(tb.shape[0],tb.shape[1]*tb.shape[2])
fig,ax=pl.subplots(figsize=(4,3))
pl.title('Distribution of each Radiometer', fontsize=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i in range(test_dat.shape[0]):
    pl.hist(test_dat[i],bins=np.arange(200,300,1),alpha=0.6,label='%i GHz Channel'%(chans[i]))
pl.legend()
pl.xlabel('Brightness Temperature (K)',fontsize=12)
pl.tight_layout()
outname='TempestD_radiometer_Tb_distributions'
#pl.savefig(out_fp+outname,bbox_to_inches='tight')
#pl.close('all')
pl.show()

