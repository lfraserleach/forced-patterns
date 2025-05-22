# %%
#load packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt
import os
import pickle
import cartopy.crs as ccrs
from snplib import load_2d_ts, flatten_ts

# %%
# USER INPUT
plt.style.use('prb')
model = 'NorESM2-LM'
exp_c = 'ssp370-ramip'
exp_p = 'ssp370-126aer'
reload_data = False
recalculate_eofs = False
datadir = '/project/p/pjk/lfl/ramip'
ensdir_c = f"{datadir}/{model}/{exp_c}"
ensdir_p = f"{datadir}/{model}/{exp_p}"
outputdir = '/project/p/pjk/lfl/ramip/snp_data' # directory for saving Pickle files (copy files and change to a directory you have write access)
plots_dir = '/home/p/pjk/lfl/ramip/snp_plots' # directory for saving plots (copy files and change to a directory you have write access)
name = f"{model}-{exp_p}-{exp_c}" # name used in Pickle files
table = 'Amon' # CMOR table name
varnam = 'tas' # set to ts for full ts field, ts50 for 50°S to 50°N
decades = [2015, 2020, 2030, 2040] # decades to use for analysis
T = np.arange(2015,2050,1/12) # historical simulations start in 1920 for CESM, 1850 for MPI (first input to arrange must match)
# Second input to arrange can be chosen to select period of interest, but change name used in pickle files for end-dates other than 2006

# %%
# Preprocess SST data (or load from Pickle file)

# def load_2d_ts(
#         varnam, table, exp_p,
#         exp_c='ssp370-ramip', memi=4, memf=10 model='NorESM2-LM',
#         decades=[2015, 2020, 2030, 2040], datadir='/project/p/pjk/lfl/ramip',
#         outputdir='/project/p/pjk/lfl/ramip/snp_data', reload_data=False
#         ):
memi = 4
memf = 10
table = 'Omon'
varnam = 'tos'
T=np.arange(2015,2050,1/12)
name = f"{model}-{exp_p}-{exp_c}"

try:
    if reload_data:
        raise Exception('reload_data is True, so model output will be '\
        + 're-loaded and pickled')
    # load pre-processed SST data from Pickle
    ts_all = pickle.load(open(f"{outputdir}/{name}_{varnam}_all.p", "rb" ))
    ts_clim_all = pickle.load(open(f"{outputdir}/{name}_{varnam}_clim_all.p", "rb" ))
    lat = ts_all.lat
    lon = ts_all.lon
    time = ts_all.time
        
except:
    # preprocess SST data and save to Pickle
    # get data files
    members = [f"r{i}i1p1f1" for i in range(memi,memf+1)]
    n = len(members)
    ne = np.empty(n)
    # define axes
    ds0list = []
    for dec in decades:
        deci = dec
        decf = dec + 9 - dec%10
        filename0 = f"{ensdir_c}/{members[0]}/{table}/{varnam}/gn/v20230810/"\
            + f"{varnam}_{table}_{model}_{exp_c}_{members[0]}_gn_{deci}01-{decf}12.nc"
        ds0dec = xr.open_dataset(filename0)
        ds0list.append(ds0dec)
    ds0 = xr.concat(ds0list, dim="time")
    if table in ['Omon', 'SImon']:
        lat = ds0.latitude
        lon = ds0.longitude
    else:
        lat = ts_all.lat
        lon = ts_all.lon
    time = ds0.time
    nt = len(time)
    nt_cut = len(T)
    month = np.linspace(1, 12, 12)

    ts_all = np.empty((n,nt_cut,len(lat),len(lon)))
    ts_clim_all = np.empty((n,12,len(lat),len(lon)))
    time = time[0:nt_cut]
    if table in ['Omon', 'SImon']:
        ts_all = xr.DataArray(ts_all, coords=[ne, time, lat, lon], 
            dims=["member", "time", "latitude", "longitude"])
        ts_clim_all = xr.DataArray(ts_clim_all, coords=[ne, month, lat, lon], 
            dims=["member", "month", "latitude", "longitude"])
    else:
        ts_all = xr.DataArray(ts_all, coords=[ne, time, lat, lon], 
            dims=["member", "time", "lat", "lon"])
        ts_clim_all = xr.DataArray(ts_clim_all, coords=[ne, month, lat, lon], 
            dims=["member", "month", "lat", "lon"])
    # concatenate all ensemble members into one dataset
    for ii, member in enumerate(members):
        print(ii)
        dslist_c = []
        dslist_p = []
        for dec in decades:
            deci = dec
            decf = dec + 9 - dec%10
            filename_c = f"{ensdir_c}/{member}/{table}/{varnam}/gn/v20230810/"\
                + f"{varnam}_{table}_{model}_{exp_c}_{member}_gn_{deci}01-{decf}12.nc"
            filename_p = f"{ensdir_p}/{member}/{table}/{varnam}/gn/v20230810/"\
                + f"{varnam}_{table}_{model}_{exp_p}_{member}_gn_{deci}01-{decf}12.nc"
            dsdec_c = xr.open_dataset(filename_c)
            dsdec_p = xr.open_dataset(filename_p)
            dslist_c.append(dsdec_c)
            dslist_p.append(dsdec_p)
        ds_member_c = xr.concat(dslist_c, dim="time")
        ds_member_p = xr.concat(dslist_p, dim="time")

        ts = ds_member_p[varnam][-nt:,:,:] - ds_member_c[varnam][-nt:,:,:]
        ts_clim = ts.groupby('time.month').mean('time')
        ts_anom = ts.groupby('time.month')-ts_clim
        ts_all[ii,:,:,:] = ts_anom
        ts_clim_all[ii,:,:,:] = ts_clim
        
    pickle.dump(ts_all, open(f"{outputdir}/{name}_{varnam}_all.p", "wb" ),protocol=4)
    pickle.dump(ts_clim_all, open(f"{outputdir}/{name}_{varnam}_clim_all.p", "wb" ))

#     return ts_all, ts_clim_all, lat, lon, time

# ts_all, ts_clim_all = load_2d_ts('tos', 'Omon', 'ssp370-126aer')

ne = ts_all.member
nt = len(ts_all.time)

# %%
# sanity check plot, just shows changes in temperature over the simulation

field = ts_all.values
field = np.mean(field,axis=0)
field_diff = np.mean(field[912:1031,:,:],axis=0)-np.mean(field[0:119,:,:],axis=0)
field_diff = (ts_all.sel(time=slice('2040-01-01','2049-01-01')).mean(dim='time')-ts_all.sel(time=slice('2015-01-01','2020-01-01')).mean(dim='time')).mean(dim='member')
f=plt.figure()
plt.contourf(ts_all.lon.values,ts_all.lat.values,field_diff,np.arange(-3,3.1,0.1),cmap=plt.cm.RdBu_r)
cbar = plt.colorbar()

# %%
# # Preprocessing for Large Ensemble EOFs

# lon = ts_all.lon
# lat = ts_all.lat
# cosw = np.sqrt(np.cos(lat*np.pi/180))
# normvec  = cosw/np.sum(cosw);
# scale = np.sqrt(normvec);

# X=ts_all*scale

# X_ensmean=X.mean('member')
# X_flat = X.stack(index=['time','member']).stack(shape=['lat','lon'])
# X_ensmean_flat = X_ensmean.stack(shape=['lat','lon'])

# # keep unscaled copies of these variables
# Xt_ensmean=ts_all.mean('member')
# Xt_flat = ts_all.stack(index=['time','member']).stack(shape=['lat','lon'])
# Xt_ensmean_flat = Xt_ensmean.stack(shape=['lat','lon'])

X_flat, X_ensmean_flat, Xt_flat, Xt_ensmean_flat = flatten_ts(ts_all)

index = X_flat.index
n = len(index)

# %%time
# Perform ensemble EOF analysis (takes a few minutes), or load from Pickle if it has already been done

try:
    if recalculate_eofs:
        raise Exception('recalculate_eofs is True, so EOFs will be '\
        + 're-loaded and pickled')
    # load PCA output from Pickle
    pcvec,evl = pickle.load(open(f"{outputdir}/{name}_{varnam}_EIG.p", "rb" ))

except:
    # Large Ensemble EOFs
    Cov = np.matmul(X_flat.values.T,X_flat.values)/(n-1)
    print('computing eigenvalues (this may take about 15 minutes)...')
    evl,pcvec = np.linalg.eig(Cov)
    print('done')
    pickle.dump([pcvec,evl], open(f"{outputdir}/{name}_{varnam}_EIG.p", "wb" ),protocol=4)
    
s=np.sqrt(evl)

# %%
# S/N Maximizing (Forced) Pattern analysis

neof=200 # number of EOFs retained in S/N maximizing pattern analysis

# Large Ensemble Forced Patterns

S=np.matmul(pcvec[:,0:neof],np.diag(1/s[0:neof]))
Sadj=np.matmul(np.diag(s[0:neof]),pcvec[:,0:neof].T)

ensmeanPCs=np.matmul(X_ensmean_flat.values,S) # ensemble-mean principal components

gamma=np.cov(ensmeanPCs.T)  # covariance matrix of ensemble-mean principal components

u2,signal_frac,v2=np.linalg.svd(gamma)

SNP=np.matmul(v2,Sadj)
SNPs_reshaped=SNP.reshape(neof,len(lat),len(lon))/scale.values[None,:,None]

weights = np.matmul(S,v2.T)
weights = weights.reshape(len(lat),len(lon),neof)*scale.values[:,None,None]

weights=weights.reshape(len(lat)*len(lon),neof)

tk = np.matmul(Xt_flat.values,weights) # compute timeseries from full data matrix
tk_emean = np.matmul(Xt_ensmean_flat.values,weights) # compute ensemble-mean timeseries from ensemble-mean data

sign_eof = np.ones((neof,1))

for ii in range(neof):
    if np.mean(SNP[ii,:]) < 0:
        SNPs_reshaped[ii,:,:] = -SNPs_reshaped[ii,:,:]
        SNP[ii,:] = -SNP[ii,:]
        tk[:,ii] = -tk[:,ii]
        tk_emean[:,ii] = -tk_emean[:,ii]
        sign_eof[ii] = -1
        
pickle.dump([tk,tk_emean,SNPs_reshaped,weights,signal_frac], open(f"{outputdir}/{name}_{varnam}_SNP{str(neof)}.p", "wb" ),protocol=4)

# %%
print(signal_frac[0:30])
plt.plot(signal_frac,marker='o')
plt.xlim(0,30)
plt.title('Signal Fraction')
plt.savefig(f'{plots_dir}/signal_fraction.pdf')

#signal_frac_check = np.zeros(31)
#for ii in range(31): 
#    signal_frac_check[ii] = np.mean(np.square(tk_emean[:,ii]))/np.mean(np.square(tk[:,ii]))
    
#print(signal_frac_check)
#f=plt.figure()
#plt.plot(signal_frac_check,marker='o')
#plt.xlim(0,30)
#plt.title('Signal Fraction Check')

# %%
# Plot S/N maximizing patterns (SNPs)
Ne = 3
nrows = 1
ncols = Ne
min_tas = -1.5
max_tas = 1.5
step_tas = 0.15
levels_tas = np.arange(min_tas,max_tas+step_tas,step_tas)
cmap_tas = plt.cm.RdBu_r
proj = ccrs.PlateCarree(central_longitude=180)
f, ax = plt.subplots(nrows,ncols,figsize=(3.5*ncols,1.8*nrows),
    subplot_kw={'projection': proj},constrained_layout=True)
for neof_plot in range(Ne):
    ctr = ax[neof_plot].contourf(lon.values,lat.values,
        np.squeeze(SNPs_reshaped[neof_plot,:,:]),
        np.arange(min_tas,max_tas,step_tas),cmap=plt.cm.RdBu_r,
        transform=ccrs.PlateCarree())
    [ax[neof_plot].plot(T,tk_reshape[:,mm,neof_plot],color='crimson') for mm in range(7)];
    ax[neof_plot].plot(T,tk_emean[:,neof_plot])
    ax[neof_plot].set_title(f'SNP{neof_plot+1}')
    # cbar = ax.colorbar()
    ax[neof_plot].coastlines()
f.colorbar(ctr, label='$\delta{T} [K]$')
f.savefig(f'{plots_dir}/snp_maps.pdf')

# %%
# Plot forced pattern timeseries
tk_reshape=tk.reshape(nt,len(ne),neof)
f, ax = plt.subplots(nrows,ncols,figsize=(3.5*ncols,1*nrows),sharey=True,
    constrained_layout=True)
for neof_plot in range(Ne): 
    [ax[neof_plot].plot(T,tk_reshape[:,mm,neof_plot],color='crimson') for mm in range(7)];
    ax[neof_plot].plot(T,tk_emean[:,neof_plot])
    # ax[neof_plot].title('SNP'+str(neof_plot+1))
f.savefig(f'{plots_dir}/snp_timeseries.pdf')

# %%
# Forced component from leading forced patterns

M = 13  # number of forced patterns to retain, choose cutoff based on eigenvalue spectrum, or check significant patterns with bootstrapping

X_forced = np.matmul(tk_emean[:,0:M],SNPs_reshaped[0:M,:,:].reshape(M,len(lat)*len(lon)))
X_forced = X_forced.reshape(len(T),len(lat),len(lon))
# X_forced_land = X_forced*landmask[None,:,:]
X_forced = xr.DataArray(X_forced, coords=[T,lat,lon], dims=["time","lat","lon"])
# X_forced_land = xr.DataArray(X_forced_land, coords=[T,lat,lon], dims=["time","lat","lon"])
# Xt_ensmean_land = Xt_ensmean*landmask

GMST_forced = X_forced.mean('lon').mean('lat')
GMST_ensmean = Xt_ensmean.mean('lon').mean('lat')

#Arctic_forced = X_forced.sel(lat=slice(90,65)).mean('lon').mean('lat')
#Arctic_ensmean = Xt_ensmean.sel(lat=slice(90,65)).mean('lon').mean('lat')

# tropical_land_forced = X_forced_land.sel(lat=slice(10,-10)).mean('lon').mean('lat')
# tropical_land_ensmean = Xt_ensmean_land.sel(lat=slice(10,-10)).mean('lon').mean('lat')

# US_land_forced = X_forced_land.sel(lon=slice(235,295),lat=slice(45,30)).mean('lon').mean('lat')
# US_land_ensmean = Xt_ensmean_land.sel(lon=slice(235,295),lat=slice(45,30)).mean('lon').mean('lat')

Nino34_forced = X_forced.sel(lon=slice(190,240),lat=slice(5,-5)).mean('lon').mean('lat')
Nino34_ensmean = Xt_ensmean.sel(lon=slice(190,240),lat=slice(5,-5)).mean('lon').mean('lat')

EEP_forced = X_forced.sel(lon=slice(210,270),lat=slice(6,-6)).mean('lon').mean('lat')
EEP_ensmean = Xt_ensmean.sel(lon=slice(210,270),lat=slice(6,-6)).mean('lon').mean('lat')

WEP_forced = X_forced.sel(lon=slice(120,180),lat=slice(6,-6)).mean('lon').mean('lat')
WEP_ensmean = Xt_ensmean.sel(lon=slice(120,180),lat=slice(6,-6)).mean('lon').mean('lat')

# %%
f=plt.figure(figsize=(3,3),constrained_layout=True)
plt.plot(T,GMST_ensmean+ts_clim_all.mean())
plt.plot(T,GMST_forced+ts_clim_all.mean())
plt.title('GMST')
plt.legend(('Ens. Mean','SNP Filtered'),loc='lower right')
plt.savefig(f'{plots_dir}/gmst_timeseries.pdf')

#%%
# # global mean temperature response vs time
# f = pplt.figure(refwidth=3)
# ax = f.subplot(111)
# ax.plot(T,(ts_all.groupby('time.month').mean('time')+ts_clim_all).mean(dim=['member','lat','lon']))
# # ax.plot(T,(ts_all.groupby('time.month')+ts_clim_all).mean(dim=['member','lat','lon']))

# first attempt at a timeseries of monthly averages (may be useful later)
# ts_all_monthly = ts_all.copy()
# ts_all_monthly.drop('time')
# ts_all_monthly.coords['time'] = ts_all['time.month']
# for yr in range(decades[0],decades[-1]+9):
#     for mon in range(1,13):
#         ts_all_monthly.loc[dict(time=f'{yr}-{mon:02d}')] = (
#             ts_all.sel(time=slice(f'{yr}-01-01',f'{yr}-12-31'))
#             .where(ts_all['time.month']==f'{yr}-{mon:02d}').mean('time'))

#%%
# plot of forced pattern vs ensemble mean
yri = 2040
yrf = 2049
min_tas = -2
max_tas = 2
step_tas = 0.2
levels_tas = np.arange(min_tas,max_tas+step_tas,step_tas)
cmap_tas = plt.cm.RdBu_r
array = [
    [1],
    [2],
]
f = pplt.figure(refwidth=4)
axs = f.subplots(array,proj='cyl')
axs.format(
    abc=False, abcloc='ul',
    xlabel='xlabel', ylabel='ylabel',
    coast=True
)
ctr0 = axs[0].contourf(lon,lat,
    X_forced.sel(time=slice(f'{yri}-01-01',f'{yrf}-12-31')).mean('time')
    + ts_clim_all.mean(['member','month']),
    levels=levels_tas,cmap=cmap_tas,extend='both',
    transform=ccrs.PlateCarree())
axs[0].set_title(r'$X_{SNP}$, mean over 2040-2049')
ctr1 = axs[1].contourf(lon,lat,
    Xt_ensmean.sel(time=slice(f'{yri}-01-01',f'{yrf}-12-31')).mean('time')
    + ts_clim_all.mean(['member','month']),
    levels=levels_tas,cmap=cmap_tas,extend='both',
    transform=ccrs.PlateCarree())
axs[1].set_title(r'$\langle{X}\rangle$, mean over 2040-2049')
f.colorbar(ctr0, loc='r', label='$\delta{T} [K]$')
f.savefig(f'{plots_dir}/ensmean_vs_snp.pdf')

# %%
f=plt.figure()
plt.plot(T,Nino34_ensmean)
plt.plot(T,Nino34_forced)
plt.title('Nino3.4')
plt.legend(('Ens. Mean','SNP Filtered'))

# %%
f=plt.figure()
plt.plot(T,Nino34_ensmean-GMST_ensmean)
plt.plot(T,Nino34_forced-GMST_forced)
plt.title('Nino34 Anomaly')
plt.legend(('Ens. Mean','SNP Filtered'))

# %%
f=plt.figure()
plt.plot(T,EEP_ensmean-WEP_ensmean)
plt.plot(T,EEP_forced-WEP_forced)
plt.title('Pacific SST Gradient')
plt.legend(('Ens. Mean','SNP Filtered'))

# # %%
# f=plt.figure()
# plt.plot(T,US_land_ensmean)
# plt.plot(T,US_land_forced)
# plt.title('U.S. Land Surface Temperature')
# plt.legend(('Ens. Mean','SNP Filtered'))

# # %%
# f=plt.figure()
# plt.plot(T,tropical_land_ensmean)
# plt.plot(T,tropical_land_forced)
# plt.title('Tropical Land Surface Temperature')
# plt.legend(('Ens. Mean','SNP Filtered'))

# %%
