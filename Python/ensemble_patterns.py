# %%
# load builtin packages
import os
import pickle
#load packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
# load user-defined packages
import snplib as snp

#%%
# Set plot format
plt.style.use('prb')

# %%
# USER INPUT
COMPUTE_EOFS = True
DATA_DIR = 'scratch'
EOF_DIR = '/scratch/p/pjk/lfl/snp_eofs/synthetic-data-rwills/'
EXP_C = 'ssp370'
EXP_P = 'ssp370-126aer'
MODEL = 'synthetic-data'
PICKLE_DIR_TS = '/scratch/p/pjk/lfl/snp_data_matrices/'
PRECISE_FNAMES = False
REALIZATIONS = np.arange(1, 11)
VAR_NAME = 'tas'
VAR_TABLE = 'Amon'
YRI_AVG = 2040
YRF_AVG = 2049
T = np.arange(2015,2051,1/12) # historical simulations start in 1920 for CESM, 1850 for MPI (first input to arrange must match)
# Second input to arrange can be chosen to select period of interest, but change name used in pickle files for end-dates other than 2006

#%%
ts_pickle_name = f"{MODEL}-{EXP_P}-{EXP_C}_{VAR_NAME}.p"
ts_pickle_path = f"{PICKLE_DIR_TS}/{MODEL}/{EXP_P}/{ts_pickle_name}"
ts_anom_coarse, ts_clim_coarse = snp.compute_coarsen_pickle_ts(
    exp_c=EXP_C, exp_p=EXP_P, model=MODEL, table=VAR_TABLE, var_name=VAR_NAME,
    pickle_path=ts_pickle_path, precise_fnames=PRECISE_FNAMES,
    realizations=REALIZATIONS, coarsen_factor=4, derived=False,
    directory=DATA_DIR)
ts_all = ts_anom_coarse
ts_clim = ts_clim_coarse
ne = ts_all.realization
nt = len(ts_all.time)
n_lat = len(ts_all.lat)
n_lon = len(ts_all.lon)

# %%
# sanity check plot, just shows changes in temperature over the simulation

# field = ts_all.values
field = ts_all
field = ts_all.mean(dim='realization')
# field = np.mean(field,axis=0)
field_diff = (
    ts_all.sel(time=slice(f'{YRI_AVG}-01-01', f'{YRF_AVG}-12-31'))
    .mean(dim=['time', 'realization'])
    - ts_all.sel(time=slice(f'2015-01-01', f'2024-12-31'))
    .mean(dim=['time', 'realization'])
)
# field_diff = np.mean(field[912:1031,:,:],axis=0)-np.mean(field[0:119,:,:],axis=0)
f=plt.figure()
plt.contourf(
    ts_all.lon.values, ts_all.lat.values, field_diff, np.arange(-1,1.1,0.1),
    cmap=plt.cm.RdBu_r)
cbar = plt.colorbar()

# %%
# Preprocessing for Large Ensemble EOFs

lon = ts_all.lon
lat = ts_all.lat
time = ts_all.time
cosw = np.sqrt(np.cos(lat*np.pi/180))
normvec  = cosw/np.sum(cosw);
scale = np.sqrt(normvec);

X=ts_all*scale

X_ensmean=X.mean('realization')
X_flat = X.stack(index=['time','realization']).stack(shape=['lat','lon'])
X_ensmean_flat = X_ensmean.stack(shape=['lat','lon'])

# keep unscaled copies of these variables
Xt_ensmean=ts_all.mean('realization')
Xt_flat = ts_all.stack(index=['time','realization']).stack(shape=['lat','lon'])
Xt_ensmean_flat = Xt_ensmean.stack(shape=['lat','lon'])

index = X_flat.index
n = len(index)

#%%
# Check scale
plt.figure()
plt.plot(lat, cosw)
plt.figure()
plt.plot(lat, scale)

#%%
# Check data matrix.
X_flat_sampled = X_flat.sel(
    time=slice(f'{YRI_AVG}-01-01', f'{YRF_AVG}-12-31'))
X_flat_reshaped = np.reshape(
    np.mean(X_flat_sampled.data, axis=0), [n_lat, n_lon])
plt.imshow(X_flat_reshaped, vmin=0.0, vmax=0.10)
plt.colorbar(orientation='horizontal')
pickle_path = f"{PICKLE_DIR_TS}/{MODEL}/{EXP_P}/datamat_rwills.p"
with open(pickle_path, "wb+") as pickle_file:
    pickle.dump(X_flat, pickle_file, protocol=4)

# %%time
# Perform ensemble EOF analysis (takes a few minutes), or load from Pickle if it has already been done

if not COMPUTE_EOFS:
    # load PCA output from Pickle
    pcvec,evl = pickle.load( open(EOF_DIR+MODEL+"_"+VAR_NAME+"_EOF.p", "rb" ))
else: 
    # Large Ensemble EOFs
    Cov = np.matmul(X_flat.values.T,X_flat.values)/(n-1)
    evl,pcvec = np.linalg.eig(Cov)
    pickle.dump([pcvec,evl], open(EOF_DIR+MODEL+"_"+VAR_NAME+"_EOF.p", "wb" ),protocol=4)

s=np.sqrt(evl)

## keeping the below as a reminder that SVD is much slower than eigenvalue analysis for datasets with long time dimension

#%%
# Check EOFs
eof_reshaped = np.reshape(pcvec[:, 0], [n_lat, n_lon])
plt.imshow(-eof_reshaped, vmin=0.0, vmax=0.08)
plt.colorbar(orientation='horizontal')

#%%time
# Perform ensemble EOF analysis (SVD takes ~20-25 minutes), or load from Pickle if it has already been done

#try:
#    # load SVD output from Pickle
#    u,s = pickle.load( open(EOF_DIR+MODEL+"_"+VAR_NAME+"_SVD.p", "rb" ))

#except:
    # Large Ensemble EOFs
    
#    #u,s = np.linalg.svd(np.transpose(X_flat.values)/np.sqrt(n-1))
#    u,s = np.linalg.svd(np.transpose(X_flat[0:int(n/10),:].values)/np.sqrt(n/10-1))
#    pickle.dump([u,s], open(EOF_DIR+MODEL+"_"+VAR_NAME+"_SVD.p", "wb" ),protocol=4)
    
#eigvals=np.diag(s*s)

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
        
pickle.dump([tk,tk_emean,SNPs_reshaped,weights,signal_frac], open(EOF_DIR+MODEL+"_"+VAR_NAME+"_SNP"+str(neof)+".p", "wb" ),protocol=4)

#%%
# Compute forced response.
M = 2  # number of forced patterns to retain, choose cutoff based on eigenvalue spectrum, or check significant patterns with bootstrapping
# Compute forced response as estimated by SNP filtering.
forced_response = np.real(np.matmul(tk_emean[:, 0:M], SNP[0:M, :]))
snps_reshaped = np.reshape(SNP, [n_lat, n_lon, neof])
forced_response_reshaped = np.reshape(forced_response, [nt, n_lat, n_lon])
forced_response_da = xr.DataArray(
    forced_response_reshaped,
    coords={'time': time, 'lat': lat, 'lon': lon},
    name=VAR_NAME, )

#%%
# Debug.
fr_plot_db = (
    (forced_response_da / scale)
    .sel(time=slice(f'{YRI_AVG}-01-01', f'{YRF_AVG}-12-31')).mean(dim='time')
)

f1, ax1 = plt.subplots()
fr1 = ax1.imshow(np.abs(fr_plot_db))
f1.colorbar(fr1, orientation='horizontal')
# f2, ax2 = plt.subplots()
# fr2 = ax2.imshow(
#     np.abs(
#         (fr_plot_db).data
#         / em_plot.coarsen(lat=4, lon=4, boundary='trim').mean().data),
#         vmin=0, vmax=0.5)
# f2.colorbar(fr2)

# %%
print(signal_frac[0:30])
plt.plot(signal_frac,marker='o')
plt.xlim(0,30)
plt.ylim(0,1)
plt.title('Signal Fraction')

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
for neof_plot in range(2): 
    f=plt.figure()
    plt.contourf(lon.values,lat.values,np.squeeze(SNPs_reshaped[neof_plot,:,:]),np.arange(-0.6,0.65,0.05),cmap=plt.cm.RdBu_r)
    cbar = plt.colorbar()

# %%
# Plot forced pattern timeseries
tk_reshape=tk.reshape(nt,len(ne),neof)

for neof_plot in range(2): 
    f=plt.figure()
    [plt.plot(T,tk_reshape[:,mm,neof_plot],color='crimson') for mm in range(10)];
    plt.plot(T,tk_emean[:,neof_plot], color='k')
    plt.title('SNP'+str(neof_plot+1))

# %%
# Forced component from leading forced patterns

M = 2  # number of forced patterns to retain, choose cutoff based on eigenvalue spectrum, or check significant patterns with bootstrapping

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

Nino34_forced = X_forced.sel(lon=slice(190,240),lat=slice(-5,5)).mean('lon').mean('lat')
Nino34_ensmean = Xt_ensmean.sel(lon=slice(190,240),lat=slice(-5,5)).mean('lon').mean('lat')

EEP_forced = X_forced.sel(lon=slice(210,270),lat=slice(-6,6)).mean('lon').mean('lat')
EEP_ensmean = Xt_ensmean.sel(lon=slice(210,270),lat=slice(-6,6)).mean('lon').mean('lat')

WEP_forced = X_forced.sel(lon=slice(120,180),lat=slice(-6,6)).mean('lon').mean('lat')
WEP_ensmean = Xt_ensmean.sel(lon=slice(120,180),lat=slice(-6,6)).mean('lon').mean('lat')

#%%
# Plot surface temperature response.
X_forced_plot = (
    X_forced.sel(time=slice(f'{YRI_AVG}-01-01', f'{YRF_AVG}-12-31'))
    .mean(dim='time')
)
X_ensmean_plot = (
    ts_all.sel(time=slice(f'{YRI_AVG}-01-01', f'{YRF_AVG}-12-31'))
    .mean(dim=['time', 'realization'])
)
f=plt.figure()
plt.contourf(
    lon.values, lat.values, X_forced_plot, np.arange(-0.6,0.65,0.1),
    cmap=plt.cm.RdBu_r)
cbar = plt.colorbar()
f=plt.figure()
plt.contourf(
    lon.values, lat.values, X_ensmean_plot, np.arange(-0.6,0.65,0.1),
    cmap=plt.cm.RdBu_r)
cbar = plt.colorbar()
f=plt.figure()
plt.contourf(
    lon.values, lat.values, X_forced_plot - X_ensmean_plot,
    cmap=plt.cm.RdBu_r)
cbar = plt.colorbar()

# %%
f=plt.figure()
plt.plot(T,GMST_ensmean)
plt.plot(T,GMST_forced)
plt.title('GMST')
plt.legend(('Ens. Mean','SNP Filtered'))

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

# %%
