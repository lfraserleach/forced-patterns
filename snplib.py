import numpy as np
import xarray as xr
import pickle

def load_2d_ts(
        varnam, table, exp_p,
        exp_c='ssp370-ramip', memi=4, memf=10, model='NorESM2-LM',
        decades=[2015, 2020, 2030, 2040], datadir='/project/p/pjk/lfl/ramip',
        outputdir='/project/p/pjk/lfl/ramip/snp_data', reload_data=False
        ):
    """
    Load 2D timeseries of difference of a variable between two experiments
    (with multiple ensemble members), and return the timeseries and the monthly 
    climatologies.
    """
    T=np.arange(2015,2050,1/12)
    name = f"{model}-{exp_p}-{exp_c}"
    ensdir_c = f"{datadir}/{model}/{exp_c}"
    ensdir_p = f"{datadir}/{model}/{exp_p}"

    try:
        if reload_data:
            raise Exception('reload_data is True, so model output will be '\
            + 're-loaded and pickled')
        # load pre-processed SST data from Pickle
        ts_all = pickle.load(open(f"{outputdir}/{name}_{varnam}_all.p", "rb" ))
        ts_clim_all = pickle.load(open(f"{outputdir}/{name}_{varnam}_clim_all.p", "rb" ))
        if table in ['Omon', 'SImon']:
            lat = ts_all.latitude
            lon = ts_all.longitude
        else:
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
        lon = ds0.lon
        lat = ds0.lat
        time = ds0.time
        nt = len(time)
        nt_cut = len(T)
        month = np.linspace(1, 12, 12)

        ts_all = np.empty((n,nt_cut,len(lat),len(lon)))
        ts_clim_all = np.empty((n,12,len(lat),len(lon)))
        time = time[0:nt_cut]
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
    
    return ts_all, ts_clim_all

def flatten_scale_ts(ts_all):
    """
    Flatten the 2D timeseries ts_all into a 1D area weighted array X_flat, scale
    by grid cell area, and return the area-weighted ensemble mean 
    X_ensmean_flat.
    """
    # calculate area-weighted mean
    lat = ts_all.lat
    cosw = np.cos(lat*np.pi/180)
    normvec  = cosw/np.sum(cosw);
    scale = np.sqrt(normvec);
    X=ts_all*scale
    X_ensmean=X.mean('member')
    X_flat = X.stack(index=['time','member']).stack(shape=['lat','lon'])
    X_ensmean_flat = X_ensmean.stack(shape=['lat','lon'])
    # keep unscaled copies of these variables
    Xt_ensmean=ts_all.mean('member')
    Xt_flat = ts_all.stack(index=['time','member']).stack(shape=['lat','lon'])
    Xt_ensmean_flat = Xt_ensmean.stack(shape=['lat','lon'])

    index = X_flat.index
    n = len(index)

    return X_flat, X_ensmean_flat, Xt_ensmean, Xt_flat, Xt_ensmean_flat, scale
