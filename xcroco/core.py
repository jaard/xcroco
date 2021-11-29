import xarray as xr
import pandas as pd
from xgcm import Grid
import numpy as np
from scipy.interpolate import griddata
from typing import Sequence
import dask
import warnings
import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import glob
import time, sys

    
def croco_dataset(model_output, time_dim='time', grid=None, xgcm_grid=None, *args, **kwargs):

    if isinstance(model_output, xr.Dataset):
        da = model_output
    else:
        da = xr.open_mfdataset(model_output, decode_times=False, decode_cf=False, combine='by_coords')
        da = xr.decode_cf(da)
    
    # Check if a separate grid file was supplied,
    # if this is the case load & merge with output file
    if grid:
        grid_path = grid
        gr = xr.open_dataset(grid_path)
        # If the grid files has the redundant dimensions eta_u/xi_v then rename them
        try:
            gr = gr.rename({'eta_u':'eta_rho', 'xi_v':'xi_rho'})
        except ValueError:
            pass
        droplist = [v for v in list(gr.data_vars) if v in list(da.data_vars) + list(da.coords)]
        for v in droplist:
            da = da.drop(v)
        if 'spherical' in gr.data_vars:
            da['spherical'] = gr['spherical']
            gr = gr.drop('spherical')
        gr = gr.astype('float32')
        da2 = xr.merge((da, gr))

    # If no grid file supplied, make sure redundant time dimension is removed from concatenated variables
    else:
        notime_vars = ['spherical','xl','el','Vtransform','sc_r','sc_w','Cs_r','Cs_w','hc','h','f','pm','pn','angle','mask_rho']
        for vv in notime_vars:
            try:
                if time_dim in da[vv].dims:
                    da[vv] = da[vv].isel(**{time_dim:0})
            except KeyError:
                pass
        da2 = da
        
    # Overwrite xi / eta variables in case they are not continuous
    for co in ['xi_rho','xi_u','eta_rho','eta_v']:
        if co not in da2.coords:
            continue  # untested for now! TODO: test if it works for dataset without u,v
        co_attr = da2[co].attrs
        da2.coords['a'] = (co, np.arange(da2[co].shape[0]).astype('float'))
        da2 = da2.set_index(**{co:'a'})
        da2[co].attrs = co_attr

    # move lat/lon to from data variables to coordinates
    latlon_vars = [v for v in list(da2.data_vars) if 'lat' in v or 'lon' in v]
    for v in latlon_vars:
        da2.coords[v] = da2[v]
    
    # Remove time dimension from lat/lon coordinates, if present
    latlon_coords = [v for v in list(da2.coords) if 'lat' in v or 'lon' in v]
    for v in latlon_coords:
        v_da = da2.coords[v]
        if time_dim in v_da.dims:
            v_da = v_da.isel(**{time_dim:0})
            da2.coords[v] = v_da.drop('time')
    
    # Copy attrs and create XGCM grid object
    da2.attrs = da.attrs
    if xgcm_grid:
        da2.attrs['xgrid'] = xgcm_grid
    else:
        #da2.attrs['xgrid'] = Grid(da2, periodic=False)
        da2.attrs['xgrid'] = Grid(da2,
                                  coords = {'X': {'center': 'xi_rho', 'inner': 'xi_u'},
                                            'Y': {'center': 'eta_rho', 'inner': 'eta_v'},
                                            'Z': {'center': 's_rho', 'outer': 's_w'}},
                                  boundary={'X':'extend','Y':'extend','Z':'extend'},
                                  periodic=False)
        
    #return da2 # DEBUG

    # Read grid parameters depending on version
    # TODO: See how this version checking (taken from ROMSTOOLS) is done in the new official CROCOTOOLS 
    if np.size(da2.Tcline) is 0:
        # 'UCLA version'
        pass
    else:
        # 'AGRIF version'
        hmin = np.nanmin(da2.h)
        try:
            #da2.hc.data = np.min([hmin, da2.Tcline])
            da2.hc.values = (da2.hc.values*0+1)*np.min([hmin, da2.Tcline])
        except AttributeError:
            da2['hc'] = np.min([hmin, da2.Tcline])

    # Check which s-coordinates are used
    # 1... old, 2...new
    try:
        VertCoordType = da2.VertCoordType
        if VertCoordType == 'NEW':
            da2.attrs.s_coord = 2
    except AttributeError:
        try:
            vtrans = da2.Vtransform
            if vtrans.size is not 0:
                if vtrans.size == 1:
                    da2.attrs['s_coord'] = int(vtrans)
                else:
                    da2.attrs['s_coord'] = int(vtrans[0])
        except KeyError:
            da2.attrs['s_coord'] = 1
    if da2.attrs['s_coord'] == 2:
        da2['hc'] = da2.Tcline
    
    # Define Z coordinates
    da2.coords['z_rho'] = zlevs(da2, 'r')
    da2.coords['z_w'] = zlevs(da2, 'w')
    if 'xi_u' in da2.dims:
        da2.coords['z_u'] = rho2var(da2, da2.z_rho, da2.u)
        da2['z_u'].attrs['long_name'] = 'depth at U-points'
        da2['z_u'].name = 'z_u'
    if 'eta_v' in da2.dims:
        da2.coords['z_v'] = rho2var(da2, da2.z_rho, da2.v)
        da2['z_v'].attrs['long_name'] = 'depth at V-points'
        da2['z_v'].name = 'z_v'

    return da2


def csf(croco_dataset, sc):

    '''
    function h = csf(sc, theta_s,theta_b)
    '''

    if croco_dataset.theta_s > 0:
        csrf = (1-np.cosh(sc * croco_dataset.theta_s)) / (np.cosh(croco_dataset.theta_s)-1)
    else:
        csrf = -sc ** 2
    if croco_dataset.theta_b > 0:
        h = (np.exp(croco_dataset.theta_b * csrf)-1) / (1-np.exp(-croco_dataset.theta_b))
    else:
        h = csrf

    return h


def get_coastline_from_mask(croco_ds):
    
    '''
    Create a coastline from the RHO mask
    
    Parameters
    ----------
    croco_ds: xr.Dataset
              Provides the variables mask_rho, lon_rho, lat_rho
              
    TODO:
    Add this function to initialization of dataset and save as "coastline_rho" variable
    '''

    maskdims = croco_ds.mask_rho.dims

    dxi = croco_ds.mask_rho.diff(maskdims[1])
    dxi_land2sea = dxi.where(dxi>0, other=0)
    dxi_land2sea.coords[maskdims[1]] = (np.arange(dxi_land2sea.mean(maskdims[0]).size)).astype(float)
    dxi_sea2land = dxi.where(dxi<0, other=0)*-1
    dxi_sea2land.coords[maskdims[1]] = (np.arange(dxi_sea2land.mean(maskdims[0]).size)+1).astype(float)
    xi_coast = (dxi_land2sea + dxi_sea2land)
    xi_coast.coords[maskdims[0]] = (np.arange(dxi_land2sea.mean(maskdims[1]).size)).astype(float)

    deta = croco_ds.mask_rho.diff(maskdims[0])
    deta_land2sea = deta.where(deta>0, other=0)
    deta_land2sea.coords[maskdims[0]] = (np.arange(deta_land2sea.mean(maskdims[1]).size)).astype(float)
    deta_sea2land = deta.where(deta<0, other=0)*-1
    deta_sea2land.coords[maskdims[0]] = (np.arange(deta_sea2land.mean(maskdims[1]).size)+1).astype(float)
    eta_coast = (deta_land2sea + deta_sea2land)
    eta_coast.coords[maskdims[1]] = (np.arange(deta_land2sea.mean(maskdims[0]).size)).astype(float)

    coastsum = (xi_coast+eta_coast).astype(bool)
    coast,b = xr.align(coastsum, croco_ds.mask_rho,
                       join='outer',fill_value=0)
    
    return coast
    
    
def distance2coast(croco_ds, coast='coastline_rho', **kwargs):
    
    '''
    Calculate the distance to coast from a coastline
    
    Parameters
    ----------
    croco_ds: xr.Dataset
              Provides the variables coastline_rho, lon_rho, lat_rho
    condition: boolean array the same shape as mask_rho to mask the coastline 
               and i.e. remove islands
    '''
    
    # apply masking condition (i.e. remove islands)
    cline = croco_ds[coast]
    clon = croco_ds.lon_rho
    clat = croco_ds.lat_rho
    if kwargs and 'condition' in kwargs:
        cline = cline.where(kwargs['condition'],drop=True)
        clon = clon.where(kwargs['condition'],drop=True)
        clat= clat.where(kwargs['condition'],drop=True)

    # flatten coast to one dimension
    coast1D = cline.stack(points=(cline.dims))
    lon1D = clon.stack(points=(cline.dims)).where(coast1D,drop=True)
    lat1D = clat.stack(points=(cline.dims)).where(coast1D,drop=True)
    #lonlim = (lon1D > -84)
    #lon1D = lon1D.where(lonlim,drop=True)
    #lat1D = lat1D.where(lonlim,drop=True)
    coastline_from_mask = np.stack([lon1D.values,lat1D.values]).T
    print('Calculating distances to {} coastal points...'.format(coastline_from_mask.shape[0]))
    croco_coords = [np.array([lo,la]) for lo,la in zip(croco_ds.lon_rho.stack(n=(croco_ds.mask_rho.dims)).values,
                                                       croco_ds.lat_rho.stack(n=(croco_ds.mask_rho.dims)).values)]
    
    geo = cgeo.Geodesic()
    dist_list = []
    update_progress(0)
    for ind,cro in enumerate(croco_coords):
        dists = geo.inverse(cro,coastline_from_mask).base[:,0]/1000
        dist_list.append(np.min(dists))
        if ind % 100 == 0:
            update_progress(ind/len(croco_coords))
    distarray = np.array(dist_list)
    update_progress(1)

    dist2coast = xr.zeros_like(croco_ds.mask_rho).stack(n=croco_ds.mask_rho.dims)
    dist2coast.values = distarray.T
    dist2coast = dist2coast.unstack('n')
    
    return dist2coast


def get_vertical_dimension(var):
    try:
        return [d for d in var.dims if 's_' in d][0]
    except IndexError:
        raise IndexError('{} is a 2D-H variable'.format(var.name))


def zlevs(croco_ds, typ, **kwargs):

    '''
    this method computes the depth of rho or w points for ROMS

    Parameters
    ----------
    croco_ds: xr.Dataset
              Provides the variables h and zeta
    typ : String
          'r' for rho-grid or 'w' for w-grid

    Returns
    -------
    z : xr.DataArray
        Depths (m) of RHO- or W-points (3D matrix).
    '''
    
    if typ is 'w':
        vertical_dim = 's_w'
    else:
        vertical_dim = 's_rho'
        
    if kwargs and 'zeta' in kwargs:
        zeta = kwargs['zeta']
    else:
        zeta = croco_ds.zeta
    
    # Create output dimensions of correct shape
    dims = list((croco_ds[vertical_dim] * zeta).dims)
    output_dims = tuple([a for a in dims if 'time' in a] + [a for a in dims if 'time' not in a])
    h = croco_ds.h

    try:
        hc = croco_ds.hc.isel(time=0).data
    except ValueError:
        hc = croco_ds.hc.data
    N = croco_ds.s_rho.size

    hshape = np.shape(h)
    if len(hshape) == 1:
        L = hshape[0]
        M = 1
    else:
        L, M = np.shape(h)

    try:
        vtransform = croco_ds.s_coord
    except AttributeError:
        try:
            vtransform = croco_ds.attrs.s_coord
        except AttributeError:  
            warnings.warn('no vtransform defined', Warning)
            vtransform = 1
            warnings.warn('Default S-coordinate system use : Vtransform = 1 (old one)', Warning)

    # Set S-Curves in domain [-1 < sc < 0] at vertical W- and RHO-points.
    
    sc_r = np.zeros([N])
    Cs_r = np.zeros([N])
    sc_w = np.zeros([N+1])
    Cs_w = np.zeros([N+1])
            
    if vtransform == 2:
        if typ is 'w':
            sc_w[0] = -1.0
            sc_w[N] = 0
            Cs_w[0] = -1.0
            Cs_w[N] = 0
            sc_w[1:N] = (np.arange(1, N) - N) / N
            Cs_w = csf(croco_ds, sc_w)
            N += 1
        else:
            sc = (np.arange(1, N+1)-N-0.5) / N
            Cs_r = csf(croco_ds, sc)
            sc_r = sc

    elif vtransform == 1:
        cff1 = 1. / np.sinh(croco_ds.theta_s)
        cff2 = 0.5 / np.tanh(0.5 * croco_ds.theta_s)
        if typ is 'w':
            sc = (np.arange(0, N+1)-N) / N
            N += 1
        else:
            sc = (np.arange(1, N+1)-N-0.5) / N

        Cs = ((1. - croco_ds.theta_b) * cff1 * np.sinh(croco_ds.theta_s * sc)
              + croco_ds.theta_b * (cff2 * np.tanh(croco_ds.theta_s * (sc+0.5))-0.5))

    # Create S-coordinate system: based on model topography h(i,j),
    # fast-time-averaged free-surface field and vertical coordinate
    # transformation metrics compute evolving depths of of the three-
    # dimensional model grid. Also adjust zeta for dry cells.

    Dcrit = 0.2  # min water depth in dry cells
    h = h.where(h > 0, other=1e-14)
    # not sure if next line works corrrectly yet... not tested
    zeta = zeta.where(zeta > (Dcrit-h), other=(Dcrit-h.where(zeta < (Dcrit-h))))

    hinv = 1. / h

    # initialize z as xr.DataArray
    z = xr.zeros_like(zeta * croco_ds[vertical_dim])

    if vtransform == 2:
        if typ is 'w':
            cff1 = Cs_w
            cff2 = sc_w+1
            sc = xr.DataArray(sc_w, dims=(vertical_dim))
            Cs = xr.DataArray(Cs_w, dims=(vertical_dim))
            sc = sc_w
        else:
            cff1 = Cs_r
            cff2 = sc_r+1
            sc = xr.DataArray(sc_r, dims=(vertical_dim))
            Cs = xr.DataArray(Cs_r, dims=(vertical_dim))
            sc = sc_r

        h2 = (h+hc)
        cff = hc*sc
        h2inv = 1. / h2
        cff1_xr = xr.DataArray(cff1,dims=(vertical_dim,))
        cff_xr = xr.DataArray(cff,dims=(vertical_dim,))

        z0 = cff_xr + cff1_xr * h
        z = zeta * (1. + z0 * h2inv) + z0 * h / h2
        z = z.transpose(*output_dims)

    elif vtransform == 1:
        cff1 = Cs
        cff2 = sc + 1
        sc = xr.DataArray(sc, dims=(vertical_dim))
        Cs = xr.DataArray(Cs, dims=(vertical_dim))
        cff = hc * (sc-Cs)
        cff2 = sc + 1
        cff1_xr = xr.DataArray(cff1,dims=(vertical_dim,))
        cff_xr = xr.DataArray(cff,dims=(vertical_dim,))

        z0 = cff_xr + cff1_xr * h
        z = zeta * (1. + z0 * hinv) + z0
        z = z.transpose(*output_dims)
        
    if typ is 'w':
        z.attrs['long_name'] = 'depth at W-points'
        z.name = 'z_w'
    else:
        z.attrs['long_name'] = 'depth at RHO-points'
        z.name = 'z_rho'
    z.attrs['units'] = 'meter'
    z.attrs['field'] = 'depth, scalar, series'
    z.attrs['standard_name'] = 'depth'

    return z

def add_coords(croco_ds, var, coords):
    for co in coords:
        var.coords[co] = croco_ds.coords[co]

def var2rho(croco_ds, var):

    '''
    Interpolate any variable to RHO points,
    position on grid is checked automatically
    '''
    latlon_rho = ['lon_rho','lat_rho']
    grid = croco_ds.attrs['xgrid']
    if 'xi_u' in var.dims:
        var_rho = grid.interp(var,'X', boundary='extend')
    elif 'eta_v' in var.dims:
        var_rho = grid.interp(var,'Y', boundary='extend')
        add_coords(croco_ds, var_rho, latlon_rho)
        var_rho.attrs = var.attrs
        var_rho.name = var.name
    elif 's_w' in var.dims:
        var_rho = grid.interp(var,'Z')
    else:
        warnings.warn('Variable already at rho points, returning input array.', Warning)
        return var

    var_rho.attrs = var.attrs
    var_rho.name = var.name
    #var_rho.attrs['long_name'] = '{} at RHO-points'.format(var_rho.name)
    if 'standard_name' in var.attrs:
        var_rho.attrs['standard_name'] = '{}rho{}'.format(var_rho.attrs['standard_name'][:-10],
                                                          var_rho.attrs['standard_name'][-9:])
    # new lat/lon coordinates are not added by xgcm need to be
    # added by hand from the Dataset to the DataArray
    add_coords(croco_ds, var_rho, latlon_rho)

    return var_rho

def rho2var(croco_ds, var_rho, var_target):

    if any(dimension in var_rho.dims for dimension in ['eta_v','xi_u','s_w']):
        raise ValueError('Input array is not on RHO-grid')
    grid = croco_ds.attrs['xgrid']
    
    try:
        del var_rho.coords['z_rho']
    except KeyError:
        pass
    
    if 'xi_u' in var_target.dims:
        var = grid.interp(var_rho,'X', boundary='extend')
        add_coords(croco_ds, var, ['lat_u','lon_u'])
        var.attrs = var_rho.attrs
        var.name = var_rho.name
        var.attrs['long_name'] = '{} at U-points'.format(var.name)
    elif 'eta_v' in var_target.dims:
        var = grid.interp(var_rho,'Y', boundary='extend')
        add_coords(croco_ds, var, ['lat_v','lon_v'])
        var.attrs = var_rho.attrs
        var.name = var_rho.name
        var.attrs['long_name'] = '{} at V-points'.format(var.name)
    elif 's_w' in var_target.dims:
        var = grid.interp(var_rho,'Z', boundary='extend')
        add_coords(croco_ds, var, ['lat_rho','lon_rho'])
        var.attrs = var_rho.attrs
        var.name = var_rho.name
        var.attrs['long_name'] = '{} at W-points'.format(var.name)
    else:
        warnings.warn('Both arrays already at rho points, returning input array.', Warning)
        return var_rho 

    return var


def get_depths(croco_ds, var):

    '''
    Get the depths of the sigma levels

    Parameters
    ----------
    var : xr.DataArray
          on arbitrary grid (rho, u, v or w)

    Returns
    -------
    z : xr.DataArray
        Depths (m) of RHO- or W-points (3D matrix).
    '''

    if 's_w' in var.dims:
        depths = zlevs(croco_ds, 'w')
    else:
        depths = zlevs(croco_ds, 'r')
        if 'xi_u' in var.dims:
            depths = rho2var(croco_ds, depths, var)
            depths.attrs['long_name'] = 'depth at U-points'
            depths.name = 'z_u'
        if 'eta_v' in var.dims:
            depths = rho2var(croco_ds, depths, var)
            depths.attrs['long_name'] = 'depth at V-points'
            depths.name = 'z_v'
    if 'time' in depths.dims:
        depths = depths.sel(time=var.time)
    
    return depths


def valid_levels(levels):
    
    '''
    Make the specified levels compatible with vinterp() if possible
    '''
    try:
        validlevs = bool(list(levels))
    except TypeError:
        validlevs = bool([levels])
    if not validlevs:
        raise ValueError('Please specify at least one valid interpolation depth')
        
    if hasattr(levels, '__iter__'):
        if type(levels) != str:
            levels = np.array(levels)
            
            # Check if all positive or negative depth definition
            if (levels <= 0).all():
                pass
            elif (levels * -1 <= 0).all():
                levels *= -1
            else:
                raise ValueError('Please use either negative or positive depth values')
                
            # Check if continuously decreasing, otherwise sort
            if all(np.diff(levels) < 0):
                pass
            else:
                levels = np.unique(levels)
                levels.sort()
                levels = levels[::-1]
            return list(levels)

        else:
            try:
                levels = float(levels)
                intlevs = int(levels)
                if intlevs == levels:
                    levels = intlevs
            except ValueError:
                raise ValueError('String could not be converted to depth level')
    return levels
    
    
def vinterp(var, z, depth):

    '''
    function  vnew = vinterp(var,z,depth)

    This function interpolates a 3D variable on a horizontal level of constant
    depth

    Parameters
    ----------
    var     xr.DataArray
            Variable to process (3D matrix).
    depth   Slice depth (scalar; meters, negative).

    Returns
    -------
    vnew    xr.DataArray
            Horizontal slice (2D matrix).
    '''

    depth = valid_levels(depth)

    if var.shape != z.shape:
        try:
            z = z.transpose(*var.dims)
        except ValueError:
            raise ValueError('Number of dimensions mismatch between Variable and Depth arrays')
        if var.shape != z.shape:
            raise ValueError('Shape mismatch between Variable and Depth arrays')

    vertical_dim = get_vertical_dimension(var)
    N = len(var[vertical_dim])

    if isinstance(depth,list):
        # Loop over depth list
        zslice_list = []
        for dd in depth:
            # Find the grid position of the nearest vertical levels
            levs = (z < dd).sum(vertical_dim)
            levs = levs.where(levs<N, other=N-1)
            levs.load()
            # Do the interpolation
            z1 = z.isel(**{str(vertical_dim):levs})
            z2 = z.isel(**{str(vertical_dim):levs-1})
            v1 = var.isel(**{str(vertical_dim):levs})
            v2 = var.isel(**{str(vertical_dim):levs-1})
            vnew = ((v1-v2)*dd + v2*z1 - v1*z2) / (z1-z2)
            vnew = vnew.where(levs>0)
            
            vnew.coords['z'] = dd
            vnew = vnew.expand_dims('z')

            zslice_list.append(vnew)
        
        vnew = xr.concat(zslice_list, dim='z')
        
        #raise NotImplementedError('Interpolation on full 3D grid not implemented yet')

    else:
        # Find the grid position of the nearest vertical levels
        levs = (z < depth).sum(vertical_dim)
        levs = levs.where(levs<N, other=N-1)
        #levs = levs.where(levs>0) # <-- invalid indexer array, no integer

        #warnings.warn('{} MB will be loaded into memory!'.format(levs.nbytes*1e-6),Warning)
        levs.load()
        # Do the interpolation
        z1 = z.isel(**{str(vertical_dim):levs})
        z2 = z.isel(**{str(vertical_dim):levs-1})
        v1 = var.isel(**{str(vertical_dim):levs})
        v2 = var.isel(**{str(vertical_dim):levs-1})

        vnew = ((v1-v2)*depth + v2*z1 - v1*z2) / (z1-z2)
        vnew = vnew.where(levs>0)
        #vnew = mask(vnew)
        vnew.coords['z'] = depth
    
    vnew.coords['z'].attrs['long_name'] = 'depth of Z-levels'
    vnew.coords['z'].name = 'z'
    vnew.coords['z'].attrs['units'] = 'meter'
    vnew.coords['z'].attrs['field'] = 'depth, scalar, series'
    vnew.coords['z'].attrs['standard_name'] = 'depth'
    vnew.attrs = var.attrs

    return vnew



def vinterp_anyvar(var, g, g_slice):

    '''
    function  vnew = vinterp(var, g, g_slice)

    This function interpolates a 3D variable on a horizontal level of constant
    variable g

    Parameters
    ----------
    var       xr.DataArray
              Variable to process (3D matrix).
    g         xr.DataArray
              Grid variable on which to slice (3D matrix).
    g_slice   g value of Slice

    Returns
    -------
    vnew    xr.DataArray
            Horizontal slice (2D matrix).
    '''

    if var.shape != g.shape:
        g = g.transpose(*var.dims)
        if var.shape != g.shape:
            raise ValueError('Shape mismatch between Variable and Grid arrays')

    vertical_dim = get_vertical_dimension(var)
    N = len(var[vertical_dim])

    if isinstance(g_slice,list):
        # Loop over g_slice list
        gslice_list = []
        for dd in g_slice:
            # Find the grid position of the nearest vertical levels
            levs = (g < dd).sum(vertical_dim)
            levs = levs.where(levs<N, other=N-1)
            levs.load()
            # Do the interpolation
            g1 = g.isel(**{str(vertical_dim):levs})
            g2 = g.isel(**{str(vertical_dim):levs-1})
            v1 = var.isel(**{str(vertical_dim):levs})
            v2 = var.isel(**{str(vertical_dim):levs-1})
            vnew = ((v1-v2)*dd + v2*g1 - v1*g2) / (g1-g2)
            vnew = vnew.where(levs>0)
            
            vnew.coords[g.name] = dd
            vnew = vnew.expand_dims(g.name)

            zslice_list.append(vnew)
        
        vnew = xr.concat(gslice_list, dim=g.name)
        
        #raise NotImplementedError('Interpolation on full 3D grid not implemented yet')

    else:
        # Find the grid position of the nearest vertical levels
        levs = (g < g_slice).sum(vertical_dim)
        levs = levs.where(levs<N, other=N-1)
        #levs = levs.where(levs>0) # <-- invalid indexer array, no integer

        #warnings.warn('{} MB will be loaded into memory!'.format(levs.nbytes*1e-6),Warning)
        levs.load()
        # Do the interpolation
        g1 = g.isel(**{str(vertical_dim):levs})
        g2 = g.isel(**{str(vertical_dim):levs-1})
        v1 = var.isel(**{str(vertical_dim):levs})
        v2 = var.isel(**{str(vertical_dim):levs-1})

        vnew = ((v1-v2)*g_slice + v2*g1 - v1*g2) / (g1-g2)
        vnew = vnew.where(levs>0)
        #vnew = mask(vnew)
        vnew.coords[g.name] = g_slice

    return vnew



def mask(croco_ds, var):

    '''
    Masks a single variable

    Parameters
    ----------
    var : xarray.DataArray

    Returns
    -------
    masked : xarray.DataArray
             (nan:land)
    '''
    if 'xi_u' in var.coords:
        return var.where(croco_ds.mask_u)
    elif 'eta_v' in var.coords:
        return var.where(croco_ds.mask_v)
    else:
        return var.where(croco_ds.mask_rho)

    
def hslice(croco_ds, var, zlevs, masked=False):
    
    '''
    get a horizontal slice of a CROCO variable

    Parameters
    ----------
    var     xarray.DataArray
            3D or 4D variable array
            
    level   real < 0
            interpolate a horizontal slice at z=level
            
    masked  True or False
            uses variable 'salt' to compute and mask the bottom

    Returns
    -------
    var     xarray.DataArray
            2D or 3D array
    '''
    
    try:
        iter(zlevs)
    except TypeError:
        zlevs = [zlevs]  
        
    #zdata = get_depths(croco_ds, var)
    zkey = [z for z in list(var.coords) if 'z_' in z][0]
    zdata = var[zkey]
    var_zgrid = croco_ds.attrs['xgrid'].transform(var,'Z',zlevs,target_data=zdata)
    var_zgrid = var_zgrid.rename({zkey:'rz'})
    
    if masked:
        mask = croco_ds.attrs['xgrid'].transform(croco_ds.salt,'Z',zlevs,target_data=croco_ds.z_rho)>0
        if any(dimension in var.dims for dimension in ['eta_v','xi_u','s_w']):
            mask = rho2var(croco_ds, mask, var)
        mask = mask.rename({'z_rho':'z'})
        mask.coords['z'] = var_zgrid['z']
            #zcoord = [c for c in var_zgrid.coords if 'z_' in c][0]
            #mask.coords[zcoord] = var_zgrid[zcoord]
        var_zgrid = var_zgrid.where(mask)
    
    return var_zgrid



def vslice(croco_ds, var, zlevels=None, **kwargs):
    
    '''
    get a vertical slice of a CROCO variable
    '''
    
    var = mask(croco_ds, var)
    pm = croco_ds.pm
    pn = croco_ds.pn
    
    if kwargs and len(kwargs) == 2:
        
        dim1, dim2 = list(kwargs.keys())[0], list(kwargs.keys())[1]
        dim1_rho = dim1.split('_')[0] + '_rho'
        dim2_rho = dim2.split('_')[0] + '_rho'
        X, Y = kwargs[dim1], kwargs[dim2]
        Npoints = len(X)

        # Make temporary DataArrays for initial interpolations
        X0 = xr.DataArray(X, dims=["distance"])
        Y0 = xr.DataArray(Y, dims=["distance"])
        interpdim1 = {dim1:X0}
        interpdim2 = {dim2:Y0}
        interpdict = {dim1:X0, dim2:Y0}
        rho_interpdict = {dim1_rho:X0, dim2_rho:Y0}

        if np.sum(['xi' in k or 'eta' in k for k in kwargs]) == 2:
            Xgrid_dist = np.diff(X)
            Ygrid_dist = np.diff(Y)
        
        elif np.sum(['lat' in k for k in kwargs]) == 1 and np.sum(['lon' in k for k in kwargs]) == 1:
        # If lat/lon exist as 1D coordinates, make them dimensions to be able to interpolate
            
            L1, L2 = var[dim1], var[dim2]
            L1_rho, L2_rho = pm[dim1_rho], pm[dim2_rho]
            xi = [dim for dim in var.dims if 'xi' in dim][0]
            eta = [dim for dim in var.dims if 'eta' in dim][0]
            xi_rho = xi.split('_')[0] + '_rho'
            eta_rho = eta.split('_')[0] + '_rho'
            changedims = {}
            changedims_rho = {}

            if L1.diff(dim=xi).sum().data == 0 and L2.diff(dim=eta).sum().data == 0:
                var.coords[dim1] = ((eta), L1.isel(**{xi:0}))
                pm.coords[dim1_rho] = ((eta_rho), L1_rho.isel(**{xi_rho:0}))
                pn.coords[dim1_rho] = ((eta_rho), L1_rho.isel(**{xi_rho:0}))
                changedims[eta] = dim1
                changedims_rho[eta_rho] = dim1_rho
                var.coords[dim2] = ((xi), L2.isel(**{eta:0}))
                pm.coords[dim2_rho] = ((xi_rho), L2_rho.isel(**{eta_rho:0}))
                pn.coords[dim2_rho] = ((xi_rho), L2_rho.isel(**{eta_rho:0}))
                changedims[xi] = dim2
                changedims_rho[xi_rho] = dim2_rho
                
                # Swap the xi/eta for lat/lon dimensions
                var = var.swap_dims(changedims)
                pm = pm.swap_dims(changedims_rho)
                pn = pn.swap_dims(changedims_rho)
                
                Xgrid_dist = np.diff(var[xi].interp(**interpdim2).values)
                Ygrid_dist = np.diff(var[eta].interp(**interpdim1).values)
                
            elif L1.diff(dim=eta).sum().data == 0 and L2.diff(dim=xi).sum().data == 0:
                var.coords[dim1] = ((xi), L1.isel(**{eta:0}))
                pm.coords[dim1_rho] = ((xi_rho), L1_rho.isel(**{eta_rho:0}))
                pn.coords[dim1_rho] = ((xi_rho), L1_rho.isel(**{eta_rho:0}))
                changedims[xi] = dim1
                changedims_rho[xi_rho] = dim1_rho
                var.coords[dim2] = ((eta), L2.isel(**{xi:0}))
                pm.coords[dim2_rho] = ((eta_rho), L2_rho.isel(**{xi_rho:0}))
                pn.coords[dim2_rho] = ((eta_rho), L2_rho.isel(**{xi_rho:0}))
                changedims[eta] = dim2
                changedims_rho[eta_rho] = dim2_rho
                
                # Swap the xi/eta for lat/lon dimensions
                var = var.swap_dims(changedims)
                pm = pm.swap_dims(changedims_rho)
                pn = pn.swap_dims(changedims_rho)
            
                Xgrid_dist = np.diff(var[xi].interp(**interpdim1).values)
                Ygrid_dist = np.diff(var[eta].interp(**interpdim2).values)
                
            else:
                raise NotImplementedError('Interpolating on a rotated or curvilinear lat/lon grid not yet supported. \nPlease specify the section on the eta/xi grid.')

        else:
            raise TypeError('Please specify either xi/eta or lat/lon dimensions')
    else:
        raise AttributeError('Please specify a series of points in 2 dimensions')

    # Distance between section points
    pm = pm.interp(**rho_interpdict).values
    pn = pn.interp(**rho_interpdict).values
    dX = 2 * Xgrid_dist / (pm[:-1] + pm[1:])
    dY = 2 * Ygrid_dist / (pn[:-1] + pn[1:])
    dS = np.sqrt(dX * dX + dY * dY)
    # Cumulative distance along the section
    distance = np.concatenate(([0], np.cumsum(dS))) / 1000.0  # unit = km
    X0["distance"] = distance
    Y0["distance"] = distance

    # Interpolate to the section making an intermediate Dataset
    B0 = var.interp(**interpdict)
    
    B0['distance'].attrs['long_name'] = 'distance along section'
    B0['distance'].name = 'distance'
    B0['distance'].attrs['units'] = 'kilometer'
    B0['distance'].attrs['field'] = 'distance, scalar, series'
    B0['distance'].attrs['standard_name'] = 'distance'
    
    # Remove distance dimension for single station
    if B0['distance'].size == 1:
        B0 = B0.isel(**{'distance':0})
        del B0['distance']
    
    if "s_rho" in var.coords:
        # Transpose dimensions for z_rho
        dimlist = list(B0.dims); dimlist.remove('s_rho'); dimlist = ['s_rho'] + dimlist
        B0['z_rho'] = B0['z_rho'].transpose(*dimlist)
    
    # Transform the vertical coordinate if Z-levels are specified
    try:
        zswitch = any(zlevels)
    except TypeError:
        zswitch = bool(zlevels)

    if zswitch:
        D0 = xr.Dataset()
        D0.attrs['xgrid'] = Grid(B0, periodic=False)
        B0z = hslice(D0, B0, zlevels, masked=False)
        return B0z
    else:
        return B0


def update_progress(progress):
    bar_length = 40
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    text = "[{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    sys.stdout.write("\r" + str(text))
    sys.stdout.flush()
    if progress == 1:
        print('\n')
        
        
def w2omega(croco_ds, xgrid=None):
    
    if not xgrid:
        xgrid = croco_ds.attrs['xgrid']
    
    Wet = croco_ds.v * xgrid.diff(croco_ds.z_rho, 'Y') * xgrid.interp(croco_ds.pn, 'Y')
    Wxi = croco_ds.u * xgrid.diff(croco_ds.z_rho, 'X') * xgrid.interp(croco_ds.pm, 'X')
    
    w = croco_ds.w - xgrid.interp(Wxi, 'X') - xgrid.interp(Wet, 'Y')
    bound = (croco_ds.w.eta_rho == 0) + (croco_ds.w.eta_rho == croco_ds.w.eta_rho[-1]) + (croco_ds.w.xi_rho == 0) + (croco_ds.w.xi_rho == croco_ds.w.xi_rho[-1])
    w = w.where(np.invert(bound),other=croco_ds.w.where(bound))
    z_r = croco_ds.z_rho
    z_w = croco_ds.z_w
    
    zw_center = z_w #.sel(**{'s_w':croco_ds.s_w[1:-1]});
    zr_left = z_r.sel(**{'s_rho':croco_ds.s_rho[:-1]}); del zr_left.coords['s_rho']; zr_left = zr_left.rename({'s_rho':'s_w'}); zr_left.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:-1]});
    zr_right = z_r.sel(**{'s_rho':croco_ds.s_rho[1:]}); del zr_right.coords['s_rho']; zr_right = zr_right.rename({'s_rho':'s_w'}); zr_right.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:-1]});
    w_left = w.sel(**{'s_rho':croco_ds.s_rho[:-1]}); del w_left.coords['s_rho']; w_left = w_left.rename({'s_rho':'s_w'}); w_left.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:-1]});
    w_right = w.sel(**{'s_rho':croco_ds.s_rho[1:]}); del w_right.coords['s_rho']; w_right = w_right.rename({'s_rho':'s_w'}); w_right.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:-1]});

    # interpolation on w grid conserving volume flux
    omega = ( (zr_right-zw_center) * w_right + (zw_center-zr_left) * w_left ) / (zr_right-zr_left)
    # set top and bottom layers to zero
    omega = xr.concat((xr.zeros_like(croco_ds.z_w.isel(s_w=0)), omega, xr.zeros_like(croco_ds.z_w.isel(s_w=-1))),'s_w')
    
    return omega


def croco_horiz_trcflux(croco_ds, var, dim, Flx, xgrid=None):
    
    if not xgrid:
        xgrid = croco_ds.attrs['xgrid']
    
    coor = xgrid.axes[dim].coords['center']
    coor_new = xgrid.axes[dim].coords['inner']
    
    mask = xgrid.interp(croco_ds.mask_rho,dim,boundary='extend')
    curv = xgrid.diff((xgrid.diff(var,dim)*mask),dim)
    curv_left = curv.sel(**{coor:croco_ds[coor][:-1]}); del curv_left.coords[coor]; curv_left = curv_left.rename({coor:coor_new});
    curv_right = curv.sel(**{coor:croco_ds[coor][1:]}); del curv_right.coords[coor]; curv_right = curv_right.rename({coor:coor_new});
    TXadv = -xgrid.diff(xgrid.interp(var,dim) * Flx - 0.166666666666 * (curv_left * Flx.where(Flx>0,other=0) + curv_right * Flx.where(Flx<0,other=0)), dim) * croco_ds.mask_rho
    Ttrun = -xgrid.diff((0.041666667 * xgrid.diff(curv,dim) * np.abs(Flx)),dim) * croco_ds.mask_rho
    
    return TXadv, Ttrun


def croco_vert_trcflux(croco_ds, var, W, xgrid=None):
    
    if not xgrid:
        xgrid = croco_ds.attrs['xgrid']
    
    epsil = 1e-16

    FC = xgrid.diff(var,'Z'); bot = FC.isel(s_w=1); del bot.coords['s_w']; bot.coords['s_w'] = FC.coords['s_w'].isel(s_w=0); top = FC.isel(s_w=-2); del top.coords['s_w']
    bot_out = FC.isel(s_w=0); top_out = FC.isel(s_w=-1)
    FC = FC.sel(**{'s_w':croco_ds.s_w[1:-1]}); output_dims = FC.dims
    FC = xr.concat([bot.expand_dims('s_w'),FC,top.expand_dims('s_w')],dim='s_w').transpose(*output_dims)

    FC_left = FC.sel(**{'s_w':croco_ds.s_w[:-1]}); del FC_left.coords['s_w']; FC_left.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:]});
    FC_right = FC.sel(**{'s_w':croco_ds.s_w[1:]}); del FC_right.coords['s_w']; FC_right.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:]});
    cff = 2 * (FC_right*FC_left)
    CF = cff / (FC_right+FC_left)
    CF = CF.where(cff >= epsil, other=0)

    CF_left = CF.sel(**{'s_w':croco_ds.s_w[1:-1]});
    CF_right = CF.sel(**{'s_w':croco_ds.s_w[2:]}); del CF_right.coords['s_w']; CF_right.coords['s_w'] = croco_ds['s_w'].sel(**{'s_w':croco_ds.s_w[1:-1]});
    FC = 0.5 * (2 * xgrid.interp(var,'Z') - 0.333333 * (CF_right-CF_left)) * W; del FC.coords['z_w']
    FC = xr.concat([bot_out.expand_dims('s_w'),FC,top_out.expand_dims('s_w')],dim='s_w').transpose(*output_dims)

    Tvadv = -xgrid.diff(FC,'Z')

    return Tvadv
