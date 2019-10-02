import xarray as xr
import pandas as pd
from xgcm import Grid
import numpy as np
from scipy.interpolate import griddata
from typing import Sequence
import dask
import warnings
import cartopy.crs as ccrs
import glob
import time, sys

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
        
def get_rhovar(D):
    maxdims = np.max(np.array([len(D[var].dims) for var in list(D.data_vars)]))
    ll = list(D.data_vars)
    for var in ll:
        vardims = D[var].dims
        if 'eta_rho' in vardims and 'xi_rho' in vardims and len(vardims) == maxdims:
            rhovar = var
            break
    return rhovar
    
def croco_dataset(model_output, time_dim='time', grid=None, *args, **kwargs):

    if isinstance(model_output, xr.Dataset):
        da = model_output
    else:
        da = xr.open_mfdataset(model_output)
    
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
        
    # Overwrite xi / eta variables in case they are not continuousv
    for co in ['xi_rho','xi_u','eta_rho','eta_v']:
        co_attr = da2[co].attrs
        da2.coords['a'] = (co, np.arange(da2[co].shape[0]).astype('float'))
        da2 = da2.set_index(**{co:'a'})
        da2[co].attrs = co_attr
    
    #da2.xi_rho.data = (np.arange(da2.xi_rho.shape[0])+1).astype('float')
    #da2.xi_u.data = (np.arange(da2.xi_u.shape[0])+1).astype('float')
    #da2.eta_rho.data = (np.arange(da2.eta_rho.shape[0])+1).astype('float')
    #da2.eta_v.data = (np.arange(da2.eta_v.shape[0])+1).astype('float')

    # move lat/lon to coordinates
    latlon_vars = [v for v in list(da2.data_vars) if 'lat' in v or 'lon' in v]
    for v in latlon_vars:
        da2.coords[v] = da2[v]
        #da2 = da2.drop(v)
    da2.attrs = da.attrs
    da2.attrs['xgcm-Grid'] = Grid(da2, periodic=False)

    
    # Read grid parameters depending on version
    # TODO: See how this version checking (taken from ROMSTOOLS) is done in the new official CROCOTOOLS 
    if np.size(da2.Tcline) is 0:
        # 'UCLA version'
        pass
    else:
        # 'AGRIF version'
        hmin = np.nanmin(da2.h)
        try:
            da2.hc.data = np.min([hmin, da2.Tcline])
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
                    da2.attrs.s_coord = int(vtrans)
                else:
                    da2.attrs.s_coord = int(vtrans[0])
        except KeyError:
            da2.attrs.s_coord = 1
    if da2.attrs.s_coord == 2:
        da2['hc'] = da2.Tcline
    
    # Define Z coordinates
    rhovar = get_rhovar(da2)
    da2.coords['z_rho'] = zlevs(da2, da2[rhovar])
    
    zr = da2['z_rho']
    zr.name = 'z_rho'
    zr.attrs['long_name'] = 'depth at RHO-points'
    zr.attrs['units'] = 'meter'
    zr.attrs['field'] = 'depth, scalar, series'
    zr.attrs['positive'] = 'up'
    zr.attrs['standard_name'] = 'depth'

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

def get_vertical_dimension(var):
    try:
        return [d for d in var.dims if 's_' in d][0]
    except IndexError:
        raise IndexError('{} is a 2D-H variable'.format(var.name))

def zlevs(croco_ds, var, **kwargs):

    '''
    this method computes the depth of rho or w points for ROMS

    Parameters
    ----------
    croco_ds: xr.Dataset
              Provides the variables h and zeta
    var : xr.DataArray
          on arbitrary grid (rho, u, v or w)

    Returns
    -------
    z : xr.DataArray
        Depths (m) of RHO- or W-points (3D matrix).
    '''

    if 's_w' in var.dims:
        typ = 'w'
    else:
        typ = 'r'
        
    input_dims = var.dims
    input_dims = ['xi_rho' if x=='xi_u' else x for x in input_dims]
    input_dims = ['eta_rho' if x=='eta_v' else x for x in input_dims]

    h = croco_ds.h
    try:
        hc = croco_ds.hc.isel(time=0).data
    except ValueError:
        hc = croco_ds.hc.data
    N = croco_ds.s_rho.size

    # find zeta time slices that correspond to input array
    # check time dim so it works with time averages
    if kwargs and 'zeta' in kwargs:
        zeta = kwargs['zeta']
    elif 'time' in var.dims:
        zeta = croco_ds.zeta.sel(time=var.time)
    elif 'time' in croco_ds.zeta.dims:
        zeta = croco_ds.zeta.mean('time')
    else:
        zeta = croco_ds.zeta

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

    sc_r = np.zeros([N, 1])
    Cs_r = np.zeros([N, 1])
    sc_w = np.zeros([N+1, 1])
    Cs_w = np.zeros([N+1, 1])

    if vtransform == 2:
        ds = 1 / N
        if typ is 'w':
            sc_w[0] = -1.0
            sc_w[N] = 0
            Cs_w[0] = -1.0
            Cs_w[N] = 0
            sc_w[1:N] = ds * (np.arange(1, N) - N)
            Cs_w = csf(sc_w)
            N += 1
     
            sc = ds * (np.arange(1, N+1)-N-0.5)
            Cs_r = csf(sc)
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
    #TODO: include option to choose dask array here
    if typ is 'w':
        z = xr.zeros_like(var)
        z.attrs['long_name'] = 'depth at W-points'
        z.name = 'z_w'
    else:
        if 'time' in var.dims:
            z = xr.zeros_like(croco_ds.temp.sel(time=var.time))
        elif 'time' in croco_ds.temp.dims:
            z = xr.zeros_like(croco_ds.temp.mean('time'))
        else:
            z = xr.zeros_like(croco_ds.temp)
        z.attrs['long_name'] = 'depth at RHO-points'
        z.name = 'z_rho'
    z.attrs['units'] = 'meter'
    z.attrs['field'] = 'depth, scalar, series'
    z.attrs['standard_name'] = 'depth'

    vertical_dim = get_vertical_dimension(var)

    if vtransform == 2:
        if typ is 'w':
            cff1 = Cs_w
            cff2 = sc_w+1
            sc = sc_w
        else:
            cff1 = Cs_r
            cff2 = sc_r+1
            sc = sc_r

        h2 = (h+hc)
        cff = hc*sc
        h2inv = 1. / h2
        cff1_xr = xr.DataArray(cff1,dims=(vertical_dim,))
        cff_xr = xr.DataArray(cff,dims=(vertical_dim,))

        z0 = cff_xr + cff1_xr * h
        z = zeta * (1. + z0 * h2inv) + z0 * h / h2
        z = z.transpose(*input_dims)

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
        z = z.transpose(*input_dims)

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
    grid = croco_ds.attrs['xgcm-Grid']
    if 'xi_u' in var.dims:
        var_rho = grid.interp(var,'X', boundary='extend')
    elif 'eta_v' in var.dims:
        var_rho = grid.interp(var,'Y', boundary='extend')
        add_coords(var_rho, latlon_rho)
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
    add_coords(var_rho, latlon_rho)

    return var_rho

def rho2var(croco_ds, var_rho, var_target):

    if any(dimension in var_rho.dims for dimension in ['eta_v','xi_u','s_w']):
        raise ValueError('Input array is not on RHO-grid')
    grid = croco_ds.attrs['xgcm-Grid']

    if 'xi_u' in var_target.dims:
        var = grid.interp(var_rho,'X')
        add_coords(croco_ds, var, ['lat_u','lon_u'])
        var.attrs = var_rho.attrs
        var.name = var_rho.name
        var.attrs['long_name'] = '{} at U-points'.format(var.name)
    elif 'eta_v' in var_target.dims:
        var = grid.interp(var_rho,'Y')
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

    depths = zlevs(croco_ds, var)
    if any(dimension in var.dims for dimension in ['eta_v','xi_u']):
        depths = rho2var(croco_ds, depths, var)

    return depths


def valid_levels(levels):
    
    '''
    Make the specified levels compatible with vinterp() if possible
    '''
    
    if not bool(list(levels)):
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
        #display(var)
        #display(z)
        z = z.transpose(*var.dims)
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
    
 #   vnew.coords['z'].attrs['long_name'] = 'g_slice of Z-levels'
 #   vnew.coords['z'].name = 'z'
 #   vnew.coords['z'].attrs['units'] = 'meter'
 #   vnew.coords['z'].attrs['field'] = 'g_slice, scalar, series'
 #   vnew.coords['z'].attrs['standard_name'] = 'g_slice'
 #   vnew.attrs = var.attrs

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

    
def hslice(croco_ds, var, level):

    '''
    get a horizontal slice of a CROCO variable

    Parameters
    ----------
    var     xarray.DataArray
            3D or 4D variable array
            
    level   real < 0 or real > 0 
            vertical level of the slice (scalar), interpolate a horizontal slice at z=level

    Returns
    -------
    var     xarray.DataArray
            2D or 3D array
    '''

    vertical_dim = get_vertical_dimension(var)

    #
    # Get a horizontal level of a 3D variable
    #

    # Get the depths of the sigma levels
    z = get_depths(croco_ds, var)

    # Do the interpolation
    if 'time' in var.dims and var.time.shape[0] > 1:
        timesteps = var.time.shape[0]
        print("Looping over time dimension...")
        slicelist = []
        update_progress(0)
        for tt in range(timesteps):
            timesl = vinterp(var.isel(time=tt), z.isel(time=tt), level)
            try:
                timesl.compute()
            except AttributeError:
                pass
            slicelist.append(timesl)
            update_progress((tt+1)/timesteps)
        vnew = xr.concat(slicelist, dim='time')
    else:
        vnew = vinterp(var, z, level)
    vnew.coords['depth'] = np.array(level).astype('float32')
        
    vnew = mask(croco_ds, vnew)
    vnew.attrs = var.attrs
    vnew.name = var.name
    
    return vnew


def vslice(croco_ds, var, **kwargs):
    
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

    if "s_rho" not in var.coords:
        raise ValueError("Section requires vertical dimension s_rho")

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
    
    # Transpose dimensions for z_rho
    dimlist = list(B0.dims); dimlist.remove('s_rho')
    dimlist = ['s_rho'] + dimlist
    B0['z_rho'] = B0['z_rho'].transpose(*dimlist)
    
    return B0
