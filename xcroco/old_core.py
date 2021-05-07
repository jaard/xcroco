########## Old code from here ##########


class Xcroco:

    '''
    Wrapper class to store all data and methods needed for
    basic analysis of a ROMS/CROCO (+PISCES) simulation. All output is given
    as xarray.DataArray objects including coordinates present
    in original output file.
    
    #TODO
    - add functionality to get_depths that buffers z-levels
      if they have been used before
    - implement get_section so that rho points are extrapolated
      to free surface; interpolate lat/lon correctly, this is 
      not done so far
    - change rho2var into rho2var(DataArray, 'grid')
      no dataArray as input, adapt get_depths
    - generalize calc_optic_romspisces for output that does not contain the variable AKt'
    '''

    def __init__(self, outfile, gridfile, *args, **kwargs):

        # open the ROMS output file or list of files
        if type(outfile) is list:
            if 'chunks' in kwargs:
                self.chunks = kwargs['chunks']
            else:
                self.chunks = {'time':1,'s_rho':32, 'eta_rho':193,'xi_rho':245}
            ds = xr.open_mfdataset(outfile, chunks=self.chunks, decode_times=False, decode_cf=False)
            self.dataset = xr.decode_cf(ds)
        elif type(outfile) is str:
            self.chunks = None
            self.dataset = xr.open_dataset(outfile)
        elif type(outfile) is xr.core.dataset.Dataset:
            self.dataset = outfile
        else:
            raise ValueError('Please provide a path or an opened xarray dataset')
            
        # open the diabio file
        if args:
            self.dataset_diabio = xr.open_dataset(args[0])
            self.dataset_diabio.coords['time'] = self.dataset.time

        # open the ROMS gridfile
        if type(gridfile) is str:
            self.gridfile = xr.open_dataset(gridfile)
        elif type(gridfile) is xr.core.dataset.Dataset:
            self.gridfile = gridfile
        else:
            raise ValueError('Please provide a path or an opened xarray dataset')

        # create an xgcm grid object
        self.grid = Grid(self.dataset, periodic=False)

        self.lon = self.dataset.lon_rho
        self.lat = self.dataset.lat_rho
        
        self.h = self.gridfile.h
        self.zeta = self.dataset.zeta
        
        # get the masks for rho, u and v grids
        try:
            self.mask_rho = self.gridfile.mask_rho
        except AttributeError:
            try:
                self.mask_rho = self.dataset.mask_rho
            except AttributeError:
                self.mask_rho = 1 + 0 * self.lon
        if self.mask_rho.size is 0:
            self.mask_rho = 1 + 0 * self.lon
        
        #TODO: ADD TRY-CATCH BLOCKS HERE
        self.mask_u = self.gridfile.mask_u.rename({'eta_u': 'eta_rho'})
        self.mask_v = self.gridfile.mask_v.rename({'xi_v': 'xi_rho'})
            
        # Read grid parameters depending on version
        # 'AGRIF/UCLA version'
        self.theta_s = self.dataset.theta_s
        self.theta_b = self.dataset.theta_b
        self.Tcline = self.dataset.Tcline
        #print(self.dataset.hc)
        #self.hc = float(self.dataset.hc)
        self.hc = self.dataset.hc

        if np.size(self.Tcline) is 0:
            # 'UCLA version'
            self.hc = self.dataset.hc
        else:
            hmin = np.nanmin(self.h)
            self.hc = np.min([hmin, self.Tcline])

        self.N = self.dataset.s_rho.size
        self.T = self.dataset.time.size
        
        # Check which s-coordinates are used
        # 1... old, 2...new
        try:
            VertCoordType = self.dataset.VertCoordType
            if VertCoordType == 'NEW':
                self.s_coord = 2
        except AttributeError:
            try:
                vtrans = self.dataset.Vtransform
                if vtrans.size is not 0:
                    if vtrans.size == 1:
                        self.s_coord = int(vtrans)
                    else:
                        # self.s_coord = int(vtrans.isel(time=0))   <-- fails without time dimension
                        self.s_coord = int(vtrans[0])
            except KeyError:
                self.s_coord = 1
        if self.s_coord == 2:
            self.hc = self.Tcline

        # Check if zeta is available, otherwise set to zero
        if self.zeta.size is 0:
            self.zeta = self.h * 0


    def mask(self, var):
        
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
            return var.where(self.mask_u==1)
        elif 'eta_v' in var.coords:
            return var.where(self.mask_v==1)
        else:
            return var.where(self.mask_rho==1)
        
        
    def mask_all(self):
        
        '''
        Masks all variables of the dataset
        with more than 2 dimensions.

        WARNING: Can take a long time!
        '''

        var_keylist = [v for v in list(self.dataset) if len(self.dataset[v].dims)>2]
        for var_key in var_keylist:
            var = self.dataset[var_key]
            if 'xi_u' in var.coords:
                self.dataset[var_key] = var.where(self.mask_u==1)
            elif 'eta_v' in var.coords:
                self.dataset[var_key] = var.where(self.mask_v==1)
            else:
                self.dataset[var_key] = var.where(self.mask_rho==1)
        

    def _read_mask(var):

        '''
        Read the latitude, the longitude
        and the mask from a ROMS variable

        Parameters
        ----------
        var : xarray.DataArray
              
        Returns
        -------
        mask : xarray.DataArray
               Mask (1:sea - nan:land)
        '''

        if typ == 'u':
            lat = rho2u_2d(lat)
            lon = rho2u_2d(lon)
            mask = mask[:, :Lp-1] * mask[:, 1: Lp]
        elif typ == 'v':
            lat = rho2v_2d(lat)
            lon = rho2v_2d(lon)
            mask = mask[:Mp-1, :] * mask[1: Mp, :]
        mask[mask == 0] = 'nan'

        return mask


    def _add_coords(self, var, coords):
        for co in coords:
            var.coords[co] = self.dataset.coords[co]
    
    def var2rho(self, var):

        '''
        Interpolate any variable to RHO points,
        position on grid is checked automatically
        '''
        latlon_rho = ['lon_rho','lat_rho']
        if 'xi_u' in var.dims:
            var_rho = self.grid.interp(var,'X', boundary='extend')
        elif 'eta_v' in var.dims:
            var_rho = self.grid.interp(var,'Y', boundary='extend')
            self._add_coords(var_rho, latlon_rho)
            var_rho.attrs = var.attrs
            var_rho.name = var.name
        elif 's_w' in var.dims:
            var_rho = self.grid.interp(var,'Z')
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
        self._add_coords(var_rho, latlon_rho)

        return var_rho
    
    def rho2var(self, var_rho, var_target):

        if any(dimension in var_rho.dims for dimension in ['eta_v','xi_u','s_w']):
            raise ValueError('Input array is not on RHO-grid')
        
        if 'xi_u' in var_target.dims:
            var = self.grid.interp(var_rho,'X')
            self._add_coords(var, ['lat_u','lon_u'])
            var.attrs = var_rho.attrs
            var.name = var_rho.name
            var.attrs['long_name'] = '{} at U-points'.format(var.name)
        elif 'eta_v' in var_target.dims:
            var = self.grid.interp(var_rho,'Y')
            self._add_coords(var, ['lat_v','lon_v'])
            var.attrs = var_rho.attrs
            var.name = var_rho.name
            var.attrs['long_name'] = '{} at V-points'.format(var.name)
        elif 's_w' in var_target.dims:
            var = self.grid.interp(var_rho,'Z', boundary='extend')
            self._add_coords(var, ['lat_rho','lon_rho'])
            var.attrs = var_rho.attrs
            var.name = var_rho.name
            var.attrs['long_name'] = '{} at W-points'.format(var.name)
        else:
            warnings.warn('Both arrays already at rho points, returning input array.', Warning)
            return var_rho 

        return var
        
    def get_depths(self, var):

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
 
        depths = self._zlevs(var)
        if any(dimension in var.dims for dimension in ['eta_v','xi_u']):
            depths = self.rho2var(depths, var)
            
        #depths = self.mask(depths)

        return depths

    @staticmethod
    def __spheric_dist(lat1, lat2, lon1, lon2):

        '''
        compute distances for a simple spheric earth

        Parameters
        ----------
        lat1 : float
            latitude of first point (matrix)
        lon1 : float
            longitude of first point (matrix)
        lat2 : float
            latitude of second point (matrix)
        lon2 : float
            longitude of second point (matrix)

        Returns
        -------
        dist : distance from first point to second point (matrix)
        '''

        R = 6367442.76

        # Determine proper longitudinal shift.
        londiff = np.array(abs(lon2-lon1))
        londiff[londiff >= 180] = 360 - londiff[londiff >= 180]

        # Convert Decimal degrees to radians.
        deg2rad = np.pi / 180
        lat1 = lat1 * deg2rad
        lat2 = lat2 * deg2rad
        londiff = londiff * deg2rad

        # Compute the distances

        dist = (R * np.sin(np.sqrt(((np.sin(londiff) * np.cos(lat2)) ** 2) +
                (((np.sin(lat2) * np.cos(lat1)) -
                  (np.sin(lat1) * np.cos(lat2) * np.cos(londiff))) ** 2))))

        return dist
    
    def __oacoef(self, londata, latdata, lon, lat, ro=5e5):

        '''
        def oacoef(londata, latdata, lon, lat, ro)
        function extrfield = oacoef(londata,latdata,lon,lat,ro)

        compute an objective analysis on a scalar field.

        Parameters
        ----------
        londata   : longitude of data points (vector)
        latdata   : latitude of data points (vector)
        lon       : longitude of the estimated points (vector)
        lat       : latitude of the estimated points (vector)
        ro        : decorrelation scale

        Returns
        -------
        coef : oa matrix
        extrfield = mdata+coef*(data-mdata)
        '''

        i = np.arange(0, len(londata))
        j = np.arange(0, len(lon))
        I, J = np.meshgrid(i, i)
        r1 = self.__spheric_dist(latdata[I], latdata[J], londata[I], londata[J])

        I, J = np.meshgrid(i, j)
        r2 = self.__spheric_dist(lat[J], latdata[I], lon[J], londata[I])

        # np.linalg.lstsq(B, b)
        B = np.array(np.exp(-r2 / ro))
        A = np.array(np.exp(-r1 / ro))
        # coef = B / A
        coef = np.linalg.lstsq(A.T, B.T)[0].T

        return coef

    def __csf(self, sc):

        '''
        function h = csf(sc, theta_s,theta_b)
        '''

        if self.theta_s > 0:
            csrf = (1-np.cosh(sc * self.theta_s)) / (np.cosh(self.theta_s)-1)
        else:
            csrf = -sc ** 2
        if self.theta_b > 0:
            h = (np.exp(self.theta_b * csrf)-1) / (1-np.exp(-self.theta_b))
        else:
            h = csrf

        return h

    def _zlevs(self, var, **kwargs):

        '''
        this method computes the depth of rho or w points for ROMS

        Parameters
        ----------
        var : xr.DataArray
              on arbitrary grid (rho, u, v or w)
        zeta (optional): if not provided, self.dataset.zeta is used
                         must have the same shape as input var

        Returns
        -------
        z : xr.DataArray
            Depths (m) of RHO- or W-points (3D matrix).
        '''
        
        if 's_w' in var.dims:
            typ = 'w'
        else:
            typ = 'r'
        
        h = self.h
        N = self.N
        
        # find zeta time slices that correspond to input array
        # check time dim so it works with time averages
        if kwargs and 'zeta' in kwargs:
            zeta = kwargs['zeta']
        elif 'time' in var.dims:
            zeta = self.zeta.sel(time=var.time)
        elif 'time' in self.zeta.dims:
            zeta = self.zeta.mean('time')
        else:
            zeta = self.zeta
        
        hshape = np.shape(h)
        if len(hshape) == 1:
            L = hshape[0]
            M = 1
        else:
            L, M = np.shape(h)
            
        try:
            vtransform = self.s_coord
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
                Cs_w = self.__csf(sc_w)
                N += 1
            else:
                sc = ds * (np.arange(1, N+1)-N-0.5)
                Cs_r = self.__csf(sc)
                sc_r = sc

        elif vtransform == 1:
            cff1 = 1. / np.sinh(self.theta_s)
            cff2 = 0.5 / np.tanh(0.5 * self.theta_s)
            if typ is 'w':
                sc = (np.arange(0, N+1)-N) / N
                N += 1
            else:
                sc = (np.arange(1, N+1)-N-0.5) / N

            Cs = ((1. - self.theta_b) * cff1 * np.sinh(self.theta_s * sc)
                  + self.theta_b * (cff2 * np.tanh(self.theta_s * (sc+0.5))-0.5))

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
        else:
            if 'time' in var.dims:
                z = xr.zeros_like(self.dataset.temp.sel(time=var.time))
            elif 'time' in self.dataset.temp.dims:
                z = xr.zeros_like(self.dataset.temp.mean('time'))
            else:
                z = xr.zeros_like(self.dataset.temp)
            z.attrs['long_name'] = 'depth at RHO-points'
        z.name = 'depth'
        z.attrs['units'] = 'meter'
        z.attrs['field'] = 'depth, scalar, series'
        z.attrs['standard_name'] = 'depth'

        vertical_dim = self._get_vertical_dimension(var)
        
        if vtransform == 2:
            if typ is 'w':
                cff1 = Cs_w
                cff2 = sc_w+1
                sc = sc_w
            else:
                cff1 = Cs_r
                cff2 = sc_r+1
                sc = sc_r

            h2 = (h+self.hc)
            cff = self.hc*sc
            h2inv = 1. / h2
            cff1_xr = xr.DataArray(cff1,dims=(vertical_dim,))
            cff_xr = xr.DataArray(cff,dims=(vertical_dim,))

            z0 = cff_xr + cff1_xr * h
            z = zeta * (1. + z0 * h2inv) + z0 * h / h2
            z = z.transpose(*var.dims)

        elif vtransform == 1:
            cff1 = Cs
            cff2 = sc + 1
            cff = self.hc * (sc-Cs)
            cff2 = sc + 1
            cff1_xr = xr.DataArray(cff1,dims=(vertical_dim,))
            cff_xr = xr.DataArray(cff,dims=(vertical_dim,))

            z0 = cff_xr + cff1_xr * h
            z = zeta * (1. + z0 * hinv) + z0
            z = z.transpose(*var.dims)

        return z
    
    @staticmethod
    def _get_vertical_dimension(var):
        try:
            return [d for d in var.dims if 's_' in d][0]
        except IndexError:
            raise IndexError('{} is a 2D-H variable'.format(var.name))
            
    
    def _vinterp(self, var, z, depth):

        '''
        function  vnew = vinterp(var,depth)

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

        if var.shape != z.shape:
            #display(var)
            #display(z)
            raise ValueError('Shape mismatch between Variable and Depth arrays')

        vertical_dim = self._get_vertical_dimension(var)
        N = len(var[vertical_dim])
        
        if isinstance(depth,list):
            raise NotImplementedError('Interpolation on full 3D grid not implemented yet')
            # Find the grid position of the nearest vertical levels
            for dep in depth:
                levs = (z < depth).sum(vertical_dim)
                levs = levs.where(levs<N, other=N-1)
        else:
            # Find the grid position of the nearest vertical levels
            levs = (z < depth).sum(vertical_dim)
            levs = levs.where(levs<N, other=N-1)
            #levs = levs.where(levs>0) # <-- invalid indexer array, no integer
            
            warnings.warn('{} MB will be loaded into memory!'.format(levs.nbytes*1e-6),Warning)
            levs.load()
            # Do the interpolation
            z1 = z.isel(**{str(vertical_dim):levs})
            z2 = z.isel(**{str(vertical_dim):levs-1})
            v1 = var.isel(**{str(vertical_dim):levs})
            v2 = var.isel(**{str(vertical_dim):levs-1})

            vnew = ((v1-v2)*depth + v2*z1 - v1*z2) / (z1-z2)
            vnew = vnew.where(levs>0)
            vnew = self.mask(vnew)

        return vnew
    
    
    def _isointerp(self, var, z, isovalue):

        '''
        function  vnew = vinterp(var,depth)

        This function interpolates the depth on value horizontal level of constant
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

        if var.shape != z.shape:
            #display(var)
            #display(z)
            raise ValueError('Shape mismatch between Variable and Depth arrays')

        vertical_dim = self._get_vertical_dimension(var)
        N = len(var[vertical_dim])
        
        if isinstance(isovalue,list):
            raise NotImplementedError('Interpolation on full 3D grid not implemented yet')

        else:
            # Find the grid position of the nearest vertical levels
            levs = (var < isovalue).sum(vertical_dim)
            levs = levs.where(levs<N, other=N-1)
            #levs = levs.where(levs>0) # <-- invalid indexer array, no integer
            
            warnings.warn('{} MB will be loaded into memory!'.format(levs.nbytes*1e-6),Warning)
            levs.load()
            # Do the interpolation
            z2 = z.isel(**{str(vertical_dim):levs})
            z1 = z.isel(**{str(vertical_dim):levs-1})
            v2 = var.isel(**{str(vertical_dim):levs})
            v1 = var.isel(**{str(vertical_dim):levs-1})

            #vnew = ((v1-v2)*depth + v2*z1 - v1*z2) / (z1-z2)
            isodep = ((z1-z2)*isovalue + z2*v1 - z1*v2) / (v1-v2)
            #isodep = isodep.where(levs>0)
            #isodep = self.mask(isodep)

        return isodep


    def get_hslice(self, var, level):

        '''
        function var=get_hslice(fname,gname,vname,tindex,level,typ)
        get an horizontal slice of a ROMS variable

        Parameters
        ----------
        var     xarray.DataArray
                3D or 4D array
        level    vertical level of the slice (scalar):
             level =   integer >= 1 and <= N
                       take a slice along a s level (N=top))
             level =   real < 0
                       interpole a horizontal slice at z=level
        typ    type of the variable (character):
             r for 'rho' for zeta, temp, salt, w(!)
             w for 'w'   for AKt
             u for 'u'   for u, ubar
             v for 'v'   for v, vbar

        Returns
        -------
        var     xarray.DataArray
                2D or 3D array
        '''
        
        vertical_dim = self._get_vertical_dimension(var)
        
        if level == 0:
            #
            # 2D variable
            #
            vnew = var
            warnings.warn('Please specify a positive or negative number', Warning)
        elif level > 0:
            #
            # Get a sigma level of a 3D variable
            #
            vnew = var.isel(**{str(vertical_dim):level})
        else:
            #
            # Get a horizontal level of a 3D variable
            #
            # Get the depths of the sigma levels
            #
            z = self.get_depths(var)
            #
            # Do the interpolation
            #
            vnew = self._vinterp(var, z, level)

        vnew.coords['depth'] = np.array(level).astype('float32')
        #vnew = vnew.expand_dims('depth')
        vnew = self.mask(vnew)
        
        vnew.attrs = var.attrs
        vnew.name = var.name

        return vnew
    
    
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
    
    def get_isosurface(self, var, value):

        '''
        Compute the depth of an isosurface of the given variable at
        the specified value
        '''
        
        vertical_dim = self._get_vertical_dimension(var)
        
        # Get the depths of the sigma levels
        #
        z = self.get_depths(var)
        #
        # Do the interpolation
        #
        isodep = self._isointerp(var, z, value)

        isodep.coords[var.name] = np.array(value).astype('float32')
        #vnew = vnew.expand_dims('depth')
        isodep = self.mask(isodep)
        
        isodep.attrs = z.attrs
        isodep.name = z.name
        
        return isodep
        
    def get_section(self, var, lonsec, latsec, ro=100e3):

        ''' def get_section(var, longitudes, latitudes):

        Extract a vertical slice in any direction (or along a curve)
        from a ROMS netcdf file.

        Parameters
        ----------
        var: xarray.Dataarray
            ....
        lonsec : list
            Longitudes of the points of the section.
            (vector or [min max] or single value if N-S section).
            (default: [12 18])
        latsec : list
            Latitudes of the points of the section.
            (vector or [min max] or single value if E-W section)
            (default: -34)
        ro : float
            Decorrelation scale for objective analysis

        NB: if lonsec and latsec are vectors, they must have the same length.

        Returns
        -------
        VAR         xarray.Dataarray
        '''

        interp_type = 'linear'

        # Find maximum grid angle size (dl)
        lat = self.gridfile.lat_rho
        lon = self.gridfile.lon_rho
        mask = self.mask_rho

        dl = 1.5 * np.max([np.max(np.abs(self.grid.diff(self.gridfile.lon_rho, 'X'))),
                           np.max(np.abs(self.grid.diff(self.gridfile.lon_rho, 'Y'))),
                           np.max(np.abs(self.grid.diff(self.gridfile.lat_rho, 'X'))),
                           np.max(np.abs(self.grid.diff(self.gridfile.lat_rho, 'Y')))])

        # Read point positions
        #
        # TODO: Refactor and implement get_type
        if 's_w' in var.dims:
            typ = 'w'
        elif 'eta_v' in var.dims:
            typ = 'v'
        elif 'xi_u' in var.dims:
            typ = 'u'
        else:
            typ = 'r'
        
        # Throws an exception if input is 2-D variable
        vertical_dim = self._get_vertical_dimension(var) 
    
        # Find minimal subgrids limits
        minlon = np.min(lonsec)-dl
        minlat = np.min(latsec)-dl
        maxlon = np.max(lonsec)+dl
        maxlat = np.max(latsec)+dl
        bounding_box = (lon > minlon) * (lon < maxlon) * (lat > minlat) * (lat < maxlat)
        bounding_box.name = 'bounding box'
        if np.sum(bounding_box) == 0:
            raise ValueError('Section out of the domain')

        # Get bounding box subgrid
        lon = lon.where(bounding_box, drop=True)
        lat = lat.where(bounding_box, drop=True)
        mask = mask.where(bounding_box, drop=True)

        # Put latitudes and longitudes of the section in the correct vector form
        if np.size(lonsec) == 1:
            print('N-S section at longitude: ' + str(lonsec))
            if np.size(latsec) == 1:
                raise ValueError('Need more points to do a section')
            elif len(latsec) == 2:
                latsec = np.arange(latsec[0], latsec[1], dl)

            lonsec = 0 * latsec + lonsec
        elif np.size(latsec) == 1:
            print('E-W section at latitude: ' + str(latsec))
            if len(lonsec) == 2:
                lonsec = np.arange(lonsec[0], lonsec[1], dl)
            latsec = 0 * lonsec + latsec

        elif (len(lonsec) == 2) and (len(latsec) == 2):
            Npts = int(np.ceil(np.max([np.abs(lonsec[1]-lonsec[0])/dl,
                                       np.abs(latsec[1]-latsec[0])/dl])))
            if lonsec[0] == lonsec[1]:
                lonsec = lonsec[0] + np.zeros(Npts)
            else:
                lonsec = np.arange(lonsec[0], lonsec[1],
                                   (lonsec[1]-lonsec[0]) / Npts)
            if latsec[0] == latsec[1]:
                latsec = latsec[0] + np.zeros(Npts)
            else:
                latsec = np.arange(latsec[0], latsec[1],
                                   (latsec[1]-latsec[0]) / Npts)

        elif len(lonsec) != len(latsec):
            raise TypeError('Section latitudes and longitudes are not of' +
                            ' the same length')
        Npts = len(lonsec)

        # Get the section subgrid
        sub = lon * 0
        sub.name = 'subgrid'

        for ii in np.arange(0, Npts):
            sub = sub.where(np.invert((lon.data > lonsec[ii]-dl) * (lon.data < lonsec[ii]+dl) *
                                      (lat.data > latsec[ii]-dl) * (lat.data < latsec[ii]+dl)), other=1)
        sub = sub.astype(bool)
        
        # Get the mask
        # moving to numpy arrays here
        maskdata = mask.data[sub]
        londata = lon.data[sub]
        latdata = lat.data[sub]
        m1 = griddata((londata, latdata), maskdata, (lonsec, latsec), method='nearest')
        
        maskdata = maskdata.astype(bool)
        londata = londata[maskdata]
        latdata = latdata[maskdata]
        
        #  Get the vertical levels
        h = self.h
        hmin = np.min(h)
        h = h.where(bounding_box, drop=True)
        #h = h.where(sub, drop=True)
        hdata = h.data[sub]
        hdata = hdata[maskdata]
        h_sec = griddata((londata, latdata), hdata, (lonsec, latsec), method=interp_type)

        
        # Get zeta
        zeta = self.zeta.sel(time=var.time).where(bounding_box, drop=True)
        mask = mask.where(sub, drop=True)
        
        zeta = zeta.where(sub, drop=True)
        zeta = zeta.where(mask, drop=True)
        zeta = zeta.stack(x=mask.dims).dropna('x')
        #zetadata = zeta.to_masked_array().data
        
        # Get lon-/latdata again for compatibility
        lon = lon.where(sub, drop=True)
        lon = lon.where(mask, drop=True)
        lon = lon.stack(x=mask.dims).dropna('x')
        londata = lon.to_masked_array().data
        lat = lat.where(sub, drop=True)
        lat = lat.where(mask, drop=True)
        lat = lat.stack(x=mask.dims).dropna('x')
        latdata = lat.to_masked_array().data

        # create new xarray with section dimensions
        if self.chunks == None:
            zeta_sec = xr.DataArray(np.zeros((len(var.time),len(h_sec)),dtype=float)*np.nan,
                                    dims=('time','x'),
                                    coords={'time':var.time.data,'x':np.arange(len(h_sec))+1})
        else:
            chunks = self.chunks.copy()
            removedims = [k for k in chunks.keys() if 'eta' in k or 'xi' in k or 's_' in k]
            [chunks.pop(dim) for dim in removedims]
            zeta_sec = xr.DataArray(np.zeros((len(var.time),len(h_sec)),dtype=float)*np.nan,
                                    dims=('time','x'),
                                    coords={'time':var.time.data,'x':np.arange(len(h_sec))+1}).chunk(chunks=chunks)

        for timeindex, time in enumerate(var.time):
            zetadata = zeta.isel(time=timeindex).to_masked_array().data
            zeta_timeslice = griddata((londata, latdata), zetadata, (lonsec, latsec),
                                  method=interp_type)
            zeta_sec = zeta_sec.where(zeta_sec.time != time, other=zeta_timeslice)
        
        
        
        return zeta_sec, zeta_timeslice

        #########################################################################
        #########################################################################
        #########################################################################

        # zlevs() needs to be modified again or Z obtained in different way
        # idea: use var.h if available, otherwise self.h ... can maybe mostly keep current zlevs()
        Z = np.squeeze(zlevs(h, zeta, theta_s, theta_b, hc, N, typ, s_coord))
        N, Nsec = np.shape(Z)
        
        #########################################################################
        #########################################################################
        #########################################################################

        # Loop on the vertical levels
        
        # create new xarray with section dimensions (see zeta) .... need to get vertical dim
        # VAR = .........

        for k in np.arange(0, N):
            var = np.squeeze(nc.variables[vname][tindex, k, jmin:jmax+1,
                                                 imin:imax+1])
            var = var[sub == 1]
            var = var[mask == 1]
            var = griddata((londata, latdata), var,
                           (lonsec, latsec), method=interp_type)
            VAR[k, :] = m1 * var

        nc.close()
        
        # Get the distances
        dist = spheric_dist(latsec[0], latsec, lonsec[0], lonsec)/1e3
        X = np.squeeze(tridim(dist, N))
        # X[np.isnan(Z)] = 'nan'

        return var_section
    
    
    def calc_nut_colim_romspisces(self):
        
        '''
        Compute colimitations for nutrients
        1= nitrogen
        2=phosphate
        3=iron
        4=silicate

        Parameters
        ----------
        None, uses dataset of xroms object from which it is called. 

        Returns
        -------
        nlimnano: xarray.DataArray
        nlimdia: xarray.DataArray
        klimnano: xarray.DataArray
        klimdia: xarray.DataArray
        
        (output is stored in dataset of xroms object from which it was called)
        '''
        
        print('### calculating nutrient co-limitations ###')
        nc = self.dataset
        
        no3 = nc['NO3']*1e-6
        fer = nc['FER']*1e-6
        dia = nc['DIA']*1e-6
        phy = nc['NANO']*1e-6
        nh4 = nc['NH4']*1e-6
        sil = nc['Si']*1e-6
        po4 = nc['PO4']*1e-6

        zno3 = no3 * 1e6
        zferlim = 1.5e-11*no3*no3/40**2
        zferlim = zferlim.where(zferlim > 3e-12, other=3e-12)
        zferlim = zferlim.where(zferlim < 1.5e-11, other=1.5e-11)
    
        fer = fer.where(fer > zferlim, other=zferlim)
        print('zfemax computed')
        
        # parameters for ecosystem PISCES (run r5 8/11/06)
        # verif namelist peru du 24/10/2016

        conc0 = 2e-6
        conc1 = 1e-5
        conc2 = 1e-11
        conc3 = 1e-10
        rtrn = 1e-15
        xksi1 = 1.5e-6
        concnnh4 = 1e-7
        concdnh4 = 5e-7
        
        xconctemp = dia-5e-7
        xconctemp = xconctemp.where(xconctemp > 0, other=0)
        xconctemp2 = dia.where(dia < 5e-7, other=5e-7)
        xconctempn = phy-1e-6
        xconctempn = xconctempn.where(xconctempn > 0, other=0)
        xconctempn2 = phy.where(phy < 1e-6, other=1e-6)
        concdfe = (xconctemp2*conc3+0.4e-9*xconctemp)/(xconctemp2+xconctemp+rtrn)
        concdfe = concdfe.where(concdfe > conc3, other=conc3)
        
        concnfe = (xconctempn2*conc2+0.08e-9*xconctempn)/(xconctempn2+xconctempn+rtrn)
        concnfe = concnfe.where(concnfe > conc2, other=conc2)
        print('Iron coeff computed')
        xnanono3 = no3*concnnh4/(conc0*concnnh4+concnnh4*no3+conc0*nh4)
        xnanonh4 = nh4*conc0/(conc0*concnnh4+concnnh4*no3+conc0*nh4)
        xlim1 = xnanono3+xnanonh4  # azote
        xlim2 = po4/(po4+concnnh4)  # phosphate
        xlim3 = fer/(fer+concnfe)  # fer
        
        xlimphy = 0*xlim1
        xlimdia = 0*xlim1
        klimphy = 0*xlim1
        klimdia = 0*xlim1

        a = (xlim1<xlim2)*(xlim1<xlim3)
        b = (xlim2<xlim1)*(xlim2<xlim3)
        c = (xlim3<xlim1)*(xlim3<xlim2)
        xlimphy = xlimphy.where(~a, other=1)  # azote
        xlimphy = xlimphy.where(~b, other=2)  # phosphate
        xlimphy = xlimphy.where(~c, other=3)  # fer
        klimphy = klimphy.where(~a, other=xlim1)
        klimphy = klimphy.where(~b, other=xlim2)
        klimphy = klimphy.where(~c, other=xlim3)
        print('xlimphy computed')
        
        xdiatno3 = no3*concdnh4/(conc1*concdnh4+concdnh4*no3+conc1*nh4)
        xdiatnh4 = nh4*conc1/(conc1*concdnh4+concdnh4*no3+conc1*nh4)
        xlim1 = xdiatno3+xdiatnh4  # azote
        xlim2 = po4/(po4+concdnh4)  # phosphate
        xlim3 = fer/(fer+concdfe)  # fer
        xlim4 = sil/(sil+xksi1)  # silice
        print('diatoms limitation computed')
        
        a = (xlim1<xlim2)*(xlim1<xlim3)*(xlim1<xlim4)
        b = (xlim2<xlim1)*(xlim2<xlim3)*(xlim2<xlim4)
        c = (xlim3<xlim1)*(xlim3<xlim2)*(xlim3<xlim4)
        d = (xlim4<xlim1)*(xlim4<xlim2)*(xlim4<xlim3)
        xlimdia = xlimdia.where(~a, other=1)  # nitrogen
        xlimdia = xlimdia.where(~b, other=2)  # phosphate
        xlimdia = xlimdia.where(~c, other=3)  # iron
        xlimdia = xlimdia.where(~d, other=4)  # silicate
        klimdia = klimdia.where(~a, other=xlim1)
        klimdia = klimdia.where(~b, other=xlim2)
        klimdia = klimdia.where(~c, other=xlim3)
        klimdia = klimdia.where(~d, other=xlim4)

        self.dataset['nlimnano'] = xlimphy
        self.dataset['nlimdia'] = xlimdia
        self.dataset['klimnano'] = klimphy
        self.dataset['klimdia'] = klimdia

        print('write output')
        return xlimphy, xlimdia, klimphy, klimdia
    
    
    def calc_optic_romspisces(self, krgb_path):
        '''
        calculation of mean light available in the mixed layer
        
        (original MATLAB code in comments provided for reference / debugging)
        '''
        print('### calculating optical parameters ###')
        

        lon = self.gridfile['lon_rho']
        lon = self.gridfile['lat_rho']
        #Lp = self.gridfile['xi_rho'].shape[0]
        #Mp = self.gridfile['eta_rho'].shape[0]
        h = self.h

        print('read data chl:');
        nchl = self.dataset['NCHL']
        dchl = self.dataset['DCHL']
        hbl = self.dataset['hbl']
        # scrum_time = self.dataset['scrum_time']
        rtrn = 1e-15
        chl = nchl + dchl + rtrn
        zmeu = xr.zeros_like(nchl.isel(s_rho=0))
        emoy = xr.zeros_like(nchl.isel(s_rho=0))
        #etot = xr.zeros_like(nchl.isel(s_rho=0))
        etot = xr.zeros_like(nchl)

        theta_s = self.theta_s
        theta_b =  self.theta_b
        hc = self.hc
        N = self.N

        # read solar flux
        print('read solar flux:')
        qsr = self.dataset['swrad']
        # read only on value for month "mth"
        qsr1 = qsr

        print('compute z levels')
        
        try:
            zeta = 0 * self.h
            zeta.name = 'zeta'
            zeta.coords['eta_rho'] = self.dataset.coords['eta_rho']
            zeta.coords['xi_rho'] = self.dataset.coords['xi_rho']
            if 'time' in self.dataset.AKt.dims:
                zroms = self._zlevs(self.dataset.AKt.isel(time=0), zeta=zeta)
            else:
                zroms = self._zlevs(self.dataset.AKt, zeta=zeta)
        except KeyError:
            raise NotImplementedError('This function needs to be generalized for output that does not contain the variable AKt')
        
        print('read band rgb:')
        a = np.array(pd.read_table(krgb_path, skipinitialspace=True, header=None, sep='   ', engine='python'))
        xkrgb = a[:,1:]
        xkrgb = xr.DataArray(xkrgb,dims=('value','bgr'))
        
        # calcul du time step
        # t = scrum_time
        qsr0 = qsr

        parlux = 0.43/3
        hbl0 = hbl

        xchl = chl
        xchl = xchl.where(xchl > 0.03, other=0.03)  # min(xchl, 0.03)
        xchl = xchl.where(xchl < 10, other=10)  # max(xchl, 10)

        mrgb = (41+20*xr.ufuncs.log10(xchl)+rtrn).round().astype(int)
        
        #return xkrgb, mrgb
        
        #ek = xkrgb.isel(value=(mrgb-1)) # <-- fails with dask array
        def select_xkrgb_b(index):
            return np.array(xkrgb.isel(bgr=0))[index]
        def select_xkrgb_g(index):
            return np.array(xkrgb.isel(bgr=1))[index]
        def select_xkrgb_r(index):
            return np.array(xkrgb.isel(bgr=2))[index]

        ekb = xr.apply_ufunc(select_xkrgb_b,(mrgb-1),dask='parallelized',output_dtypes=[xkrgb.dtype])
        ekg = xr.apply_ufunc(select_xkrgb_g,(mrgb-1),dask='parallelized',output_dtypes=[xkrgb.dtype])
        ekr = xr.apply_ufunc(select_xkrgb_r,(mrgb-1),dask='parallelized',output_dtypes=[xkrgb.dtype])
    
        # append line of zeros to the top to match zroms shape
        for ek in [ekb,ekg,ekr]:
            top_zeros = xr.zeros_like(ek.isel(s_rho=31))
            top_zeros.coords['s_rho'] = 0
            ek = xr.concat((ek,top_zeros),dim='s_rho')

        #ekb = ...isel(bgr=0)
        #ekg =
        #ekr =
        
        # start building etot and integrating em, zm

        print('calculate etot,em,zm')
        dz_3d = self.grid.diff(zroms, 'Z')

        dz = dz_3d.isel(s_rho=N-1)
        zblight = 0.5*ekb.isel(s_rho=N-1)*dz
        zglight = 0.5*ekg.isel(s_rho=N-1)*dz
        zrlight = 0.5*ekr.isel(s_rho=N-1)*dz
        e1 = parlux*qsr0*xr.ufuncs.exp(-zblight)
        e2 = parlux*qsr0*xr.ufuncs.exp(-zglight)
        e3 = parlux*qsr0*xr.ufuncs.exp(-zrlight)

        etot = etot.where(etot['s_rho'] != etot['s_rho'][N-1], other=e1+e2+e3)
        zw = -1 * zroms.isel(s_w=N-1)
        empty = zw
        empty.attrs = dict()
        em = xr.zeros_like(empty)  # em = 0
        zm = xr.zeros_like(empty)  # zm = 0
        zmeu = xr.zeros_like(empty)  # zm = 0
        hmld = hbl0
        
        em = em.where(zw > hmld, other=(em+etot.isel(s_rho=N-1)*dz))  # if zw <= hmld
        zm = zm.where(zw > hmld, other=zm+dz)  # if zw <= hmld
        
        seuil = 0.0043*qsr0

        for k in np.arange(N-2,0,-1):
            print('vertical level: k={}'.format(k))
            dz = dz_3d.isel(s_rho=k)
            dz1 = dz_3d.isel(s_rho=k+1)

            zblight = 0.5*ekb.isel(s_rho=k)*dz + 0.5*ekb.isel(s_rho=k+1)*dz1
            zglight = 0.5*ekg.isel(s_rho=k)*dz + 0.5*ekg.isel(s_rho=k+1)*dz1
            zrlight = 0.5*ekr.isel(s_rho=k)*dz + 0.5*ekr.isel(s_rho=k+1)*dz1
            e1 = e1*xr.ufuncs.exp(-zblight)
            e2 = e2*xr.ufuncs.exp(-zglight)
            e3 = e3*xr.ufuncs.exp(-zrlight)

            etot = etot.where(etot['s_rho'] != etot['s_rho'][k], other=e1+e2+e3)

            zmeu = zmeu.where(etot.isel(s_rho=k) < seuil, other=-1*zroms.isel(s_w=k))
            
            zw = -1*zroms.isel(s_w=k)

            # dz = zroms(k+1,j,i)-zroms(k,j,i)

            em = em.where(zw > hmld, other=(em+etot.isel(s_rho=k)*dz))  # if zw <= hmld
            zm = zm.where(zw > hmld, other=zm+dz)  # if zw <= hmld

        
        emoy = emoy.where(zm <= 0, other=em/zm)  # if zm > 0 ... otherwise 0, already assigned
        
        if 'time' in zmeu.dims:
            zmeu = zmeu.transpose('time','eta_rho','xi_rho')

        print('write output')
        self.dataset['zmeu'] = zmeu
        self.dataset['emoy'] = emoy
        self.dataset['etot'] = etot
        
        return emoy, etot, zmeu
    

    def calc_lightlim_romspisces(self):
        '''
        calculation of mean light available in the mixed layer

        % VERIFY NAMELIST! %
        '''
        print('### calculating light limitation ###')
        
        lon = self.gridfile['lon_rho']
        lon = self.gridfile['lat_rho']
        h = self.h

        print('read data chl')
        nchl = self.dataset['NCHL']
        dchl = self.dataset['DCHL']
        emoy = self.dataset['emoy']
        etot = self.dataset['etot']
        limdia = xr.zeros_like(etot)
        limnano = xr.zeros_like(etot)
        N = self.N
        zmeu = self.dataset['zmeu']
        hbl = self.dataset['hbl']
        tn = self.dataset['temp']
        phy = self.dataset['NANO']
        dia = self.dataset['DIA']

        # constants
        rtrn = 1e-15
        rjjss = 86400
        prmax2 = 1.2/rjjss
        pislope = 3
        pislope2 = 3
        
        for k in np.arange(N-1,0,-1):
            print('vertical level: k={}'.format(k))
            tn0 =  tn.isel(s_rho=k)
            nchl0 = nchl.isel(s_rho=k)
            dchl0 = dchl.isel(s_rho=k)
            phy0 = phy.isel(s_rho=k)
            dia0 = dia.isel(s_rho=k)
            ztn = tn0-15
            ztn = ztn.where(ztn > 0, other=0)  # max(0.,tn0-15.)
            # % version Roms_Agrif_V3.0_21112013/PISCES
            zadap = 0+1*ztn/(2+ztn)
            zadap2 = xr.zeros_like(ztn)
            # % version antérieure
            # %  zadap=1.+2*ztn/(2.+ztn);
            # %  zadap2=1.;
            # %  disp('emoy:')
            # %  emoy(l,j,i)
            zfact = xr.ufuncs.exp(-0.21*emoy)
            pislopead = pislope*(1+zadap*zfact)
            pislopead2 = pislope2*(1+zadap2*zfact)

            pislopen = pislopead*nchl0/(rtrn+phy0*12)/(prmax2*rjjss+rtrn)
            pislope2n = pislopead2*dchl0/(rtrn+dia0*12)/(prmax2*rjjss+rtrn)

            exp = xr.ufuncs.exp
            limnano_k = 1-exp(-pislopen*etot.isel(s_rho=k))
            limnano = limnano.where(limnano['s_rho'] != limnano['s_rho'][k], other=limnano_k)
            # limnano(l,k,j,i) = 1.-exp(-pislopen*etot(l,k,j,i)) MATLAB
            limdia_k = 1-exp(-pislope2n*etot.isel(s_rho=k))
            limdia = limdia.where(limdia['s_rho'] != limdia['s_rho'][k], other=limdia_k)
            # limdia(l,k,j,i) = 1.-exp(-pislope2n*etot(l,k,j,i)) MATLAB

        print('write output')
        self.dataset['llimnano'] = limnano
        self.dataset['llimdia'] = limdia
        
        return limnano, limdia
    

    def calc_total_colim_romspisces(self):
        '''
        calculation of total co-limitation (nutrients or light)
        '''
        print('### calculating total co-limitation ###')
        
        totlimnano = self.dataset.nlimnano.where(self.dataset.klimnano < self.dataset.llimnano, other=5)
        totlimdia = self.dataset.nlimdia.where(self.dataset.klimdia < self.dataset.llimdia, other=5)
        
        totlimnano = self.mask(totlimnano)
        totlimdia = self.mask(totlimdia)
        
        print('write output')
        self.dataset['totlimnano'] = totlimnano
        self.dataset['totlimdia'] = totlimdia
        
        return totlimnano, totlimdia







    
# def get_section(*args):

#     ''' def get_section(fname, gname, lonsec, latsec, vname, tindex):

#     Extract a vertical slice in any direction (or along a curve)
#     from a ROMS netcdf file.

#     Parameters
#     ----------
#     fname : string
#         History NetCDF file name.
#     gname : string
#         Grid NetCDF file name.
#     lonsec : list
#         Longitudes of the points of the section.
#         (vector or [min max] or single value if N-S section).
#         (default: [12 18])
#     latsec : list
#         Latitudes of the points of the section.
#         (vector or [min max] or single value if E-W section)
#         (default: -34)

#     NB: if lonsec and latsec are vectors, they must have the same length.

#     vname
#         NetCDF variable name to process (character string).
#         (default: temp)
#     tindex : int
#         Netcdf time index (integer).
#         (default: 1)

#     Returns
#     -------
#     X           Slice X-distances (km) from the first point (2D matrix).
#     Z           Slice Z-positions (matrix).
#     VAR         Slice of the variable (matrix).
#     '''

#     nargin = len(args)
#     interp_type = 'linear'

#     #
#     # Defaults values
#     #
#     try:
#         fname = args[0]
#     except IndexError:
#         raise NameError('You must specify a file name')
#     try:
#         gname = args[1]
#     except IndexError:
#         gname = fname
#         print(['Default grid name: ' + gname])
#     try:
#         lonsec = args[2]
#     except IndexError:
#         lonsec = [-79, -73]
#         print(['Default longitude: ' + str(lonsec)])
#     try:
#         latsec = args[3]
#     except IndexError:
#         latsec = -16
#         print(['Default latitude: ' + str(latsec)])
#     try:
#         vname = args[4]
#     except IndexError:
#         vname = 'temp'
#         print(['Default variable to plot: ' + vname])
#     try:
#         tindex = args[5]
#     except IndexError:
#         tindex = 1
#         print(['Default time index: ' + str(tindex)])

#     #
#     # Find maximum grid angle size (dl)
#     #
#     lat, lon, mask = read_latlonmask(gname, 'r')
#     M, L = np.shape(lon)
#     dl = 1.5 * np.max([np.max(np.abs(lon[1:M, :]-lon[:M-1, :])),
#                        np.max(np.abs(lon[:, 1:L]-lon[:, :L-1])),
#                        np.max(np.abs(lat[1:M, :]-lat[:M-1, :])),
#                        np.max(np.abs(lat[:, 1:L]-lat[:, :L-1]))])
#     #
#     # Read point positions
#     #
#     typ, vlevel = get_type(fname, vname, 10)
#     if (vlevel == 0):
#         print(vname + ' is a 2D-H variable')
#         return
#     lat, lon, mask = read_latlonmask(gname, typ)
#     M, L = np.shape(lon)
#     #
#     # Find minimal subgrids limits
#     #
#     minlon = np.min(lonsec)-dl
#     minlat = np.min(latsec)-dl
#     maxlon = np.max(lonsec)+dl
#     maxlat = np.max(latsec)+dl
#     sub = (lon > minlon) * (lon < maxlon) * (lat > minlat) * (lat < maxlat)
#     if np.sum(sub) == 0:
#         print('Section out of the domain')
#         return

#     ival = np.sum(sub, 0)
#     jval = np.sum(sub, 1)
#     imin = np.min(np.where(ival != 0))
#     imax = np.max(np.where(ival != 0))
#     jmin = np.min(np.where(jval != 0))
#     jmax = np.max(np.where(jval != 0))
#     #
#     # Get subgrids
#     #
#     lon = lon[jmin:jmax+1, imin:imax+1]
#     lat = lat[jmin:jmax+1, imin:imax+1]
#     sub = sub[jmin:jmax+1, imin:imax+1]
#     mask = mask[jmin:jmax+1, imin:imax+1]
#     #
#     # Put latitudes and longitudes of the section in the correct vector form
#     #
#     if np.size(lonsec) == 1:
#         print('N-S section at longitude: ' + str(lonsec))
#         if np.size(latsec) == 1:
#             raise ValueError('Need more points to do a section')
#         elif len(latsec) == 2:
#             latsec = np.arange(latsec[0], latsec[1], dl)

#         lonsec = 0 * latsec + lonsec
#     elif np.size(latsec) == 1:
#         print('E-W section at latitude: ' + str(latsec))
#         if len(lonsec) == 2:
#             lonsec = np.arange(lonsec[0], lonsec[1], dl)
#         latsec = 0 * lonsec + latsec

#     elif (len(lonsec) == 2) and (len(latsec) == 2):
#         Npts = int(np.ceil(np.max([np.abs(lonsec[1]-lonsec[0])/dl,
#                                    np.abs(latsec[1]-latsec[0])/dl])))
#         if lonsec[0] == lonsec[1]:
#             lonsec = lonsec[0] + np.zeros(Npts)
#         else:
#             lonsec = np.arange(lonsec[0], lonsec[1],
#                                (lonsec[1]-lonsec[0]) / Npts)
#         if latsec[0] == latsec[1]:
#             latsec = latsec[0] + np.zeros(Npts)
#         else:
#             latsec = np.arange(latsec[0], latsec[1],
#                                (latsec[1]-latsec[0]) / Npts)

#     elif len(lonsec) != len(latsec):
#         raise TypeError('Section latitudes and longitudes are not of' +
#                         ' the same length')
#     Npts = len(lonsec)
#     #
#     # Get the subgrid
#     #
#     sub = 0 * lon
#     for ii in np.arange(0, Npts):
#         sub[(lon > lonsec[ii]-dl) * (lon < lonsec[ii]+dl) *
#             (lat > latsec[ii]-dl) * (lat < latsec[ii]+dl)] = 1
#     #
#     #  get the coefficients of the objective analysis
#     #
#     londata = lon[sub == 1]
#     latdata = lat[sub == 1]
#     coef = oacoef(londata, latdata, lonsec, latsec, 100e3)
#     #
#     # Get the mask
#     #
#     mask = mask[sub == 1]
#     m1 = griddata((londata, latdata), mask, (lonsec, latsec), method='nearest')
#     # mask(isnan(mask)) = 0
#     # mask = mean(mask)+coef*(mask-mean(mask))
#     # mask(mask>0.5) = 1
#     # mask(mask< = 0.5) = NaN
#     londata = londata[mask == 1]
#     latdata = latdata[mask == 1]
#     #
#     #  Get the vertical levels
#     #
#     nc = ncread(gname)
#     h = nc.variables['h'][:]
#     hmin = np.min(h)
#     h = h[jmin:jmax+1, imin:imax+1]
#     nc.close()
#     h = h[sub == 1]
#     h = h[mask == 1]
#     # h = mean(h)+coef*(h-mean(h))
#     h = griddata((londata, latdata), h, (lonsec, latsec), method=interp_type)
#     #
#     nc = ncread(fname)

#     # zeta = np.squeeze(nc.variables['zeta'][tindex, jmin:jmax+1, imin:imax+1])
#     try:
#         zeta = np.squeeze(nc.variables['zeta'][tindex, jmin:jmax+1,
#                                                imin:imax+1])
#         if np.size(zeta) == 0:
#             zeta = 0 * h
#         else:
#             zeta = zeta[sub == 1]
#             zeta = zeta[mask == 1]
#             # zeta = mean(zeta)+coef*(zeta-mean(zeta))
#             zeta = griddata((londata, latdata), zeta, (lonsec, latsec),
#                             method=interp_type)
#     except KeyError:
#         zeta = 0 * h

#     theta_s = nc.theta_s

#     if np.size(theta_s) == 0:
#         # print('Rutgers version')
#         theta_s = nc.variables['theta_s']
#         theta_b = nc.variables['theta_b']
#         Tcline = nc.variables['Tcline']
#     else:
#         # print('UCLA version')
#         theta_b = nc.theta_b
#         Tcline = nc.Tcline

#     if np.size(Tcline) == 0:
#         # print('UCLA version 2')
#         hc = nc.hc
#     else:
#         hmin = np.nanmin(h)
#         hc = np.min([hmin, Tcline])

#     N = len(nc.variables['s_rho'][:])
#     s_coord = 1
#     try:
#         VertCoordType = nc.VertCoordType
#         if VertCoordType == 'NEW':
#             s_coord = 2
#     except AttributeError:
#         try:
#             vtrans = nc.variables['Vtransform'][:]
#             if np.size(vtrans) != 0:
#                 s_coord = vtrans
#         except KeyError:
#             pass
#     if s_coord == 2:
#         hc = Tcline

#     # print('h: '+str(h))
#     # print('hc: '+str(hc))
#     # print('theta_s: '+str(theta_s))
#     # print('theta_b: '+str(theta_b))
#     Z = np.squeeze(zlevs(h, zeta, theta_s, theta_b, hc, N, typ, s_coord))
#     N, Nsec = np.shape(Z)
#     #
#     # Loop on the vertical levels
#     #
#     VAR = 0 * Z
#     for k in np.arange(0, N):
#         var = np.squeeze(nc.variables[vname][tindex, k, jmin:jmax+1,
#                                              imin:imax+1])
#         var = var[sub == 1]
#         var = var[mask == 1]
#         var = griddata((londata, latdata), var,
#                        (lonsec, latsec), method=interp_type)
#         VAR[k, :] = m1 * var

#     nc.close()
#     #
#     # Get the distances
#     #
#     dist = spheric_dist(latsec[0], latsec, lonsec[0], lonsec)/1e3
#     X = np.squeeze(tridim(dist, N))
#     # X[np.isnan(Z)] = 'nan'

#     return X, Z, VAR
