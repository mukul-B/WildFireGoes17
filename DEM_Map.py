import os

import numpy as np
import rioxarray
import sys
import xarray as xr
from goes_ortho.geometry import LonLat2ABIangle
import subprocess
from distutils.spawn import find_executable

import requests

class DEM_Map:
    def __init__(self,dem_path):
        self.dem_filepath = dem_path
        file_path = 'secrets/DEM_API_MAP_KEY'
        key_map = {}
        with open(file_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(' ')
                key_map[key] = value

        self.api_key=key_map['DEM_API']
        
        
    def get_dem(self,demtype, bounds, out_fn=None, proj="EPSG:4326",output_res=30):
        """
        download a DEM of choice from OpenTopography World DEM (modified by Shashank Bhushan, first written by David Shean)

        Parameters
        ------------
        demtype : str
            type of DEM to fetch (e.g., SRTMGL1, SRTMGL1_E, SRTMGL3 etc)
        bounds : list
            geographic aoi extent in format (minx,miny,maxx,maxy)
        out_fn : str
            path to output filename
        t_srs : str
            output DEM projection

        Returns
        -----------
        out_DEM : str
            path to output DEM (useful if the downloaded DEM is reprojected to custom proj)

        Examples
        ------------

        """
        api_key = self.api_key
        
        out_fn = self.dem_filepath
        ### From David Shean
        base_url = "https://portal.opentopography.org/API/globaldem?demtype={}&west={}&south={}&east={}&north={}&outputFormat=GTiff&API_Key={}"
        if out_fn is None:
            out_fn = "{}.tif".format(demtype)
        if not os.path.exists(out_fn):
            # Prepare API request url
            # Bounds should be [minlon, minlat, maxlon, maxlat]
            url = base_url.format(demtype, *bounds, api_key)
            #print(url)
            # Get
            response = requests.get(url)
            # Check for 200
            # Write to disk
            open(out_fn, "wb").write(response.content)
        if proj != "EPSG:4326":
            # Could avoid writing to disk and directly reproject with rasterio, using gdalwarp for simplicity
            proj_fn = os.path.splitext(out_fn)[0] + "_proj.tif"
            if not os.path.exists(proj_fn):
                # output_res = 30
                gdalwarp = find_executable("gdalwarp")
                gdalwarp_call = f"{gdalwarp} -r cubic -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -tr {output_res} {output_res} -t_srs '{proj}' {out_fn} {proj_fn}"
                # print(gdalwarp_call)
                self.run_bash_command(gdalwarp_call)
            out_DEM = proj_fn
        else:
            out_DEM = out_fn
        return out_DEM



    def make_ortho_map(self, out_filepath=None):
        """
        For the entire DEM, determine the ABI scan angle coordinates for every DEM grid cell, taking into account the underlying terrain and satellite's viewing geometry. Create the mapping between GOES-R ABI pixels (netCDF input file) and a DEM grid (geotiff input file)

        Parameters
        ------------
        goes_filepath : str
            filepath to GOES ABI NetCDF file
        dem_filepath : str
            filepath to digital elevation model (DEM), GeoTiff file
        out_filepath : str
            optional filepath and filename to save this map to, defaults to None

        Returns
        ------------
        ds : xarray.Dataset
            dataset of the map relating ABI Fixed Grid coordinates to latitude and longitude

        Examples
        ------------

        """

        # print("\nRUNNING: make_ortho_map()")

        req,rpol,H,lon_0,e = self.req,self.rpol,self.H,self.lon_0,self.e
        

        # Load DEM
        # print("\nOpening DEM file...")
        dem_filepath = self.dem_filepath
        dem = rioxarray.open_rasterio(dem_filepath)
        dem = dem.where(dem != dem.attrs["_FillValue"])[0, :, :]  # replace nodata with nans
        dem = dem.fillna(
            0
        )  # fill nans with zeros for the ocean (temporary fix for fog project)
        # dem = dem.where(dem!=0) # replace zeros with nans
        # Create 2D arrays of longitude and latitude from the DEM
        # print("\nCreate 2D arrays of longitude and latitude from the DEM")
        X, Y = np.meshgrid(dem.x, dem.y)  # Lon and Lat of each DEM grid cell
        Z = dem.values  # elevation of each DEM grid cell
        # print("...done")

        # For each grid cell in the DEM, compute the corresponding ABI scan angle (x and y, radians)
        # print(
        #     "\nFor each grid cell in the DEM, compute the corresponding ABI scan angle (x and y, radians)"
        # )
        abi_grid_x, abi_grid_y = LonLat2ABIangle(X, Y, Z, H, req, rpol, e, lon_0)
        # print("...done")

        # Create metadata dictionary about this map (should probably clean up metadata, adhere to some set of standards)
        # print("\nCreate metadata dictionary about this map")
        metadata = {
            # Information about the projection geometry:
            "longitude_of_projection_origin": lon_0,
            "semi_major_axis": req,
            "semi_minor_axis": rpol,
            "satellite_height": H,
            "grs80_eccentricity": e,
            "longitude_of_projection_origin_info": "longitude of geostationary satellite orbit",
            "semi_major_axis_info": "semi-major axis of GRS 80 reference ellipsoid",
            "semi_minor_axis_info": "semi-minor axis of GRS 80 reference ellipsoid",
            "satellite_height_info": "distance from center of ellipsoid to satellite (perspective_point_height + semi_major_axis_info)",
            "grs80_eccentricity_info": "eccentricity of GRS 80 reference ellipsoid",
            # Information about the DEM source file
            "dem_file": dem_filepath,
            #'dem_crs' : dem.crs,
            #'dem_transform' : dem.transform,
            #'dem_res' : dem.res,
            #'dem_ifov': -9999, # TO DO
            "dem_file_info": "filename of dem file used to create this mapping",
            "dem_crs_info": "coordinate reference system from DEM geotiff",
            "dem_transform_info": "transform matrix from DEM geotiff",
            "dem_res_info": "resolution of DEM geotiff",
            "dem_ifov_info": "instantaneous field of view (angular size of DEM grid cell)",
            # For each DEM grid cell, we have...
            "dem_px_angle_x_info": "DEM grid cell X coordinate (east/west) scan angle in the ABI Fixed Grid",
            "dem_px_angle_y_info": "DEM grid cell Y coordinate (north/south) scan angle in the ABI Fixed Grid",
            "longitude_info": "longitude from DEM file",
            "latitude_info": "latitude from DEM file",
            "elevation_info": "elevation from DEM file",
        }
        # print("...done")

        # Create pixel map dataset
        # print("\nCreate pixel map dataset")
        ds = xr.Dataset(
            {"elevation": (["latitude", "longitude"], dem.values)},
            coords={
                "longitude": (["longitude"], dem.x.data),
                "latitude": (["latitude"], dem.y.data),
                "dem_px_angle_x": (["latitude", "longitude"], abi_grid_x),
                "dem_px_angle_y": (["latitude", "longitude"], abi_grid_y),
            },
            attrs=metadata,
        )
        print(ds)
        # print("...done")

        if out_filepath is not None:
            # print("\nExport this pixel map along with the metadata (NetCDF with xarray)")
            # Export this pixel map along with the metadata (NetCDF with xarray)
            ds.to_netcdf(out_filepath, mode="w")
            # print("...done")

        # Return the pixel map dataset
        # print("\nReturn the pixel map dataset.")
        
        return ds

    def get_artho_coff_from_GOES(self,goes_filepath):
        # print("\nOpening GOES ABI image...")
        abi_image = xr.open_dataset(goes_filepath, decode_times=False)
        # NOTE: for some reason (?) I sometimes get an error "ValueError: unable to decode time units 'seconds since 2000-01-01 12:00:00' with the default calendar. Try opening your dataset with decode_times=False." so I've added decode_times=False here.
        # Get inputs: projection information from the ABI radiance product (values needed for geometry calculations)
        # print("\nGet inputs: projection information from the ABI radiance product")
        req = abi_image.goes_imager_projection.semi_major_axis
        rpol = abi_image.goes_imager_projection.semi_minor_axis
        H = (
            abi_image.goes_imager_projection.perspective_point_height
            + abi_image.goes_imager_projection.semi_major_axis
        )
        lon_0 = abi_image.goes_imager_projection.longitude_of_projection_origin
        e = 0.0818191910435  # GRS-80 eccentricity
        # print("...done")
        self.req,self.rpol,self.H,self.lon_0,self.e = req,rpol,H,lon_0,e
    
        return req,rpol,H,lon_0,e

    def get_artho_coff_by_GOES_version(self,GOES_version):
        req,rpol,H= 6378137.0, 6356752.31414, 42164160.0
        e = 0.0818191910435  # GRS-80 eccentricity
        if(GOES_version == 'G16'):
            lon_0 = -75.0
        else:
            lon_0 = -137.0
        self.req,self.rpol,self.H,self.lon_0,self.e = req,rpol,H,lon_0,e
        return req,rpol,H,lon_0,e
    
    
    def run_bash_command(self,cmd):
        # written by Scott Henderson
        # move to asp_binder_utils
        """Call a system command through the subprocess python module."""
        # print(cmd)
        try:
            retcode = subprocess.call(cmd, shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)