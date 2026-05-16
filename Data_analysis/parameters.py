import os


class Parameters:
    def __init__(self):
        # Keep paths stable whether main.py is run from the project root or Data_analysis.
        self.FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.WORKING_DIRECTORY = os.path.dirname(self.FILE_PATH)
        self.DATA_PATH = os.path.join(self.WORKING_DIRECTORY, 'Data_files')
        self.RESULT_PATH = os.path.join(self.WORKING_DIRECTORY, 'result')

        # Input data paths.
        self.TXT_PATH = os.path.join(self.DATA_PATH, 'GLB.Ts+dSST.txt')
        self.NETCDF_GZ_PATH = os.path.join(self.DATA_PATH, 'gistemp1200_GHCNv4_ERSSTv5.nc.gz')

        # NetCDF settings used by the NASA GISS grid file.
        self.NETCDF_TEMPERATURE_VARIABLE = 'tempanomaly'
        self.NETCDF_LAT_VARIABLE = 'lat'
        self.NETCDF_LON_VARIABLE = 'lon'
        self.NETCDF_FIRST_YEAR = 1880

        # Figure settings matched to the paper text.
        self.SPATIAL_MAP_YEAR = 2024
        self.TREND_START_YEAR = 1970
        self.TREND_END_YEAR = 2025

        # Which subprocess will run.
        self.SUBPROCESS_TO_RUN = {
            'Analyse_TXT': True,
            'Analyse_NetCDF': True,
        }

        # The file encoding.
        self.ENCODING = 'utf-8'
