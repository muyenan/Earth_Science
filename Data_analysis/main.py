from draw_netcdf import Draw_netcdf
from draw_txt import Draw_txt
from parameters import Parameters
from read_netcdf import Read_netcdf
from read_txt import Read_txt


parameters = Parameters()
read_txt = Read_txt()
draw_txt = Draw_txt()
read_netcdf = Read_netcdf()
draw_netcdf = Draw_netcdf()

if parameters.SUBPROCESS_TO_RUN['Analyse_TXT']:
    print("Reading the txt...")

    years = read_txt.read_txt(0)
    annual_anomaly = read_txt.read_txt(1)

    print("Finish reading. Start analyzing and drawing...")

    draw_txt.draw_txt(years, annual_anomaly)

if parameters.SUBPROCESS_TO_RUN['Analyse_NetCDF']:
    print("Reading the NetCDF grid data...")

    years, lat, lon, annual_maps = read_netcdf.read_annual_temperature_anomaly()

    print("Finish reading. Start drawing spatial maps...")

    draw_netcdf.draw_spatial_anomaly_map(years, lat, lon, annual_maps)
    draw_netcdf.draw_trend_map(years, lat, lon, annual_maps)
