from eomaps import Maps
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

df = gpd.read_file(gpd.datasets.get_path("nybb"))

m = Maps(crs=Maps.CRS.Mercator.GOOGLE)
m.set_extent((-105.140714, -105.138887, 40.595478, 40.596143))

# m.add_gdf(df, column="BoroName", legend=True)
m.add_wms.ESRI_ArcGIS.SERVICES.World_Imagery.add_layer.xyz_layer()


# Data
m_data = m.new_layer()


leak_data_raw = pd.read_csv("EOG-Resources-Dataset-main/sensor_readings.csv")
sensor_locations = [sensor_location.replace(" ", "")[7:].split("_")[:2] for sensor_location in leak_data_raw.columns[2:]]
sensor_lon = [float(sensor_location[0]) for sensor_location in sensor_locations]
sensor_lat = [float(sensor_location[1]) for sensor_location in sensor_locations]

leak_data = pd.DataFrame(dict(lon=sensor_lon, lat=sensor_lat, value=[1]*len(sensor_lon)))
m_data.set_data(leak_data, x="lon", y="lat", parameter="value",crs=4326,)



m.set_shape.scatter_points(size=5,marker="o",)
m_data.plot_map()

plt.show()