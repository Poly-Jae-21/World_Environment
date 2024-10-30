import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from pyogrio import read_dataframe
import gemgis as gg

class Polygon_to_matrix():
    def __init__(self):
        self.data = 0

    def transform_data_landuse(self, gdf):

        # Step 2: Define the geometry and transform

        # Ensure the land use codes are in the dataframe
        if 'LANDUSE' not in gdf.columns:
            raise ValueError("The shapefile does not contain a 'land_use_code' column.")
        try:
            gdf['LANDUSE'] = gdf['LANDUSE'].astype(int)
        except ValueError:
            raise ValueError("The 'land_use_code' column contains non-numeric values.")

        Not_land_list = [1000, 1100, 1110, 1111, 1112, 1130, 1140, 1150, 1151, 1340, 1350, 1370, 1510, 1511, 1512, 1520,
                         1530, 1550, 1560, 1561, 1562, 1563, 1564, 1565, 1570, 2000, 3400, 3500, 4200, 4210, 4220, 4230,
                         4240, 5000, 6000, 9999]
        Ok_land_list = [1210, 1211, 1212, 1214, 1215, 1216, 1220, 1240, 1250, 1300, 1310, 1320, 1321, 1322, 1330, 1380,
                        1410, 1420, 1430, 1431, 1432, 1433, 1450, 1540, 3000, 3100, 3200, 3300, 4000, 4100, 4110, 4120,
                        4130, 4140]

        gdf['LANDUSE'] = gdf['LANDUSE'].apply(lambda x: 1 if x in Ok_land_list else 2)


        bounds = gdf.total_bounds  # get bounds of the shapefile
        resolution = 10  # define your desired resolution

        # Calculate the number of columns and rows for the raster
        x_min, y_min, x_max, y_max = bounds

        width = int((x_max - x_min) / resolution)
        height = int((y_max - y_min) / resolution)
        x_min, y_min, x_max, y_max = x_min / resolution, y_min / resolution, x_max / resolution, y_max / resolution

        transform = rasterio.transform.from_origin(west=x_min, north=y_max, xsize=resolution, ysize=resolution)

        # Step 3: Rasterize the geometries with land use codes
        shapes = ((geom, code) for geom, code in zip(gdf.geometry, gdf['LANDUSE']))
        raster = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')

        # Step 4: Convert the raster to a NumPy array
        numpy_array = np.array(raster)

        return numpy_array, x_min, y_min, x_max, y_max

    def transform_data_community_boundary(self, gdf):
        if 'area_numbe' not in gdf.columns:
            raise ValueError("The shapefile does not contain a 'area_number' in column.")
        try:
            gdf['area_numbe'] = gdf['area_numbe'].astype(int)
        except:
            raise ValueError("The 'area_number' column contains non-numeric values.")

        bounds = gdf.total_bounds
        resolution = 10
        x_min, y_min, x_max, y_max = bounds
        width = int((x_max - x_min) / resolution)
        height = int((y_max - y_min) / resolution)
        x_min, y_min, x_max, y_max = x_min / resolution, y_min / resolution, x_max / resolution, y_max / resolution

        transform = rasterio.transform.from_origin(west=x_min, north=y_max, xsize=resolution, ysize=resolution)

        shapes = ((geom, code) for geom, code in zip(gdf.geometry, gdf['area_numbe']))
        raster = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')

        numpy_array = np.array(raster)

        return numpy_array, x_min, y_min, x_max, y_max

    def transform_data_transmission(self, gdf):

        # Ensure the land use codes are in the dataframe
        if 'EV_DC_Fast_Count' not in gdf.columns:
            raise ValueError("The shapefile does not contain a 'EV_DC_Fast_Count' column.")
        try:
            gdf['EV_DC_Fast_Count'] = gdf['EV_DC_Fast_Count'].astype(int)
        except ValueError:
            raise ValueError("The 'EV_DC_Fast_Count' column contains non-numeric values.")

        # Step 2: Define the geometry and transform
        bounds = gdf.total_bounds  # get bounds of the shapefile
        resolution = 10  # define your desired resolution

        # Calculate the number of columns and rows for the raster
        x_min, y_min, x_max, y_max = bounds
        width = int((x_max - x_min) / resolution)
        height = int((y_max - y_min) / resolution)

        transform = rasterio.transform.from_origin(west=x_min, north=y_max, xsize=resolution, ysize=resolution)

        # Step 3: Rasterize the geometries with land use codes
        shapes = ((geom, code) for geom, code in zip(gdf.geometry, gdf['EV_DC_Fast_Count']))
        raster = rasterize(shapes=shapes, out_shape=(height, width), transform=transform, fill=0, dtype='int32')

        # Step 4: Convert the raster to a NumPy array
        numpy_array = np.array(raster)
        numpy_array[numpy_array != 0] = 1

        return numpy_array, gdf, x_min, y_min, x_max, y_max

    def save_data(self, numpy_array):
        csv_path = 'env/utils/output_matrix_v5.csv'
        np.savetxt(csv_path, numpy_array, delimiter=",", fmt='%d')

