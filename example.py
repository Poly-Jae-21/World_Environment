import numpy as np
from template.env_name.envs.utils.data_conversion import Polygon_to_matrix, Density
PtM = Polygon_to_matrix()
from pyogrio import read_dataframe

# Import The landuse data (matrix format) and convert it into numpy format for sub_MAP
read_landuse_data = read_dataframe('C:/Users\S2HubLab\PycharmProjects\World_Environment/template\env_name\envs\data\landuse_map\Landuse2018_Dissolve_Pr_Clip.shp')
landuse_numpy, landuse_minX, landuse_maxX, landuse_minY, landuse_maxY = PtM.transform_data_landuse(read_landuse_data)

