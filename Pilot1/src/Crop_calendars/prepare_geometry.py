from functools import partial
import pyproj
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
import utm

# function to convert the field to UTM projection
# and apply an inward buffer of 10 m
def to_utm_inw_buffered(epsg_original, epsg_utm, field):
    project = partial(
        pyproj.transform,
        pyproj.Proj('epsg:{}'.format(str(epsg_original))),
        pyproj.Proj('epsg:{}'.format(str(epsg_utm)))
    )
    if field.type == 'Polygon':
        lat_list = [field.coordinates[0][p][1] for p in range(len(field.coordinates[0]))]
        lon_list = [field.coordinates[0][p][0] for p in range(len(field.coordinates[0]))]
    elif field.type == 'MultiPolygon':
        lat_list = [field.coordinates[0][0][p][1] for p in range(len(field.coordinates[0][0]))]
        lon_list = [field.coordinates[0][0][p][0] for p in range(len(field.coordinates[0][0]))]
    poly_reproject = transform(project ,Polygon(zip(lon_list, lat_list))).buffer(-10, cap_style = 1, join_style = 2, resolution  = 4) # inward buffering of the polygon
    poly_reproject_WGS = UTM_to_WGS84(epsg_utm, poly_reproject)
    return poly_reproject_WGS
def UTM_to_WGS84(epsg_utm, field):
    project = partial(
        pyproj.transform,
        pyproj.Proj('epsg:{}'.format(str(epsg_utm))),
        pyproj.Proj('epsg:{}'.format(str(4326)))
    )

    poly_WGS84 = transform(project, field)
    return poly_WGS84

# function to get the epsg of the UTM zone
def _get_epsg(lat, zone_nr):
    if lat >= 0:
        epsg_code = '326' + str(zone_nr)
    else:
        epsg_code = '327' + str(zone_nr)
    return int(epsg_code)
# function that prepares the geometry of the fields so
# that they are suitable for applying the crop calendar model
def prepare_geometry(gj):
    polygons_inw_buffered = []
    poly_too_small_buffer = []
    for field_loc in range(len(gj.features)):
        if gj.features[0].geometry.type == 'Polygon':
            lon = gj['features'][field_loc].geometry.coordinates[0][0][0]
            lat = gj['features'][field_loc].geometry.coordinates[0][0][1]
        elif gj.features[0].geometry.type == 'MultiPolygon':  # in case the data is stored as a multipolygon
            lon = gj['features'][field_loc].geometry.coordinates[0][0][0][0]
            lat = gj['features'][field_loc].geometry.coordinates[0][0][0][1]
        utm_zone_nr = utm.from_latlon(lat, lon)[2]
        epsg_UTM_field = _get_epsg(lat, utm_zone_nr)
        poly_inw_buffered = to_utm_inw_buffered('4326', epsg_UTM_field, gj.features[field_loc].geometry)
        if poly_inw_buffered.is_empty:
            poly_too_small_buffer.append(gj['features'][field_loc].geometry)
            continue
        polygons_inw_buffered.append(poly_inw_buffered)
    return polygons_inw_buffered, poly_too_small_buffer

def remove_small_poly(polygons, poly_too_small_buffer):
    for poly_remove in poly_too_small_buffer:
        gj_reduced = [item for item in polygons.features if item.geometry != poly_remove]
        polygons.features = gj_reduced
    return polygons