# -*- coding: utf-8 -*-


def change_projection_to_overhead(df, lat, lon):
    """Changes the latitude and longitude projection so that it is from overhead
    
    :param df: Dataframe with original latitude and longitude
    :type df: dataframe
    :param lat: Latitude column name
    :type lat: string
    :param lon: Longitude column name
    :type lon: string
    :return: Dataframe with changed projection
    :rtype: dataframe
    """

    import pyproj

    crs = pyproj.CRS.from_epsg(4326)
    projection = pyproj.Transformer.from_crs(crs, crs.geodetic_crs)

    for row in range(len(df)):
        df.at[row, lat] = projection.transform(df[lon][row], df[lat][row])[0]
        df.at[row, lon] = projection.transform(df[lon][row], df[lat][row])[1]

    return df


def change_projection_to_original(df, lat, lon):
    """Changes the latitude and longitude projection from overhead to original
    
    :param df: Dataframe with changed latitude and longitude projection
    :type df: dataframe
    :param lat: Latitude column name
    :type lat: string
    :param lon: Longitude column name
    :type lon: string
    :return: Dataframe with original projection
    :rtype: [type]
    """

    import pyproj

    crs = pyproj.CRS.from_epsg(4326)
    projection = pyproj.Transformer.from_crs(crs.geodetic_crs, crs)

    for row in range(len(df)):
        df.at[row, lat] = projection.transform(df[lon][row], df[lat][row])[0]
        df.at[row, lon] = projection.transform(df[lon][row], df[lat][row])[1]

    return df
