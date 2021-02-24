#This script allows to retrieve some metadata from S1 by using the Terracope catalogue opensearch functionality
# The aim is to get the RO intersection with a certain BBOX in a certain period of interest

import requests
import datetime
import json
import re
import sys
import dateutil.parser as dp
import numpy as np
from shapely.geometry import Polygon


class OpenSearch:

    def __init__(self):
        self._collections = None

    def getCollectionParameters(self, ElasticSearchURL, printURL=False):
        if self._collections is None:
            'get parameters from the collections'

            if printURL:
                print(ElasticSearchURL + 'collections')

            CatalogCollections = requests.get(ElasticSearchURL + 'collections')
            CollJson = CatalogCollections.json()
            collections = []
            featJson = CollJson['features']

            for f in featJson:
                colldetails = {}
                keylist = list(f.keys())

                for k in keylist:
                    if k != 'properties':
                        colldetails[k] = f[k]
                    else:
                        prop = f['properties']
                        pkeys = list(prop.keys())
                        propdict = {}

                        for p in pkeys:
                            propdict[p] = prop[p]
                        colldetails['properties'] = propdict
                collections.append(colldetails)
            self._collections = (collections)

        return self._collections


    def OpenSearch_metadata_retrieval(self, start, end, geo):

        def coord_to_bbox(gj_poly):

            ###################### DEFINE GEOMETRY OF POLGYON #######################
            #########################################################################
            if gj_poly.geometry.type == 'Polygon':
                x_coord = [float(item[0]) for item in gj_poly.geometry['coordinates'][0]]
                y_coord = [float(item[1]) for item in gj_poly.geometry['coordinates'][0]]
            elif gj_poly.geometry.type == 'MultiPolygon':
                x_coord = [float(item[0]) for item in gj_poly.geometry['coordinates'][0][0]]
                y_coord = [float(item[1]) for item in gj_poly.geometry['coordinates'][0][0]]

            field = Polygon(zip(x_coord, y_coord))
            minx, miny, maxx, maxy = field.bounds
            return minx, miny, maxx, maxy

        def findProducts(urn, ElasticSearchURL, start=datetime.datetime(2015, 1, 1, 0, 0, 0).isoformat(),
                         end=datetime.datetime(2023, 12, 31, 23, 59, 59).isoformat(),latmin = -90.0, latmax = 90.0, lonmin = -180.0, lonmax = 180.0,
                        ccmin=0.0, ccmax=100.0,
                         prstart=datetime.datetime(2015, 1, 1, 0, 0, 0).isoformat(),
                         prend=datetime.datetime(2021, 12, 31, 23, 59, 59).isoformat(),
                         tID='', onTerrascope=True, printURL=False):
            dict_metadata_ascending = dict()
            dict_metadata_descending = dict()
            productsList = []
            bbox = str(lonmin) + ',' + str(latmin) + ',' + str(lonmax) + ',' + str(latmax)
            requestbasestring = ElasticSearchURL + 'products?collection=' + urn + "&start=" + str(start) + "&end=" + str(
                end) + '&bbox=' + bbox
            requestbasestring = requestbasestring + '&modificationDate=[' + str(prstart) + ',' + str(prend) + "["

            if 'S2' in urn:  # cloud cover is not relevant for S1 products
                requestbasestring = requestbasestring + '&cloudCover=[' + str(ccmin) + ',' + str(ccmax) + ']'
                if tID != '':  # there are tile IDs only for S2 products
                    requestbasestring = requestbasestring + '&sortKeys=title,,0,0&tileId=' + tID
                else:
                    requestbasestring = requestbasestring + '&sortKeys=title,,0,0'
            if onTerrascope:
                requestbasestring = requestbasestring + '&accessedFrom=MEP'
            if printURL:
                print(requestbasestring + '&startIndex=1')  # printing this is useful if you want to paste it in a browser

            products = requests.get(requestbasestring + '&startIndex=1')
            productsJson = products.json()
            numProducts = productsJson['totalResults']
            print(str(numProducts) + ' products found between ' + str(start) + ' and ' + str(end) + ' produced between ' + str(
                prstart) + ' and ' + str(prend))
            itemsPerPage = int(productsJson['itemsPerPage'])

            if numProducts > 10000:
                print('too many results (max 10000 allowed), please narrow down your search')
                return (['too many results'])
            else:
                if numProducts > 0:
                    for ind in range(int(numProducts / itemsPerPage) + 1):
                        startindex = ind * itemsPerPage + 1
                        products = requests.get(requestbasestring + '&startIndex=' + str(startindex))
                        productsJson = products.json()
                        features = productsJson['features']

                        for f in features:
                            productdetail = {}
                            # productdetail['productID'] = f['id']
                            # productdetail['bbox'] = f['bbox']
                            productdetail['productDate'] = f['properties']['date'].rsplit('T')[0] # only take the rounded data wihtout info on hours

                            #productdetail['productPublishedDate'] = f['properties']['published']
                            productdetail['productTitle'] = f['properties']['title']
                            productdetail['relativeOrbit'] = \
                            f['properties']['acquisitionInformation'][1]['acquisitionParameters']['relativeOrbitNumber']
                            #productdetail['productType'] = f['properties']['productInformation']['productType']

                            # if 'S2' in f['id']:
                            #     productdetail['cloudcover'] = f['properties']['productInformation']['cloudCover']
                            #     productdetail['tileID'] = f['properties']['acquisitionInformation'][1]['acquisitionParameters'][
                            #         'tileId']
                            # else:
                            #     productdetail['cloudcover'] = ''
                            #     productdetail['tileID'] = ''

                            # filelist = []
                            # linkkeys = f['properties']['links'].keys()
                            #
                            # for l in linkkeys:
                            #     for fil in f['properties']['links'][l]:
                            #         filedetails = {}
                            #         filedetails['filetype'] = l
                            #
                            #         if onTerrascope:
                            #             filedetails['filepath'] = fil['href'][7:]
                            #         else:
                            #             filedetails['filepath'] = fil['href']
                            #
                            #         if l == 'previews':
                            #             filedetails['category'] = fil['category']
                            #             filedetails['title'] = fil['category']
                            #
                            #         if (l == 'alternates') | (l == 'data'):
                            #             filedetails['category'] = fil['title']
                            #             filedetails['title'] = fil['title']
                            #
                            #         if l == 'related':
                            #             filedetails['category'] = fil['category']
                            #             filedetails['title'] = fil['title']
                            #         filedetails['length'] = fil['length']
                            #         filelist.append(filedetails)
                            #
                            # productdetail['files'] = filelist
                            productsList.append(productdetail)
                            if 'DESCENDING' in productdetail['productTitle']:
                                dict_metadata_descending.update({productdetail['productDate']:  productdetail['relativeOrbit'] })
                            else:
                                dict_metadata_ascending.update({productdetail['productDate']:  productdetail['relativeOrbit'] })
                    return dict_metadata_descending, dict_metadata_ascending

        # URL catalogue
        esOpsURL='https://services.terrascope.be/catalogue/' #ops environment
        collectionInformation=self.getCollectionParameters(esOpsURL,printURL=True)

        S1_SIGMA_list = []
        for f in collectionInformation:
            if 'S1_GRD_SIGMA' in f['id']:
               S1_SIGMA_list.append(f['id'])

        #### some parameters to define for retrieving metadata from the S1 SIGMA catalogue
        producttype   = S1_SIGMA_list[0]
        startdate     = datetime.datetime.strptime(start,'%Y-%m-%d').date()
        enddate       = datetime.datetime.strptime(end,'%Y-%m-%d').date()

        minx, miny, maxx, maxy = coord_to_bbox(geo)
        # latitudemin   = -90
        # latitudemax   = 90
        # longitudemin  = -180
        # longitudemax  = 180
        dict_metadata_descending, dict_metadata_ascending = \
            findProducts(ElasticSearchURL= esOpsURL,urn=producttype,
                               start = startdate,
                               end   = enddate,
                               latmin=miny, latmax=maxy, lonmin=minx, lonmax=maxx)
        return dict_metadata_descending, dict_metadata_ascending
