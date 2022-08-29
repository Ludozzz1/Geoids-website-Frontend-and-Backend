import re
import string
from typing import List
from numpy import array, mat
import requests
import shapefile
import numpy as np
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon
import scipy as sp
import scipy.interpolate
from copy import copy, deepcopy
import os

class Geoide:
    metaPattern = re.compile("(?P<key>\S+(?:\s\S+)?)\s*(?::|=)\s+(?P<value>.*)")
    matrixPattern = re.compile("-?\d+\.\d+")
    def __init__(self, path : str = None, url : str = None) -> None:
        self.metadata = {}
        lines : List[str]
        status : int = 0
        matrix : List[List[float]] = []
        if url:
            response = requests.get(url,verify=False)
            if response.status_code != 200:
                raise RuntimeError("cannot fetch geoid from {}".format(url))
            lines = response.text.split('\n')
        elif path:
            with open(path,'r') as file:
                lines = file.readlines()
        else:
            raise RuntimeError("path or url is needed")
        for line in lines:
            if not line: continue
            match status:
                case 0:
                    if "begin_of_head" in line:
                        status = 1
                case 1:
                    if "end_of_head" in line:
                        status = 2
                    else:
                        match = Geoide.metaPattern.match(line)
                        if match:
                            self.metadata[match.group('key')] = match.group('value')
                case 2:
                    matrix.append(list(map(lambda match: float(match[0]), Geoide.matrixPattern.finditer(line))))
        self.matrix = array(matrix,dtype=object)

def bilinear_resampling(geoide,lat_min,lat_max,nrows,lon_min,lon_max,ncols):
    y = np.linspace(float(geoide.metadata['lat max']),float(geoide.metadata['lat min']),int(geoide.metadata['nrows']))
    x = np.linspace(float(geoide.metadata['lon min']),float(geoide.metadata['lon max']),int(geoide.metadata['ncols']))
    matrice = geoide.matrix
    fun = sp.interpolate.interp2d(x,y,matrice,'linear',True,False,-9999)
    ynew = np.linspace(lat_max,lat_min,nrows)
    xnew = np.linspace(lon_min,lon_max,ncols)
    nuova_mat = fun(xnew,ynew)
    return nuova_mat

def bilinear_resampling_glob(geoide,lat_min,lat_max,nrows,lon_min,lon_max,ncols): 
    y = np.linspace(float(geoide.metadata['lat max']),float(geoide.metadata['lat min']),int(geoide.metadata['nrows']))
    x = np.linspace(float(geoide.metadata['lon min']),float(geoide.metadata['lon max']),int(geoide.metadata['ncols'])-2)
    matrice = geoide.matrix
    new_mat = deepcopy(matrice)
    new_mat = np.delete(np.delete(new_mat, -1, 1), 0, 1)
    fun = sp.interpolate.interp2d(x,y,new_mat,'linear',True,False,-9999)
    ynew = np.linspace(lat_max,lat_min,nrows)
    xnew = np.linspace(lon_min,lon_max,ncols)
    nuova_mat = fun(xnew,ynew)
    return nuova_mat


def eliminate_residui_matrix(nrows,ncols,matrix,matrix_glob): 
    new_mat = np.full((nrows,ncols),-9999.0000,dtype=float)
    for i in range(0,nrows,1):
        for j in range(0,ncols,1):
            if matrix[i][j] != -9999 and matrix_glob[i][j] != -9999:
                new_mat[i][j] = matrix[i][j]-matrix_glob[i][j]
            elif matrix[i][j] != -9999 and matrix_glob[i][j] == -9999:
                new_mat[i][j] = matrix[i][j]  
    return new_mat

def add_residui_matrix(nrows,ncols,matrix,matrix_res): 
    new_mat = np.full((nrows,ncols),-9999.0000,dtype=float)
    for i in range(0,nrows,1):
        for j in range(0,ncols,1):
            if matrix[i][j] != -9999 and matrix_res[i][j] != -9999:
                new_mat[i][j] = matrix[i][j]+matrix_res[i][j]
            elif matrix[i][j] != -9999 and matrix_res[i][j] == -9999:
                new_mat[i][j] = matrix[i][j]  
    return new_mat

def if_in_shape(lat,lon,polygon): 
    point = Point(lon,lat)
    return polygon.contains(point)

def final_cal(k,lat,lon): 
    r = shapefile.Reader(k)
    shapes = r.shapes()
    polygon = shape(shapes[0])
    point = Point(lon,lat)
    shortest_distance = polygon.distance(point)
    return shortest_distance

def calculate_systematism(lat,lon,coeff): 
   systematism = coeff[0]*lat + coeff[1]*lon + coeff[2]*lat*lon + coeff[3]
   return systematism

def create_matrix(delta_lat,delta_lon,nrows,ncols,k,offset_lat,offset_lon,matrix): 
    matrice_list = []
    residui_giusti = []
    r = shapefile.Reader(k)
    shapes = r.shapes()
    polygon = shape(shapes[0])
    for i in range(0,nrows,1):
        for j in range(0,ncols,1):
            if if_in_shape((-i*delta_lat)+offset_lat,(j*delta_lon)+offset_lon,polygon):
                matrice_list.append([(-i*delta_lat)+offset_lat,(j*delta_lon)+offset_lon,((-i*delta_lat)+offset_lat)*((j*delta_lon)+offset_lon),1])
                residui_giusti.append(matrix[i][j])
    matrice_list = np.array(matrice_list,dtype=object)
    residui_giusti = np.array(residui_giusti,dtype=object)
    coeffi = coeff(matrice_list,residui_giusti)
    return coeffi

def coeff(matrix, residui): 
    matrix_t = np.transpose(matrix)
    appoggio = np.dot(matrix_t,matrix)
    appoggio = appoggio.astype('float64')
    coeff = np.dot(np.dot(np.linalg.inv(appoggio),matrix_t),residui)
    coeff = np.transpose(coeff)
    return coeff

def remove_systematism(nrows,ncols,matrix,coeff,delta_lat,offset_lat,delta_lon,offset_lon): 
    for i in range(0,nrows,1):
        for j in range(0,ncols,1):
            if matrix[i][j] == -9999:
                matrix[i][j] = -9999
            else:
                matrix[i][j] = matrix[i][j] - calculate_systematism((-i*delta_lat)+offset_lat,(j*delta_lon)+offset_lon,coeff)
    return matrix

def add_systematism(nrows,ncols,matrix,coeff,delta_lat,offset_lat,delta_lon,offset_lon):
    for i in range(0,nrows,1):
        for j in range(0,ncols,1):
            if matrix[i][j] == -9999:
                matrix[i][j] = -9999
            else:
                matrix[i][j] = matrix[i][j] + calculate_systematism((-i*delta_lat)+offset_lat,(j*delta_lon)+offset_lon,coeff)
    return matrix

def cal_weight(shape_file_array,lat,lon,alpha): 
    distance = []
    weight = []
    count = 0
    see = 0
    contatore_ennesimo = 0
    k = len(shape_file_array)
    for x in shape_file_array:
        r = shapefile.Reader(x)
        shapes = r.shapes()
        polygon = shape(shapes[0])
        if if_in_shape(lat,lon,polygon):
            see = 1
            for y in shape_file_array:
                if y != x:
                    distance.append(final_cal(y,lat,lon))
                else: distance.append(-9999)
    if see == 1:
        for x in distance:
            if x != -9999:
                weight.append((1)/((x**alpha)+k))
                count = count + (1)/((x**alpha)+k)
            else: 
                weight.append(-9999)
        for x in weight:
            if weight[contatore_ennesimo] == -9999:
                weight[contatore_ennesimo] = 1-count
                contatore_ennesimo = contatore_ennesimo +1
            else: contatore_ennesimo = contatore_ennesimo+1
        return weight
    else:
        for x in shape_file_array:
             distance.append(final_cal(x,lat,lon))
        for x in distance:
            weight.append((1)/((x**alpha)+k))
            count = count + (1)/((x**alpha)+k)
        for x in weight:
            weight[contatore_ennesimo] = weight[contatore_ennesimo]/count
            contatore_ennesimo = contatore_ennesimo+1
        return weight

def semi_final_result(matrix_array,weight,i,j): 
    k = len(weight)
    result = 0
    weight_tot = 0
    count = 0
    var = 0
    while var < k:
        matrix = matrix_array[var]
        if matrix[i][j] == -9999:
            matrix_array.pop(var)
            weight.pop(var)
            k = k-1
        else: var = var+1
    for s in weight:
        weight_tot = weight_tot + s
    for s in weight: 
        weight[count] = weight[count]/weight_tot
        count = count +1
    for x in range(0,k):
        matrix = matrix_array[x]
        result = result + matrix[i][j]*weight[x]
        return result
    result = -9999
    return result

def final_matrix(matrix_array,nrows,ncols,offset_lat,offset_lon,delta_lat,delta_lon,shape_file_array,alpha,master,master_coeff,matr_residui):
    new_mat = np.full((nrows,ncols),-9999.0000,dtype=float)
    copia_serie = []
    for i in range(0,nrows,1):
        print(i)
        for j in range(0,ncols,1):
            copia_serie = matrix_array.copy()
            weight = cal_weight(shape_file_array,(-i*delta_lat)+offset_lat,(j*delta_lon)+offset_lon,alpha)
            new_mat[i][j] = semi_final_result(copia_serie,weight,i,j)
    if master:
        new_mat = add_systematism(nrows,ncols,new_mat,master_coeff,delta_lat,offset_lat,delta_lon,offset_lon)
    new_mat = add_residui_matrix(nrows,ncols,new_mat,matr_residui)
    return new_mat


def final_creation(lat_min,lat_max,lon_min,lon_max,delta_lat,delta_lon,matrix,alpha,global_geoid_name,nations_chosen_array,nrows,ncols,nodata):
    i = lon_min
    j = lat_max
    f = open("New_Geoid.txt","x")
    f.write('Glob geoid : '+ global_geoid_name + '\n')
    f.write('Nations:   : ')
    k = 0
    for x in nations_chosen_array:
        k = k+1
    for x in range(0,k-1):
        f.write(nations_chosen_array[x]+', ')
    f.write(nations_chosen_array[k-1] + '\n')
    f.write('Alpha par  =    '+ str(alpha) + '\n')
    f.write('begin_of_head ================================================\n')
    f.write('units      : meters\n')
    f.write('lat min    =    '+str(lat_min)+'\n')
    f.write('lat max    =    '+str(lat_max)+'\n')
    f.write('lon min    =    '+str(lon_min)+'\n')
    f.write('lon max    =    '+str(lon_max)+'\n')
    f.write('delta lat  =    '+str(delta_lat)+'\n')
    f.write('delta lon  =    '+str(delta_lon)+'\n')
    f.write('nrows      =    '+str(nrows)+'\n')
    f.write('ncols      =    '+str(ncols)+'\n')
    f.write('nodata     = '+str(nodata)+'\n')
    f.write('end_of_head ==================================================\n')
    matrix = np.round(matrix,4)
    for row in matrix:
        f.write('    '.join([f"{a:.4f}" for a in row]) + '\n')
    f.close()
    os.rename("New_Geoid.txt","New_Geoid.isg")

def trans_into_geoid(series_url): 
    return [Geoide(None, url) for url in series_url]

def turn_deg(num):
    deg, minutes, seconds,no = re.split('[°\'"]',num)
    return str((float(deg) + float(minutes)/60 + float(seconds)/(60*60)))


def end(global_url,delta_lat,delta_lon,geoids_url,geoids_shapefile,alpha,master,master_number,global_geoid_name,nations_name,nrows,ncols,lat_max,lat_min,lon_max,lon_min): 
    k = 0
    s = 0
    if '°' in delta_lat:
        delta_lat = turn_deg(delta_lat)
    if '°' in delta_lon:
        delta_lon = turn_deg(delta_lon)
    if '°' in lat_max:
        lat_max = turn_deg(lat_max)
    if '°' in lat_min:
        lat_min = turn_deg(lat_min)
    if '°' in lon_max:
        lon_max = turn_deg(lon_max)
    if '°' in lon_min:
        lon_min = turn_deg(lon_min)
    master_coeff = None
    glob_geoid = Geoide(None,global_url) 
    residui_globali = bilinear_resampling_glob(glob_geoid,lat_min,lat_max,nrows,lon_min,lon_max,ncols)
    geoids_series = trans_into_geoid(geoids_url)
    residui_globali_gen = []
    for x in geoids_series:
        residui_globali_gen.append(bilinear_resampling_glob(glob_geoid,float(x.metadata['lat min']),float(x.metadata['lat max']),int(x.metadata['nrows']),float(x.metadata['lon min']),float(x.metadata['lon max']),int(x.metadata['ncols'])))
    for x in geoids_series:
        x.matrix = eliminate_residui_matrix(int(x.metadata['nrows']),int(x.metadata['ncols']),x.matrix,residui_globali_gen[k])
        k += 1
    print('Residui eliminati')
    for x in geoids_series:
        coeffi = create_matrix(float(x.metadata['delta lat']),float(x.metadata['delta lon']),int(x.metadata['nrows']),int(x.metadata['ncols']),geoids_shapefile[s],float(x.metadata['lat max']),float(x.metadata['lon min']),x.matrix)
        x.matrix = remove_systematism(int(x.metadata['nrows']),int(x.metadata['ncols']),x.matrix,coeffi,float(x.metadata['delta lat']),float(x.metadata['lat max']),float(x.metadata['delta lon']),float(x.metadata['lon min']))
        if s == master_number:
            master_coeff = coeffi
        s += 1
    print('Sistematismo calcolato')
    residui_matrix_series = []
    for x in geoids_series:
        matrix = bilinear_resampling(x,lat_min,lat_max,nrows,lon_min,lon_max,ncols)
        residui_matrix_series.append(matrix)
    matr = final_matrix(residui_matrix_series,nrows,ncols,lat_max,lon_min,delta_lat,delta_lon,geoids_shapefile,alpha,master,master_coeff,residui_globali)
    final_creation(lat_min,lat_max,lon_min,lon_max,delta_lat,delta_lon,matr,alpha,global_geoid_name,nations_name,nrows,ncols,-9999.0000)

global_url = r'C:\Users\ludov\Downloads\test-samples\test-samples\global models\WORLD_2015_EIGEN-6C4_gravG_20220523.isg'
delta_lat = 0.01
delta_lon = 0.015
geoids_url = ['https://www.isgeoid.polimi.it/Geoid/Europe/Italy/public/ITG2009_20170606.isg','https://www.isgeoid.polimi.it/Geoid/Europe/France/public/QGF96_20161103.isg']
geoids_shapefile = [r'C:\Users\ludov\Downloads\test-samples\test-samples\shapefiles\3 PUBLIC\Italy\Italy.shp',r'C:\Users\ludov\Downloads\test-samples\test-samples\shapefiles\3 PUBLIC\France\France.shp']
alpha = 1
master = None
master_number = None
glob_geoid_name = 'World 2015 EIGEN-6C4'
nations_name = ['Italy','France']
nrows = 200
ncols= 200
lat_max = 45.5
lat_min = 43.5
lon_max = 8.9
lon_min = 5.9
#end(global_url,delta_lat,delta_lon,geoids_url,geoids_shapefile,alpha,master,master_number,glob_geoid_name,nations_name,nrows,ncols,lat_max,lat_min,lon_max,lon_min)