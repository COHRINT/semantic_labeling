#!/usr/bin/env python

""" Summary:
    Creates a map object from an inputed 'map.yaml' file (in models dir)
        with softmax LIKELIHOODs
    Map includes:
        1) General info: name, bounds.max_x_y, bounds.min_x_y, origin
        2) Object hash: 'self.objects', each member is a Map_Object
        3) Rooms : self.rooms['room_name']['lower_l' OR 'upper_r' OR 'likelihood']
            access the room's lower left coordinate and upper right coord
    Map_Object includes:
        name, color, centroid[x, y], major axis, minor axis,
        orientation from the object's major axis to the map's positive x axis,
        shape (available shapes: oval and rectangle),
        softmax likelihood

How to make a new occ_grid:
1) Make the yaml file
2) python map_maker.py "name-of-map-file" # pass in name of map file
--------DON'T INCLUDE .yaml----
3) View the ouput named "name-of-map-file._occupancy.png"
"""

__author__ = "LT"
__copyright__ = "Copyright 2017, COHRINT"
__credits__ = ["Luke Babier", "Ian Loefgren", "Nisar Ahmed"]
__license__ = "GPL"
__version__ = "2.0.1" # Command line args added for creating occ_grid
__maintainer__ = "LT"
__email__ = "luba6098@colorado.edu"
__status__ = "Development"

from pdb import set_trace

import sys # command line args
import png
import math
import yaml
import os 			# path capabilities
from collections import OrderedDict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches
from softmaxModels import *

class Map(object):
    """ Map Object from map.yaml file (located in models dir)

    Map includes:
        1) General info: self.name (str),
                        self.size[max_x, max_y] (float list),
                        self.origin[x, y] (float list)
        2) Object hash: 'self.objects', each member is a Map_Object

    Parameters
    ----------
    yaml_file : map1.yaml, map2.yaml, etc

    """

    def __init__(self, yaml_file):

        # load yaml file as a dictionary
        # cfg = self._find_yaml(yaml_file) #TODO: generalize this!

        # cfg = yaml.load(open('../models/' + yaml_file, 'r'));

        cfg = None
        with open('../'+yaml_file,'r') as file:
            cfg = yaml.load(file)

        if cfg is not None:

            # Get map's general info
            self.name = cfg['info']['name']
            self.bounds = [cfg['info']['bounds']['min_x'],cfg['info']['bounds']['min_y'],
                            cfg['info']['bounds']['max_x'],cfg['info']['bounds']['max_y']]
            self.origin = [cfg['info']['origin']['x_coord'], cfg['info']['origin']['y_coord']]

            # Add room boundaries to the map
            try:
                self.rooms = {}
                lower_l = list()
                upper_r = list()
                for room in cfg['info']['rooms']:
                    min_x = cfg['info']['rooms'][room]['min_x']
                    min_y = cfg['info']['rooms'][room]['min_y']
                    max_x = cfg['info']['rooms'][room]['max_x']
                    max_y = cfg['info']['rooms'][room]['max_y']
                    self.rooms[room] = {}
                    self.rooms[room]['min_x'] = min_x
                    self.rooms[room]['min_y'] = min_y
                    self.rooms[room]['max_x'] = max_x
                    self.rooms[room]['max_y'] = max_y
                    length = max_x - min_x
                    width = max_y - min_y
                    cent = [min_x + length/2,min_y+width/2]
                    self.rooms[room]['softmax'] = Softmax()
                    self.rooms[room]['softmax'].buildOrientedRecModel(cent, 0.0, length, width,steepness=5)
                    for i in range(0,len(self.rooms[room]['softmax'].weights)):
                        self.rooms[room]['softmax'].weights[i] = [0,0,self.rooms[room]['softmax'].weights[i][0],self.rooms[room]['softmax'].weights[i][1]];
                    self.rooms[room]['objects'] = cfg['info']['rooms'][room]['objects']
            except KeyError as e:
                print('No rooms found, skipping...')

            # Store map's objects in self.objects hash
            self.objects = {}
            for item in cfg:
                if item != 'info':	# if not general info => object on map
                    map_obj = Map_Object(cfg[item]['name'],
                                        cfg[item]['color'],
                                        [cfg[item]['centroid_x'], cfg[item]['centroid_y']],
                                        cfg[item]['x_len'],
                                        cfg[item]['y_len'],
                                        cfg[item]['orientation'],
                                        cfg[item]['shape']
                                        )
                    self.objects[map_obj.name] = map_obj
        else:
            print("***ERROR***, Could not find map yaml file!")
            print("Check the 'No such file or directory' error above's search path")
            print("If the map_.yaml name is correct, check _find_yaml() in map_maker.py")
            raise IOError

    # Searches the yaml_dir for the given yaml_file
    # Returns a python dictionary if successful
    # Returns 'None' for failure
    def _find_yaml(self, yaml_file):
        yaml_dir = 'models'
        try:
            # navigate to yaml_dir
            cfg_file = os.path.dirname(__file__) \
                + '/../' + yaml_dir + '/' + yaml_file
            # return dictionary of yaml file
            with open(cfg_file, 'r') as file:
                 return yaml.load(file)
        except IOError as ioerr:
            print(str(ioerr))
            return None

class Occupancy_Grid(object):
    # Initializing the object creates and saves a greyscale png of the map in the current directory
    # also will create the proper yaml

    #NOTE we can only turn 90 degrees with the current form of setup

    # TODO make this modular
    
    def __init__(self, _map, res):
        self._map = _map
        
        # TODO pull 1000 and 500 from the map bounds
        self.x_pix = 1000
        self.y_pix = 500
        # main rows file to organize pixels
        self.rows = []
        
        self.white_out()
        self.load_objs()

        self.save_image()
        self.make_occupancy_yaml(_map.name,_map.bounds,res)

    def white_out(self):
        white = 255
        for i in range(self.y_pix):
            cols = []
            for i in range(self.x_pix):
                cols.append(white)
            self.rows.append(cols)

    def load_objs(self):
        objects = self._map.objects
        for obj in objects:
            print(objects[obj].name)
            cent = objects[obj].centroid
            x_len = objects[obj].x_len
            y_len = objects[obj].y_len
            orient = objects[obj].orient
            self.fill_rectangle(cent,x_len,y_len,orient)

    def fill_rectangle(self, cent,x_len, y_len,orient):
        # wall x_len = 10m, y_len = 1m centroid = [0,2] no orientation
        # start at centroid, jump left and move right filling in pixels by rows
        cent_indices = self.centroid2rowIndex(cent)
#        print("cent indices: " + str(cent_indices))
        if orient == 90 or orient == 270:
            x_len, y_len = y_len,x_len
            
        ul_x = int(cent_indices[0][0] - (math.ceil(x_len*50) -1))
        ul_y = int(cent_indices[0][1] - (math.ceil(y_len*50) -1))
        ul_index = [ul_x, ul_y]
        lr_x = int(cent_indices[3][0] + (math.ceil(x_len*50) -1))
        lr_y = int(cent_indices[3][1] + (math.ceil(y_len*50) -1))
        lr_index = [lr_x,lr_y]

        # outer loop = rows
        # inner loop = columns
        try:
            for i in range(ul_y,lr_y+1): # loop through rows
                for j in range(ul_x,lr_x+1):
                    self.rows[i][j] = 0
                    #print(str(i)+","+str(j))
        except IndexError:
            print("Index Error")
            print("i: "+str(i))
            print("j: "+str(j))

    def centroid2rowIndex(self,centroid):
        # returns list of 4 tuples surrounding the centroid location
        # [0,2] -> [ [499,49] [500,49]
        #            [499,50] [500,50]]
        # [0,0] -> [ [499, 249] [500,249] # 2m -> 200cm
        #            [499, 250] [500,250]
        # [2,0] -> [ [699, 249] [700,249]
        #            [699, 250] [700, 250]
        # returns -1 in a location if that location is out of bounds

        # start at [0,0]
        #cents = [[499,249],[500,249],[499,250],[500,250]]
        x = round(centroid[0],2) # -1
        y = round(centroid[1],2) # 2

        # check x input
        if -5 <= x <= 5 and -2.5 <= y <= 2.5:
            pass
        else:
            print("Bad y or x value")
            print("X: " + str(x) + " Y: " + str(y))
            set_trace()
        
        
        cent_x = int(499 + (x*100)) # resolution here
        cent_y = int(249 - (y*100)) # y's go up for lower values
        cent_x2 = cent_x + 1
        cent_y2 = cent_y + 1

        off_map_value = -1
        if cent_x < 0 or cent_x > 999:
            cent_x = off_map_value
        if cent_x2 < 0 or cent_x2 > 999:
            cent_x2 = off_map_value
        if cent_y < 0 or cent_y > 499:
            cent_y = off_map_value
        if cent_y2 < 0 or cent_y2 > 499:
            cent_y2 = off_map_value
        return [[cent_x,cent_y],[cent_x2,cent_y],[cent_x,cent_y2],[cent_x2,cent_y2]]

        
    def save_image(self):
        w = png.Writer(self.x_pix, self.y_pix, greyscale=True) # first comes png cols then png rows
        f = open(self._map.name.lower()+'_occupancy.png', 'wb')
        w.write(f, self.rows)
        f.close()

    def make_occupancy_yaml(self,map_name, bounds,res,occ_thresh=0.2,free_thresh=0.65):
        yaml_content = {'image': map_name.lower()+'_occupancy.png',
                        'resolution': res,
                        'origin': [bounds[0],bounds[1],0.0],
                        'occupied_thresh': occ_thresh,
                        'free_thresh': free_thresh,
                        'negate': 0}
        
        file_name = os.path.dirname(__file__) + '' + map_name.lower() + '_occupancy.yaml'

        with open(file_name,'w') as yaml_file:
            yaml.safe_dump(yaml_content,yaml_file,allow_unicode=False)

        
class Map_Object(object):
    """
    Objects like chairs, bookcases, etc to be included in the map object
    -Derived from a map.yaml file (in models dir)

    Map_Object includes:
        name (str), color (str), centroid[x, y] (float list), major axis (float),
        minor axis (float),
        orientation from the object's major axis to the map's positive x axis (float)
        shape (str) (available shapes: oval and rectangle)
        softmax likelihood

    Parameters
    ----------
    name: str
        Name of obj
    color: str
        Color of obj
    centroid : 2x1 float list
        Centroid location [x, y] [m]
    x_len: float
        x axis length of obj [m] (before rotation)
    y_len: float
        y axis width of obj [m] (before rotation)
    orient : float
        Radians of turn from upward direction to the left (rotation on its centroid)
    shape : str
        Values accepted: 'rectangle' or 'oval'
    """
    def __init__(self,
                name='wall',
                color='darkblue',
                centroid=[0.0,0.0],
                x_len = 0.0,
                y_len = 0.0,
                orient=0.0,
                shape = 'rectangle'
                ):
        self.name = name
        self.color = color
        self.centroid = centroid
        self.centroid[0] = float(centroid[0])
        self.centroid[1] = float(centroid[1])
        self.x_len = x_len
        self.y_len = y_len
        self.orient = orient

        self._pick_shape(shape)

        # create the objects likelihood
        self.softmax = Softmax()
        self.get_likelihood()

    def get_likelihood(self):
        """
        Create and store corresponding likelihood.
        Approximate all shapes as rectangles
        """
        # 90 on rotation b/c coords are different
        self.softmax.buildOrientedRecModel(self.centroid,self.orient + 90, self.x_len, self.y_len, steepness=1)
        for i in range(0,len(self.softmax.weights)):
            self.softmax.weights[i] = [0,0,self.softmax.weights[i][0],self.softmax.weights[i][1]];

    # Selects the shape of the obj
    # Default = 'rectangle' --- 'oval' also accepted
    def _pick_shape(self, shape):

        if shape == 'oval':
            self.shape = 'oval'
        else:
            self.shape = 'rectangle'

def test_map_obj():
    map1 = Map('map2.yaml')

    if hasattr(map1, 'name'): # check if init was successful
        print(map1.name)
        print(map1.objects['dining table'].color)
        print(map1.rooms['dining room']['lower_l'])
        print(map1.rooms['kitchen']['upper_r'])

    else:
        print('fail')

def test_likelihood():
    map2 = Map('map2.yaml')
    if hasattr(map2, 'name'):
        for obj in map2.objects:
            print(obj)
        print("Dining table:")
        print (map2.objects['dining table'].softmax.weights)
        print (map2.objects['dining table'].softmax.bias)
        print (map2.objects['dining table'].softmax.size)
        print("Mars Poster:")
        print(map2.objects['mars poster'].softmax.weights)
        print("Dining Room: ")
        print(map2.rooms['dining room']['softmax'].weights)
    else:
        print("Failed to initialize Map Object.")
        raise

def test_occ_grid_gen():
    if len(sys.argv) != 2:
        print("ERROR: Please pass in a map name")
        print("E.g: $ python map_maker.py mapa")
        raise
    map_name = sys.argv[1]
    if '.yaml' not in map_name:
        map_name = map_name + '.yaml'
    print("Map name: " + map_name)
    _map = Map(map_name)
    res = 0.01
    occ = Occupancy_Grid(_map,res)
#    print(occ.fill_rectangle([-2.0,-2.19],1.0,0.62,0))
#    print(occ.fill_rectangle([0,0],1.0,0.62,90))
#    occ.save_image()
#    print(occ.centroid2rowIndex([-1,2]))

if __name__ == "__main__":
    #test_map_obj()
    # test_likelihood()
    #test_occ_grid_gen()
    test_occ_grid_gen() # this implementation will use python's png
