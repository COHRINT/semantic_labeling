#!/usr/bin/env python

"""
Training data creation module for learning semantic labels of objects in space

Author: Ian Loefgren
Date created: 03/20/2018
Last modified:

Data generator class contains methods:
    1. sample rand points in the space (self._map_bounds)
    2. get the softmax model classes for each rand sampled point

"""

import itertools
import re
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches

from map_maker import Map
from softmaxModels import *


class DataGenerator(object):

    def __init__(self,map_yaml):

        self.map = Map(map_yaml)
        self.num_samples = 1000

        self.models = []
        self.objects = []


    def sample_points(self):
        """
        Samples <self.num_samples> number of points from space, evaluates all
        softmax classes at that point, and then does a weighted sample from
        the classes bases on the evaluations. This is the observation model

        Returns:
            - <self.num_samples> observation models
        """
        x = np.random.uniform(low=self.map.bounds[0],high=self.map.bounds[2],size=self.num_samples)
        y = np.random.uniform(low=self.map.bounds[1],high=self.map.bounds[3],size=self.num_samples)
        
        data = []
        # print(self.models)
        for i in range(0,len(x)):
            probs = []
            c = len(self.models)*5 # five softmax classes per model
            for mod in self.models:
                for j in range(0,5):
                    probs.append(mod[0].pointEvalND(j,[x[i],y[i]]))
            probs = [x/len(self.models) for x in probs]
            obs = np.random.choice(c,p=probs)
            data.append([[x[i],y[i]],obs])

        return data

    def make_models(self):
        for obj in self.map.objects:
            model = Softmax()
            model.buildOrientedRecModel(self.map.objects[obj].centroid,
                                        self.map.objects[obj].orient,
                                        self.map.objects[obj].x_len,
                                        self.map.objects[obj].y_len)
            self.models.append([model,obj])
            self.objects.append(obj)
        self.combinations = list(itertools.permutations(self.objects))
        # print(self.combinations)

    def models2obs(self,models):
        pass

    def obs2models(self,obs):
        """Map received observation to the appropriate softmax model and class.
        Observation may be a str type with a pushed observation or a list with
        question and answer.
        """
        # print(obs)
        sign = None
        model = None
        model_name = None
        room_num = None
        class_idx = None
        # check if observation is statement (str) or question (list)
        if type(obs) is str:
            # obs = obs.split()
            if 'not' in obs:
                sign = False
            else:
                sign = True
        else:
            sign = obs[1]
            obs = obs[0]

        # find map object mentioned in statement
        for obj in self.map.objects:
            if re.search(obj,obs.lower()):
                model = self.map.objects[obj].softmax
                model_name = self.map.objects[obj].name
                for i, room in enumerate(self.map.rooms):
                    if obj in self.map.rooms[room]['objects']: # potential for matching issues if obj is 'the <obj>', as only '<obj>' will be found in room['objects']
                        room_num = i+1
                        # print(self.map.rooms[room]['objects'])
                        # print(room_num)
                break

        # if observation is relative to the cop
        # if re.search('cop',obs.lower()):
        #     model = Softmax()
        #     model.buildOrientedRecModel((pose[0],pose[1]),pose[2]*180/np.pi,0.5,0.5,steepness=2)
        #     room_num = 1
        #     for i in range(0,len(model.weights)):
        #         model.weights[i] = [0,0,model.weights[i][0],model.weights[i][1]]

        # if no model is found, try looking for room mentioned in observation
        # if model is None:
        #     for room in self.map_.rooms:
        #         if re.search(room,obs.lower()):
        #             model = self.map_.rooms[room]['softmax']
        #             room_num = 0
        #             break

        # find softmax class index
        if re.search('inside',obs.lower()) or re.search('in',obs.lower()):
            class_idx = 0
        if re.search('front',obs.lower()):
            class_idx = 1
        elif re.search('right',obs.lower()):
            class_idx = 2
        elif re.search('behind',obs.lower()):
            class_idx = 3
        elif re.search('left',obs.lower()):
            class_idx = 4
        # elif 'near' in obs:
        # 	class_idx = 5

        # print(model,model_name,class_idx,sign)
        return model, model_name, class_idx, sign

    def bayes(self,obs):
        '''
        compute P(C | O) using Bayes' rule
        '''
        probs = []
        p_c = 1/(len(self.combinations))
        p_o = sum([x*p_c for x in self.likelihoods])
        for i in range(0,len(self.combinations)):
            p_c_o = 0
            p_o_c = 0
            for o in obs:
                obj = o[0]
                c_idx = o[1]
                p_o_c += np.log(self.likelihoods[self.combinations[i].index(obj)][c_idx])

            p_c_o = p_o_c + np.log(p_c) - np.log(p_o)
            probs.append(p_c_o)
        print(np.exp(probs))
        return (max(probs),probs.index(max(probs)))

    def compute_likelihoods(self,samples):
        '''
        - compute P(O | C) using the naive bayes assumption:
            - P(O | C) = prod_O ( P(O_i | C) ) = 
            sum(ln(P(O_i | C)))
        '''
        self.likelihoods = {}
        for s in samples:
            if int(s[1]/5) not in self.likelihoods:
                self.likelihoods[int(s[1]/5)] = [0,0,0,0,0]
            # else:
                # self.likelihoods[s[1]] += 1
            self.likelihoods[int(s[1]/5)][s[1]%5] += 1
        for key in self.likelihoods:
            for i in range(0,len(self.likelihoods[key])):
                self.likelihoods[key][i] /= len(samples)
                
            # self.likelihoods[key] /= len(samples)

        print(self.likelihoods) 

    def visual(self):
        '''
        display objects and their correct labels
        '''
        # fig = Figure
        # fig = plt.figure()
        # canvas = FigureCanvas(fig)
        # ax = fig.add_subplot(111)
        # ax = plt.gca()
        fig, ax = plt.subplots(1)

        self.delta = 0.1
        x_space,y_space = np.mgrid[self.map.bounds[0]:self.map.bounds[2]:self.delta,self.map.bounds[1]:self.map.bounds[3]:self.delta];

        m = self.map
        cnt = 0
        for obj in m.objects:
            cent = m.objects[obj].centroid;
            x = m.objects[obj].x_len;
            y = m.objects[obj].y_len;
            theta = m.objects[obj].orient;
            col = m.objects[obj].color
            # tmp = patches.Ellipse((cent[0],cent[1]),width = x, height=y,angle=theta,fc=col,ec='black');
            # tmp = patches.Ellipse((cent[0] - x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
    
            tmp = patches.Rectangle(self.findLLCorner(m.objects[obj]),width = x, height=y,angle=theta,fc=col,ec='black');
            # tmp = patches.Rectangle((cent[0]- x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');

            ########
                # tmp = patches.Rectangle((cent[0]- x/2,cent[1]-y/2),width = x, height=y,angle=theta,fc=col,ec='black');
            #tmp = patches.Rectangle(self.findLLCorner(m.objects[obj]),width = x, height=y,angle=theta,fc=col,ec='black');
            ax.add_patch(tmp)
            ax.text(cent[0]+0.5,cent[1],obj)
            ax.text(cent[0],cent[1],cnt)
            cnt += 1

        # figsize = fig.get_size_inches()
        # fig.set_size_inches(figsize[0],figsize[1])

        ax.set_xlim(self.map.bounds[0],self.map.bounds[2])
        ax.set_ylim(self.map.bounds[1],self.map.bounds[3])

        plt.show()

    def findLLCorner(self, obj):
        """ Returns a 2x1 tuple of x and y coordinate of lower left corner """
                # LOL the x and y dimensions are defined to be length and width in map_maker...
                # Below they are used oppositely
        length = obj.y_len
        width = obj.x_len

        theta1 = obj.orient*math.pi/180;
        h = math.sqrt((length/2)*(length/2) + (width/2)*(width/2));
        theta2 = math.asin((length/2)/h);

        s1 = h*math.sin(theta1+theta2);
        s2 = h*math.cos(theta1+theta2);

        return (obj.centroid[0]-s2, obj.centroid[1]-s1)  



if __name__ == "__main__":
    map_fn = '2obj_map.yaml'
    dg = DataGenerator(map_fn)
    dg.make_models()
    l = dg.sample_points()
    dg.compute_likelihoods(l)
    obs = ["The robot is left of the checkers table.",
           "The robot is in front of the bookcase.",
            "The robot is in front of the desk."]
    obs_classes = []
    for o in obs:
        _,m_name,class_idx,_ = dg.obs2models(o)
        obs_classes.append([m_name,class_idx])
    prob, prob_idx = dg.bayes(obs_classes)
    print(dg.combinations[prob_idx])
    dg.visual(dg.combinations[prob_idx])
    print(prob,prob_idx)
    # print(l)
