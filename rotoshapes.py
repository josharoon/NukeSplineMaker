"""This file provodes functionalilty for creating and extracting data from Nuke and requrires a Nuke License to run
it is key to the creation of the datasets and importing predictions back into Nuke. The data created using this code
is supplied as part of the submission incase of lack of access to a Nuke License."""

import fnmatch
import json
import math
import os
import pickle
import random
from copy import copy, deepcopy
import bezier
import torch
from typing import List
import cv2
from Lib import glob
import viz
import nuke
import nuke.rotopaint as rp
import numpy as np


def createRotoNode():
    """create a roto node"""
    rotoNode = nuke.createNode("Roto")
    #set output to rgb
    rotoNode['output'].setValue('rgb')
    #set input to None
    rotoNode.setInput(0, None)
    return rotoNode

def createTransformNode(Input=None):
    """create a transform node"""
    transformNode = nuke.createNode("Transform")
    #set input to None
    transformNode.setInput(0, Input)
    #set transform expression to oscilate between 0 and 5 over 10 frames
    transformNode['translate'].setExpression("sin(frame/10)*5")
    return transformNode

def createBlurNode(Input=None):
    """create a blur node"""
    blurNode = nuke.createNode("Blur")
    #set input to None
    blurNode.setInput(0, Input)
    #set blur expression to oscilate between 0 and 5 over 10 frames
    blurNode['size'].setExpression("sin(frame/10)*5")
    return blurNode

def createGradeNode(Input=None):
    """create a grade node"""
    gradeNode = nuke.createNode("Grade")
    #set input to None
    gradeNode.setInput(0, Input)
    #set red gain to oscilate between 0.1 and 1 over 9 frames
    gradeNode['white'].setSingleValue(False)
    gradeNode['white'].setExpression("((sin(frame/3)*.5)+0.5)*0.9+0.1",0)
    print(gradeNode['white'].getValue(0))
    gradeNode['white'].setExpression("((sin(frame/4)*.5)+0.5)*0.9+0.1",1)
    print(gradeNode['white'].getValue(1))
    gradeNode['white'].setExpression("((sin(frame/6)*.5)+0.5)*0.9+0.1",2)
    print(gradeNode['white'].getValue(2))

    return gradeNode


class point2D:
    def __init__(self, vertex, lftTang, rhtTang):

        self.vertex = vertex[:2]
        self.lftTang = lftTang[:2]
        self.rhtTang = rhtTang[:2]




class nShapeMaster:
    def __init__(self, nPoints=None, rotoNode=None):
        self.nPoints = nPoints
        self.shape = None
        self.ctrlPoints = []
        self.points = []
        self.nPoints = None
        self.rotoNode = None
        self.shape = None
        self.windowSize=[224,224]




    def getCtrlPoints(self):
        """get nuke points from shape"""
        for p in range(len(self.shape)):
            self.ctrlPoints.append(self.shape[p])

    def addPoint(self, shape, x, y, ltx=0, lty=0, rtx=0, rty=0):

        ctrlPoint = rp.ShapeControlPoint(x, y)
        ctrlPoint.leftTangent = (ltx, lty)
        ctrlPoint.featherLeftTangent = (ltx, lty)
        ctrlPoint.rightTangent = (rtx, rty)
        ctrlPoint.featherRightTangent = (rtx, rty)
        shape.append(ctrlPoint)

    def addPoints(self, shape1):
        """add points to shape"""
        for point in self.points:

            if len(point) == 3:
                self.addPoint(shape1, point[0][0], point[0][1], point[1][0], point[1][1], point[2][0], point[2][1])

            else:
                assert len(point) == 2, "point must be a list of 1 or 3 points"
                self.addPoint(shape1, point[0], point[1])

    def printPoint(self,point, time):

        if type(point) == rp.ShapeControlPoint:
            print(
            "center:{} leftTanget: {} rightTangent: {} featherCentre: {} featherleftTangent: {} featherRightTangent: {}".format(point.center.getPosition(time), point.leftTangent.getPosition(time),
                                                               point.rightTangent.getPosition(time) , point.featherCenter.getPosition(time), point.featherLeftTangent.getPosition(time), point.featherRightTangent.getPosition(time)))
        #if numpy array
        elif type(point) == np.ndarray:
            print("center:{}".format(point[1]))
            print("leftTangent:{}".format(point[0]))
            print("rightTangent:{}".format(point[2]))

    def getPointDict(self,point,time):
        """get a dictionary of a point at a position in time  points"""
        pointDict={}
        pointDict["center"]=str(point.center.getPosition(time))
        pointDict["leftTangent"]=str(point.leftTangent.getPosition(time))
        pointDict["rightTangent"]=str(point.rightTangent.getPosition(time))
        pointDict["featherCenter"]=str(point.featherCenter.getPosition(time))
        pointDict["featherLeftTangent"]=str(point.featherLeftTangent.getPosition(time))
        pointDict["featherRightTangent"]=str(point.featherRightTangent.getPosition(time))
        return pointDict

    def printPoints(self, time):
        for point in self.shape:

            self.printPoint(point, time)

    def randomisePoint(self, pointIndex, time, xRange, yRange):
        """add a randomised shuffle and key frame to a single control point"""
        point = self.shape[pointIndex]
        xy = self.getPtCoord(point, time)
        xy[0] += random.randrange(-xRange, xRange)
        xy[1] += random.randrange(-yRange, yRange)
        #don't allow the angle between the point and centre to cross with that of adjacent points
        #get the centre point
        centre = self.getCentre(time)
        #get the angle between the centre and the point
        angle = self.lineNormalAngle(centre, xy)
        #get the angle between the centre and the adjacent points
        #get index of agjacent points and wrap around if needed

        angleLeft = self.lineNormalAngle(centre, self.getPtCoord(self.shape[(pointIndex-1+len(self.shape))%len(self.shape)], time))
        angleRight = self.lineNormalAngle(centre, self.getPtCoord(self.shape[(pointIndex+1)%len(self.shape)], time))
        #check if the angle is between the adjacent points
        if angle < angleLeft or angle > angleRight:
            #don't change the point
            return
        point.center.addPositionKey(time, xy)

    def getPtCoord(self, point, time):
        """get the x,y coords of a point in list format"""
        xy = list(point.center.getPosition(time))[:2]
        return xy

    def calcCentre(self,points):
        """get the centre point of the shape"""
        x=0
        y=0
        for point in points:
            x+=point[0]
            y+=point[1]
        x/=len(points)
        y/=len(points)
        return [x,y]

    def getPtTangentCoord(self,point,time):
        """get the x,y coords of a points tangents in list format"""
        xyLeft = list(point.leftTangent.getPosition(time))[:2]
        xyRight = list(point.rightTangent.getPosition(time))[:2]
        return [xyLeft,xyRight]

    def getAllPointCoords(self,time):
        """get the x,y coords of all points in list format"""
        coords=[]
        for point in self.shape:
            coords.append(self.getPtCoord(point,time))
        return coords

    def getCentre(self,time):
        """get the centre point of the shape"""
        coords=self.getAllPointCoords(time)
        x=0
        y=0
        for coord in coords:
            x+=coord[0]
            y+=coord[1]
        x/=len(coords)
        y/=len(coords)
        return [x,y]

    def growPoints(self,time,amount):
        """grow all from the centre point"""
        centre=self.getCentre(time)
        for point in self.shape:
            xy=self.getPtCoord(point,time)
            x=xy[0]-centre[0]
            y=xy[1]-centre[1]
            x*=amount
            y*=amount
            xy[0]+=x
            xy[1]+=y
            point.center.addPositionKey(time,xy)

    def translatePoints(self,time,x,y):
        """translate all points"""

        for point in self.shape:
            xy=self.getPtCoord(point,time)
            xy[0]+=x
            xy[1]+=y
            point.center.addPositionKey(time,xy)

    def growPointTangent(self,time,pointIndex,amount):
        """grow the tangent of a single point"""
        point=self.shape[pointIndex]
        xyLeft,xyRight=self.getPtTangentCoord(point,time)
        centrePoint=self.getPtCoord(point,time)
        x=xyLeft[0]
        y=xyLeft[1]
        x+=amount
        y-=amount
        #check that the normal between x and y and x and y right point outwards from the centre
        centre=self.getCentre(time)
        normalAngle=self.lineNormalAngle([x, y], xyRight)
        centreAngle=self.lineNormalAngle(centrePoint, centre)
        #print("normalAngle: {} centreAngle: {} pointIndex: {}".format(normalAngle,centreAngle,pointIndex))
        if normalAngle>centreAngle:
            x=xyLeft[0]
            y=xyLeft[1]
            x-=amount
            y+=amount
        point.leftTangent.addPositionKey(time,[x,y])
        point.featherLeftTangent.addPositionKey(time,[x,y])
        xyLeft=[x,y]
        x=xyRight[0]
        y=xyRight[1]
        x-=amount
        y+=amount
        #check that the normal between x and y and x and y left point outwards from the centre
        normalAngle=self.lineNormalAngle([x, y], xyLeft)
        centreAngle=self.lineNormalAngle(centrePoint, centre)
        if normalAngle>centreAngle:
            x=xyRight[0]
            y=xyRight[1]
            x+=amount
            y-=amount
        point.rightTangent.addPositionKey(time,[x,y])
        point.featherRightTangent.addPositionKey(time,[x,y])

    def lineNormalAngle(self, leftPoint, rightPoint):
        """get the normal of a line"""
        x= rightPoint[0] - leftPoint[0]
        y= rightPoint[1] - leftPoint[1]
        x,y=y,-x
        #convert to angle
        angle=math.atan2(y,x)
        return angle

    def growPointsTangent(self,time,amount):
        """grow the tangent of all points"""
        for point in range(len(self.shape)):
            self.growPointTangent(time,point,amount)
        self.getListofCtrlPoints()

    def resetPointsTangent(self,time):
        """reset the tangent of all points"""
        for point in range(len(self.shape)):
            self.resetPointTangent(time,point)
        self.getListofCtrlPoints()

    def resetPointTangent(self,time,pointIndex):
        """reset the tangent of a single point"""
        point=self.shape[pointIndex]
        point.leftTangent.addPositionKey(time,[0,0])
        point.featherLeftTangent.addPositionKey(time,[0,0])
        point.rightTangent.addPositionKey(time,[0,0])
        point.featherRightTangent.addPositionKey(time,[0,0])

    def randomisePointTangent(self,time,pointIndex,xRange,yRange):
        """randomise the tangent of a single point"""
        point=self.shape[pointIndex]
        xyLeft,xyRight=self.getPtTangentCoord(point,time)
        x=xyLeft[0]
        y=xyLeft[1]
        x+=random.randrange(-xRange,xRange)
        y+=random.randrange(-yRange,yRange)
        point.leftTangent.addPositionKey(time,[x,y])
        point.featherLeftTangent.addPositionKey(time,[x,y])
        x=xyRight[0]
        y=xyRight[1]
        x+=random.randrange(-xRange,xRange)
        y+=random.randrange(-yRange,yRange)
        point.rightTangent.addPositionKey(time,[x,y])
        point.featherRightTangent.addPositionKey(time,[x,y])

    def randomisePointsTangent(self,time,xRange,yRange):
        """randomise the tangent of all points"""
        for point in range(len(self.shape)):
            self.randomisePointTangent(time,point,xRange,yRange)

    def randomisePoints(self,time,xRange,yRange):
        """randomise all points"""
        for point in range(len(self.shape)):
            self.randomisePoint(point,time,xRange,yRange)

    def getListofCtrlPoints(self):
        """return a list of control points from the nukeShape"""
        #print("getting list of control points")
        self.npCtrlPoints=[]
        for i in range(self.nPoints):
            #convert point into an numpy array (lefttangent, center, righttangent)
            npPoint=np.array([self.shape[i].leftTangent.getPosition(1),self.shape[i].center.getPosition(1),self.shape[i].rightTangent.getPosition(1)])
            npPoint=npPoint[:,:2]
            self.npCtrlPoints.append(npPoint)

    def dictShape2Points(self,shape):
        """convert a dictionary shape to a list of points and tangents"""
        # e.g. {'center': '{ 362, 240, 1 }', 'leftTangent': '{ 362, 240, 1 }', 'rightTangent': '{ 362, 240, 1 }'}=point2D([362.0, 240.0], [362.0, 240.0], [362.0, 240.0])
        #points is numpy array of zeros with shape (nPoints,nCoords)
        points=np.zeros((len(shape),6))
        idx=0
        for dictPoint in shape:
            #print(dictPoint)

            dictPoint['center']=dictPoint['center'].replace('{','').replace('}','').split(',')
            dictPoint['center']=[float(x) for x in dictPoint['center']]
            points[idx,0:2]=dictPoint['center'][:2]
            dictPoint['leftTangent']=dictPoint['leftTangent'].replace('{','').replace('}','').split(',')
            dictPoint['leftTangent']=[float(x) for x in dictPoint['leftTangent']]
            points[idx,2:4]=dictPoint['leftTangent'][:2]
            dictPoint['rightTangent']=dictPoint['rightTangent'].replace('{','').replace('}','').split(',')
            dictPoint['rightTangent']=[float(x) for x in dictPoint['rightTangent']]
            points[idx,4:6]=dictPoint['rightTangent'][:2]
            #points.append(point2D(dictPoint['center'],dictPoint['leftTangent'],dictPoint['rightTangent']))
            idx+=1
        return points


class shapeFromRotopaint(nShapeMaster):
    """create a shape from a rotopaint node"""
    def __init__(self, rotoNode):
        super().__init__()
        self.rotoNode = rotoNode
        self.name=f"{self.rotoNode.name()}_shape"
        self.createShape()

    def loopKeyFrames(self, start_frame, end_frame, num_loops):

        # Get the shapes from the roto node
        self.getCtrlPoints()

        # Calculate the original frame range length
        frame_range_length = end_frame - start_frame
        datagen = Datagen(self, range=[start_frame, end_frame])
        pointsDict = datagen.pointsDict
        expanded_points_dict = {}
        for i in range(start_frame, start_frame + frame_range_length * num_loops):
            # Shift frame range so it starts from 0
            shifted_frame = i - start_frame
            current_frame = shifted_frame % frame_range_length
            if current_frame == 0:
                current_frame = frame_range_length
            # Shift back after calculation but make sure it cycles within the start and end frame
            current_frame = start_frame + (current_frame - 1) % frame_range_length
            expanded_points_dict[i] = pointsDict[current_frame]

        # Starting at frame_range_length+1, go through the expanded dict and set keyframes
        for i in range(start_frame + frame_range_length, start_frame + frame_range_length * num_loops):
            frameDict = expanded_points_dict[i]
            frameDict2 = deepcopy(frameDict)
            keyframes = self.dictShape2Points(frameDict2)
            #print(i)
            for point in range(len(self.shape)):
                self.ctrlPoints[point].center.addPositionKey(i, keyframes[point][:2])
                self.ctrlPoints[point].leftTangent.addPositionKey(i, keyframes[point][2:4])
        self.rotoNode['curves'].rootLayer.append(self.shape)

    import numpy as np

    def convert_to_bezier_format(self,point_array):
        bezier_curves = []
        n = len(point_array)
        for i in range(n):
            start_point = point_array[i][1]  # center of current curve
            control_point1 = start_point + point_array[i][2]  # right tangent
            control_point2 = point_array[(i + 1) % n][1] + point_array[(i + 1) % n][0]  # left tangent of next curve
            end_point = point_array[(i + 1) % n][1]  # center of next curve
            bezier_curves.append(np.vstack((start_point, control_point1, control_point2, end_point)))
        return bezier_curves

    def reduceDegree(self):
        """Convert from cubic bezier to quadratic bezier"""
        bezier_curves=self.convert_to_bezier_format(self.npCtrlPoints)
        reduced_curves = []
        for curve in bezier_curves:

            b_curve=bezier.Curve(curve.T, degree=3)
            deg2Curve = b_curve.reduce_()
            reduced_curves.append(deg2Curve)

        return reduced_curves

    def createtemplate(self):
        """create a template from the rotopaint node"""
        curves=self.reduceDegree()
        curves=self.subdivideCurves(curves)
        control_points = []
        for curve in curves:
            for point in curve.nodes.T:# Ignore last point
                control_points.append(point[0])  # X coordinate
                control_points.append(point[1])  # Y coordinate
        #normalize control points
        control_points=np.array(control_points)
        control_points-=control_points.min()
        control_points/=control_points.max()
        return list(control_points)

    def subdivideCurves(self, curves, nCurves=15):
        while len(curves) < nCurves:
            # subdivide the curves till we have 15
            # get longest curve
            max_length = 0
            max_index = 0
            for i, curve in enumerate(curves):

                length = curve.length
                if length > max_length:
                    max_length = length
                    max_index = i
            longest_curve = curves[max_index]
            left, right = longest_curve.subdivide()

            # remove the longest curve
            curves.pop(max_index)

            # insert the subdivided curves at the position of the removed curve
            curves.insert(max_index, left)
            curves.insert(max_index + 1, right)
        return curves

    def find_first_shape(self,layer):
        for item in layer:
            if isinstance(item, nuke.rotopaint.Shape):
                return item
            elif isinstance(item, nuke.rotopaint.Layer):
                shape = self.find_first_shape(item)
                if shape:
                    return shape
        return None

    def createShape(self):
        """create a shape from a rotopaint node"""
        root_layer = self.rotoNode['curves'].rootLayer
        self.shape = self.find_first_shape(root_layer)
        if self.shape:
            self.nPoints = len(self.shape)
            self.getListofCtrlPoints()
        else:
            raise ValueError("No shape found in the rotopaint node")





class dpsLoader(nShapeMaster):
    def __init__(self, outputDir, gtLabels=None):
        super().__init__()
        self.outputDir = outputDir
        self.ptFiles = glob.glob(f'{outputDir}/*.pt')
        self.gtLabels = gtLabels
        self.pts = None
        self.loadPts()
        self.createShape()
        # Add other initialization code here

    def loadPts(self):
        """Load .pt files into a list of NumPy arrays"""
        pts = []
        for pt_file in self.ptFiles:
            pt = torch.load(pt_file)
            # viz.plotQuadraticSpline(pt.unsqueeze(0), title=pt_file)
            # Convert points from tensors to quadratic bezier curves
            curves = self.makeCubicSpline(pt)
            #subtract yaxis from 1
            # viz.plotCubicSpline(torch.tensor(curves), title="cubic")
            curves[:, :, 1] = 1 - curves[:, :, 1]
            curves*=224
            # viz.plotCubicSpline(torch.tensor(curves), title="cubic expanded")


            pts.append(curves)

        self.pts=pts

    def makeCubicSpline(self, points: np.ndarray) -> np.ndarray:
        """Convert points to quadratic bezier curves using the bezier curve library"""
        curves = np.zeros((points.shape[0], 4, 2))
        for i in range(points.shape[0]):
            curve = bezier.Curve.from_nodes(points[i].T,2)
            curves[i] = curve.elevate().nodes.T

        return curves

    def createShape(self):
        """Create a shape from a rotopaint node"""
        self.rotoNode = nuke.createNode('Roto')
        self.shape = rp.Shape(self.rotoNode['curves'])

        num_curves = len(self.pts[0])
        spline = []
        #set control points on the first frame.
        for i in range(num_curves):
            self.shape.append(rp.ShapeControlPoint())

        for  frame in range(0,len(self.pts)):
           for i in range(num_curves):
               curve = self.pts[frame][i]
               next_curve = self.pts[frame][(i + 1) % num_curves]  # Wrap around to the first curve
               # Create a new spline in the shape
               spline = []
               # Set the points of the spline
               points = curve.reshape(-1, 2)  # Reshape the curve array to have (nPoints, 2) shape
               # Extract the control point values
               p0 = points[0]  # First control point
               p1 = points[1] - p0  # Second control point (right tangent of the first control point)
               # Set the end control point as the first control point of the next curve
               p3 = next_curve[0]
               p2 = points[2] - p3  # Third control point (left tangent of the end control point)
               self.shape[i].center.addPositionKey(frame, (p0[0], p0[1]))
               self.shape[i].rightTangent.addPositionKey(frame, (p1[0], p1[1]))
               self.shape[i].featherRightTangent.addPositionKey(frame, (p1[0], p1[1]))
               self.shape[(i+1) % num_curves].leftTangent.addPositionKey(frame, (p2[0], p2[1]))
               self.shape[(i+1) % num_curves].featherLeftTangent.addPositionKey(frame, (p2[0], p2[1]))
               self.shape[(i+1) % num_curves].center.addPositionKey(frame, (p3[0], p3[1]))










class shapeLoader(nShapeMaster):
    """load shapes from pickle file into a roto node"""
    def __init__(self, outputDir,gtLabels=None):
        super().__init__()
        self.outputDir = outputDir
        self.pickeFiles=self.getFiles()
        self.pickleFile = None
        self.npShape = None
        self.imagePath = None
        self.filesDict= None
        self.gtLabels = gtLabels
        #self.loadShape()
        #self.rotoNode = nuke.createNode('Roto')
        self.getFiles()
        self.nEpochs=len(self.filesDict[0])
        self.allLoss=np.zeros([self.nEpochs,3])
        if gtLabels is not None:
            self.loadGTJson()
        self.createShapes()
        self.allLoss/=len(self.filesDict)
        self.printLoss()

    def printLoss(self):
        """print epoch Num then loss"""
        for i in range(self.nEpochs):
            print("Epoch: {} totalLoss: {}  shapeLoss: {}  tangentLoss: {}".format(i,self.allLoss[i,0],self.allLoss[i,1],self.allLoss[i,2]))


        # this has been copied and pasted -refactor when time

    def getFiles(self):
        """get all the pickle files in the output directory as a list of lists """
        # search recursively for all pickle files in the output directory
        files = []
        for root, dirnames, filenames in os.walk(self.outputDir):
            for filename in fnmatch.filter(filenames, '*.pkl'):
                files.append([root, filename])

        filesDict = {}
        for file in files:
            key = int(file[1].split('_')[1].split('.')[0])
            if key in filesDict:
                filesDict[key].append(os.path.join(file[0], file[1]))
            else:
                filesDict[key] = [os.path.join(file[0], file[1])]

        self.filesDict=filesDict

    def loadGTJson(self):
        """load the ground truth labels from a json file"""
        with open(self.gtLabels) as f:
            self.gtLabels = json.load(f)

    def calcLoss(self):
        """calculate the loss between the ground truth labels and the current shape"""
        #convert self.GTshape to a numpy array
        norm=self.pointMatchLoss(self.gtPoints,self.npShape,6)
        pointNorm=self.pointMatchLoss(self.gtPoints[:,:2],self.npShape[:,:2],2)
        tanNorm=self.pointMatchLoss(self.gtPoints[:,2:],self.npShape[:,2:],4)
        loss=np.array([norm,pointNorm,tanNorm])

        return loss


    def createShapes(self):
        shapesList=self.filesDict.keys()
        for shape in shapesList:
            #create a new roto node named after the shape number
            for n in nuke.selectedNodes():
                n.setSelected(False)
            self.rotoNode = nuke.createNode('Roto', inpanel=False)
            self.rotoNode['name'].setValue('shape'+str(shape))
            #sort the value in filesDict[shape] by the first number in the file name before the underscore
            #e.g. 2_0.pkl = 2 and 3_0.pkl = 3
            self.filesDict[shape].sort(key=lambda x: int(x.split('_')[-2].split('\\')[-1]))
            self.loadShape(self.filesDict[shape][0])
            #add a read node with the shapes image path
            self.loadImage()
            # set the input of the rotonode to the read node
            #print("connecting read node {} to roto node {}".format(self.readNode.name(),self.rotoNode.name()))
            self.rotoNode.setInput(0, self.readNode)
            #create the shape
            #set current time in nuke to frame 1
            frame=1
            nuke.frame(frame)
            self.createShape()
            #interate through the rest of the shapes in the list
            idx=0
            for shape in self.filesDict[shape][1:]:
                nuke.frame(frame)
                self.loadShape(shape)
                self.allLoss[idx]+=self.calcLoss()
                self.addPointsToShape()
                frame+=1
                idx+=1

    def addPointsToShape(self):
        """add point keyframes to the existing shape"""
        self.npShape2points()
        for i in range(len(self.points)):
            #add point at current time
            point=self.shape[i]
            point.center.addPositionKey(nuke.frame(),self.points[i][0])





    def loadShape(self,path):
        with open(path, 'rb') as f:
            name, shape = pickle.load(f)
        self.npShape = shape
        self.imagePath = name
    def npShape2points(self):
        """convert a numpy shape to a list of points"""
        self.points=[]
        for point in self.npShape:
            if len(point) == 2:
                self.points.append(point)
            elif len(point) == 6:
                #split into 3 points
                ctrlPoint = [list(point[:2])]
                ctrlPoint.append(list(point[2:4]))
                ctrlPoint.append(list(point[4:6]))
                self.points.append(ctrlPoint)



    def addGTShape(self):
        #get the shape number from the image name e.g 'D:/pyG/data/points/\\spoints.6871.png' = 6871
        shapeNum = int(self.imagePath.split('.')[-2].split('\\')[-1])
        #get the shape from the gtLabels dictionary
        self.gtShape = self.gtLabels[str(shapeNum)]
        #convert the shape to a list of points
        self.gtPoints = self.dictShape2Points(self.gtShape)
        #add shape to roto node
        self.GTshape=rp.Shape(self.curveKnob)
        for point in self.gtPoints:
            self.addPoint(self.GTshape, point[0], point[1], point[2], point[3], point[4], point[5])
        att=self.GTshape.getAttributes()
        att.set(nuke.frame(), 'ro', 0.0)
        att.set(nuke.frame(), 'go', 1.0)
        att.set(nuke.frame(), 'bo', 0.0)


    def createShape(self):
        self.curveKnob = self.rotoNode['curves']
        # if gt shape exists add it to the roto node
        if self.gtLabels is not None:
            self.addGTShape()
        self.shape = rp.Shape(self.curveKnob)
        self.npShape2points()
        self.addPoints(self.shape)

    def pointMatchLoss(self,shape1, shape2, nCoords=6):
        # shape1 and shape2 are tensors of shape (K,2)
        # shape1 = shape1.cpu().detach().numpy()
        # shape2 = shape2.cpu().detach().numpy()
        K = shape1.shapes[0]
        assert K == shape2.shapes[0]
        sumNorm = 0
        #rewrite the below but with numpy instead of tensors
        Norms = np.zeros((K, nCoords))
        for jInd in range(K):
            L1Norm = 0
            for iInd in range(K):
                p2i = (iInd + jInd) % K
                # print(f"j={jInd}, i={iInd}, p2Index={p2i}")
                L1Norm += abs((shape1[iInd] - shape2[p2i]))
                # print(f"L1Norm = {L1Norm}")
            Norms[jInd] = L1Norm
        #print(f"Norms = {Norms}")
        # sum the x and y components of the norms then grab the minimum
        sumNorm = np.sum(Norms, axis=1)
        # print(f"sumNorm = {sumNorm}")
        minNorm = np.min(sumNorm)
        #print(f"minNorm = {minNorm}")
        return minNorm



    def loadImage(self):
        """convert image path from processed path to raw path then load
        image into new read node"""
        self.convertPath()
        self.loadPathtoReadNode()

    def convertPath(self):
        """convert processed path to raw path e.g  "D:\pyG\data\points\processed\spoints.0001.pt"
        to "D:\pyG\data\points\spoints.0001.png """
        path = self.imagePath[0]
        path = path.replace('processed', '')
        path = path.replace('.pt', '.png')
        self.imagePath = path

    def loadPathtoReadNode(self):
        """ create read node and load image into it"""
        self.readNode = nuke.createNode('Read')
        self.readNode['file'].setValue(self.imagePath)
        self.readNode['first'].setValue(1)
        self.readNode['last'].setValue(1)
        self.readNode['origfirst'].setValue(1)
        self.readNode['origlast'].setValue(1)

    def splitCurveNuke(self,curve, t):
        """given the control points of a bezier curve in list format [p0,p1,p2,p3]split it at t, retyrn the control points at the split"""
        # extract the first 2 values from 'center' and 'rightTangent' and convert to list
        p0centre = curve[0]
        p0left = curve[1]
        p0centre = np.array(p0centre).astype(np.float32)
        p0left = np.array(p0left).astype(np.float32)
        p0left = p0centre - p0left
        # now get hafway points betwen centre and tangent control points
        p0a = p0centre + (p0left - p0centre) * t
        # now do the same for p1 but use the 'leftTangent' and 'center' values

        p1Right = curve[2]
        p1Centre = curve[3]
        p1Centre = np.array(p1Centre).astype(np.float32)
        p1Right = np.array(p1Right).astype(np.float32)
        p1Right = p1Centre + p1Right
        # now lerp points betwen centre and tangent control points
        p1a = p1Centre + (p1Right - p1Centre) * t
        # now get halfway point between p0left and p1Right
        p01a = p0left + (p1Right - p0left) * t
        # now get halfway point between p0a and p01a
        nLftCtrl = p0a + (p01a - p0a) * t
        # now get halfway point between p01a and p1a
        nRgtCtrl = p01a + (p1a - p01a) * t
        # now get halfway point between nLftCtrl and nRgtCtrl
        nCentre = nLftCtrl + (nRgtCtrl - nLftCtrl) * t
        # subtract the centre poitt from tangent cotrol points
        # points = np.array([p0centre, p0left, p1Right, p1Centre])
        # newPoints = np.zeros((3, 2))
        # for i in range(3):
        #     x = (1-t) * points[i][0] + t * points[i+1][0]
        #     y = (1-t) * points[i][1] + t * points[i+1][1]
        #     newPoints[i] = np.array([x,y])
        # nLftCtrl = newPoints[0]
        # nRgtCtrl = newPoints[2]
        # nCentre = newPoints[1]
        nLftCtrl = nLftCtrl - nCentre
        nRgtCtrl = nRgtCtrl - nCentre
        newPoints = np.array([nLftCtrl, nCentre, nRgtCtrl])
        return newPoints

    def getPointsAtEdge(self, edgeIndex,frame):
        """get the points at the edge of a shape"""
        # get the correct points e.g in a 5 point shape edge 1 is (0,1), edge 5 is (4,0)
        point1=(edgeIndex-1)%self.nPoints
        point2=edgeIndex%self.nPoints
        point1centre=self.shape[point1].center.getPosition(frame)
        point2centre=self.shape[point2].center.getPosition(frame)
        point1left=self.shape[point1].leftTangent.getPosition(frame)
        point2right=self.shape[point2].rightTangent.getPosition(frame)

        points=np.array([point1centre,point1left,point2right,point2centre])
        return points

    def insertPointAtEdge(self,frame,edgeIndex, t):
        points=self.getPointsAtEdge(edgeIndex,frame)[:,:2]
        newPoints=self.splitCurveNuke(points, t)
        self.ctrlPoints.insert(edgeIndex,newPoints)


class nShapeCreator(nShapeMaster):
    """n point shape class for creating and working with n sided shapes in nuke"""
    def __init__(self, nPoints, rotoNode):
        super().__init__()
        self.rotoNode = rotoNode
        self.nPoints = nPoints
        self.createShape()
        self.getListofCtrlPoints()

    def rebuildShape(self):
        """rebuild the shape from the control points"""
        #get rotonode curve knob and add new shape
        curveKnob = self.rotoNode['curves']
        self.shape = rp.Shape(curveKnob)

        for point in self.npCtrlPoints:
            self.shape.append(point[1])
            self.shape[-1].leftTangent.addPositionKey(1,point[0])
            self.shape[-1].rightTangent.addPositionKey(1,point[2])
            self.shape[-1].featherLeftTangent.addPositionKey(1,point[0])
            self.shape[-1].featherRightTangent.addPositionKey(1,point[2])


    def createShape(self):
        curveKnob = self.rotoNode['curves']
        self.shape = rp.Shape(curveKnob)
        self.calculatePoints()
        self.addPoints(self.shape)

    def calculatePoints(self,radius=50):
        """calculate n evenly spaced points on a circle"""
        windowCentre = self.windowSize[0] / 2, self.windowSize[1] / 2
        for i in range(self.nPoints):
            x = math.cos(2 * math.pi * i / self.nPoints) * radius
            y = math.sin(2 * math.pi * i / self.nPoints) * radius
            x += windowCentre[0]
            y += windowCentre[1]
            self.points.append([x, y])

    def deCasteljau(self,points, t):
        """de casteljau algorithm for bezier curves"""
        #first add point 0 to point 1 and point 3 to point 2

        points=np.array(points)
        points[1]=points[0]+points[1]
        points[2]=points[3]+points[2]

        deg=3

        B=np.zeros((4,4,2))
        for i in range(len(points)):
            B[0][i][0]=points[i][0]
            B[0][i][1]=points[i][1]


        for k in range(1,deg+1):
            for i in range(k,deg+1):
                B[k][i]=(1-t)*B[k-1][i-1]+t*B[k-1][i]
        ctrlPoints=np.array((B[1][1],B[2][2],B[3][3],B[2][3],B[1][3]))
        #subtract the centre point from the tangent control points
        print(f"ctrlPoints left: {ctrlPoints[0]} centre: {ctrlPoints[1]} right: {ctrlPoints[2]}")
        ctrlPoints[0]=ctrlPoints[0]-B[0][0]
        ctrlPoints[1]=ctrlPoints[1]-ctrlPoints[2]
        ctrlPoints[3]=ctrlPoints[3]-ctrlPoints[2]
        ctrlPoints[4]=ctrlPoints[4]-B[0][3]
        return ctrlPoints





    def splitCurveNuke(self,curve, t):
        """given the control points of a bezier curve in list format [p0,p1,p2,p3]split it at t, retyrn the control points at the split"""
        # extract the first 2 values from 'center' and 'rightTangent' and convert to list
        p0centre = curve[0]
        p0left = curve[1]
        p0centre = np.array(p0centre).astype(np.float32)
        p0left = np.array(p0left).astype(np.float32)
        p0left = p0centre - p0left
        # now get hafway points betwen centre and tangent control points
        p0a = p0centre + (p0left - p0centre) * t
        # now do the same for p1 but use the 'leftTangent' and 'center' values

        p1Right = curve[2]
        p1Centre = curve[3]
        p1Centre = np.array(p1Centre).astype(np.float32)
        p1Right = np.array(p1Right).astype(np.float32)
        p1Right = p1Centre + p1Right
        # now lerp points betwen centre and tangent control points
        p1a = p1Centre + (p1Right - p1Centre) * t
        # now get halfway point between p0left and p1Right
        p01a = p0left + (p1Right - p0left) * t
        # now get halfway point between p0a and p01a
        nLftCtrl = p0a + (p01a - p0a) * t
        # now get halfway point between p01a and p1a
        nRgtCtrl = p01a + (p1a - p01a) * t
        # now get halfway point between nLftCtrl and nRgtCtrl
        nCentre = nLftCtrl + (nRgtCtrl - nLftCtrl) * t
        nLftCtrl = nLftCtrl - nCentre
        nRgtCtrl = nRgtCtrl - nCentre
        newPoints = np.array([nLftCtrl, nCentre, nRgtCtrl])
        return newPoints

    def getPointsAtEdge(self, edgeIndex,frame):
        """get the points at the edge of a shape"""
        # get the correct points e.g in a 5 point shape edge 1 is (0,1), edge 5 is (4,0)
        point1=(edgeIndex-1)%self.nPoints
        point2=edgeIndex%self.nPoints
        point1centre=self.shape[point1].center.getPosition(frame)
        point2centre=self.shape[point2].center.getPosition(frame)
        point1right=self.shape[point1].rightTangent.getPosition(frame)
        point2left=self.shape[point2].leftTangent.getPosition(frame)

        points=np.array([point1centre,point1right,point2left,point2centre])
        return points

    def insertPointAtEdge(self,frame,edgeIndex, t):
        points=self.getPointsAtEdge(edgeIndex,frame)[:,:2]
        newPoints=self.deCasteljau(points, t)
        self.npCtrlPoints.insert(edgeIndex,newPoints[1:4])
        self.npCtrlPoints[edgeIndex - 1][2] = newPoints[0]
        self.npCtrlPoints[edgeIndex + 1][0] = newPoints[4]
        self.rebuildShape()


class Datagen:
    """create image sequences and spline data from an nShape object"""
    def __init__(self, shapes=None,switch_frames=[] ,lastNode=None,switch=None, timeID="unknown", range=[1, 200]):
        self.shapes=shapes
        self.timeID=timeID
        self.root = "D:/pyG/data/points/"
        self.frameRange= range
        self.switch_frames=switch_frames
        self.switch=switch
        self.pointsDict={}
        self.createPointsDict()
        self.lastNode=lastNode
        #self.attachWriteNode()

    # def createPointsDict(self):
    #     """create a dictionary of points for each frame"""
    #     for frame in range(self.frameRange[0],self.frameRange[1]+1):
    #         pointDictList=[]
    #         for point in self.shapes.shapes:
    #             pointAtTime=self.shapes.getPointDict(point, frame)
    #             pointDictList.append(pointAtTime)
    #         self.pointsDict[frame]=pointDictList

    def createPointsDict(self):
        """create a dictionary of points for each frame"""
        shape_index = 0
        for frame in range(self.frameRange[0], self.frameRange[1] + 1):

            # Switch shapes at the specified frames
            # Check if self.switch_frames is a list before trying to index it
            if isinstance(self.switch_frames, list) and shape_index < len(self.switch_frames) and frame == \
                    self.switch_frames[shape_index]:
                shape_index += 1
                print(f"switching to shape {shape_index} at frame {frame}")

                # Get the current shape
                shape = self.shapes[shape_index]

                # every 1000 frames print the shapes name
                if frame % 1000 == 0:
                    print(f"frame: {frame} shape: {shape.name}")
                print(f"setting keyframe for switch at frame {frame} , value {shape_index}")
                self.switch["which"].setValueAt(shape_index, frame)
            else:
                if isinstance(self.shapes, list):
                    shape = self.shapes[shape_index]
                else:
                    shape = self.shapes
            # Generate the points for the current frame
            pointDictList = []
            for point in shape.shape:
                pointAtTime = shape.getPointDict(point, frame)
                pointDictList.append(pointAtTime)
            self.pointsDict[frame] = pointDictList

        print("finished creating points dict")

    def printPointsDict(self):
        """print the points dictionary"""
        for frame in self.pointsDict:
            print("frame: ",frame)
            for point in self.pointsDict[frame]:
                print(point)
            print("")

    def savePointsDict(self,name):
        """save the points dictionary"""
        jsonobj=json.dumps(self.pointsDict)
        with open("{}points{}.json".format(self.root,name),"w") as f:
            f.write(jsonobj)




    def attachWriteNode(self):
        """attach a write node to the roto node"""
        writeNode=nuke.createNode("Write")
        writeNode["file_type"].setValue("png")

        writeNode["file"].setValue("{}{}/spoints.####.png".format(self.root,self.timeID))
        writeNode["create_directories"].setValue(1)
        writeNode["colorspace"].setValue("linear")
        writeNode["channels"].setValue("rgb")
        writeNode["create_directories"].setValue(1)
        writeNode["datatype"].setValue("8 bit")

        #set first and last frame
        writeNode["first"].setValue(self.frameRange[0])
        writeNode["last"].setValue(self.frameRange[1])
        if self.lastNode is not None:
            writeNode.setInput(0,self.lastNode)
        else:
            writeNode.setInput(0, self.shapes.rotoNode)
        self.outWriteNode=writeNode

    def render(self):
        """render the write node"""
        nuke.render(self.outWriteNode, start=self.frameRange[0], end=self.frameRange[1])


class CornerPinRoto(shapeFromRotopaint):
    """the 4 points animation from a cornerpin node and apply it to a roto node"""
    def __init__(self, cornerPinNode, rotoNode=None, frameRange=[1,200]):
        super().__init__(rotoNode)
        self.cornerPinNode=cornerPinNode
        self.rotoNode=rotoNode
        self.frameRange=frameRange


    def getCornerPin(self,frame):
            to4 = [self.cornerPinNode['to4'].getValueAt(frame,0),self.cornerPinNode['to4'].getValueAt(frame,1)]
            to3 = [self.cornerPinNode['to3'].getValueAt(frame,0),self.cornerPinNode['to3'].getValueAt(frame,1)]
            to1 = [self.cornerPinNode['to1'].getValueAt(frame,0),self.cornerPinNode['to1'].getValueAt(frame,1)]
            to2 = [self.cornerPinNode['to2'].getValueAt(frame,0),self.cornerPinNode['to2'].getValueAt(frame,1)]
            print(f"to1: {to1} to2: {to2} to3: {to3} to4: {to4}")
            return [to1,to2,to3,to4]


    def getCornerPinFrom(self,frame):
            from4 = [self.cornerPinNode['from4'].getValueAt(frame,0),self.cornerPinNode['from4'].getValueAt(frame,1)]
            from3 = [self.cornerPinNode['from3'].getValueAt(frame,0),self.cornerPinNode['from3'].getValueAt(frame,1)]
            from1 = [self.cornerPinNode['from1'].getValueAt(frame,0),self.cornerPinNode['from1'].getValueAt(frame,1)]
            from2 = [self.cornerPinNode['from2'].getValueAt(frame,0),self.cornerPinNode['from2'].getValueAt(frame,1)]
            print(f"from1: {from1} from2: {from2} from3: {from3} from4: {from4}")
            return [from1,from2,from3,from4]

    def getCornerPinFrame(self):
        """get the cornerpin points for each frame"""
        cornerPinFrameList=[]
        for frame in range(self.frameRange[0],self.frameRange[1]+1):
            cornerPinFrameList.append(self.getCornerPin(frame))
        return cornerPinFrameList

    def getTransformMatrix(self,frame):
        """get the transform matrix from the cornerpin node"""
        matrix=cv2.getPerspectiveTransform(np.float32(self.getCornerPinFrom(frame)),np.float32(self.getCornerPin(frame)))
        return matrix

    def getAffineTransformMatrix(self,frame):
        """get the transform matrix from the cornerpin node"""
        matrix=cv2.getAffineTransform(np.float32(self.getCornerPinFrom(frame)[:3]),np.float32(self.getCornerPin(frame)[:3]))
        return matrix
    def transformPoint(self,point,matrix):
        """transform a point with a matrix"""
        point=np.array([point[0],point[1],1])
        point=np.dot(matrix,point)
        return [point[0],point[1]]

    def transformRoto(self):
        self.getCtrlPoints()
        pointOriginList = []
        for point in self.ctrlPoints:
            pointOriginList.append(point.center.getPosition(self.frameRange[0]))
        for frame in range(self.frameRange[0],self.frameRange[1]):
            matrix=self.getAffineTransformMatrix(frame)
            idx=0
            for point in self.ctrlPoints:
                point.center.addPositionKey(frame,self.transformPoint(pointOriginList[idx],matrix))
                idx+=1
        self.rotoNode["curves"].rootLayer.append(self.shape)




class RandCornerPinRoto(shapeFromRotopaint):
    """the 4 points animation from a cornerpin node and apply it to a roto node"""
    def __init__(self, cornerPinNode, rotoNode, frameRange=None):
        super().__init__(rotoNode)
        self.cornerPinNode=cornerPinNode
        self.rotoNode=rotoNode
        self.frameRange=frameRange
        self.applyRandomTransform()

    def random_affine_transform(self, frame):
        # Set the seed for the random number generator
        np.random.seed(frame)
        # Generate a random translation (tx, ty) between -10 and 10
        tx, ty = np.random.uniform(-50, 50, 2)

        # Generate a random rotation in radians
        theta =np.random.uniform(-np.pi / 90, np.pi / 90)

        # Generate a random scale factor
        scale = np.random.uniform(1, 1.1)

        # Generate a random shear factor
        shear = np.random.uniform(-1,1)

        # Create the 2D affine transformation matrix
        transform_matrix = np.array([
            [scale * np.cos(theta) - shear * np.sin(theta), -np.sin(theta), tx],
            [np.sin(theta), scale * np.cos(theta) + shear * np.sin(theta), ty],
            [0, 0, 1]
        ])

        inverse_transform_matrix = transform_matrix.T

        return transform_matrix, inverse_transform_matrix

    def convert_to_nuke_matrix(self,matrix_3x3):
        # Start with a 4x4 identity matrix
        matrix_4x4 = np.identity(4)

        # Copy over the 3x3 transform to the top-left of the 4x4 matrix
        matrix_4x4[:3, :3] = matrix_3x3[:3, :3]

        # Transpose to switch from row-major to column-major
        matrix_4x4 = matrix_4x4.T

        # Flatten to 1D array
        matrix_4x4 = matrix_4x4.flatten()

        return matrix_4x4

    def transformPoint(self,point,matrix):
        """transform a point with a matrix"""
        point=np.array([point[0],point[1],1])
        point=np.dot(matrix,point)
        return [point[0],point[1]]

    def applyRandomTransform(self):
        # Capture all point positions at each frame before transforming them
        self.getCtrlPoints()
        point_positions = {}
        # make sure cornerPin node extra matrix is animated
        #clear all keyframes in corner pin
        # for curve in self.cornerPinNode["transform_matrix"].animations():
        #     curve.clear()
        print(f"clearing animation and setting keyframes on node: {self.cornerPinNode.name()}")
        self.cornerPinNode["transform_matrix"].clearAnimated()
        #self.cornerPinNode["transform_matrix"].setDefaultValue()
        self.cornerPinNode["transform_matrix"].setAnimated()
        #set keyframes at existing value before first frame
        #set all values to identity matrix at first frame
        for i in range(16):
            self.cornerPinNode["transform_matrix"].setValueAt(1 if i%5==0 else 0, self.frameRange[0]-1, i)

        for frame in range(self.frameRange[0], self.frameRange[1]):
            point_positions[frame] = [point.center.getPosition(frame) for point in self.ctrlPoints]

        # Now apply transformations
        for frame in range(self.frameRange[0],self.frameRange[1]):
            matrix, invMatrix=self.random_affine_transform(frame)
            nukeMatrix=self.convert_to_nuke_matrix(matrix)

            for i in range(16):
                self.cornerPinNode["transform_matrix"].setValueAt(nukeMatrix[i], frame, i)
            self.transformRoto(invMatrix,frame,point_positions[frame])
        #set all values to identity matrix at last frame +1
        for i in range(16):
            self.cornerPinNode["transform_matrix"].setValueAt(1 if i%5==0 else 0, self.frameRange[1]+1, i)
        self.rotoNode["curves"].rootLayer.append(self.shape)



    def transformRoto(self, matrix, frame, pointOriginList):
        #print(f"frame {frame} transform roto")
        for idx, point in enumerate(self.ctrlPoints):
            transformed_point = self.transformPoint(pointOriginList[idx], matrix)
            #print(f"point {idx} {transformed_point}")
            point.center.addPositionKey(frame, transformed_point)
            #print(f"point {idx} {point.center.getPosition(frame)}")









def randOffsetShapeImage(rotoName, frameRange, transName):

    roto = nuke.toNode(rotoName)
    trans = nuke.toNode(transName)
    trans["translate"].setAnimated()
    trans["scale"].setAnimated()
    #set to default values
    trans["translate"].setValueAt(0, 1)
    trans["scale"].setValueAt(0, 1)
    #make trans["translate"] is animated


    shape = shapeFromRotopaint(roto)
    total_offset_x = 0
    total_offset_y = 0
    scale = 0
    print("generating shape 2 keyframes")
    for frame in range(frameRange[0],frameRange[1]):
        shape.growPoints(frame, -scale)
        shape.growPointsTangent(frame, -scale)
        # Generate random x and y offsets
        xmax = 15
        x = random.randint(-xmax, xmax)
        ymax = 5
        y = random.randint(-ymax, ymax)
        scaleMax = 0.1
        scale = random.uniform(0, scaleMax)

        # Check if the new x and y offsets will breach the limits
        if total_offset_x + x > 50:
            x = random.randint(-xmax, 0)
        elif total_offset_x + x < -50:
            x = random.randint(0, xmax)

        if total_offset_y + y > 30:
            y = random.randint(-ymax, 0)
        elif total_offset_y + y < -30:
            y = random.randint(0, ymax)

        # Update total offsets
        total_offset_x += x
        total_offset_y += y

        shape.translatePoints(frame, x, y)
        shape.growPoints(frame, scale)
        shape.growPointsTangent(frame, scale)

        transx = trans["translate"].valueAt(frame, 0)
        transy = trans["translate"].valueAt(frame, 1)
        #scaleXY = trans["scale"].valueAt(frame, 0)

        trans["translate"].setValueAt(x + transx, frame, 0)
        trans["translate"].setValueAt(y + transy, frame, 1)
        trans["scale"].setValueAt(1 + scale, frame, 0)
    roto['curves'].rootLayer.append(shape.shape)


if __name__ == '__main__':

    # nkScript = "D:/DeepParametricShapes/nukeScripts/Cadis_example_v10.nk"
    # nuke.scriptOpen(nkScript)
    #grade=nuke.toNode("Grade1")
    #roto=nuke.toNode("Roto1")
    #blur = createBlurNode(roto)
    #grade = nuke.toNode("Grade1")
    #trans=nuke.toNode("Transform1")
    #shape=shapeFromRotopaint(roto)
    # total_offset_x = 0
    # total_offset_y = 0
    #
    # for frame in range(1, 10000):
    #     # Generate random x and y offsets
    #     x = random.randint(-20, 20)
    #     y = random.randint(-10, 10)
    #     scale=random.uniform(-0.005,0.005)
    #
    #     # Check if the new x and y offsets will breach the limits
    #     if total_offset_x + x > 150:
    #         x = random.randint(-20, 0)
    #     elif total_offset_x + x < -150:
    #         x = random.randint(0, 10)
    #
    #     if total_offset_y + y > 100:
    #         y = random.randint(-10, 0)
    #     elif total_offset_y + y < -100:
    #         y = random.randint(0, 10)
    #
    #     # Update total offsets
    #     total_offset_x += x
    #     total_offset_y += y
    #
    #     shape.translatePoints(frame, x, y)
    #     shape.growPoints(frame,scale)
    #     shape.growPointsTangent(frame, scale)
    #
    #     transx = trans["translate"].valueAt(frame, 0)
    #     transy = trans["translate"].valueAt(frame, 1)
    #     scaleXY= trans["scale"].valueAt(frame, 0)
    #
    #     trans["translate"].setValueAt(x + transx, frame, 0)
    #     trans["translate"].setValueAt(y + transy, frame, 1)
    #     trans["scale"].setValueAt(scaleXY+scale, frame, 0)
    #


    # # #shape.loopKeyFrames(1,172,10)
    # switchNode=nuke.toNode("Switch1")
    # dataGenenerator=Datagen(shapes=[shape,shape2],switch_frames=[5000],  lastNode=switchNode,switch=switchNode,timeID="transform_test",range=[1,10000])
    # # # dataGenenerator.frameRange=[0,172]
    # #dataGenenerator.createPointsDict()
    # # # dataGenenerator.printPointsDict()
    # dataGenenerator.savePointsDict("transform_test")
    # #dataGenenerator.attachWriteNode()
    # nuke.scriptSave("D:/DeepParametricShapes/nukeScripts/Cadis_example_01.nk")
    # print("script saved")
    # #dataGenenerator.render()

    # nuke.Undo().disable()
    #
    # rotoListI=["ri1", "ri2", "ri3", "ri4", "ri5", "ri6", "ri7", "ri8"]
    # rotoListP=["rp1", "rp2", "rp3", "rp4", "rp5", "rp6", "rp7", "rp8"]
    # transList=["tcp1", "tcp2", "tcp3", "tcp4", "tcp5", "tcp6", "tcp7", "tcp8"]
    # cpList=["cpp1","cpp2","cpp3","cpp4","cpp5","cpp6","cpp7","cpp8"]
    # frameRanges=[[1,172],[0, 773], [981, 1145], [3691, 3999], [6346, 6728], [6919, 7376], [3573, 4012], [4539, 4686], [4496, 4540]]
    # frameRangesOffset=[[0, 2000], [2000, 4000], [4000, 6000], [6000, 8000], [8000, 10000], [10000, 12000], [12000, 14000], [14000, 16000]]
    #rotoList=["ri4"]
    # transList=[transList[-1]]
    # frameRanges=[frameRanges[-1]]

    # frameRangesOffset=[frameRangesOffset[-1]]
    # rotoList = rotoList[6:]
    # transList = transList[6:]
    # frameRanges = frameRanges[1:]
    # cpList = cpList[6:]
    # frameRangesOffset = frameRangesOffset[6:]
    # # enumerate through zipped rotoList,cpList and frameRanges
    # for i,(roto,cp,frameRange) in enumerate(zip(rotoListI+rotoListP, cpList+cpList, frameRanges+frameRanges)):
    # #     #convertTracking Data to Roto.
    #       cp=nuke.toNode(cp)
    #       roto=nuke.toNode(roto)
    # #     #print(f"applying tracking data to {roto.name()}")
    # #     # cp2roto=CornerPinRoto(cp,roto,frameRange=frameRange)
    # #     # cp2roto.transformRoto()
    # #
    #       #loop keyframes
    #       print(f"looping keyframes {roto.name()}")
    #       shape=shapeFromRotopaint(roto)
    #       shape.loopKeyFrames(frameRange[0],frameRange[1],200)
    # #
    # for i, (roto, frameRange, trans) in enumerate(zip(rotoListI, frameRangesOffset, transList)):
    #     cp = nuke.toNode(trans)
    #     roto = nuke.toNode(roto)
    #     print(f"applying random offset to {roto.name()}")
    #     Randcp2roto = RandCornerPinRoto(cp, roto, frameRange=frameRange)
    #
    # for i, (roto, frameRange, trans) in enumerate(zip(rotoListP, frameRangesOffset, transList)):
    #     cp = nuke.toNode(trans)
    #     roto = nuke.toNode(roto)
    #     print(f"applying random offset to {roto.name()}")
    #     Randcp2roto = RandCornerPinRoto(cp, roto, frameRange=frameRange)


    # #convertTracking Data to Roto.
    # cp=nuke.toNode("CornerPin2D6")
    # roto=nuke.toNode("Roto10")
    # cp2roto=CornerPinRoto(cp,roto,frameRange=[981,1144])
    # cp2roto.transformRoto()
    #
    # #loop keyframes
    # print(f"looping keyframes roto10")
    # shape=shapeFromRotopaint(nuke.toNode("Roto10"))
    # shape.loopKeyFrames(981,1144,48)
    #convertTracking Data to Roto.
    # cp=nuke.toNode("CornerPin2D5")
    # roto=nuke.toNode("Roto11")
    # cp2roto=CornerPinRoto(cp,roto,frameRange=[3691,3998])
    # cp2roto.transformRoto()

    # #loop keyframes
    # print("looping keyframes roto11")
    # shape=shapeFromRotopaint(nuke.toNode("Roto11"))
    # shape.loopKeyFrames(3691,3998,24)

    #create shapes from roto list



    #
    # switchNode=nuke.toNode("Switch1")
    # # # # #clear animation from switch node which
    # switchNode["which"].clearAnimated()
    # switchNode["which"].setAnimated()
    # switchNode["which"].setValueAt(0,0)
    # #
    # shapes = [shapeFromRotopaint(nuke.toNode(roto)) for roto in rotoListI]
    # dataGenenerator=Datagen(shapes=shapes,switch_frames=[2000,4000,6000,8000,10000,12000,14000],  lastNode=switchNode,switch=switchNode,timeID="transform_test",range=[1,16000])
    # # # # dataGenenerator.frameRange=[0,172]
    # # #dataGenenerator.createPointsDict()
    # # # # dataGenenerator.printPointsDict()
    # dataGenenerator.savePointsDict("transform_test_instrument")
    # # # # #dataGenenerator.attachWriteNode()
    # nuke.scriptSave(nkScript)
    #dataGenenerator.render()

    nuke.scriptOpen(r"D:\ThesisData\nukeScripts\testOuts.nk")


    # nuke.scriptClose()
    # nuke.scriptNew(r"D:\ThesisData\nukeScripts\testOuts.nk")
    DpsResults=dpsLoader(r"D:\ThesisData\demo\instrument1")
    nuke.scriptSave(r"D:\ThesisData\nukeScripts\testOuts.nk")

    # #create new nuke script
    # nuke.scriptNew("D:/pyG/data/OutputLoader.nk")
    # shapeLoader=shapeLoader(r"D:\pyG\temp\RES\05-06_11-09-59\epoch_5\outputs",gtLabels=r"D:\pyG\data\points\120423_183451\points120423_183451.json")
    # #save nuke script
    # nuke.scriptSave("D:/pyG/data/OutputLoader.nk")



