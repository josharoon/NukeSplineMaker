import fnmatch
import json
import math
import os
import pickle
import random
from copy import copy, deepcopy

import cv2

#import cv2
print(os.environ['PATH'])
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
            print(dictPoint)

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
        shape = self.find_first_shape(self.rotoNode['curves'].rootLayer)
        self.getCtrlPoints()

        # Calculate the original frame range length
        frame_range_length = end_frame - start_frame + 1
        datagen= Datagen(self, start_frame, end_frame)
        pointsDict= datagen.pointsDict
        expanded_points_dict = {}
        for i in range(1, frame_range_length * num_loops + 1):
            current_frame = i % frame_range_length
            if current_frame == 0:
                current_frame = frame_range_length
            expanded_points_dict[i] = pointsDict[current_frame]

        # starting at frame_range_length+1. go through the exapnded dict and set keyframes
        for i in range(frame_range_length+1, frame_range_length * num_loops + 1):
            frameDict= expanded_points_dict[i]
            frameDict2=deepcopy(frameDict)
            keyframes= self.dictShape2Points(frameDict2)
            print(i)
            for point in range(len(self.shape)):
                self.ctrlPoints[point].center.addPositionKey(i,keyframes[point][:2])
                self.ctrlPoints[point].leftTangent.addPositionKey(i,keyframes[point][2:4])






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
        #print ("loaded shape {} from {}".format(name,path))
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
        for frame in range(self.frameRange[0],self.frameRange[1]+1):
            # Switch shapes at the specified frames
            if shape_index < len(self.switch_frames) and frame == self.switch_frames[shape_index]:
                shape_index += 1
                print(f"switching to shape {shape_index} at frame {frame}")


            # Get the current shape
            shape = self.shapes[shape_index]
            # every 1000 frames print the shapes name
            if frame % 1000 == 0:
                print(f"frame: {frame} shape: {shape.name}")
            self.switch["which"].setValueAt(shape_index,frame)


            # Generate the points for the current frame
            pointDictList=[]
            for point in shape.shape:
                pointAtTime=shape.getPointDict(point,frame)
                pointDictList.append(pointAtTime)
            self.pointsDict[frame]=pointDictList

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
        for frame in range(self.frameRange[0],self.frameRange[1]+1):
            matrix=self.getAffineTransformMatrix(frame)
            idx=0
            for point in self.ctrlPoints:
                point.center.addPositionKey(frame,self.transformPoint(pointOriginList[idx],matrix))
                idx+=1
        self.rotoNode["curves"].rootLayer.append(self.shape)






if __name__ == '__main__':
    #load nuke script
    #nuke.scriptOpen("D:/pyG/data/DatasetGen.nk")
    #nuke.scriptNew("D:/pyG/data/point_split_test.nk")
    # roto = createRotoNode()
    # shape=nShapeCreator(5, roto)
    # shape.printPoints(0)
    # shape.randomisePoints(100,15,15)
    # shape.growPointsTangent(100,10)
    # shape.printPoints(100)
    # shape.randomisePoints(150, 15, 15)
    # shape.randomisePointsTangent(25,5,5)
    # shape.randomisePointsTangent(200,5,4)
    # shape.growPoints(110,0.4)
    # shape.printPoints(200)
    # shape.growPoints(135, -0.3)
    # shape.translatePoints(200, 15,15)
    # shape.translatePoints(400, -30,-20)
    # shape.growPoints(600, 0.5)
    # shape.growPoints(700, -0.5)
    # shape.randomisePoints(900, 30, 30)
    # shape.randomisePoints(1800, 5, 5)
    # shape.growPointsTangent(2000, 2)
    # shape.growPoints(8000, 0.5)
    # shape.randomisePoints(3000, 15, 15)
    # shape.randomisePoints(4000, 15, 15)
    # shape.randomisePoints(5000, 15, 15)
    # shape.randomisePoints(6000, 15, 15)
    # shape.resetPointsTangent(6100)
    # shape.translatePoints(7000, -15, 15)
    # shape.randomisePoints(7100, 15, 15)
    # shape.randomisePoints(8000, 15, 15)
    # shape.randomisePoints(9000, 15, 15)
    # shape.randomisePoints(10000, 15, 15)





    #nuke.scriptSave("D:/pyG/data/point_split_test.nk")


    # daymonthtime = datetime.now().strftime("%d%m%y_%H%M%S")
    # scriptname = "D:/pyG/data/DatasetGen{}.nk".format(daymonthtime)
    # nuke.scriptNew(scriptname)
    # #set full size format of nuke script to be 224x224
    # p2sFormat='224 224 p2s'
    # nuke.addFormat(p2sFormat)
    # nuke.root()["format"].setValue("p2s")
    # nuke.scriptNew(scriptname)
    # roto = createRotoNode()
    # blur=createBlurNode(roto)
    # grade=createGradeNode(blur)
    # shape=nShapeCreator(5, roto)
    # shape.randomisePoints(100,15,15)
    # shape.growPointsTangent(100,10)
    # shape.printPoints(100)
    # shape.randomisePoints(150, 15, 15)
    # shape.randomisePointsTangent(25,5,5)
    # shape.randomisePointsTangent(200,5,4)
    # shape.growPoints(110,0.4)
    # shape.printPoints(200)
    # shape.growPoints(135, -0.3)
    # shape.translatePoints(200, 15,15)
    # shape.translatePoints(400, -30,-20)
    # shape.growPoints(600, 0.5)
    # shape.growPoints(700, -0.5)
    # shape.resetPointsTangent(750)
    # shape.randomisePoints(900, 30, 30)
    # shape.randomisePoints(1800, 5, 5)
    # shape.growPointsTangent(2000, 5)
    # shape.growPoints(8000, 0.5)
    # shape.randomisePoints(3000, 15, 15)
    # shape.randomisePoints(4000, 15, 15)
    # shape.randomisePoints(5000, 15, 15)
    # shape.resetPointsTangent(5100)
    # shape.randomisePoints(6000, 15, 15)
    # shape.growPointsTangent(6100, 0.5)
    # shape.translatePoints(7000, -15, 15)
    # shape.randomisePoints(7000, 15, 15)
    # shape.growPointsTangent(7100, 2)
    # shape.randomisePoints(8000, 15, 15)
    # shape.randomisePoints(9000, 15, 15)
    # shape.growPointsTangent(9100, 10)
    # shape.randomisePoints(10000, 15, 15)
    # dataGenenerator=Datagen(shape,grade,daymonthtime)
    # dataGenenerator.frameRange=[1,10000]
    # dataGenenerator.createPointsDict()
    # dataGenenerator.printPointsDict()
    # dataGenenerator.savePointsDict(daymonthtime)
    # # # #
    # # # # #render write Node
    # dataGenenerator.render()
    # nuke.scriptSave(scriptname)
    # nuke.scriptClose()

    nuke.scriptOpen("D:/DeepParametricShapes/nukeScripts/Cadis_example_01.nk")
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
    # Frange=10000
    # nuke.scriptOpen("D:/DeepParametricShapes/nukeScripts/Cadis_example_01.nk")
    # # grade=nuke.toNode("Grade1")
    # roto = nuke.toNode("Roto4")
    # # blur = createBlurNode(roto)
    # grade = nuke.toNode("Grade2")
    # trans = nuke.toNode("Transform3")
    #shape2 = shapeFromRotopaint(roto)
    # total_offset_x = 0
    # total_offset_y = 0
    # print("generating shape 2 keyframes")
    # for frame in range(1, Frange):
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
    #     shape2.translatePoints(frame, x, y)
    #     shape2.growPoints(frame,scale)
    #     shape2.growPointsTangent(frame, scale)
    #
    #     transx = trans["translate"].valueAt(frame, 0)
    #     transy = trans["translate"].valueAt(frame, 1)
    #     scaleXY= trans["scale"].valueAt(frame, 0)
    #
    #     trans["translate"].setValueAt(x + transx, frame, 0)
    #     trans["translate"].setValueAt(y + transy, frame, 1)
    #     trans["scale"].setValueAt(scaleXY+scale, frame, 0)





    #roto['curves'].rootLayer.append(shape2.shape)
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


    # test cornerPointoRoto
    cp=nuke.toNode("CornerPin2D4")
    roto=nuke.toNode("Roto5")
    cp2roto=CornerPinRoto(cp,roto,frameRange=[0,172])
    cp2roto.transformRoto()

    nuke.scriptSave("D:/DeepParametricShapes/nukeScripts/Cadis_example_01.nk")

    nuke.scriptClose()



    # #create new nuke script
    # nuke.scriptNew("D:/pyG/data/OutputLoader.nk")
    # shapeLoader=shapeLoader(r"D:\pyG\temp\RES\04-12-14-45-10",gtLabels=r"D:\pyG\data\points\120423_183451\points120423_183451.json")
    # #save nuke script
    # nuke.scriptSave("D:/pyG/data/OutputLoader.nk")



