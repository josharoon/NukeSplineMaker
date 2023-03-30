import fnmatch
import json
import math
import os
import pickle
import random
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

    def addPoint(self, x, y, ltx=0, lty=0, rtx=0, rty=0):
        ctrlPoint = rp.ShapeControlPoint(x, y)
        ctrlPoint.leftTangent = (ltx, lty)
        ctrlPoint.rightTangent = (rtx, rty)
        self.shape.append(ctrlPoint)

    def addPoints(self):
        """add points to shape"""
        for point in self.points:
            self.addPoint( point[0], point[1])

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
        point.center.addPositionKey(time, xy)

    def getPtCoord(self, point, time):
        """get the x,y coords of a point in list format"""
        xy = list(point.center.getPosition(time))[:2]
        return xy

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
        x=xyLeft[0]
        y=xyLeft[1]
        x+=amount
        y-=amount
        point.leftTangent.addPositionKey(time,[x,y])
        point.featherLeftTangent.addPositionKey(time,[x,y])
        x=xyRight[0]
        y=xyRight[1]
        x-=amount
        y+=amount
        point.rightTangent.addPositionKey(time,[x,y])
        point.featherRightTangent.addPositionKey(time,[x,y])

    def growPointsTangent(self,time,amount):
        """grow the tangent of all points"""
        for point in range(len(self.shape)):
            self.growPointTangent(time,point,amount)
        self.getListofCtrlPoints()

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
        print("getting list of control points")
        self.npCtrlPoints=[]
        for i in range(self.nPoints):
            #convert point into an numpy array (lefttangent, center, righttangent)
            npPoint=np.array([self.shape[i].leftTangent.getPosition(1),self.shape[i].center.getPosition(1),self.shape[i].rightTangent.getPosition(1)])
            npPoint=npPoint[:,:2]
            self.npCtrlPoints.append(npPoint)




class shapeLoader(nShapeMaster):
    """load shapes from pickle file into a roto node"""
    def __init__(self, outputDir):
        super().__init__()
        self.outputDir = outputDir
        self.pickeFiles=self.getFiles()
        self.pickleFile = None
        self.npShape = None
        self.imagePath = None
        self.filesDict= None
        #self.loadShape()
        #self.rotoNode = nuke.createNode('Roto')
        self.getFiles()
        self.createShapes()


    def getFiles(self):
        """get all the pickle files in the output directory as a list of lists """
        # search recursively for all pickle files in the output directory
        files = []
        for root, dirnames, filenames in os.walk(self.outputDir):
            for filename in fnmatch.filter(filenames, '*.pkl'):
                files.append([root, filename])
        # get the last number of the file name separated by underscore
        # e.g. 2_0.pkl = 0 and add the file path to a dictionary with the number as the key
        # e.g. {0: ['D:\pyG\data\points\processed\2_0.pkl']}
        # if the key already exists append the file path to the list
        # e.g. {0: ['D:\pyG\data\points\processed\2_0.pkl','D:\pyG\data\points\processed\3_0.pkl']}
        filesDict = {}
        for file in files:
            key = int(file[1].split('_')[1].split('.')[0])
            if key in filesDict:
                filesDict[key].append(os.path.join(file[0], file[1]))
            else:
                filesDict[key] = [os.path.join(file[0], file[1])]

        self.filesDict=filesDict

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
            print("connecting read node {} to roto node {}".format(self.readNode.name(),self.rotoNode.name()))
            self.rotoNode.setInput(0, self.readNode)
            #create the shape
            #set current time in nuke to frame 1
            frame=1
            nuke.frame(frame)
            self.createShape()
            #interate through the rest of the shapes in the list
            for shape in self.filesDict[shape][1:]:
                frame+=1
                nuke.frame(frame)
                self.loadShape(shape)
                self.addPointsToShape()

    def addPointsToShape(self):
        """add point keyframes to the existing shape"""
        self.npShape2points()
        for i in range(len(self.points)):
            #add point at current time
            point=self.shape[i]
            point.center.addPositionKey(nuke.frame(),self.points[i])





    def loadShape(self,path):
        with open(path, 'rb') as f:
            name, shape = pickle.load(f)
        self.npShape = shape
        self.imagePath = name
        print ("loaded shape {} from {}".format(name,path))
    def npShape2points(self):
        """convert a numpy shape to a list of points"""
        self.points=[]
        for point in self.npShape:
            self.points.append([point[0], point[1]])
    def createShape(self):
        self.curveKnob = self.rotoNode['curves']
        self.shape = rp.Shape(self.curveKnob)
        self.npShape2points()
        self.addPoints()

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
        self.addPoints()

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
    def __init__(self,shape):
        self.shape=shape
        self.root = "D:/pyG/data/points/"
        self.frameRange=[1,200]
        self.pointsDict={}
        self.createPointsDict()
        self.attachWriteNode()

    def createPointsDict(self):
        """create a dictionary of points for each frame"""
        for frame in range(self.frameRange[0],self.frameRange[1]+1):
            pointDictList=[]
            for point in self.shape.shape:
                pointAtTime=self.shape.getPointDict(point,frame)
                pointDictList.append(pointAtTime)
            self.pointsDict[frame]=pointDictList

    def printPointsDict(self):
        """print the points dictionary"""
        for frame in self.pointsDict:
            print("frame: ",frame)
            for point in self.pointsDict[frame]:
                print(point)
            print("")

    def savePointsDict(self):
        """save the points dictionary"""

        jsonobj=json.dumps(self.pointsDict)
        with open("{}points.json".format(self.root),"w") as f:
            f.write(jsonobj)


    def attachWriteNode(self):
        """attach a write node to the roto node"""
        writeNode=nuke.createNode("Write")
        writeNode["file_type"].setValue("png")

        writeNode["file"].setValue("{}spoints.####.png".format(self.root))
        writeNode["colorspace"].setValue("linear")
        writeNode["channels"].setValue("rgb")
        writeNode["create_directories"].setValue(1)
        writeNode["datatype"].setValue("8 bit")

        #set first and last frame
        writeNode["first"].setValue(self.frameRange[0])
        writeNode["last"].setValue(self.frameRange[1])
        writeNode.setInput(0,self.shape.rotoNode)
        self.outWriteNode=writeNode

    def render(self):
        """render the write node"""
        nuke.render(self.outWriteNode, start=self.frameRange[0], end=self.frameRange[1])






if __name__ == '__main__':
    #load nuke script
    #nuke.scriptOpen("D:/pyG/data/DatasetGen.nk")
    nuke.scriptNew("D:/pyG/data/point_split_test.nk")
    roto = createRotoNode()
    shape=nShapeCreator(5, roto)
    shape.growPointsTangent(0, 10)
    print("tangents grown")
    shape.printPoints(0)
    print(shape.npCtrlPoints)
    shape.insertPointAtEdge(0,1, 0.25)
    print("addedPoint")
    shape.printPoints(0)
    print(shape.shape.name)
    #close script
    #nuke.scriptClose()




    nuke.scriptSave("D:/pyG/data/point_split_test.nk")
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
    # shape.translatePoints(7000, -15, 15)
    # shape.randomisePoints(7000, 15, 15)
    # shape.randomisePoints(8000, 15, 15)
    # shape.randomisePoints(9000, 15, 15)
    # shape.randomisePoints(10000, 15, 15)
    # dataGenenerator=Datagen(shape)
    # dataGenenerator.frameRange=[1,10000]
    # dataGenenerator.createPointsDict()
    # dataGenenerator.printPointsDict()
    # dataGenenerator.savePointsDict()
    # # #
    # # # #render write Node
    # dataGenenerator.render()
    # nuke.scriptSave("D:/pyG/data/DatasetGen.nk")
    # nuke.scriptClose()
    # # #save nuke script
    #
    # #create new nuke script
    # nuke.scriptNew("D:/pyG/data/OutputLoader.nk")
    # shapeLoader=shapeLoader(r"D:\pyG\temp\RES\03-28_11-12-24")
    #save nuke script
    #nuke.scriptSave("D:/pyG/data/OutputLoader.nk")



