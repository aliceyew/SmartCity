import cv2
import numpy
import matplotlib.pyplot as pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
import scipy.linalg
import random
import math
import time 


#start_time = time.time()

fig = pyplot.figure()
ax = fig.gca(projection='3d')

# generate x, y and z coordinates
#X = numpy.arange(0, 620) #512
#Y = numpy.arange(0, 460) #424
#z = numpy.loadtxt("24in_RealCrack1.txt", delimiter=" ")
#z = numpy.loadtxt("frame4190.txt", delimiter=" ")

z = numpy.loadtxt("d_org130.txt", delimiter="\t")

# to remove the edges
#z = z[7:-7, 7:-7]
#z = z[:,420:-115]

# No hardcoding
x,y = z.shape
X = numpy.arange(0, y)
Y = numpy.arange(0, x)

# to subtract height from ground
#z = z - 609.6
#z = z - 0.5334
#z = z - 0.6096
#z = z * 100
#z = z - 61
#z = z - 62
#z= z - 12
# plot 3d graph x, y and depth data
#z[z<-30]=0
X, Y = numpy.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
pyplot.xlabel('X pixels')
pyplot.ylabel('Y pixels')
ax.set_zlabel('Z: Depth values(mm)')
#pyplot.show()
fig.savefig("dorg130") #save the image
#numpy.savetxt("testremove.txt", z, delimiter=" ", fmt = '%.4f')
# combine x, y and z into a 3 column matrix
#rows = 285200
rows = x * y
columns = 3
maxlength = y
#maxlength = 620
maxwidth = x
#maxwidth = 460
B = numpy.empty((rows, columns))
row = 0
for width in range(maxwidth):
	for length in range(maxlength):
		B[row, 0] = width
		B[row, 1] = length
		B[row, 2] = z[width, length]
		row = row + 1

# run ransac on dataset B (1st trial)
#model_robust, inliers = ransac(B, LineModelND, min_samples=3, residual_threshold=1, max_trials=1000)
# get the inverse of inliers
#outliers = inliers == False

# prepare the B dataset for trial 2
point_list = []
bbox = numpy.array([float('Inf'),-float('Inf'),float('Inf'),-float('Inf'),float('Inf'),-float('Inf')])

points = numpy.array(B)


for xyz in points:
    bbox[0] = min(bbox[0], xyz[0]) # min x
    bbox[1] = max(bbox[1], xyz[0]) # max y
    bbox[2] = min(bbox[2], xyz[1]) # min x
    bbox[3] = max(bbox[3], xyz[1]) # max y
    bbox[4] = min(bbox[4], xyz[2]) # min z
    bbox[5] = max(bbox[5], xyz[2]) # max z

bbox_corners = numpy.array([
    [bbox[0],bbox[2], bbox[4]],
    [bbox[0],bbox[2], bbox[5]],
    [bbox[0],bbox[3], bbox[5]],
    [bbox[0],bbox[3], bbox[4]],
    [bbox[1],bbox[3], bbox[4]],
    [bbox[1],bbox[2], bbox[4]],
    [bbox[1],bbox[2], bbox[5]],
    [bbox[1],bbox[3], bbox[5]]]);

bbox_center = numpy.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2]);

#run ransac on dataset B (2nd trial)
#code taken from https://github.com/minghuam/point-visualizer/blob/master/point_visualizer.py
#http://www.cse.yorku.ca/~kosta/CompVis_Notes/ransac.pdf
# tolerance for distance, e.g. 0.0027m for kinect
TOLERANCE = 0.78 #5
# ratio of inliers
THRESHOLD = 0.05
N_ITERATIONS = 1000
iterations = 0
solved = 0
while iterations < N_ITERATIONS and solved == 0:
	iterations += 1
	max_error = -float('inf')
	max_index = -1
    # randomly pick three non-colinear points
    CP = numpy.array([0,0,0]);
    while CP[0] == 0 and CP[1] == 0 and CP[2] == 0:
    	[A,B,C] = points[random.sample(range(len(points)), 3)];
        # make sure they are non-collinear
        CP = numpy.cross(A-B, B-C);
    # calculate plane coefficients
    abc = numpy.dot(numpy.linalg.inv(numpy.array([A,B,C])), numpy.ones([3,1]))
    # get distances from the plane
    d = math.sqrt(abc[0]*abc[0]+abc[1]*abc[1]+abc[2]*abc[2])
    dist = abs((numpy.dot(points, abc) - 1)/d)
    #print max(dist),min(dist)
    ind = numpy.where(dist < TOLERANCE)[0];
    ratio = float(len(ind))/len(points)
    if ratio > THRESHOLD:
    	# satisfied, now fit model with the inliers
        # least squares reference plane: ax+by+cz=1
        inliers = numpy.take(points, ind, 0)
        print('\niterations: {0}, ratio: {1}, {2}/{3}'.format(iterations, ratio,len(points),len(inliers)))
        [a,b,c] = numpy.dot(numpy.linalg.pinv(inliers), numpy.ones([len(inliers), 1]))
        plane_pts = numpy.array([
           	[bbox[0], bbox[2], (1-a*bbox[0]-b*bbox[2])/c],
           	[bbox[0], bbox[3], (1-a*bbox[0]-b*bbox[3])/c],
           	[bbox[1], bbox[3], (1-a*bbox[1]-b*bbox[3])/c],
           	[bbox[1], bbox[2], (1-a*bbox[1]-b*bbox[2])/c]]);
        print('Least squares solution coeffiecients for ax+by+cz=1')
        print (a,b,c);
        solved = 1

# plot ransac
#fig = pyplot.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(points[ind][:,0], points[ind][:,1], points[ind][:,2], c='b', marker='o', label='Inlier data')
#fig.savefig("dorg_130ransacInliers") #save the image
#ax.scatter(points[outliers][:,0], points[outliers][:,1], points[outliers][:,2], c='b', marker='o', label='Outlier data')
#pyplot.show()
#fig.savefig("dorg_19ransacOutliers") #save the image

# Linear plane eq aX + bY + cZ = 1 (trial 1)
Z = (1 - a*X - b*Y)/c 
# Linear plane eq trial 2 Z = ax + by + c
#Z = a*X + b*Y + c 
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=1, cstride=1, alpha=0.2)
surf = ax.plot_surface(X,Y,z, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
#pyplot.show()
pyplot.xlabel('X pixels')
pyplot.ylabel('Y pixels')
ax.set_zlabel('Z: Depth values(mm)')
fig.savefig("dorg_26planetest")

# Depth image subtracted from fitted plane
#depthdiff = Z  - z
#This one works
depthdiff = z - Z 
# Set to 0 depth diff greater than 5mm
depthdiff[depthdiff > 5] = 0
depthdiff[depthdiff > 0] = 0
#this one works
#mask = (depthdiff > 0) & (depthdiff < 25)
#mask = (depthdiff > 0) and (depthdiff < 5)
#depthdiff[mask] = 0

# Plot test image of both the plane and the subtracted depth data
fig = pyplot.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,depthdiff, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)
#pyplot.show()
pyplot.xlabel('X pixels')
pyplot.ylabel('Y pixels')
ax.set_zlabel('Z: Depth values(mm)')
fig.savefig("dorg26plots")
#fig.savefig("dorg26plots")
pyplot.show()
# trial 2 because 5mm seems to still have a lot of points above the plane
# this time set to 2mm
#depthdiff[depthdiff > 5] = 0
#depthdiff[depth > 0] = 0
#depthdiff[depthdiff > -5] = 0

#Find the max depth in the array 
deepest = min(depthdiff.flatten())
#divide by deepest - normalized
test = depthdiff/ deepest # why are the values not 0 to 1?
#show grayscale img
pyplot.imshow(test)
pyplot.show()
#pyplot.imsave("depthdiff19.png", test)

#Need to adjust what i change to 0 because currently it only shows the left part
from skimage import filters
otsuimg = filters.threshold_otsu(test)
pyplot.imshow(test > otsuimg, cmap='gray', interpolation='nearest')
#pyplot.show()
xlim, ylim = pyplot.xlim(), pyplot.ylim()
pyplot.plot(x,y,"o")
pyplot.xlim(xlim)
pyplot.ylim(ylim)
pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
#pyplot.savefig('test')
#pyplot.show()
#pyplot.imsave("test.png", test > otsuimg, cmap='gray')


#Values positive negative problem?

#calculate percentage and occurrences of number of times it exceeds the threshold
#count = 0
#for i in range(maxwidth):
#    for j in range(maxlength):
#        if test[i][j] > otsuimg:
#            count = count + 1
#count


# clear plot
#pyplot.clf()

#save the image
pyplot.axis('off')
pyplot.savefig('test.png', bbox_inches='tight', pad_inches=0)

#read in the image
img = cv2.imread('test.png', 0)
img3 = cv2.imread('otsu19.png', 0)
black = cv2.imread('blackbg.png', 0)

#convert to binary
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#Insted of binary, use canny edges, see if its more accurate
#edges = cv2.Canny(img, 100, 200)
#find contours
#test, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
test, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
#after test drawing out the contours, correct contour is 8
cnt = contours[8]
# for test drawing
#cv2.drawContours(img2, contours, -1, (255,255,0), 3)
#pyplot.imshow(img2)
#pyplot.show()
#find bounding box
rect = cv2.minAreaRect(cnt)
#rect (center(x,y),(width, height), angle of rotation)
box = cv2.boxPoints(rect)
box = numpy.int0(box)
#cv2.drawContours(img3, [box], 0, (255,255,0),3)

#pyplot.imshow(img3)
#ax.set_title('X length %.2f', rect[1][0])
#pyplot.xlabel('X pixels')
#pyplot.ylabel('Y pixels')
#pyplot.show()

#Calculations!!!!
#find the area of the contour?
#area = cv2.contourArea(cnt)
#rect[1]
horizontalField = 57 #angle degree
verticalField = 43 #degree
height = 778 #mm
widthRes = 640 #pixel
lengthRes = 480 #pixel
kinectAngle = math.radians(horizontalField) #convert angle from degree to radians
lengthAct = 2*height*math.tan(kinectAngle/2)
pixelLength =  lengthAct/widthRes
widthdiameter = rect[1][0]
lengthdiameter = rect[1][1]
widthdiameter = widthdiameter * pixelLength
lengthdiameter = lengthdiameter * pixelLength
avgDiameter = (widthdiameter+lengthdiameter)/2

#if else statements for severity level using @avgDiameter and @deepest
deepest = abs(deepest)
if deepest <= 25: 
    if avgDiameter <= 200:
        print("L")
    elif avgDiameter > 200 and avgDiameter <= 450:
        print("L")
    elif avgDiameter > 450:
        print("M")
elif deepest > 25 and deepest <= 50:
    if avgDiameter <= 200:
        print("L")
    elif avgDiameter > 200 and avgDiameter <= 450:
        print("M")
    elif avgDiameter > 450:
        print("H")
elif deepest > 50:
    if avgDiameter <= 200:
        print("M")
    elif avgDiameter > 200 and avgDiameter <= 450:
        print("M")
    elif avgDiameter > 450:
        print("H")
elif deepest < 13 and avgDiameter < 100:
    print("Not a pothole")

#print(time.time() - start_time)
