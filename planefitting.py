import cv2
import numpy
import matplotlib.pyplot as pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import LineModelND, ransac
import scipy.linalg
import random
import math


fig = pyplot.figure()
ax = fig.gca(projection='3d')

# generate x, y and z coordinates
#X = numpy.arange(0, 620) #512
#Y = numpy.arange(0, 460) #424
z = numpy.loadtxt("24in_RealCrack1.txt", delimiter=" ")
# No hardcoding
x,y = z.shape
X = numpy.arange(0, x)
Y = numpy.arange(0, y)

# plot 3d graph x, y and depth data
X, Y = numpy.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
pyplot.show()
fig.savefig("d_org19") #save the image

# combine x, y and z into a 3 column matrix
rows = 285200
columns = 3
maxlength = 620
maxwidth = 460
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
TOLERANCE = 0.78 #0.78
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
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[ind][:,0], points[ind][:,1], points[ind][:,2], c='b', marker='o', label='Inlier data')
#fig.savefig("dorg_130ransacInliers") #save the image
#ax.scatter(points[outliers][:,0], points[outliers][:,1], points[outliers][:,2], c='b', marker='o', label='Outlier data')
#pyplot.show()
fig.savefig("dorg_19ransacOutliers") #save the image

# Linear plane eq aX + bY + cZ = 1 (trial 1)
#Z = (1 - a*X - b*Y)/c
# Linear plane eq trial 2 Z = ax + by + c
Z = a*X + b*Y + c 
fig = pyplot.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=1, cstride=1, alpha=0.2)
#pyplot.show()
fig.savefig("dorg_19planetest2")

# Depth image subtracted from fitted plane
#depthdiff = Z  - z
#This one works
depthdiff = z - Z 
# Set to 0 depth diff greater than 5mm
#depthdiff[depthdiff > 5] = 0
#this one works
mask = (depthdiff > 0) & (depthdiff < 25)
depthdiff[mask] = 0

# Plot test image of both the plane and the subtracted depth data
fig = pyplot.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X,Y,depthdiff, cmap=cm.coolwarm, linewidth = 0, antialiased=False)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
fig.colorbar(surf, shrink=0.5, aspect=5)
#pyplot.show()
fig.savefig("dorg19planeandplots")
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
#pyplot.show()
pyplot.imsave("depthdiff19.png", test)

#Need to adjust what i change to 0 because currently it only shows the left part
from skimage import filters
otsuimg = filters.threshold_otsu(test)
pyplot.imshow(test < otsuimg, cmap='gray', interpolation='nearest')
#pyplot.show()
pyplot.imsave("otsu19.png", test < otsuimg, cmap='gray')

#Values positive negative problem?

#calculate percentage and occurrences of number of times it exceeds the threshold
count = 0
for i in range(maxwidth):
    for j in range(maxlength):
        if test[i][j] > otsuimg:
            count = count + 1
count


# clear plot
pyplot.clf()
