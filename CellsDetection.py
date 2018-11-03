import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)

# Import the image
image = cv2.imread('C:\Users\la vita e bella\Desktop\Scr.png')

# Resize the image
IMAGE = cv2.resize(image, (1080, 800))
IMAGE = np.array(IMAGE, dtype=np.uint8)

# Create a range of colors, we want to include a little bit of blue and red
lower_Colour = np.array([10, 0, 10])
upper_Colour = np.array([255, 11, 255])

# Create a mask in range of the lower and upper Colour,the result is a binary. White colour are the detected cells
mask = cv2.inRange(IMAGE, lower_Colour, upper_Colour)
mask = cv2.dilate(mask, (3, 3), iterations=7)

# Copy the threshold image.
im_floodfill = mask.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = mask.shape[:2]
hey = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, hey, (0, 0), 255)

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = mask | im_floodfill_inv

# covert the im_out into uint8 type
im_out = np.uint8(im_out)

# create the kernels
kernel = np.ones((5, 5), np.uint8)
nnkernel = np.ones((6, 6), np.uint8)
nkernel = np.ones((4, 4), np.uint8)
krnl = np.ones((3, 3), np.uint8)

# noise removal
opening = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel, iterations=2)

# GROUP 1                                                                     GROUP 1

'''
 Now we will do four iterations fo our method, which are group 1, group 2, group 3 and finally group 4
 the groups do the same think but each group can detect with more accuracy starting from the minimum and finish to the
 maximum 
'''

# We make a sure background area dilating the detected cells
sure_bg = cv2.dilate(opening, kernel, iterations=5)

# Finding sure foreground area
dt = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
ret, sure_fg1 = cv2.threshold(dt, 0.4*dt.max(), 255, 0)

# clean the sure foreground area with morphologyEx
OPsure_fg1 = cv2.morphologyEx(sure_fg1, cv2.MORPH_OPEN, kernel, iterations=2)

# We made an area to be abstracted from the threshold of group2 to find the new sure foreground area
ForAbstract = cv2.dilate(OPsure_fg1, nnkernel, iterations=5)

# Type uint8 GROUP 1
sure_bg = np.uint8(sure_bg)
OPsure_fg1 = np.uint8(OPsure_fg1)
ForAbstract = np.uint8(ForAbstract)

# GROUP 2                                                    GROUP 2

# Finding the sure foreground area
ret, sure_fg2 = cv2.threshold(dt, 0.3*dt.max(), 255, 0)

# type uint8
sure_fg2 = np.uint8(sure_fg2)
sure_fg2 = cv2.dilate(sure_fg2, kernel, iterations=4)

# We want to find the bounds of cells which we did not found them in group1
# so we use the bitwise_xor function to "delete" the same detected cells from group 1 and 2
new_areas = cv2.bitwise_xor(ForAbstract, sure_fg2)

# we made the new sure foreground
OPnew_areas = cv2.morphologyEx(new_areas, cv2.MORPH_OPEN, nnkernel, iterations=5)
OPsure_fg2 = cv2.erode(OPnew_areas, kernel, iterations=5)

# We made the new sure background
sure_bg2 = cv2.dilate(OPsure_fg2, kernel, iterations=15)
ForAbstract2 = cv2.dilate(OPsure_fg2, nnkernel, iterations=5)

# type uint8 group 2
new_areas = np.uint8(new_areas)
sure_bg2 = np.uint8(sure_bg2)
OPsure_fg2 = np.uint8(OPsure_fg2)
ForAbstract2 = np.uint8(ForAbstract2)


# GROUP 3                                                    GROUP 3


# Finding the sure foreground area
ret, sure_fg3 = cv2.threshold(dt, 0.2*dt.max(), 255, 0)

# type uint8
sure_fg3 = np.uint8(sure_fg3)
sure_fg3 = cv2.dilate(sure_fg3, kernel, iterations=3)

# We want to find the bounds of cells which we did not found them in group1 and group2
# so we use the bitwise_xor function for that purpose
new_areasF1 = cv2.bitwise_xor(ForAbstract, sure_fg2)
OPnew_areasF1 = cv2.morphologyEx(new_areasF1, cv2.MORPH_OPEN, kernel, iterations=3)

new_areasF2 = cv2.bitwise_xor(ForAbstract2, sure_fg3)
OPnew_areasF2 = cv2.morphologyEx(new_areasF2, cv2.MORPH_OPEN, kernel, iterations=3)

# the new detected cells are:
new_areasG3 = cv2.bitwise_and(OPnew_areasF1, OPnew_areasF2)

# we made the new sure foreground
OPnew_areasG3 = cv2.morphologyEx(new_areasG3, cv2.MORPH_OPEN, nnkernel, iterations=4)
OPsure_fg3 = cv2.erode(OPnew_areasG3, kernel, iterations=4)

# We made the new sure background
sure_bg3 = cv2.dilate(OPsure_fg3, kernel, iterations=15)
OPsure_fg3 = np.uint8(OPsure_fg3)

ForAbstract3 = cv2.dilate(OPsure_fg3, nkernel, iterations=9)
ForAbstract3 = np.uint8(ForAbstract3)


# GROUP 4                                                    GROUP 4


_, sure_fg4 = cv2.threshold(dt, 0.1*dt.max(), 255, 0)

# type uint8
sure_fg4 = np.uint8(sure_fg4)

ForAbstract = cv2.dilate(ForAbstract, kernel, iterations=2)
ForAbstract2 = cv2.dilate(ForAbstract2, kernel, iterations=2)

# Finding the sure foreground area
sure_fg4 = cv2.dilate(sure_fg3, kernel, iterations=3)

# we remove the detected cells from previous groups
new_areasF1G4 = cv2.bitwise_xor(ForAbstract, sure_fg4)
OPnew_areasF1G4 = cv2.morphologyEx(new_areasF1G4, cv2.MORPH_OPEN, kernel, iterations=4)

new_areasF2G4 = cv2.bitwise_xor(ForAbstract2, sure_fg4)
OPnew_areasF2G4 = cv2.morphologyEx(new_areasF2G4, cv2.MORPH_OPEN, kernel, iterations=4)

new_areasF3 = cv2.bitwise_xor(ForAbstract3, sure_fg4)
OPnew_areasF3 = cv2.morphologyEx(new_areasF3, cv2.MORPH_OPEN, kernel, iterations=5)

new_areasAND1G4 = cv2.bitwise_and(OPnew_areasF1G4, OPnew_areasF2G4)
OPnew_areasAND1G4 = cv2.morphologyEx(new_areasAND1G4, cv2.MORPH_OPEN, kernel, iterations=3)

new_areasG4 = cv2.bitwise_and(OPnew_areasF3, OPnew_areasAND1G4)
OPnew_areasG4 = cv2.morphologyEx(new_areasG4, cv2.MORPH_OPEN,nnkernel, iterations=6)
OPsure_fg4 = cv2.erode(OPnew_areasG4, kernel, iterations=3)

# We made the new sure background
sure_bg4 = cv2.dilate(OPsure_fg4, kernel, iterations=6)
OPsure_fg4 = np.uint8(OPsure_fg4)
sure_bg4 = np.uint8(sure_bg4)

# Finding unknown region
UnknownG1 = cv2.subtract(sure_bg, OPsure_fg1)
UnknownG2 = cv2.subtract(sure_bg2, OPsure_fg2)
UnknownG3 = cv2.subtract(sure_bg3, OPsure_fg3)
UnknownG4 = cv2.subtract(sure_bg4, OPsure_fg4)

# Marker labelling
ret, markersG1 = cv2.connectedComponents(OPsure_fg1)
ret, markersG2 = cv2.connectedComponents(OPsure_fg2)
_, markersG3 = cv2.connectedComponents(OPsure_fg3)
_, markersG4 = cv2.connectedComponents(OPsure_fg4)

# Add one to all labels so that sure background is not 0, but 1
markersG1 = markersG1 + 1
markersG2 = markersG2 + 1
markersG3 = markersG3 + 1
markersG4 = markersG4 + 1

# Now, mark the region of unknown with zero
markersG1[UnknownG1 == 255] = 0
markersG2[UnknownG2 == 255] = 0
markersG3[UnknownG3 == 255] = 0
markersG4[UnknownG4 == 255] = 0

# we apply the watershed method to IMAGE using the previous unknown marked regions
markersG1 = cv2.watershed(IMAGE, markersG1)
image3 = IMAGE.copy()
# whe paint the bounds to IMAGE
IMAGE[markersG1 == -1] = [0, 0, 0]
IMAGE[markersG1 > 1] = [255, 255, 255]

# we repeat the process
markersG2 = cv2.watershed(IMAGE, markersG2)
# im_out[markersG2 == -1] = 0
IMAGE[markersG2 == -1] = [0, 0, 0]
IMAGE[markersG2 > 1] = [255,255, 255]


markersG3 = cv2.watershed(IMAGE, markersG3)
# im_out[markersG3 == -1] = 0
IMAGE[markersG3 == -1] = [0, 0, 0]
IMAGE[markersG3 > 1] = [255, 255, 255]


markersG4 = cv2.watershed(IMAGE, markersG4)
# im_out[markersG4 == -1] = 0
IMAGE[markersG4 == -1] = [0, 0, 0]
IMAGE[markersG4 > 1] = [255, 255, 255]

# we want to create a binary image with separated detected cells to find the contours
gray = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
_, Separate_image = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

# we clean the noise
Separate_image = cv2.morphologyEx(Separate_image, cv2.MORPH_OPEN, krnl, iterations=7)

# we apply findContour to find the boundaries
im2, contours, hierarchy = cv2.findContours(Separate_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(IMAGE, contours, -1, (255, 255, 0), 1)

# now we will find the center x,y of Minimum Enclosing Circle
listcenter = []
for i, j in enumerate(contours):
    (x, y), radius = cv2.minEnclosingCircle(contours[i])
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(image3, center, radius, (0, 255, 0), 1)
    cv2.rectangle(image3, (int(x) - 2, int(y) - 2), (int(x) + 2, int(y) + 2), (0, 255, 0), 1)
    listcenter.append(center)





