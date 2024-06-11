#Note: this is the first code provided by the tutorial with my corrections added.
import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l

nx=401 # Number of pixels in image plane
ny=401 # Number of pixels in source plane
L=xl=2. # Half size of image plane covered (in "Einstein" radii)
yl=2. # Half size of source plane covered (in Einstein radii)
xs=2.*xl/(nx-1) # pixel size on the image map. This is the size of the cell
ys=2.*yl/(ny-1) # pixel size on the source map. This is the size of the cell

# Coordinates of lens 1
xd= 0.
yd= 0.

# Coordinates of lens 2
xd2= - 0.5
yd2= 0.

# Mass of lenses
ml=1.
ml2=0.5
# Source parameters

xpos= 0.0 # Source position. X coordinate
ypos= 0.0 # Source position. Y coordinate
rad= 0.1 # Radius of source

xd= 0.
yd = 0.
xpos= 0.0 # Source position. X coordinate
ypos= 0.0 # Source position. Y coordinate

"""
THe notation below suggested that ipos and jpos were pixel labels - in this case the original implementation did not make any sense...
xpos/ys is not a pixel coordinate - (yl+xpos)/ys is...
"""
ipos=int(round(xpos/ys)) # Convert source parameters to pixels
#ipos=int(round((yl+xpos)/ys)) # Convert source parameters to pixels
jpos=int(round(ypos/ys)) #Note: In the original code, there was a negative sign in front of ypos
#jpos=int(round((yl-ypos)/ys)) #Note: In the original code, there was a negative sign in front of ypos
rpix=int(round(rad/ys))
print((ipos,jpos))
#print("source pixel coordinates = "(jpos,ipos))

a=s.gcirc(ny,rpix,jpos,ipos) # This is a circular gaussian source
print(a.shape)
b=np.zeros((nx,nx)) # This is the image plane

#This is the main loop over pixels at the image plane
# the apparently flipped order of j1 and j2  has to do with the fact that the 2D array b[i,j] is plotted as an image with i running vertically and j horizontally. 
# so i runs over the vertical dimension, j runs over the horizonatal. 
for j1 in range(nx):
    for j2 in range(nx):
        x1 = -xl + j2 * xs  # Convert pix to coords on image
        x2 = +xl - j1 * xs
        #print("x1,x2",(x1,x2))
        y1, y2 = l.Point(x1, x2, xd, yd, ml)  # Deflect X,Y coordinates
        i2 = int(round((y1 + yl) / ys))  # Convert coordinates to pixels
        i1 = int(round((yl - y2) / ys))
        # If deflected ray hits a pixel within source then set image
        # to brightness on that pixel
        #print("this is j1 and j2", j1, j2, "this is i1 and i2", i1, i2)
        #print((i1, i2))
        if 0 <= i1 < ny and 0 <= i2 < ny:
            b[j1, j2] = a[i1, i2]

#This next nested for loop is for the case of 2 lenses.

# # This is the main loop over pixels at the image plane
# for j1 in range(nx):
#     for j2 in range(nx):
#         x1=-xl+j2*xs # Convert pix to coords on image
#         x2=-xl+j1*xs
#         y1,y2=l.TwoPoints(x1,x2,xd,xd2,yd,yd2,ml,ml2) # Deflect X,Y coordinates
#         i2=int(round((y1+yl)/ys)) # Convert coordinates to pixels
#         i1=int(round((y2+yl)/ys))
#         # If deflected ray hits a pixel within source then set image
#         # to brightness on that pixel
#         if 0 <= i1 < ny and 0 <= i2 < ny:
#             b[j1, j2] = a[i1, i2]




#This reflects along the y=0 line, as it appears this will be needed
# for j1 in range(nx//2):
#    for j2 in range(nx):
#        a[nx - j1 - 1, j2], a[j1, j2] = a[j1, j2], a[nx - j1 - 1,j2]
#        b[nx - j1 - 1, j2], b[j1, j2] = b[j1, j2], b[nx - j1 - 1,j2]

# Plotting
plt.subplot(121)
plt.imshow(a, extent=(-yl, yl, -yl, yl), cmap='hot')
plt.title('Source Plane')
#plt.colorbar(label='Brightness')
plt.subplot(122)
plt.imshow(b, extent=(-xl, xl, -xl, xl), cmap='hot')
plt.title('Image Plane')
#plt.colorbar(label='Brightness')

plt.show()