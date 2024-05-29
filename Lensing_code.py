import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l

nx=401 # Number of pixels in image plane
ny=401 # Number of pixels in source plane
xl=2. # Half size of image plane covered (in "Einstein" radii)
yl=2. # Half size of source plane covered (in Einstein radii)
xs=2.*xl/(nx-1) # pixel size on the image map. This is the size of the cell
ys=2.*yl/(ny-1) # pixel size on the source map. This is the size of the cell

# Coordinates of lens 1
xd= 0.
yd= 0.5

# Coordinates of lens 2
xd2= - 0.5
yd2= 0.

# Mass of lenses
ml=1.
ml2=0.5
# Source parameters

xpos= 0.0 # Source position. X coordinate
ypos= 1.0 # Source position. Y coordinate
rad= 0.1 # Radius of source
ipos=int(round(xpos/ys)) # Convert source parameters to pixels
jpos=int(round(ypos/ys))
rpix=int(round(rad/ys))

a=s.gcirc(ny,rpix,jpos,ipos) # This is a circular gaussian source
b=np.zeros((nx,nx)) # This is the image plane

#This is the main loop over pixels at the image plane
for j1 in range(nx):
    for j2 in range(nx):
        x1 = -xl + j2 * xs  # Convert pix to coords on image
        x2 = -xl + j1 * xs
        y1, y2 = l.Point(x1, x2, xd, yd, ml)  # Deflect X,Y coordinates
        i2 = int(round((y1 + yl) / ys))  # Convert coordinates to pixels
        i1 = int(round((y2 + yl) / ys))
        # If deflected ray hits a pixel within source then set image
        # to brightness on that pixel
        if 0 <= i1 < ny and 0 <= i2 < ny:
            b[j1, j2] = a[i1, i2]



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
for j1 in range(nx//2):
    for j2 in range(nx):
        a[nx - j1 - 1, j2], a[j1, j2] = a[j1, j2], a[nx - j1 - 1,j2]
        b[nx - j1 - 1, j2], b[j1, j2] = b[j1, j2], b[nx - j1 - 1,j2]

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