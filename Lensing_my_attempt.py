#Note: This includes my attempts at changing some of the code to produce a correct result, but is detached from the tutorial somewhat.
import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l
import math
#Recall the lens equation y=x-alpha(x), where y and x are coordinates at source and lens plane respectively (vectors)

#Begin with defining parameters needed for lensing

#Lens parameters (coords and mass)
x_l, y_l = 0.0, 0.0
m_l= 1.0

#Source parameters (position and radius)
x_source, y_source= 0.0, 0.0
r=0.1

#Step 1: Divide source and image plane regions into cells/pixels.

n_x, n_y= 401, 401 #pixels in image plane. The larger these are, the more rays there will be. 
xl=5 #Half size of region at lens plane we care about 
yl=5 #Half size of region at source plane we care about

#These are the size of the cells, where x_s is the lens plane and y_s is the source plane
x_s=2*xl/(n_x-1)
y_s=2*yl/(n_y-1)

#Step 2: Convert parameters to pixels. The pixels range from [0,n_y] for the source plane.
#The bottom left coordinate corresponds to pixel (0,0), so we add half of the length of the region (xl or yl) and divide by the size of each cell and take the floor of it.

def convert_to_pixel_coordinates(x_coord, y_coord, region_size, cell_size,n):
    
    # Calculate the pixel indices for the source plane
    i = math.floor(((x_coord + region_size) / cell_size))
    j = math.floor((y_coord + region_size) / cell_size)
    
    # Ensure the indices are within bounds
    i = max(min(i, n - 1), 0)
    j = max(min(j, n - 1), 0)
    
    return i, j
#Convert source parameters
x_source_pix, y_source_pix = convert_to_pixel_coordinates(x_source,y_source, yl,y_s,n_y)

#Convert radius too
r_pix=r/y_s


#Step 3: Set surface brightness at source plane.
a=s.gcirc(n_y,r_pix,x_source_pix,y_source_pix)
#Construct Image Plane
b=np.zeros((n_x,n_x))

for j1 in range(n_x):
    for j2 in range(n_x):
        x1 = -xl + j2 * x_s  # Convert pix to coords on image
        x2 = -xl + j1 * x_s
        y1, y2 = l.Point(x1, x2, x_l, y_l, m_l)  # Deflect X,Y coordinates
        i2 = int(round((y1 + yl) / y_s))  # Convert coordinates to pixels
        i1 = int(round((y2 + yl) / y_s))
        # If deflected ray hits a pixel within source then set image
        # to brightness on that pixel
        if 0 <= i1 < n_y and 0 <= i2 < n_y:
            b[j1, j2] = a[i1, i2]


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