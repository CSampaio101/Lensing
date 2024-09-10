import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l

def pixel_to_pos(pixel, origin, pixel_size):
    return(origin + pixel*pixel_size)

def pos_to_pixel(coord, origin, pixel_size):
    """
    coord = origin + pixel*pixel_size
    This transformation is appropriate when the plane origin (i.e. where pixel index is 0) is at "origin", and the pixel indices increase towards upper right
    In other words, indices [0,0] correspond to the lower left corner of the plane
    """
    return(np.rint((np.asarray(coord) - origin) / pixel_size).astype(int))

# Parameters
nx = 401  # Number of pixels in image plane
ny = 401  # Number of pixels in source plane
xl = 2.0  # Size of image plane covered (in "Einstein" radii)
yl = 2.0  # Size of source plane covered (in Einstein radii)

# Lens parameters
xlens = 0.5
ylens = 0.0
mlens = 0.5

xlens2 = -0.5
ylens2 = 0.0
mlens2 = 0.5

xs = 2.0 * xl / (nx - 1)  # Pixel size on the image map
ys = 2.0 * yl / (ny - 1)  # Pixel size on the source map

# Source parameters: Circular Gaussian Source
xpos = 0.05
ypos = 0.4
rad = 0.1
jpos = pos_to_pixel(xpos, -yl, ys)
ipos = pos_to_pixel(ypos, -yl, ys)
print("Source position:", ipos, jpos)
rpix = int(round(rad / ys))

# Create a circular Gaussian source
x = np.linspace(-yl, yl, ny)
y = np.linspace(-yl, yl, ny)
X, Y = np.meshgrid(x, y)
a = np.exp(-((X - xpos)**2 + (Y - ypos)**2) / (2 * rad**2)) / (2.*np.pi*rad**2)
print("a shape = ", a.shape)
plt.imshow(a, origin='lower')
plt.title('Source Plane Before Lensing')
plt.show()
b = np.zeros((nx, nx))  # Image plane
b_no_source = np.zeros((nx, nx))  # Image plane

# Implicit main loop: Operations on numpy arrays (faster)
j1, j2 = np.mgrid[0:nx, 0:nx]
x1 = pixel_to_pos(j2,-xl,xs)  # Convert pixels to coordinates on image horizontal
x2 = pixel_to_pos(j1,-xl,xs)  # Convert pixels to coordinates on image vertical

# Deflect rays using the Point lens model
#y1, y2 = l.Point(x1, x2, xlens, ylens, mlens)
y1, y2 = l.TwoPoints(x1,x2,xlens,ylens,xlens2,ylens2,mlens,mlens2)

# Convert deflected coordinates to source plane pixels
i2 = pos_to_pixel(y1, -yl, ys)
i1 = pos_to_pixel(y2, -yl, ys)

# Filter valid rays hitting the source plane
ind = (i1 >= 0) & (i1 < ny) & (i2 >= 0) & (i2 < ny)
i1n = i1[ind]
i2n = i2[ind]
j1in = j1[ind]
j2in = j2[ind]

# Calculate magnification by summing contributions from all rays
for i in range(np.size(i1n)):
    b[j1in[i], j2in[i]] += a[i1n[i], i2n[i]]

# Plot the results
fig = plt.figure(1)

# Source plane
ax = plt.subplot(121)
#ax.imshow(a, extent=(-yl, yl, -yl, yl), cmap='hot')
ax.imshow(a, cmap='hot', origin='lower', extent=(-yl, yl, -yl, yl))
fa = np.sum(a*ys*ys)  # Flux on source plane
ax.set_title(f'Source Plane; Flux = {fa:.2f}')  # Set title for subplot 1

# Image plane
ax = plt.subplot(122)
#ax.imshow(b, extent=(-xl, xl, -xl, xl), cmap='hot')
ax.imshow(b, cmap='hot', origin='lower', extent=(-xl, xl, -xl, xl))
fb = np.sum(b) * (xs**2)  # Flux on image plane (considering pixel size)
ax.set_title(f'Image plane; Flux = {fb:.2f}')  # Set title for subplot 2

plt.show()