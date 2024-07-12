import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l

# Parameters
nx = 401  # Number of pixels in image plane
ny = 401  # Number of pixels in source plane
xl = 2.0  # Size of image plane covered (in "Einstein" radii)
yl = 2.0  # Size of source plane covered (in Einstein radii)

# Lens parameters
xlens = 0.0
ylens = 0.0
mlens = 1.0

xs = 2.0 * xl / (nx - 1)  # Pixel size on the image map
ys = 2.0 * yl / (ny - 1)  # Pixel size on the source map

# Source parameters: Circular Gaussian Source
xpos = 0.0
ypos = 1.0
rad = 0.1
ipos = int(round((yl + xpos) / ys))  # Convert source parameters to pixels
jpos = int(round((yl - ypos) / ys))  # Correct y-coordinate handling
print("Source position:", ipos, jpos)
rpix = int(round(rad / ys))

# Create a circular Gaussian source
x = np.linspace(-yl, yl, ny)
y = np.linspace(-yl, yl, ny)
X, Y = np.meshgrid(x, y)
a = np.exp(-((X - xpos)**2 + (Y - ypos)**2) / (2 * rad**2))

b = np.zeros((nx, nx))  # Image plane

# Implicit main loop: Operations on numpy arrays (faster)
j1, j2 = np.mgrid[0:nx, 0:nx]
x1 = -xl + j2 * xs  # Convert pixels to coordinates on image x
x2 = +xl - j1 * xs  # Convert pixels to coordinates on image y

# Deflect rays using the Point lens model
y1, y2 = l.Point(x1, x2, xlens, ylens, mlens)

# Convert deflected coordinates to source plane pixels
i2 = np.rint((y1 + yl) / ys).astype(int)
i1 = np.rint((yl - y2) / ys).astype(int)

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
ax.imshow(a, extent=(-yl, yl, -yl, yl), cmap='hot')
fa = np.sum(a)  # Flux on source plane
ax.set_title(f'Flux = {fa:.2f}')  # Set title for subplot 1

# Image plane
ax = plt.subplot(122)
ax.imshow(b, extent=(-xl, xl, -xl, xl), cmap='hot')
fb = np.sum(b) * (xs**2) / (ys**2)  # Flux on image plane (considering pixel size)
ax.set_title(f'Flux = {fb:.2f}')  # Set title for subplot 2

plt.show()

# Using the parameters from the PDF for magnifications
mu_plus_val = 1.17  # Calculated earlier
mu_minus_val = 0.17  # Calculated earlier

print(f'Magnification factors: mu+ = {mu_plus_val:.2f}, mu- = {mu_minus_val:.2f}')
