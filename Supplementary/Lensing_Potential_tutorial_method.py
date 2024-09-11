#NOTE: This code has an issue where the magnification does not increase near the caustic. I am not sure what's causing this just yet.

import numpy as np
import matplotlib.pyplot as plt

# Parameters (replace with actual values as needed)
phi_11 = 0.8
phi_22 = 0.0
phi_111 = 0.6
phi_112 = 0.05
phi_122 = 0.03
phi_222 = 0.05



C = phi_111 / 2 - phi_122 * phi_112 / phi_222 + phi_122 ** 3 / (2 * phi_222 ** 2)
B = phi_112 / 2 - phi_122 ** 2 / (2 * phi_222)

def lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222):
    y1 = 0.5 * x1 ** 2 * phi_111 + x1 * x2 * phi_112 + x1 * phi_11 + 0.5 * x2 ** 2 * phi_122
    y2 = 0.5 * x1 ** 2 * phi_112 + x1 * x2 * phi_122 + 0.5 * x2 ** 2 * phi_222 + x2 * phi_22
    return y1, y2

def compute_jacobian(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222):
    A11 = x1 * phi_111 + x2 * phi_112 + phi_11
    A12 = x1 * phi_112 + x2 * phi_122
    A21 = x1 * phi_112 + x2 * phi_122
    A22 = x1 * phi_122 + x2 * phi_222 + phi_22
    return np.array([[A11, A12], [A21, A22]])

def compute_determinant_jacobian(x1, x2):
    A11 = x1 * phi_111 + x2 * phi_112 + phi_11
    A12 = x1 * phi_112 + x2 * phi_122
    A21 = x1 * phi_112 + x2 * phi_122
    A22 = x1 * phi_122 + x2 * phi_222 + phi_22
    det_A = A11 * A22 - A12 * A21
    return det_A

# Magnification map setup
#ny enhances the resolution of the image.
ny = 801
#yl affects the range at which you're looking at. As it increases, more of the magnification/critical curve is shown.
yl = 3
b = np.zeros((ny, ny))
raypix = 25.0
sqrpix = np.sqrt(raypix)
ys = 2.0 * yl / (ny - 1)
xs = ys / sqrpix
xl = 2.0 * yl
nx = int(np.round(2 * xl / xs)) + 1
x1, x2 = np.mgrid[-xl:xl:nx*1j, -xl:xl:nx*1j]

# x2 = -x1 * phi_122 / phi_222

print(np.shape(x1), np.shape(x2))

# Calculate the lens mapping
y1, y2 = lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222)

# Calculate the determinant of the Jacobian
det_A = compute_determinant_jacobian(x1, x2)

# Magnification is the inverse of the determinant of the Jacobian
threshold = 1e-4
with np.errstate(divide='ignore', invalid='ignore'):
    magnification = np.where(np.abs(det_A) > threshold, 1.0 / np.abs(det_A), 1.0 / threshold)

print("Shape of det_A:", np.shape(det_A))
print("Non-zero entries in det_A:", np.count_nonzero(det_A == 0))

# print(np.where(det_A == 0))

# Calculate source plane coordinates
i1 = np.rint((y1 + yl) / ys).astype(int)
i2 = np.rint((yl - y2) / ys).astype(int)

ind = (i1 >= 0) & (i1 < ny) & (i2 >= 0) & (i2 < ny)
i1 = i1[ind]
i2 = i2[ind]

# Populate the magnification map
np.add.at(b, (i2, i1), magnification[ind])

# Normalize the magnification map
b /= raypix

print(np.mean(b))

# Plot the magnification map on the image plane
plt.figure()
plt.imshow(b, extent=(-xl, xl, -4 * yl, 4 * yl))
plt.title('Magnification Map on Image Plane')
plt.colorbar()

# Overlay the caustic curve
plt.contour(x1, x2, det_A, levels=[0], colors='r')
plt.title('Magnification Map on Image Plane with Caustic Curve')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# # Convert source plane positions to numpy arrays for plotting
# source_plane_x = y1[ind]
# source_plane_y = y2[ind]

# # Plot the source plane
# plt.figure()
# plt.scatter(source_plane_x, source_plane_y, s=1)
# plt.title('Source Plane Mapping')
# plt.xlabel('y1')
# plt.ylabel('y2')
# plt.show()
