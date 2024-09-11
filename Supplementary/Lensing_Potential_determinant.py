import numpy as np
import matplotlib.pyplot as plt

# Parameters (replace with actual values as needed)
phi_11 = 7
phi_22 = 0
phi_111 = 6
phi_112 = 10
phi_122 = 3
phi_222 = 4

C = phi_111/2 - phi_122*phi_112 / phi_222 + phi_122**3 / (2 * phi_222**2)
B = phi_112/2 - phi_122**2 / (2 * phi_222)

def lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222):
    y1 = 0.5 * x1**2 * phi_111 + x1 * x2 * phi_112 + x1 * phi_11 + 0.5 * x2**2 * phi_122
    y2 = 0.5 * x1**2 * phi_112 + x1 * x2 * phi_122 + 0.5 * x2**2 * phi_222 + x2 * phi_22
    return y1, y2

def compute_jacobian(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222):
    A11 = x1 * phi_111 + x2 * phi_112 + phi_11
    A12 = x1 * phi_112 + x2 * phi_122
    A21 = x1 * phi_112 + x2 * phi_122
    A22 = x1 * phi_122 + x2 * phi_222 + phi_22
    return np.array([[A11, A12], [A21, A22]])

def is_near_caustic(x1, x2, threshold):
    distance = np.sqrt(x1**2 + x2**2)
    return distance < threshold

# Magnification map setup
ny = 401
yl = 2.0
b = np.zeros((ny, ny))
raypix = 15.0
sqrpix = np.sqrt(raypix)
sqrinpix = np.sqrt(1.0 / raypix)
ys = 2.0 * yl / (ny - 1)
xs = ys / sqrpix
xl = 2.0 * yl
nx = int(np.round(2 * xl / xs)) + 1
yr = np.arange(nx)
y, x = np.mgrid[0.0:nx, 0:nx]
percent0 = 5.0
percent = 5.0

# Threshold for near caustic region
caustic_threshold = 0.5

# Arrays to store source plane positions
source_plane_x = []
source_plane_y = []

# Array to store determinant of the Jacobian
determinant_A = np.zeros((nx, nx))

# Lens mapping integration
for i in range(nx):
    if ((i * 100 / nx) >= percent):
        percent += percent0
        print(round(i * 100 / nx), "%")

    for j in range(nx):
        x1 = -xl + y[i, j] * xs
        x2 = +xl - x[i, j] * xs

        # Compute lens mapping
        y1, y2 = lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222)

        # Store source plane positions
        source_plane_x.append(y1)
        source_plane_y.append(y2)

        # Compute the Jacobian matrix and its determinant
        A = compute_jacobian(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222)
        det_A = np.linalg.det(A)
        determinant_A[i, j] = det_A

        i1 = np.rint((y1 + yl) / ys).astype(int)
        i2 = np.rint((yl - y2) / ys).astype(int)

        ind = (i1 >= 0) & (i1 < ny) & (i2 >= 0) & (i2 < ny)

        i1n = i1[ind]
        i2n = i2[ind]

        for k in range(np.size(i1n)):
            b[i2n[k], i1n[k]] += 1

y += 1.0
b /= raypix

print(np.mean(b))

# Plot the magnification map on the image plane
plt.figure()
plt.imshow(b, vmin=0, vmax=15)
plt.title('Magnification Map on Image Plane')
plt.colorbar()
plt.show()

# Convert source plane positions to numpy arrays for plotting
source_plane_x = np.array(source_plane_x)
source_plane_y = np.array(source_plane_y)

# Plot the source plane
plt.figure()
plt.scatter(source_plane_x, source_plane_y, s=1)
plt.title('Source Plane Mapping')
plt.xlabel('y1')
plt.ylabel('y2')
plt.show()

# Plot the caustic
plt.figure()
plt.contour(x, y, determinant_A, levels=[0], colors='r')
plt.title('Caustic Curve')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
