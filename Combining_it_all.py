import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, iv
import math as m
import gaussian_random_fields as gr  # Import the Gaussian random fields module

# phi parameters. These correspond to the lens.
phi_11 = 0.8
phi_22 = 0.0
phi_111 = 0.6
phi_112 = 0.5
phi_122 = 0.3
phi_222 = 0.5

g = 2 / (phi_11 ** 2 * phi_222)

def lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222, random_field_x1=None, random_field_x2=None):
    y1 = 0.5 * x1 ** 2 * phi_111 + x1 * x2 * phi_112 + x1 * phi_11 + 0.5 * x2 ** 2 * phi_122
    y2 = 0.5 * x1 ** 2 * phi_112 + x1 * x2 * phi_122 + 0.5 * x2 ** 2 * phi_222 + x2 * phi_22
    if random_field_x1 is not None and random_field_x2 is not None:
        y1 += random_field_x1
        y2 += random_field_x2
    return y1, y2

def generate_gaussian_random_field(alpha, size):
    return gr.gaussian_random_field(alpha, size, True)

def pixel_to_pos(pixel, origin, pixel_size):
    return origin + pixel * pixel_size

def pos_to_pixel(coord, origin, pixel_size):
    return np.rint((np.asarray(coord) - origin) / pixel_size).astype(int)

# Parameters
nx = 401  # Number of pixels in image plane
ny = 401  # Number of pixels in source plane
xl = 4.0  # Size of image plane covered (in "Einstein" radii)
yl = 4.0  # Size of source plane covered (in Einstein radii)

xs = 2.0 * xl / (nx - 1)  # Pixel size on the image map
ys = 2.0 * yl / (ny - 1)  # Pixel size on the source map

# Source parameters: Circular Gaussian Source
xpos = 0.0
rad = 0.01

# Loop over different ypos to vary the source distance from the caustic
ypos_values = np.linspace(-3.0, 3.0, 200)
magnifications = []
magnifications_field = []

# Generate the Gaussian random field
random_field_x1 = generate_gaussian_random_field(1, nx)  # Ensure the size matches the image plane
random_field_x2 = generate_gaussian_random_field(1, nx)  # Ensure the size matches the image plane

for ypos in ypos_values:
    jpos = pos_to_pixel(xpos, -yl, ys)
    ipos = pos_to_pixel(ypos, -yl, ys)
    rpix = int(round(rad / ys))

    # Create a circular Gaussian source
    x = np.linspace(-yl, yl, ny)
    y = np.linspace(-yl, yl, ny)
    X, Y = np.meshgrid(x, y)
    a_gaussian = np.exp(-((X - xpos)**2 + (Y - ypos)**2) / (2 * rad**2)) / (2.*np.pi*rad**2)
    
    b = np.zeros((nx, nx))  # Image plane for model without random field
    b_field = np.zeros((nx, nx))  # Image plane for model with random field

    # Grid for the image plane
    j1, j2 = np.mgrid[0:nx, 0:nx]
    x1 = pixel_to_pos(j2, -xl, xs)
    x2 = pixel_to_pos(j1, -xl, xs)

    # Deflect rays using the lens mapping without the Gaussian random field
    y1, y2 = lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222)
    
    # Deflect rays using the lens mapping with the Gaussian random field
    y1_field, y2_field = lens_mapping(x1, x2, phi_11, phi_22, phi_111, phi_112, phi_122, phi_222, 0.01 * random_field_x1, 0.01 * random_field_x2)
    
    # Convert deflected coordinates to source plane pixels (without random field)
    i2 = pos_to_pixel(y1, -yl, ys)
    i1 = pos_to_pixel(y2, -yl, ys)

    # Convert deflected coordinates to source plane pixels (with random field)
    i2_field = pos_to_pixel(y1_field, -yl, ys)
    i1_field = pos_to_pixel(y2_field, -yl, ys)

    # Filter valid rays hitting the source plane (without random field)
    ind = (i1 >= 0) & (i1 < ny) & (i2 >= 0) & (i2 < ny)
    i1n = i1[ind]
    i2n = i2[ind]
    j1in = j1[ind]
    j2in = j2[ind]

    # Filter valid rays hitting the source plane (with random field)
    ind_field = (i1_field >= 0) & (i1_field < ny) & (i2_field >= 0) & (i2_field < ny)
    i1n_field = i1_field[ind_field]
    i2n_field = i2_field[ind_field]
    j1in_field = j1[ind_field]
    j2in_field = j2[ind_field]

    # Calculate magnification by summing contributions from all rays (without random field)
    for i in range(np.size(i1n)):
        b[j1in[i], j2in[i]] += a_gaussian[i1n[i], i2n[i]]

    # Calculate magnification by summing contributions from all rays (with random field)
    for i in range(np.size(i1n_field)):
        b_field[j1in_field[i], j2in_field[i]] += a_gaussian[i1n_field[i], i2n_field[i]]

    fa_gaussian = np.sum(a_gaussian * ys * ys)  # Flux on source plane
    fb = np.sum(b) * (xs ** 2)  # Flux on image plane (without random field)
    fb_field = np.sum(b_field) * (xs ** 2)  # Flux on image plane (with random field)

    total_magnification = fb / fa_gaussian
    total_magnification_field = fb_field / fa_gaussian

    magnifications.append(total_magnification)
    magnifications_field.append(total_magnification_field)

# Define analytic expression
def gaussian_source_magnification(w):
    norm = np.pi / 2.
    if w > 0:
        return 0.5 * np.sqrt(w * np.pi / 2.) * np.exp(-0.5 * w ** 2) * kv(0.25, 0.5 * w ** 2) / norm
    else:
        return 0.25 * np.power(np.pi, 1.5) * np.sqrt(-w) * np.exp(-0.5 * w ** 2) * (iv(-0.25, 0.5 * w ** 2) + iv(0.25, 0.5 * w ** 2)) / norm

def asymptotic_source_magnification(w):
    if w > 0.:
        return 0.
    else:
        return 1. / np.sqrt(-w)

analytic = [np.sqrt(g / rad) * gaussian_source_magnification(-ypos / rad) for ypos in ypos_values]
asymptotic = [np.sqrt(g / rad) * asymptotic_source_magnification(-ypos / rad) for ypos in ypos_values]

# Plot the total magnification with and without Gaussian random field
plt.figure()
plt.plot(ypos_values, magnifications, 'b-', label='Magnification without Random Field')
plt.plot(ypos_values, magnifications_field, 'g-', label='Magnification with Random Field')
plt.plot(ypos_values, analytic, '--', label="Analytic solution for Gaussian Source")
plt.plot(ypos_values, asymptotic, 'r--', label="Asymptotic solution for Gaussian Source")

plt.xlabel('Distance from Caustic (ypos)')
plt.ylabel('Value')
plt.title('Total Magnification vs Vertical Distance from Caustic')
plt.legend()
plt.grid(True)
plt.show()
