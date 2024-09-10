# Copyright 2017 Bruno Sciolla. All Rights Reserved.
# ==============================================================================
# Generator for 2D scale-invariant Gaussian Random Fields
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import power_spectrum as ps
# Main dependencies
import numpy
import scipy.fftpack
import power_spectrum as ps

#Parameters needed for power spectrum here
params = {'M':1e-6, 'c': 10, 'DL': 1.35, 'DS':1.79, 'DLS':0.95, 'dm_mass_fraction':1}

def fftind(size, domain_size):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, numpy array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
                
        Example:
        
            print(fftind(5))
            
            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]
            
        """
    k_ind = numpy.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    k_ind = k_ind * (2*numpy.pi / domain_size)
    return( k_ind )



def gaussian_random_field(size = 128, 
                          flag_normalize = True,
                          domain_size = 1):
    """ Returns a numpy array of shifted Fourier coordinates k_x k_y.
        
        Input args:
            alpha (double, default = 3.0): 
                The power of the power-law momentum distribution
            size (integer, default = 128):
                The size of the square output Gaussian Random Fields
            flag_normalize (boolean, default = True):
                Normalizes the Gaussian Field:
                    - to have an average of 0.0
                    - to have a standard deviation of 1.0

        Returns:
            gfield (numpy array of shape (size, size)):
                The random gaussian random field
                
        Example:
        import matplotlib
        import matplotlib.pyplot as plt
        example = gaussian_random_field()
        plt.imshow(example)
        """
        
        # Defines momentum indices
    k_idx = fftind(size, domain_size)
    #print("shape", numpy.shape(k_idx))
        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    
    k_magnitude = numpy.sqrt(k_idx[0]**2 + k_idx[1]**2 + 1e-10)
    #amplitude = numpy.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 ) #Note: This is the original amplitude function
    
    #This new amplitude is obtained from our calculated power spectrum
    amplitude = [ps.Pkappa_angular(q, params) for q in k_magnitude.flatten()]
    amplitude = numpy.array(amplitude).reshape(size, size)
    amplitude[0, 0] = 0  # Set the zero frequency component to zero
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = numpy.random.normal(size = (size, size)) \
        + 1j * numpy.random.normal(size = (size, size))
    
        # To real space. Note, this is done twice since the delta alpha is a vector, based on which component of k you use..
    gfield_x1 = numpy.fft.ifft2((2 * 1j * numpy.pi * k_idx[0] / k_magnitude**2 ) * noise * amplitude).real
    gfield_x2 = numpy.fft.ifft2((2 * 1j * numpy.pi * k_idx[1] / k_magnitude**2 ) * noise * amplitude).real
        # Sets the standard deviation to one
    if flag_normalize:
        gfield_x1 = gfield_x1 - numpy.mean(gfield_x1)
        gfield_x2 = gfield_x2 - numpy.mean(gfield_x2)
        gfield_x1 = gfield_x1/numpy.std(gfield_x1)
        gfield_x2 = gfield_x2/numpy.std(gfield_x2)    
    return gfield_x1, gfield_x2




def main():
    import matplotlib
    import matplotlib.pyplot as plt
    example = gaussian_random_field()
    # plt.imshow(example, cmap='gray')
    # plt.show()
    
if __name__ == '__main__':
    main()