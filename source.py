import numpy as np

from astropy.io import fits

#Parameters are location, size, structure

def gcirc(ny,rad,x1=0.0,y1=0.0): # Default position is 0,0
    x,y=np.mgrid[0:ny,0:ny]
    r2=(x-x1-ny/2)**2+(y-y1-ny/2)**2
    a=np.exp(-r2*0.5/rad**2)
    return a/a.sum()

def fitsim(filename):
    a=fits.getdata(filename) # Read the file
    if (len(a.shape) > 2): a=a[0] # Take first plane if there are many.
    return (1.0*a)/a.sum() # Return the normalized image