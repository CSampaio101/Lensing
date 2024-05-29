# This gives us the coordinates for the deflected ray at the source plane (y1,y2)
import numpy as np
def Point(x1,x2,x1l,x2l,ml): # Point lens of mass ml at x1l,x2l
    x1ml=(x1-x1l) # Distance along x axis of ray to lens position
    x2ml=(x2-x2l) # Distance along y axis of ray to lens position
    d=x1ml*x1ml+x2ml*x2ml+1.0e-12 # Distance between ray and lens squared
    y1=x1-ml*(x1-x1l)/d # Lens equation for x coordinate
    y2=x2-ml*(x2-x2l)/d # Lens equation for y coordinate
    return (y1,y2)

def TwoPoints(x1,x2,x1l1,x2l1,x1l2,x2l2,ml1,ml2): #Two diff lenses.
    x1ml1=(x1-x1l1)
    x2ml1=(x2-x2l1)
    x1ml2=(x1-x1l2)
    x2ml2=(x2-x2l2)
    d1=x1ml2*x1ml2+x2ml1*x2ml1+1.0e-12 # Add a tiny number to avoid division by zero
    d2=x1ml2*x1ml2+x2ml2*x2ml2+1.0e-12
    y1=x1-ml1*(x1-x1l1)/d1-ml2*(x1-x1l2)/d2 # Lens equation for x coordinate
    y2=x2-ml1*(x2-x2l1)/d1-ml2*(x2-x2l2)/d2
    return (y1,y2)

def ChangRefsdal(x1,x2,x1l,x2l,ml,k,g): #This one accounts for gravitational field
    x1ml=(x1-x1l)
    x2ml=(x2-x2l)
    d=x1ml*x1ml+x2ml*x2ml+1.0e-12 # Add a tiny number to avoid division by zero
    y1=x1*(1.0-k-g)-ml*(x1-x1l)/d
    y2=x2*(1.0-k+g)-ml*(x2-x2l)/d
    return (y1,y2)

def SIS(x1,x2,x1l,x2l,k): #This is the isothermal sphere
    x1ml=(x1-x1l)
    x2ml=(x2-x2l)
    d=np.sqrt(x1ml*x1ml+x2ml*x2ml+1.0e12) # Add a tiny number to void division by zero
    y1=x1-k*(x1-x1l)/d # Lens equation for x coordinate
    y2=x2-k*(x2-x2l)/d
    return (y1,y2)
