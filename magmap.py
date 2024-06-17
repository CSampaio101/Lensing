import numpy as np
import lens as l
import matplotlib.pyplot as plt
ny=401
yl=2.
b=np.zeros((ny,ny))
raypix=15. # This is the number of rays per pixel in absence of lensing.
sqrpix=np.sqrt(raypix) # Rays per pixel square root (rays/pix in one dir)
sqrinpix=np.sqrt(1./raypix)
ys=2.*yl/(ny-1) # Pixel size on source plane
xs=ys/sqrpix # Side of the square area transported back by a ray.
xl=2.*yl # Size of the shooting region at the mage plane
# BEWARE. This may need to be larger for certain lens models!!!!
nx=np.round(2*xl/xs)+1 # Number of rays on a column/row at the image plane
yr=np.arange(nx) # This is an array with pixels on y direction
y, x=np.mgrid[0.0:nx,0:nx] #NOTE I fixed the mgrid, as it initially was 0:1, 0:nx. This produced nothign. Grid with pixel coordinates for a row at the image
percent0=5. # Percentage step for printing progress
percent=5. # Initial value for perc
for i in yr: # Loop over rows or rays
    if ((i*100/nx)>=percent): # Check if we have already completed perc.
        percent=percent+percent0 # Increase perc.
        print(round(i*100/nx),"% ") # Print progress
x1=-xl+y*xs # Convert pixels to coordinates in the image plane
x2=+xl-x*xs
y1,y2=l.TwoPoints(x1,x2,-0.75,0.,0.75,0.,0.5,2.5) # Deflect rays.
# We can set another lens model here easily.
i1=np.rint((y1+yl)/ys) # Convert coords to pixels at the source plane
i2=np.rint((yl-y2)/ys)
i1=i1.astype(int)  # Make pixel corrds integer number
i2=i2.astype(int)
ind=(i1>=0) & (i1<ny) & (i2>=0) & (i2<ny) # Indices of rays falling into our source plane
#print("ind",ind)
print("i1", i1, "i2", i2)
i1n=i1[ind] # Coordinates of pixels hitting our source plane
i2n=i2[ind]
index=0
print("size",np.size(i1n))
for i in range(np.size(i1n)): # Loop over hits "on target"
    b[i2n[i],i1n[i]]+=1 # Increase magnification at those pixels
    #index+=1
    #print("index",index)
y=y+1.0 # Increase the y coordinate of the pixel/rays
b=b/raypix # Normalize magnfication with N r
print(np.mean(b)) # Print mean magnification
plt.imshow(b,vmin=0,vmax=15) # Show image
plt.show()