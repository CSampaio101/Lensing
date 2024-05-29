import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l
nx=21 # Number of pixels in image plane
ny=21 # Number of pixels in source plane
xl=2. # Size of image plane covered (in "Einstein" radii)
yl=2. # Size of source plane covered (in Einstein radii)
# Lens parameters
xlens=0.0
ylens=0.0
mlens=1.0
xs=2.*xl/(nx-1) # pixel size on the mage map
ys=2.*yl/(ny-1) # pixel size on the source map
# Source parameters
xpos=0.0
ypos=0.0
rad=0.10
ipos=int(round(xpos/ys)) #Convert source parameters to pixels
jpos=int(round(-ypos/ys))
rpix=int(round(rad/ys))
a=s.gcirc(ny,rpix,jpos,ipos) # This is the source plane
b=np.zeros((nx,nx)) # This is the image plane
# In this version, the main loop is implicit
# We use operations on numpy arrays instead which is faster
j1,j2=np.mgrid[0:nx,0:nx]
x1=-xl+j2*xs # Pix to coord on image x
x2=-xl+j1*xs # Pix to coord on image y
y1,y2=l.SIS(x1,x2, xlens+0.1,ylens,1.2) # This line calculates the deflection
i2=np.round((y1+yl)/ys)
i1=np.round((y2+yl)/ys) # If deflected ray hits a pixel within source then set image
print("here",i1)
print("there",i2)
# to brightness on that pixel
ind= (i1>= 0) & (i1< ny) & (i2>= 0) & (i2< ny) # Now this is an array
print("boolean", ind)
# which is True if the ray
# hits the source plane.
i1n=i1[ind]
i2n=i2[ind]
j1in=j1[ind]
j2in=j2[ind]

for i in range(np.size(i1n)): # Loop over pixels that hit the source plane
    b[j1in[i],j2in[i]]=a[i1n[i],i2n[i]]

# Plot stuff including Fluxes in both plane
fig=plt.figure(1)
ax=plt.subplot(121)
ax.imshow(a,extent=(-yl,yl,-yl,yl))
fa=np.sum(a) # Flux on source plane
#ax.set_title(’Flux=’+str(fa)) # Set title for subplot 1
ax=plt.subplot(122)
ax.imshow(b,extent=(-xl,xl,-xl,xl))
fb=np.sum(b)*(xs**2)/(ys**2) # Flux on image pl. (taking into account pix size)
#ax.set_title(’Flux=’+str(fb)) # Set title for subplot 2
plt.show()