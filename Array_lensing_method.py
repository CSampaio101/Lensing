import numpy as np
import matplotlib.pyplot as plt
import source as s
import lens as l
nx = 401 # Number of pixels in image plane
ny = 401 # Number of pixels in source plane
xl = 2. # Size of image plane covered (in "Einstein" radii)
yl = 2. # Size of source plane covered (in Einstein radii)
# Lens parameters
xlens = 0.5
ylens = 0.
mlens = 0.5

xlens2 = -.5
ylens2 = 0.
mlens2 = 0.5

xs = 2.*xl/(nx-1) # pixel size on the mage map
ys = 2.*yl/(ny-1) # pixel size on the source map

# Source parameters
xpos=0.05
ypos=0.4
rad=0.10
ipos=int(round((yl+xpos)/ys)) #Convert source parameters to pixels
jpos=int(round((yl-ypos)/ys))
rpix=int(round(rad/ys))
a=s.gcirc(ny,rpix,jpos,ipos) # This is the source plane
b=np.zeros((nx,nx)) # This is the image plane
# In this version, the main loop is implicit
# We use operations on numpy arrays instead which is faster

j1,j2=np.mgrid[0:nx,0:nx]
x1=-xl+j2*xs # Pix to coord on image x
x2=+xl-j1*xs # Pix to coord on image y
y1,y2=l.TwoPoints(x1,x2,xlens,ylens,xlens2,ylens2,mlens,mlens2) # Deflect X,Y coordinates
#y1,y2=y1, y2 = l.Point(x1, x2, xlens, ylens, mlens)
#y1,y2=l.SIS(x1,x2, xlens+0.1,ylens,1.2) # This line calculates the deflection
i2=np.rint((y1+yl)/ys)
i2=i2.astype(int) #NOTE: This copies the previous array entirely which is inefficient, but this is needed to satisfy the for loop below.
i1=np.rint((yl-y2)/ys) # If deflected ray hits a pixel within source then set image
i1=i1.astype(int) 
# to brightness on that pixel
print("i1",i1)
ind= (i1>= 0) & (i1< ny) & (i2>= 0) & (i2< ny) # Now this is an array
# which is True if the ray
# hits the source plane.
i1n=i1[ind]
i2n=i2[ind]
j1in=j1[ind]
j2in=j2[ind]
print("ind", ind)
print("j1in",j1in, "j2in", j2in, "i1n",i1n, "i2n", i2n)
print("size", np.size(i1n))
for i in range(np.size(i1n)): # Loop over pixels that hit the source plane
    #print("here",j1in[i], j2in[i],i1n[i],i2n[i])
    b[j1in[i],j2in[i]]=a[i1n[i],i2n[i]]
    
# Plot stuff including Fluxes in both plane
fig=plt.figure(1)
ax=plt.subplot(121)
ax.imshow(a,extent=(-yl,yl,-yl,yl), cmap='hot')
fa=np.sum(a) # Flux on source plane
ax.set_title('Flux='+str(fa)) # Set title for subplot 1
ax=plt.subplot(122)
ax.imshow(b,extent=(-xl,xl,-xl,xl), cmap='hot')
fb=np.sum(b)*(xs**2)/(ys**2) # Flux on image pl. (taking into account pix size)
ax.set_title('Flux='+str(fb)) # Set title for subplot 2
plt.show()