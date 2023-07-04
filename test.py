import numpy as np
from scipy.special import gamma, zeta
#from sympy.ntheory.factor_ import totient
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def _x(theta,phi,r_1,r_2):
    return (r_2 + r_1 * np.cos(phi)) * np.cos(theta)

def _y(theta,phi,r_1,r_2):
    return (r_2 + r_1 * np.cos(phi)) * np.sin(theta)

def _z(phi,r_1):
    return r_1 * np.sin(phi)

class torus_plotter:
    def __init__(self,r_1,r_2,division_parameter=4,is_ticks=False):
        self.r_1 = r_1
        self.r_2 = r_2

def make_torus_plot(r_1,r_2,division_parameter=4,is_ticks=False):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    ax.axes.set_xlim3d(left=-(r_1 + r_2), right=(r_1 + r_2))
    ax.axes.set_ylim3d(bottom=-(r_1 + r_2), top=(r_1 + r_2))
    ax.axes.set_zlim3d(bottom=-(r_1 + r_2), top=(r_1 + r_2))
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)

    theta = np.linspace(-np.pi,np.pi,65)
    phi = np.linspace(-np.pi,np.pi,65)

    ang_divs = 2**division_parameter
    k = 2 * np.pi / ang_divs
    ang = -np.pi
    for i in range(ang_divs):
        x_theta = _x(ang,phi,r_1,r_2)
        y_theta = _y(ang,phi,r_1,r_2)
        z_theta = _z(phi,r_1)
        x_phi = _x(theta,ang,r_1,r_2)
        y_phi = _y(theta,ang,r_1,r_2)
        z_phi = _z(ang,r_1)
        if ang == 0:
            ax.plot(x_theta,y_theta,z_theta, color='dimgray', linewidth=1.2)
            ax.plot(x_phi,y_phi,z_phi, color='dimgray', linewidth=1.2)
        elif ang == -np.pi / 2 or ang == np.pi / 2 or ang == -np.pi:
            ax.plot(x_theta,y_theta,z_theta, color='darkgray', linewidth=1.2)
            ax.plot(x_phi,y_phi,z_phi, color='darkgray', linewidth=1.2)
        else:
            ax.plot(x_theta,y_theta,z_theta, color='lightgray', linewidth=1)
            ax.plot(x_phi,y_phi,z_phi, color='lightgray', linewidth=1)
        ang += k

    if is_ticks == True:
        ax.text(_x(0,0,r_1,r_2),_y(0,0,r_1,r_2),_z(0,r_1),'0',None)
        ax.text(_x(0,np.pi/2,r_1,r_2),_y(0,np.pi/2,r_1,r_2),_z(np.pi/2,r_1),r_1,None)
        ax.text(_x(0,-np.pi/2,r_1,r_2),_y(0,-np.pi/2,r_1,r_2),_z(-np.pi/2,r_1),-r_1,None)
        ax.text(_x(0,np.pi,r_1,r_2),_y(0,np.pi,r_1,r_2),_z(np.pi,r_1),'∞',None)
        ax.text(_x(np.pi/2,0,r_1,r_2),_y(np.pi/2,0,r_1,r_2),_z(0,r_1),r_2,None)
        ax.text(_x(-np.pi/2,0,r_1,r_2),_y(-np.pi/2,0,r_1,r_2),_z(0,r_1),-r_2,None)
        ax.text(_x(np.pi,0,r_1,r_2),_y(np.pi,0,r_1,r_2),_z(0,r_1),'∞',None)

    return fig, ax

def torus_plot(x,y):
    pass

#Torus radii
r_1 = 3 #Inner
r_2 = 10 #Outer

#Torus domain
theta = np.linspace(-np.pi,np.pi,65)
phi = np.linspace(-np.pi,np.pi,65)
T, P = np.meshgrid(theta,phi)

#Cartesian
X1 = (r_2 + r_1 * np.cos(P)) * np.cos(T)
Y1 = (r_2 + r_1 * np.cos(P)) * np.sin(T)
Z1 = r_1 * np.sin(P)

fig,ax = make_torus_plot(r_1,r_2,is_ticks=True)




samples = 2000
x_0, y_0 = 0, 0
theta_in = np.linspace(-np.pi,np.pi,samples)
x_in = np.array(x_0 + (r_1 + r_2) * np.tan(theta_in / 2))
Theta = np.empty(samples)
Phi = np.empty(samples)
X = np.empty(samples)
Y = np.empty(samples)
Z = np.empty(samples)

for i in range(samples):
    try:
        y = (x_in[i])
    except:
        y = 10**16
    Phi[i] = 2 * np.arctan((y - y_0) / r_1)
    R = r_2 + r_1 * np.cos(Phi[i])
    Theta[i] = 2 * np.arctan((x_in[i] - x_0) / R)
    X[i] = (r_2 + r_1 * np.cos(Phi[i])) * np.cos(Theta[i])
    Y[i] = (r_2 + r_1 * np.cos(Phi[i])) * np.sin(Theta[i])
    Z[i] = r_1 * np.sin(Phi[i])

ax.plot(X,Y,Z, color='r', linewidth=3.0)


"""
Theta1 = np.empty(samples)
Phi1 = np.empty(samples)
X2 = np.empty(samples)
Y2 = np.empty(samples)
Z2 = np.empty(samples)

for i in range(samples):
    y = -np.log(x_in[i])
    Phi1[i] = 2 * np.arctan((y - y_0) / r_1)
    R = r_2 + r_1 * np.cos(Phi1[i])
    Theta1[i] = 2 * np.arctan((x_in[i] - x_0) / R)
    X2[i] = (r_2 + r_1 * np.cos(Phi1[i])) * np.cos(Theta1[i])
    Y2[i] = (r_2 + r_1 * np.cos(Phi1[i])) * np.sin(Theta1[i])
    Z2[i] = r_1 * np.sin(Phi1[i])

ax.plot(X2,Y2,Z2, color='r', linewidth=5.0)
"""

plt.show()
