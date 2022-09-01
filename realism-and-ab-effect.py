####################### Source code for the manuscript "Coherence and realism in the Aharonov-Bohm effect" #######################

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ABphase = np.pi/5 #Value of AB phase (or average AB phase) used throughout the manuscript
qK = (2./5)*ABphase #Associated value of q*K for the given ABphase



################## Functions used for the calculations ##################


###Variables that appear in multiple functions

#b -> base of logarithm

#angle -> theta

#vec -> array whose components are eigenstates of angular momentum present in the
#state of the cylinder, which is an even superposition of them with real positive components.
#For instance, vec=[2, 3] corresponds to the state (|2>+|3>)/sqrt(2). A classical flux can be
#simulated with a single component, which does not need to be integer. For instance, vec=[2.5]
#is associated with ABphase considered in the graphs in Section IV of the manuscript

#g -> function associated with the phase of the observable of interest

#g0 -> function associated with the relative phase of the state of the charge

#val -> constant value associated with operators of the type sigma_{f+val}^{ml}

#fixedml -> 'None' if operator is associated with multiple values of ml, like Sigma_x and Sigma_y.
#In case the operator is associated with a single value, say -1 or 2.5, we have fixedml=-1
#and fixedml=2.5, respectively.

#nat -> if 'local', rho=rho_S (i.e., the charge alone). If 'global', rho includes
#system R (i.e., the flux)


###Functions

#Ploting functions of theta. In cases of curves with discontinuity at theta=pi/2,
#the curve is plotted without the line connecting the two "disjoint parts"
def plot(xx,yy,color,linewidth,linestyle='solid'):
    if np.abs(yy[len(xx[xx<=np.pi/2])] - yy[len(xx[xx<=np.pi/2])-1]) <= 0.01:
        plt.plot(xx, yy, color=color, linewidth=linewidth, linestyle=linestyle)
    else:
        plt.plot(xx[xx<=np.pi/2], yy[0:len(xx[xx<=np.pi/2])], color=color, linewidth=linewidth, linestyle=linestyle)
        plt.plot(xx[xx>np.pi/2+.01], yy[len(xx[xx<=np.pi/2]):-1], color=color, linewidth=linewidth, linestyle=linestyle)

#Logarithm in a desirable base
def log(v,b):
    return np.log(v)/np.log(b)

#Shannon entropy for two-level systems
def h(v,b):
    return -v * log(v,b) - (1-v) * log(1-v,b)

#Auxiliary function for part of the calculation of entropies
def uc(angle, vec, g0, g, val, fixedml):
    temp = 0.
    if fixedml is None:
        for i in vec:
            temp += np.cos(g0(angle,i) - g(angle,i,val))
    else:
        for i in vec:
            temp += np.cos(g0(angle,i) - g(angle,fixedml,val))
    return (1 + temp/len(vec))/2

#Auxiliary function for diffrealism
def u(v):
    return (1 + np.cos(v))/2

#Entanglement entropy between the charge and the flux
def ent(angle, vec, g0, b):
    temp = 0.
    for i in vec:
        temp += np.exp(-1j*(g0(angle,i)))
    varlambda = (1./2) * (1 - np.abs(temp)/len(vec))
    sol = np.array([0.0 for i in range(len(varlambda))])
    for i in range(len(varlambda)):
        if varlambda[i]==0 or varlambda[i]==1:
            sol[i] = 0.0
        else:
            sol[i] = h(varlambda[i],b)
    return sol

#von Neumann entropy of the system after a dephasing map (i.e., map Phi_O)
def entmap(angle, vec, g0, g, val, b, fixedml=None, nat='local'):
    if nat == 'local':
        varlambda = uc(angle, vec, g0, g, val, fixedml)
        temp = 1
    else:
        temp = len(vec)
    sol = np.array([0.0 for i in range(len(angle))])
    for i in range(temp):
        if nat == 'global':
            varlambda = uc(angle, [vec[i]], g0, g, val, None)
        for j in range(len(varlambda)):
            if varlambda[j]==0 or varlambda[j]==1:
                sol[j] += 0.0
            else:
                sol[j] += h(varlambda[j], b)
    return sol/temp

#Difference of realism, as defined in Eq. (15)
def diffrealism(val, phase, b):
    varlambda0, varlambda1 = u(-val), u(phase-val)
    if varlambda0==0 or varlambda0==1:
        irreal0 = 0.0
    else:
        irreal0 = h(varlambda0,b)
    if varlambda1==0 or varlambda1==1:
        irreal1 = 0.0
    else:
        irreal1 = h(varlambda1,b)
    return irreal0 - irreal1

#Function f in Eq. (6). Here, we set f(theta)=theta/3
def f(v):
    return v/3


###Functions used for the definition of the desirable relative phases
#Phase associated with sigma_x
def xf(angle, ml=None, val=None):
    return 0.
#Phase associated with sigma_y
def yf(angle, ml=None, val=None):
    return np.pi/2
#Phase associated with the state in Eq. (9) in the Coulomb gauge
def fa0(angle, ml):
    return f(angle) + qK*ml*angle/np.pi
#Phase associated with sigma_{f+val}^{ml} using the notation set in Eq. (11)
def fa(angle, ml, val):
    return np.piecewise(angle, [angle <= np.pi/2, angle > np.pi/2],
           [lambda w: f(w) + qK*ml*w/np.pi + val, lambda w: f(w) - qK*ml*(np.pi - w)/np.pi + val])





dimS = 2 #Dimension of system S, i.e., the trajectory of the charge

theta = np.arange(0, np.pi, 0.01)



################## No flux ##################



z0 = 0*theta #Realism of sigma_z


#Realism of sigma_x, sigma_y, and sigma_f

x0 = 1 - entmap(theta,[0],fa0,xf,0,dimS)
y0 = 1 - entmap(theta,[0],fa0,yf,0,dimS)
f0 = 1 - entmap(theta,[0],fa0,fa,0,dimS)

disc = len(theta[theta<=np.pi/2])

plot(theta,z0,'indigo',3.0,'dashed')
plot(theta,x0,'xkcd:goldenrod',3.0)
plot(theta,y0,'xkcd:green',3.0,'dashdot')
plot(theta,f0,'crimson',3.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.yticks(np.arange(0, 1+.01, step=0.5))

plt.gca().set_aspect(3)
plt.tight_layout()

plt.savefig('fig1.svg', format='svg', facecolor=None, dpi=300)




################## Classical flux ##################

####### Realism of sigma_z, sigma_x^A, and sigma_y^A #######

x = 1 - entmap(theta,[2.5],fa0,fa,0,dimS) #sigma_x^A
y = 1 - entmap(theta,[2.5],fa0,fa,np.pi/2,dimS) #sigma_y^A

plt.figure()

plot(theta,x,'xkcd:blue',3.0,(0,(2, 3, 1, 3)))
plot(theta,z0,'xkcd:beige',3.0)
plot(theta,y,'xkcd:teal',3.0,'dashed')

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/4)), ['0','π/4','π/2','3π/4','π'])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.0,0.01)
plt.gca().set_aspect(1.2)
plt.tight_layout()

plt.savefig('fig2.svg', format='svg', facecolor=None, dpi=300)



####### Change in realism #######

delta = np.linspace(0, np.pi/2, num=80, endpoint=True)
phaseAB = np.linspace(0, np.pi, num=20, endpoint=True)

DeltaR = np.zeros((len(phaseAB), len(delta)))

for j in range(len(phaseAB)):
    for i in range(len(delta)):
        DeltaR[j,i] = diffrealism(delta[i],phaseAB[j],dimS)

DR = np.zeros(len(delta))
for i in range(len(delta)):
    DR[i] = diffrealism(delta[i],np.pi/5,dimS)

plt.figure()

colors = plt.cm.autumn(phaseAB*(1./np.pi))
for i in range(len(phaseAB)):
    plt.plot(delta, DeltaR[i], color=colors[i])

plt.plot(delta, DR, color='xkcd:violet', linewidth=3.0)

plt.xticks(np.arange(0, np.pi/2+.01, step=(np.pi/4)), ['0','π/4','π/2'])
plt.yticks(np.arange(-1, 1.01, step=(1.)))

plt.margins(0.0,0.01)
plt.gca().set_aspect(0.65)
plt.tight_layout()

normalize = mcolors.Normalize(vmin=0, vmax=np.pi)
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap='autumn')
scalarmappaple.set_array(phaseAB)
plt.colorbar(scalarmappaple, orientation='horizontal', ticks=[0, np.pi/2, np.pi])


plt.savefig('fig3.svg', format='svg', facecolor=None, dpi=300)




################## Quantized flux: local realism ##################

####### "Realism" of sigma_z for the reduced state of the charge #######

z0 = ent(theta,[2.5],fa0,dimS)
z1 = ent(theta,[2,3],fa0,dimS)
z2 = ent(theta,[1,2,3,4],fa0,dimS)
z3 = ent(theta,[0,1,2,3,4,5],fa0,dimS)
z4 = ent(theta,[-1,0,1,2,3,4,5,6],fa0,dimS)

plt.figure()

plot(theta,z0,'xkcd:blue',6.0)
plot(theta,z1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,z2,'xkcd:wheat',6.0,'dashdot')
plot(theta,z3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,z4,'xkcd:tomato',6.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.ylim([-0.02, 1.02])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig4.svg', format='svg', facecolor=None, dpi=300)



####### "Realism" of sigma_x for the reduced state of the charge #######

x0 = 1. + z0 - entmap(theta,[2.5],fa0,xf,0,dimS)
x1 = 1. + z1 - entmap(theta,[2,3],fa0,xf,0,dimS)
x2 = 1. + z2 - entmap(theta,[1,2,3,4],fa0,xf,0,dimS)
x3 = 1. + z3 - entmap(theta,[-0,1,2,3,4,5],fa0,xf,0,dimS)
x4 = 1. + z4 - entmap(theta,[-1,0,1,2,3,4,5,6],fa0,xf,0,dimS)

plt.figure()

plot(theta,x0,'xkcd:blue',6.0)
plot(theta,x1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,x2,'xkcd:wheat',6.0,'dashdot')
plot(theta,x3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,x4,'xkcd:tomato',6.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.ylim([-0.02, 1.02])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig5.svg', format='svg', facecolor=None, dpi=300)




####### "Realism" of sigma_y for the reduced state of the charge #######

y0 = 1. + z0 - entmap(theta,[2.5],fa0,yf,0,dimS)
y1 = 1. + z1 - entmap(theta,[2,3],fa0,yf,0,dimS)
y2 = 1. + z2 - entmap(theta,[1,2,3,4],fa0,yf,0,dimS)
y3 = 1. + z3 - entmap(theta,[-0,1,2,3,4,5],fa0,yf,0,dimS)
y4 = 1. + z4 - entmap(theta,[-1,0,1,2,3,4,5,6],fa0,yf,0,dimS)


plt.figure()

plot(theta,y0,'xkcd:blue',6.0)
plot(theta,y1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,y2,'xkcd:wheat',6.0,'dashdot')
plot(theta,y3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,y4,'xkcd:tomato',6.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.ylim([-0.02, 1.02])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig6.svg', format='svg', facecolor=None, dpi=300)




####### "Realism" of sigma_f^{A_{val_ml}} for the reduced state of the charge #######

val_ml = -1
f0 = 1. + z0 - entmap(theta,[2.5],fa0,fa,0,dimS,val_ml)
f1 = 1. + z1 - entmap(theta,[2,3],fa0,fa,0,dimS,val_ml)
f2 = 1. + z2 - entmap(theta,[1,2,3,4],fa0,fa,0,dimS,val_ml)
f3 = 1. + z3 - entmap(theta,[-0,1,2,3,4,5],fa0,fa,0,dimS,val_ml)
f4 = 1. + z4 - entmap(theta,[-1,0,1,2,3,4,5,6],fa0,fa,0,dimS,val_ml)


plt.figure()

plot(theta,f0,'xkcd:blue',6.0)
plot(theta,f1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,f2,'xkcd:wheat',6.0,'dashdot')
plot(theta,f3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,f4,'xkcd:tomato',6.0,(0,(1,1)))

plt.ylim([-0.02, 1.02])
plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig7.svg', format='svg', facecolor=None, dpi=300)




################## Quantized flux: global realism ##################

l = 6
dimR = 2*l+1
dimSR = 2*dimR
ref = [log(dimR,dimSR) for i in theta]


####### Realism of Sigma_x #######

globalX0 = 1. - log(1,dimSR) - entmap(theta,[2.5],fa0,fa,0,dimSR,nat='global')
globalX1 = 1. - log(2,dimSR) - entmap(theta,[2,3],fa0,fa,0,dimSR,nat='global')
globalX2 = 1. - log(4,dimSR) - entmap(theta,[1,2,3,4],fa0,fa,0,dimSR,nat='global')
globalX3 = 1. - log(6,dimSR) - entmap(theta,[-0,1,2,3,4,5],fa0,fa,0,dimSR,nat='global')
globalX4 = 1. - log(8,dimSR) - entmap(theta,[-1,0,1,2,3,4,5,6],fa0,fa,0,dimSR,nat='global')


plt.figure()

#Region where the realism of observables of the charge alone (local observables) lie
plt.fill_between(theta, ref, 1, color='xkcd:silver', alpha=.5)

plot(theta,globalX0,'xkcd:blue',6.0)
plot(theta,globalX1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,globalX2,'xkcd:wheat',6.0,'dashdot')
plot(theta,globalX3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,globalX4,'xkcd:tomato',6.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.ylim([-0.02, 1.02])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig8.svg', format='svg', facecolor=None, dpi=300)


####### Realism of Sigma_y #######

globalY0 = 1. - log(1,dimSR) - entmap(theta,[2.5],fa0,fa,np.pi/2,dimSR,nat='global')
globalY1 = 1. - log(2,dimSR) - entmap(theta,[2,3],fa0,fa,np.pi/2,dimSR,nat='global')
globalY2 = 1. - log(4,dimSR) - entmap(theta,[1,2,3,4],fa0,fa,np.pi/2,dimSR,nat='global')
globalY3 = 1. - log(6,dimSR) - entmap(theta,[-0,1,2,3,4,5],fa0,fa,np.pi/2,dimSR,nat='global')
globalY4 = 1. - log(8,dimSR) - entmap(theta,[-1,0,1,2,3,4,5,6],fa0,fa,np.pi/2,dimSR,nat='global')

plt.figure()

#Region where the realism of observables of the charge alone (local observables) lie
plt.fill_between(theta, ref, 1, color='xkcd:silver', alpha=.5)

plot(theta,globalY0,'xkcd:blue',6.0)
plot(theta,globalY1,'xkcd:teal',6.0,(0,(6,1.5)))
plot(theta,globalY2,'xkcd:wheat',6.0,'dashdot')
plot(theta,globalY3,'xkcd:pink',6.0,(0,(3,1,1,1,1,1)))
plot(theta,globalY4,'xkcd:tomato',6.0,(0,(1,1)))

plt.xticks(np.arange(0, np.pi+.01, step=(np.pi/2)), ['0','π/2','π'])
plt.ylim([-0.02, 1.02])
plt.yticks(np.arange(0, 1.1, step=(0.5)))

plt.margins(0.,0.)
plt.gca().set_aspect(np.pi)
plt.tight_layout()

plt.savefig('fig9.svg', format='svg', facecolor=None, dpi=300)
