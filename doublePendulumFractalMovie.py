
import random
import numpy as np
import png
import ColorFunctions as cf

gL1=1.#FFFFFF
gL2=1.
gM1=1.
gM2=1.
gG=9.81



def writeImage( fName, colorTensor ):
    p = [
        [ int( 255 * x ) for element in row for x in element ]
        for row in colorTensor
    ]
    w = png.Writer( size, size, greyscale=False )
    with open( fName, 'wb') as f:
        w.write( f, p )

def ePot1( x ):
    global gL1, gL2, gM1, gM2, gG
    return gG * ( -np.cos( x[2] ) * gL1 * gM1 )

def ePot2( x ):
    global gL1,gL2, gM1,gM2,gG
    return -gG * ( np.cos( x[2] ) + np.cos( [x[4]] ) ) * gL2 * gM2

def ePotTot( x ):
    return ePot1( x ) + ePot2( x )

def eKin1( x ):
    global gL1, gL2, gM1, gM2, gG
    return 0.5 * gM1 * gL1**2 * x[1]**2

def eKin2( x ):
    global gL1,gL2, gM1,gM2,gG
    return 0.5 * gM2 * (gL1**2 * x[1]**2 + gL2**2 * x[3]**2 + 2.* gL1*gL2 *x[1]*x[3]*np.cos(x[4]-x[2]))

def eKinTot( x ):
    return eKin1( x ) + eKin2( x )

def eTot(x):
    return eKinTot( x ) + ePotTot( x )

def speed2(x):
    global gL1, gL2, gM1, gM2, gG
    return np.array([
        gL1 * np.cos( x[2] ) *x[1] + gL2 * np.cos( x[4] ) *x[3],
        gL1 * np.sin( x[2] ) *x[1] + gL2 * np.sin( x[4] ) *x[3]
    ])

def toPosition( t, u, v, y, z):
    global gL1, gL2, gM1, gM2
    x1 = +gL1 * np.sin(v)
    y1 = -gL1*np.cos(v)
    x2 = x1 + gL2 * np.sin( z )
    y2 = y1 - gL2 * np.cos( z )
    return [ x1, y1, x2, y2 ]

#  u = tau1
#  v = theta1
#  y = tau2
#  z = theta2

def dgl1( t, u, v, y, z ): #tau_dt= theta1_dtdt
    global gL1, gL2, gM1, gM2, gG
    cd = np.cos( z - v )
    sd = np.sin( z - v )
    
    tau1_dt = gM2 * gL2 * u**2 * sd * cd
    tau1_dt += gM2 * gL2 * y**2 * sd
    tau1_dt += gG * gM2 * cd * np.sin( z )
    tau1_dt -= gG * ( gM1 + gM2 ) * np.sin( v )
    tau1_dt /= ( gL1 * ( gM1 + gM2 * sd**2 ) )
    return tau1_dt

def dgl2( t, u, v, y, z ): #theta1_dt =tau1
    return u

def dgl3( t, u, v, y, z ): #tau_dt= theta1_dtdt
    global gL1, gL2, gM1, gM2, gG
    cd = np.cos( z - v )
    sd = np.sin( z - v )

    tau2_dt = -( gM1 + gM2 ) * gL1 * u**2 * sd
    tau2_dt -=   gM2 * gL2 * y**2 * sd * cd
    tau2_dt -= gG *( gM1+gM2)*np.sin(z)
    tau2_dt += gG* ( gM1 + gM2 ) * cd * np.sin( v )
    tau2_dt /= (gL2 * ( gM1 + gM2 * sd**2 ) )
    return tau2_dt

def dgl4( t, u, v, y, z ): #theta2_dt =tau2
    return y

def coupledRK( t, u, v, y, z ):
    h=.01
    i1 = h * dgl1( t, u, v, y, z )
    j1 = h * dgl2( t, u, v, y, z )
    k1 = h * dgl3( t, u, v, y, z )
    l1 = h * dgl4( t, u, v, y, z )

    i2 = h * dgl1( t + h/2., u + i1/2., v + j1/2.,  y + k1/2., z + l1/2.)
    j2 = h * dgl2( t + h/2., u + i1/2., v + j1/2.,  y + k1/2., z + l1/2.)
    k2 = h * dgl3( t + h/2., u + i1/2., v + j1/2.,  y + k1/2., z + l1/2.)
    l2 = h * dgl4( t + h/2., u + i1/2., v + j1/2.,  y + k1/2., z + l1/2.)

    i3 = h * dgl1( t + h/2., u + i2/2., v + j2/2., y + k2/2., z + l2/2.)
    j3 = h * dgl2( t + h/2., u + i2/2., v + j2/2., y + k2/2., z + l2/2.)
    k3 = h * dgl3( t + h/2., u + i2/2., v + j2/2., y + k2/2., z + l2/2.)
    l3 = h * dgl4( t + h/2., u + i2/2., v + j2/2., y + k2/2., z + l2/2.)

    i4 = h * dgl1( t + h, u + i3, v + j3, y + k3, z + l3)
    j4 = h * dgl2( t + h, u + i3, v + j3, y + k3, z + l3)
    k4 = h * dgl3( t + h, u + i3, v + j3, y + k3, z + l3)
    l4 = h * dgl4( t + h, u + i3, v + j3, y + k3, z + l3)

    i0 = 1./6. * (i1 + 2. * i2 + 2. * i3 + i4)
    j0 = 1./6. * (j1 + 2. * j2 + 2. * j3 + j4)
    k0 = 1./6. * (k1 + 2. * k2 + 2. * k3 + k4)
    l0 = 1./6. * (l1 + 2. * l2 + 2. * l3 + l4)

    return [ t + h, u + i0, v + j0,  y + k0, z + l0 ]


# ~size = 2 * 1000
size = 2 * 50

S = np.zeros((size,size,5))
for i in range( size // 2):
    for j in range(size):
        theta1=i/(1.*size-1.)*2.*np.pi
        theta2=j/(1.*size-1.)*2.*np.pi
        S[i][j][0],S[i][j][1],S[i][j][2],S[i][j][3],S[i][j][4]=0,0,theta1,0,theta2


snapshotI = 10
snapshotS = 100
B = np.zeros( ( size, size, 3 ) )#For Positions
PotTotMax = gG * ( gM1 * gL1 + gM2*( gL1 + gL2 ) )
Pot1Max = gG * gM1 * gL1
Pot2Max=gG * gM2 *( gL1 + gL2 )
v2Max=np.sqrt( 2.0 * 2.0 * PotTotMax / gM2 )

KT = np.zeros( ( size, size, 3 ) )#totalKitetic
KT1 = np.zeros( ( size, size, 3 ) )# first Kitetic
KT2 = np.zeros( ( size, size, 3 ) )#secondKitetic
PT1 = np.zeros( ( size, size, 3 ) )#first potential
PT2 = np.zeros( ( size, size, 3 ) )#second potential
SPEED = np.zeros( ( size, size, 3 ) )#totalKitetic

for s in range( 150 + 1 ):
    for i in range( size // 2 ):
        print( s, i )
        for j in range(size):
            # ~print( s, i, j )
            newState = coupledRK( *( S[i][j] ) )
            S[i][j] = newState
            if not( s % snapshotI ): 
                th1 = ( newState[2]%(2.*np.pi ) )
                th2 = ( newState[4]%(2.*np.pi ) )
                th3 = ( -th1 % (2.0 * np.pi ) )
                th4 = ( -th2 % (2.0 * np.pi ) )
                B[i][j][0],B[i][j][1],B[i][j][2]=.7 * np.sin(th1/2.)**2+.15*(1+np.sin(th2)),0.35*(1+np.sin(th1))+.15*(1+np.sin(th2)),np.sin(th2/2.)**2 
                #fill symmetry
                B[size-i-1][size-j-1][0],B[size-i-1][size-j-1][1],B[size-i-1][size-j-1][2]=.7 * np.sin(th3/2.)**2+.15*(1+np.sin(th4)),0.35*(1+np.sin(th3))+.15*(1+np.sin(th4)),np.sin(th4/2.)**2 
                ke=eKinTot(newState)/(2.*PotTotMax)#normalized
                KT[i][j][0],KT[i][j][1],KT[i][j][2]=cf.STMColor(ke)
                KT[size-i-1][size-j-1][0],KT[size-i-1][size-j-1][1],KT[size-i-1][size-j-1][2]=cf.STMColor(ke)
                ke1=eKin1(newState)/(2.*PotTotMax)#normalized
                KT1[i][j][0],KT1[i][j][1],KT1[i][j][2]=cf.RedScale(ke1)
                KT1[size-i-1][size-j-1][0],KT1[size-i-1][size-j-1][1],KT1[size-i-1][size-j-1][2]=cf.RedScale(ke1)
                ke2=eKin2(newState)/(2.*PotTotMax)#normalized
                KT2[i][j][0],KT2[i][j][1],KT2[i][j][2]=cf.BlueScale(ke2)
                KT2[size-i-1][size-j-1][0],KT2[size-i-1][size-j-1][1],KT2[size-i-1][size-j-1][2]=cf.BlueScale(ke2)

                pe1=(Pot1Max+ePot1(newState))/(2.*Pot1Max)#normalized...can be negative
                PT1[i][j][0],PT1[i][j][1],PT1[i][j][2]=cf.RBWhiteColor(pe1)
                PT1[size-i-1][size-j-1][0],PT1[size-i-1][size-j-1][1],PT1[size-i-1][size-j-1][2]=cf.RBWhiteColor(pe1)
                
                pe2=(Pot2Max+ePot2(newState))/(2.*Pot2Max)#normalized
                PT2[i][j][0],PT2[i][j][1],PT2[i][j][2]=cf.Sunset(pe2)
                PT2[size-i-1][size-j-1][0],PT2[size-i-1][size-j-1][1],PT2[size-i-1][size-j-1][2]=cf.Sunset(pe2)

                sp=speed2(newState)/v2Max
                vAbs=np.linalg.norm(sp)*np.pi

                if vAbs < 1.e-6:
                    sp=[0,0,np.cos(.9*np.pi)]
                else:
                    sp*=np.sin(0.8*vAbs)/vAbs
                    spz=-np.cos(0.8*vAbs)
                    sp=np.append(sp,spz)
                SPEED[i][j][0],SPEED[i][j][1],SPEED[i][j][2]=cf.cColor43D(sp)
                sp[0]*=-1.
                SPEED[size-i-1][size-j-1][0],SPEED[size-i-1][size-j-1][1],SPEED[size-i-1][size-j-1][2]=cf.cColor43D(sp)

    if not( s % snapshotI ):
        posName=( "posI_%03d.png" % s )
        writeImage( posName, B )
        ktName=("kin_tot_%03d.png"%s)
        writeImage(ktName, KT)
        kt1Name=("kin_1_%03d.png"%s)
        writeImage(kt1Name, KT1)
        kt2Name=("kin_2_%03d.png"%s)
        writeImage(kt2Name, KT2)
        
        pt1Name=("pot_1_%03d.png"%s)
        writeImage(pt1Name, PT1)
        pt2Name=("pot_2_%03d.png"%s)
        writeImage(pt2Name, PT2)
        
        speedName=("speed_%03d.png"%s)
        writeImage(speedName, SPEED)
    if not( s % snapshotS ):
        stateName=("state_%03d.png"%s)
        np.save(stateName,S)
