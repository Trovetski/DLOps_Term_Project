#import statements
import numpy as np
from sklearn.neighbors import BallTree
from PIL import Image, ImageTk, ImageDraw

#environmental parameters
n_obs = 4

n_sides = 7

pps = 50

n_rays = 360*2

#CONSTANTS
SIN5 = np.sin(5*np.pi/180)
COS5 = np.cos(5*np.pi/180)

COS = np.cos(np.linspace(0, 2*np.pi, n_rays,False))[None].T
SIN = np.sin(np.linspace(0, 2*np.pi, n_rays,False))[None].T

class Environment:

    def __init__(self, batch_size, size):
        #number of instances of environments
        self.batch_size = batch_size

        #size of the environemnts
        self.size = size

        #pursuer, target coordinates and velocities
        self.p = np.random.randint(0,self.size-1,(self.batch_size,2))*1.0

        self.t = np.random.randint(0,self.size-1,(self.batch_size,2))*1.0

        self.th = 2*np.pi*np.random.rand(self.batch_size)
        self.pv = np.zeros((self.batch_size,2),'float64')
        self.pv[:,0] = 0.9*self.size*np.cos(self.th)/200
        self.pv[:,1] = 0.9*self.size*np.sin(self.th)/200

        self.tv = np.zeros_like(self.pv)
        
        #generating the target point cloud
        self.point_cloud = np.zeros((self.batch_size,pps,2))
        th = np.linspace(0,2*np.pi,pps,False)[None].T
        r = self.size/60

        self.point_cloud[:,:,0] = (self.t[:,0] + r*np.cos(th)).T
        self.point_cloud[:,:,1] = (self.t[:,1] + r*np.sin(th)).T

        self.index = [BallTree(i) for i in self.point_cloud]

        #other environment variables
        self.episode_steps = np.zeros((self.batch_size,))

        self.MOVE_REWARD = -1
        self.CAPTURE_REWARD = 150
        self.COLLISION_REWARD = -50
    
    def step(self, delTh):
        #increase the steps
        self.episode_steps += 1

        #move the pursuer and change its direction
        self.p += self.pv

        mask = delTh!=0
        temp = self.pv[:,0][mask]
        self.pv[:,0][mask] = temp*COS5 - (delTh*self.pv[:,1]*SIN5)[mask]
        self.pv[:,1][mask] = self.pv[:,1][mask]*COS5 + delTh[mask]*temp*SIN5
        
        '''
            check for terminal conditions i.e.

                1. If the pursuer is out of bounds
                2. If the pursuer has captured the target
                3. If the pursuer has exausted episode steps
            
            take the logical or of all these and return reward
        '''
        #check for condition 1
        isOutOfBounds = (self.p>self.size) + (self.p<0)
        isOutOfBounds = isOutOfBounds[:,0] + isOutOfBounds[:,1]

        #check for condition 2
        hasCaptured = np.sum((self.p-self.t)**2,axis=1)<100

        #check for condition 3
        isOutOfSteps = self.episode_steps>300

        #calculate reward and return everything
        reward = self.MOVE_REWARD + self.CAPTURE_REWARD*hasCaptured + self.COLLISION_REWARD*isOutOfBounds

        done = isOutOfBounds + hasCaptured + isOutOfSteps

        state = self.getState(20)
        
        #reset the environments that are terminated
        self.reset(done)

        if(hasCaptured.sum()>0):
            print("CAPTURE")
        if(isOutOfBounds.sum()>0):
            print("COLLISION")

        return state, reward, done
    
    def reset(self, which):
        n = which.sum()
        if(n==0):
            return False
        
        self.p[which] = np.random.randint(0,self.size-1,(n,2))

        self.t[which] = np.random.randint(0,self.size-1,(n,2))

        self.th = 2*np.pi*np.random.rand(n)
        self.pv[:,0][which] = 0.9*self.size*np.cos(self.th)/200
        self.pv[:,1][which] = 0.9*self.size*np.sin(self.th)/200

        self.tv[which] = np.zeros_like(self.pv[which])
        
        #generating the target point cloud
        self.point_cloud = np.zeros((self.batch_size,pps,2))
        th = np.linspace(0,2*np.pi,pps,False)[None].T
        r = self.size/60

        self.point_cloud[:,:,0] = (self.t[:,0] + r*np.cos(th)).T
        self.point_cloud[:,:,1] = (self.t[:,1] + r*np.sin(th)).T

        self.index = [BallTree(i) for i in self.point_cloud]

        self.episode_steps[which] = 0
    
    def query(self, points):
        out = np.zeros((points.shape[0],points.shape[1]))

        for i in range(points.shape[0]):
            dist, ind = self.index[i].query(points[i])
            out[i] = dist.T[0]
        
        return out
    
    def getOutPoints(self,steps:int):
        #unit rays staring from the velocity vector in all directions
        unit_rays = np.zeros((self.batch_size,n_rays,2))
        unit_rays[:,:,0] = (self.pv[:,0]*COS - self.pv[:,1]*SIN).T/2.7
        unit_rays[:,:,1] = (self.pv[:,1]*COS + self.pv[:,0]*SIN).T/2.7

        #the source of rays i.e. the pursuer
        unit_step = np.expand_dims(self.p,1) + unit_rays

        #global minimum distance of the pursuer from the environment
        globalMinD = self.query(unit_step)

        #ray marching algorithm
        for _ in range(steps):
            currentMinD = self.query(unit_step+np.expand_dims(globalMinD,2)*unit_rays)
            currentMinD[currentMinD<0.6] = 0
            currentMinD[currentMinD>600] = 0
            globalMinD += currentMinD
        
        endPoints = np.expand_dims(globalMinD,2)*unit_rays
        outPoints = np.expand_dims(self.p,1) + endPoints

        #slope of all the rays and padding
        m = endPoints[:,:,1]/endPoints[:,:,0]
        pad = 3

        #top -> y > HIEGHT-padding
        mask = outPoints[:,:,1]>self.size-pad

        outPoints[mask,0] = (self.p[:,0][None].T + (self.size-pad-self.p[:,1][None].T)/m)[mask]
        outPoints[mask,1] = self.size-pad

        #bottom -> y < 0+padding
        mask = outPoints[:,:,1]<pad

        outPoints[mask,0] = (self.p[:,0][None].T + (pad-self.p[:,1][None].T)/m)[mask]
        outPoints[mask,1] = pad

        #left -> x < 0+padding
        mask = outPoints[:,:,0]<pad

        outPoints[mask,1] = (self.p[:,1][None].T + (pad-self.p[:,0][None].T)*m)[mask]
        outPoints[mask,0] = pad
        
        #right -> x > WIDTH+padding
        mask = outPoints[:,:,0]>self.size-pad

        outPoints[mask,1] = (self.p[:,1][None].T + (self.size-pad-self.p[:,0][None].T)*m)[mask]
        outPoints[mask,0] = self.size-pad

        #get the distance vector state
        out = np.sqrt(np.sum((outPoints-np.expand_dims(self.p,1))**2,axis=2))
        out = np.expand_dims(out,1)

        return outPoints
    
    def getState(self, steps:int):
        outPoints = self.getOutPoints(steps)

        #get the distance vector state
        out = np.sqrt(np.sum((outPoints-np.expand_dims(self.p,1))**2,axis=2))
        out = np.expand_dims(out,1)

        return out


    def renderState(self):
        #BEFORE RUNNING CHANGE RETURN IN GETSTATE TO OUTPOINTS
        img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8), 'RGB')
        draw = ImageDraw.Draw(img)

        draw.ellipse((self.p[0][0]-8,self.p[0][1]-8,self.p[0][0]+8,self.p[0][1]+8), fill=(255,0,0), outline=(200,200,200))
        draw.line((self.p[0][0],self.p[0][1],self.p[0][0]+self.pv[0][0]*20,self.p[0][1]+self.pv[0][1]*20),width=3)

        for p in self.getOutPoints(20)[0]:
            draw.point((p[0],p[1]),fill=(255,255,255))
        
        return img