import AirSimClient
import time
import copy
import numpy as np
from PIL import Image
import cv2

goal_threshold = 3
np.set_printoptions(precision=3, suppress=True)
IMAGE_VIEW = True

class drone_env:#无人机控制
	def __init__(self,start = [0,0,-5],aim = [32,38,-4]):
		self.start = np.array(start)
		self.aim = np.array(aim)
		self.client = AirSimClient.MultirotorClient()#获取airsimclient  remove call 连接
		self.client.confirmConnection()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.threshold = goal_threshold
		
	def reset(self):
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.moveToPosition(self.start.tolist()[0],self.start.tolist()[1],self.start.tolist()[2],5,max_wait_seconds = 10)
		time.sleep(2)
		
		
	def isDone(self):
		pos = self.client.getPosition()
		if distance(self.aim,pos) < self.threshold:#靠近到一定地步则算到了
			return True
		return False
		
	def moveByDist(self,diff, forward = False):
		temp = AirSimClient.YawMode()
		temp.is_rate = not forward
		sta=self.client.getVelocity()

		#print("valo")
		#print(sta2[2])
		#print("-------")
		self.client.moveByVelocity(diff[0], diff[1], diff[2]*0.8, 1 ,drivetrain = AirSimClient.DrivetrainType.ForwardOnly, yaw_mode = temp)
		time.sleep(0.5)
		
		return 0
		
	def render(self,extra1 = "",extra2 = ""):
		pos = v2t(self.client.getPosition())
		goal = distance(self.aim,pos)
		print (extra1,"distance:",int(goal),"position:",pos.astype("int"),extra2)
		
	def help(self):
		print ("drone simulation environment")
		
		
#-------------------------------------------------------
# grid world
		
class drone_env_gridworld(drone_env):
	def __init__(self,start = [0,0,-5],aim = [32,38,-4],scaling_factor = 5):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		
	def interpret_action(self,action):
		scaling_factor = self.scaling_factor
		if action == 0:
			quad_offset = (0, 0, 0)
		elif action == 1:
			quad_offset = (scaling_factor, 0, 0)
		elif action == 2:
			quad_offset = (0, scaling_factor, 0)
		elif action == 3:
			quad_offset = (0, 0, scaling_factor)
		elif action == 4:
			quad_offset = (-scaling_factor, 0, 0)	
		elif action == 5:
			quad_offset = (0, -scaling_factor, 0)
		elif action == 6:
			quad_offset = (0, 0, -scaling_factor)
		
		return np.array(quad_offset).astype("float64")
	
	def step(self,action):
		diff = self.interpret_action(action)
		drone_env.moveByDist(self,diff)
		
		pos_ = v2t(self.client.getPosition())
		vel_ = v2t(self.client.getVelocity())
		state_ = np.append(pos_, vel_)
		pos = self.state[0:3]
		
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		reawrd = reward / 50
		if action == 0:
			reward -= 10
		if self.isDone():
			done = True
			reward = 100
			info = "success"
		if self.client.getCollisionInfo().has_collided:
			reward = -100
			done = True
			info = "collision"
		if (distance(pos_,self.aim)>150):
			reward = -100
			done = True
			info = "out of range"
			
		self.state = state_
		
		return state_,reward,done,info
	
	def reset(self):
		drone_env.reset(self)
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		state = np.append(pos, vel)
		self.state = state
		return state
		
	def rewardf(self,state,state_):
		
		dis = distance(state[0:3],self.aim)
		dis_ = distance(state_[0:3],self.aim)
		reward = dis - dis_
		reward = reward * 1
		reward -= 1
		return reward
		
#-------------------------------------------------------
# height control
# continuous control
		
class drone_env_heightcontrol(drone_env):
	def __init__(self,start = [-23,0,-8],aim = [-23,125,-8],scaling_factor = 2,img_size = [64,64]):
		drone_env.__init__(self,start,aim)
		self.scaling_factor = scaling_factor
		self.aim = np.array(aim)
		self.height_limit = -30
		self.count = 0
		self.initDistance=1000
		self.rand = False

		if aim == None:
			self.rand = True
			self.start = np.array([0,0,-10])
		else:
			self.aim_height = self.aim[2]
	
	def reset_aim(self):
		self.aim = (np.random.rand(3)*200).astype("int")-100
		self.aim[2] = -np.random.randint(3) - 2

		print ("Our aim is: {}".format(self.aim).ljust(80," "),end = '\r')
		self.aim_height = self.aim[2]
		
	def reset(self):
		if self.rand:
			self.reset_aim()
		drone_env.reset(self)
		self.count = 0
		self.state = self.getState()


		relativeState=copy.deepcopy(self.state)
		relativeState[1][0]=self.state[1][0]-self.aim[0]
		relativeState[1][1] = self.state[1][1] - self.aim[1]
		self.initDistance=np.sqrt(abs(relativeState[1][0]) ** 2 + abs(relativeState[1][1]) ** 2 )
		relativeState[1][2] =relativeState[1][1]/relativeState[1][0]

	#	norm_state = copy.deepcopy(relativeState)
	#	norm_state[1] = norm_state[1] / 100

		return relativeState
		
	def getState(self):
		pos = v2t(self.client.getPosition())
		vel = v2t(self.client.getVelocity())
		img = self.getImg()
		state = [img, np.array([pos[0],pos[1],pos[2] - self.aim_height])]
		
		return state
		
	def step(self,action):#一步行为
		pos = v2t(self.client.getPosition())
		dpos = self.aim - pos

		if abs(action[0]) > 1:
			print ("action value error")
			action[0] = action[0] / abs(action[0])
		if abs(action[1]) > 1:
			print ("action value error")
			action[1] = action[1] / abs(action[1])

		self.count+=1
		temp = np.sqrt(dpos[0]**2 + dpos[1]**2)
		dx = action[0] * self.scaling_factor
		dy = action[1]* self.scaling_factor

		
		state_ = self.getState()
		pos = state_[1][2]
		dz =self.aim[2]-pos
		print ("direction: ",dx,dy,dz,end = "\r")
		drone_env.moveByDist(self,[dx,dy,dz],forward = True)
		info = None
		done = False
		reward = self.rewardf(self.state,state_)
		#print("reward")
		#print(self.state[1])
		#print(state_[1])
		#print(reward)
		cache=reward
		if self.isDone():
			if self.rand:
				done = True
				reward = 50
				info = "success"
				self.reset_aim()
			else:
				done = True
				reward = 50
				info = "success"
			
		if self.client.getCollisionInfo().has_collided:
			reward = -50
			done = True
			info = "collision"
		if (pos ) < self.height_limit-8:
			done = True
			info = "too high"
			reward = -50
		if (self.count) >= 400:
			done = True
			info = "too slow"
			reward = -50
			
		self.state = state_

		relativeState = copy.deepcopy(state_)

		relativeState[1][0] = self.state[1][0] - self.aim[0]
		relativeState[1][1] = self.state[1][1] - self.aim[1]
		relativeState[1][2]=relativeState[1][1]/relativeState[1][0]
		reward /= 50

		norm_state = relativeState
		#norm_state[1] = norm_state[1]/100

		#print(norm_state[1])

		return norm_state,reward,done,info
		
	def isDone(self):
		pos = v2t(self.client.getPosition())
		pos[2] = self.aim[2]
		if distance(self.aim,pos) < self.threshold:
			return True
		return False
		
	def rewardf(self,state,state_):
		pos = state[1][2]
		pos_ = state_[1][2]
		reward = - abs(pos_) + 5
		dis = distance(state[1][0:2], self.aim[0:2])
		dis_ = distance(state_[1][0:2], self.aim[0:2])
		reward2 = (self.initDistance-dis_)/self.initDistance
		reward2 = reward2 *50
		print("distance: {}".format(dis_).ljust(20," "),"position: {}".format(state_[1]).ljust(20," "),end = "\r")

		return reward2
		
	def getImg(self):
		
		responses = self.client.simGetImages([AirSimClient.ImageRequest(0, AirSimClient.AirSimImageType.DepthPerspective, True, False)])
		img1d = np.array(responses[0].image_data_float, dtype=np.float)
		#print("===========================================")

		img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

		image = Image.fromarray(img2d)
		im_final = np.array(image.resize((64, 64)).convert('L'), dtype=np.float)/255
		im_final.resize((64,64,1))
		if IMAGE_VIEW:
			cv2.imshow("view",im_final)
			key = cv2.waitKey(1) & 0xFF;
		return im_final
		
def v2t(vect):
	if isinstance(vect,AirSimClient.Vector3r):
		res = np.array([vect.x_val, vect.y_val, vect.z_val])
	else:
		res = np.array(vect)
	return res

def distance(pos1,pos2):
	pos1 = v2t(pos1)
	pos2 = v2t(pos2)

	#dist = np.sqrt(abs(pos1[0]-pos2[0])**2 + abs(pos1[1]-pos2[1])**2 + abs(pos1[2]-pos2[2]) **2)
	dist = np.linalg.norm(pos1-pos2)
	#print(dist)
	return dist