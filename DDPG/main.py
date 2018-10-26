import tensorflow as tf
import numpy as np
import os
from DDPG import DDPG_agent
from drone_env import drone_env_heightcontrol

PATH = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(PATH, "data")
tf.set_random_seed(22)
PREMODEL = True
np.set_printoptions(precision=3, suppress=True)

def main():

	with tf.device("/gpu:0"):#网络运行

		config = tf.ConfigProto(allow_soft_placement=True)#tf.ConfigProto()函数用在创建session的时候，用来对session进行参数配置
		config.gpu_options.allow_growth = True#当allow_growth设置为True时，分配器将不会指定所有的GPU内存，而是根据需求增长
		with tf.Session(config=config) as sess:

			globe_episode = tf.Variable(0, dtype=tf.int32, trainable=False, name='globe_episode')#设定trainable=False 可以防止该变量被数据流图的 GraphKeys.TRAINABLE_VARIABLES 收集,
                                                                                                 # 这样我们就不会在训练的时候尝试更新它的值。
			env = drone_env_heightcontrol(aim = None)#外调函数
			state = env.reset()#外调函数
			state_shape = 3
			action_bound = 1
			action_dim = 1
			agent = DDPG_agent(sess, state_shape, action_bound, action_dim)#DDPG使用
			saver = tf.train.Saver()
			print(DIR)

			if not agent.load(saver, DIR):
				sess.run(tf.global_variables_initializer())
				if not os.path.exists(DIR):
					os.mkdir(DIR)
			else:
				print ("coninnue------------------")
			
			if PREMODEL:
				prepath = os.path.join(PATH, 'premodel\checkpoint')
				ckpt = tf.train.get_checkpoint_state(os.path.dirname(prepath))
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(agent.sess, ckpt.model_checkpoint_path)
					agent.action_noise.reset()
					print ("------------pretrained model loaded-------------")#貌似没见调用到
			e, success, episode_reward, step_count = 0, 0, 0, 0

			while True:


				action = agent.act(state)

				next_state, reward, terminal, info = env.step(action)

				episode_reward += reward
				agent.observe(state, action, reward, next_state, terminal)
				agent.train()
				state = next_state
				step_count += 1
				print ("aim height: {}".format(env.aim_height).ljust(20," "),"reward: {:.5f}.".format(reward).ljust(20," "),"steps: {}".format(step_count).ljust(20," "),end = "\r")

				if terminal:

					if info == "success":
						success += 1
					print (" "*80,end = "\r")
					print("episode {} finish, average reward: {:.5f}, total success: {} result: {} step: {}".format(e, episode_reward/step_count, success, info, step_count).ljust(80," "))
					episode_reward = 0
					step_count = 0
					e += 1
					total_episode = sess.run(globe_episode.assign_add(1))
					if e % 10 == 0:
						nDir = os.path.join(PATH, "data/"+str(int(e//10)))
						if not os.path.exists(nDir):
							os.mkdir(nDir)
						agent.save(saver,nDir)
						print("total training episode: {}".format(total_episode))
					state = env.reset()#重置状态

if __name__ == "__main__":
	main()