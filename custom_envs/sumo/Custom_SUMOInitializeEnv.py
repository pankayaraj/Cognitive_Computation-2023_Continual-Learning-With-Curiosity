from custom_envs.sumo.custom_sumo_env import SUMOEnv
import os
import traci

class SUMOEnv_Initializer(SUMOEnv):
	def __init__(self,mode='gui', port_no=8870):
		super(SUMOEnv_Initializer, self).__init__(mode=mode, simulation_end=3600, port_no=port_no)
