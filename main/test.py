import numpy as np
import torch
from custom_envs.custom_pendulum import PendulumEnv

import argparse
import os
import sys
import pandas as pd
print(os.environ)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from custom_envs.sumo_rl import SumoEnvironment



env = SumoEnvironment(net_file='nets/4x4-Lucas/4x4.net.xml',
                          route_file='nets/4x4-Lucas/4x4c1c2c1c2.rou.xml',
                          use_gui=True,
                          num_seconds=80000,
                          max_depart_delay=0)