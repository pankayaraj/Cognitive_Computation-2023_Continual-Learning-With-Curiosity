# Continual-Learning-With-Curiosity

This repository contains the experiments of the paper:

"Using Curiosity for an Even Representation of Tasks in Continual Offline Reinforcement Learning" 2023 published in Cognitive Computation, Springer.

by: Pankayaraj Pathmanathan, Natalia Díaz-Rodríguez and Javier Del Ser.




## Description

### 1 . Reproducing the results in the paper
    In order to reproduce the results in this paper just use the three arguments along with the main file run.py
    
    superseding = True
    supersede_env : use the env you want (check below (1.1)
    supersede_buff : use the algo you want to run (see below (1.2) for available algorithms)
    
    
#### 1.1 Environments
    "Pendulum" : results in running the classic control pendulum env
    "Cartpole" : results in running the classic control cartpole env
    "Hopper"   : results in running the pybullet based roboschool hopper env
    "Walker2D" : results in running the pybullet based roboschool Walker2D env

#### 1.2 Algorithms

    "FIFO" :  First in first out buffer
    "HRF" :   Reservoir buffer with a small fifo element 
    "MTR_low" : Multi time scale replay buffer with 2 buffers for cartpole and 3 for the rest
    "MTR_high" :  Multi time scale replay buffer with 5 buffers 
    "TS_HRF"  : Resorvoir buffer with task separation (introduced in this paper)
    "TS_C_HRF" : Curiosity based Resorvior buffer with task separation (introduced in this paper)


Questions?
 p.pankayaraj@gmail.com

    
### 2. Invariant risk minimization 

    Since IRM didn't produce much of a difference in the outcomes we didn't include it in the results. But if you wanted to try you can set superseding = False and manually try changing the parameters on main/run.py
