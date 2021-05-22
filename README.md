# Continual-Learning-With-Curiosity

## Description

### 1 . Reproducing the results in the pape
    Inorder to reproduce the results in this paper just used the three arguments aling with the main file run.py
    
    superseding = True
    supersede_env : use the env you want (check below (1.1)
    supersede_buff : use the algo you want to run (see below (1.2) for available algorithms)
    
    
#### 1.1 Enviornments
    "Pendulum" : results in running the classic control pendulum env
    "Cartpole" : results in running the classic control cartpole env
    "Hopper"   : results in running the pybullet based roboschool hopper env
    "Walker2D" : results in running the pybullet based roboschool Walker2D env

#### 1.2 Algorithms

    "FIFO" :  First in first out buffer
    "HRF" :   Reservior buffer with a samll fifo element 
    "MTR_low" : Multi time scale replay buffer with 2 buffers for cartpole and 3 for the rest
    "MTR_high" :  Multi time scale replay buffer with 5 buffers 
    "TS_HRF"  : Resorvior buffer with task seperation (introduced in this paper)
    "TS_C_HRF" : Curiosity based Resorvior buffer with task seperation (introduced in this paper)
    
### 2. Invarient risk minimization 

    Since IRM didn't produce much of a difference in the results we didn't include it in the results. But if you wanted to try you can set superseding = False and manually try changing the parameters on run.py
