from Microgrids import Preprocessing as PP

import numpy as np
import random
import pandas as pd

class Environment:

    def __init__(self, mode_mg = "connected",pv_penetration = 52,day= 4,dataset = "X", mode_learning = True):
        
        self.DF = PP.Prepro(day, pv_penetration, dataset)
        
        self.mode_mg = mode_mg

        if mode_learning == True:

            self.data = self.DF.dt_training
            self.random_seed = 42
            
        else:

            self.data = self.DF.dt_testing
            self.random_seed = 84
        
        self.netdemand = self.data[0]
        self.num_step = len(self.data)        
        self.timestep = 0
        self.hour = self.data.index[self.timestep].hour
        self.ND_Category = 0
        self.battery_initial = 10
        self.battery_min = 0
        self.battery_max = 100
        self.battery_capacity = self.battery_initial
        
        self.outage_list = self.generate_weak_grid_profile(2)
        
        
        if self.mode_mg == "islanded":

            self.RE = 0

        elif self.mode_mg == "connected":

            self.RE = 1
        
        elif self.mode_mg == "both":

            self.RE = self.outage_list[0]

        else:

            print("Error in microgrid mode definition")

        
        self.done = False
        self.reward = 0
                        ## 0   1    2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
        self.grid_price = [0.3,0.3,0.3,0.3,0.3,0.8,0.8,2.0,2.0,0.8,0.8,0.8,0.8,2.0,0.3,0.3,0.8,2.0,2.0,2.0,2.0,0.8,0.3,0.3]
        

        self.state = (self.netdemand,self.battery_capacity, self.RE, self.hour, self.ND_Category)
        self.observation_space = 5
        self.action_space = 5


    def generate_weak_grid_profile(self, outage_per_day):
        
        np.random.seed(self.random_seed)
    
        #weak_grid_timeseries = np.random.random_integers(0,1, int(nb_time_step_per_year+1) ) #for a number of time steps, value between 0 and 1
        #generate a timeseries of 8760/timestep points based on np.random seed
        #profile of ones and zeros
        weak_grid_timeseries = np.random.random(self.num_step+1) #for a number of time steps, value between 0 and 1
    
    
        weak_grid_timeseries = [0 if weak_grid_timeseries[i] < outage_per_day/24 else 1 for i in range(len(weak_grid_timeseries))]
    
  #duration of the outage
# =============================================================================
#         for i in range(len(weak_grid_timeseries)):
#             if weak_grid_timeseries[i] == 0:
#                 for j in range(1, int(duration_of_outage/timestep)):
#                     if i-j > 0:
#                         weak_grid_timeseries[i-j] = 0
# 
# =============================================================================

        #print weak_grid_timeseries

        return weak_grid_timeseries

    def step(self, action):
        
        if self.netdemand != 0:
            
            test_capacity = self.battery_capacity + self.netdemand
        
            #Discharge priority list
            if (self.netdemand < 0) and action == 0 and (self.battery_capacity > self.battery_min):
    
                if (test_capacity >= self.battery_min):
                
                    self.reward = -abs(self.netdemand) * 0.27 #fixed price
                    self.battery_capacity = test_capacity
                    self.done = self._end_Episode()
                    
                else:
                    
                    reste = abs(test_capacity)
                    self.reward = -(reste * 2.8)  - (self.battery_capacity*0.27)
                    self.battery_capacity = self.battery_min
                    self.done = self._end_Episode()
                    
           # Charge
            elif (self.netdemand > 0) and (action == 1) and (test_capacity <= self.battery_max):
                
                self.reward = -abs(self.netdemand) * 0.27
                self.battery_capacity = test_capacity
                self.done = self._end_Episode()
                
            #Genset
            elif (self.netdemand < 0) and  action == 2:
    
                self.reward = -abs(self.netdemand) * 2.8
                self.done = self._end_Episode()
           
            #Buy
            elif (self.netdemand < 0) and  (self.RE == 1) and action == 3:
    
                self.reward = -abs(self.netdemand) * 0.34 #* self.grid_price[self.hour]
                self.done = self._end_Episode()
            
            #Sell
            elif (self.netdemand > 0) and (self.RE == 1) and action == 4:
    
                self.reward = abs(self.netdemand) * 0.12#* self.grid_price[self.hour]
                self.done = self._end_Episode()    
           
            #Curtailement 
            else:

                self.reward = -abs(self.netdemand)*10
                self.done = self._end_Episode()       
        
        self.timestep += 1
        self.netdemand = self.data[self.timestep]
        self.hour = self.data.index[self.timestep].hour
        
        
        if self.netdemand < 0: 
            
            self.ND_Category = 1
        
        else:   
            

            self.ND_Category = 0
            
        
        if self.mode_mg == "islanded":

            self.RE = 0

        elif self.mode_mg == "connected":

            self.RE = 1
            
        elif self.mode_mg == "both":

            self.RE = self.outage_list[self.timestep]

            
        self.state = (self.netdemand,self.battery_capacity, self.RE, self.hour,self.ND_Category)            
        self.state= np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])
        
        return (self.state, self.reward, self.done)


    def reset(self, mode_learning, random_bat):
        
        self.timestep = 0     
        self.netdemand = self.data[self.timestep]
        

        if mode_learning == True:

            self.data = self.DF.dt_training
            self.random_seed = 42

        else:

            self.data = self.DF.dt_testing
            self.random_seed = 84
        
        self.outage_list = self.generate_weak_grid_profile(2)
        
        if random_bat == True :

            self.battery_init_choice = [0,10,20,30]
            self.battery_initial = random.choice(self.battery_init_choice)
            #self.battery_initial = 30

        else:
            self.battery_initial = 30
            
            
    
        self.num_step = len(self.data)        
        self.hour = self.data.index[self.timestep].hour
        
        self.battery_min = 0
        self.battery_max = 100
        self.battery_capacity = self.battery_initial

        self.done = False
        self.reward = 0

        if self.mode_mg == "islanded":

            self.RE = 0

        elif self.mode_mg == "connected":

            self.RE = 1
            
        elif self.mode_mg == "both":

            self.RE = self.outage_list[self.timestep]

        
        if self.netdemand < 0: 
            
            self.ND_Category = 1
        
        else:   
            
            self.ND_Category = 0
        

        self.state = (self.netdemand,self.battery_capacity, self.RE,self.hour, self.ND_Category)
        self.state = np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])

        return self.state
        
        
    def _end_Episode(self):

        end = False


        if self.timestep+2 == self.num_step:

            end = True

        return end
