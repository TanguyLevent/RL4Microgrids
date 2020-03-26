from Preprocessing import Preprocessing_X as PP

import numpy as np
import random
from numpy import array


class Environment:

    def __init__(self):

        self.data = PP
        
        self.netdemand = self.data[0]
                
        self.num_step = len(self.data)        
        self.timestep = 0
        self.hour = 0
        self.ND_Category = 0
        
        self.battery_initial = 10
        self.battery_min = 0
        self.battery_max = 100
        self.battery_capacity = self.battery_initial
        
        self.RE = 1
        
        self.done = False
        self.reward = 0
                        ## 0   1    2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
        self.grid_price = [0.3,0.3,0.3,0.3,0.3,0.8,0.8,2.0,2.0,0.8,0.8,0.8,0.8,2.0,0.3,0.3,0.8,2.0,2.0,2.0,2.0,0.8,0.3,0.3]
        

        self.state = (self.netdemand,self.battery_capacity, self.RE, self.hour, self.ND_Category)
        self.observation_space = 5
        self.action_space = 5


    def step(self, action):
        
        if self.netdemand != 0:
            
            test_capacity = self.battery_capacity + self.netdemand
        
            #Discharge priority list
            if (self.netdemand < 0) and action == 0 and (self.battery_capacity > self.battery_min):
    
                if (test_capacity >= self.battery_min):
                
                    self.reward = -abs(self.netdemand) * 0.5 #fixed price
                    self.battery_capacity = test_capacity
                    self.done = self._end_Episode()
                    
                else:
                    
                    reste = abs(test_capacity)
                    self.reward = -(reste * 1.5)  - (self.battery_capacity*0.5)
                    self.battery_capacity = self.battery_min
                    self.done = self._end_Episode()
                    
           # Charge
            elif (self.netdemand > 0) and (action == 1) and (test_capacity <= self.battery_max):
                
                self.reward = -abs(self.netdemand) * 0.5
                self.battery_capacity = test_capacity
                self.done = self._end_Episode()
                
            #Genset
            elif (self.netdemand < 0) and  action == 2:
    
                self.reward = -abs(self.netdemand) * 1.5
                self.done = self._end_Episode()
            #Buy
            elif (self.netdemand < 0) and  (self.RE == 1) and action == 3:
    
                self.reward = -abs(self.netdemand) * self.grid_price[self.hour]
                self.done = self._end_Episode()
            #Sell
            elif (self.netdemand > 0) and (self.RE == 1) and action == 4:
    
                self.reward = abs(self.netdemand) * self.grid_price[self.hour]
                self.done = self._end_Episode()    
            #GAME OVER
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
            
        self.state = (self.netdemand,self.battery_capacity, self.RE, self.hour,self.ND_Category)            
        self.state= np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])
        
        return (self.state, self.reward, self.done)


    def reset(self, dt, random_bat):
        
        self.data = dt
        
        self.timestep = 0     
        self.netdemand = self.data[self.timestep]
        
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
        self.RE =1
        
        if self.netdemand < 0: 
            
            self.ND_Category = 1
        
        else:   
            
            self.ND_Category = 0
        
        #self.netdemand_predicted = self.netdemand	

        self.state = (self.netdemand,self.battery_capacity, self.RE,self.hour, self.ND_Category)
        self.state = np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])

        return self.state
        
        
    def _end_Episode(self):

        end = False


        if self.timestep+2 == self.num_step:

            end = True

        return end
