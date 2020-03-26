import numpy as np
import random

class Environment:

    def __init__(self,PP):

        self.data = PP.dt_testing
        
        self.netdemand1 = self.data[0]
        # self.netdemand3 = self.data[1]
        # self.netdemand2 = self.data[2]
        # self.netdemand1 = self.data[3]
        
        self.num_step = len(PP.dt_training)        
        self.timestep = 0
        self.hour = 0
        self.ND_Category = 0
        
        self.battery_initial = 10
        self.battery_min = 0
        self.battery_max = 100
        self.battery_capacity = self.battery_initial
        
        self.RE = 0
        
        self.done = False
        self.reward = 0
        
        self.state = (self.netdemand1, self.battery_capacity, self.RE, self.ND_Category)
        
        self.observation_space = 4 
        self.action_space = 3
        
    def step(self, action):
        
        if self.netdemand1 != 0:
            
            test_capacity = self.battery_capacity + self.netdemand1
        
            #Discharge priority list
            if (self.netdemand1 < 0) and action == 0 and (self.battery_capacity > self.battery_min):
    
                if (test_capacity >= self.battery_min):
                
                    self.reward = -abs(self.netdemand1) * 0.5 #fixed price
                    self.battery_capacity = test_capacity
                    self.done = self._end_Episode()
                    
                else:
                    
                    reste = abs(test_capacity)
                    self.reward = -(reste * 1.5)  - (self.battery_capacity*0.5)
                    self.battery_capacity = self.battery_min
                    self.done = self._end_Episode()
                    
           # Charge
            elif (self.netdemand1 > 0) and (action == 1) and (test_capacity <= self.battery_max):
                
                self.reward = -abs(self.netdemand1) * 0.5
                self.battery_capacity = test_capacity
                self.done = self._end_Episode()
                
            #Genset
            elif (self.netdemand1 < 0) and  action == 2:
    
                self.reward = -abs(self.netdemand1) * 1.5
                self.done = self._end_Episode()
            
            #GAME OVER
            else:

                self.reward = -abs(self.netdemand1)*10
                self.done = self._end_Episode()       
        
        self.timestep += 1
        self.netdemand1 = self.data[self.timestep-3]
        # self.netdemand3 = self.data[self.timestep-2]
        # self.netdemand2 = self.data[self.timestep-1]
        # self.netdemand1 = self.data[self.timestep]
        self.hour = self.data.index[self.timestep].hour
        
        if self.netdemand1 < 0: 
            
            self.ND_Category = 1
        
        else:   
            
            self.ND_Category = 0
        
        #if (self.timestep % 24) ==0:
            
         #   self.battery_capacity = self.battery_initial
       
        self.state = (self.netdemand1,self.battery_capacity, self.RE, self.ND_Category)
        self.state= np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])
        
        return (self.state, self.reward, self.done)
    
    def reset(self, dt,random_bat):
        
        self.data = dt
        
        self.timestep = 3
        
        self.netdemand1 = self.data[self.timestep-3]
        # self.netdemand3 = self.data[self.timestep-2]
        # self.netdemand2 = self.data[self.timestep-1]
        # self.netdemand1 = self.data[self.timestep]
        
        if random_bat == True :

            #self.battery_init_choice = [0,10,20,30]
            #self.battery_initial = random.choice(self.battery_init_choice)
            self.battery_initial = 30

        else:
            self.battery_initial = 30
            
        self.num_step = len(self.data)        
        
        self.hour = self.data.index[self.timestep].hour

        
        self.battery_min = 0
        self.battery_max = 100
        self.battery_capacity = self.battery_initial

        self.done = False
        self.reward = 0
        self.RE = 0
        
        if self.netdemand1 < 0: 
            
            self.ND_Category = 1
        
        else:   
            
            self.ND_Category = 0
        
        self.state = (self.netdemand1,self.battery_capacity, self.RE, self.ND_Category)
        self.state = np.asarray(self.state)
        self.state = np.reshape(self.state, [1, self.observation_space])

        return self.state
        
        
    def _end_Episode(self):

        end = False


        if (self.timestep + 2) == self.num_step:

            end = True

        return end
