from collections import deque
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:

    ######### BEGIN INIT ############
    
    def __init__(self,env):
        
        self.state_size = env.observation_space
        self.action_size = env.action_space
        #self.count_slurm = int(os.environ["SLURM_ARRAY_TASK_ID"])
        self.memory_size =100000

        self.memory = deque(maxlen=self.memory_size)
        self.memory_10_size = 10
        self.memory_10 = deque(maxlen=self.memory_10_size)
        self.batch_size = 32
        
        self.gamma = 0.1 #self.count_slurm/10
        self.epsilon = 1.0

        self.epsilon_min = 0.1
        self.learning_rate = 0.00025
                
        self.policy_network = self.build_model()
        self.target_network = self.build_model()
       
        self.update_target_model()       
 
        self.bool_test = False
        self.bool_random_action = False
        self.history = []
    
    ######### END INIT ############   

    ######### BEGIN BUILD MODEL ############
        
    def build_model(self):

        model = Sequential()
        model.add(Dense(self.state_size, activation='linear'))        
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer= Adam(lr=self.learning_rate))
        
        return model
    
    ######### END BUILD MODEL ############
    
    ######### BEGIN ADD MEMORIES ############
    
    
    def add_replay_memory(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        
    ######### END ADD MEMORIES ############
    
    
    ######### BEGIN UPDATES: EPS and TARGET NETWORK ############
    
    def update_epsilon(self,step_hour_count):
        
        self.epsilon = 1/(1.1)**step_hour_count
        
        if self.epsilon < self.epsilon_min:
            
            self.epsilon = self.epsilon_min
        
        return self.epsilon
    
    def update_target_model(self):
       
        self.target_network.set_weights(self.policy_network.get_weights())

    ######### END UPDATES: EPS and TARGET NETWORK ############
    
    
    ######### BEGIN ACTION AGENT: EXPLOITATION/EXPLORATION DILEMNA ############
               
    def act(self, state, step,e):

        self.greedy = np.random.rand()       

        if (self.greedy <= self.epsilon):
            
            action = random.randrange(self.action_size)
            self.bool_random_action = True
            return action
        
        else:   
            
            act_values = self.policy_network.predict(state)
            action = np.argmax(act_values[0])
            self.bool_random_action = False
            return action
        

    
    ######### END AGENT: EXPLOITATION/EXPLORATION DILEMNA ############ 
    
     ######### TRAIN : Experience Replay ############ 
    
    def experience_replay(self):
        
        minibatch = random.sample(self.memory, self.batch_size-1)
        minibatch.append(self.memory[-1])
        states = []
        newStates=[]
        
        for sample in minibatch:
            
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)
        
        states = np.array(states).reshape(self.batch_size, self.state_size)
        newStates = np.array(newStates).reshape(self.batch_size, self.state_size)
        
        targets = self.policy_network.predict(states) #targets = array of 32*4 car 4 Q(s,a) pour chaque example (car 4 actions)
        next_targets = self.policy_network.predict(newStates)
        new_state_targets = self.target_network.predict(newStates)
        
        i = 0
        
        for sample in minibatch:
            
            state, action, reward, new_state, done = sample
            
            if done:
                
                targets[i][action] = reward
                
            else:
                
                next_best_action = np.argmax(next_targets[i])
                targets[i][action] = reward + self.gamma * new_state_targets[i][next_best_action]
                
            i+=1
        
        result = self.policy_network.fit(states, targets, epochs=1, verbose=0)
        self.history.append(result.history['loss'])
 
