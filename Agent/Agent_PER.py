from collections import deque
import random
import numpy as np
import os
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:

    ######### BEGIN INIT ############
    
    def __init__(self,mg):
        
        self.state_size = mg.observation_space
        self.action_size = mg.action_space
        
        self.memory_size = 20000
        self.memory = deque(maxlen=self.memory_size)
        self.priorities = deque(maxlen= self.memory_size)
        self.memory_10_size = 10
        self.memory_10 = deque(maxlen=self.memory_10_size)
        self.batch_size = 32
        
        
        self.epsilon = 1.0
        self.epsilon_min = 0.0004
        
        self.gamma = 0.02
        self.learning_rate = 0.00025
        
        self.policy_network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_model()
     
        self.bool_test = False
        
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
        self.priorities.append(max(self.priorities, default= 1))
        
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
  
    ######### BEGIN PER ############.
    
    
    def get_probabilities(self, priority_scale):
        
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        
        return sample_probabilities
    
    def get_importance(self, probabilities):
        
        importance = 1/self.memory_size * 1/probabilities
        importance_normalized = importance/max(importance)
        
        return importance_normalized
        
    def sample_buffer(self, priority_scale = 1.0):
                       
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = np.random.choice(range(len(self.memory)), self.batch_size, p = sample_probs)
        samples = np.array(self.memory)[sample_indices]
                    
        return samples, sample_indices
    
    
    def get_TD_Error(self,target, qsa):
        
        offset= 0.1
        error = abs(target - qsa) + offset
        
        return error
    
    def set_priorities(self, idx, error):
        
        self.priorities[idx] = error
    
        return error
    
    ######### END PER ############     
    
    
    def experience_replay(self):
        
        minibatch,idx = self.sample_buffer()
   
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
            
            state, action, reward, new_state, done= sample
            
            if done:
                
                targets[i][action] = reward
                
            else:
                
                next_best_action = np.argmax(next_targets[i])
                target = reward + self.gamma * new_state_targets[i][next_best_action]
                error = self.get_TD_Error(target, targets[i][action])
                self.set_priorities(idx[i],error)
                targets[i][action] = target
            
            i+=1
        

        result = self.policy_network.fit(states, targets, epochs=1, verbose=0)
        self.history.append(result.history['loss'])      
