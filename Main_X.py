from Agent import Agent_DDQN as ag
from Microgrids import Microgrid as mg
import time
import numpy as np
import pandas as pd

############################################## BEGIN MAIN #####################################################

################################## BEGIN TRAINING PHASE ##################################

#Unblock comment if using on supercomputer
#count_slurm=int(os.environ["SLURM_ARRAY_TASK_ID"])
#job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])

mode_mg = "connected" #choice: islanded, connected, both
pv_penetration = 52
day = 4
dataset = "X" #choice: X, Tahiti
mode_learning = True #True = training, False = testing

env = mg.Environment(mode_mg,pv_penetration,day,dataset, mode_learning)
agent = ag.DQNAgent(env)

episode = 200000
rate_step_hour = 0.003
target_network_frequency = 0

min_cost = 0
cost_memory = []
mean_cumulated_cost_memory = []
mean_last_100_cost = 0
mean_last_100_cost_memory = []

t0 = time.time()

print("\nJour: {}, Penetration PV: {}, Microgrid: {}, Dataset: {}".format(int(day), int(pv_penetration), str(mode_mg), str(dataset)))
print("Nombre d'épisode:", episode)
print("Gamma equal to:", agent.gamma)
print("Learning rate:", agent.learning_rate)
print("Variable pour epsilon update:",rate_step_hour)
print("Epsilon start", agent.epsilon)
print("Eps_min to:", agent.epsilon_min)
print("Memory & batch:", agent.memory_size, agent.batch_size,"\n")

for e in range(episode+1):

    total_cost_episode = 0
    step_episode = 0
    epsilon_update_value = e*rate_step_hour

    state = env.reset(mode_learning,True)
 
    while not env.done:
          
        action = agent.act(state, step_episode,e)
        next_state, reward, done = env.step(action)
        agent.add_replay_memory(state, action, reward, next_state, done)
        state = next_state

        total_cost_episode += reward

        step_episode += 1
        target_network_frequency +=1
 
 
        if target_network_frequency % 5000 == 0 :
         
            agent.update_target_model()
        
    agent.update_epsilon(epsilon_update_value)
  
    if len(agent.memory) > agent.batch_size:

        agent.experience_replay()    
        
################################## END TRAINING PHASE ##################################

################################## BEGIN Record Part ##################################
    
    cost_memory.append(total_cost_episode)
    mean_cumulated_cost_memory.append(np.around(np.mean(cost_memory), decimals=1))
    min_cost = max(cost_memory)
    
    if e > 100:
        
        mean_last_100_cost= np.sum(cost_memory[-100:])/100
        mean_last_100_cost_memory.append(mean_last_100_cost)


    if e % 100 == 0 :
        t1 = time.time()
        print("episode: {}/{} , epsilon: {:.2}, meancost: {:.2}, mincost: {:.2}, in {:.2} secondes".format(int(e), int(episode), float(agent.epsilon) , float(mean_last_100_cost), float (min_cost),float(t1-t0)))
        print('Mean Loss over the last 10 loss {:.2}'.format(np.mean(agent.history[-10:])))


    if e % 50000 == 0:
                
        df = {"Mean 100 cost memory": mean_last_100_cost_memory}#,"Mean 100 cost w eps 0":mean_last_100_cost_memory_wo_eps, "Mean 100 cost w eps 0 test": mean_last_100_cost_memory_wo_eps_test}
        df = pd.DataFrame(df)
        name_result="result_Mean_Cost_.csv" #+str(job_id)+"_"+str(count_slurm)+".csv"
        df.to_csv(name_result, index= False)

        df = {"Cost per episode with eps": cost_memory}#,"Cost per episode with eps 0": cost_memory_wo_eps,"Cost per episode with eps 0 test": cost_memory_wo_eps_test}
        df = pd.DataFrame(df)
        name_result="result_Cost_.csv" #+str(job_id)+"_"+str(count_slurm)+".csv"
        df.to_csv(name_result, index= False)

        df = {"Loss_State_Space": agent.history}
        df = pd.DataFrame(df)
        df.to_csv("result_Loss_.csv") #+str(job_id)+"_"+str(count_slurm)+".csv", index = False)
 
################################## END Record Part ##################################
        
################################## BEGIN TRAINING & TESTING GREEDY MODE ##################################
                 
total_cost_episode = 0
step_episode = 0
state = env.reset(mode_learning,False)
 
print("\nTraining Set Policy Network:")
  
while not env.done:
     
    action = agent.policy_network.predict(state)
    action = np.argmax(action)
    next_state, reward, done = env.step(action)
     
    if done:
        
        print(step_episode+1,"-",state,"-",action,"-",total_cost_episode)
 
    state = next_state
    total_cost_episode += reward
    step_episode += 1

total_cost_episode = 0
step_episode = 0
mode_learning = False   
state = env.reset(mode_learning,False)
 
  
print("\nTesting set Policy Network:")
  
while not env.done:
 
    action = agent.policy_network.predict(state)
    action = np.argmax(action)
    next_state, reward, done = env.step(action)
 
    if done:
 
        print(step_episode+1,"-",state,"-",action,"-",total_cost_episode)

    state = next_state
    total_cost_episode += reward
    step_episode += 1
    
################################## END TRAINING & TESTING GREEDY MODE ##################################
    
############################################## END MAIN #####################################################
