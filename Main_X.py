from Preprocessing import Preprocessing_X as PP
from Agent import Agent_X as ag
from Microgrids import Microgrid_connected_5SP as mg
import time
import numpy as np
import pandas as pd
import os

############# CALL PREPOCESSING #################
PP
############### END PREPROCESSING ################

##################################### BEGIN MAIN ##################################
count_slurm=int(os.environ["SLURM_ARRAY_TASK_ID"])
job_id = int(os.environ["SLURM_ARRAY_JOB_ID"])

env = mg.Environment(PP.dt_training)
#env2 = mg.Environment(PP.dt_training)
#env3 = mg.Environment(PP.dt_testing)

agent = ag.DQNAgent(env)

episode = 200000
rate_step_hour = 0.003
print("variable pour epsilon:",rate_step_hour)
target_network_frequency = 0

cost_memory = []
#cost_memory_wo_eps = []
#cost_memory_wo_eps_test = []

mean_cumulated_cost_memory = []
#mean_cumulated_cost_memory_wo_eps = []
#mean_cumulated_cost_memory_wo_eps_test = []

mean_last_100_cost= 0
mean_last_100_cost_memory = []

#mean_last_100_cost_wo_eps = 0
#mean_last_100_cost_memory_wo_eps = []

#mean_last_100_cost_wo_eps_test = 0
#mean_last_100_cost_memory_wo_eps_test = []

counter = 0

t0 = time.time()

print("Gamma equal to:", agent.gamma)
print("Learning rate:", agent.learning_rate)
print("Epsilon", agent.epsilon)
print("Eps_min to:", agent.epsilon_min)
print("Memory & batch:", agent.memory_size, agent.batch_size)

for e in range(episode+1):

    total_cost_episode = 0
    total_cost_episode_wo_eps = 0
    total_cost_episode_wo_eps_test = 0

    step_episode = 0
    step_hour_count = e*rate_step_hour

    state = env.reset(PP.dt_training,True)
    #state2 = env2.reset(PP.dt_training,True)
    #state3 = env3.reset(PP.dt_testing,True)
  
    while not env.done:
          
        action = agent.act(state, step_episode,e)
        next_state, reward, done = env.step(action)
        
        #act_values_wo_eps = agent.policy_network.predict(state2)
        #action_wo_eps = np.argmax(act_values_wo_eps[0])
        #next_state_wo_eps, reward_wo_eps, done_wo_eps = env2.step(action_wo_eps)

        #act_values_wo_eps_test = agent.policy_network.predict(state3)
        #action_wo_eps_test = np.argmax(act_values_wo_eps_test[0])        
        #next_state_wo_eps_test, reward_wo_eps_test, done_wo_eps_test = env3.step(action_wo_eps_test)

        agent.add_replay_memory(state, action, reward, next_state, done)
        state = next_state
        #state2 = next_state_wo_eps
        #state3 = next_state_wo_eps_test
        
        total_cost_episode += reward
        #total_cost_episode_wo_eps += reward_wo_eps
        #total_cost_episode_wo_eps_test += reward_wo_eps_test

        step_episode += 1
        target_network_frequency +=1
        counter += 1    
 
        if target_network_frequency % 5000 == 0 :
         
            agent.update_target_model()
        
    agent.update_epsilon(step_hour_count)
  
    if len(agent.memory) > agent.batch_size:

        agent.experience_replay()    

########## BEGIN Record Part ##########
    
    cost_memory.append(total_cost_episode)
    #cost_memory_wo_eps.append(total_cost_episode_wo_eps)
    #cost_memory_wo_eps_test.append(total_cost_episode_wo_eps_test)    

    #min_cost_memory = max(cost_memory_wo_eps)
    #min_cost_memory_test = max(cost_memory_wo_eps_test)

    mean_cumulated_cost_memory.append(np.around(np.mean(cost_memory), decimals=1))
    #mean_cumulated_cost_memory_wo_eps.append(np.around(np.mean(cost_memory_wo_eps), decimals=1))
    #mean_cumulated_cost_memory_wo_eps_test.append(np.around(np.mean(cost_memory_wo_eps_test), decimals=1))

    if e > 100:
        
        mean_last_100_cost = np.sum(cost_memory[-100:])/100
        mean_last_100_cost_memory.append(mean_last_100_cost)

        #mean_last_100_cost_wo_eps = np.sum(cost_memory_wo_eps[-100:])/100
        #mean_last_100_cost_memory_wo_eps.append(mean_last_100_cost_wo_eps)

        #mean_last_100_cost_wo_eps_test = np.sum(cost_memory_wo_eps_test[-100:])/100
        #mean_last_100_cost_memory_wo_eps_test.append(mean_last_100_cost_wo_eps_test)

    eps = agent.epsilon

    if e % 100 == 0 :
        t1 = time.time()
        print("episode: {}/{} , epsilon: {:.2}, meancost: {:.2}, in {:.2} secondes".format(int(e), int(episode), float(eps) , float(mean_last_100_cost), float(t1-t0)))
        print('Mean Loss over the last 10 loss',np.mean(agent.history[-10:]))
       # print("Cout minimum training set:", min_cost_memory,"- Cout mini testing", min_cost_memory_test)        


    if e % 50000 == 0 :
                
        df = {"Mean 100 cost memory": mean_last_100_cost_memory}#,"Mean 100 cost w eps 0":mean_last_100_cost_memory_wo_eps, "Mean 100 cost w eps 0 test": mean_last_100_cost_memory_wo_eps_test}
        df = pd.DataFrame(df)
        name_result="result_Mean_Cost_"+str(job_id)+"_"+str(count_slurm)+".csv"
        df.to_csv(name_result, index= False)

        df = {"Cost per episode with eps": cost_memory}#,"Cost per episode with eps 0": cost_memory_wo_eps,"Cost per episode with eps 0 test": cost_memory_wo_eps_test}
        df = pd.DataFrame(df)
        name_result="result_Cost_"+str(job_id)+"_"+str(count_slurm)+".csv"
        df.to_csv(name_result, index= False)

        df = {"Loss_State_Space": agent.history}
        df = pd.DataFrame(df)
        df.to_csv("result_Loss_"+str(job_id)+"_"+str(count_slurm)+".csv", index = False)
 
#print("min cost:", min_cost_memory)    

df = {"Mean 100 cost memory": mean_last_100_cost_memory}#,"Mean 100 cost w eps 0":mean_last_100_cost_memory_wo_eps, "Mean 100 cost w eps 0 test": mean_last_100_cost_memory_wo_eps_test}
df = pd.DataFrame(df)
name_result="result_Mean_Cost_"+str(job_id)+"_"+str(count_slurm)+".csv"
df.to_csv(name_result, index= False)

df = {"Cost per episode with eps": cost_memory}#,"Cost per episode with eps 0": cost_memory_wo_eps,"Cost per episode with eps 0 test": cost_memory_wo_eps_test}
df = pd.DataFrame(df)
name_result="result_Cost_"+str(job_id)+"_"+str(count_slurm)+".csv"
df.to_csv(name_result, index= False)

df = {"Loss_State_Space": agent.history}
df = pd.DataFrame(df)
df.to_csv("result_Loss_"+str(job_id)+"_"+str(count_slurm)+".csv", index = False)
 
# model_json = agent.policy_network.to_json()
# # #with open("train_network_8SP_"+str(job_id)+"_"+str(count_slurm)+".json", "w") as json_file:
# # #    json_file.write(model_json)
# # #agent.policy_network.save_weights("train_network_8SP_"+str(job_id)+"_"+str(count_slurm)+".h5")
# # 
# 
total_cost_episode = 0
step_episode = 0
state = env.reset(PP.dt_training,False)
 
print("Training Set Policy Network:\n")
  
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
state = env.reset(PP.dt_testing,False)
 
  
print("Testing set Policy Network\n:")
  
while not env.done:
 
    action = agent.policy_network.predict(state)
    action = np.argmax(action)
    next_state, reward, done = env.step(action)
 
    if done:
 
        print(step_episode+1,"-",state,"-",action,"-",total_cost_episode)

    state = next_state
    total_cost_episode += reward
    step_episode += 1
    
########## END Record Part ##########
    
##################################### END MAIN ##################################
