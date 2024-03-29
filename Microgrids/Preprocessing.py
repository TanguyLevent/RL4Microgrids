import pandas as pd

class Prepro:

    def __init__(self, day, PV_Penetration, dataset):
        
        if dataset == "X":
 
            path = r"C:/Users/Tanguy LEVENT/Documents/RL_for_Microgrids/Microgrids/Data_5min.csv"
            dt = pd.read_csv(path)
            dt = dt.drop(dt.columns[[1,2,3,4,5,6,8,9,10,11,12,13,14,15,17,18,19,20,21]], axis=1)
            
            dt.rename(columns={"Date and time (UTC)": "DateTime", "Zone1 electricity consumption (kW)": "Load_Consumption",
            		                   "Pmpp FranceWatts panel (W)": "PV_Production"},inplace=True)
            
            dt['PV_Production'] = dt['PV_Production']/1000
            dt['PV_Production'] = dt['PV_Production']*int(PV_Penetration)
            dt['Net_Demand'] = dt['PV_Production'] - dt['Load_Consumption']
            dt['DateTime'] = pd.to_datetime(dt.DateTime)
              
# =============================================================================
#             print("\nThe first ten rows of the dataset:\n")
#             print(dt.head(10))
#             print("\nMissing values processing by columns: \n")
#             print(dt.isnull().sum(),"\n")
# =============================================================================
              
            
            #print("Shape dataframe:",dt.shape)
            dt = dt.loc[dt.DateTime.dt.weekday == int(day),:]
            dt = dt.set_index("DateTime")
            time_schedule = "H"
            multiplicateur = 24
            dt = dt.resample(time_schedule).asfreq()
            dt = dt.dropna(how='any',axis=0) 
            #print("\nDataframe reshape:",dt.shape)
            dt = dt.drop(dt.columns[[0,1]], axis = 1)
               
            training_size = round((dt.shape[0]/multiplicateur)*0.50)
            self.dt_training = dt[0:training_size*multiplicateur].Net_Demand
            self.dt_testing = dt[training_size*multiplicateur:].Net_Demand
