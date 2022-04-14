import numpy as np
import pandas as pd
import pickle
import pickle_compat
pickle_compat.patch()
from sklearn import model_selection
from sklearn import svm
from sklearn import decomposition, preprocessing
from scipy.fftpack import fft

def fetch_meal_no_meal_data(meal_time, start_time, end_time, is_data_meal, glucose_data_1):
    meal_data = []
    for t in meal_time:
        idx_meal = glucose_data_1[glucose_data_1['datetime'].between(t + pd.DateOffset(hours = start_time), t + pd.DateOffset(hours = end_time))]
        if 24 > idx_meal.shape[0]:
            continue
        val_glucose = idx_meal['Sensor Glucose (mg/dL)'].to_numpy()
        mean = idx_meal['Sensor Glucose (mg/dL)'].mean()
        if is_data_meal:
            count_missing_val = 30 - len(val_glucose)
            if count_missing_val > 0:
                for i in range(count_missing_val):
                    val_glucose = np.append(val_glucose, mean)
            meal_data.append(val_glucose[0:30])
        else:
            meal_data.append(val_glucose[0:24])
            
    return pd.DataFrame(data=meal_data)

def no_meal_and_meal_slots(slots_first,slot_diff):
    
    slots_1 = slots_first[0:len(slots_first)-1]
    slots_2 = slots_first[1:len(slots_first)]
    diff = list(np.array(slots_1) - np.array(slots_2))
    val = list(zip(slots_1, slots_2, diff))
    meal_slots = []
    for i in val:
        if slot_diff>i[2]:
            meal_slots.append(i[0])
    return meal_slots

def no_meal_and_meal_data(insulin_DF,glucose_DF):
    
    no_meal_Times =[]  
    meal_Times=[]
    
    scaler_std = preprocessing.StandardScaler()
    pca = decomposition.PCA(n_components=5)
    
    no_meal_data = pd.DataFrame()
    meal_data = pd.DataFrame()
    
    glucose_DF= glucose_DF[::-1]
    glucose_DF['Sensor Glucose (mg/dL)'] = glucose_DF['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    glucose_DF['datetime'] = pd.to_datetime(glucose_DF["Date"].astype(str) + " " + glucose_DF["Time"].astype(str))
    glucose_DF_1 = glucose_DF[['datetime','Sensor Glucose (mg/dL)']]
    
    insulin_DF= insulin_DF[::-1]
    insulin_DF['datetime'] = pd.to_datetime(insulin_DF["Date"].astype(str) + " " + insulin_DF["Time"].astype(str))
    insulin_DF_1 = insulin_DF[['datetime','BWZ Carb Input (grams)']]
    insulin_DF_1 = insulin_DF_1[(insulin_DF_1['BWZ Carb Input (grams)'].notna()) & (insulin_DF_1['BWZ Carb Input (grams)']>0) ]
    new_slots = list(insulin_DF_1['datetime'])
    
      
    no_meal_Times = no_meal_and_meal_slots(new_slots,pd.Timedelta('0 days 240 min'))
    no_meal_data = fetch_meal_no_meal_data(no_meal_Times,2,4,False, glucose_DF_1)
    no_meal_data_DFFeatures = glucoseFeatures(no_meal_data)
    no_meal_standard = scaler_std.fit_transform(no_meal_data_DFFeatures)
    no_meal_pca = pd.DataFrame(pca.fit_transform(no_meal_standard))
    no_meal_pca['class'] = 0

    
    meal_Times = no_meal_and_meal_slots(new_slots,pd.Timedelta('0 days 120 min'))
    meal_data = fetch_meal_no_meal_data(meal_Times,-0.5,2,True, glucose_DF_1)
    meal_data_DFFeatures = glucoseFeatures(meal_data)
    meal_standard = scaler_std.fit_transform(meal_data_DFFeatures)
    pca.fit(meal_standard)
    meal_pca = pd.DataFrame(pca.fit_transform(meal_standard))
    meal_pca['class'] = 1   
    data = meal_pca.append(no_meal_pca)
    data.index = [i for i in range(data.shape[0])]
    return data


def glucose_Entropy(row):
    
    entropy = 0
    row_len = len(row)
    
    if row_len > 1:
        value, cnt = np.unique(row, return_counts=True)
        ratio = cnt / row_len
        nzr = np.count_nonzero(ratio)
        if nzr <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy
    else:
        return 0
    
def FFT(row):
    amp = []
    FFT = fft(row)
    row_len = len(row)
    frequency = np.linspace(0, row_len * 2/300, row_len)
    
    for a in FFT:
        amp.append(np.abs(a))
        
    amp_sorted = amp
    amp_sorted = sorted(amp_sorted)
    amp_max = amp_sorted[(-2)]
    freq_max = frequency.tolist()[amp.index(amp_max)]
    return [amp_max, freq_max]

def absolute_mean(row):
    mean = 0
    for p in range(0, len(row) - 1):
        mean = mean + np.abs(row[(p + 1)] - row[p])
    return mean / len(row)

def root_mean_square(row):
    rms = 0
    for p in range(0, len(row) - 1):
        rms = rms + np.square(row[p])
    return np.sqrt(rms/ len(row))

def glucoseFeatures(meal_no_meal_DF):
    
    features=pd.DataFrame()
    for i in range(0, meal_no_meal_DF.shape[0]):
        row = meal_no_meal_DF.iloc[i, :].tolist()
        features = features.append({ 
         'root_mean_square':root_mean_square(row),
         'glucoseEntropy':glucose_Entropy(row),
         'min':min(row), 
         'max':max(row),
         'range': max(row) - min(row),
         'absolute_value_mean1':absolute_mean(row[:13]), 
         'absolute_value_mean2':absolute_mean(row[13:]),  
         'FFT1':FFT(row[:13])[0], 
         'FFT2':FFT(row[:13])[1], 
         'FFT3':FFT(row[13:])[0], 
         'FFT4':FFT(row[13:])[1]},
          ignore_index=True)
    return features
  
if __name__=='__main__':
    
    glucose_DataFrame_1=pd.read_csv("CGM_patient2.csv",low_memory=False)
    insulin_DataFrame_1=pd.read_csv("Insulin_patient2.csv",low_memory=False)
    glucose_DataFrame=pd.read_csv("CGMData.csv",low_memory=False)
    insulin_DataFrame=pd.read_csv("InsulinData.csv",low_memory=False)
    
    glucose_data=pd.concat([glucose_DataFrame_1,glucose_DataFrame])
    insulin_data=pd.concat([insulin_DataFrame_1,insulin_DataFrame])
    
    
    data = no_meal_and_meal_data(insulin_data,glucose_data)
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    model = svm.SVC(kernel='linear',C=0.1, gamma = 0.1)
    kfold = model_selection.KFold(5, True, 1)
    
    for train, test in kfold.split(x, y):
        x_train, x_test = x.iloc[train], x.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        
        model.fit(x_train, y_train)

    with open('model.pkl', 'wb') as (file):
        pickle.dump(model, file)
