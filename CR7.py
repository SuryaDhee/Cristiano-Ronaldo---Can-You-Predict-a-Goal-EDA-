#!/usr/bin/env python
# coding: utf-8

# In[675]:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,roc_auc_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
data = pd.read_csv('data.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
mode =  data['match_event_id'].mode()[0]
data['match_event_id'] = data['match_event_id'].fillna(mode)
data['location_x'] = data['location_x'].fillna(data['location_x'].mean())
data['location_y'] = data['location_y'].fillna(data['location_y'].mean())
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'remaining_min.1']) and pd.notnull(data.loc[i,'remaining_min'])):
        data.loc[i,'remaining_min.1'] = data.loc[i,'remaining_min']
    if(pd.isnull(data.loc[i,'power_of_shot.1']) and pd.notnull(data.loc[i,'power_of_shot'])):
        data.loc[i,'power_of_shot.1'] = data.loc[i,'power_of_shot']
    if(pd.isnull(data.loc[i,'knockout_match']) and (data.loc[i,'knockout_match.1']==1 or data.loc[i,'knockout_match.1']==0)):
        data.loc[i,'knockout_match'] = data.loc[i,'knockout_match.1']
data['remaining_min.1'] = data['remaining_min.1'].fillna(data['remaining_min.1'].mean())
data['remaining_min'] = data['remaining_min'].fillna(data['remaining_min'].mean())
data['power_of_shot.1'] = data['power_of_shot.1'].fillna(data['power_of_shot.1'].mean())
data['power_of_shot'] = data['power_of_shot'].fillna(data['power_of_shot'].mean())
data.drop('knockout_match.1',axis=1,inplace=True)
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'knockout_match'])):
        if(i<26198):
            data.loc[i,'knockout_match'] = 0
        else:
            data.loc[i,'knockout_match'] = 1
data['game_season'] = data['game_season'].interpolate(method ='pad',limit=10)
data['game_season'] = data['game_season'].rank(method='dense', ascending=False).astype(int)
for i in range(data.shape[0]):
    if(pd.notnull(data.loc[i,'remaining_sec.1'])):
        data.loc[i,'remaining_min.1']+=data.loc[i,'remaining_sec.1']/60
        data.loc[i,'remaining_sec.1']= data.loc[i,'remaining_sec.1']%60
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'remaining_sec.1']) and pd.notnull(data.loc[i,'remaining_sec'])):
        data.loc[i,'remaining_sec.1'] = data.loc[i,'remaining_sec']
data['remaining_sec.1'] = data['remaining_sec.1'].fillna(data['remaining_sec.1'].mean())
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'remaining_sec']) and pd.notnull(data.loc[i,'remaining_sec.1'])):
        data.loc[i,'remaining_sec'] = data.loc[i,'remaining_sec.1']
data['remaining_sec'] = data['remaining_sec'].fillna(data['remaining_sec'].mean())
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'distance_of_shot.1']) and pd.notnull(data.loc[i,'distance_of_shot'])):
        data.loc[i,'distance_of_shot.1'] = data.loc[i,'distance_of_shot']
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'distance_of_shot']) and pd.notnull(data.loc[i,'distance_of_shot.1'])):
        data.loc[i,'distance_of_shot'] = data.loc[i,'distance_of_shot.1']
data['distance_of_shot.1'] = data['distance_of_shot.1'].fillna(data['distance_of_shot.1'].mean())
data['distance_of_shot'] = data['distance_of_shot'].fillna(data['distance_of_shot'].mean())
data['area_of_shot'] = data['area_of_shot'].fillna('Center(C)')
data['area_of_shot'] = data['area_of_shot'].rank(method='dense', ascending=False).astype(int)
data['shot_basics'] = data['shot_basics'].fillna('Mid Range')
data['shot_basics'] = data['shot_basics'].rank(method='dense', ascending=False).astype(int)
data['range_of_shot'] = data['range_of_shot'].interpolate(method ='pad',limit=10)
data['range_of_shot'] = data['range_of_shot'].rank(method='dense', ascending=False).astype(int)
data.drop('team_name',axis=1,inplace = True)
data['date_of_game'] = data['date_of_game'].astype('datetime64')
data['year'] = data.date_of_game.dt.year
data['month'] = data.date_of_game.dt.month
data['day'] = data.date_of_game.dt.day
data['year'] = data['year'].interpolate(method ='pad',limit=10)
data['month'] = data['month'].interpolate(method ='pad',limit=10)
data['day'] = data['day'].interpolate(method ='pad',limit=10)
data.drop('date_of_game',axis=1,inplace=True)
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'type_of_shot']) and pd.notnull(data.loc[i,'type_of_combined_shot'])):
        data.loc[i,'type_of_shot'] = data.loc[i,'type_of_combined_shot']
data.drop('type_of_combined_shot',axis=1,inplace=True)
data['match_id'] = data['match_id'].rank(method='dense', ascending=False).astype(int)
data['team_id'] = data['team_id'].rank(method='dense', ascending=False).astype(int)
for i in range(data.shape[0]):
    try:
        data.loc[i,'home/away']= data.loc[i,'home/away'].replace('@','vs.')
    except:
        continue
data['home/away'] = data['home/away'].interpolate(method ='pad',limit=10)
data['home/away'] = data['home/away'].rank(method='dense', ascending=False).astype(int)
for i in range(data.shape[0]):
    try:
        ls = data.loc[i,'lat/lng'].split(', ')
        data.loc[i,'lat'] = ls[0]
        data.loc[i,'ling'] = ls[1]
    except:
        data.loc[i,'lat'] = None
        data.loc[i,'ling'] = None
data.drop('lat/lng',axis=1,inplace=True)
data['lat'] = data['lat'].interpolate(method ='pad',limit=10)
data['ling'] = data['ling'].interpolate(method ='pad',limit=10)
data['type_of_shot'] = data['type_of_shot'].rank(method='dense', ascending=False).astype(int)
data['lat'] = data['lat'].astype(float)
data['ling'] = data['ling'].astype(float)
for i in range(data.shape[0]):
    if(pd.isnull(data.loc[i,'shot_id_number'])):
        data.loc[i,'shot_id_number'] = data.loc[i-1,'shot_id_number']+1
test = data[pd.isnull(data['is_goal'])]
test = data.drop('is_goal',axis=1)
test.to_csv('Test.csv',index = False)
data = data[pd.notnull(data['is_goal'])]
target = data['is_goal']
data.drop('is_goal',axis=1,inplace=True)
data['is_goal'] = target
data.to_csv('CR7.csv',index=False)




X = data.iloc[:,:-1]
y = data.iloc[:,-1]


#data = pd.read_csv('CR7.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


clf = XGBClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 4,verbosity =0, random_state = 0,n_jobs=-1)
clf.fit(X_train, y_train)
y_predict = clf.predict_proba(X_test)
sub = pd.DataFrame(y_predict)
sub.drop(0,axis=1,inplace=True)
sub['shot_id_number'] = test['shot_id_number']
target = sub[1]
sub.drop(1,axis=1,inplace=True)
sub['is_goal'] = target
print(target)
sub.to_csv('Submission.csv')

#
#
#lr = [0.05,0.1,0.2,0.3,0.4,0.5]
#max_depth = [2,3,4,5,6]
#maxauc = 0
#maxd = 0
#maxlr =0
#minmae = 1000000000
#for l in lr:
#    for md in max_depth:
#        clf = XGBClassifier(n_estimators=100, learning_rate = l, max_depth = md,verbosity =0, random_state = 0,n_jobs=-1)
#        clf.fit(X_train_enc, y_train)
#        y_p = clf.predict(X_test_enc)
#        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_p)
#        auc = metrics.auc(fpr, tpr)
#        mae = mean_absolute_error(y_true, y_pred)
#        if(minmae>mae):
#            maxauc = auc
#            maxd = md
#            y_pred = y_p
#            maxlr = l
#            minmae = mae
#
#print('AUC,depth')
#print(maxauc)
#print(maxd)
#print(maxlr)
#print("MAE")
#print(minmae)
