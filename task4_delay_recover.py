"""IAB353 Model 2: leakage-aware XGBoost experiment.
Run: python train_xgboost_selected_16.py
Requires pandas, numpy, scikit-learn and xgboost.
Uses only the three source monthly CSVs beside this script.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'/'xgboost_selected_16_features'; OUT.mkdir(parents=True,exist_ok=True)
cols=['FlightDate','Cancelled','Diverted','DepDelay','ArrDelay','Distance','CRSElapsedTime','Reporting_Airline','Origin','Dest','OriginState','DestState','DayOfWeek','CRSDepTime','CRSArrTime','DepTime','Flight_Number_Reporting_Airline']
frames=[]
for month in ['01','02','12']:
    path=ROOT/f'flight_delays_2025_{month}.csv'
    if not path.exists(): path=ROOT/'US Airline Data'/path.name
    d=pd.read_csv(path,usecols=cols,low_memory=False)
    print(f'Month {month}: {len(d):,} input rows',flush=True)
    d=d[(d.Cancelled==0)&(d.Diverted==0)&(d.ArrDelay.notna())&(d.DepDelay>=15)].copy()
    frames.append(d)
df=pd.concat(frames,ignore_index=True); del frames
print('Eligible flights:',len(df),flush=True)
df['Delay_Recovery_Status']=np.select([df.ArrDelay<15,(df.ArrDelay>=15)&(df.ArrDelay<df.DepDelay)],[0,1],default=2).astype('int8')
df.drop(columns=['ArrDelay','Cancelled','Diverted'],inplace=True)
df.FlightDate=pd.to_datetime(df.FlightDate)
for c in ['DepDelay','Distance','CRSElapsedTime','DayOfWeek','CRSDepTime','CRSArrTime','DepTime','Flight_Number_Reporting_Airline']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df['Month']=df.FlightDate.dt.month
df['DayOfMonth']=df.FlightDate.dt.day
for fld in ['CRSDepTime','CRSArrTime','DepTime']:
    raw=df[fld].fillna(0).astype('int32'); mins=(raw//100)*60+raw%100
    df[fld+'_Minutes']=mins.astype('float32')
    df[fld+'_Hour']=((raw//100)%24).astype('float32')
    df[fld+'_Sin']=np.sin(2*np.pi*mins/1440).astype('float32')
    df[fld+'_Cos']=np.cos(2*np.pi*mins/1440).astype('float32')
df['Route']=df.Origin.astype(str)+'_'+df.Dest.astype(str)
df['AirlineOrigin']=df.Reporting_Airline.astype(str)+'_'+df.Origin.astype(str)
df['AirlineDest']=df.Reporting_Airline.astype(str)+'_'+df.Dest.astype(str)
df['AirlineRoute']=df.Reporting_Airline.astype(str)+'_'+df.Route
df['FlightCode']=df.Reporting_Airline.astype(str)+'_'+df.Flight_Number_Reporting_Airline.fillna(-1).astype(int).astype(str)
df['FlightNumberBand']=(df.Flight_Number_Reporting_Airline//500).fillna(-1).astype(int).astype(str)
df['FlightWeekday']=df.DayOfWeek.astype(str)+'_'+df.Reporting_Airline.astype(str)
df['Schedule_Per_Mile']=(df.CRSElapsedTime/df.Distance.clip(lower=1)).astype('float32')
df['Planned_Speed']=(60*df.Distance/df.CRSElapsedTime.clip(lower=1)).astype('float32')
df['Schedule_Extra']=(df.CRSElapsedTime-df.Distance/7.8).astype('float32')
df['Delay_Ratio']=(df.DepDelay/df.CRSElapsedTime.clip(lower=1)).astype('float32')
df['Scheduled_Arrival_After_Departure']=((df.CRSArrTime_Minutes-df.CRSDepTime_Minutes)%1440).astype('float32')
df['Actual_Departure_Lateness_Clock']=((df.DepTime_Minutes-df.CRSDepTime_Minutes+720)%1440-720).astype('float32')
# Focused, explainable Model 2 feature set (available after gate departure).
cat=['Reporting_Airline','Origin','Dest','Route','AirlineRoute']
num=['DepDelay','CRSElapsedTime','Distance','Delay_Ratio','Schedule_Per_Mile',
     'DayOfWeek','Month','CRSDepTime_Hour','DepTime_Hour','CRSArrTime_Hour','DayOfMonth']
features=cat+num
# Stratified split separately within each month, ensuring all three months are in all splits.
from sklearn.model_selection import train_test_split
rng=42
train=np.zeros(len(df),dtype=bool); val=np.zeros(len(df),dtype=bool); test=np.zeros(len(df),dtype=bool)
for month in [1,2,12]:
    ids=np.flatnonzero(df.FlightDate.dt.month.to_numpy()==month)
    labels=df.iloc[ids]['Delay_Recovery_Status']
    tr,rest=train_test_split(ids,test_size=.30,stratify=labels,random_state=rng)
    va,te=train_test_split(rest,test_size=.50,stratify=df.iloc[rest]['Delay_Recovery_Status'],random_state=rng)
    train[tr]=True; val[va]=True; test[te]=True
    print('Month',month,'train',len(tr),'validation',len(va),'test',len(te),flush=True)
for c in cat:
    s=df[c].fillna('Unknown').astype(str)
    vocab=pd.Index(s[train].unique())
    df[c]=pd.Categorical(s,categories=vocab).codes.astype('int32')
for c in num:
    s=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    df[c]=s.fillna(s[train].median()).astype('float32')
Xtr=df.loc[train,features]; Xva=df.loc[val,features]; Xte=df.loc[test,features]
ytr=df.loc[train,'Delay_Recovery_Status']; yva=df.loc[val,'Delay_Recovery_Status']; yte=df.loc[test,'Delay_Recovery_Status']
print('Split sizes',len(ytr),len(yva),len(yte),'Feature count',len(features),flush=True)
(OUT/'features.json').write_text(json.dumps({'categorical':cat,'numerical':num,'prediction_time':'after gate departure','test':'stratified 15% within each of three months; not a future-month test'},indent=2))
# Configuration selected using the mixed-month validation set; test rows stay held out.
configs=[dict(max_depth=5,min_child_weight=12,learning_rate=.065,subsample=.85,colsample_bytree=.85,reg_lambda=8),
         dict(max_depth=7,min_child_weight=25,learning_rate=.05,subsample=.85,colsample_bytree=.80,reg_lambda=12)]
models=[];rows=[]
for i,cfg in enumerate(configs,1):
    model=XGBClassifier(objective='multi:softprob',num_class=3,n_estimators=280,tree_method='hist',max_bin=128,n_jobs=6,random_state=42,eval_metric='mlogloss',early_stopping_rounds=25,**cfg)
    model.fit(Xtr,ytr,eval_set=[(Xva,yva)],verbose=False)
    pred=model.predict(Xva)
    r={'configuration':i,'validation_accuracy':accuracy_score(yva,pred),'validation_macro_f1':f1_score(yva,pred,average='macro'),'best_iteration':int(model.best_iteration),'parameters':str(cfg)}
    rows.append(r);models.append(model);print('VALIDATION',r,flush=True)
pd.DataFrame(rows).to_csv(OUT/'validation_model_comparison.csv',index=False)
idx=max(range(len(rows)),key=lambda i:(rows[i]['validation_macro_f1'],rows[i]['validation_accuracy']))
model=models[idx]; pred=model.predict(Xte); probs=model.predict_proba(Xte)
metrics={'best_configuration':idx+1,'test_rows':len(yte),'test_accuracy':float(accuracy_score(yte,pred)),'test_macro_f1':float(f1_score(yte,pred,average='macro')),'test_balanced_accuracy':float(balanced_accuracy_score(yte,pred)),'test_roc_auc_ovr':float(roc_auc_score(yte,probs,multi_class='ovr'))}
print('HELD-OUT ALL-MONTHS TEST:',json.dumps(metrics,indent=2),flush=True)
print(classification_report(yte,pred,digits=4),flush=True)
(OUT/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
(OUT/'classification_report.txt').write_text(classification_report(yte,pred,digits=4))
pd.DataFrame(confusion_matrix(yte,pred),index=['Actual_0','Actual_1','Actual_2'],columns=['Predicted_0','Predicted_1','Predicted_2']).to_csv(OUT/'confusion_matrix.csv')
pd.DataFrame({'feature':features,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
model.save_model(str(OUT/'xgboost_model.json'))
