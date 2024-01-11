import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd 
from PIL import Image
from imblearn.over_sampling import SMOTE 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
import pickle 

#import model 
xgb = pickle.load(open('XGB.pkl','rb'))

#load dataset
data = pd.read_csv('Stroke Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Prediksi Stroke')

html_layout1 = """
<br>
<div style="background-color:red ;border-radius: 25px;box-shadow: -3px -3px 2px rgba(0,0,0,0.4); padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Stroke Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['XGB','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Stroke</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())
    
sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)
    st.header('**Visualisasi Data**')
    groupby_column=st.selectbox("Which Categrical Variabels would you like to visualisasion",
      (
        'stroke','work_type','gender','age'
      ),
    )
    st.subheader(f'Graph showing the distributin of {groupby_column}')
    fig= sns.displot(data,x=groupby_column,hue='stroke',multiple="dodge",shrink=0.8)
    st.pyplot(fig)

#Handling missing value dengan mean data
data['bmi'].fillna(math.floor(data['bmi'].mean()),inplace=True)
#Membuang data Other
data=data[data['gender']!='Other']
#Membuang coloum yg tidak di perlukan
data=data.drop(['id','Residence_type'],axis=1)
#Transformasi data

encode = LabelEncoder()
data['gender']=encode.fit_transform(data['gender'].values)
data['work_type']=encode.fit_transform(data['work_type'].values)
data['ever_married']=encode.fit_transform(data['ever_married'].values)
data['smoking_status']=encode.fit_transform(data['smoking_status'].values)

#train test split
#SMOTE
#Dilakukan SMOTE untuk menangani data imbalance
X  = data.drop('stroke',axis=1)
y = data['stroke']
sm= SMOTE(random_state=30)
#sampling SMOTE
X_sm,y_sm= sm.fit_resample (X,y)

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, train_size=0.3  , random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    gender = st.sidebar.slider('Kelamin',0,1,1)
    age = st.sidebar.slider('Usia',0.8,100.0,1.0)
    hypertensi = st.sidebar.slider('Darah Tinggi',0,1,0)
    heart = st.sidebar.slider('Penyakit Jantung',0,1,0)
    married = st.sidebar.slider('Pernah Menikah',0,1,0)
    work=st.sidebar.slider('Type Pekerjaan',0,3,1)
    glucose=st.sidebar.slider('Rata-rata Gula Darah',55.0,280.0,1.0)
    bmi = st.sidebar.slider('BMI',10.3,100.0,25.0)
    smoke = st.sidebar.slider('Status Merokok', 0,3,0)  
    user_report_data = {
        'gender':gender,
        'age':age,
        'hypertension':hypertensi,
        'heart_disease':heart,
        'ever_married':married,
        'work_type':work,
        'avg_glucose_level':glucose,
        'bmi':bmi,
        'smoking_status':smoke
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = xgb.predict(user_data)
xgb_score = accuracy_score(y_test,xgb.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena Stroke'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(xgb_score*100)+'%')