import pickle
from pathlib import Path
#import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import time

import streamlit as st
import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, plot_roc_curve
import joblib 

st.set_page_config(page_title='Diabetes Prediction App', page_icon=':bar_chart:',layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)


#def show_loading_page():
#    st.image("diabetesgif.gif", use_column_width=True)
 #   time.sleep(2)

#def main():

    # load hashed passwords
  #  file_path = Path(__file__).parent / "hashed_pw.pkl"
 #   with file_path.open("rb") as file:
 #       hashed_passwords = pickle.load(file)

#    credentials = {
 #           "usernames":{
 #               "nanji":{
 #                   "name":"Nanji Freddy",
  #                  "password":hashed_passwords[0]
  #                  },
  #              "abdou":{
   #                 "name":"Abdouraman",
   #                 "password":hashed_passwords[1]
   #                 }            
    #            }
    #        }

   # authenticator = stauth.Authenticate(credentials,
   #     "Diabetes_Prediction_App", "abcd", cookie_expiry_days=0)

   # name, authentication_status, username = authenticator.login("Login", "main")

  #  if authentication_status == False:
    #    st.error("Username/password is incorrect")

   # if authentication_status == None:
   #     st.warning("Please enter your username and password")

  #  if authentication_status:
        
    
        
data = pd.read_csv('C46-Diabetes.csv')

# load pre-trained model from pickle file
# model = joblib.load('Final_diabetes_model_4Avril2022.pkl')
load_model = pickle.load(open('Final_diabetes_model.sav','rb'))





st.write(""" 
        
            <h1 style='text-align:center;'>DIADETECT<h1>
    """, unsafe_allow_html=True)
st.write("""
        <p style='text-align:center; margin-top:-55px;font-size:14px;color:red'>... a diabetes detection system</p>
    """ , unsafe_allow_html=True)
st.write("""
        <p style='text-align:center; margin-top:-55px;font-size:20px'>This app uses machine learning to predict the presence of diabetes based on several clinical features.</p>
    """ , unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs([":mask: Prediction",":clipboard: Data",":bar_chart: Visualisation"])

#sidebar
#authenticator.logout("Logout", "sidebar")

# st.sidebar.title(f"Welcome {name}")
st.sidebar.write(f"""
<span style='color:orange;font-size:28px;'>Welcome to diadetect<span>
""", unsafe_allow_html=True)
st.sidebar.header('User Input Parameters')
# state1 = st.sidebar.checkbox("Display Training Data",value=False)
# if state1:
#     st.subheader('Training data')
#     st.dataframe(data.describe())
# else:
#     pass

# def user_input_features():
#     pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
#     glucose = st.sidebar.slider('Glucose', 0,200, 120 )
#     blood_pressure = st.sidebar.slider('Blood Pressure', 0,122, 70 )
#     bmi = st.sidebar.slider('BMI', 0,67, 20 )
#     dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
#     age = st.sidebar.slider('Age', 21,88, 33 )
#     data ={
#         'Pregnancies': pregnancies,
#         'Glucose': glucose,
#         'BloodPressure': blood_pressure,
#         'BMI': bmi,
#         'DiabetesPedigreeFunction': dpf,
#         'Age': age
#     }
#     input_data = pd.DataFrame(data, index=[0])
#     return input_data
def user_input_features():
    # option = st.sidebar.radio("Choose Input Method", ("Upload Dataset", "Input Parameters"))
    global option
    option = st.sidebar.radio("Choose Input Method", ("Upload Dataset", "Input Parameters"))
    pregnancies = 3
    glucose = 120
    blood_pressure = 70
    bmi= 20
    dpf=0.47
    age=33
    if option == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            # Read the uploaded file into a DataFrame
            data = pd.read_csv(uploaded_file)
            data = data.drop(["Insulin","SkinThickness","Outcome"],axis=1)
            return data

    elif option == "Input Parameters":
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
        glucose = st.sidebar.slider('Glucose', 0, 200, 120)
        blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
        bmi = st.sidebar.slider('BMI', 0, 67, 20)
        dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
        age = st.sidebar.slider('Age', 21, 88, 33)

    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    }

    input_data = pd.DataFrame(data)
    return input_data


st.markdown("""
<style>
    
    .css-1kyxreq{
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .css-1v0mbdj img{
        width: 50vw !important;
        align-items:center;
    }
    .main{
        padding:0 5%
    }
    .css-1v0mbdj{
    }
    }
    
    
</style>
""", unsafe_allow_html=True)
df =user_input_features()
if df is not None:

    # prediction = model.predict(df)
    prediction = load_model.predict(df)


#main
# st.subheader('User Input parameters')

with tab1:
    st.subheader('User Input parameters')
    
    st.write(df)
    # OUTPUT
    if st.button("Predict"):
        if option == "Input Parameters" :
            output=''
            if prediction[0]==0:
                output = ' Congratulations , You are not Diabetic'
                st.success(output,icon="✅")
            else:
                output = 'You either have diabetes or are likely to have it. Please visit the doctor as soon as possible.'
                st.warning(output, icon='⚠️')
        else:
            result = pd.concat([df, pd.DataFrame({'Prediction': prediction})], axis=1)
            result['Prediction'] = result['Prediction'].replace({0: "You don't have diabetes", 1: "You have diabetes"})
            # st.write(result)
            st.dataframe(result, width=None)



with tab2:
    Graphes = st.selectbox("Data",options=("Training Data","Dataset"))
    if Graphes=="Dataset":
        st.subheader('Diabetes Dataset')
        st.write(data)
    else:
        st.subheader('Training Data')
        st.dataframe(data.describe())
with tab3:
    Graphes = st.selectbox("Analytical graph",options=(None,"Breakdown of diabetics and non-diabetics","number of wholesales in each class"
                ,"risk according to the age of each class"))
    if Graphes=="Breakdown of diabetics and non-diabetics": 
        def frequence_diabete():
            plt.figure(figsize=(8, 6))
            data['Outcome'].value_counts().plot(kind ='bar')
            plt.title("Repartition des diabetiques et non-diabetiques")
            plt.ylabel("Frequence")
            plt.tight_layout() 
            return st.pyplot()
        frequence_diabete()
    if Graphes == "number of wholesales in each class":
        def pregnancy_diabete():
            pregnancy_0 = data[data['Outcome']==0]['Pregnancies'] #ressortir la classe 0
            pregnancy_1 = data[data['Outcome']==1]['Pregnancies'] #ressortir la classe 1
            sns.set(style="darkgrid")
            sns.histplot(pregnancy_0, label='Nbre de grossesses sans diabetes',color= 'blue',kde= True ,  )
            sns.histplot(pregnancy_1, label='Nbre de grossesses avec diabetes',color ='orange',kde= True )
            plt.title('histogramme du nombre de grosseses de chaque classe')
            plt.legend()
            return st.pyplot()
        pregnancy_diabete()	
    if Graphes == "risk according to the age of each class":
        def Age_diabete():
            pregnancy_0 = data[data['Outcome']==0]['Age'] #ressortir la classe 0
            pregnancy_1 = data[data['Outcome']==1]['Age'] #ressortir la classe 1
            sns.set(style="darkgrid")
            sns.histplot(pregnancy_0, label='Age sans diabetes',color= 'blue',kde= True )
            sns.histplot(pregnancy_1, label='Age avec diabetes',color ='orange',kde= True )
            plt.title("histogramme du risque en fonction de l'age de chaque classe")
            plt.legend()
            return st.pyplot()
        Age_diabete()
    if Graphes ==None:
        st.write('Use the dropbox above to visualise data analysis')
    else:
        st.warning("No input data provided.")


# state2 = st.checkbox("Visualise Predictions Accuracy",value=False)

# if state2:

#     predict = load_model.predict(data.drop(['Outcome','Insulin','SkinThickness'],axis=1))
#     accuracy = metrics.accuracy_score(data['Outcome'],predict)
#     st.subheader('Accuracy : '+str(accuracy))

#     col1, col2 = st.columns(2)

#     with col1:
#         st.write('Confusion Matrix')
#         cm = confusion_matrix(data['Outcome'],predict)
#         # Plot the confusion matrix as heatmap
#         sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
#         st.pyplot()


#     with col2:
#         #AUC curve
#         st.write('AUC-ROC Curve')
#         # Generate AUC-ROC curve
#         fpr, tpr, thresholds = roc_curve(data['Outcome'],predict)

#         # Plot the AUC-ROC curve
#         roc_auc = plot_roc_curve(load_model, data.drop(['Outcome','Insulin','SkinThickness'],axis=1), data['Outcome'])
#         st.pyplot()
# else:
#     st.write('Use the checkbox above to visualise Predictions Accuracy')


# state3 = st.checkbox("Visualise Patients Reports",value=False)



# # COLOR FUNCTION
# colors = ['green','orange']
# sns.set_palette(colors)
# if prediction[0]==0:
#     color = colors[0]
# else:
#     color = colors[1]

# if state3:

#     st.subheader('Visualisations')
#     col1, col2 = st.columns(2)



#     with col1:
#         # Age vs Pregnancies
#         st.write('Pregnancy count Graph (Others vs Yours)')
#         fig_preg = plt.figure()
#         ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = data, hue = 'Outcome', palette = sns.color_palette())
#         ax2 = sns.scatterplot(x = df['Age'], y = df['Pregnancies'], s = 300, color = color)
#         plt.xticks(np.arange(10,100,5))
#         plt.yticks(np.arange(0,20,2))
#         plt.title('0 - Healthy & 1 - Unhealthy')
#         st.pyplot(fig_preg)
    
#         # Age vs Glucose
#         st.write('Glucose Value Graph (Others vs Yours)')
#         fig_glucose = plt.figure()
#         ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = data, hue = 'Outcome' , palette=sns.color_palette())
#         ax4 = sns.scatterplot(x = df['Age'], y = df['Glucose'], s = 300, color = color)
#         plt.xticks(np.arange(10,100,5))
#         plt.yticks(np.arange(0,220,10))
#         plt.title('0 - Healthy & 1 - Unhealthy')
#         st.pyplot(fig_glucose)

#     with col2:
#         # Age vs Bp
#         st.write('Blood Pressure Value Graph (Others vs Yours)')
#         fig_bp = plt.figure()
#         ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = data, hue = 'Outcome', palette=sns.color_palette())
#         ax6 = sns.scatterplot(x = df['Age'], y = df['BloodPressure'], s = 300, color = color)
#         plt.xticks(np.arange(10,100,5))
#         plt.yticks(np.arange(0,130,10))
#         plt.title('0 - Healthy & 1 - Unhealthy')
#         st.pyplot(fig_bp)



#         # Age vs BMI
#         st.write('BMI Value Graph (Others vs Yours)')
#         fig_bmi = plt.figure()
#         ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = data, hue = 'Outcome', palette=sns.color_palette())
#         ax12 = sns.scatterplot(x = df['Age'], y = df['BMI'], s = 300, color = color)
#         plt.xticks(np.arange(10,100,5))
#         plt.yticks(np.arange(0,70,5))
#         plt.title('0 - Healthy & 1 - Unhealthy')
#         st.pyplot(fig_bmi)


#     with col1:
#         # Age vs Dpf
#         st.write('DPF Value Graph (Others vs Yours)')
#         fig_dpf = plt.figure()
#         ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = data, hue = 'Outcome', palette=sns.color_palette())
#         ax14 = sns.scatterplot(x = df['Age'], y = df['DiabetesPedigreeFunction'], s = 300, color = color)
#         plt.xticks(np.arange(10,100,5))
#         plt.yticks(np.arange(0,3,0.2))
#         plt.title('0 - Healthy & 1 - Unhealthy')
#         st.pyplot(fig_dpf)
    
# else:
#     st.write('Use the checkbox above to visualise patients records')

#if __name__ == '__main__':
    
    # show_loading_page()
 #   main()
   
