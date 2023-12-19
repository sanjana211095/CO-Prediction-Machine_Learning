import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import pickle 

st.image("Image.jpg",width=600)
st.set_option('deprecation.showPyplotGlobalUse',False)

st.write("""
# Carbon Concentration Prediction App
This web app will predict the **carbon concentration** in air
""")

st.write('---')
st.write('**Description of Dataset**')
st.write('**CO(GT)** - True hourly averaged concentration CO in mg/m^3 (reference analyzer)')
st.write('**PT08.S1(CO)** - PT08.S1 (tin oxide) hourly averaged sensor response (nominally CO targeted)')
st.write('**C6H6(GT)** - True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)') 
st.write('**PT08.S2(NHMC)** - PT08.S2 (titania) hourly averaged sensor response (nominally NMHC targeted)')
st.write('**NOx(GT)** - True hourly averaged NOx concentration in ppb (reference analyzer)')
st.write('**PT08.S3(NOx)** - PT08.S3 (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)')
st.write('**NO2(GT)** - True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)')
st.write('**PT08.S4(NO2)** - PT08.S4 (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)')
st.write('**PT08.S5(O3)** - PT08.S5 (indium oxide) hourly averaged sensor response (nominally O3 targeted)')
st.write('**T** - Temperature in °C')
st.write('**RH** - Relative Humidity (%)')
st.write('**AH** - AH Absolute Humidity')

# Import dataset
df1=pd.read_csv("Air_Quality.csv")
# Mean
pt_co_mean = df1["PT08.S1(CO)"].mean()
c6h6_mean = df1["C6H6(GT)"].mean()
pt_nhmc_mean = df1["PT08.S2(NMHC)"].mean()
nox_mean = df1["NOx(GT)"].mean()
pt_no_mean = df1["PT08.S3(NOx)"].mean()
no2_mean = df1["NO2(GT)"].mean()
pt_no2_mean = df1["PT08.S4(NO2)"].mean()
pt_o3_mean = df1["PT08.S5(O3)"].mean()
t_mean = df1["T"].mean()
rh_mean = df1["RH"].mean()
ah_mean = df1["AH"].mean()


# Standard deviation
pt_co_std = df1["PT08.S1(CO)"].std()
c6h6_std = df1["C6H6(GT)"].std()
pt_nhmc_std = df1["PT08.S2(NMHC)"].std()
nox_std = df1["NOx(GT)"].std()
pt_no_std = df1["PT08.S3(NOx)"].std()
no2_std = df1["NO2(GT)"].std()
pt_no2_std = df1["PT08.S4(NO2)"].std()
pt_o3_std = df1["PT08.S5(O3)"].std()
t_std = df1["T"].std()
rh_std = df1["RH"].std()
ah_std = df1["AH"].std()


st.write(df1) 
st.markdown('<span style = "color:red">*Note : Ignore comma(,) it is just a streamlit issue</span>',unsafe_allow_html=True )


Columns = list(df1.columns)
st.sidebar.header('Plot setting')
plot_selection = st.sidebar.selectbox(label = 'Type of plot',options = ['Scatterplot','Boxplot'])

if plot_selection == 'Scatterplot':
    st.sidebar.subheader("Scatterplot settings")
    x_value = st.sidebar.selectbox(label = 'X-axis',options = Columns)
    y_value = st.sidebar.selectbox(label = 'Y-axis',options = Columns)
    plot = px.scatter(data_frame = df1,x = x_value,y = y_value)
    st.write(plot)

if plot_selection == 'Boxplot':
    st.sidebar.subheader('Boxplot settings')
    value = st.sidebar.selectbox(label = "Select Column",options = Columns)
    plot = px.box(df1,y = value)
    st.write(plot)

st.sidebar.subheader("Model selection")
select_model = st.sidebar.selectbox(label = 'Select Model', options = ["Linear Regression","Ridge Regression","Lasso Regression","ElasticNet"])

if select_model == 'Linear Regression':
    X = df1.drop("CO(GT)",axis=1)
    Y = df1["CO(GT)"]
    st.header('Specify Input Parameter')
    def input_features():
        PT08_CO = st.slider("**PT08.S1(CO)**",X["PT08.S1(CO)"].min(),X["PT08.S1(CO)"].max(),X["PT08.S1(CO)"].mean())
        C6H6 = st.slider("**C6H6(GT)**",X["C6H6(GT)"].min(),X["C6H6(GT)"].max(),X["C6H6(GT)"].mean())
        PT08_NHMC = st.slider("**PT08.S2(NMHC)**",X["PT08.S2(NMHC)"].min(),X["PT08.S2(NMHC)"].max(),X["PT08.S2(NMHC)"].mean())
        NOx = st.slider("**NOx(GT)**",X["NOx(GT)"].min(),X["NOx(GT)"].max(),X["NOx(GT)"].mean())
        PT08_NOx = st.slider("**PT08.S3(NOx)**",X["PT08.S3(NOx)"].min(),X["PT08.S3(NOx)"].max(),X["PT08.S3(NOx)"].mean())
        NO2 = st.slider("**NO2(GT)**",X["PT08.S3(NOx)"].min(),X["PT08.S3(NOx)"].max(),X["PT08.S3(NOx)"].mean())
        PT08_NO2 = st.slider("**PT08.S4(NO2)**",X["PT08.S4(NO2)"].min(),X["PT08.S4(NO2)"].max(),X["PT08.S4(NO2)"].mean())
        PT08_O3 = st.slider("**PT08.S5(O3)**",X["PT08.S5(O3)"].min(),X["PT08.S5(O3)"].max(),X["PT08.S5(O3)"].mean())
        T = st.slider("**T**",X["T"].min(),X["T"].max(),X["T"].mean())
        RH = st.slider("**RH**",X["RH"].min(),X["RH"].max(),X["RH"].mean())
        AH = st.slider("**AH**",X["AH"].min(),X["AH"].max(),X["AH"].mean())

       #Normalize input 
        PT08_CO_norm = (PT08_CO - pt_co_mean)/pt_co_std
        C6H6_norm = (C6H6 - c6h6_mean)/c6h6_std
        PT08_NHMC_norm = (PT08_NHMC - pt_nhmc_mean)/pt_nhmc_std
        NOx_norm = (NOx - nox_mean)/nox_std
        PT08_NOx_norm = (PT08_NOx - pt_no_mean)/pt_no_std
        NO2_norm = (NO2 - no2_mean)/no2_std
        PT08_NO2_norm = (PT08_NO2 - pt_no2_mean)/pt_no2_std
        PT08_O3_norm =  (PT08_O3 - pt_o3_mean)/pt_o3_std
        T_norm = (T - t_mean)/t_std
        RH_norm = (RH - rh_mean)/rh_std
        AH_norm = (AH - ah_mean)/rh_std
        

        data = {
            "PT08.S1(CO)" : PT08_CO_norm,
            "C6H6(GT)" : C6H6_norm,
            "PT08.S2(NMHC)" : PT08_NHMC_norm,
            "NOx(GT)" : NOx_norm,
            "PT08.S3(NOx)" : PT08_NOx_norm,
            "NO2(GT)" : NO2_norm,
            "PT08.S4(NO2)" : PT08_NO2_norm,
            "PT08.S5(O3)" : PT08_O3_norm,
            "T" : T_norm,
            "RH" : RH_norm,
            "AH" : AH_norm
        }
        feature = pd.DataFrame([data])
        return feature
    X = input_features()
    st.write(X)

    st.write("Predicted CO")

    model = pickle.load(open("model.sav","rb"))

    pred = model.predict(X)[0]
    res = "The CO(GT) of input feature is " +str(pred)
    st.success(res)




















if select_model == 'Ridge Regression':
    X = df1.drop("CO(GT)",axis=1)
    Y = df1["CO(GT)"]
    st.header('Specify Input Parameter')
    def input_features():
        PT08_CO = st.slider("**PT08.S1(CO)**",X["PT08.S1(CO)"].min(),X["PT08.S1(CO)"].max(),X["PT08.S1(CO)"].mean())
        C6H6 = st.slider("**C6H6(GT)**",X["C6H6(GT)"].min(),X["C6H6(GT)"].max(),X["C6H6(GT)"].mean())
        PT08_NHMC = st.slider("**PT08.S2(NMHC)**",X["PT08.S2(NMHC)"].min(),X["PT08.S2(NMHC)"].max(),X["PT08.S2(NMHC)"].mean())
        NOx = st.slider("**NOx(GT)**",X["NOx(GT)"].min(),X["NOx(GT)"].max(),X["NOx(GT)"].mean())
        PT08_NOx = st.slider("**PT08.S3(NOx)**",X["PT08.S3(NOx)"].min(),X["PT08.S3(NOx)"].max(),X["PT08.S3(NOx)"].mean())
        NO2 = st.slider("**NO2(GT)**",X["PT08.S3(NOx)"].min(),X["PT08.S3(NOx)"].max(),X["PT08.S3(NOx)"].mean())
        PT08_NO2 = st.slider("**PT08.S4(NO2)**",X["PT08.S4(NO2)"].min(),X["PT08.S4(NO2)"].max(),X["PT08.S4(NO2)"].mean())
        PT08_O3 = st.slider("**PT08.S5(O3)**",X["PT08.S5(O3)"].min(),X["PT08.S5(O3)"].max(),X["PT08.S5(O3)"].mean())
        T = st.slider("**T**",X["T"].min(),X["T"].max(),X["T"].mean())
        RH = st.slider("**RH**",X["RH"].min(),X["RH"].max(),X["RH"].mean())
        AH = st.slider("**AH**",X["AH"].min(),X["AH"].max(),X["AH"].mean())

        #Normalize input 
        PT08_CO_norm = (PT08_CO - pt_co_mean)/pt_co_std
        C6H6_norm = (C6H6 - c6h6_mean)/c6h6_std
        PT08_NHMC_norm = (PT08_NHMC - pt_nhmc_mean)/pt_nhmc_std
        NOx_norm = (NOx - nox_mean)/nox_std
        PT08_NOx_norm = (PT08_NOx - pt_no_mean)/pt_no_std
        NO2_norm = (NO2 - no2_mean)/no2_std
        PT08_NO2_norm = (PT08_NO2 - pt_no2_mean)/pt_no2_std
        PT08_O3_norm =  (PT08_O3 - pt_o3_mean)/pt_o3_std
        T_norm = (T - t_mean)/t_std
        RH_norm = (RH - rh_mean)/rh_std
        AH_norm = (AH - ah_mean)/rh_std
        

        data = {
            "PT08.S1(CO)" : PT08_CO_norm,
            "C6H6(GT)" : C6H6_norm,
            "PT08.S2(NMHC)" : PT08_NHMC_norm,
            "NOx(GT)" : NOx_norm,
            "PT08.S3(NOx)" : PT08_NOx_norm,
            "NO2(GT)" : NO2_norm,
            "PT08.S4(NO2)" : PT08_NO2_norm,
            "PT08.S5(O3)" : PT08_O3_norm,
            "T" : T_norm,
            "RH" : RH_norm,
            "AH" : AH_norm
        }
        feature = pd.DataFrame([data])
    # col = list(feature.columns)
    # scale = MinMaxScaler()
    # feature.loc[:,col] = scale.fit_transform(feature)
    # feature = sm.add_constant(feature)
        return feature
    X = input_features()
    st.write(X)
    st.write("Predicted CO")

    model = pickle.load(open("RidgeModel.sav","rb"))

    pred = model.predict(X)[0]
    res = "The CO(GT) of input feature is " +str(pred)
    st.success(res)

if select_model == 'Lasso Regression':
    st.header("Work in Progress....")
if select_model == 'ElasticNet':
    st.header("Work in Progress....")
