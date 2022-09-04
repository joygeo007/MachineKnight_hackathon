import streamlit as st
import pandas as pd
import numpy as np
import pickle

header = st.container()
dataset = st.container()
features = st.container()

with header:
    st.title("Machineknight Hackathon!")
    st.text("In this project we will predict the rent of properties")

with dataset:
    st.header("Housing Dataset")

    housing_data = pd.read_csv('test.csv')
    if st.checkbox('Show Training Dataframe'):
        housing_data
     


with features:
    st.header("Here we'll take the inputs")
    st.text("Please enter the relevant data for rent prediction")
    sel_col, disp_col = st.columns([8,1])

    type_sel = sel_col.selectbox("Enter the type of the property:",options=["1BHK","2BHK","3BHK","4BHK","1RK","BHK4plus"], index = 0)
    locality_sel = sel_col.text_input("Enter the locality of the property:","location")
    lat_sel = sel_col.number_input("Enter the latitude of the property",step=1e-6,format="%.5f")
    long_sel = sel_col.number_input("Enter the longitude of the property",step=1e-6,format="%.5f")
    lease_sel = sel_col.selectbox("Lease_type",options=["FAMILY","BACHELOR","COMPANY","ANYONE"], index = 3)
    amenities = sel_col.write("Select the amenities provided")
    gym_sel = sel_col.checkbox("Gym")
    lift_sel = sel_col.checkbox("Lift")
    net_sel = sel_col.checkbox("Internet")
    ac_sel = sel_col.checkbox("AC")
    club_sel = sel_col.checkbox("Club")
    intercom_sel = sel_col.checkbox("Intercom")
    sev_sel = sel_col.checkbox("Servant")
    pool_sel = sel_col.checkbox("Pool")
    cpa_sel = sel_col.checkbox("CPA")
    secure_sel = sel_col.checkbox("Security")
    sc_sel = sel_col.checkbox("SC")
    gp_sel = sel_col.checkbox("GP")
    park_sel = sel_col.checkbox("Park")
    rwh_sel = sel_col.checkbox("RWH")
    HK_sel = sel_col.checkbox("HK")
    PB_sel = sel_col.checkbox("PB")
    stp_sel = sel_col.checkbox("STP")
    vp_sel = sel_col.checkbox("VP")
    swimpool_sel = sel_col.checkbox("Swimming pool")
    nego_sel = sel_col.selectbox("Is the price negotiable",options=["Yes","No"], index = 0)
    furnish_sel = sel_col.selectbox("Furnishing",options=["Fully furnished","Semi furnished","Not furnished"], index = 0)
    park_sel = sel_col.selectbox("Parking",options=["Two wheeler","Four wheeler","Both","None"], index = 0)
    propsize_sel = sel_col.slider('Property_size', min_value=1, max_value=50000,value=20,step=1)
    propage_sel = sel_col.slider('Property_age', min_value=1, max_value=400,value=2,step=1)
    propage_sel = sel_col.slider('Bathroom', min_value=1, max_value=10,value=2,step=1)
    facing_sel = sel_col.selectbox("Facing",options=["N","S","E","W","NE","NW","SE","SW"], index = 0)
    cupboard_sel = sel_col.slider('Cupboard', min_value=0, max_value=20,value=2,step=1)
    floor_sel = sel_col.slider('Floor', min_value=0, max_value=25,value=2,step=1)
    totalfloor_sel = sel_col.slider('Total_Floor', min_value=1, max_value=26,value=2,step=1)
    watersupply_sel = sel_col.selectbox("Water_supply",options=["CORP_BORE","CORPORATION","BOREWELL"], index = 0)
    buildingtype_sel = sel_col.selectbox("Building_type",options=["IF","AP","IH","GC"], index = 0)
    balconies_sel = sel_col.slider("balconies",min_value=0,max_value=9,value=0,step=1)

pkl_filename = "pickle.pkl"
with open(pkl_filename,'wb') as file:
    pickle.dump(model,file)

with open(pkl_filename,'rb') as file:
    pickle_model = pickle.load(file)

score = pickle_model.score(Xtest, Ytest)
print("Test score: {0:.2f}%".format(100*score))
Ypredict = pickle_model.predict(Xtest)



    
    




