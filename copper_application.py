import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import pickle

alignment = """

<style>
#the-title {
  text-align: center
}
</style>
"""

#set streamlit page
st.set_page_config(page_title="Industrial_copper_modeling",layout='wide')
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
padding_top = 0

def status(x,y,status_features):
   # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier()

    # Train the classifier on the training data
    rf_classifier.fit(x, y)

    # Predict the target variable for the test set
    statuss = rf_classifier.predict(status_features)

    st.write('The predicted status is: ', 'Won' if statuss == 1 else 'Lost')
  

def main():  

  df = pd.read_csv("C:\\Users\\Natarajan\\Desktop\\Dhivya\\DS\\capstone\\Industrial_copper\\copper_final.csv")
  st.write( f'<h3 style="color:#009999;">Industrial Copper Modeling Application</h3>', unsafe_allow_html=True )
  tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"])
  status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']
  item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
  country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
  application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
  product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']
  with tab2:
    col1,col2 = st.columns([5,5],gap= "medium")
    with col1:
      quantity = st.text_input("Quantity Tons",placeholder="Min")
      thickness = st.text_input("Thickness ")
      width = st.text_input("Width ")
      customer = st.text_input("Customer ID ")
      selling_price = st.text_input("Selling Price")
    with col2:
       item_type = st.selectbox("Item Type", item_type_options, key=21)
       country = st.selectbox("Country", sorted(country_options), key=31)
       application = st.selectbox("Application", sorted(application_options), key=41)
       product_ref = st.selectbox("Product Reference", product, key=51)
       predict_status = st.button(label="PREDICT STATUS") 

    
    
    #item_type_data = {'W':5, 'WI':6, 'S':3, 'Others':1, 'PL':2, 'IPL':0, 'SLAWR':4}
    # Define a mapping of categories to numeric values
    category_mapping = {'W': 0,
                        'S': 1,
                        'Others': 2,
                        'PL': 3,
                        'WI': 4,
                        'IPL': 5,
                        'SLAWR': 6}
    # dataframe for provided features
    status_data = {'quantity_tons': quantity,
                   'thickness': thickness,
                   'width': width,
                   'customer':customer,
                   'selling_price': selling_price,
                   'item type':category_mapping.get(item_type) ,
                   'country':country,
                   'application':application,
                   'product_ref':product_ref
                   }
    status_features = pd.DataFrame(status_data, index=[0])
    
     
    """def itemType_func(item_type):
          x = item_type_data.get(item_type)     
          return x"""
    x = df[['quantity tons_log','thickness_log','selling_price_log','customer','country','item type','application','width','product_ref']].values
    y = df['status'] 
    if st.button('predict status'):
        # calling function to predict status
        status(x, y, status_features)

if __name__ == "__main__":
    # This block will be executed when the script is run as the main program
    main()        
         
        
        

        
         


        

