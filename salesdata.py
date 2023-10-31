import pickle
import streamlit as st

model = pickle.load(open('sales_data.sav','rb'))

st.title('sales data')
Product_ean = st.number_input('Product ean')
Price_Each = st.number_input('Price Each')
Order_ID = st.number_input('margin')
Quantity_Ordered = st.number_input('Quantity Ordered')

predict = ''

if st.button('sales data'):
  predict = model.predict(
      [[Product_ean, Price_Each, margin, Quantity_Ordered]]
  )
  st.write('sales data:', predict)
