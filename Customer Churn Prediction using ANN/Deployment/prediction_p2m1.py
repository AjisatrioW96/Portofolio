import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

#model inferencing





with open ('pipeline.pkl' , 'rb') as file_10:
    pipeline = pickle.load(file_10)

model_functional_final = tf.keras.models.load_model('model_functional_final.h5')


def run() :

    st.write('# Customer Churn Prediction ')
    background_color_3 = "#a66827"
    st.markdown(
                f"""
                <div style="background-color: {background_color_3}; padding: 10px; border-radius: 5px;">
                    <p><strong>This model prediction still needs to be improved in order to enhance its accuracy and reliability. While it provides valuable insights and predictions, it is essential to use the results with caution.</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
    st.markdown('')
    with st.form('key=Form Prediction Churn Customer'):
        st.write ('### Fill out the Following:')

        gender_options = {'Female': 'F', 'Male': 'M'}
        region_category_options = {'City': 'City', 'Town': 'Town', 'Village': 'Village'}
        membership_category_options = {'No Membership': 'No Membership', 'Platinum Membership': 'Platinum Membership', 'Basic Membership': 'Basic Membership', 'Gold Membership': 'Gold Membership', 'Silver Membership': 'Silver Membership', 'Premium Membership': 'Premium Membership'}
        joined_through_referral_options = {'No': 'No', 'Yes': 'Yes'}
        preferred_offer_types_options = {'Gift Vouchers/Coupons': 'Gift Vouchers/Coupons', 'Credit/Debit Card Offers': 'Credit/Debit Card Offers', 'Without Offers': 'Without Offers'}
        medium_of_operation_options = {'Smartphone': 'Smartphone', 'Both': 'Both', 'Desktop': 'Desktop'}
        internet_options_options ={'Fiber_Optic' : 'Fiber Optic' , 'Wi-Fi' : 'Wi-Fi', 'Mobile Data' : 'Mobile_Data'}
        used_special_discount_options = {'No': 'No', 'Yes': 'Yes'}
        offer_application_preference_options = {'Yes': 'Yes', 'No': 'No'}
        past_complain_options = {'Yes': 'Yes', 'No': 'No'}
        complaint_status_options = {'Not Applicable': 'Not Applicable', 'Solved': 'Solved', 'Unsolved': 'Unsolved', 'Solved in Follow-up': 'Solved in Follow-up', 'No Information Available': 'No Information Available'}
        feedback_options = {'No reason specified': 'No reason specified', 'Too many ads': 'Too many ads', 'Poor Website': 'Poor Website', 'Poor Customer Service': 'Poor Customer Service', 'Reasonable Price': 'Reasonable Price', 'Poor Product Quality': 'Poor Product Quality', 'User Friendly Website': 'User Friendly Website', 'Products always in Stock': 'Products always in Stock', 'Quality Customer Care': 'Quality Customer Care'}


        Name = st.text_input('Tuliskan nama customer', 'Abidin')

        gender = st.selectbox('Apa Gender Customer yang anda cari ?', list(gender_options.keys()), index=0)
        gender_value = gender_options[gender]

        region_category = st.radio('Tinggal di manakah Customer yang anda cari ?', list(region_category_options.keys()), index=0)
        region_category_value = region_category_options[region_category]

        membership_category = st.radio('Jenis Membership apa yang dimiliki oleh customer', list(membership_category_options.keys()), index=0)
        membership_category_value = membership_category_options[membership_category]

        joined_through_referral = st.selectbox('Apakah customer join melalui referral ?', list(joined_through_referral_options.keys()), index=0)
        joined_through_referral_value = joined_through_referral_options[joined_through_referral]

        preferred_offer_types = st.radio('Jenis penawaran apa yang disenangi oleh customer ?', list(preferred_offer_types_options.keys()), index=0)
        preferred_offer_types_value = preferred_offer_types_options[preferred_offer_types]

        medium_of_operation = st.radio('Gadget jenis apa yang digunakan customer dalam bertransaski ?', list(medium_of_operation_options.keys()), index=0)
        medium_of_operation_value = medium_of_operation_options[medium_of_operation]

        internet = st.radio('Jenis Internet apakah yang dipakai oleh customer ?', list(internet_options_options.keys()), index=0)
        internet_options_value = internet_options_options[internet]


        used_special_discount = st.selectbox('Apakah customer menggunakan penawaran diskon saat diberikan ?', list(used_special_discount_options.keys()), index=0)
        used_special_discount_value = used_special_discount_options[used_special_discount]

        offer_application_preference = st.radio('Apakah customer suka diberikan penawaran ?', list(offer_application_preference_options.keys()), index=0)
        offer_application_preference_value = offer_application_preference_options[offer_application_preference]

        past_complain = st.selectbox('Apakah customer suka diberikan penawaran ?', list(offer_application_preference_options.keys()), index=0)
        past_complain_value = past_complain_options[past_complain]


        complaint_status = st.radio('Apakah customer pernah mengajukan komplain ? dan apakah sudah terselesaikan ?', list(complaint_status_options.keys()), index=0)
        complaint_status_value = complaint_status_options[complaint_status]

        feedback = st.radio('Feedback apa yang diberikan oleh customer ?', list(feedback_options.keys()), index=0)
        feedback_value = feedback_options[feedback]




        age = st.number_input('Berapa umur dari customer ?', min_value=10 ,max_value=100, value=24)
        joining_date = st.date_input('Tanggal berapa customer berlangganan product kita',  min_value=datetime(2015, 1, 1), max_value=datetime(2017, 12, 31), value=datetime(2017, 1, 1) )
        days_since_last_login = st.slider('Kapan terakhir kali customer login ke dalam website (dalam hari)?',  min_value=-100, max_value=100, value=25)
        avg_time_spent = st.slider('Berapa rata rata waktu yang dihabiskan customer dalam menggunakan website (dalam menit) ?',  min_value=0, max_value=3000, value=160)
        avg_transaction_value = st.slider('Berapa rata rata total transaksi yang dikeluarkan oleh customer ?',  min_value=0, max_value=100000, value=30000)
        avg_frequency_login_days = st.number_input('Berapa lama rata rata hari customer login kedalam website?',  min_value=0, max_value=80, value=25)
        points_in_wallet = st.slider('Berapa banyak poin yang terkumpul di akun customer?',  min_value=0, max_value=3000, value=250)

        submitted = st.form_submit_button('Predict')

        #membuat data baru
        data_inf = {
                    'age' : age, 
                    'gender' : gender_value, 
                    'region_category' : region_category_value,
                    'membership_category' : membership_category_value,
                    'joining_date' : joining_date, 
                    'joined_through_referral' : joined_through_referral_value,
                    'preferred_offer_types' : preferred_offer_types_value,  
                    'medium_of_operation' : medium_of_operation_value, 
                    'internet_option' : internet_options_value, 
                    'days_since_last_login' : days_since_last_login,
                    'avg_time_spent' : avg_time_spent,
                    'avg_transaction_value' : avg_transaction_value,
                    'avg_frequency_login_days' : avg_frequency_login_days,
                    'points_in_wallet' : points_in_wallet,
                    'used_special_discount' : used_special_discount_value,
                    'offer_application_preference' : offer_application_preference_value,
                    'past_complaint' : past_complain_value,
                    'complaint_status' : complaint_status_value,
                    'feedback' : feedback_value
        }
        data_inf = pd.DataFrame([data_inf])



    if submitted: 
            #splitting data
        data_inf_transform = pipeline.transform(data_inf)
        y_pred_inf = model_functional_final.predict(data_inf_transform)
        background_color_1 = "#e63946"
        background_color_2 = "#3d85c6" 
        background_color_4 = "#70e000"
        
        if y_pred_inf == 1:
            st.write(f'###### Halo, berdasarkan dari model yang diberikan Customer dengan nama {Name} mempunyai : ')
            st.markdown(
                f"""
                <div style="background-color: {background_color_1}; padding: 10px; border-radius: 5px; ">
                    <p><strong>EXPECTED TO CHURN</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
            st.markdown(
                f"""
                <div style="background-color: {background_color_2}; padding: 10px; border-radius: 5px;">
                    <p>The training file for the CHURN dataset contains a collection of 440,882 customer records along with their respective features and churn labels. This file serves as the primary resource for training machine learning models to predict customer churn. The algorithm used in the model is based on Decision Tree, which has an accuracy of 99.7294% in predicting whether a customer will churn or not. However, even though the model has high accuracy, it is essential to recognize that it is not entirely flawless. It's important to understand that while the model performs well in terms of accuracy, there could still be room for improvement and potential challenges to consider.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
        else:
            st.write(f'###### Halo, berdasarkan dari model yang diberikan Customer dengan nama {Name} mempunyai : ')
            st.markdown(
                f"""
                <div style="background-color: {background_color_4}; padding: 10px; border-radius: 5px;">
                    <p><strong>LOW RISK OF CHURN</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')
            st.markdown(
                f"""
                <div style="background-color: {background_color_2}; padding: 10px; border-radius: 5px;">
                    <p>The training file used for this prediction contains data for churn customers from several companies. The goal is to predict whether a customer is likely to churn. The dataset consists of approximately 37,010 records with 22 features. The data serves as the training set for a machine learning model, which utilizes an Artificial Neural Network (ANN) with the architecture of Functional API. The model has achieved an impressive accuracy of 92% in predicting whether a customer is likely to churn or not. However, it's important to acknowledge that even though the model exhibits high accuracy, it may not be entirely flawless. There could be room for improvement and potential challenges to consider in its performance and predictions."</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('')


if __name__ == '__main__':
    run()
