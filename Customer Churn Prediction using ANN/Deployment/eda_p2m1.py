import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
from scipy.stats import mode


st.set_page_config(
    page_title = 'Customer Churn Prediction',
    page_icon="🧊",
    
    initial_sidebar_state='expanded',
        menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
            

    }
)


colors = sns.color_palette('PuBu')
colors_1 = sns.color_palette('PuBu')[4:7]
colors_2 = sns.color_palette('PuBu')[2:7]
colors_3 = sns.color_palette('PuBu')[1:7]
colors_4 = ['#7209B7', '#4895EF', '#560BAD', '#480CA8', '#4361EE' ]


def run(): 

    #Membuat Title
    st.title('Customer Churn Prediction')

    #Membuat sub header
    st.subheader('Exploratory Data for Customer Churn Prediction')

    image = Image.open('COVER-CHURN.png')
    st.image(image, caption = 'Churn')


    c1, c2 = st.columns((5,5))
    with c1: 
        st.write('Customer churn refers to the phenomenon where customers discontinue their relationship or subscription with a company or service provider. It represents the rate at which customers stop using a companys products or services within a specific period. Churn is an important metric for businesses as it directly impacts revenue, growth, and customer retention')
    with c2:
        st.write('In the context of the Churn dataset, the churn label indicates whether a customer has churned or not. A churned customer is one who has decided to discontinue their subscription or usage of the companys services. On the other hand, a non-churned customer is one who continues to remain engaged and retains their relationship with the company.')

    st.markdown("##")
    st.markdown("##")   
     

    st.write("On this page, you can explore the contents of the dataset. It provides an overview of the data points, features, and information related to churn customers in the companies.  ")

    st.write('<h3 style="text-align: center;">Dataframe of Customer </h3>', unsafe_allow_html=True)
    data = pd.read_csv('churn.csv')
    data.dropna(inplace=True)
    st.dataframe(data)


    

    st.write('<h3 style="text-align: center;"> Exploratory Data of Customer </h3>', unsafe_allow_html= True )

    st.markdown("##")
    
    st.sidebar.header("Please Filter Here:")

    # Update the session state variables when the user selects an option
    selected_tep1 = st.sidebar.selectbox(
        "Select to Categorical Columns:", [ 'gender',
                                            'region_category',
                                            'membership_category',
                                            'joined_through_referral',
                                            'preferred_offer_types',
                                            'medium_of_operation',
                                            'internet_option',
                                            'used_special_discount',
                                            'offer_application_preference',
                                            'past_complaint',
                                            'complaint_status',
                                            'feedback']
    ) 
    
    selected_tep2 = st.sidebar.selectbox(
        "Select to Numerical Columns:", [   'age',
                                            'days_since_last_login',
                                            'avg_time_spent',
                                            'avg_transaction_value',
                                            'avg_frequency_login_days',
                                            'points_in_wallet' ] 
    )

    
    st.markdown("##")

    def eda_meanplot(x, y, data):
        feat = x
        hue = y
        hue_type = data[hue].dtype.type
        groups = data[feat].unique()
        proportions = data.groupby(feat)[hue].mean()
        total_percentage = proportions.sum()
        proportions_percent = proportions / total_percentage * 100 
        color_palette = sns.color_palette('YlGnBu')
        ax = proportions.plot(kind='bar', color=color_palette)
        for index, value in enumerate(proportions_percent):
            ax.text(index, value, f'{value:.1f}%', ha='center', va='bottom')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

        
    def eda_countplot(x, y, data):
        feat = x
        hue = y
        hue_type = data[hue].dtype.type
        groups = data[feat].dropna().unique()
        data_cleaned = data.dropna(subset=[hue])
        proportions = data_cleaned.groupby(feat)[hue].value_counts(normalize=True).unstack()
        sns.set(style="whitegrid", rc={'figure.figsize': (10, 6)})
        color_palette = sns.color_palette('YlGnBu')
        ax = sns.countplot(x=feat, hue=hue, data=data_cleaned, palette=color_palette)
        for c in ax.containers:
            labels = [f'{proportions.loc[g, hue_type(c.get_label())]:.1%}' for g in groups]
            ax.bar_label(c, labels)
        plt.xlabel(x)
        plt.ylabel('Count')
        plt.legend(title=hue)
        ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=30, ha='right')
        plt.show()


    def eda_displot(x, y, data):
        ax = sns.kdeplot(data=data, x=x, hue=y, multiple="stack", palette ='YlGnBu', alpha=.7, fill = True)
        mode_number = data[data[y] == 1][x].value_counts().idxmax()
        median_number = np.median(data[x])
        plt.text(1.08, 0.6, f'Mode {x}: {mode_number:.2f}\nMedian {x}: {median_number:.2f}', transform=plt.gca().transAxes)
        plt.axvline(mode_number, color='red', linestyle='--', linewidth=1)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.xlabel(x)
        plt.ylabel('Count')
        plt.show()
        

    e1, e2, e3 = st.columns((2, 15, 2))
    with e2 :
        image = Image.open('CHURN-PIC2.png')
        st.image(image, use_column_width=True)
        st.caption('Churn')
    
    st.write("In this section, we can see plots made for each feature available in the dataset. The purpose of EDA is to gain valuable insights, explore data distribution, identify patterns, and understand the relationship between different features and the target variable 'Churn.")


    st.markdown(f"<h2 style='text-align: center; font-size: 20px;'> Barplot {selected_tep2} groupby {selected_tep1} </h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(facecolor='None')
    fig.set_size_inches(13 , 6)
    eda_meanplot(selected_tep1, selected_tep2 , data)
    st.pyplot(fig)


    st.markdown(f"<h2 style='text-align: center; font-size: 20px;'> {selected_tep1} by Churn Risk Score </h2>", unsafe_allow_html=True)
    fig, ax1 = plt.subplots()
    fig, ax = plt.subplots(facecolor='None')
    fig.set_size_inches(5, 4)
    eda_countplot(selected_tep1, 'churn_risk_score' , data)
    st.pyplot(fig)



    st.markdown(f"<h2 style='text-align: center; font-size: 20px;'> {selected_tep2} histogram by Churn Risk Score  </h2>", unsafe_allow_html=True)
    fig, ax1 = plt.subplots()
    fig, ax = plt.subplots(facecolor='None')
    fig.set_size_inches(5, 4)
    eda_displot(selected_tep2, 'churn_risk_score', data)
    st.pyplot(fig)




if __name__ == '__main__':
    run()

