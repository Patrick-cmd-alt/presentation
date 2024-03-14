import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
# define here all data frames





df = pd.read_csv("archive/Patrick_first.csv")
df2 = pd.read_csv("archive/Patrick_second.csv")
top20_rf = pd.read_csv("archive/Patrick_top20.csv")

# defines streamlit layout
st.title("Tennis prediction project : binary classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Exploration", "Data Vizualization","Betting Strategies", "Modelling1", "Modelling2","Summary and Outlook", "Demo", "Test Page1", "Test Page 2", "Test Page 3"]
page=st.sidebar.radio("Go to", pages)

# first page introduction to tennis betting project.

if  page == pages[0]:
    
    st.write("""
    In the following work, we aim to design a machine learning model, employing
    standard ML algorithms like Decision Tree, Random Forest, Support Vector Machine (SVM), 
    and more advanced deep learning models like Convolutional Neural Network (CNN), capable 
    of predicting the outcomes of tennis matches played on the ATP tour. We aim to better 
    understand which features influence the outcome of a tennis game. Additionally, we 
    intend to develop a betting strategy that minimizes investment losses and maximizes 
    return on investment by placing bets on the odds provided by two bookmakers: Bet365 
    and Pinnacle Sports. In general, we aim to develop a betting tool which:
    
    1. Predicts tennis matches with high accuracy.
    2. Yields a net plus when betting money on tennis matches on Bet 365 and Pinnacle 
       with their respective odds.
       
    This work, carried out by Vahid Toomani with a scientific background in math and
    physics, and Patrick Schall with a scientific background in molecular biology, will
    not only assist gamblers in maximizing their ROI and aid bookmakers in improving
    their odds, but it will also help tennis players and their coaches better understand
    the main features influencing the outcome of a tennis game.
    """)

#defines text on first page [second headline]
if page == pages[1]:
    st.write("### Presentation of Data")

    # Display the first 10 lines of the dataframe
    # df data frame
    st.write("Initial Data Frame")
    st.dataframe(df.head(10))
    st.write("Info of Initial Data Frame")
    # info of 
    st.dataframe(df.info())
    st.write(df.shape)
    st.dataframe(df.describe())
    # df2 data frame

    st.write("Enginneerd Data Frame")
    st.dataframe(df2.head(20))
    st.write("Info of Engineered Data Frame")
    
    # info 0f
    st.dataframe(df2.info())
    st.write(df2.shape)
    st.dataframe(df2.describe())

    # top 20 data frame 
    st.write("Top 20 features for Random Forest")
    st.dataframe(top20_rf.head(20))
    
    st.write("Info of Top 20 features for Random Forest")
    st.dataframe(top20_rf.info())
    st.write(top20_rf.shape)
    st.dataframe(top20_rf.describe())

# write is like print in python and st.dataframe displays the dataframe 
   
# greates a checkbox to show non null characters
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())

# heading for the second page of the webapplication
if  page == pages[2]: 
    st.write("### DataVizualization")
#   Inserting an image from a file path
 
    
    
    st.image("archive/match-count-year-dist.png", caption='Match Count Year Distribution', use_column_width=True)
    st.image("archive/elo-rates-dist.png", caption='Elo Rates Distribution', use_column_width=True)
    st.image("archive/top-25-elo-rates.png", caption='Top 25 Elo Rates', use_column_width=True)
    st.image("archive/bottom-25-elo-rates.png", caption='Bottom 25 Elo Rates', use_column_width=True)
    st.image("archive/atp-point-dist.png", caption='ATP Point Distribution', use_column_width=True)
    
    st.image("archive/elo-rate-field-type-dist.png", caption='Elo Rate Field Type Distribution', use_column_width=True)
   
    st.image("archive/match-count-field-type-dist.png", caption='Match Count Field Type Distribution', use_column_width=True)
   
    

    st.image("archive/ps-odds-dist.png", caption='PS Odds Distribution', use_column_width=True)
    st.image("archive/b365-odds-dist.png", caption='B365 Odds Distribution', use_column_width=True)
    st.image("archive/top-10-elo-rates-indoor-carpet.png", caption='Top 10 Elo Rates Indoor Carpet', use_column_width=True)
    st.image("archive/top-10-elo-rates-indoor-clay.png", caption='Top 10 Elo Rates Indoor Clay', use_column_width=True)
    st.image("archive/top-10-elo-rates-indoor-hard.png", caption='Top 10 Elo Rates Indoor Hard', use_column_width=True)
    st.image("archive/top-10-elo-rates-outdoor-clay.png", caption='Top 10 Elo Rates Outdoor Clay', use_column_width=True)
    st.image("archive/top-10-elo-rates-outdoor-grass.png", caption='Top 10 Elo Rates Outdoor Grass', use_column_width=True)
    st.image("archive/top-10-elo-rates-outdoor-hard.png", caption='Top 10 Elo Rates Outdoor Hard', use_column_width=True)
    st.image("archive/top-25-elo-rates.png", caption='Top 25 Elo Rates', use_column_width=True)
    st.image("archive/top-40-players-match-wins.png", caption='Top 40 Players Match Wins', use_column_width=True)
    st.image("archive/top-40-players-tournament-wins.png", caption='Top 40 Players Tournament Wins', use_column_width=True)


if  page == pages[3]:
    st.write("###Modelling by Vahid")


if  page == pages[4]:
    st.write("###Modelling by Vahid")


if  page == pages[5]:
    st.write("#Data")
    # modeling patrick
    st.write("###Modelling by Patrick")
    st.image("archive/most-important-ada.png", caption='Most Important ADA', use_column_width=True)
    st.image("archive/most-important-dt.png", caption='Most Important DT', use_column_width=True)
    st.image("archive/most-important-gb.png", caption='Most Important GB', use_column_width=True)
   
    st.image("archive/most-important-rf.png", caption='Most Important RF', use_column_width=True)
    
    st.image("archive/top-10-features.png", caption='Top 10 Features', use_column_width=True)
   
    X = top20_rf.drop(['PlayerA_Wins', 'proba_elo_PlayerB_Wins'], axis=1)
    y = top20_rf['PlayerA_Wins']


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=573)

    # Initialize the models
    dt_model2 = DecisionTreeClassifier()
    rf_model2 = RandomForestClassifier()
    ab_model2 = AdaBoostClassifier()
    gb_model2 = GradientBoostingClassifier()

    # Fit the models
    dt_model2.fit(X_train, y_train)
    rf_model2.fit(X_train, y_train)
    ab_model2.fit(X_train, y_train)
    gb_model2.fit(X_train, y_train)
    # predicting with the model
    ypred_dt2=dt_model2.predict(X_test)
    ypred_rf2=rf_model2.predict(X_test)
    ypred_ab2=ab_model2.predict(X_test)
    ypred_gb2=gb_model2.predict(X_test)

    

    # Calculate accuracy scores
    accuracy_dt = accuracy_score(y_test, ypred_dt2)
    accuracy_rf = accuracy_score(y_test, ypred_rf2)
    accuracy_ab = accuracy_score(y_test, ypred_ab2)
    accuracy_gb = accuracy_score(y_test, ypred_gb2)

    

    # Save Decision Tree model
    dump(dt_model2, 'decision_tree_model.joblib')

    # Save Random Forest model
    dump(rf_model2, 'random_forest_model.joblib')

    # Save AdaBoost model
    dump(ab_model2, 'adaboost_model.joblib')

    # Save Gradient Boosting model
    dump(gb_model2, 'gradient_boosting_model.joblib')


    # Display accuracy scores
    st.write("Decision Tree Classifier Accuracy:", accuracy_dt)
    st.write("Random Forest Classifier Accuracy:", accuracy_rf)
    st.write("AdaBoost Classifier Accuracy:", accuracy_ab)
    st.write("Gradient Boosting Classifier Accuracy:", accuracy_gb)
    #   Inserting an image from a file path
    st.image("archive/accuracy-score-models.png", caption='accuracy score models', use_column_width=True) 
    st.image("archive/confusion-matrix.png", caption='Confusion Matrix', use_column_width=True)

  
    
    
    st.image("archive/betting.png", caption='Betting', use_column_width=True)
    
    # to calculate metrics of the different models, uncomment the following code
    """
   

    # Calculate accuracy for each model
    accuracy_dt = accuracy_score(y_test, ypred_dt2)
    accuracy_rf = accuracy_score(y_test, ypred_rf2)
    accuracy_ab = accuracy_score(y_test, ypred_ab2)
    accuracy_gb = accuracy_score(y_test, ypred_gb2)

    # Calculate precision for each model
    precision_dt = precision_score(y_test, ypred_dt2)
    precision_rf = precision_score(y_test, ypred_rf2)
    precision_ab = precision_score(y_test, ypred_ab2)
    precision_gb = precision_score(y_test, ypred_gb2)

    # Calculate recall for each model
    recall_dt = recall_score(y_test, ypred_dt2)
    recall_rf = recall_score(y_test, ypred_rf2)
    recall_ab = recall_score(y_test, ypred_ab2)
    recall_gb = recall_score(y_test, ypred_gb2)

    # Calculate F1-score for each model
    f1_dt = f1_score(y_test, ypred_dt2)
    f1_rf = f1_score(y_test, ypred_rf2)
    f1_ab = f1_score(y_test, ypred_ab2)
    f1_gb = f1_score(y_test, ypred_gb2)

    # Print the metrics for each model
    print("Decision Tree:")
    print("Accuracy:", accuracy_dt)
    print("Precision:", precision_dt)
    print("Recall:", recall_dt)
    print("F1 Score:", f1_dt)
    print()

    print("Random Forest:")
    print("Accuracy:", accuracy_rf)
    print("Precision:", precision_rf)
    print("Recall:", recall_rf)
    print("F1 Score:", f1_rf)
    print()

    print("AdaBoost:")
    print("Accuracy:", accuracy_ab)
    print("Precision:", precision_ab)
    print("Recall:", recall_ab)
    print("F1 Score:", f1_ab)
    print()

    print("Gradient Boosting:")
    print("Accuracy:", accuracy_gb)
    print("Precision:", precision_gb)
    print("Recall:", recall_gb)
    print("F1 Score:", f1_gb)

    """ 
    
     




if  page == pages[6]:
    st.write("""
                We were able to train five different models that can correctly predict three out
            of four tennis matches. Furthermore, we identified the most important features for
            correctly predicting the outcome of a tennis match. Unfortunately, the prediction
            accuracy of about 75% is not enough to generate a positive return on investment
            (ROI) using the odds from Pinnacle Sports. We also attempted to train our models
            on matches where both players have a high difference in Elo ratings, making it easier
            for the algorithm to correctly predict the winner (Data not shown). Although this
            strategy significantly improves the prediction accuracy, the odds on those games
            were so low that we wouldn’t achieve any positive ROI by betting on them. Since
            bookmakers’ odds are designed to ensure a positive net income for the bookmaker
            37
            FIG. 25. Confusion matrix of Decision Tree, Random Forest, AdaBoost and Gradient
            Boosting. The confusion matrix has been produced with the predicted values for each
            model compared to the y_test (Player A wins) set.
            in the long run, we should be cautious about including them in our model training.
            The odds set by bookmakers inherently favor the bookmaker and may negatively
            influence our prediction model. For future prediction models, we should not rely
            on bookmaker odds and instead focus more on the Elo system as well as calculated
            features such as a player’s specific win ratio against an opponent. It might also be
            beneficial to aggregate important features into a single variable, such as the win
            ratio per match at a specific tournament and round. Another way to improve our
            models would be to utilize advanced neural networks like recurrent neural networks,
            especially Long Short-Term Memory (LSTM) networks, which incorporate time
            variables into their calculations and could enhance our prediction model.
            .
    """)


if  page == pages[7]:
    st.write("Here comes the demo of the model")
   


if  page == pages[8]:
    st.write("###Modelling by Vahid")
    st.write("This Page is just for testing!")
    st.image("archive/confusion-matrix-nn.png", caption='Confusion Matrix - NN', use_column_width=True)
    st.image("archive/dataframes-patrick.png", caption='Dataframes - Patrick', use_column_width=True)
    st.image("archive/most-important-nn.png", caption='Most Important NN', use_column_width=True)
    st.image("archive/nn-model.png", caption='NN Model', use_column_width=True)

     
    print("Type of top20_rf:", type(top20_rf))  # Check the type of top20_rf
    print("Shape of top20_rf:", top20_rf.shape)  # Check the shape of top20_rf
    print("Head of top20_rf:")  # Print the head of top20_rf
    print(top20_rf.head())

    # Display the DataFrame and its info
    st.write("Top 20 features for Random Forest")
    st.dataframe(top20_rf.head(20))
    st.write("Info of Top 20 features for Random Forest")
    st.write(top20_rf.info())
if  page == pages[9]:
    st.write("###Modelling by Vahid")
    
    # Display the first 10 lines of the dataframe
    # df data frame
    st.write("Initial Data Frame")
    
    st.write("Info of Initial Data Frame")
    # info of 
    st.dataframe(df.info())
    st.write(df.shape)
    st.dataframe(df.describe())
    # df2 data frame

    st.write("Enginneerd Data Frame")
   
    st.write("Info of Engineered Data Frame")
    
    # info 0f
    st.dataframe(df2.info())
    st.write(df2.shape)
    st.dataframe(df2.describe())

    # top 20 data frame 
    st.write("Top 20 features for Random Forest")
    
    
    st.write("Info of Top 20 features for Random Forest")
    st.dataframe(top20_rf.info())
    st.write(top20_rf.shape)
    st.dataframe(top20_rf.describe())

if  page == pages[10]:

    # THIS PAGE IS FOR TESTING CHOICE BOXES

    def prediction(classifier):
         X = top20_rf.drop(['PlayerA_Wins', 'proba_elo_PlayerB_Wins'], axis=1)
         y = top20_rf['PlayerA_Wins']


         # Train-test split
         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=573)
         if classifier == 'Random Forest':
             clf = RandomForestClassifier()
         elif classifier == 'AdaBoost':
             clf = AdaBoostClassifier()
         elif classifier == 'DecisionTree':
            clf = DecisionTreeClassifier()
         elif classifier == "GradientBoosting":
             clf = GradientBoostingClassifier()
         clf.fit(X_train, y_train)
         return clf

  
    
    
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))
        
    choice = ['Random Forest', 'AdaBoost', 'DecisionTree, GradientBoosting']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))