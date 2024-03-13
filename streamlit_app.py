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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# define here all data frames





df = pd.read_csv("archive/Patrick_first.csv")
df2 = pd.read_csv("archive/Patrick_second.csv")
top20_rf = pd.read_csv("archive/Patrick_top20.csv")

# defines streamlit layout
st.title("Tennis prediction project : binary classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Exploration", "DataVizualization", "Modelling1","Modelling2","Summary and Outlook", "Test Page"]
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
if  page == pages[1]:
    st.write("### Presentation of Data")

    #display the first 10 lines of the dataframe
    st.dataframe(df.head(20))
    st.dataframe(df2.head(20))
    st.dataframe(top20_rf.head(20))

# write is like print in python and st.dataframe displays the dataframe 
    st.write(df.shape)
    st.dataframe(df.describe())
# greates a checkbox to show non null characters
    if st.checkbox("Show NA") :
        st.dataframe(df.isna().sum())

# heading for the second page of the webapplication
if  page == pages[2] : 
    st.write("### DataVizualization")
#   Inserting an image from a file path
 
    st.image("archive/betting.png", caption='Match count', use_column_width=True)


    #diplaying a countplot calculated from the data frame
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)
    gig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)

    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)

    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)


if  page == pages[3] : 
    # preproccesing and modeling of data
    st.write("### Modelling")
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf
if  page == pages[4]:
    st.write("#Data")
    # modeling patrick
   
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

    # Display accuracy scores
    st.write("Decision Tree Classifier Accuracy:", accuracy_dt)
    st.write("Random Forest Classifier Accuracy:", accuracy_rf)
    st.write("AdaBoost Classifier Accuracy:", accuracy_ab)
    st.write("Gradient Boosting Classifier Accuracy:", accuracy_gb)
    #   Inserting an image from a file path
    st.image("archive/betting.png", caption='Match count', use_column_width=True)
   

    # Displaying each image individually
    st.image("archive/accuracy-score-models.png", caption='Match count', use_column_width=True)
    st.image("archive/betting.png", caption='Match count', use_column_width=True)
    st.image("archive/confusion-matrix-nn.png", caption='Match count', use_column_width=True)
    st.image("archive/confusion-matrix.png", caption='Match count', use_column_width=True)
    st.image("archive/dataframes-patrick.png", caption='Match count', use_column_width=True)
    st.image("archive/most-important-ada.png", caption='Match count', use_column_width=True)
    st.image("archive/most-important-dt.png", caption='Match count', use_column_width=True)
    st.image("archive/most-important-gb.png", caption='Match count', use_column_width=True)
    st.image("archive/most-important-nn.png", caption='Match count', use_column_width=True)
    st.image("archive/most-important-rf.png", caption='Match count', use_column_width=True)
    st.image("archive/nn-model.png", caption='Match count', use_column_width=True)
    st.image("archive/top-10-features.png", caption='Match count', use_column_width=True)
    st.image("archive/top-40-players-match-wins.png", caption='Match count', use_column_width=True)
    st.image("archive/top-40-players-tournament-wins.png", caption='Match count', use_column_width=True)

if  page == pages[5]:
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


if  page == pages[6]:
     st.write("This Page is just for testing!")

