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
from joblib import load
import keras
import tensorflow as tf


# define here all data frames



df_tournament_ID = pd.read_csv("archive/df_tournament.csv")
df_names = pd.read_csv("archive/player_names_ids.csv")
df = pd.read_csv("archive/Patrick_first.csv")
df2 = pd.read_csv("archive/Patrick_second.csv")
top20_rf = pd.read_csv("archive/Patrick_top20.csv")

# defines streamlit layout
st.title("Tennis prediction project : binary classification project")
st.sidebar.title("Table of contents")
pages=["Introduction", "Exploration", "Data Vizualization", "Strategies", "Betting Strategies", "Strategy 1", "Strategy 2",'Tennis Match Winner Predictor ', "Summary and Outlook"]
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
    and Pinnacle Sports. In general, we aim to develop a betting tool which :
    
    1. Predicts tennis matches with high accuracy.
    2. Yields a net plus when betting money on tennis matches on Bet 365 and Pinnacle 
       with their respective odds.
       
    This work will not only assist gamblers in maximizing their ROI and aid bookmakers in improving
    their odds, but it will also help tennis players and their coaches better understand
    the main features influencing the outcome of a tennis game.
    """)

#defines text on first page [second headline]
if page == pages[1]:
    st.write("### Presentation of Data")

    # Display the first 10 lines of the dataframe
    # df data frame
    st.write("Initial Data Frame")
    st.dataframe(df) 

    st.write(df.shape)
    if st.checkbox("Show Description of Initial Data Frame"):
        st.dataframe(df.describe())
    if st.checkbox("Show Columns Initial Dataframe"):
        st.image("archive/enginedf.tif", caption="Initial Dataframe", use_column_width=True)
    
    # df2 data frame

    st.write("Enginneerd Data Frame")
    st.dataframe(df2)
    st.write(df2.shape)
    if st.checkbox("Show Description of Enginneerd Data Frame"):
        st.dataframe(df2.describe())
    columnsdf2 = st.selectbox("Columns Enginneerd Data Frame", df2.columns)

    # top 20 data frame 
    top20_rf_mod = top20_rf.drop(["PlayerA_Wins", "proba_elo_PlayerB_Wins"], axis=1)
    st.write("Top 20 features for Random Forest")
    st.dataframe(top20_rf_mod)
    st.write(top20_rf_mod.shape)
    if st.checkbox("Show Description of Top 20 Random Forest"):
        st.dataframe(top20_rf_mod.describe())
# write is like print in python and st.dataframe displays the dataframe 
    if st.checkbox("Show Columns TOP 20 Random Forest"):
        st.image("archive/top20rf.png", caption='Top 20 Random Forest', use_column_width=True)

   
        

    # heading for the second page of the webapplication
if  page == pages[2]: 
    st.write("### Data Vizualization")
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
    st.write(" Two Strategies")
    
    st.image("archive/Slide1.tiff", caption='Strategies', use_column_width=True)
if  page == pages[4]:
    st.write("###Modelling by Vahid")


if  page == pages[5]:
    st.write("###Modelling by Vahid")


if  page == pages[6]:
    st.write("###Modelling by Patrick")
    st.write("#Data")
    # modeling patrick
    X = top20_rf.drop(['PlayerA_Wins', 'proba_elo_PlayerB_Wins'], axis=1)
    y = top20_rf['PlayerA_Wins']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=573)
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = load('archive/random_forest_model.joblib')
        elif classifier == 'AdaBoost':
            clf = load('archive/adaboost_model.joblib')
        elif classifier == 'DecisionTree':
            clf = load("archive/decision_tree_model.joblib")
        elif classifier == "GradientBoosting":
            clf = load('archive/gradient_boosting_model.joblib')
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    choice = ['Random Forest', 'AdaBoost', "DecisionTree", "GradientBoosting"]
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

    
    st.image("archive/most-important-ada.png", caption='Most Important ADA', use_column_width=True)
    st.image("archive/most-important-dt.png", caption='Most Important DT', use_column_width=True)
    st.image("archive/most-important-gb.png", caption='Most Important GB', use_column_width=True)
   
    st.image("archive/most-important-rf.png", caption='Most Important RF', use_column_width=True)
    
    st.image("archive/top-10-features.png", caption='Top 10 Features', use_column_width=True)



    

   
  
    #   Inserting an image from a file path
    st.image("archive/accuracy-score-models.png", caption='accuracy score models', use_column_width=True) 
    

  
    
    
    st.image("archive/betting.png", caption='Betting', use_column_width=True)
    
    # to calculate metrics of the different models, uncomment the following code

if  page == pages[7]:
   
    

    st.title('Tennis Match Winner Predictor')

    # Select player A and B by user
    player_A_name = st.selectbox('Select Player A', df_names["Name"])
    player_B_name = st.selectbox('Select Player B', df_names["Name"])
    player_A_id = df_names.loc[df_names['Name'] == player_A_name, 'ID'].iloc[0]
    player_B_id = df_names.loc[df_names['Name'] == player_B_name, 'ID'].iloc[0]

    # Lists of columns for Player A and Player B stats
    player_A_columns = ["PS_PlayerA",
                        "Wins_Per_Match_ratio_PlayerA",
                        "B365_PlayerA",
                        "proba_elo_PlayerA_Wins",
                        "Wins_Per_Match_Ratio_PlayerA_Hard",
                        "PlayerA_Pts",
                        "Wins_Per_Match_Ratio_PlayerA_Grass",
                        "Wins_Per_Match_Ratio_PlayerA_Clay",
                        "elo_PlayerA",
                        "Wins_Player_A"]

    player_B_columns = ["PS_PlayerB",
                        "Wins_Per_Match_ratio_PlayerB",
                        "B365_PlayerB",
                        "proba_elo_PlayerB_Wins",
                        "Wins_Per_Match_Ratio_PlayerB_Hard",
                        "PlayerB_Pts",
                        "Wins_Per_Match_Ratio_PlayerB_Grass",
                        "Wins_Per_Match_Ratio_PlayerB_Clay",
                        "elo_PlayerB",
                        "Wins_Player_B"]

    # Initialize empty DataFrames to store player A and player B stats
    player_A_stats = pd.DataFrame(columns=player_A_columns)
    player_B_stats = pd.DataFrame(columns=player_B_columns)

    # Flag to keep track of whether player A data is found
    player_A_found = False

    # Iterate through df2 from bottom to top for Player A
    for index, row in df2[::-1].iterrows():
        if player_A_found:
            break
        # Check if the ID matches player_A_id
        if row["PlayerA_ID"] == player_A_id:
            player_A_stats = pd.concat([player_A_stats, pd.DataFrame(row[player_A_columns]).transpose()], ignore_index=True)
            player_A_found = True
        # Check if the ID matches player_B_id
        elif row["PlayerB_ID"] == player_A_id:
            # Swap Player A and Player B stats
            for col in player_A_columns:
                if "Wins_Player_A" in col:
                    row[col], row[col.replace("Wins_Player_A", "Wins_Player_B")] = row[col.replace("Wins_Player_A", "Wins_Player_B")], row[col]
                else:
                    row[col], row[col.replace("PlayerA", "PlayerB")] = row[col.replace("PlayerA", "PlayerB")], row[col]
            player_A_stats = pd.concat([player_A_stats, pd.DataFrame(row[player_A_columns]).transpose()], ignore_index=True)
            player_A_found = True

    # Flag to keep track of whether player B data is found
    player_B_found = False

    # Iterate through df2 from bottom to top for Player B
    for index, row in df2[::-1].iterrows():
        if player_B_found:
            break
        # Check if the ID matches player_B_id
        if row["PlayerA_ID"] == player_B_id:
            # Swap Player A and Player B stats
            for col in player_B_columns:
                if "Wins_Player_B" in col:
                    row[col], row[col.replace("Wins_Player_B", "Wins_Player_A")] = row[col.replace("Wins_Player_B", "Wins_Player_A")], row[col]
                else:
                    row[col], row[col.replace("PlayerB", "PlayerA")] = row[col.replace("PlayerB", "PlayerA")], row[col]
            player_B_stats = pd.concat([player_B_stats, pd.DataFrame(row[player_B_columns]).transpose()], ignore_index=True)
            player_B_found = True
        # Check if the ID matches player_B_id
        elif row["PlayerB_ID"] == player_B_id:
            player_B_stats = pd.concat([player_B_stats, pd.DataFrame(row[player_B_columns]).transpose()], ignore_index=True)
            player_B_found = True

    # Now player_A_stats contains the stats for player A and player_B_stats contains the stats for player B
    #st.dataframe(player_A_stats)
    #st.dataframe(player_B_stats)

    # only display certain values, which will be fixed for calculation the other values will be added by the usher        
    # These data frames are used for visualization for the user
    
    # Columns related to Player A
    columns_PlayerA = [
    'elo_PlayerA',
    'PlayerA_Pts',
    'Wins_Player_A',
    'Wins_Per_Match_ratio_PlayerA',
    'Wins_Per_Match_Ratio_PlayerA_Hard',
    'Wins_Per_Match_Ratio_PlayerA_Grass',
    'Wins_Per_Match_Ratio_PlayerA_Clay'
        ]

    # Columns related to Player B
    columns_PlayerB = [
    'elo_PlayerB',
    'PlayerB_Pts',
    'Wins_Player_B',
    'Wins_Per_Match_ratio_PlayerB',
    'Wins_Per_Match_Ratio_PlayerB_Hard',
    'Wins_Per_Match_Ratio_PlayerB_Grass',
    'Wins_Per_Match_Ratio_PlayerB_Clay'
       ]

    # Select columns for player A and player B
    player_A_stats_mod = player_A_stats[columns_PlayerA]
    player_B_stats_mod = player_B_stats[columns_PlayerB]

    st.dataframe(player_A_stats_mod)
    st.dataframe(player_B_stats_mod)
    # merging user created dataframes
  
   # Merge player_B_stats into player_A_stats dataframe
   

    X_test_user = player_A_stats.merge(player_B_stats, left_index=True, right_index=True)

    
    # Define the desired column order
    desired_column_order = [
    'B365_PlayerB',
    'PS_PlayerA',
    'PS_PlayerB',
    'Wins_Per_Match_ratio_PlayerB',
    'Wins_Per_Match_ratio_PlayerA',
    'B365_PlayerA',
    'proba_elo_PlayerA_Wins',
    'Wins_Per_Match_Ratio_PlayerB_Hard',
    'Wins_Per_Match_Ratio_PlayerA_Hard',
    'PlayerA_Pts',
    'elo_PlayerB',
    'PlayerB_Pts',
    'Wins_Per_Match_Ratio_PlayerB_Clay',
    'Wins_Per_Match_Ratio_PlayerB_Grass',
    'Wins_Per_Match_Ratio_PlayerA_Grass',
    'Wins_Per_Match_Ratio_PlayerA_Clay',
    'elo_PlayerA',
    'Wins_Player_A',
    'Wins_Player_B',
    'Tournament_ID',

    'proba_elo_PlayerB_Wins'
                            ]

    # Reorder the columns of X_test_user dataframe
    X_test_user = X_test_user.reindex(columns=desired_column_order)
    # Get the Elo ratings of player A and player B from the dataframe
    elo_PlayerA = X_test_user.at[0, 'elo_PlayerA']
    elo_PlayerB = X_test_user.at[0, 'elo_PlayerB']

    # Calculate the proba_elo_PlayerA_Wins using the given formula
    proba_elo_PlayerA_Wins = 1 / (1 + 10**((elo_PlayerB - elo_PlayerA) / 400))

    # Update the value of proba_elo_PlayerA_Wins in the dataframe
    X_test_user.at[0, 'proba_elo_PlayerA_Wins'] = proba_elo_PlayerA_Wins


    # Drop the column "proba_elo_PlayerB_Wins" from X_test_user
    X_test_user.drop(columns=["proba_elo_PlayerB_Wins"], inplace=True)
    # Add a new column named "Tournament_ID" with initial value 0 to the X_test_user dataframe
    X_test_user["Tournament_ID"] = 0

    # Display an input box for the user to enter a number for Tournament_ID
   
  

    # Assuming df_tournament_ID is a dataframe mapping tournament names to their IDs
    tournament_name = st.selectbox('Select Tournament', df_tournament_ID["Tournament"])

    # Get the corresponding tournament ID from df_tournament_ID based on the selected tournament name
    tournament_name_id = df_tournament_ID.loc[df_tournament_ID['Tournament'] == tournament_name, 'Tournament_ID'].iloc[0]

    # Update the "Tournament_ID" column in the X_test_user dataframe with the selected tournament ID
    X_test_user["Tournament_ID"] = tournament_name_id
    


    # Assuming X_test_user is the dataframe containing the columns PS_PlayerA, PS_PlayerB, B365_PlayerA, and B365_PlayerB

    # Ask the user to input odds for Player A and Player B for Pinnacle
    pinnacle_odds_PlayerA = st.number_input("Enter odds for Player A in Pinnacle:", min_value=0.0)
    pinnacle_odds_PlayerB = st.number_input("Enter odds for Player B in Pinnacle:", min_value=0.0)

    # Ask the user to input odds for Player A and Player B for Bet365
    bet365_odds_PlayerA = st.number_input("Enter odds for Player A in Bet365:", min_value=0.0)
    bet365_odds_PlayerB = st.number_input("Enter odds for Player B in Bet365:", min_value=0.0)

    # Update the corresponding columns in the X_test_user dataframe with the user inputs
    X_test_user.loc[0, 'PS_PlayerA'] = pinnacle_odds_PlayerA
    X_test_user.loc[0, 'PS_PlayerB'] = pinnacle_odds_PlayerB
    X_test_user.loc[0, 'B365_PlayerA'] = bet365_odds_PlayerA
    X_test_user.loc[0, 'B365_PlayerB'] = bet365_odds_PlayerB


    # Print the merged dataframe
    X_test_user_mod = X_test_user[["PS_PlayerA","PS_PlayerB","B365_PlayerA", "B365_PlayerB"]]
    st.dataframe(X_test_user_mod)
    #testing predinction
    clf_user = load('archive/random_forest_model.joblib')
    ypred_user = clf_user.predict(X_test_user)
    
    if ypred_user == True:
        st.write(player_A_name, "will be the winner")
    else:
        st.write (player_B_name, "will be the winner")
  
    # Assuming ypred_user contains the predicted probabilities for each class

    # Get the predicted probabilities for each class
    predicted_probabilities = clf_user.predict_proba(X_test_user)

    # Get the probability of Player A winning (class 1)
    probability_player_A_wins = predicted_probabilities[0, 1]  # Assuming class 1 represents Player A winning

    # Display the probability
    st.write("Probability of",player_A_name,"winning:", probability_player_A_wins)
    st.write("Probability of",player_B_name,"winning:", 1 - probability_player_A_wins)
   


    
     




if  page == pages[8]:
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







                