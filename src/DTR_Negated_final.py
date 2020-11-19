# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# Access data store
data_store = pd.HDFStore("processed_data.h5")

# Retrieve data using key
match_df = data_store["preprocessed_df"]
data_store.close()

match_df.head()

match_df["Winner_team"] = match_df["Winner"]

for ind in match_df.index:
    if match_df["Winner"][ind] == match_df["Team_1"][ind]:
        match_df["Winner_team"][ind] = 1
    elif match_df["Winner"][ind] == match_df["Team_2"][ind]:
        match_df["Winner_team"][ind] = 2
    else:
        match_df["Winner_team"][ind] = 0

match_df["Winner_team"].value_counts()

match_df.head()

np.random.seed(60)

"""##Calculating Net Run Rate

###Import Data
"""

attributes = pd.read_csv("../Data/attributes.csv")
attributes.head()

scorecard = open("../Data/scorecard.json",)
scorecard_data = json.load(scorecard)

tmap = open("../Data/tmap.json",)
tmap_data = json.load(tmap)

"""###Get NNR"""

match = match_df.copy()

match["NRR_team1"] = ""
match["NRR_team2"] = ""

name_to_code = {
    "Afghanistan": "AFG",
    "Australia": "AUS",
    "Bangladesh": "BAN",
    "England": "ENG",
    "India": "IND",
    "Ireland": "IRE",
    "Kenya": "KEN",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Pakistan": "PAK",
    "South Africa": "SAF",
    "Scotland": "SCO",
    "Sri Lanka": "SRL",
    "West Indies": "WIN",
    "Zimbabwe": "ZIM",
}

skip_keys = ["1282", "1761", "1765", "1770", "1862", "1866", "2528"]


def check_allOut(scorecard_data, matchCode, team_num):

    bat = "BATTING" + str(team_num)

    dismissal = [i[1] for i in scorecard_data[matchCode][bat]]

    if "not out" in dismissal or "" in dismissal:
        return False

    return True


def get_totalOvers(scorecard_data, matchCode, team_num):

    bat = "BATTING" + str(team_num)

    balls = [i[3] for i in scorecard_data[matchCode][bat] if i[3] != -1]

    overs = sum(balls) / 6
    return overs


for ind in match.index:
    if match["Winner_team"][ind] == 0:
        match["NRR_team1"][ind] = 0
        match["NNR_team2"][ind] = 0
    else:
        team_num = 2
        match_code = str(match["MatchCode"][ind])
        if match_code in skip_keys:
            continue
        order = scorecard_data[match_code]["ORDER"]
        if name_to_code[order[1]] == match["Team_2"][ind]:
            team_num = 1
        runRate_team1 = match["Score_1"][ind] / 50
        if check_allOut(scorecard_data, match_code, team_num):
            runRate_team2 = match["Score_2"][ind] / 50
        else:
            if match["Winner_team"][ind] == 2:
                runRate_team2 = match["Score_2"][ind] / get_totalOvers(
                    scorecard_data, match_code, team_num
                )
            else:
                runRate_team2 = match["Score_2"][ind] / 50

        match["NRR_team1"][ind] = runRate_team1 - runRate_team2
        match["NRR_team2"][ind] = runRate_team2 - runRate_team1

match.head()

len(match)

match = match[~match["MatchCode"].isin(skip_keys)]

len(match)

"""###Store the NNR dataframe"""

# Save the final dataframe
data_store = pd.HDFStore("nnr_data.h5")

data_store["preprocessed_df"] = match
data_store.close()

"""#Flipped Dataset"""

# Access data store
data_store = pd.HDFStore("nnr_data.h5")

# Retrieve data using key
match_df = data_store["preprocessed_df"]
data_store.close()

match_df.head()

match_flipped_df = match_df.copy()

for ind in match_flipped_df.index:
    match_flipped_df["Team_1"][ind], match_flipped_df["Team_2"][ind] = (
        match_df["Team_2"][ind],
        match_df["Team_1"][ind],
    )
    match_flipped_df["Score_1"][ind], match_flipped_df["Score_2"][ind] = (
        match_df["Score_2"][ind],
        match_df["Score_1"][ind],
    )
    match_flipped_df["NRR_team1"][ind], match_flipped_df["NRR_team2"][ind] = (
        match_df["NRR_team2"][ind],
        match_df["NRR_team1"][ind],
    )

    if match_df["TOSS"][ind] == 1:
        match_flipped_df["TOSS"][ind] = 2
    else:
        match_flipped_df["TOSS"][ind] = 1

    if match_df["Venue"][ind] == 1:
        match_flipped_df["Venue"][ind] = 2
    elif match_df["Venue"][ind] == 2:
        match_flipped_df["Venue"][ind] = 1

for ind in match_flipped_df.index:
    if match_flipped_df["Winner"][ind] == match_flipped_df["Team_1"][ind]:
        match_flipped_df["Winner_team"][ind] = 1
    else:
        match_flipped_df["Winner_team"][ind] = 2

match_flipped_df.head()

# Access data store
data_store = pd.HDFStore("processed_data.h5")

# Retrieve data using key
match_df = data_store["preprocessed_df"]
data_store.close()

frames = [match_df, match_flipped_df]

final_df = pd.concat(frames)

final_df.head()

len(final_df)

# Save the final dataframe
data_store = pd.HDFStore("flipped_data.h5")

data_store["flip_df"] = final_df
data_store.close()

"""####Flipped dataframe"""

# Access data store
data_store = pd.HDFStore("flipped_data.h5")

# Retrieve data using key
flipped_df = data_store["flip_df"]
data_store.close()

flipped_df.head()

len(flipped_df)

"""#Models using Flipped Data"""

enc = OneHotEncoder(handle_unknown="ignore")

enc_df = pd.DataFrame(enc.fit_transform(flipped_df[["Winner_team", "TOSS"]]).toarray())

flipped_df = flipped_df.join(enc_df)
flipped_df.head()

flipped_df.rename(columns={0: "Win1", 1: "Win2", 2: "Toss1", 3: "Toss2"}, inplace=True)

labelencoder = LabelEncoder()

flipped_df["Team_1Enc"] = labelencoder.fit_transform(flipped_df["Team_1"])
flipped_df["Team_2Enc"] = labelencoder.fit_transform(flipped_df["Team_2"])

flipped_df.head()

X = flipped_df[
    ["Date", "Team_1Enc", "Team_2Enc", "Venue", "GroundCode", "TOSS", "Toss1", "Toss2"]
].copy()
y = flipped_df[
    ["Winner_team", "Win1", "Win2", "Score_1", "Score_2", "NRR_team1", "NRR_team2"]
].copy()

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=100
)

X_train.head()

y_train.head()

"""##Training on Scores"""


def print_model_scores(model, data, y, predictors, target):
    """
    A generic function to generate the performance report of the
    model in question on the data passed to it
    Args:
        model: ML Model to be checked
        data: data on which the model needs to pe trained
        y: data containing the target variables
        predictors: independent feature variable
        target: target variable
    """
    model.fit(data[predictors], y[target])
    predictions = model.predict(data[predictors])
    rms = sklearn.metrics.mean_squared_error(predictions, y[target]) ** 0.5
    print("RMS : %s" % "{0:.2%}".format(rms))
    r2 = sklearn.metrics.r2_score(predictions, y[target])
    print("R2 : %s" % "{0:.2%}".format(r2))
    return np.asarray(predictions)


def winner_prediction(model, data, y, predictors, winner):
    """
    A generic function to predict the winner for the model in question 
    
    Args:
        model: ML Model to be checked
        data: data on which the model needs to be trained
        y: data containing the target variables
        predictors: independent feature variable
        winner: winning team
  """
    pred1 = print_model_scores(model, X_train, y_train, predictor_var, ["Score_1"])
    pred2 = print_model_scores(model, X_train, y_train, predictor_var, ["Score_2"])

    pred = pred1 - pred2

    for i in range(len(pred)):
        if (pred[i]) > 0:
            pred[i] = 1
        else:
            pred[i] = 2

    print("Model Accuracy is: ")
    print(sum(1 for x, y in zip(pred, winner) if x == y) / len(winner))


"""##Model1 - Toss + GroundCode """

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = RandomForestRegressor(n_estimators=100, random_state=0)
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = LinearRegression()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = DecisionTreeRegressor()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = LassoCV()
winner_prediction(model, X_train, y_train, predictor_var, winner)

"""##Model2 - Toss + GroundCode + Venue"""

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = RandomForestRegressor(n_estimators=100, random_state=0)
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = LinearRegression()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = DecisionTreeRegressor()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = LassoCV()
winner_prediction(model, X_train, y_train, predictor_var, winner)

"""##Training on NRR"""


def print_model_scores(model, data, y, predictors, target):
    """
    A generic function to generate the performance report of the
    model in question on the data passed to it using cross-validation
    
    Args:
        model: ML Model to be checked
        data: data on which the model needs to pe trained
        predictors: independent feature variable
        target: target variable
    """
    model.fit(data[predictors], y[target])
    predictions = model.predict(data[predictors])
    # scores = cross_val_score(model, data[predictors], y[target], scoring="neg_mean_squared_error", cv=5)
    # print('Cross-Validation Score :{}'.format(np.sqrt(-scores)))
    rms = sklearn.metrics.mean_squared_error(predictions, y[target]) ** 0.5
    print("RMS : %s" % "{0:.2%}".format(rms))
    # print(f"Average RMSE: {np.sqrt(-scores).mean()}")
    r2 = sklearn.metrics.r2_score(predictions, y[target])
    print("R2 : %s" % "{0:.2%}".format(r2))
    return np.asarray(predictions)


def winner_prediction(model, data, y, predictors, winner):
    pred = print_model_scores(model, X_train, y_train, predictor_var, ["NRR_team1"])

    for i in range(len(pred)):
        if (pred[i]) > 0:
            pred[i] = 1
        elif pred[i] == 0:
            pred[i] = 0
        else:
            pred[i] = 2

    print("Model Accuracy is: ")
    print(sum(1 for x, y in zip(pred, winner) if x == y) / len(winner))


"""##Model1 - Toss + GroundCode """

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = RandomForestRegressor(n_estimators=100, random_state=0)
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = LinearRegression()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = DecisionTreeRegressor()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS"]
model = LassoCV()
winner_prediction(model, X_train, y_train, predictor_var, winner)

"""##Model2 - Toss + GroundCode + Venue"""

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = RandomForestRegressor(n_estimators=100, random_state=0)
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = LinearRegression()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = DecisionTreeRegressor()
winner_prediction(model, X_train, y_train, predictor_var, winner)

winner = y_train["Winner_team"]
predictor_var = ["Team_1Enc", "Team_2Enc", "GroundCode", "TOSS", "Venue"]
model = LassoCV()
winner_prediction(model, X_train, y_train, predictor_var, winner)
