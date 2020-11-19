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

matches_df = pd.read_csv("../Data/matches.csv")

matches_df.head()

len(matches_df)

"""## Data Preprocessing"""

matches_df["Team_1"].value_counts()

matches_df["Team_2"].value_counts()

f = open("../Data/scorecard.json",)
data = json.load(f)

df = pd.DataFrame(data)

df = df.transpose()

df.head()

len(df)

df = df.drop(["BATTING1", "BOWLING1", "BATTING2", "BOWLING2"], axis=1)

df.head()

df["MatchCode"] = df.index

df.head()

df = df.astype({"MatchCode": int})

df.info()

match_df = matches_df.merge(df, on="MatchCode")

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

match_df.replace({"TOSS": name_to_code}, inplace=True)
match_df.head()

match_df[["team1", "team2"]] = pd.DataFrame(
    match_df.ORDER.tolist(), index=match_df.index
)
match_df.head()

match_df[["score1", "score2"]] = pd.DataFrame(
    match_df.SCORES.tolist(), index=match_df.index
)
match_df.head()

match_df = match_df.drop(["ORDER", "SCORES"], axis=1)
match_df.head()

match_df.replace({"team1": name_to_code, "team2": name_to_code}, inplace=True)
match_df.head()

match_df["Score_1"] = match_df["score1"]
match_df["Score_2"] = match_df["score2"]
match_df.head()

for ind in match_df.index:
    if match_df["Team_1"][ind] == match_df["team1"][ind]:
        match_df["Score_1"][ind] = match_df["score1"][ind]
        match_df["Score_2"][ind] = match_df["score2"][ind]
    else:
        match_df["Score_1"][ind] = match_df["score2"][ind]
        match_df["Score_2"][ind] = match_df["score1"][ind]

match_df.head()

match_df = match_df.drop(["team1", "team2", "score1", "score2"], axis=1)
match_df.head()

venue_encoding = {"Home": 0, "Away": 1, "Neutral": 2}

match_df.replace({"Venue": venue_encoding}, inplace=True)
match_df.head()

match_df["Winner"] = match_df["TOSS"]

for ind in match_df.index:
    if match_df["Score_1"][ind] > match_df["Score_2"][ind]:
        match_df["Winner"][ind] = match_df["Team_1"][ind]
    else:
        match_df["Winner"][ind] = match_df["Team_2"][ind]

for ind in match_df.index:
    if match_df["TOSS"][ind] == match_df["Team_1"][ind]:
        match_df["TOSS"][ind] = 1
    else:
        match_df["TOSS"][ind] = 2

match_df.head()

# Save the final dataframe
data_store = pd.HDFStore("processed_data.h5")

data_store["preprocessed_df"] = match_df
data_store.close()

"""##Data Analysis"""

# Access data store
data_store = pd.HDFStore("processed_data.h5")

# Retrieve data using key
match_df = data_store["preprocessed_df"]
data_store.close()

np.random.seed(42)

match_df.head()

ss = match_df["TOSS"] == match_df["Winner"]
ss.groupby(ss).size()

"""From the above numbers, it does seem that there might be a slight correlation between winning the toss and winning the match"""

sns.countplot(ss)

df_group = match_df.groupby("GroundCode")

df_group.head()

df_group["Winner"].value_counts()[:9].sort_index(ascending=False).plot(kind="barh")

"""According to the analysis, there seems to be clear correlation between Australia playing a game in ground 1 and Australia winning. It also seems that Bangladesh is unable to perform well when playing in Ground 1. Let's use Causal Inference techniques to confirm that the ground influences the winning team."""

labelencoder = LabelEncoder()

match_df["Winner_Enc"] = labelencoder.fit_transform(match_df["Winner"])
match_df.head()

match_df["Ground_Enc"] = labelencoder.fit_transform(match_df["GroundCode"])
match_df.head()

from PyIF import te_compute as te

rand = np.random.RandomState(seed=23)

TE = te.te_compute(
    np.array(match_df["Ground_Enc"]),
    np.array(match_df["Winner_Enc"]),
    k=1,
    embedding=1,
    safetyCheck=False,
    GPU=False,
)

print(TE)

"""The transfer entropy value indicates a small positive causal relationship of the ground on the winning team"""

match_df = match_df.drop(["Ground_Enc", "Winner_Enc"], axis=1)
match_df.head()

match_df["Winner_team"] = match_df["Winner"]
match_df.head()

for ind in match_df.index:
    if match_df["Winner"][ind] == match_df["Team_1"][ind]:
        match_df["Winner_team"][ind] = 1
    else:
        match_df["Winner_team"][ind] = 2

match_df.head()

sum(match_df["Venue"] == match_df["Winner_team"])

"""The winning team was playing on the home turf approximately 30% of the time

#Models
"""

enc = OneHotEncoder(handle_unknown="ignore")

enc_df = pd.DataFrame(enc.fit_transform(match_df[["Winner_team", "TOSS"]]).toarray())

match_df = match_df.join(enc_df)
match_df.head()

match_df.rename(columns={0: "Win1", 1: "Win2", 2: "Toss1", 3: "Toss2"}, inplace=True)

labelencoder = LabelEncoder()

match_df["Team_1Enc"] = labelencoder.fit_transform(match_df["Team_1"])
match_df["Team_2Enc"] = labelencoder.fit_transform(match_df["Team_2"])

match_df.head()

X = match_df[
    ["Date", "Team_1Enc", "Team_2Enc", "Venue", "GroundCode", "TOSS", "Toss1", "Toss2"]
].copy()
y = match_df[["Winner_team", "Win1", "Win2", "Score_1", "Score_2"]]

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=100
)

X_train.head()

y_train.head()


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

# Access data store
data_store = pd.HDFStore("nnr_data.h5")

# Retrieve data using key
match_df = data_store["preprocessed_df"]
data_store.close()

enc = OneHotEncoder(handle_unknown="ignore")

enc_df = pd.DataFrame(enc.fit_transform(match_df[["Winner_team", "TOSS"]]).toarray())

match_df = match_df.join(enc_df)
match_df.head()

match_df.rename(columns={0: "Win1", 1: "Win2", 2: "Toss1", 3: "Toss2"}, inplace=True)

labelencoder = LabelEncoder()

match_df["Team_1Enc"] = labelencoder.fit_transform(match_df["Team_1"])
match_df["Team_2Enc"] = labelencoder.fit_transform(match_df["Team_2"])

match_df.head()

X = match_df[
    ["Date", "Team_1Enc", "Team_2Enc", "Venue", "GroundCode", "TOSS", "Toss1", "Toss2"]
].copy()
y = match_df[
    ["Winner_team", "Win1", "Win2", "Score_1", "Score_2", "NRR_team1", "NRR_team2"]
].copy()

# Test Train Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=100
)

X_train.head()

y_train.head()


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
