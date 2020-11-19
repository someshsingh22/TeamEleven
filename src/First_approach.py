# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import json
import datetime
import math
from random import randint

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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

"""## Importing data"""

matches = pd.read_csv("../Data/matches.csv")
matches.head()

attributes = pd.read_csv("../Data/attributes.csv")
attributes.head()

batsmen = open("../Data/batsmen.json",)
batsmen_data = json.load(batsmen)

bowlers = open("../Data/bowlers.json",)
bowlers_data = json.load(bowlers)

invtmap = open("../Data/invtmap.json",)
invtmap_data = json.load(invtmap)

scorecard = open("../Data/scorecard.json",)
scorecard_data = json.load(scorecard)

region = open("../Data/region.json",)
region_data = json.load(region)

tmap = open("../Data/tmap.json",)
tmap_data = json.load(tmap)

"""## Model 1

## Making the Database
"""


def get_matches(team_1, team_2, date, num_years=5):
    matches_team = matches[matches["Team_1"] == team_1]
    matches_team = matches_team[matches["Team_2"] == team_2]
    matches_team1 = matches[matches["Team_1"] == team_2]
    matches_team1 = matches_team1[matches["Team_2"] == team_1]
    matches_team = pd.concat([matches_team, matches_team1], axis=0)
    matches_team["Date"] = pd.to_datetime(matches_team["Date"])
    get_date = datetime.datetime.strptime(date, "%Y-%m-%d")
    min_date = datetime.datetime.strptime(
        str(get_date.year - num_years) + "-01" + "-01", "%Y-%m-%d"
    )
    matches_team_latest = matches_team[matches_team["Date"] >= min_date]
    matches_team_latest = matches_team_latest[matches_team_latest["Date"] < date]
    matches_team_latest = matches_team_latest.sort_values(by="Date")

    matches_team_latest = matches_team_latest.reset_index()
    matches_team_latest = matches_team_latest.drop(["index"], axis=1)
    return matches_team_latest


def create_Dataset(Dataset):
    Dataset["Toss"] = ""
    Dataset["Fours_team1"] = ""
    Dataset["Fours_team2"] = ""
    Dataset["Sixes_team1"] = ""
    Dataset["Sixes_team2"] = ""
    Dataset["Strike_rate_team1"] = ""
    Dataset["Strike_rate_team2"] = ""
    Dataset["Maidens_team1"] = ""
    Dataset["Maidens_team2"] = ""
    Dataset["Wickets_team1"] = ""
    Dataset["Wickets_team2"] = ""
    Dataset["Economy_rate_team1"] = ""
    Dataset["Economy_rate_team2"] = ""
    Dataset["Score_1"] = ""
    Dataset["Score_2"] = ""
    Dataset["Result"] = ""

    MatchCodes = Dataset["MatchCode"].tolist()

    for i in range(len(MatchCodes)):
        matchdata = scorecard_data[str(MatchCodes[i])]
        # Dataset = Dataset.drop(columns=['GroundCode','Team_1','Team_2'])
        if matchdata["TOSS"] == matchdata["ORDER"][0]:
            toss = 1
        else:
            toss = 2
        # Batting 1
        batting1 = matchdata["BATTING1"]
        Runs_team1 = []
        Balls_Played_team1 = []
        Fours_team1 = []
        Sixes_team1 = []
        Strike_rate_team1 = []
        for val in range(10):
            Runs_team1.append(batting1[val][2])
            Balls_Played_team1.append(batting1[val][3])
            Fours_team1.append(batting1[val][4])
            Sixes_team1.append(batting1[val][5])
            Strike_rate_team1.append(batting1[val][6])

        # Batting 2
        batting2 = matchdata["BATTING2"]
        Runs_team2 = []
        Balls_Played_team2 = []
        Fours_team2 = []
        Sixes_team2 = []
        Strike_rate_team2 = []
        for val in range(10):
            Runs_team2.append(batting2[val][2])
            Balls_Played_team2.append(batting2[val][3])
            Fours_team2.append(batting2[val][4])
            Sixes_team2.append(batting2[val][5])
            Strike_rate_team2.append(batting2[val][6])

        # Bowling 1
        bowling1 = matchdata["BOWLING1"]
        Overs_team1 = []
        Maidens_team1 = []
        Runs_given_team1 = []
        Wickets_team1 = []
        Economy_rate_team1 = []
        for val in range(10):
            if val < len(bowling1):
                Overs_team1.append(bowling1[val][1])
                Maidens_team1.append(bowling1[val][2])
                Runs_given_team1.append(bowling1[val][3])
                Wickets_team1.append(bowling1[val][4])
                Economy_rate_team1.append(bowling1[val][5])
            else:
                Overs_team1.append(-1)
                Maidens_team1.append(-1)
                Runs_given_team1.append(-1)
                Wickets_team1.append(-1)
                Economy_rate_team1.append(-1)

        # Bowling 2
        bowling2 = matchdata["BOWLING2"]
        Overs_team2 = []
        Maidens_team2 = []
        Runs_given_team2 = []
        Wickets_team2 = []
        Economy_rate_team2 = []
        for val in range(10):
            if val < len(bowling2):
                Overs_team2.append(bowling2[val][1])
                Maidens_team2.append(bowling2[val][2])
                Runs_given_team2.append(bowling2[val][3])
                Wickets_team2.append(bowling2[val][4])
                Economy_rate_team2.append(bowling2[val][5])
            else:
                Overs_team2.append(-1)
                Maidens_team2.append(-1)
                Runs_given_team2.append(-1)
                Wickets_team2.append(-1)
                Economy_rate_team2.append(-1)
        # Scores
        score = matchdata["SCORES"]
        if score[0] > score[1]:
            result = 1
        elif score[1] > score[0]:
            result = 2
        else:
            result = "Tie"

        # insert into dataset
        Dataset["Toss"].iloc[i] = toss
        Dataset["Score_1"].iloc[i] = score[0]
        Dataset["Score_2"].iloc[i] = score[1]
        Dataset["Fours_team1"].iloc[i] = Fours_team1
        Dataset["Fours_team2"].iloc[i] = Fours_team2
        Dataset["Sixes_team1"].iloc[i] = Sixes_team1
        Dataset["Sixes_team2"].iloc[i] = Sixes_team2
        Dataset["Strike_rate_team1"].iloc[i] = Strike_rate_team1
        Dataset["Strike_rate_team2"].iloc[i] = Strike_rate_team2
        Dataset["Maidens_team1"].iloc[i] = Maidens_team1
        Dataset["Maidens_team2"].iloc[i] = Maidens_team2
        Dataset["Wickets_team1"].iloc[i] = Wickets_team1
        Dataset["Wickets_team2"].iloc[i] = Wickets_team2
        Dataset["Economy_rate_team1"].iloc[i] = Economy_rate_team1
        Dataset["Economy_rate_team2"].iloc[i] = Economy_rate_team2
        Dataset["Result"].iloc[i] = result

    Dataset = Dataset.drop(columns=["Date", "GroundCode"])
    return Dataset


# Example Input
team_1 = "AUS"
team_2 = "IND"
date = "2016-01-20"

Dataset = get_matches(team_1, team_2, date)
Dataset = create_Dataset(Dataset)

Dataset.head()


"""### Data for model"""


def get_model_data(Dataset):
    venue_encoding = {"Home": 0, "Away": 1, "Neutral": 2}
    Dataset.replace({"Venue": venue_encoding}, inplace=True)
    enc = OneHotEncoder(handle_unknown="ignore")
    enc_df = pd.DataFrame(enc.fit_transform(Dataset[["Result", "Toss"]]).toarray())
    Dataset = Dataset.join(enc_df)
    Dataset.rename(columns={0: "Win1", 1: "Win2", 2: "Toss1", 3: "Toss2"}, inplace=True)
    y = Dataset[["Result", "Win1", "Win2", "Score_1", "Score_2"]]
    Dataset.drop(["Result", "Score_1", "Score_2"], inplace=True, axis=1)
    Dataset.drop(["Team_1", "Team_2"], inplace=True, axis=1)
    temp = Dataset["Venue"]
    Dataset.drop(["Win1", "Win2"], inplace=True, axis=1)
    Dataset.drop(["Venue"], inplace=True, axis=1)
    Dataset["Venue"] = ""
    Dataset["Venue"] = temp
    Dataset.drop(["Toss"], inplace=True, axis=1)
    Dataset["new"] = ""
    for i in range(len(Dataset)):
        l = []
        for j in range(1, len(Dataset.columns) - 4):
            if type(Dataset.iloc[i][j]) != list:
                l.append(Dataset.iloc[i][j])
            else:
                for k in range(len(Dataset.iloc[i][j])):
                    l.append(Dataset.iloc[i][j][k])
        Dataset["new"].iloc[i] = l

    X = Dataset[["MatchCode", "new", "Toss1", "Toss2", "Venue"]]
    return X, y


X, y = get_model_data(Dataset)

# Test Train Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 100)

y_train = y

X

"""### Predict_data"""


def allBatsmen(batting):
    batting_keys = batting.keys()
    performance = {}
    for k in batting_keys:
        for j in batting[k]:
            performance.setdefault(k, []).append(
                [j[0], j[1][1], j[1][2], j[1][3], j[1][4], j[1][5]]
            )

    performance_keys = performance.keys()

    for per in performance_keys:
        l = performance[per]
        l1 = sorted(l, reverse=True)
        performance[per] = l1

    return performance


def allBowlers(bowling):
    bowling_keys = bowling.keys()
    performance = {}
    for k in bowling_keys:
        for j in bowling[k]:
            performance.setdefault(k, []).append(
                [j[0], j[1][0], j[1][1], j[1][2], j[1][3], j[1][4]]
            )

    performance_keys = performance.keys()

    for per in performance_keys:
        l = performance[per]
        l1 = sorted(l, reverse=True)
        performance[per] = l1

    return performance


def getLatestPerf_Bat(perform, team_players, date, num_years=5):
    latest_performance = {}
    perform_keys = perform.keys()
    for k in team_players:
        for i in perform[k]:
            d = datetime.datetime.strptime(date, "%Y-%m-%d")
            d1 = datetime.datetime.strptime(i[0], "%Y-%m-%d")
            min_date = datetime.datetime.strptime(
                str(d.year - num_years) + "-01" + "-01", "%Y-%m-%d"
            )
            if d1 >= min_date and i[0] < date:
                latest_performance.setdefault(k, []).append(i)
    return latest_performance


def getAvgBat(team, index):
    final_score = []
    for k in team.keys():
        para = []
        for j in team[k]:
            para.append(j[index])

        para1 = pd.Series(para)
        f1 = para1.ewm(com=0.5, ignore_na=True, min_periods=1).mean()

        f1 = np.nanmean(f1)
        final_score.append(round(f1, 3))

    return final_score


def getLatestPerf_Bowl(perform, team_players, date, num_years=5):
    latest_performance = {}
    perform_keys = perform.keys()
    for k in team_players:
        for i in perform[k]:
            d = datetime.datetime.strptime(date, "%Y-%m-%d")
            d1 = datetime.datetime.strptime(i[0], "%Y-%m-%d")
            min_date = datetime.datetime.strptime(
                str(d.year - num_years) + "-01" + "-01", "%Y-%m-%d"
            )
            if d1 >= min_date and i[0] < date:
                latest_performance.setdefault(k, []).append(i)
    return latest_performance


def getAvgBowl(team, index):
    final_score = []
    for k in team.keys():
        para = []
        for j in team[k]:
            para.append(j[index])

        para1 = [x if x != -1 else np.nan for x in para]
        para1 = pd.Series(para1)
        f1 = para1.ewm(com=0.5).mean()
        f1 = np.nanmean(f1)
        final_score.append(round(f1, 3))

    return final_score


def adjust_length(l):
    if len(l) < 10:
        l.extend([-1] * (10 - len(l)))
    return l


def get_matches_for_test(matches):
    x = [randint(0, len(matches)) for p in range(0, 10)]
    match_codes = []
    for i in x:
        match_codes.append(matches["MatchCode"].iloc[i])
    return match_codes


def get_testing_data(X_test, match_codes, scorecard_data, matches):
    for i in range(len(match_codes)):
        match_code = match_codes[i]
        scorecard = scorecard_data[str(match_code)]
        team_1_batsmen = [scorecard["BATTING1"][i][0] for i in range(10)]
        team_2_batsmen = [scorecard["BATTING2"][i][0] for i in range(10)]
        team_1_bowlers = [
            scorecard["BOWLING1"][i][0] for i in range(len(scorecard["BOWLING1"]))
        ]
        team_2_bowlers = [
            scorecard["BOWLING2"][i][0] for i in range(len(scorecard["BOWLING2"]))
        ]
        date = matches[matches["MatchCode"] == match_code]["Date"].iloc[0]
        if scorecard["TOSS"] == scorecard["ORDER"][0]:
            toss = 1
        else:
            toss = 2
        team1_bat = getLatestPerf_Bat(bat, team_1_batsmen, date, 5)
        team2_bat = getLatestPerf_Bat(bat, team_2_batsmen, date, 5)
        team1_bowl = getLatestPerf_Bowl(bowl, team_1_bowlers, date, 5)
        team2_bowl = getLatestPerf_Bowl(bowl, team_2_bowlers, date, 5)

        row = (
            adjust_length(getAvgBat(team1_bat, 3))
            + adjust_length(getAvgBat(team2_bat, 3))
            + adjust_length(getAvgBat(team1_bat, 4))
            + adjust_length(getAvgBat(team1_bat, 4))
            + adjust_length(getAvgBat(team2_bat, 5))
            + adjust_length(getAvgBat(team2_bat, 5))
            + adjust_length(getAvgBowl(team1_bowl, 1))
            + adjust_length(getAvgBowl(team2_bowl, 1))
            + adjust_length(getAvgBowl(team1_bowl, 2))
            + adjust_length(getAvgBowl(team2_bowl, 2))
            + adjust_length(getAvgBowl(team1_bowl, 4))
            + adjust_length(getAvgBowl(team2_bowl, 4))
        )

        X_test["new"].iloc[i] = row
        X_test["Toss"].iloc[i] = toss

        score = scorecard_data[str(match_codes[i])]["SCORES"]
        if score[0] > score[1]:
            X_test["Result"].iloc[i] = 1
        elif score[1] > score[0]:
            X_test["Result"].iloc[i] = 2
        else:
            X_test["Result"].iloc[i] = 0

        X_test["Score_1"].iloc[i] = score[0]
        X_test["Score_2"].iloc[i] = score[1]

    return X_test


match_test = get_matches_for_test(matches)

bat = {}
bat = allBatsmen(batsmen_data)
print(len(bat))

bowl = {}
bowl = allBowlers(bowlers_data)
print(len(bowl))

X_test = pd.DataFrame(match_test)
X_test.columns = ["MatchCode"]
X_test["Toss"] = ""
X_test["new"] = ""
X_test["Score_1"] = ""
X_test["Score_2"] = ""
X_test["Result"] = ""

X_test = get_testing_data(X_test, match_test, scorecard_data, matches)

y_test = X_test[["Score_1", "Score_2", "Result"]]

X_train = X

enc = OneHotEncoder(handle_unknown="ignore")
enc_df = pd.DataFrame(enc.fit_transform(X_test[["Toss"]]).toarray())
X_test = X_test.join(enc_df)
X_test.rename(columns={0: "Toss1", 1: "Toss2"}, inplace=True)
X_test

len(X["new"].iloc[0])

"""### Model """

x = np.array(X["new"])
x_train = list(x)

x_test = np.array(X_test["new"])
x_test = list(x_test)


def model_eval(model, x_train, y_train, target, x_test, y_test):
    model.fit(x_train, y_train[target])
    predictions = model.predict(x_test)

    rms = sklearn.metrics.mean_squared_error(predictions, y_test[target]) ** 0.5
    print("RMS : %s" % "{0:.2%}".format(rms))

    r2 = sklearn.metrics.r2_score(predictions, y_test[target])
    print("R2 : %s" % "{0:.2%}".format(r2))
    return np.asarray(predictions)


def winner_pred(model, X_train, Y_train, x_test, y_test):
    pred1 = model_eval(model, X_train, y_train, ["Score_1"], x_test, y_test)
    pred2 = model_eval(model, X_train, y_train, ["Score_2"], x_test, y_test)

    pred = pred1 - pred2

    for i in range(len(pred)):
        if (pred[i]) > 0:
            pred[i] = 1
        else:
            pred[i] = 2

    # print(pred)
    sum = 0
    print("Model Accuracy is: ")
    for i in range(len(pred)):
        if pred[i] == y_test["Result"].iloc[i]:
            sum = sum + 1
    print(sum / len(pred))


len(y_train)

model = DecisionTreeRegressor()
winner_pred(model, x_train, y_train, x_test, y_test)

model = LinearRegression()
winner_pred(model, x_train, y_train, x_test, y_test)

model = RandomForestRegressor()
winner_pred(model, x_train, y_train, x_test, y_test)


x_train = np.array(X_train["new"])
x_train = list(x_train)

x_test = np.array(X_test["new"])
x_test = list(x_test)


model = DecisionTreeRegressor()
winner_pred(model, x_train, y_train, x_test, y_test)


"""## Model 2"""


def create_Dataset1(Dataset):
    Dataset["Toss"] = ""

    Dataset["Strike_rate_team1"] = ""
    Dataset["Strike_rate_team2"] = ""

    Dataset["Wickets_team1"] = ""
    Dataset["Wickets_team2"] = ""
    Dataset["Economy_rate_team1"] = ""
    Dataset["Economy_rate_team2"] = ""
    Dataset["Score_1"] = ""
    Dataset["Score_2"] = ""
    Dataset["Result"] = ""

    MatchCodes = Dataset["MatchCode"].tolist()

    for i in range(len(MatchCodes)):
        matchdata = scorecard_data[str(MatchCodes[i])]
        # Dataset = Dataset.drop(columns=['GroundCode','Team_1','Team_2'])
        if matchdata["TOSS"] == matchdata["ORDER"][0]:
            toss = 1
        else:
            toss = 2
        # Batting 1
        batting1 = matchdata["BATTING1"]
        Runs_team1 = []
        Balls_Played_team1 = []
        Fours_team1 = []
        Sixes_team1 = []
        Strike_rate_team1 = []
        for val in range(10):
            Runs_team1.append(batting1[val][2])
            Balls_Played_team1.append(batting1[val][3])
            Fours_team1.append(batting1[val][4])
            Sixes_team1.append(batting1[val][5])
            Strike_rate_team1.append(batting1[val][6])

        # Batting 2
        batting2 = matchdata["BATTING2"]
        Runs_team2 = []
        Balls_Played_team2 = []
        Fours_team2 = []
        Sixes_team2 = []
        Strike_rate_team2 = []
        for val in range(10):
            Runs_team2.append(batting2[val][2])
            Balls_Played_team2.append(batting2[val][3])
            Fours_team2.append(batting2[val][4])
            Sixes_team2.append(batting2[val][5])
            Strike_rate_team2.append(batting2[val][6])

        # Bowling 1
        bowling1 = matchdata["BOWLING1"]
        Overs_team1 = []
        Maidens_team1 = []
        Runs_given_team1 = []
        Wickets_team1 = []
        Economy_rate_team1 = []
        for val in range(10):
            if val < len(bowling1):
                Overs_team1.append(bowling1[val][1])
                Maidens_team1.append(bowling1[val][2])
                Runs_given_team1.append(bowling1[val][3])
                Wickets_team1.append(bowling1[val][4])
                Economy_rate_team1.append(bowling1[val][5])
            else:
                Overs_team1.append(-1)
                Maidens_team1.append(-1)
                Runs_given_team1.append(-1)
                Wickets_team1.append(-1)
                Economy_rate_team1.append(-1)

        # Bowling 2
        bowling2 = matchdata["BOWLING2"]
        Overs_team2 = []
        Maidens_team2 = []
        Runs_given_team2 = []
        Wickets_team2 = []
        Economy_rate_team2 = []
        for val in range(10):
            if val < len(bowling2):
                Overs_team2.append(bowling2[val][1])
                Maidens_team2.append(bowling2[val][2])
                Runs_given_team2.append(bowling2[val][3])
                Wickets_team2.append(bowling2[val][4])
                Economy_rate_team2.append(bowling2[val][5])
            else:
                Overs_team2.append(-1)
                Maidens_team2.append(-1)
                Runs_given_team2.append(-1)
                Wickets_team2.append(-1)
                Economy_rate_team2.append(-1)
        # Scores
        score = matchdata["SCORES"]
        if score[0] > score[1]:
            result = 1
        elif score[1] > score[0]:
            result = 2
        else:
            result = "Tie"

        # insert into dataset
        Dataset["Toss"].iloc[i] = toss
        Dataset["Score_1"].iloc[i] = score[0]
        Dataset["Score_2"].iloc[i] = score[1]

        Dataset["Strike_rate_team1"].iloc[i] = Strike_rate_team1
        Dataset["Strike_rate_team2"].iloc[i] = Strike_rate_team2

        Dataset["Wickets_team1"].iloc[i] = Wickets_team1
        Dataset["Wickets_team2"].iloc[i] = Wickets_team2
        Dataset["Economy_rate_team1"].iloc[i] = Economy_rate_team1
        Dataset["Economy_rate_team2"].iloc[i] = Economy_rate_team2
        Dataset["Result"].iloc[i] = result

    Dataset = Dataset.drop(columns=["Date", "GroundCode"])
    return Dataset


Dataset1 = get_matches(team_1, team_2, date)
Dataset1 = create_Dataset1(Dataset1)

Dataset1.head()


def get_testing_data1(X_test, match_codes, scorecard_data, matches):
    for i in range(len(match_codes)):
        match_code = match_codes[i]
        scorecard = scorecard_data[str(match_code)]
        team_1_batsmen = [scorecard["BATTING1"][i][0] for i in range(10)]
        team_2_batsmen = [scorecard["BATTING2"][i][0] for i in range(10)]
        team_1_bowlers = [
            scorecard["BOWLING1"][i][0] for i in range(len(scorecard["BOWLING1"]))
        ]
        team_2_bowlers = [
            scorecard["BOWLING2"][i][0] for i in range(len(scorecard["BOWLING2"]))
        ]
        date = matches[matches["MatchCode"] == match_code]["Date"].iloc[0]
        if scorecard["TOSS"] == scorecard["ORDER"][0]:
            toss = 1
        else:
            toss = 2
        team1_bat = getLatestPerf_Bat(bat, team_1_batsmen, date, 5)
        team2_bat = getLatestPerf_Bat(bat, team_2_batsmen, date, 5)
        team1_bowl = getLatestPerf_Bowl(bowl, team_1_bowlers, date, 5)
        team2_bowl = getLatestPerf_Bowl(bowl, team_2_bowlers, date, 5)

        row = (
            adjust_length(getAvgBat(team2_bat, 5))
            + adjust_length(getAvgBat(team2_bat, 5))
            + adjust_length(getAvgBowl(team1_bowl, 2))
            + adjust_length(getAvgBowl(team2_bowl, 2))
            + adjust_length(getAvgBowl(team1_bowl, 4))
            + adjust_length(getAvgBowl(team2_bowl, 4))
        )

        X_test["new"].iloc[i] = row
        X_test["Toss"].iloc[i] = toss

        score = scorecard_data[str(match_codes[i])]["SCORES"]
        if score[0] > score[1]:
            X_test["Result"].iloc[i] = 1
        elif score[1] > score[0]:
            X_test["Result"].iloc[i] = 2
        else:
            X_test["Result"].iloc[i] = 0

        X_test["Score_1"].iloc[i] = score[0]
        X_test["Score_2"].iloc[i] = score[1]

    return X_test


X, y = get_model_data(Dataset1)

X_train = X

match_test = get_matches_for_test(matches)

X_test = pd.DataFrame(match_test)
X_test.columns = ["MatchCode"]
X_test["Toss"] = ""
X_test["new"] = ""
X_test["Score_1"] = ""
X_test["Score_2"] = ""
X_test["Result"] = ""

X_test = get_testing_data1(X_test, match_test, scorecard_data, matches)

y_test = X_test[["Score_1", "Score_2", "Result"]]

y_train = y

"""### Model"""

x = np.array(X["new"])
x_train = list(x)

x = np.array(X_test["new"])
x_test = list(x)

model = DecisionTreeRegressor()
winner_pred(model, x_train, y_train, x_test, y_test)

model = RandomForestRegressor()
winner_pred(model, x_train, y_train, x_test, y_test)

model = LinearRegression()
winner_pred(model, x_train, y_train, x_test, y_test)
