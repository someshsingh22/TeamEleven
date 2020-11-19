# -*- coding: utf-8 -*-

import numpy as np
import torch
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as LE
import bisect
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split

np.random.seed(22)
torch.manual_seed(22)

with open("../Data/batsmen.json", "r") as f:
    batsmen = json.load(f)
with open("../Data/bowlers.json", "r") as f:
    bowlers = json.load(f)
batsmen = {k: [x for x in v if x[1][1] >= 0] for k, v in batsmen.items()}
batsmen = {k: sorted(v, key=lambda x: x[0]) for k, v in batsmen.items() if v}
bowlers = {k: sorted(v, key=lambda x: x[0]) for k, v in bowlers.items() if v}


def getBatScores(scores):
    # runs, balls, boundaries, contribs, out
    array = []
    for score in scores:
        date = score[0]
        _, runs, balls, fours, sixes, _, contrib = score[1]
        boundaries = fours + sixes * 1.5
        array.append((date, np.array([runs, balls, boundaries, contrib])))
    return array


def getBowlScores(scores):
    # overs, maidens, runs, wickets, contribs
    array = []
    for score in scores:
        date = score[0]
        overs, maidens, runs, wickets, _, contrib = score[1]
        overs = int(overs) + (overs - int(overs)) * 10 / 6
        array.append((date, np.array([overs, maidens, runs, wickets, contrib])))
    return array


batsmen_scores = {k: getBatScores(v) for k, v in batsmen.items()}
bowlers_scores = {k: getBowlScores(v) for k, v in bowlers.items()}

_batsmen_scores = {k: {_v[0]: _v[1] for _v in v} for k, v in batsmen_scores.items()}
_bowlers_scores = {k: {_v[0]: _v[1] for _v in v} for k, v in bowlers_scores.items()}

att = pd.read_csv("../Data/attributes.csv")
att["BatHand"] = 0 + (att["Bats"].str.find("eft") > 0)
att["BowlHand"] = 0 + (att["Bowls"].str.find("eft") > 0)
att["BowlType"] = 0 + (
    (att["Bowls"].str.find("ast") > 0) | (att["Bowls"].str.find("edium") > 0)
)


def getBatStats(scores):
    dates, scorelist = [score[0] for score in scores], [score[1] for score in scores]
    scorelist = np.array(scorelist)
    cumscores = np.cumsum(scorelist, axis=0)
    innings = np.arange(1, cumscores.shape[0] + 1)
    average = cumscores[:, 0] / innings
    sr = cumscores[:, 0] / (cumscores[:, 1] + 1)
    contrib = cumscores[:, 3] / innings
    stats = np.array([innings, average, sr, contrib]).T
    return [datetime.strptime(date, "%Y-%m-%d") for date in dates], stats


def getBowlStats(scores):
    dates, scorelist = [score[0] for score in scores], [score[1] for score in scores]
    scorelist = np.array(scorelist)
    cumscores = np.cumsum(scorelist, axis=0)
    overs = cumscores[:, 0]
    overs = overs.astype("int32") + 10 / 6 * (overs - overs.astype("int32"))
    runs = cumscores[:, 2]
    economy = runs / overs
    wickets = cumscores[:, 3]
    average = wickets / (runs + 1)
    sr = wickets / overs
    contrib = cumscores[:, 4] / np.arange(1, cumscores.shape[0] + 1)
    stats = np.array([overs, average, economy, sr, contrib]).T
    return [datetime.strptime(date, "%Y-%m-%d") for date in dates], stats


batsmen_stats = {key: getBatStats(getBatScores(v)) for key, v in batsmen.items()}
bowlers_stats = {key: getBowlStats(getBowlScores(v)) for key, v in bowlers.items()}

with open("../Data/scorecard.json", "r") as f:
    scorecards = json.load(f)
position = dict()
for code, match in scorecards.items():
    for pos, batsmen in enumerate(match["BATTING1"]):
        if batsmen[0] in position:
            position[batsmen[0]].append(pos + 1)
        else:
            position[batsmen[0]] = [pos + 1]
    for pos, batsmen in enumerate(match["BATTING2"]):
        if batsmen[0] in position:
            position[batsmen[0]].append(pos + 1)
        else:
            position[batsmen[0]] = [pos + 1]

position = {int(k): max(set(v), key=v.count) for k, v in position.items()}
for missing in set(att["Code"]) - set(position.keys()):
    position[missing] = 0

with open("../Data/region.json", "r") as f:
    region = json.load(f)
with open("../Data/tmap.json", "r") as f:
    tmap = json.load(f)

matches = pd.read_csv("../Data/matches.csv")
att["BatPos"] = att["Code"].apply(lambda x: position[x])
matches["GroundCode"] = matches["GroundCode"].apply(lambda x: region[str(x)])
matches = matches[pd.to_datetime(matches["Date"], format="%Y-%m-%d") > "1990-01-01"]
df_cards = pd.DataFrame(scorecards).transpose()
df_cards = df_cards[df_cards.index.astype(int).isin(matches["MatchCode"])]
matches = matches[matches["MatchCode"].isin(df_cards.index.astype(int))]

att = pd.get_dummies(att, columns=["BatPos"])
le = {
    "GC": LE(),
    "Team": LE(),
    "Venue": LE(),
}
le["Team"].fit((matches["Team_1"].tolist()) + (matches["Team_2"].tolist()))
matches["Team_1"] = le["Team"].transform(matches["Team_1"])
matches["Team_2"] = le["Team"].transform(matches["Team_2"])
matches["Venue"] = le["Venue"].fit_transform(matches["Venue"])
matches["GroundCode"] = le["GC"].fit_transform(matches["GroundCode"])
matches

patts = att[
    [
        "BatHand",
        "BowlHand",
        "BowlType",
        "BatPos_0",
        "BatPos_1",
        "BatPos_2",
        "BatPos_3",
        "BatPos_4",
        "BatPos_5",
        "BatPos_6",
        "BatPos_7",
        "BatPos_8",
        "BatPos_9",
        "BatPos_10",
    ]
].values
pcodes = att["Code"].tolist()
attdict = dict()
for i, pc in enumerate(pcodes):
    attdict[pc] = patts[i]

df_cards["MatchCode"] = df_cards.index.astype(int)
matches = matches.sort_values(by="MatchCode")
df_cards = df_cards.sort_values(by="MatchCode")
df_cards.reset_index(drop=True, inplace=True)
matches.reset_index(drop=True, inplace=True)
df_cards["BAT2"] = le["Team"].transform(df_cards["ORDER"].apply(lambda x: tmap[x[1]]))
df_cards["BAT1"] = le["Team"].transform(df_cards["ORDER"].apply(lambda x: tmap[x[0]]))
df_cards["RUN1"] = df_cards["SCORES"].apply(lambda x: x[0])
df_cards["RUN2"] = df_cards["SCORES"].apply(lambda x: x[1])
df_cards["TOSS"] = le["Team"].transform(df_cards["TOSS"].apply(lambda x: tmap[x]))
df = pd.merge(matches, df_cards)
df["PLAYERS1"] = df["BATTING1"].apply(lambda x: [y[0] for y in x])
df["PLAYERS2"] = df["BATTING2"].apply(lambda x: [y[0] for y in x])

_BAT1, _BAT2, _BOW1, _BOW2 = (
    df["PLAYERS1"].tolist(),
    df["PLAYERS2"].tolist(),
    [[_x[0] for _x in x] for x in df["BOWLING1"].tolist()],
    [[_x[0] for _x in x] for x in df["BOWLING2"].tolist()],
)
for i in range(len(_BAT1)):
    try:
        _BAT1[i].append(list(set(_BOW2[i]) - set(_BAT1[i]))[0])
        _BAT2[i].append(list(set(_BOW1[i]) - set(_BAT2[i]))[0])
    except:
        pass
df["PLAYERS1"], df["PLAYERS2"] = _BAT1, _BAT2
df = df[
    [
        "Date",
        "Team_1",
        "Team_2",
        "Venue",
        "GroundCode",
        "TOSS",
        "BAT1",
        "BAT2",
        "RUN1",
        "RUN2",
        "PLAYERS1",
        "PLAYERS2",
    ]
]

df = df[
    df["PLAYERS1"].apply(lambda x: len(x) == 11)
    & df["PLAYERS2"].apply(lambda x: len(x) == 11)
]
df.reset_index(drop=True, inplace=True)

Team_1, Team_2, BAT1, BAT2, BOWL1, BOWL2 = [], [], [], [], [], []
for t1, t2, b1, b2 in zip(
    df["Team_1"].tolist(),
    df["Team_2"].tolist(),
    df["BAT1"].tolist(),
    df["BAT2"].tolist(),
):
    if b1 == t1:
        Team_1.append(t1)
        Team_2.append(t2)
    else:
        Team_1.append(t2)
        Team_2.append(t1)
df["Team_1"] = Team_1
df["Team_2"] = Team_2
df.drop(["BAT1", "BAT2", "Venue"], axis=1, inplace=True)


def getStats(code, date):
    _date = datetime.strptime(date, "%Y-%m-%d")
    if code in batsmen_stats:
        i = bisect.bisect_left(batsmen_stats[code][0], _date) - 1
        if i == -1:
            bat = np.zeros(4)
        else:
            bat = batsmen_stats[code][1][i]
    else:
        bat = np.zeros(4)

    if code in bowlers_stats:
        i = bisect.bisect_left(bowlers_stats[code][0], _date) - 1
        if i == -1:
            bowl = np.zeros(5)
        else:
            bowl = bowlers_stats[code][1][i]
    else:
        bowl = np.zeros(5)
    if int(code) in attdict:
        patt = attdict[int(code)]
    else:
        patt = np.zeros(14)
    stats = np.concatenate([bat, bowl, patt])
    return stats


def getScores(code, date):
    if code in _batsmen_scores and date in _batsmen_scores[code]:
        bat = _batsmen_scores[code][date]
    else:
        bat = np.zeros(4)
    if code in _bowlers_scores and date in _bowlers_scores[code]:
        bowl = _bowlers_scores[code][date]
    else:
        bowl = np.zeros(5)
    return np.concatenate([bat, bowl])


P1, P2, Dates = df["PLAYERS1"].tolist(), df["PLAYERS2"].tolist(), df["Date"].tolist()
PStats1, PStats2 = (
    [[getStats(p, date) for p in team] for team, date in zip(P1, Dates)],
    [[getStats(p, date) for p in team] for team, date in zip(P2, Dates)],
)
PScores1, PScores2 = (
    [[getScores(p, date) for p in team] for team, date in zip(P1, Dates)],
    [[getScores(p, date) for p in team] for team, date in zip(P2, Dates)],
)


def getNRR(matchcode):
    card = scorecards[matchcode]
    run1, run2 = card["SCORES"]
    overs = sum([int(b[1]) + 10 / 6 * (b[1] - int(b[1])) for b in card["BOWLING2"]])
    allout = not (
        len(card["BATTING2"][-1][1]) < 2 or ("not" in card["BATTING2"][-1][1])
    )
    if allout:
        overs = 50
    return abs((run1 / 50) - (run2 / overs))


df["NRR"] = matches["MatchCode"].apply(lambda x: getNRR(str(x)))
df["TEAM1WIN"] = 0
df["TEAM1WIN"][df["RUN1"] > df["RUN2"]] = 1
df_0 = df[df["TEAM1WIN"] == 0]
df_1 = df[df["TEAM1WIN"] == 1]
df_0["NRR"] = -df_0["NRR"]
df = (df_0.append(df_1)).sort_index()

nPStats1, nPStats2, nPScores1, nPScores2 = (
    np.array(PStats1),
    np.array(PStats2),
    np.array(PScores1),
    np.array(PScores2),
)

StatMaxes = np.max(np.concatenate([nPStats1, nPStats2]), axis=(0, 1))
dfStats_N1 = nPStats1 / StatMaxes
dfStats_N2 = nPStats2 / StatMaxes
ScoreMaxes = np.max(np.concatenate([nPScores1, nPScores2]), axis=(0, 1))
dfScores_N1 = nPScores1 / ScoreMaxes
dfScores_N2 = nPScores2 / ScoreMaxes
NRRMax = np.max(df["NRR"])
df["NRR"] = df["NRR"] / NRRMax

nnPStats1 = np.concatenate([dfStats_N1, dfStats_N2], axis=0)
nnPStats2 = np.concatenate([dfStats_N2, dfStats_N1], axis=0)
nnPScores1 = np.concatenate([dfScores_N1, dfScores_N2], axis=0)
nnPScores2 = np.concatenate([dfScores_N2, dfScores_N1], axis=0)
_NRR = np.concatenate([df["NRR"].values, -df["NRR"].values])

train_idx, test_idx = train_test_split(np.arange(2 * len(df)), test_size=0.1)

import torch.nn as nn
import torch
from torch import optim


class AE(nn.Module):
    def __init__(self, input_shape=12, output_shape=1, hidden=16, dropout=0.2):
        super(AE, self).__init__()
        self.hidden = hidden
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.player_encoder = nn.Sequential(
            nn.Linear(input_shape, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )

        self.score_regressor = nn.Sequential(nn.Linear(hidden, 9), nn.Tanh(),)

        self.decoder = nn.Sequential(nn.Linear(hidden, input_shape))

        self.team_encoder = nn.Sequential(
            nn.Linear(11 * hidden, hidden * 4), nn.Tanh(), nn.Dropout(dropout),
        )

        self.nrr_regressor = nn.Sequential(
            nn.Linear(hidden * 8, hidden * 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, output_shape),
            nn.Tanh(),
        )

    def forward(self, x1, x2):
        encoded1, decoded1, scores1 = [], [], []
        encoded2, decoded2, scores2 = [], [], []
        for i in range(11):
            e1 = self.player_encoder(x1[:, i, :])
            d1 = self.decoder(e1)
            e2 = self.player_encoder(x2[:, i, :])
            d2 = self.decoder(e2)
            noise = (0.1 ** 0.5) * torch.randn(e1.size())
            e1, e2 = e1 + noise, e2 + noise
            scores1.append(self.score_regressor(e1))
            scores2.append(self.score_regressor(e2))
            encoded1.append(e1)
            decoded1.append(d1)
            encoded2.append(e2)
            decoded2.append(d2)
        team1, team2 = (
            self.team_encoder(torch.cat(tuple(encoded1), axis=1)),
            self.team_encoder(torch.cat(tuple(encoded2), axis=1)),
        )
        out = self.nrr_regressor(torch.cat((team1, team2), axis=1))
        decoded = torch.cat(tuple(decoded1 + decoded2), axis=1)
        scores1 = torch.cat(tuple(scores1), axis=1)
        scores2 = torch.cat(tuple(scores2), axis=1)
        return decoded, out, scores1, scores2


model = AE(dropout=0.3)
criterion = nn.MSELoss()
ED_Loss_train, NRR_Loss_train, Player_Loss_train = [], [], []
ED_Loss_test, NRR_Loss_test, Player_Loss_test = [], [], []
optimizer = optim.RMSprop(model.parameters(), lr=1e-4,)
epochs = 8200
for epoch in range(1, epochs + 1):
    model.train()
    inputs1 = torch.FloatTensor(nnPStats1[:, :, :12][train_idx])
    inputs2 = torch.FloatTensor(nnPStats2[:, :, :12][train_idx])
    outputs = torch.FloatTensor(_NRR[train_idx].reshape(-1, 1))
    optimizer.zero_grad()
    decoded, out, scores1, scores2 = model(inputs1, inputs2)
    inp = (inputs1).view(train_idx.shape[0], -1), (inputs2).view(train_idx.shape[0], -1)
    loss1 = criterion(decoded, torch.cat(inp, axis=1))
    loss2 = criterion(out, outputs)
    loss3 = criterion(
        scores1, torch.FloatTensor(nnPScores1[train_idx]).view(train_idx.shape[0], -1)
    )
    loss4 = criterion(
        scores2, torch.FloatTensor(nnPScores2[train_idx]).view(train_idx.shape[0], -1)
    )
    loss = 1e-5 * loss1 + 1 * loss2 + 1e-3 * (loss3 + loss4)
    loss.backward()
    ED_Loss_train.append(loss1.item())
    NRR_Loss_train.append(loss2.item())
    Player_Loss_train.append((loss3.item() + loss4.item()) / 2)
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}")
        print(
            "Train Losses Decoder: %0.3f NRR: %0.3f Player Performance %0.3f"
            % (loss1.item(), loss2.item(), (loss3.item() + loss4.item()) / 2)
        )
        model.eval()
        inputs1 = torch.FloatTensor(nnPStats1[:, :, :12][test_idx])
        inputs2 = torch.FloatTensor(nnPStats2[:, :, :12][test_idx])
        outputs = torch.FloatTensor(_NRR[test_idx].reshape(-1, 1))
        decoded, out, scores1, scores2 = model(inputs1, inputs2)
        inp = (
            (inputs1).view(test_idx.shape[0], -1),
            (inputs2).view(test_idx.shape[0], -1),
        )
        loss1 = criterion(decoded, torch.cat(inp, axis=1))
        loss2 = criterion(out, outputs)
        loss3 = criterion(
            scores1, torch.FloatTensor(nnPScores1[test_idx]).view(test_idx.shape[0], -1)
        )
        loss4 = criterion(
            scores2, torch.FloatTensor(nnPScores2[test_idx]).view(test_idx.shape[0], -1)
        )
        ED_Loss_test.append(loss1.item())
        print(
            "Validation Losses Decoder: %0.3f NRR: %0.3f Player Performance: %0.3f"
            % (loss1.item(), loss2.item(), (loss3.item() + loss4.item()) / 2)
        )
        NRR_Loss_test.append(loss2.item())
        out, outputs = out.detach().numpy(), outputs.detach().numpy()
        Player_Loss_test.append((loss3.item() + loss4.item()) / 2)
        acc = 100 * np.sum((out * outputs) > 0) / out.shape[0]
        print("Val Accuracy: %0.3f" % acc)

sns.lineplot(x=np.arange(1, 10001), y=ED_Loss_train)
sns.lineplot(x=np.arange(1, 10001, 50), y=ED_Loss_test)

sns.lineplot(x=np.arange(1, 10001), y=NRR_Loss_train)
sns.lineplot(x=np.arange(1, 10001, 50), y=NRR_Loss_test)

sns.lineplot(x=np.arange(1, 10001), y=Player_Loss_train)
sns.lineplot(x=np.arange(1, 10001, 50), y=Player_Loss_test)
