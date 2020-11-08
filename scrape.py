import pandas as pd
from itertools import combinations
import os, json
from utils import parsePage, dataPipe, getCard, parseInfo

if __name__ == "__main__":

    _DATAROOT_ = "./Data"
    MATCHPATH = "matches.csv"
    SCOREPATH = "scorecard.json"
    BATPATH = "batsmen.json"

    Teams = [
        "AFG",
        "AUS",
        "BAN",
        "ENG",
        "IND",
        "IRE",
        "KEN",
        "NED",
        "NZL",
        "PAK",
        "SCO",
        "SAF",
        "SRL",
        "WIN",
        "ZIM",
    ]

    data = pd.DataFrame(
        columns=["Date", "Team_1", "Team_2", "Venue", "MatchCode", "GroundCode"]
    )
    for teams in combinations(Teams, 2):
        for venue in ["Home", "Away", "Neutral"]:
            page = parsePage(teams, venue)
            if page is not None:
                data = data.append(page)
        print(teams)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(os.path.join(_DATAROOT_, MATCHPATH), index=False)

    matchcodes = data["MatchCode"].tolist()
    scorecards = {matchcode: dataPipe(getCard(matchcode)) for matchcode in matchcodes}

    with open(os.path.join(_DATAROOT_, SCOREPATH), "w") as fp:
        json.dump(scorecards, fp)

    with open(os.path.join(_DATAROOT_, BATPATH), "r") as fp:
        players = json.load(fp).keys()

    info = {k: parseInfo(k) for k in players}
    info = pd.DataFrame.from_dict(info, orient="index")
    info.to_csv("attributes.csv")
