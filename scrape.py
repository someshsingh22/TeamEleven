import pandas as pd
from itertools import combinations
import os, json
from utils import parsePage, dataPipe, getCard

if __name__ == "__main__":

    _DATAROOT_ = "./Data"
    MATCHPATH = "matches.csv"
    SCOREPATH = "scorecard.json"

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
    scorecards = dict()

    for matchcode in matchcodes:
        try:
            card = getCard(matchcode)
            scorecards[matchcode] = dataPipe(card)
        except:
            print(f"MatchCode {matchcode} removed due to rain")
            scorecards[matchcode] = None

    with open(os.path.join(_DATAROOT_, SCOREPATH), "w") as fp:
        json.dump(scorecards, fp)
