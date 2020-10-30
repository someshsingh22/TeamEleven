import pandas as pd
import re, time, os, requests
from bs4 import BeautifulSoup as bs
from itertools import combinations
from datetime import date, datetime

_DATAROOT_ = "./Data"


def parseRow(row, teams, venue):
    entry = dict()
    stamp = row.find_all("td")[1].text.strip()
    stamp = datetime.strptime(stamp, "%d/%m/%Y").date()
    mc, gc = row.find_all("a")
    mc, gc = mc.attrs["href"], gc.attrs["href"]
    entry["Date"] = stamp
    entry["MatchCode"] = re.sub("[^0-9]+", "", mc)
    entry["GroundCode"] = re.sub("[^0-9]+", "", gc)
    entry["Venue"] = venue
    return entry


def parsePage(teams, venue):
    # TEST
    link = f"http://www.howstat.com/cricket/Statistics/Matches/MatchListCountry_ODI.asp?A={teams[0]}&B={teams[1]}&C={venue}#odis"
    req = requests.get(link)
    page = bs(req.content, features="html.parser")
    try:
        table = page.find("table", {"class": "TableLined"}).find_all("tr")[1:]
        return [parseRow(row, teams, venue) for row in table]
    except:
        print(f"No Results for Team {teams[0]} vs. {teams[1]} at {venue}")
        return None


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

data = pd.DataFrame(columns=["Date", "Venue", "MatchCode", "GroundCode"])
for teams in combinations(Teams, 2):
    for venue in ["Home", "Away", "Neutral"]:
        page = parsePage(teams, venue)
        if page is not None:
            data = data.append(page)
    data.to_csv(os.path.join(_DATAROOT_, "matches.csv"), index=False)
data.reset_index(drop=True, inplace=True)
