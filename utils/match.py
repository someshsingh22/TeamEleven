import re, requests
from bs4 import BeautifulSoup as bs
from datetime import date, datetime


def parseRow(row, teams, venue):
    entry = dict()
    stamp = row.find_all("td")[1].text.strip()
    stamp = datetime.strptime(stamp, "%d/%m/%Y").date()
    mc, gc = row.find_all("a")
    mc, gc = mc.attrs["href"], gc.attrs["href"]
    entry["Date"] = stamp
    entry["Team_1"], entry["Team_2"] = teams
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
