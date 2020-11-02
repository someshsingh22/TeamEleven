from bs4 import BeautifulSoup as bs
import re, requests


def getCard(MatchCode):
    link = f"http://www.howstat.com/cricket/Statistics/Matches/MatchScorecard_ODI.asp?MatchCode={MatchCode}"
    return requests.get(link)


def ScoreCardScrape(card):
    soup = bs(card.content)
    page = soup.table.find_all(
        "table", {"cellpadding": 1, "cellspacing": 0, "border": 0}
    )
    T1, T2 = soup.find_all(
        "td", {"class": "TextBlackBold8", "valign": "top", "colspan": 2}
    )
    T1, T2 = T1.text.strip(), T2.text.strip()
    ORDER = (T1, T2[: T2.find("\xa0")])
    info, table = page[1], page[2]
    TOSS = info.find(
        "td", text=re.compile("Toss")
    ).next_sibling.next_sibling.text.strip()
    BAT1 = table.find_all("tr")[1:12]
    BOWL1, BOWL2 = table.find_all("table")[1::2]
    BOWL1, BOWL2 = BOWL1.find_all("tr")[1:], BOWL2.find_all("tr")[1:]
    BAT2IDX = 0
    for line in table.find_all("tr"):
        if line.find(text=re.compile("target")):
            BAT2IDX += 1
    BAT2 = table.find_all("tr")[BAT2IDX : 12 + BAT2IDX]
    return (ORDER, TOSS, (BAT1, BOWL1), (BAT2, BOWL2))


def batScore(batsman):
    cols = batsman.find_all("td")
    PlayerCode = re.sub("[^0-9]+", "", cols[0].a.attrs["href"])
    Dismissal = cols[1].text.strip()
    Runs, Balls, Fours, Sixes, SR, Perc = -1, -1, -1, -1, -1, -1
    if not Dismissal == "":
        Runs, Balls, Fours, Sixes, SR = [float(col.text.strip()) for col in cols[2:7]]
        try:
            Perc = float(cols[-1].span.next_sibling.strip()[:-1]) / 100
        except:
            Perc = 0
    return [PlayerCode, Dismissal, Runs, Balls, Fours, Sixes, SR, Perc]


def bowlScore(bowler):
    cols = bowler.find_all("td")
    PlayerCode = re.sub("[^0-9]+", "", cols[0].a.attrs["href"])
    Overs, Maidens, Runs, Wickets, ER = [float(col.text.strip()) for col in cols[1:6]]
    try:
        Perc = float(cols[-1].span.next_sibling.strip()[:-1]) / 100
    except:
        Perc = 0
    return [PlayerCode, Overs, Maidens, Runs, Wickets, ER, Perc]


def dataPipe(card):
    ORDER, TOSS, INNING1, INNING2 = ScoreCardScrape(card)
    BATTING1, BOWLING1 = INNING1
    BATTING2, BOWLING2 = INNING2
    BATTING1, BATTING2 = (
        [batScore(batsman) for batsman in BATTING1[:-1]],
        [batScore(batsman) for batsman in BATTING2[1:-1]],
    )
    BOWLING1, BOWLING2 = (
        [bowlScore(bowler) for bowler in BOWLING1],
        [bowlScore(bowler) for bowler in BOWLING2],
    )
    return {
        "ORDER": ORDER,
        "TOSS": TOSS,
        "BATTING1": BATTING1,
        "BOWLING1": BOWLING1,
        "BATTING2": BATTING2,
        "BOWLING2": BOWLING2,
    }
