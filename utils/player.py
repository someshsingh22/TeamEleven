from bs4 import BeautifulSoup as bs
import re, requests


def parseInfo(code, stats=False):
    code = str(code).zfill(4)
    link = f"http://www.howstat.com/cricket/Statistics/Players/PlayerOverview_ODI.asp?PlayerId={code}"
    page = requests.get(link).content
    page = bs(page).body
    stats = page.find("td", {"width": 303, "valign": "top"}).table
    stats = {
        field.span.text.strip(): field.next_sibling.next_sibling.text.strip()
        for field in stats.find_all("td")
        if field.span
    }
    info = dict()
    for row in page.find_all("table")[6].find_all("tr")[2:]:
        entry = row.find_all("td")[:2]
        entry = (entry[0].text.strip(), entry[1].text.strip())
        if ":" in entry[0]:
            info[entry[0][:-1]] = entry[1]
    if stats:
        return stats, info
    else:
        return info
