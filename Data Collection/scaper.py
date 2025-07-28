import requests
from bs4 import BeautifulSoup
import re

# URL of the schedule page
schedule_url = 'https://www.baseball-reference.com/leagues/MLB-schedule.shtml'

# Request the page
response = requests.get(schedule_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Regular expression to match box score URL pattern
pattern = re.compile(r'^/boxes/[A-Z]{3}/[A-Z]{3}\d{9}\.shtml$')

# Find all links to box scores that match the pattern
box_score_links = []
for link in soup.find_all('a', href=True):
    href = link['href']
    if pattern.match(href):
        box_score_links.append('https://www.baseball-reference.com' + href)

# Scrape data from each box score
def scrape_team_totals(box_score_url):
    response = requests.get(box_score_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all team batting tables within the nested div structure
    team_totals = {}
    for div in soup.find_all('div', id=re.compile(r'^all_[A-Za-z]+batting$')):
        nested_div = div.find('div', class_='table_wrapper')
        if nested_div:
            team_name = nested_div['id'].replace('all_', '').replace('batting', '')
            team_table = nested_div.find('table', class_='stats_table')
            if team_table:
                # Extract the totals row
                totals_row = team_table.find('tfoot').find('tr')
                totals = [td.text for td in totals_row.find_all('td')]
                team_totals[team_name] = totals

    return team_totals

# Example: Scrape data for the first box score link
first_box_score_link = box_score_links[0]
team_totals = scrape_team_totals(first_box_score_link)
print(team_totals)

