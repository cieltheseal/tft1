import requests
import json
import pandas as pd
from flatten_json import flatten
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

api_key = 'RGAPI-a26ccbcb-2244-467e-9487-05060c1b95b5'
#oregion = "asia"
#region = 'sea'
n = 1
server = "SG2"
servers = ["NA1", "KR", "EUW1"]
regions = {"Americas": ["NA1", "BR1"], "Asia": ["KR", "JP"], "Europe": ["EUW1"], "SEA": ["SG2"]}
username = "SirCiel"

def get_challengers(server, api_key) -> list:
    puuids = []
    get_challengers_url = ('https://' + server + '.api.riotgames.com/tft/league/v1/challenger?api_key=' + api_key)
    print(f"Extracting Challenger PUUIDs from {server}.")
    try:
        challengers_resp = requests.get(get_challengers_url, timeout=60)
        challengers_info = challengers_resp.json()
        puuids.extend(entry['puuid'] for entry in challengers_info['entries'])
    except:
        print('Request has timed out.')
    print("Challenger PUUIDs successfully extracted.")
    return puuids

def get_matches(puuids, region, n, api_key) -> list:
    print(f"Extracting {n} most recent matches for {len(puuids)} Challengers.")
    match_ids = set()
    for puuid in puuids:
        matches_url = ('https://' + region + '.api.riotgames.com/tft/match/v1/matches/by-puuid/'
                       + puuid + '/ids?start=0&count=' + str(n) + '&api_key=' + api_key)
        try:
            matches_resp = requests.get(matches_url, timeout = 60)
            match_ids.update(matches_resp.json())
        except:
            print('Request has timed out')
    print(f"{len(match_ids)} Match IDs successfully extracted.")
    return match_ids

def get_match_info(match_id, subserver, api_key) -> dict:
    match_url = ('https://' + subserver + '.api.riotgames.com/tft/match/v1/matches/' + match_id + '?api_key=' + api_key)
    try:
        match_resp = requests.get(match_url, timeout = 60)
        match_info = match_resp.json()
        return match_info["info"]
    except:
        print('Request has timed out')

def get_matches_info(matches, subserver, api_key) -> pd.DataFrame():
    match_data = pd.DataFrame()
    print("Obtaining match data.")
    for match in matches:
        match_info = get_match_info(match, subserver, api_key)
        flat_match_resp = flatten(match_info)
        keyList = list(flat_match_resp.keys())

        keyList1 = [key for key in keyList if key.startswith('participants_0')]
        keyList2 = [key for key in keyList if key.startswith('participants_1')]
        keyList3 = [key for key in keyList if key.startswith('participants_2')]
        keyList4 = [key for key in keyList if key.startswith('participants_3')]
        keyList5 = [key for key in keyList if key.startswith('participants_4')]
        keyList6 = [key for key in keyList if key.startswith('participants_5')]
        keyList7 = [key for key in keyList if key.startswith('participants_6')]
        keyList8 = [key for key in keyList if key.startswith('participants_7')]

        player1info = {k:v for k, v in flat_match_resp.items() if k in keyList1}
        player2info = {k:v for k, v in flat_match_resp.items() if k in keyList2}
        player3info = {k:v for k, v in flat_match_resp.items() if k in keyList3}
        player4info = {k:v for k, v in flat_match_resp.items() if k in keyList4}
        player5info = {k:v for k, v in flat_match_resp.items() if k in keyList5}
        player6info = {k:v for k, v in flat_match_resp.items() if k in keyList6}
        player7info = {k:v for k, v in flat_match_resp.items() if k in keyList7}
        player8info = {k:v for k, v in flat_match_resp.items() if k in keyList8}

        player1 = pd.DataFrame([player1info])
        player2 = pd.DataFrame([player2info])
        player3 = pd.DataFrame([player3info])
        player4 = pd.DataFrame([player4info])
        player5 = pd.DataFrame([player5info])
        player6 = pd.DataFrame([player6info])
        player7 = pd.DataFrame([player7info])
        player8 = pd.DataFrame([player8info])

        player1.columns = player1.columns.str.replace('participants_0_', '')
        player2.columns = player2.columns.str.replace('participants_1_', '')
        player3.columns = player3.columns.str.replace('participants_2_', '')
        player4.columns = player4.columns.str.replace('participants_3_', '')
        player5.columns = player5.columns.str.replace('participants_4_', '')
        player6.columns = player6.columns.str.replace('participants_5_', '')
        player7.columns = player7.columns.str.replace('participants_6_', '')
        player8.columns = player8.columns.str.replace('participants_7_', '')

        player1 = player1.convert_dtypes()
        player2 = player2.convert_dtypes()
        player3 = player3.convert_dtypes()
        player4 = player4.convert_dtypes()
        player5 = player5.convert_dtypes()
        player6 = player6.convert_dtypes()
        player7 = player7.convert_dtypes()
        player8 = player8.convert_dtypes()

        player1 = player1.select_dtypes(exclude=['object'])
        player2 = player2.select_dtypes(exclude=['object'])
        player3 = player3.select_dtypes(exclude=['object'])
        player4 = player4.select_dtypes(exclude=['object'])
        player5 = player5.select_dtypes(exclude=['object'])
        player6 = player6.select_dtypes(exclude=['object'])
        player7 = player7.select_dtypes(exclude=['object'])
        player8 = player8.select_dtypes(exclude=['object'])

        match_data = pd.concat([match_data, player1], ignore_index = True)
        match_data = pd.concat([match_data, player2], ignore_index = True)
        match_data = pd.concat([match_data, player3], ignore_index = True)
        match_data = pd.concat([match_data, player4], ignore_index = True)
        match_data = pd.concat([match_data, player5], ignore_index = True)
        match_data = pd.concat([match_data, player6], ignore_index = True)
        match_data = pd.concat([match_data, player7], ignore_index = True)
        match_data = pd.concat([match_data, player8], ignore_index = True)

    print(f"Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns generated.")

    return match_data

class DropEmptyColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Identify columns that do NOT end with '_name'
        self.columns_to_keep_ = [col for col in X.columns if not col.endswith('_name')]
        return self

    def transform(self, X):
        return X[self.columns_to_keep_]

def use_data_pipeline(match_data) -> 'DataFrame':

    # use pipeline for data analysis
    pipe_analysis = Pipeline([
       ("dropempty", DropEmptyColumns()),
    ])

    match_data = pipe_analysis.fit_transform(match_data)

    # write csv for data analysis
    match_data.to_csv('data/unprocessed.csv', index = False)
    print(f"Dataframe with {match_data.shape[0]} boards and {match_data.shape[1]} columns saved.")

    pipe_ml = Pipeline([
    ])

    #match_data = pipe_ml.fit_transform(match_data)

    # write csv for placement estimator
    match_data.to_csv('data/processed.csv', index = False)

    return match_data


if __name__ == '__main__':
    puuids = get_challengers(server, api_key)
    region = next((region for region, servers in regions.items() if server in servers), None)
    puuids = [puuids[0]]
    match_ids = get_matches(puuids, region, n, api_key)
    match_ids = [list(match_ids)[0]]
    db = get_matches_info(match_ids, region, api_key)
    use_data_pipeline(db)
    #with open("tft_match_summary.json", "w") as f:
    #    json.dump(match, f, indent=2)

    #profile = get_profile(server, username, region, api_key)
    #puuid = profile['puuid']
    #matches = get_matches(puuid, subserver, api_key)

"""
def get_profile(server, username, region, api_key) -> dict:
    profile_url = ('https://' + server + '.api.riotgames.com/riot/account/v1/accounts/by-riot-id/' + username
    + '/' + region+ '?api_key=' + api_key)
    try:
        profile_resp = requests.get(profile_url, timeout = 60)
        profile_info = profile_resp.json()
        print(profile_info)
        return profile_info
    except:
"""