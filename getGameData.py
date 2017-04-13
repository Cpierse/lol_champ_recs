# -*- coding: utf-8 -*-
"""
Once we have collected and sorted the summoner ids, we can use the api to
download data from their ranked games. This is the most time-intensive part 
of the code due to the API restrictions. As such, we will focus on 
Gold-ranked players in the 2016 ranked season.
"""

#%% Import and Initialize API, key variables, and load Tier data
import time
import io, json


import getTierIds
api, tiers, DeltaT = getTierIds.initialize(True)

# Static variables:
roles = ['TOP','JUNGLE','MID','ADC','SUPPORT']


#%% Key constants:
LOAD_PREV_DATA = True
# Which summoner data should we collect?
TIER = 'GOLD'
SEASON = 'SEASON2016'
QUEUE = 'TEAM_BUILDER_DRAFT_RANKED_5x5'
Summ_data_file_name = 'SummData.json'

# Changes for 2017:
LOAD_PREV_DATA = True
SEASON = 'PRESEASON2017'
QUEUE = 'TEAM_BUILDER_RANKED_SOLO, RANKED_FLEX_SR' 
Summ_data_file_name = 'SummData2017.json'

api_broken = True
queues = ['TEAM_BUILDER_RANKED_SOLO', 'RANKED_FLEX_SR'] 
year = 2017

#%% Collect information on each summoner one-by-one:
# Initialize the main dictionary. This will be of the format:
# summ_data = {TIER:{ID:{Role:[Champs]}}}
if LOAD_PREV_DATA:
    with open(Summ_data_file_name, 'r') as fp:
        summ_data = json.load(fp)
else:
    summ_data = {}
    summ_data[TIER]={}

# Go one-by-one through the summoners collecting their data.
count = 0
same_id_fails = 0
print('Currently at ' + str(len(summ_data[TIER])) + ' summoners of ' + str(len(tiers[TIER])) + '.')
print('Running...')
index = len(summ_data[TIER])
for failure_runs in range(0,10000):
    try:
        for sid in tiers[TIER][index:len(tiers[TIER])]:
            # Check if we have already collected this data
            if str(sid) in summ_data[TIER].keys():
                index += 1 
                continue
            # Pause until the api can make a request.
            while not api.can_make_request():
                time.sleep(DeltaT)
            # Getting the match history for the one summoner:
            if api_broken:
                # The api's optional arguments do not seem to work with the 2017 changes
                ml = api.get_match_list(summoner_id = sid)
                ml['matches'] = [x for x in ml['matches'] if (x['season']==SEASON) and (str(x['queue']) in queues)]
            else:
                ml = api.get_match_list(summoner_id = sid,season = SEASON,ranked_queues=QUEUE)
            # Initialize this summoner's dict {Role:[champs]}.
            role_champ_list = {}
            # Fill in the appropriate role with champion information.            
            for match in ml['matches']:
                if 'lane' in match:
                    role = str(match['lane'])
                    if role.find('BOTTOM') > -1:
                        if str(match['role']).find('CARRY')>-1:
                            role = 'ADC'
                        elif str(match['role']).find('SUPPORT')>-1:
                            role = 'SUPPORT'
                        else:
                            continue
                    if not role in role_champ_list:
                        role_champ_list[role] = []
                    role_champ_list[role].append(match['champion'])
            # Save the current result to the main dataframe
            summ_data[TIER][str(sid)] = role_champ_list
            # Add to counter:
            count += 1
            print('Success: '  + str(tiers[TIER][index]) + ' - ' + str(len(ml['matches'])) + ' matches')
            if count%100 == 0:
                print('Currently at ' + str(len(summ_data[TIER])) + ' summoners of ' + str(len(tiers[TIER])) + '.')
                with io.open(Summ_data_file_name, 'w', encoding='utf-8') as fp: 
                    fp.write(unicode(json.dumps(summ_data, ensure_ascii=False)))
            # The API requests are sometimes more than one. So sleep for 2x.
            time.sleep(1.10*DeltaT) 
            same_id_fails = 0
            index+=1
    except: 
        # Failure is typcially due to high server load. 
        same_id_fails += 1
        if same_id_fails==5:
            # Sometimes refreshing the api helps.
            api, tiers, DeltaT = getTierIds.initialize(True)
            time.sleep(2*DeltaT)
        elif same_id_fails>6:
            # Sometimes, particular ids will just not work.
            same_id_fails = 0
            print('Skipping ' + str(tiers[TIER][index]))
            index += 1
        time.sleep(2*DeltaT)
            


