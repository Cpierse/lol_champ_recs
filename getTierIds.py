# -*- coding: utf-8 -*-
"""
Here, we find and sort summoners based on their current ranking.
Starting from a seed id, we step through multiple ids and their
respective leagues. We check each league's tier, then save the 
summoners to the appropriate tier. 
"""
#%% Imports
import sys, os
import json
import time

## Finding and importing riotwatcher in parent directory.
rwpath = '//'.join(str.split(os.getcwd(),'\\')\
    [0:len(str.split(os.getcwd(),'\\'))-1]+['Riot-Watcher','riotwatcher'])
sys.path.append(rwpath)
import riotwatcher

#%% Variables:
Summ_ids_file_name = 'SummIds.json'

# Changes for 2017:
Summ_ids_file_name = 'SummIds2017.json'


#%% Initialize API and key variables:
def initialize(load=False):
    # Load the api using my api key
    api_key = open("api_key.txt").read()
    api = riotwatcher.RiotWatcher(api_key)

    ## Time Limits: 500 per 10 minutes
    DeltaT =  10.0*60.0/500.0
        
    
    if load:
        ## Load Tier info already acquired
        with open(Summ_ids_file_name, 'r') as fp:
            tiers = json.load(fp)
    else:
        ## Initialize main dict:
        all_tiers = ['BRONZE','SILVER','GOLD','PLATINUM','DIAMOND','MASTER','CHALLENGER']
        tiers = {}
        for tier in all_tiers:
            tiers[tier] = []
    
    return api, tiers, DeltaT

#%% Collect summoner ids grouped by tier:
def id_crawl(api, tiers, current_id, DeltaT):
    count = 0
    consecututive_fails = 0
    max_fails = 10000
    while consecututive_fails<max_fails:
        count += 1    
        try:
            # Get the league info from 10 summoners at a time.
            info = api.get_league(summoner_ids=range(current_id,current_id+10))
            consecututive_fails = 0
            for sid in info: # This is the result from each individual ID
                for queue in info[str(sid)]: # This isolates the league queue types
                    if queue['queue'].find('SOLO')>=0:  # Only Solo Q
                        # Which tier are they? Bronze? Master?
                        tier = queue['tier']
                        # Only add unique summoners. Do it one-by-one to make sure.
                        for peeps in info[str(sid)][0]['entries']:
                            if not int(peeps['playerOrTeamId']) in tiers[tier]:            
                                tiers[str(tier)].append(int(peeps['playerOrTeamId']))                    
        except Exception: # For when the api call fails
            consecututive_fails += 1        
            pass
        # print the results 
        if count % 10==0:
            print([x + ': ' + str(len(y)) for x,y in tiers.items()])
            if count % 100 == 0:
                # Save the results in a json file:
                with open(Summ_ids_file_name, 'w') as fp:
                    json.dump(tiers, fp)
        current_id += 10
        time.sleep(DeltaT)  # To stay within the Riot API Limit


#%% Main function:
def main():
    api, tiers, DeltaT = initialize()
    ## Choose an initial ID:
    my_id = 49921382
    id_0 = my_id
    current_id = int(float(id_0)/2)
    ## Crawl through ids seeking tiered leagues:
    id_crawl(api, tiers, current_id, DeltaT)
    
    
    
#%% Running the code:
if __name__ == "__main__":
   main()

