# -*- coding: utf-8 -*-
"""
We will use the API to create a dictionary that maps champion ids to champion
names. We will also dowload all of the champion images to a subfolder. 
"""
#%% Imports
import sys, os
import json
import requests
# Imports for images:
from PIL import Image
from StringIO import StringIO

## Finding and importing riotwatcher in parent directory.
rwpath = '//'.join(str.split(os.getcwd(),'\\')\
    [0:len(str.split(os.getcwd(),'\\'))-1]+['Riot-Watcher','riotwatcher'])
sys.path.append(rwpath)
import riotwatcher

#%% Creating dictionary of champion ids to names
api_key = open("api_key.txt").read()
api = riotwatcher.RiotWatcher(api_key)

# Getting the list of champion ids:
champ_list = api.get_all_champions()['champions']
champ_id_list = []
for champ in champ_list:
    champ_id_list.append(champ['id'])
print champ_id_list

# Creating a function to get the champion names given the id:
def get_champ_name(champ_id):
    url = 'https://global.api.pvp.net/api/lol/static-data/na/v1.2/champion/{CHAMP_ID}?api_key={APIKEY}'.format(
        CHAMP_ID = str(champ_id),
        APIKEY = 'b2678d97-bad1-4323-9f01-b5a641417273' )
    response = requests.get(url)
    return response.json()

# Using the list of champion IDs to find the champion names:
id2champ = {}
champ2id = {}

for champ_id in champ_id_list:
    champ_name = get_champ_name(champ_id)['name']
    id2champ[champ_id] = champ_name
    champ2id[champ_name] = champ_id

with open('champ2id.json', 'w') as fp:
    json.dump(champ2id, fp)

with open('id2champ.json', 'w') as fp:
    json.dump(id2champ, fp)

#%% Download all champion images:

# Find the image location:
def get_champs_image_locations(api_key=api_key):
    url = 'https://global.api.pvp.net/api/lol/static-data/na/v1.2/champion?champData=image&api_key={APIKEY}'.format(
        APIKEY = api_key )
    response = requests.get(url)
    return response.json()

# Download the image:
def save_champ_image(champ_name,champ_img_name):
    if champ_name=='Fiddlesticks':
        champ_img_name = 'FiddleSticks.png' 
    url = 'http://ddragon.leagueoflegends.com/cdn/6.24.1/img/champion/{CHAMP_IMG_NAME}'.format(
        CHAMP_IMG_NAME = str(champ_img_name).replace(" ", ""))
    response = requests.get(url)
    try:
        img = Image.open(StringIO(response.content))
        img.save('Images' + os.sep +  champ_name + '.png')
    except IOError:
        print('Could not find image for:' + champ_name)
    return url

# Download all images one-by-one:
champ_info = get_champs_image_locations()['data']
champ_urls = {}
for ind in range(0,len(champ_info)):
    champ = champ_info.keys()[ind]
    champ_id = champ_info[champ]['id']
    champ_img_name = champ_info[champ]['image']['full']
    champ_name = str(champ)
    champ_urls[champ_id] = save_champ_image(champ_name,champ_img_name)

# Save all urls to instead use ddragon to host images:
with open('champURLs.json', 'w') as fp:
    json.dump(champ_urls, fp)
