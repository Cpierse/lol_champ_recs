# lol_champ_recs
A champion recommendation system for the online game League of Legends. For details on the algorithm, please see this [post](http://cpierse.physics.ucsd.edu/champ_recs_post/).
## Components
The core of the code is broken down into four pieces: collectStaticData.py, getTierIds.py, getGameData.py, and champAnalysis.py. Each piece is detailed below.

### Prerequisites
This code uses the [Riot-Watcher](https://github.com/pseudonym117/Riot-Watcher) Python wrapper for Riot's API. You will also need an API key from Riot Games which can be acquired [here](https://developer.riotgames.com/).

### The core of the algorithm
- collectStaticData.py - downloads champion images and builds a dictionary that map champion ids to names.
- getTierIds.py - iterates through player ids and sorts them based on the ranked tier of the player. 
- getGameData.py - collects game data from the players that were identified in getTierIds. 
- champAnalysis.py - generates and rates recommendations from the game data. 

### Other notes
- The last section of champAnalysis.py is for the general analysis of  game data. Feel free to comment and ignore this section. 
- sqlSave.py converts the saved game data from json to a sql database. 