import json


def compute(role,champ):
    """Return relevant champ rec dictionary"""
    with open('Recs2016.json', 'r') as fp:
        recs = json.load(fp)
    return recs['GOLD'][role][champ]

