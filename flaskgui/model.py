from wtforms import Form, SelectField
import json
import collections



#with open('id2champ.json', 'r') as fp:
#    id2champ = json.load(fp)
#roledict = {'TOP':'Top','JUNGLE':'Jungle','MID':'Mid','ADC':'Adc','SUPPORT':'Support'}
with open('champ2id.json', 'r') as fp:
    champ2id = json.load(fp)
champ2id = collections.OrderedDict(sorted(champ2id.items()))
roledict = [('Top','TOP'),('Jungle','JUNGLE'),('Mid','MID'),('Support','SUPPORT'),('Adc','ADC')]
roledict = collections.OrderedDict(roledict)

class InputForm(Form):
    role = SelectField(label='Role', 
            choices=roledict.items())
    champ = SelectField(label='Champion', 
            choices=champ2id.items())

    
    
