from model import InputForm
from flask import Flask, render_template, request
from compute import compute
import sys

import json
with open('id2champ.json', 'r') as fp:
    id2champ = json.load(fp)

app = Flask(__name__)

try:
    template_name = sys.argv[1]
    print(template_name)
except IndexError:
    template_name = 'view_bootstrap_cent_score'
    print(template_name)

if 'flask' in template_name and 'bootstrap' in template_name:
    from flask_bootstrap import Bootstrap
    Bootstrap(app)

@app.route('/champ_recs', methods=['GET', 'POST'])
def index():
    form = InputForm(request.form )
    if request.method == 'POST':
        # Print the results to terminal
        print('Results: ' + request.form['role_req'] + ' + ' + request.form['champ_req'])
        result = compute(request.form['role_req'],request.form['champ_req'])
        requested_champ = id2champ[request.form['champ_req']]
        #print(result)
        #print(id2champ[request.form['champ_req']])
        #print(request.form)
    else:
        result = None
        requested_champ = None
    #print form, dir(form)
    #print form.keys()
    #    for f in form:
    #        print('--------------------\n')
    #        print f.id
    #        print f.name
    #        print f.label
    #        #print f.choices
    #        print('--------------------\n')

    return render_template(template_name + '.html',
                           form=form, result=result,requested_champ=requested_champ)

if __name__ == '__main__':
    app.run(debug=True)