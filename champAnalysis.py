# -*- coding: utf-8 -*-
"""
Now that we have collected game data from our summoners, we can analyze 
the results. Here, we look at correlations in champion play rates across 
roles to make meaningful recommendations. 

Note: In it's current form, the code is not written for efficiency. 

@author: Xenogearcap
"""

#%% Imports, static role and champ data, and summoner data
import json, io
import numpy as np
import copy, operator
import os
import matplotlib.pyplot as plt
from scipy.stats import expon, ttest_ind, chi2_contingency




#%% Key constants:
MIN_FRAC_ROLE = 0.20
MIN_FRAC_CHAMP_ROLE = 0.10
MIN_N_GAMES_MAIN_ROLE = 20 
MIN_N_GAMES_OFF_ROLE = 10
MIN_N_GAMES_CHAMP = 10
N_RECS = 12

summ_data_file_name = 'SummData2017.json' #'SummData.json'
rec_file_name = 'Recs2017.json'

#%% Load relevant data:
with open('id2champ.json', 'r') as fp:
    id2champ = json.load(fp)
with open('champ2id.json', 'r') as fp:
    champ2id = json.load(fp)
champs = sorted(champ2id.keys())
roles = ['TOP','JUNGLE','MID','ADC','SUPPORT']

# Load summoner data:
with open(summ_data_file_name, 'r') as fp:
    summ_data = json.load(fp)

#%% Convert each summoner into a dict of champ counts in each role.
summ_role_counts = {}
for tier in summ_data:
    summ_role_counts[tier] = {}
    for sid in summ_data[tier]:
        summ_role_counts[tier][sid] = {}
        for role in summ_data[tier][sid]:
            counts = dict()
            for i in summ_data[tier][sid][role]:
                counts[i] = counts.get(i, 0) + 1
            summ_role_counts[tier][sid][role] = counts


#%% Calculate each champ's average play frequency in each role.
#   Also, calculate the frequencies for each summoner + role
role_champ_freq_sum = {}
role_counts = {}
summ_role_freq = {}
for tier in summ_role_counts:
    role_champ_freq_sum[tier] = {}
    summ_role_freq[tier] = {}
    for role in roles:
        # Initialize the matrix:
        role_champ_freq_sum[tier][role] = {}
        role_counts[role] = 0
    for sid in summ_role_counts[tier]:
        summ_role_freq[tier][sid] = {}
        for role in summ_role_counts[tier][sid]:
            summ_role_freq[tier][sid][role]  = {}
            N = sum(summ_role_counts[tier][sid][role].values())
            if N < MIN_N_GAMES_MAIN_ROLE:
                continue
            keys = summ_role_counts[tier][sid][role].keys()
            values = np.array(summ_role_counts[tier][sid][role].values(),np.float)/sum(summ_role_counts[tier][sid][role].values())
            freq = dict( zip(keys,values) )
            summ_role_freq[tier][sid][role] = freq
            for champ, frac in freq.items():
                role_champ_freq_sum[tier][role][champ] = role_champ_freq_sum[tier][role].get(champ,0)+frac
            role_counts[role] += 1
# Normalize the sum:
role_champ_freq = {}
for tier in summ_role_counts:
    role_champ_freq[tier] = {}
    for role in roles:
        keys = role_champ_freq_sum[tier][role].keys()
        values = np.array(role_champ_freq_sum[tier][role].values())/role_counts[role]
        freq = dict( zip(keys,values) )
        role_champ_freq[tier][role] = freq

#%% Take each summoner, get their play frequency and difference from the mean.
#   Use this to get sigma
sigma_role_champ = {}
for tier in summ_role_counts:
    sigma_role_champ[tier] = {}
    for role in roles:
        sigma_role_champ[tier][role]= {}
        for champ in id2champ.keys():
            champ_freq_values = []
            for sid in summ_role_freq[tier]:
                if role in summ_role_freq[tier][sid]:
                    if int(champ) in summ_role_freq[tier][sid][role]:
                        champ_freq_values.append(summ_role_freq[tier][sid][role][int(champ)])
            sigma_role_champ[tier][role][champ]= np.std(champ_freq_values)



#%% Now onto the Pearson Correlation... kinda.
# So here's the weird part. We can calculate the pearson coorelation now....
# or we can consider the problem a bit more. Really, a person who does NOT
# play champion A often (and a person who does NOT play champion B often?)
# should not contribute to a value that will be used down the line to
# recommend champions to a player. Let's try to cut out terms where 
# a player plays below the average freq on both champs A and B. 
# Additionally, we will add the constraint that the player must have played
# more than MIN_FRAC_ROLE fraction of their games in role 1, played
# MIN_N_GAMES_CHAMP on that champ, and played MIN_FRAC_CHAMP_ROLE of their
# games in a role on that champ. This means the result is assymetric where
# the first set of (role,champ) is really a starting point.

# Computationally more expensive:
get_raw = True



# Initialize the data:
empty_dict = {}
for tier in summ_role_counts.keys():
    empty_dict[tier] = {}
    for role1 in roles:
        empty_dict[tier][role1] = {}
        for champ1 in id2champ.keys():
            empty_dict[tier][role1][champ1] = {}
            for role2 in roles:
                empty_dict[tier][role1][champ1][role2] = {}
                empty_dict[tier][role1][champ1][role2]['TOTAL'] = 0
                if get_raw: empty_dict[tier][role1][champ1][role2]['DATA'] = {}
                for champ2 in id2champ.keys():
                    empty_dict[tier][role1][champ1][role2][champ2] = 0
                    if get_raw: empty_dict[tier][role1][champ1][role2]['DATA'][champ2]=[]
#pseudo_coor = copy.deepcopy(empty_dict)
counts = copy.deepcopy(empty_dict)
single_count_recs = copy.deepcopy(empty_dict)
sliding_count_recs = copy.deepcopy(empty_dict)
sliding_max = 9.0


# Fill in the results:
for tier in summ_role_counts:
    for sid in summ_role_freq[tier].keys():
        n_games = np.sum(np.sum( [summ_role_counts[tier][sid][x].values() for x in summ_role_counts[tier][sid].keys()]))
        for role1 in summ_role_counts[tier][sid]:
            n_games_role = sum(summ_role_counts[tier][sid][role1].values())
            role_freq = np.array(n_games_role,float)/n_games
            if role_freq<MIN_FRAC_ROLE or n_games_role<MIN_N_GAMES_MAIN_ROLE:
                continue
            for role2 in summ_role_counts[tier][sid]:
                n_games_role = sum(summ_role_counts[tier][sid][role2].values())
                role_freq = np.array(n_games_role,float)/n_games
                if role_freq<MIN_FRAC_ROLE or n_games_role<MIN_N_GAMES_OFF_ROLE:
                    continue
                for champ1 in summ_role_freq[tier][sid][role1]:
                    n_games_champ = summ_role_counts[tier][sid][role1][champ1]
                    champ_freq = float(n_games_champ)/sum(summ_role_counts[tier][sid][role1].values())
                    if n_games_champ < MIN_N_GAMES_CHAMP or champ_freq < MIN_FRAC_CHAMP_ROLE:
                        continue
                    # For the summetric matrix total
                    counted_this_role = False
                    for champ2 in summ_role_freq[tier][sid][role2]:
                        if champ2=='TOTAL': continue
                        diff1 = summ_role_freq[tier][sid][role1][champ1] - role_champ_freq[tier][role1][champ1]
                        diff2 = summ_role_freq[tier][sid][role2][champ2] - role_champ_freq[tier][role2][champ2]
                        std1 = sigma_role_champ[tier][role1][str(champ1)]
                        std2 = sigma_role_champ[tier][role2][str(champ2)]
                        if diff1>0:# and not diff2<0:
                            pearson_coor_coeff_component = diff1*diff2/(std1*std2)
                            #pseudo_coor[tier][role1][str(champ1)][role2][str(champ2)] += pearson_coor_coeff_component
                            counts[tier][role1][str(champ1)][role2][str(champ2)] += 1    
                            if diff2>0:
                                single_count_recs[tier][role1][str(champ1)][role2][str(champ2)] += 1
                                sliding_count_recs[tier][role1][str(champ1)][role2][str(champ2)] += min(pearson_coor_coeff_component/(std1*std2),sliding_max)/sliding_max
                                if get_raw: sliding_count_recs[tier][role1][str(champ1)][role2]['DATA'][str(champ2)].append(min(pearson_coor_coeff_component/(std1*std2),sliding_max)/sliding_max) 
                                if not counted_this_role:
                                    counted_this_role = True
                                    single_count_recs[tier][role1][str(champ1)][role2]['TOTAL'] += 1
                                    sliding_count_recs[tier][role1][str(champ1)][role2]['TOTAL'] += 1

## Normalize the results and divide by the standard deviations:
#for tier in summ_role_counts:
#    for role1 in roles:
#        for champ1 in id2champ:
#            for role2 in roles:
#                for champ2 in id2champ:
#                    if counts[tier][role1][str(champ1)][role2][str(champ2)]>0:
#                        pseudo_coor[tier][role1][str(champ1)][role2][str(champ2)] /= counts[tier][role1][str(champ1)][role2][str(champ2)]

#%% One set of final results - top 3 lists using symmetric positive elements:
# Which dict to use - you can switch symm
source_dict = single_count_recs
#source_dict = sliding_count_recs

# Testing:
with open('champURLs.json', 'r') as fp:
    champ_urls = json.load(fp)


# Set up the reccomendation dict:
recs = copy.deepcopy(empty_dict)
for tier in recs:
    for role1 in roles:
        for champ1 in id2champ:
            for role2 in roles:
                recs[tier][role1][champ1][role2] = {}
                for pos in range(1,N_RECS+1):
                    recs[tier][role1][champ1][role2][pos]={}
# Sort and save the top N_RECS results:
for tier in recs:
    for role1 in roles:
        for champ1 in id2champ:
            for role2 in roles:
                curr_dict = source_dict[tier][role1][str(champ1)][role2]
                #curr_dict = pseudo_coor[tier][role1][str(champ1)][role2]
                champ_recs = sorted(curr_dict.items(), key=operator.itemgetter(1), reverse=True)
                pos = 1
                max_possible_score = curr_dict['TOTAL']
                recs[tier][role1][str(champ1)][role2]['N'] = max_possible_score
                for (champ2,score) in champ_recs:
                    if champ2=='TOTAL' or champ2=='DATA': continue
                    if not int(champ2)==int(champ1) or not role1==role2:
                        recs[tier][role1][str(champ1)][role2][pos]['id']=int(champ2)
                        recs[tier][role1][str(champ1)][role2][pos]['score']=int(round(score))
                        if max_possible_score>0: 
                            recs[tier][role1][str(champ1)][role2][pos]['frac_score']=int(100*float(score)/max_possible_score)
                        else:
                            recs[tier][role1][str(champ1)][role2][pos]['frac_score']=0
                        recs[tier][role1][str(champ1)][role2][pos]['champ']=id2champ[str(champ2)]
                        img_name = str(id2champ[str(champ2)]).replace('\'','')
                        img_name = img_name.replace(' ','')
                        img_name = img_name.replace('.','')
                        img_name = img_name.replace('Wukong','MonkeyKing')
                        recs[tier][role1][str(champ1)][role2][pos]['img_loc'] = os.path.join('static','images',img_name  + '.png')
                        recs[tier][role1][str(champ1)][role2][pos]['img_loc_url'] = champ_urls[str(champ2)]
                        pos+=1
                    if pos>N_RECS:
                        break


# Save the final recs:
with io.open('All'+rec_file_name, 'w', encoding='utf-8') as fp: 
    fp.write(unicode(json.dumps(recs, ensure_ascii=False)))


#%% Creating a recommendation rating based on t-test:
def get_p_vals(role1,champ1,single_counts=True,span=3):
    # Use a chi-squared test to calculate p-values to compare the recommendation 
    # distributions for the top 3 champs vs the next few recommendations.
    champ1=str(champ2id.get(champ1,champ1))
    p_vals = {}
    for role2 in recs[tier][role1][champ1]:
        p_vals[role2] = {}
        if role2=='TOTAL' or role2=='DATA': 
            continue
        for idx in range(1,4):
            values = []
            for pos_to_compare in range(idx+1,idx+1+span):
                # Get ids from recs:
                champ2_1 = str(champ2id[recs[tier][role1][champ1][role2][idx]['champ']])
                champ2_2 = str(champ2id[recs[tier][role1][champ1][role2][pos_to_compare]['champ']])
                # Get data:
                N = recs[tier][role1][champ1][role2]['N']
                if N > 10:
                    data = sliding_count_recs[tier][role1][champ1][role2]
                    champ2_1_data = np.array(data['DATA'][champ2_1] + [0]*(N-len(data['DATA'][champ2_1])))
                    champ2_2_data = np.array(data['DATA'][champ2_2] + [0]*(N-len(data['DATA'][champ2_2])))
                    if single_counts:
                        champ2_1_data[champ2_1_data>0]=1
                        champ2_2_data[champ2_2_data>0]=1
                    contingency_mat = np.array([[sum(champ2_1_data), N-sum(champ2_1_data)],[sum(champ2_2_data),N-sum(champ2_2_data)]])
                    values.append(chi2_contingency(contingency_mat)[1])
                else:
                    values.append(1)
            p_vals[role2][idx] = values
    return p_vals

def rate_recs(role1,champ1,single_counts = source_dict==single_count_recs, count=3):
    # Craft a function to rate the recommendations:
    champ1=str(champ2id.get(champ1,champ1))
    p_vals = get_p_vals(role1,champ1,single_counts = single_counts, span=6)
    # Minimum p_value for maintaining score:
    p_crit = 0.01
    N_crit = 0
    p_diff = 0.25
    score_dict = {}
    for role2 in p_vals:
        N = recs[tier][role1][champ1][role2]['N']
        if N < 10:
            scores = [0, 0, 0]
        else:
            prev_p_val = 0.0
            scores = []
            for idx in range(1,count+1):
                p_val = p_vals[role2][idx]
                if idx == 1:
                    score = 3.0
                    frac_rec = float(recs[tier][role1][champ1][role2][idx]['score'])/N
                    # Penalize the score if less than 20% of players recommend this champ:
                    while frac_rec<0.20:
                        score-=0.5
                        frac_rec+=0.05
                    # Penalize the score if only a few people play the champion
                    if N<N_crit:
                        score-=1
                else:
                    # If the p_val with previous champ is somewhat large, give
                    # the same score. Otherwise, penalize:
                    if prev_p_val < p_diff:
                        score -= 0.5
                # Penalize if p_val for next 3 champs not below the critical value
                if not any([p<p_crit for p in p_val[0:3]]):
                    score -= 0.5
                    # Even more if next 3 champs not below the critical value
                    if not any([p<p_crit for p in p_val[3:6]]):
                        score -= 1
                        if not any([p<p_crit for p in p_val[6:]]):
                            score -= 1
                # Record values:
                prev_p_val = p_val[0]
                if score<0: score = 0
                scores.append(score)
        score_dict[role2] = scores
    return score_dict


# Add this info to the current recommendations:
# Also create a top 3 truncated version for publishing online:
pub_recs = copy.deepcopy(recs)
for tier in recs:
    for role1 in roles:
        for champ1 in id2champ:
            ratings = rate_recs(role1,champ1)
            for role2 in ratings:
                for idx,rating in enumerate(ratings[role2]):
                    recs[tier][role1][str(champ1)][role2][idx+1]['rating'] = rating
                    pub_recs[tier][role1][str(champ1)][role2][idx+1]['rating'] = rating
                    # Remove unnecessary data:
                    pub_recs[tier][role1][str(champ1)][role2][idx+1].pop('frac_score')
                    pub_recs[tier][role1][str(champ1)][role2][idx+1].pop('img_loc')
                for idx in range(4,N_RECS+1):
                    pub_recs[tier][role1][str(champ1)][role2].pop(idx)


# Save the final recs:
with io.open('All'+rec_file_name, 'w', encoding='utf-8') as fp: 
    fp.write(unicode(json.dumps(recs, ensure_ascii=False)))

with io.open(rec_file_name, 'w', encoding='utf-8') as fp: 
    fp.write(unicode(json.dumps(pub_recs, ensure_ascii=False)))

#%% Plotting the play frequency for different champions:
# Function to plot the play rate distributions:
def plot_play_freq(roles,champs,tier='GOLD',plot_zeros = False, save_fig=False, multi_plot = False):  
    """
    Plots the play frequency of given champs in given roles. Both roles and
    champs can be lists or a single string. If champs is an integer, plots
    champs with the highest average (non-zero) play frequencies in those roles.
    """
    # If champs is an integer, grab top champions
    top_N_flag = False
    if type(champs)==int: 
        top_N_flag = True
        N = champs
        champs = [0]*N
    # Convert to lists if just one champion
    if type(roles)==str: roles = [roles]
    if type(champs)==str: champs = [champs]
    # Iterate through options:
    if multi_plot: 
        colors = ['r','g','b','c','m']
        f, axarr = plt.subplots(len(roles),len(champs))
    role_pos = 0 # Keep track of current role position
    for role1 in roles:
        # If input champs is a number, grab top play freq in that role:
        if top_N_flag:
            top_N = sorted(role_champ_freq[tier][role1].iteritems(), key=operator.itemgetter(1), reverse=True)[0:N]
            champs = [int(x[0]) for x in top_N]
        # Iterate through champs,:
        champ_pos = 0 # Keep track of current champ position
        y_max = 0 # For keeping track of axis in multiplot
        for champ1 in champs:
            # Convert input to strings our dictionary can use:
            champ1_name = None
            if type(champ1) == str:
                if champ1 in champ2id:
                    champ1_name = champ1
                    champ1 = int(champ2id[champ1])
                else:
                    champ1 = int(champ1)
            if not champ1_name:
                champ1_name = id2champ[str(champ1)]
            # Get play frequency for every summoner:
            play_freq = []
            for sid in summ_role_freq[tier].keys():
                if role1 not in summ_role_counts[tier][sid]:
                    continue
                n_games_role = sum(summ_role_counts[tier][sid][role1].values())
                n_games = np.sum(np.sum( [summ_role_counts[tier][sid][x].values() for x in summ_role_counts[tier][sid].keys()]))
                role_freq = np.array(n_games_role,float)/n_games
                if role_freq<MIN_FRAC_ROLE or n_games_role<MIN_N_GAMES_MAIN_ROLE:
                    continue
                if champ1 in summ_role_freq[tier][sid][role1]:
                    play_freq.append(summ_role_freq[tier][sid][role1][champ1])
                else:
                    play_freq.append(0.0)
            # Plot the play frequencies in a histogram:
            play_freq = np.array(play_freq)
            if not multi_plot:
                print(champ1_name + ': ' +  str(np.mean(play_freq)))
                if plot_zeros:
                    fig = plt.hist(play_freq,bins=20)
                else:
                    fig = plt.hist(play_freq[play_freq>0],bins=20)
                plt.title(role1.capitalize() + ' ' + champ1_name)
                plt.xlabel('In-role play frequency')
                plt.ylabel('Count')
                if save_fig:
                    plt.savefig('play_freq_plots\\'+ role1.capitalize() + '_' + str(pos+1) + '_' + champ1_name + '.png',dpi=300)
                plt.show()
            else:
                print(champ1_name + ': ' +  str(np.mean(play_freq)))
                # EXPERIMENTAL:
                exp_fit = expon.fit(play_freq)
                print(exp_fit)
                exp_fit_nz = expon.fit(play_freq[play_freq>0])
                print(exp_fit_nz)
                # END EXPERIMENTAL
                fig = axarr[role_pos,champ_pos]
                if plot_zeros:
                    fig.hist(play_freq,bins=20,color=colors[role_pos])
                else:
                    fig.hist(play_freq[play_freq>0],bins=20,color=colors[role_pos])
                y_max = max([y_max,fig.get_ylim()[1]]) 
            champ_pos+=1
        # Make sure axes are all the same:
        if multi_plot:
            for champ_pos in range(0,len(champs)):
                fig = axarr[role_pos,champ_pos]
                fig.set_ylim([0,y_max])
                #fig.text(0.95,0.80*y_max,role1.capitalize(),ha='right',va='top',fontsize=6)
                fig.text(0.95,0.95*y_max,id2champ[str(champs[champ_pos])],ha='right',va='top',fontsize=6)
                fig.xaxis.set_ticks([0, 0.5, 1])
                fig.yaxis.set_ticks([0,0.25*y_max,0.5*y_max,0.75*y_max, y_max])
                if champ_pos == 0:
                    #if role_pos == np.round(float(len(roles))/2):
                        #fig.set_ylabel('Count',fontsize=6)
                    #fig.yaxis.set_tick_params(labelsize=6,size=2)
                    fig.set_ylabel(role1.capitalize(),fontsize=6)
                else:
                    fig.yaxis.set_ticklabels([])
                if role_pos == len(roles)-1:
                    if champ_pos == np.round(float(len(champs))/2):
                        fig.set_xlabel('In-role play frequency',fontsize=6)
                    #fig.xaxis.set_tick_params(labelsize=6,size=2)
                else:
                    fig.xaxis.set_ticklabels([])
                fig.tick_params(axis='both', which='minor', labelsize=4,size=2)
                fig.tick_params(axis='both', which='major', labelsize=4,size=2)
        role_pos+=1
    if multi_plot and save_fig:
        plt.savefig('play_freq_plots\\multiplot.png',dpi=300)


# Let's look at the top 5 champs in each role:
roles = ['TOP','JUNGLE','MID','ADC','SUPPORT']
plot_play_freq(roles,5,save_fig=True, multi_plot=True)



#%% Random work in progress:

# Recs for specific champ
recs[tier]['TOP'][str(champ2id['Quinn'])]

recs[tier]['MID'][str(champ2id["Vel'Koz"])]

recs[tier]['TOP'][str(champ2id['Darius'])]

recs[tier]['MID'][str(champ2id['Lux'])]

recs[tier]['ADC'][str(champ2id['Caitlyn'])]


# If we make the (horrible) assumption that each recommendation comes from a 
# binomial distribution:
def quick_bin_confidence(N_players, N_recs, z=1.96):
    p = float(N_recs)/N_players
    delta = z*np.sqrt(1/float(N_recs)*p*(1-p))    
    N_min = N_recs - round(N_recs*delta)
    N_max = N_recs + round(N_recs*delta)
    return N_min, N_max    

#quick_bin_confidence(recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['N'],
#                     recs[tier]['TOP'][str(champ2id['Darius'])]['TOP'][1]['score'])

# How about a t-test based on this assumption?
def quick_t_test(N_players,champ1_data,champ2_data):
    champ1_data += [0]*(N_players-len(champ1_data))
    champ2_data += [0]*(N_players-len(champ2_data))
    return ttest_ind(champ1_data,champ2_data,equal_var=False)

#t_score = quick_t_test(recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['N'], 
#             sliding_count_recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['DATA'][str(champ2id['Garen'])],
#             sliding_count_recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['DATA'][str(champ2id['Nasus'])])
#
#
#t_score = quick_t_test(recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['N'], 
#             sliding_count_recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['DATA'][str(champ2id['Garen'])],
#             sliding_count_recs[tier]['TOP'][str(champ2id['Darius'])]['TOP']['DATA'][str(champ2id['Olaf'])])
#


# More detailed t-test function:
def t_test_ex(role1,champ1,single_counts=False):
    champ1=str(champ2id[champ1])
    for role2 in recs[tier][role1][champ1]:
        if role2=='TOTAL' or role2=='DATA': 
            continue
        for idx in range(1,4):
            values = []
            for pos_to_compare in range(idx+1,idx+4):
                # Get ids from recs:
                champ2_1 = champ2id[recs[tier][role1][champ1][role2][idx]['champ']]
                champ2_2 = champ2id[recs[tier][role1][champ1][role2][pos_to_compare]['champ']]
                # Get data:
                N = recs[tier][role1][champ1][role2]['N']
                data = sliding_count_recs[tier][role1][champ1][role2]
                champ2_1_data = np.array(data['DATA'][champ2_1] + [0]*(N-len(data['DATA'][champ2_1])))
                champ2_2_data = np.array(data['DATA'][champ2_2] + [0]*(N-len(data['DATA'][champ2_2])))
                if single_counts:
                    champ2_1_data[champ2_1_data>0]=1
                    champ2_2_data[champ2_2_data>0]=1
                values.append(str(ttest_ind(champ2_1_data,champ2_2_data,equal_var=False)[1]))
            print( role2 + ' ' + id2champ[champ2_1] + ' p-values: ' +  values[0] + ', ' + values[1] + ', ' + values[2])
        print('-----------------------------------------------------------------------------')

#t_test_ex('TOP','Darius')
#t_test_ex('TOP','Darius',single_counts=True)
#
#t_test_ex('MID','Lux')
#t_test_ex('MID','Lux',single_counts=True)
#
#t_test_ex('ADC','Caitlyn')
#t_test_ex('ADC','Caitlyn',single_counts=True)
#
#
#






