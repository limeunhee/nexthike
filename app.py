from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np
from scipy import sparse
import pickle5 as pickle

## For CB model
def weighted_star_rating(x):
    v = x['review_count']
    R = x['stars']
    return (v/(v+m) * R) + (m/(m+v) * C)

df = pd.read_csv('data/CA_trails.csv')
m = df['review_count'].quantile(q=0.95)
C = df['stars'].mean()
df_Q = df[df['review_count'] > m].copy()
df_Q['WSR'] = df_Q.apply(weighted_star_rating, axis=1)
top_10 = df_Q.sort_values('WSR', ascending = False)[:10]   ## Return this for top 10 in CA 


def pop_chart_per_region(region):
    df_region =  df[df['location']==region]
    m = df['review_count'].quantile(q=0.70)
    C = df['stars'].mean()
    df_Q = df_region[ df_region['review_count'] > m]
    df_Q['WSR'] = df_Q.apply(weighted_star_rating, axis=1)
    top_5_in_region = df_Q.sort_values('WSR', ascending = False)[:5]
    return top_5_in_region


PT_feature_cosine_sim = np.loadtxt('data/feature_sim.txt', delimiter =',')
cosine_sim = np.loadtxt('data/text_sim.txt', delimiter =',')

def get_recommendations_by_text_sim(idx, top_n=5):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1],  reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    trail_indices = [i[0] for i in sim_scores]
    result = df.iloc[trail_indices]
    return result

def get_recommendations_by_feature_sim(idx, top_n=5):
    sim_scores = list(enumerate(PT_feature_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1],  reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    trail_indices = [i[0] for i in sim_scores]
    result = df.iloc[trail_indices]
    return result

def get_recommendations_by_hybrid_sim(idx, top_n=5):
    hybrid_cosine_sim = 0.5*PT_feature_cosine_sim + 0.5*cosine_sim
    sim_scores = list(enumerate(hybrid_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1],  reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    trail_indices = [i[0] for i in sim_scores]
    result = df.iloc[trail_indices]
    return result

order = ['name', 'location', 'stars','distance','elevation','duration','difficulty']

## CF model

trail_indices = pd.read_csv('data/knn_trail_indices.csv')
small_df = pd.read_csv('data/small_df.csv')

filename = 'data/knn_recommendations.pickle'
with open(filename, 'rb') as handle:
    CF_model = pickle.load(handle)
 

app = Flask(__name__)

# home page
@app.route('/', methods=['GET','POST'])
def index():

    return render_template("index.html")


# Where you return list of trails that contains the string user searched for
@app.route('/CF_input', methods=['GET','POST'])
def CF_input():
    cols = ['trail', 'location', 'distance', 'difficulty', 'trail_url']
    samples = trail_indices.sample(25)[cols]

    return render_template('CF_input_form.html', samples = samples)

# Where you return list of trails that contains the string user searched for
@app.route('/CF_rec', methods=['GET','POST'])
def CF_rec():
    new_user_ratings = request.form.getlist("trail_index")

    #if len(new_user_ratings) ==0:
        
    A = [10000 for i in new_user_ratings] ## new user id
    B= [int(i) for i in new_user_ratings ] ## trail id
    C = [5 for i in new_user_ratings] ## rating value

    d = {'user_id':A, 'trail_id':B, 'stars':C}
    df_new = pd.DataFrame(data=d)
    new_small_df = small_df.append(df_new)

    sparse_mat = sparse.csr_matrix((new_small_df.stars, (new_small_df.user_id, new_small_df.trail_id,)))
    A= (sparse_mat * sparse_mat.T)[-1,:].todense()
    A = np.asarray(A).reshape(-1)
    temp = sorted(A, reverse=True)

    result = np.where(A == temp[1])[0][0]

    old_user_rec = [i[0] for i in CF_model[result]]
    new_user_rec = [x for x in old_user_rec if x not in B]

    new_user_top_5 = new_user_rec[:5]
    pop_output = top_10.loc[:,order]

    return render_template("CF_rec.html" , output = trail_indices.iloc[new_user_top_5], new_user_ratings  = trail_indices.iloc[B], pop_output = pop_output)


# Where you return list of trails that contains the string user searched for
@app.route('/trail_search_result', methods=['GET','POST'])
def trail_search_result():
    text = str(request.form['user_input'])
    output = df[(df['name'].str.contains(text, case=False)) |
                 (df['location'].str.contains(text, case=False))].loc[:,['name', 'location','distance']]
    return render_template("trail_form.html", output=output)


@app.route('/recommendations/<index_val>', methods=['GET', 'POST'])
def recommendations(index_val):

    user_input = int(index_val)
    user_trail = df[df.index==user_input]
    user_trail = user_trail[order]
    
    pop_output = top_10.loc[:,order]
    
    hybrid_output = get_recommendations_by_hybrid_sim(user_input,10)
    hybrid_output = hybrid_output[order]

    return render_template('recommendations.html', user_trail = user_trail, pop_output=pop_output, hybrid_output = hybrid_output )

       
               # Trails liked by other hikers who also liked 'user_input' <br>  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
