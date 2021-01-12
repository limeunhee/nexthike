from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np

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

app = Flask(__name__)

# home page
@app.route('/', methods=['GET','POST'])
def index():

    pop_output = top_10.loc[:,['name', 'difficulty','location','stars','review_count']]
    hybrid_output = get_recommendations_by_hybrid_sim(0,5)

    return render_template("index.html")




# Where you return list of trails that contains the string user searched for
@app.route('/trail_search_result', methods=['GET','POST'])
def trail_search_result():
    text = str(request.form['user_input'])
    output = df[(df['name'].str.contains(text, case=False)) |
                 (df['location'].str.contains(text, case=False))].loc[:,['name', 'location','distance']]
    return render_template("trail_form.html", output=output)


@app.route('/recommendations/<index_val>', methods=['GET', 'POST'])
def recommendations(index_val):
    order = ['name', 'location', 'stars','distance','elevation','duration','difficulty'] #,'short_description']
    
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
