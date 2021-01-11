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
df_Q = df[ df['review_count'] > m]
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



# Where you ask user to search for trail they liked
@app.route('/input_form')
def input_form():
    return '''
        <form action="/trail_search_result" target="_blank" method='POST'>
            <input type="text" name="user_input" />
            <input type="submit"/>
        </form>
        '''

# Where you return list of trails that contains the string user searched for
@app.route('/trail_search_result', methods=['GET','POST'])
def trail_search_result():
    text = str(request.form['user_input'])
    output = df[(df['name'].str.contains(text, case=False)) |
                 (df['location'].str.contains(text, case=False))].loc[:,['name', 'location','distance']].to_html()
    return f'{output}'


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    order = ['name', 'location', 'stars','distance','elevation','duration','difficulty'] #,'short_description']
    
    user_input = str(request.form['user_input_idx'])
    if len(user_input) == 0:
        return redirect('/')

    user_input = user_input.split(',')
    if len(user_input) == 1:
        user_input = int(user_input[0])
        user_trail = df[df.index==user_input]
        user_trail = user_trail[order]
        
        pop_output = top_10.loc[:,order]
        
        hybrid_output = get_recommendations_by_hybrid_sim(user_input,10)
        hybrid_output = hybrid_output[order]
    else:
        return "more than 1 argument"
    

        
    
    

    return f'''The trail you chose: {user_trail.to_html()} 
             Top 10 trails in CA: {pop_output.to_html()} 
             Trails similar to yours: {hybrid_output.to_html()}
             Hikers similar to you also liked: 
            '''

       
               # Trails liked by other hikers who also liked 'user_input' <br>  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
