import gc

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

### -----------------------------------------------------------------
#  Improvements
#  Consider using movie's distance to the cluster composed by likes of many users.      
### ----------------------


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

gpt2_model.resize_token_embeddings(len(tokenizer))  

def clean_text(text):
    text = text.replace('\n', ' ')  
    text = text.replace('\t', ' ')  
    text = ' '.join(text.split())   
    return text

data = {
    'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'title': [
        'Inception', 'Interstellar', 'Catch Me If You Can', 'Sherlock Holmes',
        'Star Wars: The Force Awakens', 'The Lord of the Rings: The Fellowship of the Ring',
        'Elizabeth', 'Heat', 'The Matrix', 'The Godfather', 'Avatar', 
        'Pulp Fiction', 'Fight Club', 'Titanic', 'Jurassic Park'
    ],
    'synopsis': [
        "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
        "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
        "A seasoned FBI agent pursues Frank Abagnale Jr., who, before his 19th birthday, had successfully forged millions of dollars' worth of checks while posing as a Pan Am pilot, doctor, and legal prosecutor.",
        "Detective Sherlock Holmes and his stalwart partner Watson engage in a battle of wits and brawn with a nemesis whose plot is a threat to all of England.",
        "As a new threat to the galaxy rises, Rey, a desert scavenger, and Finn, an ex-stormtrooper, must join Han Solo and Chewbacca to search for the one hope of restoring peace.",
        "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle Earth from the Dark Lord Sauron.",
        "The early years of the reign of Elizabeth I of England and her difficult task of learning what is necessary to be a monarch.",
        "A group of professional bank robbers start to feel the heat from police when they unknowingly leave a clue at their latest heist.",
        "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
        "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.",
        "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "An insomniac office worker and a devil-may-care soapmaker form an underground fight club that evolves into something much, much more.",
        "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
        "During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok."
    ],
    'genres': [
        ['Sci-Fi', 'Thriller'], ['Adventure', 'Sci-Fi'], ['Biography', 'Crime'],
        ['Action', 'Mystery'], ['Adventure', 'Sci-Fi'], ['Adventure', 'Fantasy'],
        ['Biography', 'Drama'], ['Crime', 'Thriller'],
        ['Sci-Fi', 'Action'], ['Crime', 'Drama'], ['Action', 'Adventure'],
        ['Crime', 'Drama'], ['Drama', 'Action'], ['Drama', 'Romance'],
        ['Adventure', 'Sci-Fi']
    ],
    'actors': [
        ['Leonardo DiCaprio', 'Joseph Gordon-Levitt', 'Elliot Page'],
        ['Matthew McConaughey', 'Anne Hathaway', 'Jessica Chastain'],
        ['Leonardo DiCaprio', 'Tom Hanks', 'Christopher Walken'],
        ['Robert Downey Jr.', 'Jude Law', 'Rachel McAdams'],
        ['Daisy Ridley', 'John Boyega', 'Harrison Ford'],
        ['Elijah Wood', 'Ian McKellen', 'Viggo Mortensen'],
        ['Cate Blanchett', 'Geoffrey Rush', 'Joseph Fiennes'],
        ['Al Pacino', 'Robert De Niro', 'Val Kilmer'],
        ['Keanu Reeves', 'Laurence Fishburne', 'Carrie-Anne Moss'],
        ['Marlon Brando', 'Al Pacino', 'James Caan'],
        ['Sam Worthington', 'Zoe Saldana', 'Sigourney Weaver'],
        ['John Travolta', 'Samuel L. Jackson', 'Uma Thurman'],
        ['Brad Pitt', 'Edward Norton', 'Helena Bonham Carter'],
        ['Leonardo DiCaprio', 'Kate Winslet', 'Billy Zane'],
        ['Sam Neill', 'Laura Dern', 'Jeff Goldblum']
    ],
    'directors': [
        ['Christopher Nolan'], ['Christopher Nolan'], ['Steven Spielberg'],
        ['Guy Ritchie'], ['J.J. Abrams'], ['Peter Jackson'],
        ['Shekhar Kapur'], ['Michael Mann'],
        ['Lana Wachowski', 'Lilly Wachowski'], ['Francis Ford Coppola'],
        ['James Cameron'], ['Quentin Tarantino'], ['David Fincher'],
        ['James Cameron'], ['Steven Spielberg']
    ]
}

movies = pd.DataFrame(data)

print("labeling data...")
mlb_genres = MultiLabelBinarizer()
mlb_actors = MultiLabelBinarizer()
mlb_directors = MultiLabelBinarizer()

genre_vectors = mlb_genres.fit_transform(movies['genres'])
actor_vectors = mlb_actors.fit_transform(movies['actors'])
director_vectors = mlb_directors.fit_transform(movies['directors'])

print("vectorizing data...")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
synopsis_vectors = tfidf_vectorizer.fit_transform(movies['synopsis']).toarray()

def calculate_mean_liked_vector(liked_movie_indices):
    return np.mean(synopsis_vectors[liked_movie_indices, :], axis=0)

def train_likeability_model(features, labels, liked_movie_indices):
    mean_liked_vector = calculate_mean_liked_vector(liked_movie_indices)
    synopsis_distances = cosine_similarity(features[:, -len(tfidf_vectorizer.get_feature_names_out()):], mean_liked_vector.reshape(1, -1)).flatten()
    
    enhanced_features = np.hstack((features, synopsis_distances.reshape(-1, 1)))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(enhanced_features)
    
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
    likeability_model = RandomForestClassifier(n_estimators=100, random_state=42)
    likeability_model.fit(X_train, y_train)
    
    gc.collect()

    return likeability_model, scaler

combined_features = np.hstack((genre_vectors, actor_vectors, director_vectors, synopsis_vectors))

liked_movie_indices = [0, 2, 4]  
labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])  
print("training model...")
likeability_model, scaler = train_likeability_model(combined_features, labels, liked_movie_indices)

def generate_explanation(movie1_idx, movie2_idx):
    shared_actors = np.intersect1d(movies.iloc[movie1_idx]['actors'], movies.iloc[movie2_idx]['actors']).tolist()
    shared_directors = np.intersect1d(movies.iloc[movie1_idx]['directors'], movies.iloc[movie2_idx]['directors']).tolist()
    shared_genres = np.intersect1d(movies.iloc[movie1_idx]['genres'], movies.iloc[movie2_idx]['genres']).tolist()
    synopsis_similarity = cosine_similarity([synopsis_vectors[movie1_idx]], [synopsis_vectors[movie2_idx]])[0][0]
    
    overall_similarity = cosine_similarity([combined_features[movie1_idx]], [combined_features[movie2_idx]])[0][0]

    reasons = {
        'shared_actors': shared_actors,
        'shared_directors': shared_directors,
        'shared_genres': shared_genres,
        'synopsis_similarity': synopsis_similarity,
        'features_similarity': overall_similarity 
    }

    prompt_text = (
        f"Here are the synopses of two movies:\n\n"
        f"1. '{movies.iloc[movie1_idx]['title']}': {movies.iloc[movie1_idx]['synopsis']}\n"
        f"2. '{movies.iloc[movie2_idx]['title']}': {movies.iloc[movie2_idx]['synopsis']}\n\n"
        "Based on the synopses above, without repeating them, please explain if the two movies are similar or not:\n"
    )


    prompt_text = prompt_text[:tokenizer.model_max_length]

    generator = pipeline('text-generation', model='openai-community/gpt2-medium', device= 0) 
    generated_text = generator(clean_text(prompt_text), max_length=230, num_return_sequences=1, temperature=0.4) 

    gc.collect()
    return {"reasons": reasons, "explanation": clean_text(generated_text[0]['generated_text'].split('\n')[0])}


def find_similar_movies_and_explain_and_predict_likeability(movie_id, likeability_model, scaler, top_n=3, liked_movie_indices=None):
    idx = movies.index[movies['movieId'] == movie_id].tolist()[0]
    features = np.hstack((genre_vectors, actor_vectors, director_vectors, synopsis_vectors))

    if liked_movie_indices is not None:
        mean_liked_vector = calculate_mean_liked_vector(liked_movie_indices)
        synopsis_distances = cosine_similarity(features[:, -len(tfidf_vectorizer.get_feature_names_out()):], mean_liked_vector.reshape(1, -1)).flatten()
        features = np.hstack((features, synopsis_distances.reshape(-1, 1)))

    scaled_features = scaler.transform(features) 
    
    print("... predicting likeability")
    likeability_scores = likeability_model.predict_proba(scaled_features)[:, 1]  
    
    print("... predicting similar movies")    
    similarities = cosine_similarity([scaled_features[idx]], scaled_features)[0]
    
    filtered_indices = [i for i in range(len(similarities)) if similarities[i] < 0 or similarities[i] > 0]
    
    if not filtered_indices:
        return "No similar movies found based on the criteria."

    top_indices = np.argsort(similarities[filtered_indices])[-top_n-1:-1][::-1]  
    
    print("... generating explanation")        
    explanations = []
    for i in top_indices:
        explanation_text = generate_explanation(idx, i)
        explanations.append({
            'movie': movies.iloc[i]['title'],
            'explanation': explanation_text,
            'like_probability': likeability_scores[i],
            'full_similarity_score': similarities[i]           
        })

    return explanations


print("finding movies...")
explanations = find_similar_movies_and_explain_and_predict_likeability(2, likeability_model, scaler, 3, liked_movie_indices)
for explanation in explanations:
    print(f"Recommended Movie: {explanation['movie']}")
    print(f"Reasons: {explanation['explanation']['reasons']}")
    print(f"Explanation: {explanation['explanation']['explanation']}")
    print(f"Like Probability: {explanation['like_probability']:.2f}\n")
