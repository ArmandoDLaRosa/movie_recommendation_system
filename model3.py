import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

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


def prepare_documents(data):
    documents = []
    for i in range(len(data['movieId'])):
        doc_text = f"{data['title'][i]}. {data['synopsis'][i]} Genres: {', '.join(data['genres'][i])}. "
        doc_text += f"Actors: {', '.join(data['actors'][i])}. Director: {', '.join(data['directors'][i])}."
        documents.append({"doc_id": data['movieId'][i], "text": doc_text, "title": data['title'][i]})
    return documents

documents = prepare_documents(data)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([doc['text'] for doc in documents])
tfidf_vectors = normalize(tfidf_matrix, norm='l2', axis=1).toarray()

dimension = tfidf_vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(tfidf_vectors)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

def search_index(query, vectorizer, index, documents, k=3):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    faiss.normalize_L2(query_vec)
    
    distances, indices = index.search(query_vec, k)
    
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        if (1 - distance) != 0: 
            results.append((documents[idx]['title'], documents[idx]['text']))
            if len(results) == k:
                break
    
    return results

def generate_response(query, results):
    responses = []

    text_generator = pipeline("text-generation", model="openai-community/gpt2-medium", device=0)
    for title, text in results:
        
        prompt_text = f"Movie: {title}.\n"
        prompt_text += f"Description: {text}.\n"
        prompt_text += f"Query: {query}.\n\n" 
        prompt_text += f"Purely based on the description, briefly explain how it relates to the query:\n"
        response = text_generator(prompt_text, max_length=150, num_return_sequences=1, temperature=0.3)[0]['generated_text']

        responses.append(response)

    return responses

query = "Movies about a ship"
results = search_index(query, vectorizer, index, documents, k=3)
response = generate_response(query, results)

for prompt in response:
    print(prompt)
    print("\n")
