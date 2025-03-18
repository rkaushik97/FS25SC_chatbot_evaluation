import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import json
import os
from flask import Flask, render_template, request, jsonify
import openai  

# Flask app
app = Flask(__name__)

# OpenAI API Key (replace with your actual key)
openai.api_key = "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"

# Define fuzzy variables
energy = ctrl.Antecedent(np.arange(0, 101, 1), 'energy')
tempo = ctrl.Antecedent(np.arange(0, 101, 1), 'tempo')
vocals = ctrl.Antecedent(np.arange(0, 11, 1), 'vocals')
instrumentation = ctrl.Antecedent(np.arange(0, 11, 1), 'instrumentation')
melodic = ctrl.Antecedent(np.arange(0, 11, 1), 'melodic')
lyrics_sentiment = ctrl.Antecedent(np.arange(-1, 2, 1), 'lyrics_sentiment')
mood = ctrl.Consequent(np.arange(0, 101, 1), 'mood')

# Define membership functions for energy
energy['very_low'] = fuzz.trimf(energy.universe, [0, 0, 25])
energy['low'] = fuzz.trimf(energy.universe, [0, 25, 50])
energy['medium'] = fuzz.trimf(energy.universe, [25, 50, 75])
energy['high'] = fuzz.trimf(energy.universe, [50, 75, 100])
energy['very_high'] = fuzz.trimf(energy.universe, [75, 100, 100])

# Define membership functions for tempo
tempo['very_slow'] = fuzz.trimf(tempo.universe, [0, 0, 25])
tempo['slow'] = fuzz.trimf(tempo.universe, [0, 25, 50])
tempo['moderate'] = fuzz.trimf(tempo.universe, [25, 50, 75])
tempo['fast'] = fuzz.trimf(tempo.universe, [50, 75, 100])
tempo['very_fast'] = fuzz.trimf(tempo.universe, [75, 100, 100])

# Define membership functions for vocals
vocals['very_low'] = fuzz.trimf(vocals.universe, [0, 0, 3])
vocals['low'] = fuzz.trimf(vocals.universe, [0, 3, 6])
vocals['medium'] = fuzz.trimf(vocals.universe, [3, 6, 9])
vocals['high'] = fuzz.trimf(vocals.universe, [6, 9, 10])
vocals['very_high'] = fuzz.trimf(vocals.universe, [9, 10, 10])

# Define membership functions for instrumentation
instrumentation['very_sparse'] = fuzz.trimf(instrumentation.universe, [0, 0, 3])
instrumentation['sparse'] = fuzz.trimf(instrumentation.universe, [0, 3, 6])
instrumentation['moderate'] = fuzz.trimf(instrumentation.universe, [3, 6, 9])
instrumentation['rich'] = fuzz.trimf(instrumentation.universe, [6, 9, 10])
instrumentation['very_rich'] = fuzz.trimf(instrumentation.universe, [9, 10, 10])

# Define membership functions for melodic
melodic['very_non_melodic'] = fuzz.trimf(melodic.universe, [0, 0, 3])
melodic['non_melodic'] = fuzz.trimf(melodic.universe, [0, 3, 6])
melodic['somewhat_melodic'] = fuzz.trimf(melodic.universe, [3, 6, 9])
melodic['melodic'] = fuzz.trimf(melodic.universe, [6, 9, 10])
melodic['very_melodic'] = fuzz.trimf(melodic.universe, [9, 10, 10])

# Define membership functions for lyrics sentiment
lyrics_sentiment['negative'] = fuzz.trimf(lyrics_sentiment.universe, [-1, -1, 0])
lyrics_sentiment['neutral'] = fuzz.trimf(lyrics_sentiment.universe, [-1, 0, 1])
lyrics_sentiment['positive'] = fuzz.trimf(lyrics_sentiment.universe, [0, 1, 1])

# Define membership functions for mood
mood['very_calm'] = fuzz.trimf(mood.universe, [0, 0, 20])
mood['calm'] = fuzz.trimf(mood.universe, [0, 20, 40])
mood['neutral'] = fuzz.trimf(mood.universe, [20, 40, 60])
mood['energetic'] = fuzz.trimf(mood.universe, [40, 60, 80])
mood['very_energetic'] = fuzz.trimf(mood.universe, [60, 80, 100])

# Define fuzzy rules
rules = [
    ctrl.Rule(energy['very_low'] & tempo['very_slow'] & vocals['very_low'] & instrumentation['very_sparse'] & melodic['very_non_melodic'] & lyrics_sentiment['negative'], mood['very_calm']),
    ctrl.Rule(energy['low'] & tempo['slow'] & vocals['low'] & instrumentation['sparse'] & melodic['non_melodic'] & lyrics_sentiment['neutral'], mood['calm']),
    ctrl.Rule(energy['medium'] & tempo['moderate'] & vocals['medium'] & instrumentation['moderate'] & melodic['somewhat_melodic'] & lyrics_sentiment['neutral'], mood['neutral']),
    ctrl.Rule(energy['high'] & tempo['fast'] & vocals['high'] & instrumentation['rich'] & melodic['melodic'] & lyrics_sentiment['positive'], mood['energetic']),
    ctrl.Rule(energy['very_high'] & tempo['very_fast'] & vocals['very_high'] & instrumentation['very_rich'] & melodic['very_melodic'] & lyrics_sentiment['positive'], mood['very_energetic'])
]

# Create the control system
music_ctrl = ctrl.ControlSystem(rules)
music_sim = ctrl.ControlSystemSimulation(music_ctrl)

# Song Data
songs = [
    {
        "title": "Blinding Lights",
        "artist": "The Weeknd",
        "genre": "Synthwave",
        "release_year": 2019,
        "description": "A high-energy synthwave track with a retro vibe, perfect for late-night drives."
    },
    {
        "title": "Bohemian Rhapsody",
        "artist": "Queen",
        "genre": "Rock",
        "release_year": 1975,
        "description": "An epic rock ballad with dramatic shifts in tempo and mood."
    }
]

# Function to generate ChatGPT response
def get_chatgpt_response(song, mood_description):
    # Create the prompt string
    prompt = (
        f"Analyze the song '{song['title']}' by {song['artist']} based on its mood ('{mood_description}'), "
        f"instrumentation, energy, tempo, and lyrical sentiment. Provide a detailed, engaging breakdown."
    )
    
    try:
        # Send the request to OpenAI's API with the new interface
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or any other model you want to use
            messages=[  # Use the `messages` field to pass in the conversation
                {"role": "system", "content": "You are a music critic."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the response content
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']  # Corrected this line
        else:
            return "Sorry, I couldn't generate a response at the moment."
    
    except Exception as e:
        # If an error occurs during the API call
        return f"An error occurred: {str(e)}"
    

# Homepage
@app.route('/')
def home():
    return render_template('index.html', songs=songs)

# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    # Get user input
    song_title = request.form['song_title']
    energy_input = int(request.form['energy'])
    tempo_input = int(request.form['tempo'])
    vocals_input = int(request.form['vocals'])
    instrumentation_input = int(request.form['instrumentation'])
    melodic_input = int(request.form['melodic'])
    lyrics_sentiment_input = int(request.form['lyrics_sentiment'])
    
    # Validate input values
    if not (0 <= energy_input <= 100):
        return "Error: Energy Level must be between 0 and 100."
    if not (0 <= tempo_input <= 100):
        return "Error: Tempo must be between 0 and 100."
    if not (1 <= vocals_input <= 10):
        return "Error: Vocals Prominence must be between 1 and 10."
    if not (1 <= instrumentation_input <= 10):
        return "Error: Instrumentation Richness must be between 1 and 10."
    if not (1 <= melodic_input <= 10):
        return "Error: Melodic Quality must be between 1 and 10."
    if lyrics_sentiment_input not in [-1, 0, 1]:
        return "Error: Lyrics Sentiment must be -1 (Negative), 0 (Neutral), or 1 (Positive)."
    
    # Find song details
    selected_song = next((song for song in songs if song["title"] == song_title), None)

    # Add all input variables to the system
    music_sim.input['energy'] = energy_input
    music_sim.input['tempo'] = tempo_input
    music_sim.input['vocals'] = vocals_input
    music_sim.input['instrumentation'] = instrumentation_input
    music_sim.input['melodic'] = melodic_input
    music_sim.input['lyrics_sentiment'] = lyrics_sentiment_input

    # Compute fuzzy output
    try:
        music_sim.compute()
        mood_value = music_sim.output['mood']
    except KeyError:
        return "Error: Unable to compute mood. Please check your input values."

    mood_description = "calm" if mood_value < 40 else "energetic" if mood_value < 80 else "very energetic"

    # Generate fuzzy chatbot response
    fuzzy_response = (
        f"🎶 '{selected_song['title']}' by {selected_song['artist']}' is an intriguing blend of sound and emotion, "
        f"with an overall mood that feels {mood_description}. Released in {selected_song['release_year']}, this {selected_song['genre']} track is {selected_song['description'].lower()}. "
        
        f"\n\n🎵 Energy & Tempo: The song carries an energy level of {energy_input}, making it {'gentle and introspective' if energy_input < 40 else 'vibrant and electrifying' if energy_input > 75 else 'perfectly balanced between relaxation and movement'}. "
        f"With a tempo of {tempo_input} BPM, it {'flows like a slow-burning ballad, perfect for quiet reflection' if tempo_input < 50 else 'has an infectious rhythm that keeps you engaged' if tempo_input < 80 else 'drives forward relentlessly, impossible to ignore'}. "
        
        f"\n\n🎤 Vocals & Instrumentation: The vocals here are {'soft and intimate, drawing you in with their subtlety' if vocals_input < 4 else 'rich and expressive, adding depth and emotion' if vocals_input < 7 else 'powerful and commanding, taking center stage' if vocals_input < 10 else 'booming and dynamic, leading the entire musical experience'}. "
        f"The instrumentation is {'minimal and delicate, leaving space for emotions to breathe' if instrumentation_input < 4 else 'well-balanced, giving equal weight to every instrument' if instrumentation_input < 7 else 'lush and immersive, wrapping you in a sonic embrace' if instrumentation_input < 10 else 'orchestral and grand, creating a massive, full-bodied sound'}. "
        
        f"\n\n🎼 Melody & Lyrical Sentiment: The melody leans towards {'subdued and experimental, designed to challenge traditional structures' if melodic_input < 4 else 'melancholic yet beautiful, weaving nostalgia into its core' if melodic_input < 7 else 'instantly memorable, with hooks that stay with you long after the song ends' if melodic_input < 10 else 'utterly euphoric, soaring to breathtaking highs'}. "
        f"The lyrics convey {'a deep sadness, carrying the weight of raw emotion' if lyrics_sentiment_input == -1 else 'a reflective neutrality, letting the listener interpret their meaning' if lyrics_sentiment_input == 0 else 'an uplifting sense of hope and joy, radiating positivity'}. "
        
        f"\n\n🔥 Final Verdict: Whether you’re looking to {'unwind and drift into a contemplative mood' if mood_value < 40 else 'find a song that transitions effortlessly between moments of peace and energy' if mood_value < 60 else 'get lost in a wave of pure adrenaline and motion' if mood_value < 80 else 'experience a sonic explosion of exhilaration'}, "
        f"this track delivers an unforgettable experience. Press play and let the music take over! 🎧"
    )

    # Generate ChatGPT response
    chatgpt_response = get_chatgpt_response(selected_song, mood_description)

    return render_template('index.html', response_fuzzy=fuzzy_response, response_gpt=chatgpt_response, songs=songs, selected_song=selected_song)

# Save user ratings
@app.route('/rate', methods=['POST'])
def rate():
    try:
        data = request.json
        if not os.path.exists("ratings.json"):
            with open("ratings.json", "w") as f:
                json.dump([], f)

        with open("ratings.json", "r+") as f:
            ratings = json.load(f)
            ratings.append(data)
            f.seek(0)
            json.dump(ratings, f, indent=4)

        return jsonify({"message": "Rating saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)