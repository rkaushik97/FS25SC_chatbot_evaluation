"""
A dynamic and responsive music review system that takes user input about various song attributes
(like energy, tempo, vocals, instrumentation, melodic complexity, and lyrical sentiment) 
and generates a detailed and engaging review.
"""
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import json
import os
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# Flask app
app = Flask(__name__)

# Define fuzzy variables
energy = ctrl.Antecedent(np.arange(0, 101, 1), 'energy')
tempo = ctrl.Antecedent(np.arange(0, 101, 1), 'tempo')
vocals = ctrl.Antecedent(np.arange(0, 11, 1), 'vocals')
instrumentation = ctrl.Antecedent(np.arange(0, 11, 1), 'instrumentation')
melodic = ctrl.Antecedent(np.arange(0, 11, 1), 'melodic')
lyrics_sentiment = ctrl.Antecedent(np.arange(-1, 2, 1), 'lyrics_sentiment')
mood = ctrl.Consequent(np.arange(0, 101, 1), 'mood')

# Define membership functions
def define_membership(var, labels_ranges):
    for label, pts in labels_ranges.items():
        var[label] = fuzz.trimf(var.universe, pts)

define_membership(energy, {
    'very_low': [0, 0, 25],
    'low': [0, 25, 50],
    'medium': [25, 50, 75],
    'high': [50, 75, 100],
    'very_high': [75, 100, 100]
})

define_membership(tempo, {
    'very_slow': [0, 0, 25],
    'slow': [0, 25, 50],
    'moderate': [25, 50, 75],
    'fast': [50, 75, 100],
    'very_fast': [75, 100, 100]
})

define_membership(vocals, {
    'very_low': [0, 0, 3],
    'low': [0, 3, 6],
    'medium': [3, 6, 9],
    'high': [6, 9, 10],
    'very_high': [9, 10, 10]
})

define_membership(instrumentation, {
    'very_sparse': [0, 0, 3],
    'sparse': [0, 3, 6],
    'moderate': [3, 6, 9],
    'rich': [6, 9, 10],
    'very_rich': [9, 10, 10]
})

define_membership(melodic, {
    'very_non_melodic': [0, 0, 3],
    'non_melodic': [0, 3, 6],
    'somewhat_melodic': [3, 6, 9],
    'melodic': [6, 9, 10],
    'very_melodic': [9, 10, 10]
})

define_membership(lyrics_sentiment, {
    'negative': [-1, -1, 0],
    'neutral': [-1, 0, 1],
    'positive': [0, 1, 1]
})

define_membership(mood, {
    'very_calm': [0, 0, 20],
    'calm': [0, 20, 40],
    'neutral': [20, 40, 60],
    'energetic': [40, 60, 80],
    'very_energetic': [60, 80, 100]
})

# Define fuzzy rules
rules = []

# Helper aliases
e = energy
t = tempo
s = lyrics_sentiment
m = mood

# Very Calm to Calm
rules += [
    ctrl.Rule(e['very_low'] | t['very_slow'] | s['negative'], m['very_calm']),
    ctrl.Rule(e['low'] & t['slow'], m['calm']),
    ctrl.Rule((e['low'] & s['neutral']) | (t['slow'] & s['neutral']), m['calm']),
]

# Neutral
rules += [
    ctrl.Rule(e['medium'] & t['moderate'] & s['neutral'], m['neutral']),
    ctrl.Rule((e['medium'] & t['moderate']) | (s['neutral']), m['neutral']),
]

# Energetic
rules += [
    ctrl.Rule(e['high'] & t['fast'] & s['positive'], m['energetic']),
    ctrl.Rule(e['high'] | t['fast'], m['energetic']),
]

# Very Energetic
rules += [
    ctrl.Rule(e['very_high'] & t['very_fast'] & s['positive'], m['very_energetic']),
    ctrl.Rule((e['very_high'] | t['very_fast']) & s['positive'], m['very_energetic']),
]

# Sentiment-only influence (fallback)
rules += [
    ctrl.Rule(s['positive'], m['energetic']),
    ctrl.Rule(s['negative'], m['very_calm']),
]

# Boost if vocals/instrumentation/melodic are strong
rules += [
    ctrl.Rule(vocals['very_high'] | instrumentation['very_rich'] | melodic['very_melodic'], m['energetic']),
    ctrl.Rule(vocals['very_low'] & instrumentation['very_sparse'] & melodic['very_non_melodic'], m['very_calm']),
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
    },
    {
        "title": "Shape of You",
        "artist": "Ed Sheeran",
        "genre": "Pop",
        "release_year": 2017,
        "description": "A catchy pop song with tropical house influences and romantic lyrics."
    },
    {
        "title": "Smells Like Teen Spirit",
        "artist": "Nirvana",
        "genre": "Grunge",
        "release_year": 1991,
        "description": "A defining grunge anthem with raw energy and rebellious lyrics."
    },
    {
        "title": "Rolling in the Deep",
        "artist": "Adele",
        "genre": "Soul",
        "release_year": 2010,
        "description": "A powerful soul ballad about heartbreak and empowerment."
    },
    {
        "title": "Uptown Funk",
        "artist": "Mark Ronson ft. Bruno Mars",
        "genre": "Funk",
        "release_year": 2014,
        "description": "A modern funk track with infectious energy and retro vibes."
    },
    {
        "title": "Someone Like You",
        "artist": "Adele",
        "genre": "Ballad",
        "release_year": 2011,
        "description": "A emotional piano ballad about lost love and moving on."
    },
    {
        "title": "Bad Guy",
        "artist": "Billie Eilish",
        "genre": "Electropop",
        "release_year": 2019,
        "description": "A dark, bass-heavy electropop song with whispery vocals."
    },
    {
        "title": "Sweet Child O' Mine",
        "artist": "Guns N' Roses",
        "genre": "Rock",
        "release_year": 1987,
        "description": "A classic rock song with iconic guitar riffs and emotional vocals."
    },
    {
        "title": "Despacito",
        "artist": "Luis Fonsi ft. Daddy Yankee",
        "genre": "Reggaeton",
        "release_year": 2017,
        "description": "A global reggaeton hit with infectious rhythm and romantic lyrics."
    },
    {
        "title": "Havana",
        "artist": "Camila Cabello",
        "genre": "Pop",
        "release_year": 2017,
        "description": "A Latin-infused pop song with sultry vocals and tropical rhythms."
    },
    {
        "title": "Thunderstruck",
        "artist": "AC/DC",
        "genre": "Hard Rock",
        "release_year": 1990,
        "description": "A high-voltage hard rock anthem with electrifying guitar riffs."
    }
]

# Linguistic label extraction
def get_linguistic_label(variable, score):
    print('Entered', variable)
    if variable == 'energy':
        
        if score < 30:
            return "lifeless"  # Harsh description for very low energy
        elif score < 60:
            return "low energy"
        elif score < 90:
            return "high energy"
        else:
            return "explosive energy"
        
    elif variable == 'tempo':
        if score < 30:
            return "sluggish"  # Harsh description for very slow tempo
        elif score < 60:
            return "slow-paced"
        elif score < 90:
            return "fast-paced"
        else:
            return "rapid fire tempo"
        
    elif variable == 'vocals':
        if score < 3:
            return "inaudible"  # Harsh description for very low vocals
        elif score < 6:
            return "barely noticeable"
        elif score < 8:
            return "clear and present"
        else:
            return "dominant and powerful"
        
    elif variable == 'instrumentation':
        if score < 3:
            return "almost nonexistent"  # Harsh description for minimal instrumentation
        elif score < 6:
            return "sparse instrumentation"
        elif score < 8:
            return "rich and full"
        else:
            return "overwhelming instrumentation"
        
    elif variable == 'melodic':
        if score < 3:
            return "monotonous"  # Harsh description for very simple melody
        elif score < 6:
            return "uncomplicated"
        elif score < 8:
            return "catchy"
        else:
            return "complex and intricate"
        
    elif variable == 'lyrics_sentiment':
        if score < 0:
            return "despondent"  # Harsh description for very negative sentiment
        elif score == 0:
            return "neutral"
        else:
            return "uplifting"

summarizer = pipeline("summarization", model="google-t5/t5-small")

def summarize_fuzzy_response(text):
    # Optional: Clean Markdown or emojis if model struggles
    cleaned_text = text.replace("🎶", "").replace("💡", "")
    
    # Calculate the dynamic max_length and min_length based on the text length
    text_length = len(cleaned_text.split())  # Split the text by spaces to get word count
    max_length = min(200, text_length + 50)  # Set max_length to 200 or a bit more than the text length
    min_length = max(50, int(text_length * 0.5))  # Set min_length to at least 50 or half the text length

    try:
        summary = summarizer(cleaned_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"(Could not summarize: {str(e)})"


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

# Set the device to use for generation (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to generate local LLM response using FLAN-T5
def get_local_llm_response(song, mood_description, energy_input, tempo_input, vocals_input, instrumentation_input, melodic_input, lyrics_sentiment_input, mood_value):
  
    prompt = (
    f"Analyze the song '{song['title']}' by {song['artist']} ({song['release_year']}, {song['genre']}) based on its attributes. "
    f"The song is described as: '{song['description']}'.\n\n"
    f"Step 1: Evaluate the song's core elements and consider the following numerical values:\n"
    f"- Energy (0-100): {energy_input} → Does this energy level feel high, mellow, or balanced? Does it reflect excitement or calmness?\n"
    f"- Tempo (BPM): {tempo_input} → Is the pace slow, moderate, or fast? How does this contribute to the song's feel?\n"
    f"- Vocals (0-10): {vocals_input} → Are the vocals more powerful or subtle? Do they stand out or complement the instrumentation?\n"
    f"- Instrumentation (0-10): {instrumentation_input} → Is the instrumentation sparse, rich, or balanced? How does it affect the mood?\n"
    f"- Melodic Complexity (0-10): {melodic_input} → Does the melody feel complex, intricate, or simple? Does it shift or remain steady?\n"
    f"- Lyrical Sentiment (-1 to 1): {lyrics_sentiment_input} → Does the song have uplifting, neutral, or melancholic lyrics? How does this relate to the overall feel?\n"
    f"- Overall Mood Score (0-100): {mood_value} → Does the song feel mellow, neutral, energetic, or intense based on the combination of the attributes?\n\n"
    f"Step 2: Final Verdict:\n"
    f"Provide a review that incorporates these elements: energy, tempo, vocals, instrumentation, melodic complexity, lyrical sentiment, and mood. "
    f"Highlight how the song's numerical values guide the overall assessment. Make the tone concise, engaging, and fitting for a quick review (100-150 words). "
    f"Ensure that the review aligns with the values of energy, tempo, and lyrical sentiment as they have been quantified."
    )

    print(prompt)
    try:
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_length = len(inputs.input_ids[0]) 
        max_length = prompt_length + 150
        # Generate output using FLAN-T5
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,  # Ensure it doesn't generate overly long responses
            temperature=0.7,
            top_p=1.0,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=150
        )
        
        # Decode the output to a string
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response (if it includes the prompt at the beginning)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"

def simulate_user_input():
    """Generate random user input for testing"""
    return {
        'energy': random.randint(0, 100),
        'tempo': random.randint(0, 100),
        'vocals': random.randint(0, 10),
        'instrumentation': random.randint(0, 10),
        'melodic': random.randint(0, 10),
        'lyrics_sentiment': random.choice([-1, 0, 1])
    }

def simulate_feedback(chatbot_type):
    """Simulate user feedback for a chatbot response"""
    feedback_options = {
        "positive": ["Great analysis!", "Spot on!", "Perfect review!", "Exactly what I was thinking!"],
        "neutral": ["It's okay.", "Not bad.", "Could be better.", "Decent review."],
        "negative": ["Completely off!", "This doesn't make sense.", "Wrong analysis.", "Not helpful at all."]
    }
    
    # Randomly choose feedback sentiment
    sentiment = random.choice(["positive", "neutral", "negative"])
    return random.choice(feedback_options[sentiment])

def generate_fuzzy_response(song, user_input):
    """Generate a fuzzy logic response for testing"""
    # Add all input variables to the system
    music_sim.input['energy'] = user_input['energy']
    music_sim.input['tempo'] = user_input['tempo']
    music_sim.input['vocals'] = user_input['vocals']
    music_sim.input['instrumentation'] = user_input['instrumentation']
    music_sim.input['melodic'] = user_input['melodic']
    music_sim.input['lyrics_sentiment'] = user_input['lyrics_sentiment']
    
    # Compute fuzzy output
    music_sim.compute()
    mood_value = music_sim.output['mood']
    
    mood_description = get_linguistic_label('mood', mood_value)
    energy_desc = get_linguistic_label('energy', user_input['energy'])
    tempo_desc = get_linguistic_label('tempo', user_input['tempo'])
    vocals_desc = get_linguistic_label('vocals', user_input['vocals'])
    instr_desc = get_linguistic_label('instrumentation', user_input['instrumentation'])
    melodic_desc = get_linguistic_label('melodic', user_input['melodic'])
    sentiment_desc = get_linguistic_label('lyrics_sentiment', user_input['lyrics_sentiment'])

    return (
        f"🎶 '{song['title']}' by {song['artist']} feels {mood_description} overall. "
        f"This {song['genre']} track is {song['description'].lower()}.\n\n"
        f"🔋 Energy: {energy_desc}, 🎵 Tempo: {tempo_desc}.\n"
        f"🎤 Vocals: {vocals_desc}, 🎹 Instrumentation: {instr_desc}.\n"
        f"🎼 Melody: {melodic_desc}, 📝 Lyrics Sentiment: {sentiment_desc}."
    )

def generate_llm_response(song, user_input):
    """Generate an LLM response for testing"""
    # Add all input variables to the system
    music_sim.input['energy'] = user_input['energy']
    music_sim.input['tempo'] = user_input['tempo']
    music_sim.input['vocals'] = user_input['vocals']
    music_sim.input['instrumentation'] = user_input['instrumentation']
    music_sim.input['melodic'] = user_input['melodic']
    music_sim.input['lyrics_sentiment'] = user_input['lyrics_sentiment']
    
    # Compute fuzzy output
    music_sim.compute()
    mood_value = music_sim.output['mood']
    
    mood_description = get_linguistic_label('mood', mood_value)
    
    return get_local_llm_response(
        song, mood_description, 
        user_input['energy'], user_input['tempo'], 
        user_input['vocals'], user_input['instrumentation'], 
        user_input['melodic'], user_input['lyrics_sentiment'], 
        mood_value
    )

def generate_responses_and_feedback():
    """Generate simulated responses and feedback for testing"""
    feedback_data = []

    for i in range(12):  # Simulate 12 user inputs
        print('User: ', i)
        user_input = simulate_user_input()
        print('Input: ', user_input)
        song = random.choice(songs)
        
        # Generate responses
        fuzzy_response = generate_fuzzy_response(song, user_input)
        llm_response = generate_llm_response(song, user_input)

        # Simulate feedback for both chatbots
        feedback_data.append({
            "song": song['title'],
            "artist": song['artist'],
            "user_input": user_input,
            "fuzzy_logic_response": fuzzy_response,
            "llm_response": llm_response,
            "fuzzy_logic_feedback": simulate_feedback("fuzzy_logic"),
            "llm_feedback": simulate_feedback("generative_ai")
        })
        print({
            "song": song['title'],
            "artist": song['artist'],
            "user_input": user_input,
            "fuzzy_logic_response": fuzzy_response,
            "llm_response": llm_response,
            "fuzzy_logic_feedback": simulate_feedback("fuzzy_logic"),
            "llm_feedback": simulate_feedback("generative_ai")
        })
        print('Finished processing user: ', i)
    
    # Save feedback data to a JSON file
    output_filename = f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w") as f:
        json.dump(feedback_data, f, indent=4)

    print(f"Feedback data saved to {output_filename}")
    return feedback_data

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

    mood_description = get_linguistic_label('mood', mood_value)
    energy_desc = get_linguistic_label('energy', energy_input)
    tempo_desc = get_linguistic_label('tempo', tempo_input)
    vocals_desc = get_linguistic_label('vocals', vocals_input)
    instr_desc = get_linguistic_label('instrumentation', instrumentation_input)
    melodic_desc = get_linguistic_label('melodic', melodic_input)
    sentiment_desc = get_linguistic_label('lyrics_sentiment', lyrics_sentiment_input)

    # Generate fuzzy chatbot response
    raw_fuzzy_response = (
        f"🎶 '{selected_song['title']}' by {selected_song['artist']} feels {mood_description} overall. "
        f"This {selected_song['genre']} track, released in {selected_song['release_year']}, is {selected_song['description'].lower()}.\n\n"
        f"🔋 **Energy:** {energy_desc.capitalize()}, 🎵 **Tempo:** {tempo_desc.replace('_', ' ').capitalize()}.\n"
        f"🎤 **Vocals:** {vocals_desc.replace('_', ' ')}, 🎹 **Instrumentation:** {instr_desc.replace('_', ' ')}.\n"
        f"🎼 **Melody:** {melodic_desc.replace('_', ' ')}, 📝 **Lyrics Sentiment:** {sentiment_desc}.\n\n"
        f"💡 A harshly subdued mix of sound and sentiment, this track captures a {mood_description} vibe that's "
        f"perfect for those seeking {('relaxation' if mood_value < 40 else 'motivation' if mood_value < 70 else 'intensity')}."
    )

    fuzzy_response = summarize_fuzzy_response(raw_fuzzy_response)
    llm_response = get_local_llm_response(
        selected_song, mood_description, energy_input, tempo_input, vocals_input, instrumentation_input, melodic_input, lyrics_sentiment_input, mood_value
    )

    return render_template('index.html', response_fuzzy=fuzzy_response, response_gpt=llm_response, songs=songs, selected_song=selected_song)

# Save user feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        if not os.path.exists("feedback.json"):
            with open("feedback.json", "w") as f:
                json.dump([], f)

        with open("feedback.json", "r+") as f:
            feedback_data = json.load(f)
            feedback_data.append(data)
            f.seek(0)
            json.dump(feedback_data, f, indent=4)

        return jsonify({"message": "Feedback saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Generate test data route
@app.route('/generate_test_data', methods=['GET'])
def generate_test_data():
    try:
        feedback_data = generate_responses_and_feedback()
        return jsonify({
            "message": "Test data generated successfully",
            "data": feedback_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)