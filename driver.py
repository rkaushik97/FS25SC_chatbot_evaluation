import json
import random
from datetime import datetime
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import json
import os
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


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


# Define a list of songs
songs = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "genre": "Synthwave", "release_year": 2019, "description": "A high-energy synthwave track with a retro vibe, perfect for late-night drives."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "release_year": 2017, "description": "A danceable pop track with catchy lyrics and a memorable melody."},
    {"title": "Imagine", "artist": "John Lennon", "genre": "Pop", "release_year": 1971, "description": "A hopeful anthem calling for world peace and unity."},
    {"title": "Rolling in the Deep", "artist": "Adele", "genre": "Soul", "release_year": 2010, "description": "A soulful track with deep, powerful vocals and emotional lyrics."},
    {"title": "Uptown Funk", "artist": "Mark Ronson feat. Bruno Mars", "genre": "Funk", "release_year": 2014, "description": "An energetic and funky track that encourages dancing and having a good time."},
    {"title": "Let It Be", "artist": "The Beatles", "genre": "Rock", "release_year": 1970, "description": "A timeless ballad with an uplifting message of acceptance."},
    {"title": "Billie Jean", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "description": "A catchy pop track with a deep groove and unforgettable bassline."},
    {"title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Grunge", "release_year": 1991, "description": "A defining track of the grunge movement with raw energy and angst."},
    {"title": "Hurt", "artist": "Nine Inch Nails", "genre": "Industrial", "release_year": 1994, "description": "A dark and haunting track that explores themes of despair and self-reflection."},
    {"title": "Old Town Road", "artist": "Lil Nas X", "genre": "Country Rap", "release_year": 2018, "description": "A catchy fusion of country and rap with a viral hook."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Hey Jude", "artist": "The Beatles", "genre": "Pop", "release_year": 1968, "description": "A classic anthem with emotional depth and a powerful sing-along outro."},
    {"title": "What a Wonderful World", "artist": "Louis Armstrong", "genre": "Jazz", "release_year": 1967, "description": "A timeless jazz standard with a message of gratitude and beauty."},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Rock", "release_year": 1975, "description": "An epic rock ballad with dramatic shifts in tempo and mood."},
    {"title": "Hey Ya!", "artist": "OutKast", "genre": "Hip-Hop", "release_year": 2003, "description": "A funky track that blends hip-hop and pop, famous for its catchy rhythm."},
    {"title": "Like a Rolling Stone", "artist": "Bob Dylan", "genre": "Folk Rock", "release_year": 1965, "description": "A revolutionary track with poignant lyrics and an innovative sound."},
    {"title": "I Will Always Love You", "artist": "Whitney Houston", "genre": "Pop", "release_year": 1992, "description": "A powerful ballad showcasing Whitney Houston's unmatched vocal range."},
    {"title": "Livin' on a Prayer", "artist": "Bon Jovi", "genre": "Rock", "release_year": 1986, "description": "An anthem of resilience and hope, with a catchy chorus and upbeat energy."},
    {"title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "genre": "Hard Rock", "release_year": 1987, "description": "A hard rock classic with iconic guitar riffs and heartfelt lyrics."},
    {"title": "Thriller", "artist": "Michael Jackson", "genre": "Pop", "release_year": 1982, "description": "A groundbreaking pop song with a legendary music video and spooky vibes."}
]

def get_linguistic_label(variable, score):
    if variable == 'energy':
        if score < 30:
            return "lifeless"
        elif score < 60:
            return "low energy"
        elif score < 90:
            return "high energy"
        else:
            return "explosive energy"
    elif variable == 'tempo':
        if score < 30:
            return "sluggish"
        elif score < 60:
            return "slow-paced"
        elif score < 90:
            return "fast-paced"
        else:
            return "rapid fire tempo"
    elif variable == 'vocals':
        if score < 3:
            return "inaudible"
        elif score < 6:
            return "barely noticeable"
        elif score < 8:
            return "clear and present"
        else:
            return "dominant and powerful"
    elif variable == 'instrumentation':
        if score < 3:
            return "almost nonexistent"
        elif score < 6:
            return "sparse instrumentation"
        elif score < 8:
            return "rich and full"
        else:
            return "overwhelming instrumentation"
    elif variable == 'melodic':
        if score < 3:
            return "monotonous"
        elif score < 6:
            return "uncomplicated"
        elif score < 8:
            return "catchy"
        else:
            return "complex and intricate"
    elif variable == 'lyrics_sentiment':
        if score < 0:
            return "despondent"
        elif score < 0.5:
            return "neutral"
        else:
            return "uplifting"
    elif variable == 'mood':
        if score < 20:
            return "very calm"
        elif score < 40:
            return "calm"
        elif score < 60:
            return "neutral"
        elif score < 80:
            return "energetic"
        else:
            return "very energetic"
    # Default return if none of the above matches
    return "neutral"

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
def generate_llm_response(song, user_input):
    energy_input = user_input['energy']
    # song_title = user_input['title']
    energy_input = int(user_input['energy'])
    tempo_input = int(user_input['tempo'])
    vocals_input = int(user_input['vocals'])
    instrumentation_input = int(user_input['instrumentation'])
    melodic_input = int(user_input['melodic'])
    lyrics_sentiment_input = int(user_input['lyrics_sentiment'])

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
    f"Step 2: Final Verdict:\n"
    f"Provide a review that incorporates these elements: energy, tempo, vocals, instrumentation, melodic complexity, lyrical sentiment, and mood. "
    f"Highlight how the song's numerical values guide the overall assessment. Make the tone concise, engaging, and fitting for a quick review (100-150 words). "
    f"Ensure that the review aligns with the values of energy, tempo, and lyrical sentiment as they have been quantified."
    )

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

def generate_fuzzy_response(song, user_input):
    energy_input = user_input['energy']
    # song_title = user_input['title']
    energy_input = int(user_input['energy'])
    tempo_input = int(user_input['tempo'])
    vocals_input = int(user_input['vocals'])
    instrumentation_input = int(user_input['instrumentation'])
    melodic_input = int(user_input['melodic'])
    lyrics_sentiment_input = int(user_input['lyrics_sentiment'])
    print(lyrics_sentiment_input)

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
    f"🎶 '{song['title']}' by {song['artist']} feels {mood_description} overall. "
    f"This {song['genre']} track, released in {song['release_year']}, is {song['description'].lower()}.\n\n"
    f"🔋 **Energy:** {energy_desc.capitalize()}, 🎵 **Tempo:** {tempo_desc.replace('_', ' ').capitalize()}.\n"
    f"🎤 **Vocals:** {vocals_desc.replace('_', ' ')}, 🎹 **Instrumentation:** {instr_desc.replace('_', ' ')}.\n"
    f"🎼 **Melody:** {melodic_desc.replace('_', ' ')}, 📝 **Lyrics Sentiment:** {sentiment_desc}.\n\n"
    f"💡 A harshly subdued mix of sound and sentiment, this track captures a {mood_description} vibe that's "
    f"perfect for those seeking {('relaxation' if mood_value < 40 else 'motivation' if mood_value < 70 else 'intensity')}."
    )

    fuzzy_response = summarize_fuzzy_response(raw_fuzzy_response)
    return fuzzy_response

# Helper function to simulate random user inputs
def simulate_user_input():
    return {
        'energy': random.randint(0, 100),
        'tempo': random.randint(0, 100),
        'vocals': random.randint(0, 10),
        'instrumentation': random.randint(0, 10),
        'melodic': random.randint(0, 10),
        'lyrics_sentiment': random.randint(-1, 1)
    }

# Helper function to generate simulated feedback
def simulate_feedback(chatbot_type):
    return {
        "clarity": str(random.randint(1, 5)),
        "satisfaction": str(random.randint(1, 5)),
        "engagement": str(random.randint(1, 5)),
        "chatbot_type": chatbot_type
    }

# Driver script to simulate user input and generate chatbots' responses
def generate_responses_and_feedback():
    feedback_data = []


    # for song in songs:
    for i in range(12):  # Simulate 12 user inputs
        print('User: ',i)
        user_input = simulate_user_input()
        print('Input: ', user_input)
        song = songs[random.randint(0, 11)]
        # Generate responses
        fuzzy_response = generate_fuzzy_response(song, user_input)
        llm_response = generate_llm_response(song, user_input)

        # Simulate feedback for both chatbots
        feedback_data.append({
            "user_input": user_input,
            "fuzzy_logic_response": fuzzy_response,
            "llm_response": llm_response,
            "fuzzy_logic_feedback": simulate_feedback("fuzzy_logic"),
            "llm_feedback": simulate_feedback("generative_ai")
        })
        print({
            "user_input": user_input,
            "fuzzy_logic_response": fuzzy_response,
            "llm_response": llm_response,
            "fuzzy_logic_feedback": simulate_feedback("fuzzy_logic"),
            "llm_feedback": simulate_feedback("generative_ai")
        })
        print('finished processing user: ', i)
    
    # Save feedback data to a JSON file
    output_filename = f"feedback_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w") as f:
        json.dump(feedback_data, f, indent=4)

    print(f"Feedback data saved to {output_filename}")

if __name__ == "__main__":
    generate_responses_and_feedback()
