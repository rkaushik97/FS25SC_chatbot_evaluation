import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import json
import os
from flask import Flask, render_template, request, jsonify
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Flask app
app = Flask(__name__)

# OpenAI API Key (replace with your actual key)
openai.api_key = ""

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

# Load the model and tokenizer
#model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Replace with your preferred model
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#model.to(device)

# Function to generate ChatGPT response
def get_chatgpt_response(song, mood_description):
    prompt = (
        f"Analyze the song '{song['title']}' by {song['artist']} based on its mood ('{mood_description}'), "
        f"instrumentation, energy, tempo, and lyrical sentiment. Provide a detailed, engaging breakdown."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a music critic."},
                {"role": "user", "content": prompt}
            ]
        )
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['message']['content']
        else:
            return "Sorry, I couldn't generate a response at the moment."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to generate local LLM response
def get_local_llm_response(song, mood_description, energy_input, tempo_input, vocals_input, instrumentation_input, melodic_input, lyrics_sentiment_input, mood_value):
    prompt = (
        f"Analyze the following song based on its attributes and provide a concise, engaging review (50-60 words). Follow this thought process before forming the final verdict:\n"
        f"Step 1: Evaluate the song's core elements:\n"
        f"- Energy (0-100): {energy_input} → What does this energy level suggest? Is it high-energy, mellow, or balanced?\n"
        f"- Tempo (BPM): {tempo_input} → Is the pace fast, moderate, or slow? How does it contribute to the mood?\n"
        f"- Vocals (0-10): {vocals_input} → Are they powerful, subtle, or somewhere in between?\n"
        f"- Instrumentation (0-10): {instrumentation_input} → Is the instrumental section rich, minimal, or balanced?\n"
        f"- Melodic Complexity (0-10): {melodic_input} → Does the melody feel intricate, repetitive, or smooth?\n"
        f"- Lyrical Sentiment (-1 to 1): {lyrics_sentiment_input} → Is the song emotionally uplifting (+1), neutral (0), or melancholic (-1)?\n"
        f"- Overall Mood Score (0-100): {mood_value} → How do all elements combine to define the song’s mood?\n\n"
        f"Step 2: Final Verdict:\n"
        f"Summarize the analysis into a 50-60 word review that highlights the song’s energy, emotion, and sound. "
        f"Make the tone concise, engaging, and fitting for a quick review."
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=300,
            temperature=1.0,
            top_p=0.7,
            repetition_penalty=1.5,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
    except Exception as e:
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


    #chatgpt_response = get_local_llm_response(
    #    selected_song, mood_description, energy_input, tempo_input, vocals_input, instrumentation_input, melodic_input, lyrics_sentiment_input, mood_value
    #)
    chatgpt_response = "Blinding Lights by The Weeknd is a high-energy synth-driven track with a pulsating 80 BPM rhythm. The strong vocals (8/10) and retro instrumentation (8/10) create a nostalgic yet exhilarating feel. With passionate lyrics (+1 sentiment), this song perfectly matches a lively, adventurous mood (60/100)—great for night drives and dance floors. 🎶✨"
    return render_template('index.html', response_fuzzy=fuzzy_response, response_gpt=chatgpt_response, songs=songs, selected_song=selected_song)

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

if __name__ == '__main__':
    app.run(debug=True)