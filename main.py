from flask import Flask, request, send_file
from bark.generation import generate_text_semantic, preload_models
from bark.api import semantic_to_waveform
from bark import generate_audio, SAMPLE_RATE
import nltk
import numpy as np
import os
import random
import string

app = Flask(__name__)

preload_models()
nltk.download('punkt')

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/convert', methods=['POST'])
def convert():
    text = request.form['text']
    sentences = nltk.sent_tokenize(text)
    SPEAKER = "v2/en_speaker_6"
    silence = np.zeros(int(0.25 * SAMPLE_RATE))
    pieces = []
    for sentence in sentences:
        audio_array = generate_audio(sentence, history_prompt=SPEAKER)
        pieces += [audio_array, silence.copy()]
    audio = np.concatenate(pieces)
    filename = ''.join(random.choices(string.ascii_letters + string.digits, k=17)) + '.wav'
    with open(filename, 'wb') as f:
        f.write(semantic_to_waveform(audio))
    return send_file(filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)

