

from flask import Flask, request, render_template, redirect, url_for,session,send_file
from playsound import playsound
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import re
import os
from heapq import nlargest
import string
import nltk
import nltk.data
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

dic = {
    '1': 'en',
    '2': 'hi-IN',
    '3': 'gu',
    '4': 'pa-guru-IN',
    '5': 'mr-IN'
}

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def speech_to_text(language_code):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source, phrase_time_limit=60)

    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language=language_code)
        print(f"user said {query}\n")
    except Exception as e:
        print("say that again please.....")
        return "None"
    return query

def text_to_text(text):
    translator = Translator()
    translated_text = translator.translate(text, dest='en')
    t = translated_text.text
    return t

def text_to_text1(text, to_lang):
    translator = Translator()
    translated_text = translator.translate(text, dest=to_lang)
    t1 = translated_text.text
    return t1

def text_to_speech(text, to_lang):
    speak = gTTS(text=text, lang=to_lang, slow=False)
    speak.save("captured_voice.mp3")
    playsound('captured_voice.mp3')
    file_path="E://nlp mini project//captured_voice.mp3"
    return file_path




def summarize_text(text):
    input_tensor = tokenizer.encode(text, return_tensors="pt", max_length=512)
    outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs_tensor[0])
    return summary


def summarize_text_nltk(text):
    # If the length of the text is greater than 20, take a 10th of the sentences
    if text.count(". ") > 20:
        length = int(round(text.count(". ") / 10, 0))
    # Otherwise return five sentences
    else:
        length = 2

    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    # Remove stopwords
    processed_text = [word for word in nopunc.split() if word.lower() not in nltk.corpus.stopwords.words('english')]

    # Create a dictionary to store word frequency
    word_freq = {}
    # Enter each word and its number of occurrences
    for word in processed_text:
        if word not in word_freq:
            word_freq[word] = 1
        else:
            word_freq[word] = word_freq[word] + 1

    # Divide all frequencies by max frequency to give store of (0, 1]
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_freq)

    # Create a list of the sentences in the text
    sent_list = nltk.sent_tokenize(text)
    # Create an empty dictionary to store sentence scores
    sent_score = {}
    for sent in sent_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_freq.keys():
                if sent not in sent_score.keys():
                    sent_score[sent] = word_freq[word]
                else:
                    sent_score[sent] = sent_score[sent] + word_freq[word]

    summary_sents = nlargest(length, sent_score, key=sent_score.get)
    summary = ' '.join(summary_sents)

    return summary


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        x = request.form.get('language',None)
        language_code = dic.get(x, 'en')
        query = speech_to_text(language_code)
        translated = text_to_text(query)
        session['trans11']=translated
        while query == "None":
            query = speech_to_text(language_code)
            translated = text_to_text(query)
            session['trans11']=translated

        return render_template('index.html', query=query, translated=translated)
    else:
        return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        summary = None
        summary1 = None
        translated2 = None
        translated4 = None
        speech1 = None
        speech2 = None
        t = session.get('trans11')
        lang = request.form.get('to_language',None)
        type = request.form.get('summarize_type',None)
        print('type', t)
        t1=request.form.get('text11',None)
        if type=='transformer':
            t2=text_to_text(t1)
            if (t == None):
                summary=None
            else:
                summary = summarize_text(t)
                print(summary)
            if (t2 == None):
                summary1=None
            else:
                summary1=summarize_text(t2)

            if summary!='' and summary!=None:
                translated1=text_to_text1(summary,lang)

                if translated1 == None:
                    translated2=None
                else:
                    translated2 = re.sub('<[^<]+?>', '', translated1)

                speech1=text_to_speech(translated2, lang)
                os.remove('captured_voice.mp3')
                if 'trans11' in session:
                    session.pop('trans11',None)

            if summary1!='' and summary1!=None:
                translated3 = text_to_text1(summary1, lang)
                if translated3 == None:
                    translated4=None
                else:
                    translated4 = re.sub('<[^<]+?>', '', translated3)
                speech2=text_to_speech(translated4,lang)
                os.remove('captured_voice.mp3')

        elif type=='nltk':
            t2 = text_to_text(t1)
            if (t == None):
                summary = None
            else:
                summary = summarize_text_nltk(t)
                print(summary)
            if (t2 == None):
                summary1 = None
            else:
                summary1 = summarize_text_nltk(t2)

            if summary != '' and summary != None:
                translated1 = text_to_text1(summary, lang)

                if translated1 == None:
                    translated2 = None
                else:
                    translated2 = re.sub('<[^<]+?>', '', translated1)

                speech1 = text_to_speech(translated2, lang)
                os.remove('captured_voice.mp3')
                if 'trans11' in session:
                    session.pop('trans11', None)

            if summary1 != '' and summary1 != None:
                translated3 = text_to_text1(summary1, lang)
                if translated3 == None:
                    translated4 = None
                else:
                    translated4 = re.sub('<[^<]+?>', '', translated3)
                speech2 = text_to_speech(translated4, lang)
                os.remove('captured_voice.mp3')


        return render_template('index.html', t=t, translated2=translated2,speech1=speech1,translated4=translated4,speech2=speech2)
    else:
        return render_template('index.html')

# @app.route('/audio')
# def audio():
#     # Replace 'path/to/your/audio/file.mp3' with the actual path to your audio file
#     return send_file("E://nlp mini project//captured_voice.mp3", mimetype='audio/*')

if __name__ == '__main__':
    app.run(debug=True)