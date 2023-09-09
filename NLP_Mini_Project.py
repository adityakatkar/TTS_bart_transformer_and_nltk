# %%
from playsound import playsound
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import nltk.data



dic=('हिंदी','hi','ગુજરાતી','gu','मराठी','mr','इंग्लिश','en')

x=int(input("Type 1 for English,2 for Hindi,3 for gujarati,4 for punjabi,5 for marathi: "))
if(x==1):
    language_code='en'
elif x==2:
    language_code='hi-IN'
elif x==3:
    language_code='gu'
elif x==4:
    language_code='pa-guru-IN'
else:
    language_code='mr-IN'


def takecommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source,phrase_time_limit=60)

    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language=language_code)
        print(f"user said {query}\n")
    except Exception as e:
        print("say that again please.....")
        return "None"
    return query

def takecommand_convert():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        r.pause_threshold = 1
        audio = r.listen(source,phrase_time_limit=5)

    try:
        print("Recognizing.....")
        query = r.recognize_google(audio, language=language_code)
        print(f"user said {query}\n")
    except Exception as e:
        print("say that again please.....")
        return "None"
    return query


query = takecommand()
while (query == "None"):
    query = takecommand()


def destination_language():
    print("Enter the language in which you want to convert \
    : Ex. Hindi , English , etc.")
    print()

    # Input destination language in which the user
    # wants to translate
    to_lang = takecommand_convert()
    while (to_lang == "None"):
        to_lang = takecommand_convert()
    to_lang = to_lang.lower()
    return to_lang


to_lang = destination_language()

# Mapping it with the code
while (to_lang not in dic):
    print("Language in which you are trying to convert\
    is currently not available ,please input some other language")
    print()
    to_lang = destination_language()

to_lang = dic[dic.index(to_lang) + 1]
translator = Translator()


translation = translator.translate(query,dest='en')
text = translation.text
speech_translation=text
speak = gTTS(text=text, lang=to_lang, slow=False)

# Using save() method to save the translated
# speech in capture_voice.mp3
speak.save("captured_voice.mp3")

text1=speak.text

path = os.path.expanduser('~/natural_language_toolkit_data')
print(path)
if not os.path.exists(path):
   os.mkdir(path)

path in nltk.data.path


with open(os.path.expanduser('~/natural_language_toolkit_data/wordfile.txt'),'w',encoding = 'utf-8') as f:
   f.write(text1)


t=os.path.expanduser('file://C://Users//Katka//natural_language_toolkit_data//wordfile.txt')

lm=nltk.data.load(t, format="raw")
print(lm)
# Using OS module to run the translated voice.
playsound('captured_voice.mp3')
os.remove('captured_voice.mp3')
print(text)

# %%
subtitle=text

# %% [markdown]
# Using BART for Summarization

# %%
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

# %%
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# %%
input_tensor = tokenizer.encode(subtitle, return_tensors="pt", max_length=512)

# %%
outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)

# %%
summary=tokenizer.decode(outputs_tensor[0])

# %%
import nltk
import string
from heapq import nlargest

# %%
#text="Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning. In reinforcement learning methods, expectations are approximated by averaging over samples and using function approximation techniques to cope with the need to represent value functions over large state-action spaces. Policy iteration consists of two steps: policy evaluation and policy improvement. The work on learning ATARI games by Google DeepMind increased attention to deep reinforcement learning or end-to-end reinforcement learning. Assuming full knowledge of the MDP, the two basic approaches to compute the optimal action-value function are value iteration and policy iteration. Two elements make reinforcement learning powerful: the use of samples to optimize performance and the use of function approximation to deal with large environments. Many policy search methods may get stuck in local optima (as they are based on local search)"

# %%
# If the length of the text is greater than 20, take a 10th of the sentences
if text.count(". ") > 20:
  length = int(round(text.count(". ")/10, 0))
# Otherwise return five sentences
else:
  length = 2

# %%
# Remove punctuation
nopunc = [char for char in text if char not in string.punctuation]
nopunc = ''.join(nopunc)
# Remove stopwords
processed_text =[word for word in nopunc.split() if word.lower() not in nltk.corpus.stopwords.words('english')]

# %%
# Create a dictionary to store word frequency
word_freq = {}
# Enter each word and its number of occurrences
for word in processed_text:
  if word not in word_freq:
    word_freq[word] = 1
  else:
    word_freq[word] = word_freq[word] + 1

# %%
# Divide all frequencies by max frequency to give store of (0, 1]
max_freq = max(word_freq.values())
for word in word_freq.keys():
  word_freq[word] = (word_freq[word]/max_freq)

# %%
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

# %%
summary_sents = nlargest(length, sent_score, key = sent_score.get)
summary = ' '.join(summary_sents)
print(summary)

# %%
dic=('हिंदी','hi-IN','ગુજરાતી','gu-IN','मराठी','mr')

# def destination_language():
#     print("Enter the language in which you want to convert \
#     : Ex. Hindi ,Gujarati,Marathi etc.")
#     print()

#     # Input destination language in which the user
#     # wants to translate
#     to_lang = takecommand()
#     while (to_lang == "None"):
#         to_lang = takecommand()
#     to_lang = to_lang.lower()
#     return to_lang


# to_lang = destination_language()
print("Enter the language in which you want to convert \
#     : Ex. Hindi ,Gujarati,Marathi etc.")
print()

x=int(input("Type 1 for Hindi,2 for gujarati,3 for punjabi,4 for marathi: "))
if(x==1):
    language_code='hi-IN'
elif x==2:
    language_code='gu'
elif x==3:
    language_code='pa-guru'
elif x==4:
    language_code='mr'

translator = Translator()


translation = translator.translate(summary,dest=language_code)
text = translation.text
print(text)
speak = gTTS(text=text, lang=to_lang, slow=False)

# Using save() method to save the translated
# speech in capture_voice.mp3
speak.save("captured_voice.mp3")

text1=speak.text
playsound('captured_voice.mp3')
os.remove('captured_voice.mp3')


