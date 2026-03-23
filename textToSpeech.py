from gtts import gTTS
import os
def text_to_speech(text, language='en', filename='output', slow=False):
    myobj = gTTS(text=text, lang=language, slow=slow)
    myobj.save(filename+'.mp3')
    os.system(f"start {filename}.mp3")
dave = input("Enter the text: ")
bob = input("Enter the language code. ie 'en': ")
jhonathan = input("Give the file a name: ")
thompson = input("Slow speech? True/False: ")
text_to_speech(dave, bob, jhonathan, thompson)