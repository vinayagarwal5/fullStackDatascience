import base64
import os

import streamlit as st
from gtts import gTTS
from mtranslate import translate

LANGUAGE_OPTIONS = [
    ("Afrikaans", "af"),
    ("Arabic", "ar"),
    ("Bengali", "bn"),
    ("Bulgarian", "bg"),
    ("Catalan", "ca"),
    ("Chinese", "zh-CN"),
    ("Czech", "cs"),
    ("Danish", "da"),
    ("Dutch", "nl"),
    ("English", "en"),
    ("Esperanto", "eo"),
    ("Finnish", "fi"),
    ("French", "fr"),
    ("German", "de"),
    ("Greek", "el"),
    ("Gujarati", "gu"),
    ("Hindi", "hi"),
    ("Hungarian", "hu"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Kannada", "kn"),
    ("Korean", "ko"),
    ("Malayalam", "ml"),
    ("Marathi", "mr"),
    ("Nepali", "ne"),
    ("Norwegian", "no"),
    ("Odia", "or"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Romanian", "ro"),
    ("Russian", "ru"),
    ("Sinhala", "si"),
    ("Spanish", "es"),
    ("Swedish", "sv"),
    ("Tamil", "ta"),
    ("Telugu", "te"),
    ("Thai", "th"),
    ("Turkish", "tr"),
    ("Ukrainian", "uk"),
    ("Urdu", "ur"),
    ("Vietnamese", "vi"),
]

langlist = tuple(name for name, _ in LANGUAGE_OPTIONS)
lang_array = {name: code for name, code in LANGUAGE_OPTIONS}

# layout
st.title("Language-Translation")
inputtext = st.text_area("Hi Please Enter text here to Translate", height=100)

choice = st.sidebar.radio("SELECT LANGUAGE", langlist)

speech_langs = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "od": "odia",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-CN": "Chinese",
    "ow": "Odia",
}


# function to decode audio file for download
def get_binary_file_downloader_html(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


c1, c2 = st.columns([4, 3])

# I/O
if len(inputtext) > 0:
    try:
        output = translate(inputtext, lang_array[choice])
        with c1:
            st.text_area("TRANSLATED TEXT", output, height=200)
        # if speech support is available will render autio file
        if choice in speech_langs.values():
            with c2:
                aud_file = gTTS(text=output, lang=lang_array[choice], slow=False)
                aud_file.save("lang.mp3")
                with open("lang.mp3", "rb") as audio_file_read:
                    audio_bytes = audio_file_read.read()
                st.audio(audio_bytes, format="audio/mp3")
                st.markdown(
                    get_binary_file_downloader_html("lang.mp3", "Audio File"),
                    unsafe_allow_html=True,
                )
    except Exception as e:
        st.error(e)
