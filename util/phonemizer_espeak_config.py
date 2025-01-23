# global config file for espeak backend for phonemizer
# add import phonemizer_config at the start of every script that uses phonemizer of course


import os
from phonemizer.backend.espeak.wrapper import EspeakWrapper


os.environ["PATH"] += r";C:\Program Files\eSpeak NG"
os.environ["ESPEAK_DATA_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng-data"

# Set up eSpeak-NG globally
EspeakWrapper.set_library("C:\\Program Files\\eSpeak NG\\libespeak-ng.dll")
