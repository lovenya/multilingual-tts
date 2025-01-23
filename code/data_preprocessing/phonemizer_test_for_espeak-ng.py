import os
from phonemizer.phonemize import phonemize

# Add eSpeak-NG path dynamically
os.environ["PATH"] += r";C:\Program Files\eSpeak NG"  # Adjust if necessary
os.environ["ESPEAK_DATA_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng-data"
from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library("C:\Program Files\eSpeak NG\libespeak-ng.dll")


def test_espeak_ng():
    """
    Test if eSpeak-NG is installed correctly and working with Phonemizer.
    """
    try:
        # Test text
        text = "Hello, world!"
        language = "en-us"  # English US

        # Phonemize the text
        phonemes = phonemize(
            text,
            language=language,
            backend="espeak",
            strip=True,  # Remove extra spaces
            preserve_punctuation=True,  # Keep punctuation
            with_stress=True,  # Add stress markers
        )

        print("Phonemes generated successfully!")
        print("Input Text:", text)
        print("Phonemes:", phonemes)

    except Exception as e:
        print("Error: eSpeak-NG is not installed or not configured correctly.")
        print("Details:", e)


if __name__ == "__main__":
    test_espeak_ng()
