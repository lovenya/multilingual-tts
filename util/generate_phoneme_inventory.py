def get_fixed_inventory():
    """
    Returns a fixed, unified phoneme inventory for the four languages,
    with unified prefixes:
      - English: (en-us)
      - Bhojpuri: (hi)
      - Gujarati: (gu)
      - Kannada: (kn)
    
    Note: This is an example inventory. You may refine it based on your needs.
    """
    inventory = [
        # English (en-us)
        "(en-us) p", "(en-us) b", "(en-us) t", "(en-us) d", "(en-us) k", "(en-us) g",
        "(en-us) f", "(en-us) v", "(en-us) θ", "(en-us) ð", "(en-us) s", "(en-us) z",
        "(en-us) ʃ", "(en-us) ʒ", "(en-us) h", "(en-us) tʃ", "(en-us) dʒ", "(en-us) m",
        "(en-us) n", "(en-us) ŋ", "(en-us) l", "(en-us) r", "(en-us) j", "(en-us) w",
        "(en-us) i", "(en-us) ɪ", "(en-us) e", "(en-us) ɛ", "(en-us) æ", "(en-us) ʌ",
        "(en-us) ɑ", "(en-us) ɒ", "(en-us) ɔ", "(en-us) o", "(en-us) ʊ", "(en-us) u",
        "(en-us) aɪ", "(en-us) aʊ", "(en-us) ɔɪ", "(en-us) eɪ", "(en-us) oʊ",
        
        # Bhojpuri (hi) – using a Hindi-like inventory
        "(hi) p", "(hi) b", "(hi) t̪", "(hi) d̪", "(hi) ʈ", "(hi) ɖ", "(hi) k", "(hi) g",
        "(hi) tʃ", "(hi) dʒ", "(hi) f", "(hi) s", "(hi) h", "(hi) m", "(hi) n",
        "(hi) ɳ", "(hi) n̪", "(hi) l", "(hi) r", "(hi) j",
        "(hi) ə", "(hi) a", "(hi) ɪ", "(hi) i", "(hi) ʊ", "(hi) u",
        "(hi) e", "(hi) o", "(hi) ɛ", "(hi) ɔ", "(hi) ɒ",
        
        # Gujarati (gu) – similar to Hindi/Bhojpuri
        "(gu) p", "(gu) b", "(gu) t̪", "(gu) d̪", "(gu) ʈ", "(gu) ɖ", "(gu) k", "(gu) g",
        "(gu) tʃ", "(gu) dʒ", "(gu) f", "(gu) s", "(gu) h", "(gu) m", "(gu) n",
        "(gu) ɳ", "(gu) n̪", "(gu) l", "(gu) r", "(gu) j",
        "(gu) ə", "(gu) a", "(gu) ɪ", "(gu) i", "(gu) ʊ", "(gu) u",
        "(gu) e", "(gu) o", "(gu) ɛ", "(gu) ɔ", "(gu) ɒ",
        
        # Kannada (kn) – similar to the above but may have slight differences
        "(kn) p", "(kn) b", "(kn) t", "(kn) d", "(kn) ʈ", "(kn) ɖ", "(kn) k", "(kn) g",
        "(kn) tʃ", "(kn) dʒ", "(kn) f", "(kn) s", "(kn) h", "(kn) m", "(kn) n",
        "(kn) ɳ", "(kn) n̪", "(kn) l", "(kn) r", "(kn) j",
        "(kn) ə", "(kn) a", "(kn) ɪ", "(kn) i", "(kn) ʊ", "(kn) u",
        "(kn) e", "(kn) o", "(kn) ɛ", "(kn) ɔ", "(kn) ɒ",
    ]
    return inventory

if __name__ == "__main__":
    fixed_inventory = get_fixed_inventory()
    output_file = "fixed_inventory.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for token in fixed_inventory:
            f.write(token + "\n")
    print(f"Fixed inventory saved to {output_file}")
    print(f"Total phonemes in inventory: {len(fixed_inventory)}")

# The phoneme inventory provided is based on standard phoneme sets for each language (English, Bhojpuri, Gujarati, Kannada), 
# which are commonly used in phonetic transcription systems like the International Phonetic Alphabet (IPA). 
# These phonemes represent common consonant and vowel sounds used in these languages.
# The phonemes were chosen based on standard phonetic inventories for each language, derived from:

# English (en-us): General American English phoneme set from resources like CMU Pronouncing Dictionary and espeak-ng.
# Bhojpuri (hi): Phonemes similar to Hindi, as Bhojpuri shares much of its phonetic inventory with Hindi, based on linguistic studies and espeak-ng.
# Gujarati (gu): Based on Gujarati phonetic resources and espeak-ng.
# Kannada (kn): Derived from Kannada linguistic studies and espeak-ng for TTS applications.
# These phonemes are commonly used in TTS systems for each language.