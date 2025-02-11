first run untar and unzip from util/ if you've installed dataset via the wget method (terminal/CLI)
then util/removing_uneccessary_nesting to fix the folder structure of the dataset
then we move to data_preprocessing
    - file_nomenclature.py
    - normalize_audio_sampling.py
    - normalize_transcript.py
    - metadata_generation.py
    - phoneme_generation.py


now from the util folder
    - append_suffix_to_english_phoneme_sequences.py
    - compute_phoneme_vocab.py

now we'll hafve to install nemo_toolkit
but you'll face a lot of problems

start with cuda 11.8
python 3.10.x

run pip install nemo_toolkit[all]
you'll receive some errors
if it is related to a dependency, irrespective of which one, go to the next step
and then run the following three commands in the same order:

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install nemo_toolkit[all]W

now there shoudn't be any problems
now there shoudn't be any problems
if there still arises any problems, please open an issue or mail me at lovenya2002@gmail.com


then we run
    - generate_phoneme_inventory.py

then we go to util
we have appending suffices functions to phoneme tokens, if it is needed

after all this we go to data_preprocessing again and run
    - normalize_and_clean_phoneme_sequences.py
    - metadata_integration_with_phonemes.py



then we run
    - generate_phoneme_inventory.py

then we go to util
we have appending suffices functions to phoneme tokens, if it is needed

after all this we go to data_preprocessing again and run
    - normalize_and_clean_phoneme_sequences.py
    - metadata_integration_with_phonemes.py


then we move to feature extraction
    - pitch_extraction.py
    - energy_extraction.py

then we move to data_preprocessing
    - metadata_integration_with_energies.py
    - metadata_integration_with_pitches.py

then change env for espnet2 installation
then do pip install espnet2
then run 
    - fastspeech2_train.py