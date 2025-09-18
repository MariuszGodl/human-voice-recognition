Human_voice_processing on  main [!] via  v3.12.3 (venv) 
❯ find data/processed/words/ -type d | wc -l
7728

Human_voice_processing on  main [!] via  v3.12.3 (venv) took 23m45s 
❯ find data/processed/words/ -type f | wc -l
217941

Human_voice_processing on  main [!] via  v3.12.3 (venv) 
❯ tree -L 3
.
├── data
├── dataprocessing
│   ├── api_to_chtp.py
│   ├── augmentation.py
│   ├── const.py
│   ├── dataprocessing.py
│   ├── get_global_parameters.py
│   ├── helper_funct.py
│   ├── iterate_dataset.py
│   ├── RemovePolichChars.py
│   └── words.json
├── model
│   ├── model.py
│   ├── models
│   │   └── v1_30.pth
│   ├── speech_dataset.py
│   └── trenning.py
├── readme.txt
├── temp.py
└── venv