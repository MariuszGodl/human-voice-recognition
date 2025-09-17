Human_voice_processing on  main [✘!+⇡] via  v3.12.3 (venv) 
❯ find data/processed/words/ -type d | wc -l
7902

Human_voice_processing on  main [✘!+⇡] via  v3.12.3 (venv) 
❯ find data/processed/words/ -type f | wc -l
88305

Human_voice_processing on  main [✘!+⇡] via  v3.12.3 (venv) 
❯ tree -L 3
.
├── api_to_chtp.py
├── const.py
├── data
│   ├── processed
│   │   ├── labels.csv
│   │   ├── mfcc_norm_stats.npz
│   │   └── words
│   └── raw
│       ├── 1001-1500
│       ├── 1-500
│       ├── 1501-2000
│       ├── 2001-2500
│       ├── 2501-3000
│       └── 501-1000
├── dataprocessing.py
├── get_global_parameters.py
├── helper_funct.py
├── iterate_dataset.py
├── __pycache__
├── readme.txt
├── RemovePolichChars.py
└── venv

