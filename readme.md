# Polish Speech Recognition System  

## Overview  
This project is a **Polish automatic speech recognition (ASR) system**, designed to accurately transcribe spoken Polish into text. It was developed as part of my academic and research work, with a strong focus on combining **signal processing techniques** and **deep learning models**.  

The project demonstrates practical knowledge in:  
- **Audio preprocessing** (MFCC extraction, sentence-level features, spectrogram analysis)  
- **Machine learning & neural networks** (TensorFlow-based architectures)  
- **Natural language processing (NLP)** for handling text-based features  
- **Iterative model development** across multiple experimental versions  

## Key Features  

### Feature Engineering  
- Mel-Frequency Cepstral Coefficients (MFCCs) for capturing phonetic characteristics  
- Sentence-level statistical features for context-aware recognition  
- Text-based feature integration for improved accuracy  

### Model Development  
- Multiple model iterations, evolving from baseline classifiers to trainable deep neural networks  
- Optimization of architectures, hyperparameters, and training pipelines  
- Comparative evaluation across model versions  

### Polish Language Focus  
- Special attention to Polish phonetics, diacritics, and morphology  
- Preprocessing pipeline tailored to Polish speech data  

### Data Augmentation  
To improve model robustness and generalization, several augmentation techniques were applied to the speech dataset, including:  
- **Time shifting** (small delays or advances in audio)  
- **Pitch shifting** (slight tone variations to simulate different speakers)  
- **Speed perturbation** (faster/slower playback)  
- **Background noise injection** (simulating real-world conditions such as street or office noise)  

These augmentations significantly increased dataset diversity without requiring additional labeled recordings.  

## Technical Stack  
- **Languages**: Python  
- **Libraries**: TensorFlow, NumPy, Pandas, Scikit-learn, Librosa  
- **Workflow**: Jupyter notebooks + modular Python scripts  
- **Version Control**: Git  

## Results  
- Improved recognition accuracy with each model iteration  
- Demonstrated scalability from feature-based models to trainable neural networks  
- Established a reproducible pipeline for further experimentation  

## Applications  
- Polish speech-to-text systems  
- Assistive technologies for accessibility  
- Customer service automation (call centers, chatbots)  
- Academic research in low-resource language processing  

## Next Steps  
- Expand dataset for greater generalization  
- Explore end-to-end ASR with transformer architectures (e.g., Wav2Vec2, Whisper)  
- Integrate language modeling for enhanced transcription quality  
