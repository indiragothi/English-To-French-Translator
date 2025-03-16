# English-to-French Language Translation using Seq2Seq Model

## Overview
This project demonstrates the use of a Sequence-to-Sequence (Seq2Seq) model for English-to-French translation. By leveraging deep learning techniques, particularly Neural Machine Translation (NMT), the model efficiently translates English sentences into French. The model is trained end-to-end, ensuring improved fluency and accuracy compared to traditional statistical machine translation methods.

## Dataset Used
The dataset used for training consists of English sentences and their corresponding French translations.

**Dataset Link:** [Kaggle - English-French Translation Dataset](https://www.kaggle.com/datasets/devicharith/language-translation-englishfrench)

## Python Libraries Used
- Keras
- TensorFlow
- Scikit-learn
- NumPy
- Pandas
- Seaborn
- Matplotlib

## Model Architecture
The Seq2Seq model consists of an encoder-decoder structure, where:
- The **encoder** processes the input English sentence and converts it into a context vector.
- The **decoder** uses this context vector to generate the translated French sentence.

## Implementation Steps
1. Data Preprocessing
2. Model Building (Encoder-Decoder Architecture)
3. Training the Model
4. Evaluating and Testing the Model
5. Translating Sample Sentences

## Usage
Run the provided Python script to train and evaluate the translation model. Modify the input sentences to test custom translations.

## Results
The trained Seq2Seq model provides accurate English-to-French translations with improved fluency and coherence compared to traditional translation techniques.

## References
- [Sequence-to-Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

