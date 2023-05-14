from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import numpy as np

app = Flask(__name__)

#Get the dictionary
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# Load your model
model = load_model('rnn_imdb.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']

        tokens = text.split()  # simple whitespace tokenization
        sequences = [[word_index.get(word, 0) for word in tokens]]

        # Pad the sequences
        data = pad_sequences(sequences, maxlen=500)  # same maxlen as your model expects

        # Make prediction test. Uncomment if you watn to directly get the data 
        #data = [1, 4, 2, 2, 33, 2804, 4, 2040, 432, 111, 153, 103, 4, 1494, 13, 70, 131, 67, 11, 61, 2, 744, 35, 3715, 761, 61, 5766, 452, 9214, 4, 985, 7, 2, 59, 166, 4, 105, 216, 1239, 41, 1797, 9, 15, 7, 35, 744, 2413, 31, 8, 4, 687, 23, 4, 2, 7339, 6, 3693, 42, 38, 39, 121, 59, 456, 10, 10, 7, 265, 12, 575, 111, 153, 159, 59, 16, 1447, 21, 25, 586, 482, 39, 4, 96, 59, 716, 12, 4, 172, 65, 9, 579, 11, 6004, 4, 1615, 5, 2, 7, 5168, 17, 13, 7064, 12, 19, 6, 464, 31, 314, 11, 2, 6, 719, 605, 11, 8, 202, 27, 310, 4, 3772, 3501, 8, 2722, 58, 10, 10, 537, 2116, 180, 40, 14, 413, 173, 7, 263, 112, 37, 152, 377, 4, 537, 263, 846, 579, 178, 54, 75, 71, 476, 36, 413, 263, 2504, 182, 5, 17, 75, 2306, 922, 36, 279, 131, 2895, 17, 2867, 42, 17, 35, 921, 2, 192, 5, 1219, 3890, 19, 2, 217, 4122, 1710, 537, 2, 1236, 5, 736, 10, 10, 61, 403, 9, 2, 40, 61, 4494, 5, 27, 4494, 159, 90, 263, 2311, 4319, 309, 8, 178, 5, 82, 4319, 4, 65, 15, 9225, 145, 143, 5122, 12, 7039, 537, 746, 537, 537, 15, 7979, 4, 2, 594, 7, 5168, 94, 9096, 3987, 2, 11, 2, 4, 538, 7, 1795, 246, 2, 9, 2, 11, 635, 14, 9, 51, 408, 12, 94, 318, 1382, 12, 47, 6, 2683, 936, 5, 6307, 2, 19, 49, 7, 4, 1885, 2, 1118, 25, 80, 126, 842, 10, 10, 2, 2, 4726, 27, 4494, 11, 1550, 3633, 159, 27, 341, 29, 2733, 19, 4185, 173, 7, 90, 2, 8, 30, 11, 4, 1784, 86, 1117, 8, 3261, 46, 11, 2, 21, 29, 9, 2841, 23, 4, 1010, 2, 793, 6, 2, 1386, 1830, 10, 10, 246, 50, 9, 6, 2750, 1944, 746, 90, 29, 2, 8, 124, 4, 882, 4, 882, 496, 27, 2, 2213, 537, 121, 127, 1219, 130, 5, 29, 494, 8, 124, 4, 882, 496, 4, 341, 7, 27, 846, 10, 10, 29, 9, 1906, 8, 97, 6, 236, 2, 1311, 8, 4, 2, 7, 31, 7, 2, 91, 2, 3987, 70, 4, 882, 30, 579, 42, 9, 12, 32, 11, 537, 10, 10, 11, 14, 65, 44, 537, 75, 2, 1775, 3353, 2, 1846, 4, 2, 7, 154, 5, 4, 518, 53, 2, 2, 7, 3211, 882, 11, 399, 38, 75, 257, 3807, 19, 2, 17, 29, 456, 4, 65, 7, 27, 205, 113, 10, 10, 2, 4, 2, 2, 9, 242, 4, 91, 1202, 2, 5, 2070, 307, 22, 7, 5168, 126, 93, 40, 2, 13, 188, 1076, 3222, 19, 4, 2, 7, 2348, 537, 23, 53, 537, 21, 82, 40, 2, 13, 2, 14, 280, 13, 219, 4, 2, 431, 758, 859, 4, 953, 1052, 2, 7, 5991, 5, 94, 40, 25, 238, 60, 2, 4, 2, 804, 2, 7, 4, 9941, 132, 8, 67, 6, 22, 15, 9, 283, 8, 5168, 14, 31, 9, 242, 955, 48, 25, 279, 2, 23, 12, 1685, 195, 25, 238, 60, 796, 2, 4, 671, 7, 2804, 5, 4, 559, 154, 888, 7, 726, 50, 26, 49, 7008, 15, 566, 30, 579, 21, 64, 2574]
        #data = np.reshape(data, (1, -1))

       
        prediction = model.predict(data)

        if prediction < 0.5:
            sentiment = 'Negative'
        else:
            sentiment = 'Positive'

        print(prediction)
        return render_template('index.html', sentiment=sentiment)
    
    print("Something crazy")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(port=8081)