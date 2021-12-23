import codecs

import pandas as pd
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems
import pickle
import os

def main():
    question = input("I am your google assistance ask me your question:")
    position = dict()
    df = pd.read_excel('IR1_7k_news.xlsx')
    x = df['content']
    title = df['title']
    my_normalizer = Normalizer(statistical_space_correction=True)
    my_tokenizer = Tokenizer()
    my_stemmer = FindStems()
    sw = codecs.open('Stop_words.txt', encoding='utf-8').read().split('\r\n')
    stop_words = dict()
    for i in sw:
        stop_words[i] = 1
    # stemmed = []
    # if os.path.isfile('position.pickle'):
    #     with open('position.pickle', 'rb') as handle:
    #         position = pickle.load(handle)
    # else:
    for i in range(len(x)):
        words = my_tokenizer.tokenize_words(my_normalizer.normalize(x[i]))
        # tmp = []
        for j in range(len(words)):
            if words[j] not in stop_words:
                b = my_stemmer.convert_to_stem(words[j])
                # tmp.append(b)
                if b not in position:
                    position[b] = dict()
                if i not in position[b]:
                    position.get(b)[i] = list()
                position.get(b).get(i).append(j)
            # stemmed.append(tmp)
        # with open('position.pickle', 'wb') as handle:
        #     pickle.dump(position, handle, protocol=pickle.HIGHEST_PROTOCOL)
    question_words = my_tokenizer.tokenize_words(my_normalizer.normalize(question))
    clean_q_words = []
    q_w_locations = []
    for i in range(len(question_words)):
        if question_words[i] not in stop_words:
            b = my_stemmer.convert_to_stem(question_words[i])
            if b not in position:
                print("No match found.")
                return
            q_w_locations.append(position[b])
            clean_q_words.append(b)


if __name__ == '__main__':
    main()
