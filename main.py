import codecs

import pandas as pd
from parsivar import Normalizer
from parsivar import Tokenizer
from parsivar import FindStems
import math
import pickle
import os


def main():
    question = input("I am your google assistance ask me your question:")
    Repetition = dict()
    df = pd.read_excel('IR1_7k_news.xlsx')
    x = df['content']
    title = df['title']
    my_normalizer = Normalizer(statistical_space_correction=True)
    my_tokenizer = Tokenizer()
    my_stemmer = FindStems()
    sw = codecs.open('Stop_words.txt', encoding='utf-8').read().split('\r\n')
    stop_words = dict()
    N = len(x)
    for i in sw:
        stop_words[i] = 1
    stemmed = []
    # if os.path.isfile('Repetition.pickle'):
    #     with open('Repetition.pickle', 'rb') as handle:
    #         Repetition = pickle.load(handle)
    # else:
    for i in range(N):
        words = my_tokenizer.tokenize_words(my_normalizer.normalize(x[i]))
        # tmp = []
        for j in range(len(words)):
            if words[j] not in stop_words:
                b = my_stemmer.convert_to_stem(words[j])
                # tmp.append(b)
                if b not in Repetition:
                    Repetition[b] = dict()
                if i not in Repetition[b]:
                    Repetition.get(b)[i] = 0
                temp = Repetition.get(b).get(i)
                Repetition.get(b)[i] = temp + 1
            # stemmed.append(tmp)
        # with open('Repetition.pickle', 'wb') as handle:
        #     pickle.dump(Repetition, handle, protocol=pickle.HIGHEST_PROTOCOL)

    tfidf = dict()
    for i in Repetition.keys():
        nt = len(Repetition.get(i))
        idf = math.log10(N / nt)
        tfidf[i] = dict()
        for j in Repetition.get(i).keys():
            tfidf.get(i)[j] = (1 + math.log10(Repetition.get(i).get(j))) * idf
            if i == "استقلال":
                print(Repetition.get(i).get(j), nt ,N, j)

    question_words = my_tokenizer.tokenize_words(my_normalizer.normalize(question))
    clean_q_words = []
    # q_w_locations = []
    q_repetition = dict()
    for i in range(len(question_words)):
        if question_words[i] not in stop_words:
            b = my_stemmer.convert_to_stem(question_words[i])
            if b not in Repetition:
                print("No match found.")
                return
            if b not in q_repetition:
                q_repetition[b] = 0
            q_repetition[b] = q_repetition.get(b) + 1
            # q_w_locations.append(Repetition[b])
            clean_q_words.append(b)

    q_tf = dict()
    q_vector_size = 0.0

    for i in q_repetition.keys():
        q_tf[i] = (1 + math.log10(q_repetition.get(i)))
        # print(q_repetition.get(i))
        q_vector_size = q_vector_size + (q_tf.get(i) * q_tf.get(i))
    q_vector_size = math.sqrt(q_vector_size)

    similarity = dict()
    for i in range(N):
        similarity[i] = 0.0
        sum = 0.0
        vector_size = 0.0
        for j in q_tf.keys():
            if i in tfidf.get(j):
                sum = sum + (q_tf.get(j) * tfidf.get(j).get(i))
                # print(tfidf.get(j).get(i), q_tf.get(j), i)
                vector_size = vector_size + (tfidf.get(j).get(i) * tfidf.get(j).get(i))
        if vector_size != 0.0:
            similarity[i] = sum / (q_vector_size * math.sqrt(vector_size))
            print(sum, q_vector_size, vector_size)
        else:
            del similarity[i]
    print(similarity)



if __name__ == '__main__':
    main()
