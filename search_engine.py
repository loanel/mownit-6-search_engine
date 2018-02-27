from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import linalg
import math
import numpy as np
import codecs
import os
import pickle as pic


# 1, 2, 3
def retrieve_and_stem(file_index):
    translation_table = dict.fromkeys(map(ord, '\"\'\r\n`<>-()%?.!:,;'), None)
    stemmer = PorterStemmer()
    with codecs.open("data/file" + str(file_index), 'r', 'utf-8') as file:
        file_content = file.read()
        file_words = file_content.split()
        file_words = [x.translate(translation_table) for x in file_words]
        english_stop_words = stopwords.words('english')
        stemmed_words = [stemmer.stem(x) for x in file_words if len(x) > 2 and x not in english_stop_words]
        bag_of_words = Counter(stemmed_words)
    return bag_of_words, stemmed_words


# 4
def calculate_term_idf(term, bags_of_words):
    term_appearances = 0
    total = len(bags_of_words)
    for i in range(total):
        if term in bags_of_words[i] and bags_of_words[i][term] != 0:
            term_appearances += 1
    return math.log(1.0 * total / term_appearances)


# 5
def calculate_idf(bags_of_words, terms):
    idf_holder = {}
    for term in terms:
        idf_holder[term] = calculate_term_idf(term, bags_of_words)
    for i in range(len(bags_of_words)):
        for word in bags_of_words[i].keys():
            bags_of_words[i][word] *= 1.0 * idf_holder[word]
    return bags_of_words


# 6
def create_A(bags_of_words, terms):
    matrix_A = np.zeros((len(terms), len(bags_of_words)))
    # mapping numbers to set items, makes everything look cleaner
    index_map = {}
    i = 0
    for term in terms:
        index_map[term] = i
        i += 1
    for j in range(len(bags_of_words)):
        for word in bags_of_words[j].keys():
            matrix_A[index_map[word], j] = bags_of_words[j][word]
    return matrix_A, index_map


# 8
def svd(matrix_A, k):
    U, S, V = linalg.svds(matrix_A, k)
    m = U.dot(np.diag(S)).dot(V)
    return m


def save_matrix(matrix, filename):
    with open("results/" + filename, 'wb') as file:
        matrix.dump(file)


def save_dictionary(dictionary, filename):
    with open("results/" + filename, 'wb') as file:
        pic.dump(dictionary, file)


def main():
    # change here to test other values
    k = 150
    terms = set()
    bags_of_words = []
    file_amount = len(os.listdir("data/")) - 1

    for i in range(1, file_amount + 1):
        bag_of_words, file_terms = retrieve_and_stem(i)
        bags_of_words.append(bag_of_words)
        new_terms_set = set(file_terms).difference(terms)
        for term in new_terms_set:
            terms.add(term)
    print(len(terms))
    print(len(bags_of_words))
    bags_of_words = calculate_idf(bags_of_words, terms)
    matrix_A, term_index_map = create_A(bags_of_words, terms)
    matrix_svdA = svd(matrix_A, k)
    save_matrix(matrix_A, "normal.txt")
    save_matrix(matrix_svdA, "svd.txt")
    save_dictionary(term_index_map, "terms.txt")


main()
