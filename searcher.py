import argparse
import numpy as np
import pickle as pic
import heapq
from nltk.stem.porter import *
import codecs


# assuming that nobody will write something along the lines of ,,,....text(*()) to input
def parse_input(query, terms):
    stemmer = PorterStemmer()
    query_stemmed = [stemmer.stem(x) for x in query]
    q = [0] * len(terms)
    for word in query_stemmed:
        if word in terms:
            q[terms[word]] += 1
    return q


def query_normal(query):
    with open("results/normal.txt", "rb") as file:
        matrix_A = np.load(file)
    with open("results/terms.txt", "rb") as file:
        terms = pic.load(file)

    # building vector q
    q = parse_input(query, terms)

    m = len(terms)
    n = matrix_A.shape[1]
    results = dict()
    for i in range(n):
        d = [0] * m
        for j in range(m):
            d[j] = matrix_A[j, i]
        results[i] = calculate(q, d)
    return heapq.nlargest(10, results, key=lambda x: results[x]), results


def query_svd(query):
    with open("results/svd.txt", "rb") as file:
        matrix_svdA = np.load(file)

    with open("results/terms.txt", "rb") as file:
        terms = pic.load(file)

    # building vector q
    q = parse_input(query, terms)
    m = len(terms)
    n = matrix_svdA.shape[1]
    results = dict()
    for i in range(n):
        d = [0] * m
        for j in range(m):
            d[j] = matrix_svdA[j, i]
        results[i] = calculate(q, d)
    return heapq.nlargest(10, results, key=lambda x: results[x]), results


def calculate(q, d):
    q_T = np.array(q).transpose()
    return np.dot(q_T, d) / (np.linalg.norm(q_T) * np.linalg.norm(d))


def get_urls(text_indexes):
    with codecs.open("data/data", 'r', 'utf-8') as file:
        file_content = file.readlines()
        for index in text_indexes:
            for line in file_content:
                line = line.split()
                if line[0] == "file" + str(index + 1):
                    print(line[0] + " " + line[1])


def main():
    input_parser = argparse.ArgumentParser(description="Please input search words")
    input_parser.add_argument("input_query", type=str, nargs='+')
    arguments = input_parser.parse_args()
    print(arguments.input_query)
    text_indexes , result = query_normal(arguments.input_query)
    # text_indexes, result = query_svd(arguments.input_query)
    for x in text_indexes:
        print(result[x])
    get_urls(text_indexes)


main()
