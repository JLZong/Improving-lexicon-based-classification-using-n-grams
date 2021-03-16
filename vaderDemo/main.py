# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import csv

def demo_1():
    sentences = ['I like you just so so', 'I like you a little', 'I like you', 'I like you very much',
                 'I hate you just so so', 'I hate you a little', 'I hate you', 'I hate you very much', ]
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))

def demo_2():
    sentences = [
        'I like you just so so',
        'I like you a little',
        'I like you',
        'I like you very much',
        'I hate you just so so',
        'I hate you a little',
        'I hate you',
        'I hate you very much',
    ]

    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))

def yelp_demo():
    df_yelp = pd.DataFrame(pd.read_csv('dataset/yelp_labelled.csv', names=['value', 'key']))
    sentences = df_yelp['value'].tolist()

    paragraphSentiments = 0.0
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentences), 4)))

def imdb_demo():
    df_imdb = pd.DataFrame(pd.read_csv('dataset/imdb_labelled.csv', names=['value', 'key']))
    sentences = df_imdb['value'].tolist()

    paragraphSentiments = 0.0
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentences), 4)))

def amazon_demo():
    df_amazon = pd.DataFrame(pd.read_csv('dataset/amazon_cells_labelled.csv', names=['value', 'key']))
    sentences = df_amazon['value'].tolist()

    paragraphSentiments=0.0
    analyzer = SentimentIntensityAnalyzer()
    for sentence in sentences:
        vs = analyzer.polarity_scores(sentence)
        print("{:-<65} {}".format(sentence, str(vs)))
        paragraphSentiments += vs["compound"]
    print("AVERAGE SENTIMENT FOR PARAGRAPH: \t" + str(round(paragraphSentiments / len(sentences), 4)))

if __name__ == "__main__":
    demo_1()
    print('--------------------------------------------------------------------------------------------------------------------------------------')
    demo_2()
    print()
    print('-----------------------------------------------------------------Demo of amazon labelled-----------------------------------------------------------------')
    amazon_demo()
    print()
    print('-----------------------------------------------------------------Demo of imdb labelled-----------------------------------------------------------------')
    imdb_demo()
    print()
    print('-----------------------------------------------------------------Demo of yelp labelled-----------------------------------------------------------------')
    yelp_demo()



