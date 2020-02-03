# -*- coding: utf-8 -*-
"""
configuration.py
Glenn Abastillas
Created on Thu Oct 31 09:48:49 2019


"""

from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess, lemmatize


class TopicModel(object):

    def __init__(self, corpus=None):
        self.model = None
        pass

    def preprocess(self, document, min_len=3):
        return simple_preprocess(document, min_len=min_len)

    def fit(self, corpus):
        """
        Train a model on a preprocessed corpus

        Parameters
        ----------
            corpus (list): Documents as BOW representation

        Notes
        -----
            This method takes preprocessed strings into BOW representation.
        """
        model = LdaModel(corpus)
        self.model = model

    def predict(self, document):
        pass

    def lemmatize(self, text, tag=[b"N", b"V", b"R", b"J"], sep=b"/"):
        """
        Return only the nouns, verbs, adverbs, and adjectives in a text

        Parameters
        ----------
            text (str): Original text to filter
            tag (list): POS tags to keep
            sep (str): Separator / character to split lemmas by as a byte

        Notes
        -----
            This method yields tokens that have valid POS tags.
        """
        lemmas = lemmatize(text)

        for lemma in lemmas:
            token, pos = lemma.split(sep)

            valid_pos = [pos.startswith(_) for _ in tag]

            if valid_pos:
                yield token

    def vectorize(self, corpus):
        """
        Convert text into numeric vectors.

        Parameters
        ----------
            corpus (list): List of documents with tokens to vectorize

        Notes
        -----
            Return the Dictionary corpus object and a vectorized corpus
            Input should be lemmatized text as bytes
        """
        bag = Dictionary(corpus)
        docs = [bag.doc2bow(doc) for doc in corpus]
        return bag, docs





if __name__ == "__main__":
    docx = ['this is an example document to be preprocessed', 'another document that requires preprocessing abc splurge.']

    # print(lemmatize(docx[0]))

    tk = TopicModel()
    lemmas = [list(tk.lemmatize(_)) for _ in docx]
    print(lemmas)
    corpus, docs = tk.vectorize(lemmas)
    print(corpus, docs)
