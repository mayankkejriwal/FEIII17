from gensim.models.word2vec import *
import codecs, re

def train_word2vec(text_file, output_file=None, num_dims=30):
    sentences = list()
    with codecs.open(text_file, 'r', 'utf-8') as f:
        for line in f:
            sentence = re.split(' ', line[0:-1])
            sentences.append(sentence)
    model = Word2Vec(sentences=sentences, size=num_dims, window=5, sg=1)
    model.init_sims(replace=True)
    # print model.most_similar(positive=['1', '2', '3'])
    if output_file:
        model.save(output_file)


def word2vec_custom_tester(word2vec_model_file, pos_words):
    """
    Custom file for putting in random stuff.
    :param word2vec_model_file:
    :return:
    """
    model = Word2Vec.load(word2vec_model_file)
    print model.most_similar(positive=pos_words)