from jsonpath_rw import jsonpath, parse
import codecs
import json, re
import gzip
import string
from gensim.models.word2vec import Word2Vec, LineSentence
import numpy as np


def try_jsonpath1(input_file, jsonpath_strings=['*.*.*.spacy.result.*','*.*.*.ticker.result[*].value']):
    ticker_jsonpaths = [parse(jsonpath_string) for jsonpath_string in jsonpath_strings]
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            tt = list()
            for ticker_jsonpath in ticker_jsonpaths:
                tmp = [match.value for match in ticker_jsonpath.find(json.loads(line))]
                tt += tmp
            print tt

def try_gunzip_reader(gz_input='/Users/mayankkejriwal/datasets/memex/user-worker-dig3-full-phase4-2017/gz/part-00000.txt.gz'):
    jpath_relaxed = parse('fields.phone.relaxed[*].name')
    jpath_strict = parse('fields.phone.strict[*].name')
    with gzip.open(gz_input, 'rb') as f:
        for line in f:
            # print line
            j = json.loads(re.split('\t', line[0:-1])[1])
            # print j
            strict_phones = set([match.value for match in jpath_strict.find(j)])
            relaxed_phones = set([match.value for match in jpath_relaxed.find(j)])
            # break

def sat_dataset_conversion(input_file, output_file):
    out = codecs.open(output_file, 'w')
    count = 0
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            if count == 0:
                count += 1
                nums = ['col'+str(i) for i in (range(1, 17))]
                out.write(','.join(nums))
                out.write('\n')
            else:
                fields = re.split(',',line[0:-1])
                race = 0 # did not identify as either black, hisp. or asian
                if int(fields[6]) == 1:
                    race = 1 # black
                elif int(fields[7]) == 1:
                    race = 2 # hisp
                elif int(fields[8]) == 1:
                    race = 3
                fields = fields[0:6]+[str(race)]+fields[9:]
                out.write(','.join(fields))
                out.write('\n')


    out.close()

def iterator_for_word2vec(infile, sample_text_file, only_once=False):
    flag = True
    if infile is None:
        yield iterator_for_word2vec(codecs.open(sample_text_file, 'r', 'utf-8'), sample_text_file)
    for line in infile:
        flag = False
        print line
        yield re.split(' ', line[0:-1])
    if flag and not only_once:
        # the file has ended. reopen it.
        infile.close()
        yield iterator_for_word2vec(codecs.open(sample_text_file, 'r', 'utf-8'), sample_text_file)


def online_word2vec(sample_text_file):
    """
    I want to try the gensim word2vec with an iterator. If this works, it should keep on cycling through
    the sample text file.

    If we want this to work, we should preprocess the data and go with LineSentence
    :param sample_text_file:
    :return:
    """
    infile = codecs.open(sample_text_file, 'r', 'utf-8')
    iter = 6

    model = Word2Vec(None, size=5, window=3, sg=1, min_count=0, seed=3, iter=1000)
    model.build_vocab(sentences=LineSentence(sample_text_file))

    print model['example'].tolist()
    # model.init_sims(replace=True)
    vec = model['example'].tolist()
    print np.linalg.norm(vec, 2)
    print model['2'].tolist()
    model.train(sentences=LineSentence(sample_text_file))
    # model.init_sims(replace=True)
    print model['example'].tolist()
    # sentence = iterator_for_word2vec(infile, sample_text_file)
    # while iter > 0:
    #
    #     model.train(sentences=sentence.next())
    #     print model['2'].tolist()
    #     iter -= 1
    # print model.alpha
    # while True:
    #     line = iterator_for_word2vec(infile, sample_text_file, True)
    #     if line is None:
    #         break
    #     model.build_vocab(sentences=line)
    #
    # infile.close()





#path = '/Users/mayankkejriwal/Dropbox/dig-microcap-data/'
# path = '/Users/mayankkejriwal/datasets/d3m/sat_excel/'
# path = '/Users/mayankkejriwal/datasets/tmp/sample_2_line_file.txt'
# online_word2vec(path)
# sat_dataset_conversion(path+'sat-old.csv', path+'ss-regression.csv')
#try_jsonpath1(path+'microcap_sample.json')
# try_gunzip_reader()

# st = 'gfr,gdfsdf,...453dsfsdf,cewrwe///d@'
# print 'num of puncutations in string ',st,' is: ',str(len([i for i in st if i in string.punctuation]))