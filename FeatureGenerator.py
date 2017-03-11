from FilePreprocessor import *
from DocEmbeddings import *
"""
The methods should only take the training folder as input and return dicts with roles, vectors and labels
"""


def convert_jsons_to_pos_neg_script(file_path, word2vec_model, singular_flag=True):
    plural = ['underwriters', 'insurers', 'servicers', 'affiliates', 'guarantors', 'sellers', 'trustees', 'agents',
              'counterparties', 'issuers']
    singular = ['underwriter', 'insurer', 'servicer', 'affiliate', 'guarantor', 'seller', 'trustee', 'agent',
                'counterparty', 'issuer']

    if singular_flag:
        file_array = singular
    else:
        file_array = singular+plural

    for f in file_array:
        convert_json_to_pos_neg(file_path+f+"_train.json", word2vec_model, file_path+'/train-pos-neg/'+f+'.tsv')
        convert_json_to_pos_neg(file_path + f + "_test.json", word2vec_model,
                                file_path + '/test-pos-neg/' + f + '.tsv')



def convert_json_to_pos_neg(train_file, word2vec_model_file, pos_neg_file):
    """
    Will print warnings (1) training file has no jsons (not outputting anything as a result), (2) training file
     only contains one type of label.

     Will always use normalized averaged words to represent doc vec.
    :param train_file:
    :param word2vec_model_file:
    :param pos_neg_file:
    :return:
    """
    word2vec_model = Word2Vec.load(word2vec_model_file)
    train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
    unique_labels = set()

    if len(train_json) == 0:
        print 'Quitting...empty file'
    else:
        out = codecs.open(pos_neg_file, 'w', 'utf-8')
        for element in train_json:
            vec = generate_doc_embedding(word2vec_model, tokenize_string(element['THREE_SENTENCES'].lower()))
            labels = _get_labels(element, True)
            if len(labels) == 0:
                raise Exception
            else:
                unique_labels = unique_labels.union(set(labels))
                for label in labels:
                    out.write(element['ID'] + '\t' + str(vec) + '\t' + str(label) + '\n')
        out.close()
        if len(unique_labels) == 1:
            print 'Warning: you only had one unique label '+str(unique_labels)+' in document '+train_file


def generate_raw_features(training_folder, word2vec_model_file, output_file = None, use_equiv=True, pos_neg_file=None):
    answer_dict = dict()
    word2vec_model = Word2Vec.load(word2vec_model_file)
    listOfFiles = glob.glob(training_folder + '*.json')
    # out = codecs.open(output_file, 'w', 'utf-8')
    for train_file in listOfFiles:
        train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
        for element in train_json:
            if 'ROLE' not in element or 'THREE_SENTENCES' not in element:
                continue
            else:
                if element['ROLE'].lower() not in answer_dict:
                    answer_dict[element['ROLE'].lower()] = list()
                vec = generate_doc_embedding(word2vec_model, tokenize_string(element['THREE_SENTENCES'].lower()))
                labels = _get_labels(element)
                if len(labels) == 0:
                    raise Exception
                else:
                    for label in labels:
                        tmp = dict()
                        tmp['vector'] = vec
                        tmp['label'] = label
                        answer_dict[element['ROLE'].lower()].append(tmp)
    if use_equiv:
        _condense_answer_dict(answer_dict)
    if output_file: # a simple json dump
        json.dump(answer_dict, codecs.open(output_file, 'w', 'utf-8'))
    if pos_neg_file: # something we can use in TokenSupervised
        _write_out_dict_as_pos_neg_file(answer_dict, pos_neg_file)
    return answer_dict


def _condense_answer_dict(answer_dict):
    plural = ['underwriters', 'insurers', 'servicers', 'affiliates', 'guarantors', 'sellers', 'trustees', 'agents', 'counterparties', 'issuers']
    singular = ['underwriter', 'insurer', 'servicer', 'affiliate', 'guarantor', 'seller', 'trustee', 'agent', 'counterparty', 'issuer']
    for i in range(0, 10): # change this if number of elements in plural/singular changes
        if plural[i] in answer_dict:
            if singular[i] not in answer_dict:
                answer_dict[singular[i]] = list()
            answer_dict[singular[i]] += answer_dict[plural[i]]
            answer_dict.pop(plural[i], None)


def _get_labels(json_obj, unique_only=False):
    labels = list()
    for k, v in json_obj.items():
        if 'RATING_EXPERT' not in k:
            continue
        else:
            if v.lower() == 'irrelevant':
                labels.append(0)
            else:
                labels.append(1)
    if unique_only:
        return list(set(labels))
    else:
        return labels


def _analyze_feature_dict(feature_dict):
    for k, v in feature_dict.items():
        print k,' has ',str(len(v)),' items.'


def _write_out_dict_as_pos_neg_file(feature_dict, pos_neg_file):
    out = codecs.open(pos_neg_file, 'w', 'utf-8')
    for k, v in feature_dict.items():
        for element in v:
            out.write(k+'\t'+str(element['vector'])+'\t'+str(element['label'])+'\n')
    out.close()





# path = '/Users/mayankkejriwal/datasets/FEIII17/dec-16-data/FEIIIY2_csv/'
# convert_jsons_to_pos_neg_script(path+'Training/partitions-by-role-singular/90-10/', path+'word2vec_raw_training',singular_flag=True)
# d = generate_raw_features(path+'Training/', path+'word2vec_raw', path+'raw_training_vectors.json',True,path+'raw_pos_neg.tsv')
# _analyze_feature_dict(d)