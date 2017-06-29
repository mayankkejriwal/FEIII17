from FilePreprocessor import *
from DocEmbeddings import *
from gensim.models.doc2vec import *
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

            if v.lower() == 'irrelevant' or v.lower() == 'neutral':
                labels.append(0)
            elif v.lower() == 'highly relevant':
                labels.append(1)
            elif v.lower() == 'relevant':
                labels.append(0.5)
            else:
                if len(v) == 0:
                    continue
                else:
                    print v
                    raise Exception
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

def build_role_dict_from_training_only(singular_training_json):
    """
    The 0 labels will be converted to -1.
    :param singular_training_json:
    :return:
    """
    list_of_texts = json.load(codecs.open(singular_training_json, 'r', 'utf-8'))
    answer = dict()
    for element in list_of_texts:
        element['ROLE'] = element['ROLE'].lower()
        if element['ROLE'] not in answer:
            answer[element['ROLE']] = dict()
        labels = _get_labels(element)
        for label in labels:
            if label == 0:
                label = -1
            if label not in answer[element['ROLE']]:
                answer[element['ROLE']][label] = list()
                answer[element['ROLE']][label].append('training-'+element['ID'])
    return answer

def compute_doc2vec_score(doc2vec_model, label_dict, ref_id):
    # if -1 not in label_dict:
    #     return 1
    # elif 1 not in label_dict:
    #     return 0
    # else:
        neg_score = 0.0
        pos_score = 0.0
        pos_score2 = 0.0
        length = 0
        if 1 in label_dict:
            for pos in label_dict[1]:
                pos_score += doc2vec_model.docvecs.similarity(ref_id, pos)
                length += 1
        # pos_score = pos_score/len(label_dict[1])
        if -1 in label_dict:
            for neg in label_dict[-1]:
                neg_score = neg_score - doc2vec_model.docvecs.similarity(ref_id, neg)
                length += 1
        if 0.5 in label_dict:
            for p in label_dict[0.5]:
                pos_score2 = pos_score2 + (0.5*doc2vec_model.docvecs.similarity(ref_id, p))
                length += 1

        return (((pos_score+neg_score+pos_score2)/length)+1)/2


def exhaustive_nn_on_test_data(singular_training_json, doc_embedding_model, singular_testing_json, output_file):
    model = Doc2Vec.load(doc_embedding_model)
    out = codecs.open(output_file, 'w', 'utf-8')
    role_dict = build_role_dict_from_training_only(singular_training_json)
    with codecs.open(singular_testing_json, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            id = obj['tags'][0]
            role = obj['tags'][1]
            score = compute_doc2vec_score(model, role_dict[role], id)
            out.write(id+','+str(score)+'\n')


    out.close()





# path = '/Users/mayankkejriwal/datasets/FEIII17/dec-16-data/FEIIIY2_csv/'
# exhaustive_nn_on_test_data(path+'Training/all-singular.json', path+'all-singular-doc2vec-20dims',path+'all-singular-testing-taggeddoc.jl',path+'scores-dry-run-3.csv')
# build_role_dict_from_training_only(path+'Training/all-singular.json')
# convert_jsons_to_pos_neg_script(path+'Training/partitions-by-role-singular/90-10/', path+'word2vec_raw_training',singular_flag=True)
# d = generate_raw_features(path+'Training/', path+'word2vec_raw', path+'raw_training_vectors.json',True,path+'raw_pos_neg.tsv')
# _analyze_feature_dict(d)