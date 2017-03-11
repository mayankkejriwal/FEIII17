import glob
import codecs
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from WordEmbeddings import *



def tokenize_string(string):
    """
    I designed this method to be used independently of an obj/field. If this is the case, call _tokenize_field.
    It's more robust.
    :param string: e.g. 'salt lake city'
    :return: list of tokens
    """
    list_of_sentences = list()
    tmp = list()
    tmp.append(string)
    k = list()
    k.append(tmp)
    # print k
    list_of_sentences += k  # we are assuming this is a unicode/string

    word_tokens = list()
    for sentences in list_of_sentences:
        # print sentences
        for sentence in sentences:
            for s in sent_tokenize(sentence):
                word_tokens += word_tokenize(s)

    return word_tokens


def assign_ids_to_training_files(input_training_folder, output_training_folder):
    """
    We assign a unique ID to every single dictionary in the FOLDER.
    :param input_training_folder:
    :param output_training_folder:
    :return:
    """
    listOfFiles = glob.glob(input_training_folder + '*.json')
    # print listOfFiles[0].replace(input_training_folder, output_training_folder)
    # print listOfFiles
    count = 1
    for train_file in listOfFiles:
        out = codecs.open(train_file.replace(input_training_folder, output_training_folder), 'w', 'utf-8')
        train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
        for element in train_json:
            element['ID'] = str(count)
            count += 1
        json.dump(train_json, out)
        out.close()


def split_training_files_by_role(training_folder, output_folder, singular_only=True):
    """
    We don't do the singular plural thing here. We do convert the role to lower-case
    :param training_folder:
    :param output_folder:
    :return:
    """
    listOfFiles = glob.glob(training_folder + '*.json')
    role_dict = dict()
    for train_file in listOfFiles:
        # out = codecs.open(train_file.replace(input_training_folder, output_training_folder), 'w', 'utf-8')
        train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
        for element in train_json:
            role = element['ROLE'].lower()
            if role not in role_dict:
                role_dict[role] = list()
            role_dict[role].append(element)
    plural = ['underwriters', 'insurers', 'servicers', 'affiliates', 'guarantors', 'sellers', 'trustees', 'agents',
              'counterparties', 'issuers']
    singular = ['underwriter', 'insurer', 'servicer', 'affiliate', 'guarantor', 'seller', 'trustee', 'agent',
                'counterparty', 'issuer']
    if singular_only:
        for i in range(0, len(plural)):
            if plural[i] in role_dict and singular[i] in role_dict:
                role_dict[singular[i]] += role_dict[plural[i]]
                del role_dict[plural[i]]
    for k, v in role_dict.items():
        out = codecs.open(output_folder+k+".json", 'w', 'utf-8')
        json.dump(v, out)
        out.close()


def combine_all_training(training_folder, output_file, singular_only=True):
    listOfFiles = glob.glob(training_folder + '*.json')
    all_jsons = list()
    for train_file in listOfFiles:
        # out = codecs.open(train_file.replace(input_training_folder, output_training_folder), 'w', 'utf-8')
        train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
        for element in train_json:
            element['ROLE'] = element['ROLE'].lower()
        all_jsons += train_json
    plural = ['underwriters', 'insurers', 'servicers', 'affiliates', 'guarantors', 'sellers', 'trustees', 'agents',
              'counterparties', 'issuers']
    singular = ['underwriter', 'insurer', 'servicer', 'affiliate', 'guarantor', 'seller', 'trustee', 'agent',
                'counterparty', 'issuer']
    if singular_only:
        for j in all_jsons:
            if j['ROLE'] in plural:
                i = plural.index(j['ROLE'])
                j['ROLE'] = singular[i]
    out = codecs.open(output_file, 'w', 'utf-8')
    json.dump(all_jsons, out)
    out.close()




def tokenize_sentences_from_working_file(working_file, output_file):
    # listOfFiles = glob.glob(training_folder + '*.json')
    out = codecs.open(output_file, 'w', 'utf-8')
    working_json = json.load(codecs.open(working_file, 'r', 'utf-8'))
    for element in working_json:
        if 'THREE_SENTENCES' in element and element['THREE_SENTENCES'] is not None:
            text = ' '.join(tokenize_string(element['THREE_SENTENCES'].lower()))
            out.write(text)
            out.write('\n')
    out.close()


def tokenize_sentences_from_training_folder(training_folder, output_file):
    listOfFiles = glob.glob(training_folder + '*.json')
    out = codecs.open(output_file, 'w', 'utf-8')
    for train_file in listOfFiles:
        train_json = json.load(codecs.open(train_file, 'r', 'utf-8'))
        for element in train_json:
            if 'THREE_SENTENCES' in element and element['THREE_SENTENCES'] is not None:
                text = ' '.join(tokenize_string(element['THREE_SENTENCES'].lower()))
                out.write(text)
                out.write('\n')
    out.close()


def partition_jsons(input_folder, output_folder, train_percent):
    listOfFiles = glob.glob(input_folder + '*.json')
    # test_percent = 100.0-train_percent
    for role_file in listOfFiles:
        print 'partitioning file: '+role_file
        out_train = codecs.open(role_file.replace(input_folder, output_folder).replace(".json", "_train.json"), 'w', 'utf-8')
        out_test = codecs.open(role_file.replace(input_folder, output_folder).replace(".json", "_test.json"), 'w',
                                'utf-8')
        list_of_elements = json.load(codecs.open(role_file, 'r', 'utf-8'))
        random.shuffle(list_of_elements)
        train_list = list_of_elements[0:int((train_percent/100.0)*len(list_of_elements))]
        test_list = list_of_elements[len(train_list):]
        print 'num training elements: ',str(len(train_list))
        json.dump(train_list, out_train)
        json.dump(test_list, out_test)
        out_train.close()
        out_test.close()

def partition_single_json(input_file, output_train, output_test, train_percent):
    out_train = codecs.open(output_train, 'w',
                            'utf-8')
    out_test = codecs.open(output_test, 'w',
                           'utf-8')
    list_of_elements = json.load(codecs.open(input_file, 'r', 'utf-8'))
    random.shuffle(list_of_elements)
    train_list = list_of_elements[0:int((train_percent / 100.0) * len(list_of_elements))]
    test_list = list_of_elements[len(train_list):]
    print 'num training elements: ', str(len(train_list))
    json.dump(train_list, out_train)
    json.dump(test_list, out_test)
    out_train.close()
    out_test.close()

def convert_raw_text_to_meta_text(raw_text_file, meta_text_file):
    # TO IMPLEMENT
    pass


# path = '/Users/mayankkejriwal/datasets/FEIII17/dec-16-data/FEIIIY2_csv/'
# train_word2vec(path+'training_working_sentences_raw.txt', path+'word2vec_raw_training_working')
# word2vec_custom_tester(path+'word2vec_raw', ['regulators'])
# assign_ids_to_training_files(path+'Training/by-company-without-id/', path+'Training/by-company-with-id/')
# split_training_files_by_role(path+'Training/by-company-with-id/',path+'Training/by-role-singular/')
# tokenize_sentences_from_training_folder(path+'Training/',path+'training_sentences_raw.txt')
# tokenize_sentences_from_working_file(path+'Working.json', path+'working_sentences_raw.txt')
# partition_single_json(path+'Training/all-exhaustive.json',
#                       path+'Training/partitions-all-exhaustive/10-90/all-exhaustive_train.json',
#                       path + 'Training/partitions-all-exhaustive/10-90/all-exhaustive_test.json', 10.0)
# combine_all_training(path+'Training/by-company-with-id/', path+'Training/all-exhaustive.json', singular_only=False)
# partition_jsons(path+'Training/by-role-exhaustive/', path+'Training/partitions-by-role-exhaustive/90-10/', 90.0)

