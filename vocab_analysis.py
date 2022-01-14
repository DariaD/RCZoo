import os
import re
import enchant
import langid
#from nltk.corpus import words
#from parameters import save_data_path
from parameters import data_folder, result_folder

us_dict = enchant.Dict("en_US")
gb_dict = enchant.Dict("en_GB")

domain_list = [".com", ".co.", ".tv", ".org", ".pl", ".edu"]

# #folder = "/home/pinecone/Data/Vocabulary"
# folder = "/home/pinecone/Data/Vocabulary/MAY_FROM_CLASTER"
# #folder = os.path.join(save_data_path, "Vocabulary")
# vocab_files = os.listdir(folder)

RE_NUM = re.compile(r'^[-+]?[0-9]+([.,£$%€0-9]?)+$')
RE_WORD = re.compile(r'^[a-z]+$')
RE_WEB = re.compile(r"^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$")


def is_num(word):
    if re.match(RE_NUM, word):
        return True
    return False

# We priotaritaze precision over recall
def is_web_link(word):
    if "www." in word or "http:/" in word or "https:/" in word:
        return True
    if re.match(RE_WEB, word):
#        if "www." in word or "http:/"in word or "https:/" in word:
#            return True
        for domain in domain_list:
            if word.endswith(domain) or domain+"/" in word:
                return True
    return False


def is_english_word(word):
    try:
        if us_dict.check(word) or gb_dict.check(word):
            return True
        lang, value = langid.classify(word)
        if lang == "en":
            return True
    except enchant.errors.Error:
        print("Error happend! Word:", word)

    return False


def is_not_ascii(word):
    #print(word)
    for symbol in word:
        ascii_code = ord(symbol)
        #print(ascii_code)
        if ascii_code <= 31 or ascii_code >= 128:
            return True
    return False


def write_to_file(dataset_name, words_group, vocabulary):
    # file_name = "/home/pinecone/Data/Vocabulary/debug/{}_{}.txt".format(dataset_name, words_group)
    vocabulary_path = os.path.join(result_folder, dataset_name, "vocabulary")
    if not os.path.exists(vocabulary_path):
        os.makedirs(vocabulary_path)
    file_name = os.path.join(vocabulary_path, "{}_{}.txt".format(dataset_name, words_group))
    if len(vocabulary) > 0:
        vocabulary_file = open(file_name, "w")
        for v in sorted(vocabulary):
            string_out = v + "\n"
            vocabulary_file.write(string_out)
        vocabulary_file.close()


#print("&".join([" \\bf Dataset", "\\bf English Words", "\\bf Numbers", "\\bf Not English Words",  "\\bf Not Ascii", "\\bf Web Links"])+ " \\\\\hline")


# vocab_files.sort()
# for file in vocab_files:
#     dataset_name = file.split(".")[0]
#     if "txt" not in file:
#         continue
#
#     # if "TriviaQA" not in dataset_name:
#     #     continue
#     # print(dataset_name)
#
#

def vocab_analysis_finction(file, dataset_name):
    number_list, not_ascii, ascii_symbol, web_list = [], [], [], []
    english_words, non_english_words, others = [], [], []

    with open(file) as reader:
        vocabulary_lines = reader.readlines()
        for word in vocabulary_lines:
            word = word.strip()
            if len(word) == 0:
                continue

            # Not Ascii
            if is_not_ascii(word):
                not_ascii.append(word)
                continue

            # Web Links
            if is_web_link(word):
                web_list.append(word)
                continue

            # Numbers
            if is_num(word):
                number_list.append(word)
                continue

            if is_english_word(word):
                english_words.append(word)
                continue

            non_english_words.append(word)


    write_to_file(dataset_name, "eng_words", english_words)
    write_to_file(dataset_name, "non_eng_words", non_english_words)
    write_to_file(dataset_name, "ascii_symbol", ascii_symbol)
    write_to_file(dataset_name, "not_ascii", not_ascii)
    write_to_file(dataset_name, "numbers", number_list)
    write_to_file(dataset_name, "web_links", web_list)

    output_list = [english_words,
                   number_list,
                   non_english_words,
                   #ascii_symbol,
                   not_ascii,
                   web_list
                  ]

    # print("&".join(["Dataset", "\# english words", "\# numbers", "\# ascii mix", "\# not ascii", "\# web links"])+ " \\\\\hline")
    # print("&#&".join(["Dataset", "English words", "Not English words", "Numbers", "Ascii Mix", "Not Ascii", "Web Links"])+ " \\\\\hline")

    output_str_list = [dataset_name] + ["0(0\%)" if len(x) == 0 else str(len(x)) + " ({:.1f}\%)".format(len(x)/len(vocabulary_lines)*100) for x in output_list]
    output_line = "\t&".join(output_str_list)+"\\\\  %" + str(len(vocabulary_lines))
    print(output_line)
    return output_line

