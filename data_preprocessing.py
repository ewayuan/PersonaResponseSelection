import spacy
from spacy.lang.en import English
import pickle
import time
from util import save_pickle
import en_core_web_sm

# train:
# test: 22730
# valid: 19829

# persona: 112442

spacy_nlp = spacy.load('en_core_web_sm')
all_stopwords = spacy_nlp.Defaults.stop_words
nlp = English()
# tokenizer = nlp.tokenizer

def perona_preprocissing(data_set):
    persona_path = "./cleaned/" + str(data_set) + "/persona_mini.txt"
    lines = open(persona_path,'r').read().splitlines()
    persona_dict = {}
    lineToSkip = -1
    # store each persona sentence in to dict
    for i, line in enumerate(lines):
        if i == 0:
            # the first line is user name
            persona_dict[line] = []
            cur_username = line
        elif line.find("********************") != -1:
            # if we can find ***, then its next line is user name
            if i+1 < len(lines):
                cur_username = lines[i+1]
                persona_dict[cur_username] = []
                lineToSkip = i + 1
        elif i == lineToSkip:
            continue
        else:
            # No Tokenization
            # tokens = tokenizer(line)
            # tokens_words = [str(word) for word in tokens]
            persona_dict[cur_username].append(line.strip())
            # Removing Stop words using Spacy (may should not remove stop words)
            # tokens_without_sw = [word for word in tokens if not str(word) in all_stopwords]
    print(persona_dict)
    return persona_dict


def data_preprocssing(data_set, data_type):
    lines = open("./cleaned/" + str(data_set) + "/" + str(data_type) + ".txt",'r').read().splitlines()
    data_list = []
    cur_conv_dict = {}
    for i, line in enumerate(lines):
        content_sentence_list = []
        if line == '':
            continue
        if line == "********************":
            data_list.append([(k, v) for k, v in cur_conv_dict.items()])
            cur_conv_dict = {}
            continue
        partitioned_line = line.partition('---+---')
        if partitioned_line[1].strip() == '---+---':
            user_name = partitioned_line[0].strip()
            user_content = partitioned_line[2].strip()
            # Split into list of sentences
            content = spacy_nlp(user_content)
            for sent in content.sents:
                # sent = re.sub('\W+',' ', sent )
                content_sentence_list.append(str(sent))
            cur_conv_dict[str(user_name)] = content_sentence_list
        else:
            another_user_content = partitioned_line[0].strip()
            another_content = spacy_nlp(another_user_content)
            for sent in another_content.sents:
                # sent = re.sub('\W+',' ', sent )
                content_sentence_list.append(str(sent))
            cur_conv_dict[str(user_name)].extend(content_sentence_list)
    return data_list


persona_start = time.time()
happy_persona = perona_preprocissing("happy")
# print("happy_persona: ", happy_persona)
save_pickle(happy_persona, "./data/reddit_empathetic/happy/happy_persona_mini.pkl")
persona_end = time.time()
print("persona processing time: ", persona_end - persona_start)

test_start = time.time()
happy_test_data = data_preprocssing("happy", "test_mini")
# print("happy_test_data: ", happy_test_data)
save_pickle(happy_test_data, "./data/reddit_empathetic/happy/happy_test_mini.pkl")
test_end = time.time()
print("test processing time: ", test_end - test_start)

valid_start = time.time()
happy_valid_data = data_preprocssing("happy", "valid_mini")
save_pickle(happy_valid_data, "./data/reddit_empathetic/happy/happy_valid_mini.pkl")
valid_end = time.time()
print("valid processing time: ", valid_end - valid_start)


train_start = time.time()
happy_train_data = data_preprocssing("happy", "train_mini")
save_pickle(happy_train_data, "./data/reddit_empathetic/happy/happy_train_mini.pkl")
train_end = time.time()
print("train processing time: ", train_end - train_start)

# import nltk
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer,PorterStemmer
# from nltk.corpus import stopwords
# import re
# lemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()

# sentence=str(sentence)
# sentence = sentence.lower()
# sentence=sentence.replace('{html}',"")
# cleanr = re.compile('<.*?>')
# cleantext = re.sub(cleanr, '', sentence)
# rem_url=re.sub(r'http\S+', '',cleantext)
# rem_num = re.sub('[0-9]+', '', rem_url)
# tokenizer = RegexpTokenizer(r'\w+')
# tokens = tokenizer.tokenize(rem_num)
# filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
# stem_words=[stemmer.stem(w) for w in filtered_words]
# lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
# res = " ".join(filtered_words)
