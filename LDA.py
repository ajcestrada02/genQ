from nltk import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import math
import PyPDF2
import timeit
import spacy

nlp = spacy.load('en')

# Initiate List for NER
ORG = []
PERSON = []
GPE_LOCATION = []
MONEY = []
TIME = []
EVENT = []
OTHER = []

question_list = []
text = """
Bertie Steffink, nephew of the aforementioned Luke, had early in life adopted the profession of
ne'er-do-weel; his father had been something of the kind before him. At the age of eighteen Bertie 
had commenced that round of visits to our Colonial possessions, so seemly and desirable in the case 
of a Prince of the Blood, so suggestive of insincerity in a young man of the middle-class."""

# Replace the tokens to exclude the NER tokens
# Not including the NER taggings in tokenization

# SYSTEM CODE
# PREPROCESSING
tokenized_lda = word_tokenize(text)

# Tokenized by sentence for LDA
lda_sentence_tokens = []
sentence = ""
i = 0;
for word in tokenized_lda:
    sentence += " " + word
    if (word == '.'):
        lda_sentence_tokens.append(sentence)
        # printing every sentence in a text
        sentence = ""

# LDA
# Preprocess
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# compile sample documents into a list
doc_set = lda_sentence_tokens

# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    # clean and tokenize document string
    doc = nlp(i)

    for ent in doc.ents:
        if ent.label_ == 'ORG':
            ORG.append(ent.text)
            print(ent.text, ent.label_)
        elif ent.label_ == 'PERSON':
            PERSON.append(ent.text)
            print(ent.text, ent.label_)
        elif ent.label_ == 'GPE':
            GPE_LOCATION.append(ent.text)
            print(ent.text, ent.label_)
        elif ent.label_ == 'MONEY':
            MONEY.append(ent.text)
            print(ent.text, ent.label_)
        elif ent.label_ == 'TIME':
            TIME.append(ent.text)
            print(ent.text, ent.label_)
        elif ent.label_ == 'EVENT':
            EVENT.append(ent.text)
            print(ent.text, ent.label_)
        else:
            OTHER.append(ent.text)
            print(ent.text, ent.label_)

NER = ORG + PERSON + GPE_LOCATION + MONEY + TIME + EVENT + OTHER
if " " in NER:
    NER.remove(" ")
print(NER)

for word in NER:

    for sentence in doc_set:

        if "________" in sentence:
            break
        elif word in sentence:
            question_list.append(sentence.replace(word, "________"))
index = 1
for question in question_list:
    print(str(index) + "] " +question)
    index += 1
"""
for token in range(len(NER)):
        if NER[token] in i:
            i = i.replace(NER[token], " tags" + str(token) + " ")

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    for word in range(len(tokens)):
        for token in range(len(NER)):
            if tokens[word] == (" tags" + str(token) + " "):
                tokens[word] = NER[token]

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # add tokens to list
    texts.append(stopped_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
lda_raw = ldamodel.print_topics(num_topics=2, num_words=100)

try:
    # Preprocessing
    # Gathering each words in LDA

    nstr = re.sub(r'[+|""|,|\\|'']', r'', str(lda_raw[1]))
    tokens = word_tokenize(nstr)

    lda_tokens = []
    for i in tokens:
        lda_tokens.append(str(i).split('*'))

    lda_words = []
    index = 0
    for i in lda_tokens:
        if index >= 2 and index < len(lda_tokens) - 2:
            lda_words.append(i[1])
        index += 1


        # Part-of-Speech Tagging in Templating is divided into two.
        # The POS tagging from LDA to filter out the topic in the corpora and the Original Copy which is the template that
        # change the
except IndexError:
    print("Error Occur")

# from LDA
lda_tagging = pos_tag(lda_words)
print(lda_tagging)
"""