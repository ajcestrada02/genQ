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

# ---------------------------------------------WITH PDF--------------------------------------------------------------
# Process with PDF
# creating a pdf file object
pdfFileObj = open('C:/Users/Papoy/Desktop/Folders/Thesis folder/Thesis_Final_Paper-3.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file
pages = pdfReader.numPages
text = ""

# creating a page object passing text in 
for x in range(0, pages):
    pageObj = pdfReader.getPage(x)
    text += pageObj.extractText()

# closing the pdf file object
pdfFileObj.close()

# ---------------------------------------------WITH PDF--------------------------------------------------------------

# ---------------------------------------------WITHOUT PDF--------------------------------------------------------------
# text = "" #initiate the variable

nlp = spacy.load('en')

# Initiate List for NER
ORG = []
PERSON = []
GPE_LOCATION = []
MONEY = []
TIME = []
EVENT = []
OTHER = []

# text = """Bertie Steffink, nephew of the aforementioned Luke, had early in life adopted the profession of ne'er-do-weel; his father had been something of the kind before him. At the age of eighteen Bertie had commenced that round of visits to our Colonial possessions, so seemly and desirable in the case of a Prince of the Blood, so suggestive of insincerity in a young man of the middle-class."""

# text = "Hello World Amelito Estrada, This is your system."
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

questions = []
# collection NER 
doc = nlp(text)

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
        if not ent.text == ' ':
            OTHER.append(ent.text)
            print(ent.text, ent.label_)

NER = ORG + PERSON + GPE_LOCATION + MONEY + TIME + EVENT + OTHER
print(NER)
# loop through document list
for i in doc_set:

    # clean and tokenize document string
    # remove the NER collected words before tokenize to aviod tokenized the NER collection
    for token in range(len(NER)):
        if NER[token] in i:
            i = i.replace(NER[token], " tags" + str(token) + " ")

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # recall the words collected in NER
    for word in range(len(tokens)):
        for token in range(len(NER)):

            if tokens[word] == ("tags" + str(token) + ""):
                tokens[word] = NER[token]
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    questions.append(tokens)
    # add tokens to list
    texts.append(stopped_tokens)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
lda_raw = ldamodel.print_topics(num_topics=len(NER), num_words=100)

# try:
# POST-PROCESSING
# data cleansing
lda = []
for topic in range(len(lda_raw)):
    nstr = re.sub(r'[""|,|\\|''|)|(]', r'', str(lda_raw[topic]))
    lda.append(nstr)
print("LDA ")
print(lda)
# separate by topic
pre_tokens = []
for x in lda:
    pre_tokens.append(str(x).split(" + "))

print("PRE_TOKENS")
print(pre_tokens)
# separate by weights and word
lda_tokens = []
for i in pre_tokens:
    for x in i:
        lda_tokens.append(str(x).split('*'))

print("LDA_TOKENS ")
print(lda_tokens)

lda_words = []
index = 0
for i in lda_tokens:
    if index >= 2 and index < len(lda_tokens) - 2:
        lda_words.append(i[1])
    index += 1

# Part-of-Speech Tagging in Templating is divided into two.
# The POS tagging from LDA to filter out the topic in the corpora and the Original Copy which is the template that 
# change the  
# except IndexError:
#    print(IndexError)

# print(NER)
# from LDA
lda_tagging = pos_tag(lda_words)
print(lda_tagging)from nltk import *
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


#---------------------------------------------WITH PDF--------------------------------------------------------------
#Process with PDF
# creating a pdf file object
pdfFileObj = open('C:/Users/Papoy/Desktop/Folders/Thesis folder/Thesis_Final_Paper-3.pdf','rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file
pages = pdfReader.numPages
text = ""

# creating a page object passing text in
for x in range(0,pages):
    pageObj = pdfReader.getPage(x)
    text += pageObj.extractText()

# closing the pdf file object
pdfFileObj.close()

#---------------------------------------------WITH PDF--------------------------------------------------------------

#---------------------------------------------WITHOUT PDF--------------------------------------------------------------
#text = "" #initiate the variable

nlp = spacy.load('en')

#Initiate List for NER
ORG = []
PERSON = []
GPE_LOCATION = []
MONEY = []
TIME = []
EVENT = []
OTHER = []

#text = """Bertie Steffink, nephew of the aforementioned Luke, had early in life adopted the profession of ne'er-do-weel; his father had been something of the kind before him. At the age of eighteen Bertie had commenced that round of visits to our Colonial possessions, so seemly and desirable in the case of a Prince of the Blood, so suggestive of insincerity in a young man of the middle-class."""

#text = "Hello World Amelito Estrada, This is your system."
#Replace the tokens to exclude the NER tokens
# Not including the NER taggings in tokenization

#SYSTEM CODE
# PREPROCESSING
tokenized_lda = word_tokenize(text)



#Tokenized by sentence for LDA
lda_sentence_tokens = []
sentence = ""
i = 0;
for word in tokenized_lda:
    sentence += " " + word
    if(word == '.'):
        lda_sentence_tokens.append(sentence)
        #printing every sentence in a text
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

questions = []
# collection NER
doc = nlp(text)

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
        if not ent.text == ' ':
            OTHER.append(ent.text)
            print(ent.text, ent.label_)

NER = ORG + PERSON + GPE_LOCATION + MONEY + TIME + EVENT + OTHER
print(NER)
# loop through document list
for i in doc_set:

    # clean and tokenize document string
    # remove the NER collected words before tokenize to aviod tokenized the NER collection
    for token in range(len(NER)):
        if NER[token] in i:
            i = i.replace(NER[token], " tags"+str(token)+" ")

    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # recall the words collected in NER
    for word in range(len(tokens)):
        for token in range(len(NER)):

            if tokens[word] == ("tags"+str(token)+ ""):
                tokens[word] = NER[token]
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    questions.append(tokens)
    # add tokens to list
    texts.append(stopped_tokens)
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
lda_raw = ldamodel.print_topics(num_topics=len(NER), num_words=100)


#try:
# POST-PROCESSING
# data cleansing
lda = []
for topic in range(len(lda_raw)):
    nstr = re.sub(r'[""|,|\\|''|)|(]',r'',str(lda_raw[topic]))
    lda.append(nstr)
print("LDA ")
print(lda)
# separate by topic
pre_tokens = []
for x in lda:
    pre_tokens.append(str(x).split(" + "))

print("PRE_TOKENS")
print(pre_tokens)
#separate by weights and word
lda_tokens = []
for i in pre_tokens:
    for x in i:
        lda_tokens.append(str(x).split('*'))

print("LDA_TOKENS ")
print(lda_tokens)

lda_words = []
index = 0
for i in lda_tokens:
    if index >= 2 and index < len(lda_tokens)-2:
        lda_words.append(i[1])
    index += 1


# Part-of-Speech Tagging in Templating is divided into two.
# The POS tagging from LDA to filter out the topic in the corpora and the Original Copy which is the template that
# change the
#except IndexError:
#    print(IndexError)

#print(NER)
# from LDA
lda_tagging = pos_tag(lda_words)
print(lda_tagging)