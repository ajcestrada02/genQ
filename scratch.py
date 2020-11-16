from django.shortcuts import render
from django.http import HttpRequest

#Importing packages for generate
from nltk import *
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, MWETokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import math
from random import randint
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import BytesIO
import timeit
import spacy

pdfFileObj = open('C:/Users/Doms/Downloads/what_is_science.pdf','rb')
rsrcmgr = PDFResourceManager()
sio = BytesIO()
codec = 'utf-8'
laparams = LAParams()
device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

# creating a pdf reader object
# printing number of pages in pdf file
# creating a page object passing text in

for page in PDFPage.get_pages(pdfFileObj):
    interpreter.process_page(page)
pdfFileObj.close()

# Get text from StringIO

x = sio.getvalue()
text = str(x.decode('utf-8'))
temp = ""
for line in text:
    newLine = line.rstrip('\r\n')
    temp += newLine

text = temp
# Cleanup

device.close()
sio.close()

nlp = spacy.load('en')

# Initiate list of label for NER
ORG = []
PERSON = []
GPE_LOCATION = []
MONEY = []
TIME = []
EVENT = []
OTHER = []

# Initiate list of question variable
questions = []

# ------------------------------------------------------PREPROCESSING---------------------------------------------------

# Tokenized text by sentence to become document in every
lda_sentence_tokens = sent_tokenize(text)

# Initiate tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Create English stop words list
en_stop = get_stop_words('english')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# Compile sample documents into a list
doc_set = lda_sentence_tokens

# List for tokenized documents in loop
texts = []

# Collection of NER
doc = nlp(text)
for token in doc.noun_chunks:
    print(token)
for ent in doc.ents:
    if not ent.text == ' ':
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
print("THIS IS NER")
print(NER)

NERstr = ""

for index in NER:
    NERstr += " "+ index

print("NER IN STRING")
