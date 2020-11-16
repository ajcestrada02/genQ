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
import PyPDF2

#Process with PDF
# creating a pdf file object
# PDFMiner boilerplate

pdfFileObj = open('H:/Drive/folder/Thesis_Final_Paper-3.pdf', 'rb')
rsrcmgr = PDFResourceManager(caching=False)
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

text = sio.getvalue()
x = str(text.decode('utf-8'))
# Cleanup
temp = ""
for line in x:
    newLine = line.rstrip('\r\n')
    temp += newLine

x = temp
device.close()
sio.close()

print(text)