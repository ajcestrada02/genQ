#Importing packages for generate
from nltk import *
from nltk.tokenize import RegexpTokenizer, MWETokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter#process_pdf
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import BytesIO
import timeit
import spacy


pdfFileObj = open('C:/Users/Doms/Downloads/what_is_science.pdf', 'rb')
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
DATE = []

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
		elif ent.label_ == 'DATE':
			DATE.append(ent.text)
			print(ent.text, ent.label_)
		else:
			OTHER.append(ent.text)
			print(ent.text, ent.label_)

NER = ORG + PERSON + GPE_LOCATION + MONEY + TIME + EVENT + OTHER + DATE
print("THIS IS NER")
print(NER)
# Coop through document list
for i in doc_set:

	# Clean and tokenize document string
	# Remove the NER collected words before tokenize to aviod tokenized the NER collection
	for token in range(len(NER)):
		if NER[token] in i:  # Replace the tokens to exclude the NER tokens
			i = i.replace(NER[token], " tags" + str(token) + " ")

	raw = i.lower()  #
	tokens = tokenizer.tokenize(raw)

	# Recall the words collected in NER
	for word in range(len(tokens)):
		for token in range(len(NER)):

			if tokens[word] == ("tags" + str(token) + ""):
				tokens[word] = NER[token]

	# Remove stop words from tokens
	stopped_tokens = [i for i in tokens if not i in en_stop]
	questions.append(tokens)

	# Add tokens to list
	texts.append(stopped_tokens)

# ---------------------------------------------------TOPIC MODELING (LDA)------------------------------------------------
# Turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
print(dictionary.token2id)
# Convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# Generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=20)
lda_raw = ldamodel.print_topics(num_topics=5, num_words=5000)

# ------------------------------------------- POST-PROCESSING----------------------------------------------------------------
# Data Cleansing
lda = []
for topic in range(len(lda_raw)):
	nstr = re.sub(r'[""|,|\\|''|)|(]', r'', str(lda_raw[topic]))
	lda.append(nstr)
print("LDA ")
# separate by topic
pre_tokens = []
for x in lda:
	pre_tokens.append(str(x).split(" + "))


# separate by weights and word
lda_tokens = []
for i in pre_tokens:
	for x in i:
		lda_tokens.append(str(x).split('*'))



lda_words = []
index = 0
for i in lda_tokens:
	if index >= 2 and index < len(lda_tokens) - 2:
		lda_words.append(i[1])
	index += 1

def UniqueItems(lda_words):
	seen = set()
	result = []
	for item in lda_words:
		if item not in seen:
			seen.add(item)
			result.append(item)
	return result

cleanse = UniqueItems(lda_words)
# Part-of-Speech Tagging in Templating is divided into two.
# The POS tagging from LDA to filter out the topic in the corpora and the Original Copy which is the template that
# change the
lda_tagging = pos_tag(cleanse)

print(lda_tokens)

time = timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000)
