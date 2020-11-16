import spacy
import re
from gensim import corpora, models



def dataCleansing(Words):
    nlp = spacy.load('en')
    remove_space = re.sub(r'["\n]', r' ', Words)
    parse = nlp(remove_space)
    sentences = [sent.string.strip() for sent in parse.sents]
    
    return sentences

def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc
def UniqueItems(Noun):
    seen = set()
    result = []
    for item in Noun:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

nlp = spacy.load('en')
nlp.add_pipe(remove_whitespace_entities, after='ner')
texts ="""Astronomers had long known about the existence of variable
stars—stars whose brightness changes over time, slowly
shifting between brilliant and dim—when, in 1912, Henrietta
Leavitt announced a remarkable (and totally unanticipated)
discovery about them. For these stars, the length of time
between their brightest and dimmest points seemed to be
related to their overall brightness: slower cycling stars are
more luminous. At the time, no one knew why that was the
case, but nevertheless, the discovery allowed astronomers
to infer the distances to far-off stars, and hence, to figure
out the size of our own galaxy. Leavitt’s observation was a true surprise—a discovery
in the classic sense—but one that came only after she’d spent years carefully
comparing thousands of photos of these specks of light, looking for patterns
in the darkness"""

text = ''
text = re.sub("[\(\[].*?[\)\]]", "", texts) 

nlp = nlp(text)
Noun = []
Name = []
cleanse = []
temp = []

ORG = []
PERSON = []
GPE_LOCATION = []
DATE = []
EVENT = []

questsent = ''
for ent in nlp.noun_chunks:

     print(ent.text, ent.root.dep_, ent.root.head.text)
     if ent.root.dep_ == 'nsubj':

        questsent += " "+ent.text +" "+ent.root.head.text
     elif ent.root.dep_  == 'pobj':
         questsent += " "+ent.root.head.text +" "+ ent.text
     elif ent.root.dep_ == 'dobj':
         questsent += " "+ent.root.head.text +" "+ ent.text
     elif ent.root.dep_ == 'appos':
         questsent += ent.root.head.text +" "+ ent.text
        
         
print(questsent)         
for ent in nlp.ents:
    print("name")
    print(ent.text, ent.root.dep_, ent.root.head.text)
    if not ent.text == ' ' and not ent.text == '' and not ent.text == '  ' :

        if ent.label_ == 'ORG' and not ent.text == ' ' and not ent.text == '':


            ORG.append(ent.text)

        elif ent.label_ == 'PERSON' and not ent.text == ' ' and not ent.text == '':

            PERSON.append(ent.text)

        elif ent.label_ == 'GPE' and not ent.text == ' ' and not ent.text == '':

            GPE_LOCATION.append(ent.text)

        elif ent.label_ == 'DATE':

            DATE.append(ent.text)

        elif ent.label_ == 'EVENT':

            EVENT.append(ent.text)
  
	    
NER = ORG + PERSON + GPE_LOCATION + DATE + EVENT


        
NER = UniqueItems(Noun)
nlp = spacy.load('en')
sentences = dataCleansing(text)
string = ''
chunkdep = []
aux = ''
placesents = ''
question_list = []
i = 0
for sents in sentences:
    parser = nlp(sents)
    
    for words in parser.ents:

         if words.label_ == 'ORG':

             aux = sents.replace(sents[words.start_char:words.end_char], 'what')
             
         if words.label_ == 'PERSON':
             
             aux = sents.replace(sents[words.start_char:words.end_char], 'who')

         if words.label_ == 'GPE':
             
            aux = sents.replace(sents[words.start_char:words.end_char], 'where')

         if words.label_ == 'EVENT':
             
            aux = sents.replace(sents[words.start_char:words.end_], 'when')

             

        

        
    question_list.append(aux.capitalize()+" "+placesents)
    placesents = ''

lemmas = [parser.lemma_ for parser in parser if not parser.is_stop]
print(lemmas)
print(question_list)
            

    



