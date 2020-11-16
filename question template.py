import spacy
from gensim import corpora, models

nlp = spacy.load('en')

texts ="""Alberto, K., Abella of risk management should start with risk identification. The
purpose of risk identification is to discover all factors that could lead to project
failure (Hall, 1998). These factors are connected with the technology used on
the project, software development process and organizational factors. These
areas should be observed Alberto K. Abella assessed in order to capture all of the potential
risks. It is necessary to capture details connected with the discovered risk, like
risk description, probability of risk occurrence, costs connected with
materialized risk and possible risk solutions and avoidance strategies.
The second step of the risk management process is to assess the level of
exposure for each risk. Alberto is step, discovered risks should be ranked in levels
according to risk impact (Capers, 1994). Risks should be classified according to
the degree of impact in order to choose important risks to be solved first. Risks
with a devastating impact should be assessed before risks with a low impact.
This is important because risks with a huge impact should be considered in the
early development phases, when the costs connected with risk materialization."""

nlp = nlp(texts)
Noun = []
Name = []
cleanse = []
temp = []

for token in nlp.noun_chunks:

    Noun.append(token.text)

for ner in nlp.ents:

    if ner.label_ == 'GPE':
        a = ""
    else:
        Name.extend(ner.text)

print


def UniqueItems(Noun):
    seen = set()
    result = []
    for item in Noun:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


cleanse = UniqueItems(Noun)

for words in cleanse:

    print(words)

