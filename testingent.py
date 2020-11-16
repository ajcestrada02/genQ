import spacy


def merge_phrase(matcher, doc, i, matches):
    '''
    Merge a phrase. We have to be careful here because we'll change the token indices.
    To avoid problems, merge all the phrases once we're called on the last match.
    '''
    if i != len(matches)-1:
        return None
    # Get Span objects
    spans = [(ent_id, label, doc[start : end]) for ent_id, label, start, end in matches]
    for ent_id, label, span in spans:
        span.merge(label=label, tag='NNP' if label else span.root.tag_)

nlp = spacy.load('en')
nlp.matcher.add_entity('MorganStanley', on_match=merge_phrase)
nlp.matcher.add_pattern('MorganStanley', [{'orth': 'Morgan'}, {'orth': 'Stanley'}], label='ORG')
nlp.pipeline = [nlp.tagger, nlp.entity, nlp.matcher, nlp.parser]

# Okay, now we've got our pipeline set up...
doc = nlp(u'Morgan Stanley fires Vice President')
for word in doc:
    print(word.text, word.tag_, word.dep_, word.head.text, word.ent_type_)
