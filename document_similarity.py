import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import spacy
nlp = spacy.load('en')

# to remove punctuations
translator = str.maketrans('', '', string.punctuation)

# some sample documents
resumes = ["Executive Administrative Assistant with over 10 years of experience providing thorough and skillful support to senior executives.",
"Experienced Administrative Assistant, successful in project management and systems administration.",
"10 years of administrative experience in educational settings; particular skill in establishing rapport with people from diverse backgrounds.",
"Ten years as an administrative support professional in a corporation that provides confidential case work.",
"A highly organized and detail-oriented Executive Assistant with over 15 years' experience providing thorough and skillful administrative support to senior executives.",
"More than 20 years as a knowledgeable and effective psychologist working with individuals, groups, and facilities, with particular emphasis on geriatrics and the multiple psychopathologies within that population.",
"Ten years as a sales professional with management experience in the fashion industry.",
"More than 6 years as a librarian, with 15 years' experience as an active participant in school-related events and support organizations.",
"Energetic sales professional with a knack for matching customers with optimal products and services to meet their specific needs. Consistently received excellent feedback from customers.",
"More than six years of senior software engineering experience, with strong analytical skills and a broad range of computer expertise.",
"Software Developer/Programmer with history of productivity and successful project outcomes."]

job_doc = ["""Executive Administrative with a knack for matching and effective psychologist with particular emphasis on geriatrics"""]

# combine the two
_all = resumes+job_doc

# convert each to spacy document
docs= [nlp(document) for document in _all]

# lemmatizae words, remove stopwords, remove punctuations
docs_pp = [' '.join([token.lemma_.translate(translator) for token in docs if not token.is_stop]) for docs in docs]

# get tfidf matrix
tfidf_vec = TfidfVectorizer()
tfidf_matrix = tfidf_vec.fit_transform(docs_pp).todense()

# calculate similarity
cosine_similarity(tfidf_matrix[-1,], tfidf_matrix[:-1,])
