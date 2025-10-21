import sys
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer, util
#from simcse import SimCSE

def tok(s: str):
    return s.lower().split()

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def doc2Vec_sim(S1,S2):
    docs = [
        TaggedDocument(tok(S1), ["d1"]),
        TaggedDocument(tok(S2), ["d2"]),
    ]

    model = Doc2Vec(vector_size=100, min_count=1, epochs=100, dm=1, seed=42, workers=1)
    model.build_vocab(docs)
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)

    v1 = model.infer_vector(tok(S1), epochs=50)
    v2 = model.infer_vector(tok(S2), epochs=50)

    print(f"similarity = {cosine(v1, v2):.6f}")

def sentence_bert_sim(S1, S2):
    sentences = [S1,S2]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    sim = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(sim), 2)

if __name__ == "__main__":
    S1 = "Yes, it is a foul. The defender pushed his opponent in the back with his upper body," \
    " without intending to play the ball."
    S2 = "Foul. The denfender made a very violent move on his opponent."
    print(sentence_bert_sim(S1, S2))
