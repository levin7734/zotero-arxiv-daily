import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], model: str = 'allenai/specter') -> list[ArxivPaper]:
    # Modell laden
    encoder = SentenceTransformer(model)
    
    # Corpus nach Datum sortieren (neueste zuerst)
    corpus = sorted(corpus, key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)
    
    # Lineare Zeitgewichtsfunktion
    time_decay_weight = np.linspace(1, 0.1, len(corpus))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    
    # Features aus Titel und Abstract kombinieren
    corpus_texts = [f"{paper['data']['title']} {paper['data']['abstractNote']}" 
                    for paper in corpus if 'abstractNote' in paper['data']]
    candidate_texts = [f"{paper.title} {paper.summary}" 
                       for paper in candidate if paper.summary]
    
    # Embeddings berechnen
    corpus_feature = encoder.encode(corpus_texts)
    candidate_feature = encoder.encode(candidate_texts)
    
    # Ähnlichkeit berechnen
    sim = encoder.similarity(candidate_feature, corpus_feature)  # [n_candidate, n_corpus]
    
    # Scores mit Zeitgewichtung
    scores = (sim * time_decay_weight).sum(axis=1)
    
    # Scores normalisieren auf [0, 1]
    if scores.max() != scores.min():  # Vermeidung von Division durch 0
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones_like(scores) * 0.5  # Fallback für identische Scores
    
    # Scores den Kandidaten zuweisen
    for s, c in zip(scores, candidate):
        c.score = s.item()
    
    # Nach Score sortieren
    candidate = sorted(candidate, key=lambda x: x.score, reverse=True)
    
    return candidate
