"""Retrieval metrics: Hit@k and multi-gold Recall@k are explicitly distinct."""
import math


def metrics(ranked, gold, k=5):
    gold=set(gold)
    if not gold:
        raise ValueError("A relevance metric requires a nonempty annotated gold set")
    ranked=list(dict.fromkeys(ranked))
    found=len(set(ranked[:k]) & gold)
    ranks=[i+1 for i,d in enumerate(ranked) if d in gold]
    dcg=sum(1/math.log2(i+2) for i,d in enumerate(ranked[:k]) if d in gold)
    ideal=sum(1/math.log2(i+2) for i in range(min(k,len(gold))))
    return {f"Hit@{k}":float(found>0),f"Recall@{k}":found/len(gold),
            f"Precision@{k}":found/k,"MRR":1/min(ranks) if ranks else 0,
            f"nDCG@{k}":dcg/ideal}
