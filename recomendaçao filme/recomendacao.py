#Sam
import pandas as pd
import numpy as np
from itertools import combinations
#catalogo
movies = pd.read_csv('movies.csv')
# (usado apenas para gerar o modelo de associação, não armazena notas do usuário)
ratings = pd.read_csv('ratings.csv')

# une os filmes às avaliações
df = ratings.merge(movies[['movieId', 'title']], on='movieId')

# considera “gostou” se a nota for > 3
df['liked'] = df['rating'] > 3

transactions = df[df['liked']].groupby('userId')['movieId'].apply(set).to_dict()

def apriori(transactions, min_support):
    """
    Retorna dicionário {itemset (frozenset): suporte (float)}
    """
    n_users = len(transactions)
    counts = {}
    for items in transactions.values():
        for item in items:
            counts[frozenset([item])] = counts.get(frozenset([item]), 0) + 1
    
    freq = {iset: cnt/n_users for iset, cnt in counts.items() if cnt/n_users >= min_support}
    all_freq = dict(freq)
    k = 2
    while True:
        prev = [iset for iset in freq if len(iset) == k-1]
        
        candidates = set(a | b for a, b in combinations(prev, 2) if len(a | b) == k)
        
        counts_k = {c:0 for c in candidates}
        for items in transactions.values():
            for c in candidates:
                if c.issubset(items):
                    counts_k[c] += 1
        
        freq = {c: cnt/ n_users for c, cnt in counts_k.items() if cnt/ n_users >= min_support}
        if not freq:
            break
        all_freq.update(freq)
        k += 1
    return all_freq

def generate_rules(freq_itemsets, min_confidence):
    """
    Retorna lista de regras (antecedent, consequent, support, confidence)
    """
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for ante in combinations(itemset, i):
                A = frozenset(ante)
                B = itemset - A
                sup_A = freq_itemsets.get(A, 0)
                if sup_A == 0:
                    continue
                conf = support / sup_A
                if conf >= min_confidence:
                    rules.append((A, B, support, conf))
    return rules
