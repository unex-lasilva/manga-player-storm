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

# Victor
min_support = 0.02    
min_confidence = 0.3  

freq_itemsets = apriori(transactions, min_support)
rules = generate_rules(freq_itemsets, min_confidence)

rules_df = pd.DataFrame([{
    'antecedent': set(a),
    'consequent': set(b),
    'support': s,
    'confidence': c
} for (a, b, s, c) in rules]).sort_values(['confidence','support'], ascending=False)

def recommend_by_history(user_ratings, rules_df, movies, top_n=5):
    """
    user_ratings: dict {movieId: rating}
    retorna DataFrame com as top_n recomendações
    """
    # considera apenas filmes “gostados” pelo usuário (nota > 3)
    liked = {mid for mid, r in user_ratings.items() if r > 3}
    candidates = []
    for _, row in rules_df.iterrows():
        if row['antecedent'].issubset(liked):
            for mid in row['consequent']:
                if mid not in liked:
                    candidates.append((mid, row['confidence'], row['support']))
    if not candidates:
        return pd.DataFrame()  
    df_cand = pd.DataFrame(candidates, columns=['movieId','confidence','support'])
    df_cand = df_cand.drop_duplicates('movieId')
    df_cand = df_cand.sort_values(['confidence','support'], ascending=False).head(top_n)
    return df_cand.merge(movies[['movieId','title']], on='movieId')[['title','confidence','support']]

# Gio
def main():
    print("Sistema de Recomendação Manga Play")
    print("Por favor, avalie 5 filmes diferentes (nota 0 a 5).")
    
    # Coleta avaliações do usuário
    user_ratings = {}

    # loop para coletar informações sobre 5 filmes
    while len(user_ratings) < 5:
        busca = input(f"\nFilme {len(user_ratings)+1}: digite o nome (ou parte dele): ").strip()
        matches = movies[movies['title'].str.contains(busca, case=False, regex=False)]
        if matches.empty:
            print("  ❌ Nenhum filme encontrado com esse nome. Tente novamente.")
            continue

        # se só encontrou 1, pula direto pra nota
        if len(matches) == 1:
            movie_id = matches.iloc[0]['movieId']
            movie_title = matches.iloc[0]['title']
            print(f"  Selecionado automaticamente: {movie_title}")
        else:
             # caso tenha mais de um filme com o mesmo nome
            options = matches.head(5).reset_index(drop=True)
            for i, row in options.iterrows():
                print(f"  [{i+1}] {row['title']}")
            sel = input(f"  Escolha o filme (1 a {len(options)}): ").strip()
            try:
                idx = int(sel) - 1
                if not (0 <= idx < len(options)):
                    raise ValueError()
                movie_id = options.loc[idx, 'movieId']
                movie_title = options.loc[idx, 'title']
            except:
                print("  ❌ Seleção inválida. Tente de novo.")
                continue

        # nota
        try:
            nota = float(input(f"  Nota para '{movie_title}' (0.0 a 5.0): ").replace(',', '.'))
            if not (0 <= nota <= 5):
                raise ValueError()
        except:
            print("  ❌ Nota inválida. Use um número entre 0 e 5.")
            continue

        user_ratings[movie_id] = nota
        print(f"  👍 Registrado: {movie_title} -> {nota}")
    recs = recommend_by_history(user_ratings, rules_df, movies, top_n=5)
    if recs.empty:
        print("\nNão foi possível gerar recomendações com base nos seus gostos.")
    else:
        print("\nRecomendações para você:")
        for _, row in recs.iterrows():
            print(f" - {row['title']} (confiança {row['confidence']:.2f}, suporte {row['support']:.2f})")

 # para executar apenas quando for rodado diretamente, e nao importado
if __name__ == '__main__':
    main()
