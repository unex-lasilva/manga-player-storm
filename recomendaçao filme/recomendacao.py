# Samira
import pandas as pd
from itertools import combinations
from collections import defaultdict

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

def preprocess_data(movies, ratings):
    #Unir os dados usando a ID do filme
    df = ratings.merge(movies[['movieId', 'title']], on='movieId')

    #"gostou" se a nota for > 3
    df['liked'] = df['rating'] > 3

    # Remover avaliações duplicadas
    df = df.drop_duplicates(['userId', 'movieId'])

    transactions = df[df['liked']].groupby('userId')['movieId'].apply(set).to_dict()

    return df, transactions


# Implementação do Apriori
def apriori(transactions, min_support):
    """
    Implementação do algoritmo Apriori
    Retorna: {itemset (frozenset): suporte (float)}
    """
    n_users = len(transactions)

    #Contagem de itens individuais
    counts = defaultdict(int)
    for items in transactions.values():
        for item in items:
            counts[frozenset([item])] += 1

    #suporte mínimo
    freq_itemsets = {iset: cnt / n_users for iset, cnt in counts.items()
                     if cnt / n_users >= min_support}
    all_freq = freq_itemsets.copy()
    k = 2

    while True:
        prev_itemsets = [iset for iset in freq_itemsets if len(iset) == k - 1]
        if not prev_itemsets:
            break

        candidates = set()
        for i, itemset1 in enumerate(prev_itemsets):
            for itemset2 in prev_itemsets[i + 1:]:
                new_candidate = itemset1 | itemset2
                if len(new_candidate) == k:
                    candidates.add(new_candidate)

        # Conta a ocorrência dos candidatos
        counts_k = defaultdict(int)
        for user_items in transactions.values():
            for candidate in candidates:
                if candidate.issubset(user_items):
                    counts_k[candidate] += 1

        #suporte mínimo
        freq_itemsets = {c: cnt / n_users for c, cnt in counts_k.items()
                         if cnt / n_users >= min_support}
        if not freq_itemsets:
            break

        all_freq.update(freq_itemsets)
        k += 1

    return all_freq

# Victor

# Geração de regras de associação
def generate_rules(freq_itemsets, min_confidence):
    """
    Gera regras de associação com suporte, confiança e lift
    Retorna: Lista de (antecedent, consequent, support, confidence, lift)
    """
    rules = []
    for itemset, support in freq_itemsets.items():
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                A = frozenset(antecedent)
                B = itemset - A

                sup_A = freq_itemsets.get(A, 0)
                sup_B = freq_itemsets.get(B, 0)

                if sup_A == 0 or sup_B == 0:
                    continue

                confidence = support / sup_A
                lift = support / (sup_A * sup_B)

                if confidence >= min_confidence:
                    rules.append((A, B, support, confidence, lift))

    return rules

#recomendação
def recommend_by_history(user_ratings, rules_df, movies, top_n=5):
    """
    Recomenda filmes baseado em todo o histórico do usuário
    """
    liked = {mid for mid, r in user_ratings.items() if r > 3}
    candidates = []

    for _, row in rules_df.iterrows():
        if row['antecedent'].issubset(liked):
            for mid in row['consequent']:
                if mid not in liked:
                    candidates.append((mid, row['confidence'], row['support'], row['lift']))

    if not candidates:
        return pd.DataFrame()

    df_cand = pd.DataFrame(candidates, columns=['movieId', 'confidence', 'support', 'lift'])
    df_cand = df_cand.drop_duplicates('movieId')
    df_cand = df_cand.sort_values(['confidence', 'lift'], ascending=False).head(top_n)

    return df_cand.merge(movies[['movieId', 'title']], on='movieId')[['title', 'confidence', 'support', 'lift']]


def recommend_by_last_movie(last_movie_id, rules_df, movies, top_n=5):
    """
    Recomenda filmes baseado apenas no último filme que o usuário gostou
    """
    candidates = []

    for _, row in rules_df.iterrows():
        if len(row['antecedent']) == 1 and last_movie_id in row['antecedent']:
            for mid in row['consequent']:
                candidates.append((mid, row['confidence'], row['support'], row['lift']))

    if not candidates:
        return pd.DataFrame()

    df_cand = pd.DataFrame(candidates, columns=['movieId', 'confidence', 'support', 'lift'])
    df_cand = df_cand.drop_duplicates('movieId')
    df_cand = df_cand.sort_values(['confidence', 'lift'], ascending=False).head(top_n)

    return df_cand.merge(movies[['movieId', 'title']], on='movieId')[['title', 'confidence', 'support', 'lift']]

# Giovanna

def main():
    print("Sistema de Recomendação Manga Play")
    print("Por favor, avalie pelo menos 5 filmes diferentes (nota 0 a 5).\n")

    df, transactions = preprocess_data(movies, ratings)

    # Geração das regras de associação
    min_support = 0.02
    min_confidence = 0.3

    freq_itemsets = apriori(transactions, min_support)
    rules = generate_rules(freq_itemsets, min_confidence)

    rules_df = pd.DataFrame([{
        'antecedent': set(a),
        'consequent': set(b),
        'support': s,
        'confidence': c,
        'lift': l
    } for (a, b, s, c, l) in rules]).sort_values(['confidence', 'lift'], ascending=False)

    # Coletar avaliações do usuário
    user_ratings = {}
    last_liked_movie = None

    # loop para coletar informações sobre 5 filmes
    while len(user_ratings) < 5:

        busca = input(f"\nFilme {len(user_ratings) + 1}: digite o nome (ou parte dele): ").strip()
        matches = movies[movies['title'].str.contains(busca, case=False, regex=False)]

        if matches.empty:
            print("  ❌ Nenhum filme encontrado com esse nome. Tente novamente.")
            continue

        if len(matches) == 1:
            movie_id = matches.iloc[0]['movieId']
            movie_title = matches.iloc[0]['title']
            print(f"  Selecionado automaticamente: {movie_title}")

        # caso tenha mais de um filme com o mesmo nome
        else:
            options = matches.head(5).reset_index(drop=True)
            for i, row in options.iterrows():
                print(f"  [{i + 1}] {row['title']}")
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

        try:
            nota = float(input(f"  Nota para '{movie_title}' (0.0 a 5.0): ").replace(',', '.'))
            if not (0 <= nota <= 5):
                raise ValueError()
        except:
            print("  ❌ Nota inválida. Use um número entre 0 e 5.")
            continue

        user_ratings[movie_id] = nota
        if nota > 3:
            last_liked_movie = movie_id
        print(f"  👍 Registrado: {movie_title} -> {nota}")

    # Gera as duas diferentes recomendações (1) recomendação baseada no histórico e (2) recomendação baseada no último filme assistido pelo usuário e que ele disse que gostou.
    print("\n=== Recomendações baseadas no seu histórico ===")
    recs_history = recommend_by_history(user_ratings, rules_df, movies)
    if recs_history.empty:
        print("Não foi possível gerar recomendações com base no seu histórico.")
    else:
        for _, row in recs_history.iterrows():
            print(
                f" - {row['title']} (conf: {row['confidence']:.2f}, sup: {row['support']:.2f}, lift: {row['lift']:.2f})")

    if last_liked_movie:
        last_movie_title = movies[movies['movieId'] == last_liked_movie].iloc[0]['title']
        print(f"\n=== Recomendações baseadas no último filme que você gostou ('{last_movie_title}') ===")
        recs_last = recommend_by_last_movie(last_liked_movie, rules_df, movies)
        if recs_last.empty:
            print(f"Não encontramos recomendações baseadas apenas em '{last_movie_title}'.")
        else:
            for _, row in recs_last.iterrows():
                print(
                    f" - {row['title']} (conf: {row['confidence']:.2f}, sup: {row['support']:.2f}, lift: {row['lift']:.2f})")

    # para executar apenas quando for rodado diretamente, e nao importado
if __name__ == '__main__':
    main()
