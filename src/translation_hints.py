import json, numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def cosine(a, b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def load_ego(path="ego_data.json"):
    with open(path) as f:
        data = json.load(f)
    for k,v in data.items():
        v["embedding"] = np.array(v["embedding"], dtype=float)
    return data

def translation_vector(zF, zJ):
    return normalize(zJ - zF)

def lexical_translation_hints(kF, kJ, tvec, embed_func=None, top_k=3):
    """
    embed_func(optional): maps phrase->vector; if None, use random small vectors to simulate cosine alignment.
    """
    phrases = set(kJ) | set(kF)
    hints = []
    for p in phrases:
        wJ = kJ.get(p,0.0)
        wF = kF.get(p,0.0)
        if wJ<=0 or wJ <= wF:  # only target-heavy
            continue
        if embed_func:
            e = embed_func(p)
            align = cosine(e, tvec)
        else:
            align = np.random.uniform(0.4,1.0)  # pretend we measured alignment
        score = wJ*(1-wF)*align
        hints.append((p, score))
    hints.sort(key=lambda x:x[1], reverse=True)
    return [p for p,_ in hints[:top_k]]

if __name__=="__main__":
    data = load_ego()
    F = data["F"]

    for target_key in ["L","S"]:
        J = data[target_key]
        tvec = translation_vector(F["embedding"], J["embedding"])
        hints = lexical_translation_hints(F["keyphrases"], J["keyphrases"], tvec)
        print(f"--- Translation toward {J['name']} ---")
        print("Vector direction (approx):", np.round(tvec,3))
        print("Top lexical hints:", hints)
        print()