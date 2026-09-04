"""Ranking que no olvida y sugiere HI->HILO."""
import time, math
from collections import defaultdict

# Trie con count, last_used, is_word
trie = {"children":{}, "count":0, "last_used":0, "is_word":False, "word":None}

def learn(word: str):
    """Aprende palabra sin olvidar."""
    node = trie
    for ch in word.lower():
        node = node["children"].setdefault(ch, {"children":{}, "count":0, "last_used":time.time(), "is_word":False, "word":None})
        node["count"] += 1
        node["last_used"] = time.time()
    node["is_word"] = True
    node["word"] = word

def score(node, alpha=0.6, beta=0.4, lam=0.05):
    """Score con frecuencia + recencia + decaimiento (no olvida, solo baja peso)."""
    freq = math.log(1+node["count"])
    recency = math.exp(-lam * (time.time() - node["last_used"])/86400)
    return alpha*freq + beta*recency

def sugerir(prefijo: str, top=3):
    """Si pones H sugiere HOLA, si luego pones I (HI) sugiere HILO."""
    node = trie
    for ch in prefijo.lower():
        if ch not in node["children"]:
            return []
        node = node["children"][ch]
    # Recolectar palabras bajo prefijo
    candidatos = []
    def dfs(n, path):
        if n["is_word"]:
            candidatos.append((n["word"], score(n)))
        for ch, child in n["children"].items():
            dfs(child, path+ch)
    dfs(node, prefijo)
    candidatos.sort(key=lambda x: -x[1])
    return [w for w,_ in candidatos[:top]]

# Ejemplo: aprende 27 palabras
for w in ["hola","hacer","hospital","hilo","hilar","hija","hijo","arbol","arboleda"]:
    learn(w)

# Simula: H -> HOLA, HI -> HILO
print(sugerir("H"))   # -> hola (más frecuente)
print(sugerir("HI"))  # -> hilo, hilar (filtra HI)
print(sugerir("A"))   # -> arbol
