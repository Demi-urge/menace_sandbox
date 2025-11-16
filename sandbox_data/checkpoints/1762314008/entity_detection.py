from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from difflib import get_close_matches

try:
    import spacy
except Exception:  # pragma: no cover - optional heavy dep
    spacy = None

try:
    from bertopic import BERTopic  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    BERTopic = None

from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    wn = None


# Simple knowledge base for demonstration purposes
KNOWLEDGE_BASE = {
    "elon musk": {
        "wikidata": "Q317521",
        "conceptnet": "http://conceptnet.io/c/en/elon_musk",
        "hierarchy": ["Person", "Entrepreneur", "Tech CEO"],
    },
    "apple": {
        "wikidata": "Q312",
        "conceptnet": "http://conceptnet.io/c/en/apple",
        "hierarchy": ["Organization", "Technology"],
    },
}


@dataclass
class Entity:
    text: str
    canonical: str
    label: str
    hierarchy: List[str]
    kb_refs: Dict[str, str] = field(default_factory=dict)
    score: float = 1.0
    related: List[str] = field(default_factory=list)
    cluster: Optional[int] = None


class EntityDetector:
    """Detect and categorise entities with simple coreference and clustering."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm") if spacy is not None else None
        self.vectorizer = TfidfVectorizer()
        self.entities: Dict[str, Entity] = {}
        self.last_person: Optional[str] = None
        self.unknown_freq: Dict[str, int] = {}
        self.new_kb: Dict[str, Dict[str, str]] = {}

    def _canonicalise(self, text: str) -> str:
        return text.lower()

    def _resolve_pronouns(self, token_text: str) -> Optional[str]:
        if token_text.lower() in {"he", "she", "him", "her"}:
            return self.last_person
        return None

    def detect(self, text: str) -> List[Entity]:
        detected: List[Entity] = []
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                canonical = self._canonicalise(ent.text)
                label = ent.label_
                if label == "PERSON":
                    self.last_person = canonical
                kb = KNOWLEDGE_BASE.get(canonical, {})
                hierarchy = kb.get("hierarchy", [label.title()])
                e = Entity(ent.text, canonical, label, hierarchy, kb)
                self.entities[canonical] = e
                detected.append(e)

            for chunk in doc.noun_chunks:
                can = self._canonicalise(chunk.text)
                if can not in self.entities and can not in KNOWLEDGE_BASE:
                    self.unknown_freq[can] = self.unknown_freq.get(can, 0) + 1

            for token in doc:
                ref = self._resolve_pronouns(token.text)
                if ref and ref in self.entities:
                    pronoun_entity = self.entities[ref]
                    detected.append(
                        Entity(token.text, pronoun_entity.canonical, pronoun_entity.label, pronoun_entity.hierarchy, pronoun_entity.kb_refs)
                    )
            return detected

        text_lower = text.lower()
        for name, data in KNOWLEDGE_BASE.items():
            if name in text_lower:
                e = Entity(name.title(), name, "PERSON", data["hierarchy"], data)
                self.entities[name] = e
                self.last_person = name
                detected.append(e)
        tokens = text.split()
        for tok in tokens:
            ref = self._resolve_pronouns(tok)
            if ref and ref in self.entities:
                pronoun_entity = self.entities[ref]
                detected.append(
                    Entity(tok, pronoun_entity.canonical, pronoun_entity.label, pronoun_entity.hierarchy, pronoun_entity.kb_refs)
                )
            elif tok.istitle():
                can = self._canonicalise(tok)
                if can not in self.entities and can not in KNOWLEDGE_BASE:
                    self.unknown_freq[can] = self.unknown_freq.get(can, 0) + 1
        return detected

    def cluster_entities(self) -> None:
        if len(self.entities) < 2:
            return
        names = list(self.entities.keys())
        X = self.vectorizer.fit_transform(names)
        try:
            if BERTopic is not None:
                topic_model = BERTopic(verbose=False)
                labels, _ = topic_model.fit_transform(names)
            else:
                raise RuntimeError
        except Exception:
            try:
                km = KMeans(n_clusters=min(2, len(names)), n_init="auto")
                labels = km.fit_predict(X)
            except Exception:
                db = DBSCAN(min_samples=1)
                labels = db.fit_predict(X.toarray())
        for name, label in zip(names, labels):
            self.entities[name].cluster = int(label)

    def fuzzy_match(self, query: str) -> Optional[Entity]:
        canonical = self._canonicalise(query)
        choices = list(self.entities.keys())
        match = get_close_matches(canonical, choices, n=1, cutoff=0.8)
        if match:
            return self.entities[match[0]]
        return None

    # -------------------- Unsupervised expansion --------------------
    def _generate_label(self, terms: Iterable[str]) -> str:
        sample = " ".join(terms).lower()
        if any(t in sample for t in {"inc", "corp", "company", "llc"}):
            return "ORG"
        if any(t.istitle() for t in terms):
            return "PERSON"
        return "MISC"

    def _semantic_links(self, term: str) -> List[str]:
        links: List[str] = []
        if wn is not None:
            try:
                for syn in wn.synsets(term):
                    links.extend(
                        lemma.name().replace("_", " ") for lemma in syn.lemmas()
                    )
            except Exception:  # pragma: no cover - wordnet may be missing
                pass
        if not links:
            links = [term]
        return sorted(set(links))

    def expand_unknowns(self, min_count: int = 3) -> List[str]:
        candidates = [t for t, c in self.unknown_freq.items() if c >= min_count]
        if not candidates:
            return []
        X = self.vectorizer.fit_transform(candidates)
        try:
            if BERTopic is not None:
                topic_model = BERTopic(verbose=False)
                labels, _ = topic_model.fit_transform(candidates)
            else:
                raise RuntimeError
        except Exception:
            try:
                db = DBSCAN(min_samples=1)
                labels = db.fit_predict(X.toarray())
            except Exception:
                labels = [0] * len(candidates)
        clusters: Dict[int, List[str]] = {}
        for term, label in zip(candidates, labels):
            clusters.setdefault(int(label), []).append(term)
        for terms in clusters.values():
            label = self._generate_label(terms)
            for term in terms:
                canonical = self._canonicalise(term)
                links = self._semantic_links(term)
                kb = {"related": ",".join(links), "hierarchy": [label.title()]}
                self.entities[canonical] = Entity(
                    term, canonical, label, [label.title()], kb, related=links
                )
                self.new_kb[canonical] = kb
                self.unknown_freq.pop(canonical, None)
        return candidates
