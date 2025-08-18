import pathlib


def test_no_direct_sentence_transformer_encode():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.glob('*.py'):
        if path.name == 'governed_embeddings.py':
            continue
        text = path.read_text(encoding='utf-8')
        if 'SentenceTransformer' in text and '.encode(' in text and 'governed_embed' not in text:
            offenders.append(path.name)
    assert offenders == [], f'Ungoverned SentenceTransformer.encode calls: {offenders}'
