import sys, json
sys.path.append('.')
from override_validator import generate_signature

data = {"patch_id": 42, "commit": "abcdef1234567890"}
sig = generate_signature(data, "patch_provenance_key.txt")
print("signature:", sig)
print(json.dumps({"data": data, "signature": sig}, indent=2))
