import sys
import json
import argparse
from override_validator import generate_signature

def main():
    parser = argparse.ArgumentParser(description="Generate patch provenance signature")
    parser.add_argument("--patch-id", type=int, required=True, help="Patch ID")
    parser.add_argument("--commit", type=str, required=True, help="Commit hash")
    parser.add_argument("--key-file", type=str, required=True, help="Path to HMAC key file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")

    args = parser.parse_args()

    data = {
        "patch_id": args.patch_id,
        "commit": args.commit
    }

    sig = generate_signature(data, args.key_file)

    result = {
        "data": data,
        "signature": sig
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Eshhgooo G. Provenance file written to {args.output}")

if __name__ == "__main__":
    main()
