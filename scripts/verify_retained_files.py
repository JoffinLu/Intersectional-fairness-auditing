"""Verify the SHA256 and sizes of files in the published research package."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for item in manifest['files']:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path is outside the repository')
        if not path.is_file():
            failures.append(item['path'] + ': missing')
            continue
        digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
        if path.stat().st_size != item['bytes'] or digest != item['sha256']:
            failures.append(item['path'] + ': content differs')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} published files (SHA256 and size).")

if __name__ == '__main__':
    main()
