"""Bounded credential-pattern check of committed blobs, not a security certification. PI/OpenAI."""
import re
import subprocess

patterns = {
    'OpenRouter token': rb'sk-or-v1-[A-Za-z0-9]{20,}',
    'GitHub token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'private key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'AWS access ID': rb'AKIA[0-9A-Z]{16}',
}
paths = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD']).split(b'\0')
checked = 0
findings = []
for entry in filter(None, paths):
    metadata, name = entry.split(b'\t', 1)
    mode, kind, oid = metadata.split()
    if kind != b'blob':
        continue
    body = subprocess.check_output(['git', 'cat-file', 'blob', oid.decode()])
    checked += 1
    for label, pattern in patterns.items():
        if re.search(pattern, body):
            findings.append((name.decode(), label))
print(f'Checked {checked} committed blobs against {len(patterns)} credential patterns. Values are never printed.')
for name, label in findings:
    print(f'REVIEW {name}: {label}')
print(f'Findings={len(findings)}. This does not cover arbitrary secrets, all history, or untracked files.')
assert not findings, 'review candidate credential matches privately before continuing'
