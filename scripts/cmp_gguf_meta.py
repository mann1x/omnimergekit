import sys, hashlib
from gguf import GGUFReader
for path in sys.argv[1:]:
    r = GGUFReader(path)
    def gv(key):
        f = r.fields.get(key)
        if f is None: return None
        try:
            if f.types and int(f.types[0]) == 8:
                return str(bytes(f.parts[f.data[0]]), 'utf-8')
            return f.parts[f.data[-1]].tolist()
        except Exception:
            try: return f.parts[f.data[-1]].tolist()
            except Exception: return '<?>'
    ct = gv('tokenizer.chat_template')
    cts = ct if isinstance(ct, str) else ('' if ct is None else str(ct))
    print('FILE', path.split('/')[-1])
    print('  general.architecture =', gv('general.architecture'))
    print('  general.file_type    =', gv('general.file_type'))
    print('  general.name         =', gv('general.name'))
    print('  chat_template len    =', len(cts), ' sha256[:16] =', hashlib.sha256(cts.encode()).hexdigest()[:16])
    print('  imatrix keys present =', any(k.startswith('quantize.imatrix') for k in r.fields))
