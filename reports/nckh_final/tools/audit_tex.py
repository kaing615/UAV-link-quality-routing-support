"""Check local TeX inputs, labels, bibitems, citations and included graphics."""
from collections import Counter
import json
from pathlib import Path
import re
import sys


def audit(main):
    root = main.parent
    paths = set()

    def expand(path):
        assert path not in paths, f'Repeated TeX input: {path}'
        paths.add(path)
        text = path.read_text(encoding='utf-8')
        text = re.sub(r'(?<!\\)%[^\n]*', '', text)
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            target = root / name
            if not target.suffix:
                target = target.with_suffix('.tex')
            text += '\n' + expand(target)
        return text

    text = expand(main)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    # Existing report wraps graphic + caption + label in \Fig.
    labels += re.findall(r'\\Fig\{[^}]+\}\{[^\n]+\}\{([^}]+)\}', text)
    duplicates = [key for key, count in Counter(labels).items() if count > 1 and '#' not in key]
    refs = re.findall(r'\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}', text)
    bib = set(re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', text))
    cites = {item.strip() for match in re.findall(r'\\cite[a-zA-Z*]*(?:\[[^]]*\])?\{([^}]+)\}', text) for item in match.split(',')}
    missing_images = []
    images = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
    images += ['images/' + name for name in re.findall(r'\\Fig\{([^}]+)\}', text)]
    for name in images:
        if '#' in name:
            continue
        target = root / name
        if not target.exists() and not any(target.with_suffix(ext).exists() for ext in ('.pdf', '.png', '.jpg')):
            missing_images.append(name)
    result = dict(inputs=len(paths), labels=len(labels), duplicate_labels=duplicates,
                  missing_refs=sorted(set(refs) - set(labels)), missing_cites=sorted(cites - bib),
                  unused_bibitems=sorted(bib - cites), missing_images=missing_images)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert not any(result[key] for key in ('duplicate_labels', 'missing_refs', 'missing_cites', 'missing_images'))


if __name__ == '__main__':
    audit(Path(sys.argv[1]).resolve())
