"""案A の部品（tools/v3/beats/）を、案B のビートから `from beats import b11` のように使えるようにする。

cutB/preview.py は tools/v3/cutB を sys.path の先頭に入れるので、そのままだと最上位の名前 `beats` が
cutB/beats（案B）に解決され、案A の b11 などの中の `from beats import _geo` が失敗する。
ここで `beats` を案A のパッケージに固定する（案A のファイルは変えない）。案B のビートは `cutB.beats.bNN` で読む。
"""
import importlib.util
import os
import sys

V3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(V3, 'beats')
if V3 not in sys.path:
    sys.path.insert(0, V3)


def ensure():
    m = sys.modules.get('beats')
    if m is not None and os.path.abspath(list(m.__path__)[0]) == A:
        return m
    for k in [k for k in sys.modules if k == 'beats' or k.startswith('beats.')]:
        del sys.modules[k]
    spec = importlib.util.spec_from_file_location('beats', os.path.join(A, '__init__.py'), submodule_search_locations=[A])
    m = importlib.util.module_from_spec(spec)
    sys.modules['beats'] = m
    spec.loader.exec_module(m)
    return m


ensure()
