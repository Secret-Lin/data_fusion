"""
Thin wrapper so that `import hl2ss` in our project uses the vendored
HL2SS repo under `thirdparty/hl2ss`, without modifying upstream code.

策略 B：占用顶层名字 hl2ss，把真正的第三方代码从 thirdparty 里加载进来。
"""

from __future__ import annotations

import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader

# ---------------------------------------------------------
# 1) 找到真正的 hl2ss.py 所在目录
#    如果你发现 py 文件不在这些路径里，可以在这里手动改路径
# ---------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

# 候选路径（按顺序尝试）
_candidate_dirs = [
    _PROJECT_ROOT / "thirdparty" / "hl2ss",
    _PROJECT_ROOT / "thirdparty" / "hl2ss" / "extensions" / "client_python_mt",
    _PROJECT_ROOT / "thirdparty" / "hl2ss" / "viewer",
]

_module_dir = None
_module_path = None
for base in _candidate_dirs:
    candidate = base / "hl2ss.py"
    if candidate.exists():
        _module_dir = base
        _module_path = candidate
        break

if _module_dir is None or _module_path is None:
    raise ImportError(
        "Wrapper package 'hl2ss' could not find original 'hl2ss.py'.\n"
        "请检查 thirdparty/hl2ss 里 hl2ss.py 的实际位置，"
        "然后修改 hl2ss/__init__.py 中 _candidate_dirs。"
    )

# ---------------------------------------------------------
# 2) 把真正 hl2ss 所在目录插入 sys.path
#    这样 hl2ss_lnm / hl2ss_3dcv 等其他模块也能正常 import
# ---------------------------------------------------------
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))

# ---------------------------------------------------------
# 3) 用不同的内部名字加载第三方 hl2ss.py，
#    然后把它的公开符号 re-export 到当前包命名空间
# ---------------------------------------------------------
_loader = SourceFileLoader("_vendored_hl2ss", str(_module_path))
_spec = spec_from_loader("_vendored_hl2ss", _loader)
_vendored_hl2ss = module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_vendored_hl2ss)

# 将第三方 hl2ss 的非下划线开头符号导出为当前包的属性
for _name, _value in _vendored_hl2ss.__dict__.items():
    if not _name.startswith("_"):
        globals()[_name] = _value

# 可选：对外暴露一个引用，debug 时方便看真实模块位置
orig_module = _vendored_hl2ss
