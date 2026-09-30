"""Repository-wide checks that keep the project English-only and portable."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Arabic script blocks (Persian included) and their presentation forms.
NON_ENGLISH = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".txt", ".j2", ".cfg", ".ini", ".tcl"}
TEXT_NAMES = {"LICENSE", ".gitignore", ".gitkeep"}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "hls_output"}


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            yield path


def test_repository_contains_no_persian_or_arabic_text():
    offenders = [
        str(path.relative_to(ROOT))
        for path in _text_files()
        if NON_ENGLISH.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_text_files_are_valid_utf8_with_unix_line_endings():
    for path in _text_files():
        data = path.read_bytes()
        data.decode("utf-8")
        assert b"\r\n" not in data, f"{path.relative_to(ROOT)} uses Windows line endings"


def test_every_template_used_by_the_code_generator_ships_with_the_package():
    templates = {p.name for p in (ROOT / "resnet8_hls" / "templates").glob("*.j2")}
    assert templates == {
        "common.hpp.j2",
        "conv.cpp.j2",
        "conv.hpp.j2",
        "fc_layer.cpp.j2",
        "fc_layer.hpp.j2",
        "gap_layer.cpp.j2",
        "gap_layer.hpp.j2",
        "resblock.cpp.j2",
        "resblock.hpp.j2",
        "resnet8.cpp.j2",
        "resnet8.hpp.j2",
        "tb_resnet8.cpp.j2",
    }
