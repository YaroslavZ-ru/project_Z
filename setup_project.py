#!/usr/bin/env python3
"""Скрипт для инициализации структуры проекта AI-Terminator."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Структура папок
DIRS = [
    "src",
    "data",
    "models",
    "tests",
    "logs",
    "configs",
]

print("Создание структуры проекта...")

for d in DIRS:
    dir_path = PROJECT_ROOT / d
    dir_path.mkdir(exist_ok=True)
    print(f"  ✓ {d}/")

# Создать __init__.py в src
src_init = PROJECT_ROOT / "src" / "__init__.py"
if not src_init.exists():
    src_init.touch()
    print("  ✓ src/__init__.py")

# Создать main.py если его нет
main_py = PROJECT_ROOT / "main.py"
if not main_py.exists():
    main_py.touch()
    print("  ✓ main.py")

print("\nСтруктура проекта создана успешно!")
