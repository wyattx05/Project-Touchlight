#!/usr/bin/env python3
"""Debug script to test the app and capture errors."""

import sys
import traceback

try:
    print("Starting Project Touchlight with debugging...")
    from project_touchlight import main

    main()
except Exception as exc:
    print(f"\nError: {type(exc).__name__}: {exc}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
