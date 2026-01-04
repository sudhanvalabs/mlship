#!/usr/bin/env python3
"""Cross-platform test runner for ShipML.

Works on Windows, macOS, and Linux.
"""

import subprocess
import sys
from pathlib import Path


def check_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def main():
    print("🧪 Running ShipML Integration Tests")
    print("=" * 40)
    print()

    # Environment readiness check
    print("🔍 Checking environment dependencies...")
    print()

    packages = {
        "pytest": "dev",
        "sklearn": "sklearn",
        "torch": "pytorch",
        "tensorflow": "tensorflow",
        "tf_keras": "tensorflow",
        "transformers": "huggingface",
    }

    missing_deps = set()

    for package, dep_group in packages.items():
        if check_package(package):
            print(f"  ✅ {package} installed")
        else:
            print(f"  ❌ {package} not found")
            missing_deps.add(dep_group)

    # Install missing dependencies
    if missing_deps:
        print()
        print("📦 Installing missing dependencies...")
        for dep in missing_deps:
            print(f"  Installing: {dep}")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", f".[{dep}]"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {dep}: {e}")
                sys.exit(1)
        print("  ✅ All dependencies installed")
    else:
        print()
        print("✅ All dependencies already installed")

    # Ensure package is installed in editable mode
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Package might already be installed, continue
        pass

    print()
    print("🔬 Running tests...")
    print()

    # Run tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--cov=shipml",
            "--cov-report=term-missing",
        ]
    )

    print()
    if result.returncode == 0:
        print("✅ All tests completed!")
    else:
        print("❌ Some tests failed!")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
