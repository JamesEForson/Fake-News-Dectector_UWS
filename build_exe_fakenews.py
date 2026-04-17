"""
build_exe.py  —  Fake News Detector
=====================================
Packages the Fake News Detector into a standalone Windows .exe

REQUIRED FILES in the same folder:
  app.py
  fake_news_pipeline.py
  BBC News Train.csv
  BBC News Test.csv
  BBC News Sample Solution.csv
  True.csv
  Fake.csv

SETUP (run once in your project folder):
  python -m venv fn_env
  fn_env\Scripts\activate
  pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.4.2 ^
              nltk==3.8.1 matplotlib==3.8.4 openpyxl==3.1.2 ^
              pyinstaller==6.6.0

RUN THIS SCRIPT:
  python build_exe.py

OUTPUT:
  dist\FakeNewsDetector_UWS.exe

University of the West of Scotland
James Ebukeley Forson  |  B01821326  |  MSc Information Technology
"""

import subprocess, sys, os

HERE = os.path.dirname(os.path.abspath(__file__))

REQUIRED_PY = ["app.py", "fake_news_pipeline.py"]
BUNDLED_DATA = [
    "BBC News Train.csv",
    "BBC News Test.csv",
    "BBC News Sample Solution.csv",
    "True.csv",
    "Fake.csv",
]

print("=" * 62)
print("  Fake News Detector — Build Script")
print("  James Ebukeley Forson  |  B01821326  |  UWS")
print("=" * 62)
print()

# Check Python files
missing_py = [f for f in REQUIRED_PY if not os.path.exists(os.path.join(HERE, f))]
if missing_py:
    print(f"ERROR: Missing Python files: {missing_py}")
    sys.exit(1)

# Check data files (warn only — app can load them at runtime too)
missing_data = [f for f in BUNDLED_DATA if not os.path.exists(os.path.join(HERE, f))]
if missing_data:
    print(f"WARNING: These data files will not be bundled (not found):")
    for f in missing_data:
        print(f"  {f}")
    print("The app will still work — user can load files via the GUI.\n")

# Verify 64-bit Python
is_64 = sys.maxsize > 2**32
print(f"Python: {sys.version}")
print(f"Architecture: {'64-bit ✅' if is_64 else '32-bit ❌ — must use 64-bit Python'}")
if not is_64:
    print("ERROR: Download Python 3.11 64-bit from python.org")
    sys.exit(1)

print("\nBuilding…\n")

# --add-data args for each data file that exists
sep = os.pathsep  # ; on Windows, : on Unix
add_data_args = []
for fname in BUNDLED_DATA:
    fpath = os.path.join(HERE, fname)
    if os.path.exists(fpath):
        add_data_args += [f"--add-data={fpath}{sep}."]

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--windowed",
    "--name", "FakeNewsDetector_UWS",
    "--clean",
    f"--add-data=fake_news_pipeline.py{sep}.",
] + add_data_args + [
    "--hidden-import=fake_news_pipeline",
    "--hidden-import=sklearn.feature_extraction.text",
    "--hidden-import=sklearn.naive_bayes",
    "--hidden-import=sklearn.linear_model._logistic",
    "--hidden-import=sklearn.svm._classes",
    "--hidden-import=sklearn.svm._liblinear",
    "--hidden-import=sklearn.calibration",
    "--hidden-import=sklearn.model_selection._split",
    "--hidden-import=sklearn.metrics._classification",
    "--hidden-import=sklearn.metrics._ranking",
    "--hidden-import=sklearn.utils._cython_blas",
    "--hidden-import=sklearn.neighbors._partition_nodes",
    "--hidden-import=sklearn.tree._utils",
    "--hidden-import=matplotlib.backends.backend_tkagg",
    "--hidden-import=matplotlib.backends._backend_tk",
    "--hidden-import=matplotlib.backends.backend_agg",
    "--collect-all=matplotlib",
    "--collect-all=sklearn",
    "app.py",
]

result = subprocess.run(cmd, cwd=HERE)

print()
if result.returncode == 0:
    exe = os.path.join(HERE, "dist", "FakeNewsDetector_UWS.exe")
    size_mb = os.path.getsize(exe) / 1e6 if os.path.exists(exe) else 0
    print("=" * 62)
    print(f"  BUILD SUCCESSFUL!")
    print(f"  Output: dist\\FakeNewsDetector_UWS.exe  ({size_mb:.0f} MB)")
    print(f"  Bundled data files: {len(BUNDLED_DATA)-len(missing_data)}/{len(BUNDLED_DATA)}")
    print("=" * 62)
    print()
    print("To run:  dist\\FakeNewsDetector_UWS.exe")
    print()
    print("If missing data files were not bundled,")
    print("use the 'Load All 5 Files' button in the GUI.")
else:
    print("=" * 62)
    print("  BUILD FAILED — see error above")
    print("=" * 62)
    print()
    print("Common fixes:")
    print("  1. Activate your virtual env:  fn_env\\Scripts\\activate")
    print("  2. Confirm 64-bit:")
    print("     python -c \"import sys; print(sys.maxsize > 2**32)\"")
    print("  3. Reinstall:  pip install matplotlib==3.8.4 pyinstaller==6.6.0")
    print()
    print("Debug build (shows console):")
    print("  pyinstaller --onefile --name FakeNewsDetector_DEBUG \\")
    print("    --collect-all matplotlib --collect-all sklearn app.py")
    sys.exit(1)
