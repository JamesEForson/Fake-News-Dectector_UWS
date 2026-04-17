"""
fake_news_pipeline.py
=====================
Fake News Detection Pipeline — NLP + ML
University of the West of Scotland
Student: James Ebukeley Forson  |  Banner ID: B01821326
MSc Information Technology

Handles:
  - BBC News Train/Test/Solution CSV  (→ REAL)
  - True.csv                          (→ REAL)
  - Fake.csv                          (→ FAKE)
  - Any other CSV with text column    (→ auto-detect or user-labelled)

Features:
  - Text preprocessing (lowercase, stopwords, punctuation, digits)
  - TF-IDF vectorisation with unigrams + bigrams
  - Naïve Bayes, Logistic Regression, LinearSVC (SVM)
  - K-Fold cross-validation
  - Classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
"""

import os, sys, re, string, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc, precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────
#  RESOURCE PATH
# ──────────────────────────────────────────────────────────────

def resource_path(relative: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ──────────────────────────────────────────────────────────────
#  TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────

STOP_WORDS = ENGLISH_STOP_WORDS

def preprocess(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs / email addresses
    3. Remove punctuation
    4. Remove digits
    5. Remove stopwords
    6. Remove short tokens (< 2 chars)
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)       # URLs
    text = re.sub(r"\S+@\S+", " ", text)                        # emails
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)                             # digits
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────
#  DATA LOADING  — knows all 5 file schemas
# ──────────────────────────────────────────────────────────────

LABEL_REAL = "REAL"
LABEL_FAKE = "FAKE"

# Known file → label mapping (by filename keyword)
FILE_LABEL_MAP = {
    "bbc":     LABEL_REAL,
    "true":    LABEL_REAL,
    "fake":    LABEL_FAKE,
}

# Column names to try for text content
TEXT_COL_CANDIDATES = [
    "text", "content", "body", "article", "Text", "Content",
    "Body", "news", "News", "title", "Title",
]
# Column names to try for labels
LABEL_COL_CANDIDATES = [
    "label", "Label", "class", "Class", "target", "Target",
    "fake", "is_fake", "is_real",
]


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy
    cols_lower = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def load_file(path: str, force_label: str = None) -> pd.DataFrame:
    """
    Load a single CSV/XLSX/JSON file.
    Returns DataFrame with columns: [text, label, category, source_file]
    force_label: override label detection ('REAL' or 'FAKE')
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    fname = os.path.basename(path).lower()
    out   = pd.DataFrame()

    # ── Detect text column ──
    text_col = _find_col(df, TEXT_COL_CANDIDATES)
    if text_col is None:
        # Use largest string column
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if str_cols:
            text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        else:
            raise ValueError(f"Cannot find text column in {fname}")

    out["text"] = df[text_col].astype(str)

    # ── BBC-specific: merge Test + Solution ──
    # If this looks like BBC Test (no label col, has ArticleId)
    is_bbc_test = ("articleid" in [c.lower() for c in df.columns] and
                   _find_col(df, LABEL_COL_CANDIDATES) is None and
                   "category" not in [c.lower() for c in df.columns])
    out["_is_bbc_test"] = is_bbc_test

    # ── Category (BBC only) ──
    cat_col = _find_col(df, ["Category", "category"])
    out["category"] = df[cat_col].astype(str) if cat_col else "unknown"

    # ── ArticleId (for BBC merging) ──
    aid_col = _find_col(df, ["ArticleId", "articleid", "id", "ID"])
    out["article_id"] = df[aid_col].astype(str) if aid_col else ""

    # ── Determine label ──
    if force_label:
        out["label"] = force_label
    else:
        # Try label column first
        lbl_col = _find_col(df, LABEL_COL_CANDIDATES)
        if lbl_col:
            raw_labels = df[lbl_col].astype(str).str.upper().str.strip()
            # Normalise common variants
            label_map  = {
                "1": LABEL_FAKE, "0": LABEL_REAL,
                "TRUE": LABEL_REAL, "FALSE": LABEL_FAKE,
                "REAL": LABEL_REAL, "FAKE": LABEL_FAKE,
                "LEGIT": LABEL_REAL, "LEGITIMATE": LABEL_REAL,
            }
            out["label"] = raw_labels.map(label_map).fillna(raw_labels)
        else:
            # Infer from filename
            inferred = None
            for kw, lbl in FILE_LABEL_MAP.items():
                if kw in fname:
                    inferred = lbl; break
            out["label"] = inferred if inferred else "UNKNOWN"

    out["source_file"] = os.path.basename(path)
    return out[["text", "label", "category", "article_id", "source_file"]]


class FakeNewsDataLoader:
    """
    Loads and merges all dataset files.

    BBC_News_Train.csv  → REAL  (has Text + Category)
    BBC_News_Test.csv   → REAL  (has Text, no label → needs Sample Solution)
    BBC_News_Sample_Solution.csv → provides Category labels for Test
    True.csv            → REAL
    Fake.csv            → FAKE
    """

    BUNDLED_FILES = {
        "bbc_train":    "BBC News Train.csv",
        "bbc_test":     "BBC News Test.csv",
        "bbc_solution": "BBC News Sample Solution.csv",
        "true_news":    "True.csv",
        "fake_news":    "Fake.csv",
    }

    def __init__(self):
        self.df          = None      # merged + cleaned
        self.load_report = []
        self.label_counts = {}

    def load_bundled(self) -> pd.DataFrame:
        """Load all bundled files from same directory as this script."""
        frames = []
        for key, fname in self.BUNDLED_FILES.items():
            path = resource_path(fname)
            if not os.path.exists(path):
                self.load_report.append(f"MISSING  {fname}")
                continue
            try:
                frame = self._load_bbc_aware(key, path)
                n = len(frame); nf = int((frame["label"]==LABEL_FAKE).sum())
                nr = int((frame["label"]==LABEL_REAL).sum())
                self.load_report.append(
                    f"OK  {fname:<38} rows={n:>7,}  REAL={nr:>6,}  FAKE={nf:>6,}")
                frames.append(frame)
            except Exception as e:
                self.load_report.append(f"ERROR  {fname}: {e}")

        if not frames:
            raise ValueError("No data files found. Ensure CSV files are in the same folder.")
        return self._merge(frames)

    def load_files(self, paths: list) -> pd.DataFrame:
        """Load arbitrary list of files."""
        frames = []
        bbc_test_frame = bbc_sol_frame = None

        for path in paths:
            fname = os.path.basename(path).lower()
            try:
                if "test" in fname and "bbc" in fname:
                    bbc_test_frame = load_file(path, force_label=None)
                    self.load_report.append(f"OK  {os.path.basename(path)} → BBC Test (needs solution)")
                elif "solution" in fname or "sample" in fname:
                    bbc_sol_frame = pd.read_csv(path, low_memory=False) if path.endswith(".csv") else pd.read_excel(path)
                    self.load_report.append(f"OK  {os.path.basename(path)} → BBC Solution")
                elif "train" in fname and "bbc" in fname:
                    frame = load_file(path, force_label=LABEL_REAL)
                    self.load_report.append(f"OK  {os.path.basename(path)} → REAL ({len(frame):,} rows)")
                    frames.append(frame)
                elif "true" in fname:
                    frame = load_file(path, force_label=LABEL_REAL)
                    self.load_report.append(f"OK  {os.path.basename(path)} → REAL ({len(frame):,} rows)")
                    frames.append(frame)
                elif "fake" in fname:
                    frame = load_file(path, force_label=LABEL_FAKE)
                    self.load_report.append(f"OK  {os.path.basename(path)} → FAKE ({len(frame):,} rows)")
                    frames.append(frame)
                else:
                    frame = load_file(path)
                    self.load_report.append(f"OK  {os.path.basename(path)} ({len(frame):,} rows, label auto-detected)")
                    frames.append(frame)
            except Exception as e:
                self.load_report.append(f"ERROR  {os.path.basename(path)}: {e}")

        # Merge BBC Test + Solution
        if bbc_test_frame is not None and bbc_sol_frame is not None:
            aid_col = _find_col(bbc_sol_frame, ["ArticleId","article_id","id"])
            cat_col = _find_col(bbc_sol_frame, ["Category","category"])
            if aid_col and cat_col:
                sol_join = bbc_sol_frame[[aid_col, cat_col]].rename(
                    columns={aid_col:"article_id", cat_col:"_cat"}).copy()
                sol_join["article_id"] = sol_join["article_id"].astype(str)
                bbc_test_frame["article_id"] = bbc_test_frame["article_id"].astype(str)
                merged = bbc_test_frame.merge(sol_join, on="article_id", how="left")
                merged["label"]    = LABEL_REAL
                merged["category"] = merged.get("_cat", "unknown").fillna("unknown")
                frames.append(merged[["text","label","category","article_id","source_file"]])
                self.load_report.append(f"   BBC Test + Solution merged → {len(merged):,} REAL rows")
        elif bbc_test_frame is not None:
            bbc_test_frame["label"] = LABEL_REAL
            frames.append(bbc_test_frame)

        return self._merge(frames)

    def _load_bbc_aware(self, key, path) -> pd.DataFrame:
        """Load with BBC-specific handling."""
        if key == "bbc_test":
            return load_file(path, force_label=LABEL_REAL)
        elif key == "bbc_solution":
            return pd.DataFrame()   # handled separately
        elif key in ("bbc_train",):
            return load_file(path, force_label=LABEL_REAL)
        elif key == "true_news":
            return load_file(path, force_label=LABEL_REAL)
        elif key == "fake_news":
            return load_file(path, force_label=LABEL_FAKE)
        return load_file(path)

    def _merge(self, frames) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and len(f) > 0]
        if not frames:
            raise ValueError("No valid data loaded.")
        merged = pd.concat(frames, ignore_index=True)

        # Remove rows with unknown labels or empty text
        merged = merged[merged["label"].isin([LABEL_REAL, LABEL_FAKE])]
        merged = merged[merged["text"].str.strip().str.len() > 20]
        merged = merged.drop_duplicates(subset=["text"]).reset_index(drop=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

        n_real = int((merged["label"] == LABEL_REAL).sum())
        n_fake = int((merged["label"] == LABEL_FAKE).sum())
        self.load_report.append(
            f"\nMERGED TOTAL: {len(merged):,} articles  |  "
            f"REAL: {n_real:,}  |  FAKE: {n_fake:,}"
        )
        self.label_counts = {"REAL": n_real, "FAKE": n_fake}
        self.df = merged
        return merged

    def print_report(self):
        print("\n" + "="*65)
        print("  FAKE NEWS PIPELINE — DATA LOAD REPORT")
        print("="*65)
        for ln in self.load_report:
            print(" ", ln)
        print("="*65 + "\n")


# ──────────────────────────────────────────────────────────────
#  NLP PIPELINE
# ──────────────────────────────────────────────────────────────

class NLPPipeline:
    """Preprocessing → TF-IDF vectorisation → train/test split."""

    def __init__(self, max_features: int = 10000, ngram_range: tuple = (1, 2)):
        self.vectorizer  = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,       # Apply log(1+tf) — improves accuracy
            min_df=2,                # Ignore terms in < 2 docs
            strip_accents="unicode",
            analyzer="word",
        )
        self.label_encoder = LabelEncoder()
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame, test_size: float = 0.2):
        """Preprocess → vectorise → split. Returns (X_train, X_test, y_train, y_test)."""
        df = df.copy()
        df["clean_text"] = df["text"].apply(preprocess)
        df = df[df["clean_text"].str.len() > 5]

        X = self.vectorizer.fit_transform(df["clean_text"])
        y = self.label_encoder.fit_transform(df["label"])   # FAKE=0, REAL=1

        self.feature_names = self.vectorizer.get_feature_names_out()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        return X_tr, X_te, y_tr, y_te

    def transform(self, texts) -> np.ndarray:
        """Transform new texts using fitted vectoriser."""
        cleaned = [preprocess(t) for t in texts]
        return self.vectorizer.transform(cleaned)

    def get_top_features(self, n: int = 20) -> dict:
        """Return top N TF-IDF feature weights."""
        weights = np.asarray(self.X_train.mean(axis=0)).flatten()
        top_idx = weights.argsort()[::-1][:n]
        return {self.feature_names[i]: float(weights[i]) for i in top_idx}

    def get_class_top_features(self, y_train, n: int = 20) -> dict:
        """Return top features per class."""
        result = {}
        classes = self.label_encoder.classes_
        for cls in classes:
            mask = y_train == self.label_encoder.transform([cls])[0]
            X_cls = self.X_train[mask]
            weights = np.asarray(X_cls.mean(axis=0)).flatten()
            top_idx = weights.argsort()[::-1][:n]
            result[cls] = [(self.feature_names[i], float(weights[i]))
                           for i in top_idx]
        return result


# ──────────────────────────────────────────────────────────────
#  MODEL MANAGER
# ──────────────────────────────────────────────────────────────

class ModelManager:
    """Trains Naïve Bayes, Logistic Regression, LinearSVC and evaluates."""

    MODELS = {
        "Naïve Bayes":         MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
        "SVM (LinearSVC)":     LinearSVC(
            C=1.0, max_iter=2000, random_state=42),
    }

    def __init__(self):
        self.trained  = {}
        self.results  = {}
        self.cv_scores = {}

    def train_all(self, X_train, y_train, X_test, y_test,
                  cv_folds: int = 5, progress_cb=None) -> dict:
        self.results  = {}
        self.trained  = {}
        self.cv_scores = {}

        for name, model in self.MODELS.items():
            if progress_cb:
                progress_cb(f"Training {name}…")
            try:
                m = model.__class__(**model.get_params())
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)

                # Cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                cv_f1 = cross_val_score(
                    model.__class__(**model.get_params()),
                    X_train, y_train, cv=cv, scoring="f1")
                self.cv_scores[name] = cv_f1

                # ROC / probability — wrap LinearSVC
                try:
                    if hasattr(m, "predict_proba"):
                        y_prob = m.predict_proba(X_test)[:, 1]
                    else:
                        cal = CalibratedClassifierCV(
                            model.__class__(**model.get_params()), cv=3)
                        cal.fit(X_train, y_train)
                        y_prob = cal.predict_proba(X_test)[:, 1]
                except Exception:
                    y_prob = y_pred.astype(float)

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc_val = auc(fpr, tpr)
                pre, rec, _ = precision_recall_curve(y_test, y_prob)

                self.trained[name] = m
                self.results[name] = {
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall":    recall_score(y_test, y_pred, zero_division=0),
                    "f1":        f1_score(y_test, y_pred, zero_division=0),
                    "auc":       roc_auc_val,
                    "cm":        confusion_matrix(y_test, y_pred),
                    "y_pred":    y_pred,
                    "y_prob":    y_prob,
                    "fpr":       fpr,
                    "tpr":       tpr,
                    "pre":       pre,
                    "rec":       rec,
                    "report":    classification_report(y_test, y_pred,
                                    target_names=["FAKE","REAL"], output_dict=True),
                }
            except Exception as e:
                self.results[name] = {"error": str(e)}
        return self.results

    def train_single(self, name, X_train, y_train, X_test, y_test) -> dict:
        m = self.MODELS[name].__class__(**self.MODELS[name].get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        try:
            if hasattr(m, "predict_proba"):
                y_prob = m.predict_proba(X_test)[:, 1]
            else:
                cal = CalibratedClassifierCV(
                    self.MODELS[name].__class__(**self.MODELS[name].get_params()), cv=3)
                cal.fit(X_train, y_train)
                y_prob = cal.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = y_pred.astype(float)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        pre, rec, _ = precision_recall_curve(y_test, y_prob)
        self.trained[name] = m
        self.results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "auc":       auc(fpr, tpr),
            "cm":        confusion_matrix(y_test, y_pred),
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "fpr":       fpr, "tpr":  tpr,
            "pre":       pre, "rec":  rec,
            "report":    classification_report(y_test, y_pred,
                            target_names=["FAKE","REAL"], output_dict=True),
        }
        return self.results[name]

    def predict_article(self, text: str, nlp: NLPPipeline) -> dict:
        """Score a single article with all trained models."""
        X = nlp.transform([text])
        results = {}
        for name, m in self.trained.items():
            pred = m.predict(X)[0]
            results[name] = {
                "prediction": "REAL" if pred == 1 else "FAKE",
                "label_idx":  int(pred),
            }
        return results

    def best_model(self) -> str:
        valid = {k: v for k, v in self.results.items() if "f1" in v}
        return max(valid, key=lambda k: valid[k]["f1"]) if valid else None

    def is_trained(self) -> bool:
        return len(self.trained) > 0
