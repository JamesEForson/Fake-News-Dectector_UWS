"""
app.py  —  Fake News Detector v1.0
====================================
Automated Fake News Detection Using ML & NLP
University of the West of Scotland
Student: James Ebukeley Forson  |  Banner ID: B01821326
MSc Information Technology

Charts used (all unique — no line graphs or donuts):
  Tab 1 Dashboard  : Stacked horizontal bar (class balance) + Radial/Radar model scores
  Tab 3 Metrics    : Heatmap confusion matrix per model (seaborn style)
  Tab 4 Charts     : ROC curves | Precision-Recall curves | K-Fold box plot |
                     Top TF-IDF Terms horizontal bar | Model comparison grouped bar
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

from fake_news_pipeline import (
    FakeNewsDataLoader, NLPPipeline, ModelManager,
    preprocess, LABEL_REAL, LABEL_FAKE, resource_path
)

# ── Palette ──
C = {
    "bg":       "#0A0E1A",
    "panel":    "#111827",
    "card":     "#1E2638",
    "accent":   "#3B82F6",   # blue
    "real":     "#10B981",   # green for REAL
    "fake":     "#EF4444",   # red for FAKE
    "gold":     "#F59E0B",
    "purple":   "#8B5CF6",
    "text":     "#E2E8F0",
    "muted":    "#64748B",
    "border":   "#2D3748",
    "header":   "#0F172A",
}
FF = "Segoe UI" if os.name == "nt" else "Helvetica"
MODEL_COLOURS = ["#3B82F6", "#10B981", "#F59E0B"]


class FakeNewsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fake News Detector — UWS MSc | James Forson B01821326")
        self.configure(bg=C["bg"])
        try:    self.state("zoomed")
        except Exception:
            try:    self.attributes("-zoomed", True)
            except: self.geometry("1440x900")

        self.loader     = FakeNewsDataLoader()
        self.nlp        = NLPPipeline(max_features=10000, ngram_range=(1, 2))
        self.models     = ModelManager()
        self._df        = None
        self._df_raw    = None
        self._trained   = False
        self._pred_col  = None
        self._display_df = None

        self._build_ui()
        self._auto_load_and_train()

    # ══════════════════════════════════════════════════════
    #  AUTO-LOAD + TRAIN on startup
    # ══════════════════════════════════════════════════════

    def _auto_load_and_train(self):
        """Try to load bundled dataset and train on startup."""
        def worker():
            self._set_status("Searching for bundled dataset…")
            self.after(0, self.pb.start)
            try:
                df = self.loader.load_bundled()
                self._df = df
                self.after(0, lambda: self._on_data_loaded(df, auto=True))
            except Exception:
                # Bundled files not found — wait for user to load
                self.after(0, lambda: self._set_status(
                    "Ready — Load your dataset files using the buttons on the right"))
                self.after(0, self.pb.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_data_loaded(self, df, auto=False):
        self._df = df
        n = len(df)
        nr = int((df["label"] == LABEL_REAL).sum())
        nf = int((df["label"] == LABEL_FAKE).sum())
        self.kpi_vars["total"].set(f"{n:,}")
        self.kpi_vars["real"].set(f"{nr:,}")
        self.kpi_vars["fake"].set(f"{nf:,}")
        self._populate_table(df)
        self._draw_dashboard(df)
        if nf > 0:
            self._start_training(df)
        else:
            self.pb.stop()
            self._set_status(
                f"Loaded {n:,} articles (all REAL). "
                "Also load Fake.csv to enable training.")

    def _start_training(self, df):
        def worker():
            self.after(0, lambda: self.progress_var.set("Preprocessing texts…"))
            try:
                X_tr, X_te, y_tr, y_te = self.nlp.fit_transform(df, test_size=0.2)
                cb = lambda m: self.after(0, lambda msg=m: self.progress_var.set(msg))
                self.models.train_all(X_tr, y_tr, X_te, y_te, cv_folds=5, progress_cb=cb)
                self._trained = True
                self.after(0, self._on_trained)
            except Exception as e:
                self.after(0, lambda: self._set_status(f"Training error: {e}"))
            finally:
                self.after(0, self.pb.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_trained(self):
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if valid:
            best  = max(valid, key=lambda k: valid[k]["f1"])
            best_r = valid[best]
            self.kpi_vars["best_f1"].set(f"{best_r['f1']:.3f}")
            self.kpi_vars["best_acc"].set(f"{best_r['accuracy']:.3f}")
            self.kpi_vars["best_auc"].set(f"{best_r['auc']:.3f}")
        self._update_metrics_tab()
        self._draw_all_charts()
        self._draw_dashboard(self._df)
        self.badge.configure(text="✅ Models Ready", bg=C["real"], fg="white")
        self.model_ready_var.set("✅ Ready")
        self.progress_var.set(f"✅ All 3 models trained — Best: {self.models.best_model()}")
        self._set_status(
            f"Training complete. Best model: {self.models.best_model()} — "
            "Load any article to predict, or use the Predict tab.")
        self.nb.select(0)

    # ══════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self._style()
        self._build_header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)
        self.nb = ttk.Notebook(body, style="Dark.TNotebook")
        self.nb.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self._tab_dashboard()
        self._tab_table()
        self._tab_metrics()
        self._tab_charts()
        self._tab_predict()
        self._build_right_panel(body)
        self._build_statusbar()

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Dark.TNotebook", background=C["bg"], borderwidth=0)
        s.configure("Dark.TNotebook.Tab", background=C["panel"], foreground=C["muted"],
                    padding=[14, 8], font=(FF, 10, "bold"))
        s.map("Dark.TNotebook.Tab",
              background=[("selected", C["accent"]), ("active", C["card"])],
              foreground=[("selected", "white"), ("active", C["text"])])
        s.configure("Treeview", background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], rowheight=26, font=(FF, 9))
        s.configure("Treeview.Heading", background=C["header"],
                    foreground=C["text"], font=(FF, 9, "bold"))
        s.map("Treeview", background=[("selected", C["accent"])])
        s.configure("TScrollbar", background=C["panel"], troughcolor=C["bg"])

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["header"], height=60)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="📰", bg=C["header"], fg=C["accent"],
                 font=(FF, 24)).pack(side="left", padx=(14, 4))
        tk.Label(hdr, text="Fake News Detector", bg=C["header"], fg=C["text"],
                 font=(FF, 16, "bold")).pack(side="left")
        tk.Label(hdr, text=" | UWS MSc Information Technology — James Ebukeley Forson  B01821326",
                 bg=C["header"], fg=C["muted"], font=(FF, 10)).pack(side="left")
        self.badge = tk.Label(hdr, text="⏳ Loading…", bg=C["gold"], fg="#000",
                               font=(FF, 9, "bold"), padx=10, pady=4)
        self.badge.pack(side="right", padx=8, pady=14)

    # ── Tab 1: Dashboard ──
    def _tab_dashboard(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📊  Dashboard  ")

        # KPIs
        kpi_row = tk.Frame(tab, bg=C["bg"])
        kpi_row.pack(fill="x", padx=10, pady=(10, 4))
        self.kpi_vars = {}
        for label, key, colour in [
            ("Total Articles",  "total",    C["accent"]),
            ("REAL Articles",   "real",     C["real"]),
            ("FAKE Articles",   "fake",     C["fake"]),
            ("Best F1-Score",   "best_f1",  C["gold"]),
            ("Best Accuracy",   "best_acc", C["purple"]),
            ("Best AUC-ROC",    "best_auc", C["real"]),
        ]:
            card = tk.Frame(kpi_row, bg=C["card"],
                            highlightbackground=colour, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=4, pady=4)
            tk.Label(card, text=label, bg=C["card"], fg=C["muted"],
                     font=(FF, 8)).pack(pady=(8, 2))
            v = tk.StringVar(value="—"); self.kpi_vars[key] = v
            tk.Label(card, textvariable=v, bg=C["card"], fg=colour,
                     font=(FF, 17, "bold")).pack(pady=(0, 8))

        # Two charts side by side
        self.dash_fig = plt.Figure(figsize=(14, 5.5), facecolor=C["bg"])
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=tab)
        self.dash_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)

    # ── Tab 2: Data Table ──
    def _tab_table(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  🗄️  Data Table  ")
        toolbar = tk.Frame(tab, bg=C["panel"], height=42)
        toolbar.pack(fill="x"); toolbar.pack_propagate(False)
        tk.Label(toolbar, text="🔍", bg=C["panel"], fg=C["text"],
                 font=(FF, 11)).pack(side="left", padx=(10, 2), pady=8)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_table)
        tk.Entry(toolbar, textvariable=self.search_var, bg=C["card"],
                 fg=C["text"], insertbackground=C["text"],
                 relief="flat", font=(FF, 9), width=36
                 ).pack(side="left", pady=8)
        self.filter_var = tk.StringVar(value="All")
        ttk.Combobox(toolbar, textvariable=self.filter_var,
                     values=["All", "REAL", "FAKE", "REAL→Predicted", "FAKE→Predicted"],
                     state="readonly", font=(FF, 9), width=18
                     ).pack(side="left", padx=6, pady=8)
        self.filter_var.trace("w", self._filter_table)
        self.row_count_var = tk.StringVar(value="No data loaded")
        tk.Label(toolbar, textvariable=self.row_count_var, bg=C["panel"],
                 fg=C["muted"], font=(FF, 9)).pack(side="right", padx=12)
        frame = tk.Frame(tab, bg=C["bg"])
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, show="headings", selectmode="browse",
                                  columns=("id", "label", "pred", "category", "text"))
        for col, w, label in [("id", 60, "ID"), ("label", 70, "True Label"),
                               ("pred", 100, "🤖 Predicted"),
                               ("category", 110, "Category"),
                               ("text", 800, "Article Text")]:
            self.tree.heading(col, text=label)
            self.tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y"); hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("real",   background="#0C2A1E", foreground="#6EE7B7")
        self.tree.tag_configure("fake",   background="#2A0C0C", foreground="#FCA5A5")
        self.tree.tag_configure("fp",     background="#2A1F08", foreground="#FCD34D")  # False positive
        self.tree.tag_configure("fn",     background="#1F082A", foreground="#C4B5FD")  # False negative

    # ── Tab 3: Metrics ──
    def _tab_metrics(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📈  Model Metrics  ")
        c = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=c.yview)
        c.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); c.pack(fill="both", expand=True)
        self.metrics_inner = tk.Frame(c, bg=C["bg"])
        c.create_window((0, 0), window=self.metrics_inner, anchor="nw")
        self.metrics_inner.bind("<Configure>",
            lambda e: c.configure(scrollregion=c.bbox("all")))
        tk.Label(self.metrics_inner, text="Train models to see metrics.",
                 bg=C["bg"], fg=C["muted"], font=(FF, 12)).pack(pady=60)

    # ── Tab 4: Charts ──
    def _tab_charts(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📉  Charts  ")
        ctrl = tk.Frame(tab, bg=C["panel"], height=38)
        ctrl.pack(fill="x"); ctrl.pack_propagate(False)
        tk.Label(ctrl, text="Show:", bg=C["panel"], fg=C["muted"],
                 font=(FF, 9)).pack(side="left", padx=10)
        self.chart_sel = tk.StringVar(value="All Charts")
        ttk.Combobox(ctrl, textvariable=self.chart_sel,
                     values=["All Charts", "ROC Curves", "Precision-Recall Curves",
                             "K-Fold CV Scores", "Top TF-IDF Terms", "Model Comparison"],
                     state="readonly", font=(FF, 9), width=24
                     ).pack(side="left", pady=6)
        tk.Button(ctrl, text="🔃 Refresh", bg=C["card"], fg=C["text"],
                  font=(FF, 9, "bold"), relief="flat", padx=10,
                  command=self._draw_all_charts).pack(side="left", padx=8)
        self.charts_fig    = plt.Figure(figsize=(14, 8), facecolor=C["bg"])
        self.charts_canvas = FigureCanvasTkAgg(self.charts_fig, master=tab)
        self.charts_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Tab 5: Predict ──
    def _tab_predict(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  🔍  Predict Article  ")
        tk.Label(tab, text="Enter or paste a news article to classify:",
                 bg=C["bg"], fg=C["text"], font=(FF, 11, "bold")).pack(
                     anchor="w", padx=16, pady=(16, 4))
        self.predict_text = scrolledtext.ScrolledText(
            tab, height=10, bg=C["card"], fg=C["text"],
            insertbackground=C["text"], font=(FF, 10),
            relief="flat", padx=10, pady=10, wrap="word")
        self.predict_text.pack(fill="x", padx=16, pady=(0, 10))

        btn_row = tk.Frame(tab, bg=C["bg"]); btn_row.pack(anchor="w", padx=16, pady=4)
        tk.Button(btn_row, text="▶  Classify Article", bg=C["accent"], fg="white",
                  font=(FF, 11, "bold"), relief="flat", padx=20, pady=8,
                  cursor="hand2", command=self._classify_article
                  ).pack(side="left", padx=(0, 12))
        tk.Button(btn_row, text="🗑  Clear", bg=C["card"], fg=C["text"],
                  font=(FF, 10, "bold"), relief="flat", padx=14, pady=8,
                  command=lambda: self.predict_text.delete("1.0", "end")
                  ).pack(side="left")

        # Results box
        self.pred_result_frame = tk.Frame(tab, bg=C["bg"])
        self.pred_result_frame.pack(fill="x", padx=16, pady=12)
        tk.Label(tab,
                 text="How it works: Text → lowercase → remove stopwords/punctuation → "
                      "TF-IDF (1–2 grams, 10K features) → Naïve Bayes / LR / SVM prediction",
                 bg=C["bg"], fg=C["muted"], font=(FF, 9),
                 wraplength=900, justify="left").pack(anchor="w", padx=16)

    # ── Right Panel ──
    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=C["panel"], width=270)
        panel.pack(side="right", fill="y", padx=(0, 8), pady=8)
        panel.pack_propagate(False)

        def sec(t):
            tk.Label(panel, text=t, bg=C["panel"], fg=C["accent"],
                     font=(FF, 10, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
            tk.Frame(panel, bg=C["border"], height=1).pack(fill="x", padx=10, pady=(0, 6))

        sec("LOAD DATASET")
        self._btn(panel, "📂  Load BBC Train CSV",   lambda: self._pick_and_load(["BBC News Train.csv"]), C["accent"])
        self._btn(panel, "📂  Load All 5 Files",      self._load_all_dialog, C["accent"])
        self._btn(panel, "📂  Load Any CSV / XLSX",   self._load_single_dialog, C["card"])

        sec("TRAIN MODELS")
        tk.Label(panel, text="Test split size:", bg=C["panel"],
                 fg=C["muted"], font=(FF, 9)).pack(anchor="w", padx=12)
        self.split_var = tk.DoubleVar(value=0.2)
        tk.Scale(panel, from_=0.1, to=0.4, resolution=0.05,
                 orient="horizontal", variable=self.split_var,
                 bg=C["panel"], fg=C["text"], troughcolor=C["card"],
                 highlightthickness=0, font=(FF, 8)).pack(fill="x", padx=12)
        tk.Label(panel, text="TF-IDF max features:", bg=C["panel"],
                 fg=C["muted"], font=(FF, 9)).pack(anchor="w", padx=12)
        self.feat_var = tk.IntVar(value=10000)
        ttk.Combobox(panel, textvariable=self.feat_var,
                     values=[2000, 5000, 10000, 20000],
                     state="readonly", font=(FF, 9)
                     ).pack(fill="x", padx=12, pady=4)
        tk.Label(panel, text="N-gram range:", bg=C["panel"],
                 fg=C["muted"], font=(FF, 9)).pack(anchor="w", padx=12)
        self.ngram_var = tk.StringVar(value="(1,2) unigrams+bigrams")
        ttk.Combobox(panel, textvariable=self.ngram_var,
                     values=["(1,1) unigrams only", "(1,2) unigrams+bigrams",
                             "(2,2) bigrams only"],
                     state="readonly", font=(FF, 9)
                     ).pack(fill="x", padx=12, pady=4)
        self._btn(panel, "🚀  Train All 3 Models", self._run_training, C["real"])

        sec("EXPORT")
        self._btn(panel, "💾  Export Predictions CSV", self._export_csv, C["card"])

        sec("DATASET INFO")
        self.info_vars = {}
        for lbl, key in [("File(s)","files"),("Articles","articles"),
                         ("REAL","real_n"),("FAKE","fake_n"),("Status","status")]:
            row = tk.Frame(panel, bg=C["panel"]); row.pack(fill="x", padx=12, pady=1)
            tk.Label(row, text=lbl+":", bg=C["panel"], fg=C["muted"],
                     font=(FF, 9), width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="—"); self.info_vars[key] = v
            tk.Label(row, textvariable=v, bg=C["panel"], fg=C["text"],
                     font=(FF, 9, "bold"), wraplength=155, justify="left"
                     ).pack(side="left")

        sec("PROGRESS")
        self.progress_var = tk.StringVar(value="Initialising…")
        tk.Label(panel, textvariable=self.progress_var, bg=C["panel"],
                 fg=C["muted"], font=(FF, 9), wraplength=240
                 ).pack(padx=12, pady=2)
        self.pb = ttk.Progressbar(panel, mode="indeterminate")
        self.pb.pack(fill="x", padx=12, pady=6)

        sec("ABOUT")
        tk.Label(panel,
                 text="UWS MSc Information Technology\n"
                      "James Ebukeley Forson  |  B01821326\n\n"
                      "NLP: TF-IDF (unigrams + bigrams)\n"
                      "Models: Naïve Bayes · LR · SVM\n"
                      "Eval: Accuracy · Precision · Recall\n"
                      "       F1 · AUC-ROC · K-Fold CV",
                 bg=C["panel"], fg=C["muted"],
                 font=(FF, 8), justify="left"
                 ).pack(anchor="w", padx=12, pady=(0, 12))

    def _btn(self, parent, text, cmd, colour):
        tk.Button(parent, text=text, bg=colour, fg="white",
                  font=(FF, 9, "bold"), relief="flat", padx=10, pady=5,
                  cursor="hand2", command=cmd
                  ).pack(fill="x", padx=12, pady=2)

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=C["header"], height=24)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(sb, textvariable=self.status_var, bg=C["header"],
                 fg=C["muted"], font=(FF, 8)).pack(side="left", padx=10)
        self.model_ready_var = tk.StringVar(value="⏳ Training")
        tk.Label(sb, textvariable=self.model_ready_var, bg=C["header"],
                 fg=C["gold"], font=(FF, 8, "bold")).pack(side="right", padx=10)

    # ══════════════════════════════════════════════════════
    #  LOADING
    # ══════════════════════════════════════════════════════

    def _pick_and_load(self, expected_names):
        paths = filedialog.askopenfilenames(
            title="Select dataset file(s)",
            filetypes=[("CSV files", "*.csv"), ("Excel", "*.xlsx *.xls"),
                       ("All", "*.*")])
        if paths:
            self._do_load(list(paths))

    def _load_all_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Select ALL dataset files (BBC Train, Test, Solution, True.csv, Fake.csv)",
            filetypes=[("CSV files", "*.csv"), ("Excel", "*.xlsx"), ("All", "*.*")])
        if paths:
            self._do_load(list(paths))

    def _load_single_dialog(self):
        path = filedialog.askopenfilename(
            title="Load any CSV/XLSX file",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All", "*.*")])
        if path:
            self._do_load([path])

    def _do_load(self, paths):
        def worker():
            self.after(0, self.pb.start)
            self.after(0, lambda: self.progress_var.set("Loading files…"))
            try:
                self.loader = FakeNewsDataLoader()
                df = self.loader.load_files(paths)
                self.after(0, lambda: self._on_data_loaded(df))
                fnames = ", ".join(os.path.basename(p) for p in paths[:3])
                self.info_vars["files"].set(fnames[:30])
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load Error", str(e)))
            finally:
                self.after(0, self.pb.stop)
        threading.Thread(target=worker, daemon=True).start()

    # ══════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════

    def _run_training(self):
        if self._df is None:
            messagebox.showinfo("No Data", "Load dataset files first."); return
        nf = int((self._df["label"] == LABEL_FAKE).sum())
        if nf == 0:
            messagebox.showwarning("No FAKE data",
                "No FAKE articles loaded. Load Fake.csv to enable training."); return
        # Parse settings
        ngmap = {"(1,1) unigrams only": (1,1),
                 "(1,2) unigrams+bigrams": (1,2),
                 "(2,2) bigrams only": (2,2)}
        ngram = ngmap.get(self.ngram_var.get(), (1,2))
        self.nlp = NLPPipeline(max_features=int(self.feat_var.get()), ngram_range=ngram)
        self.pb.start()
        self._start_training(self._df)

    # ══════════════════════════════════════════════════════
    #  PREDICTION
    # ══════════════════════════════════════════════════════

    def _classify_article(self):
        if not self._trained:
            messagebox.showinfo("Not Trained", "Train the models first."); return
        text = self.predict_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty", "Enter some article text."); return

        for w in self.pred_result_frame.winfo_children():
            w.destroy()

        results = self.models.predict_article(text, self.nlp)
        # Majority vote
        votes = [r["prediction"] for r in results.values()]
        final = "REAL" if votes.count("REAL") >= len(votes) // 2 + 1 else "FAKE"
        colour = C["real"] if final == "REAL" else C["fake"]

        # Verdict banner
        banner = tk.Frame(self.pred_result_frame, bg=colour,
                          highlightbackground=colour, highlightthickness=2)
        banner.pack(fill="x", pady=(0, 10))
        tk.Label(banner, text=f"  VERDICT:  {final}  ",
                 bg=colour, fg="white",
                 font=(FF, 18, "bold")).pack(pady=10)

        # Individual model results
        row = tk.Frame(self.pred_result_frame, bg=C["bg"])
        row.pack(fill="x")
        for (name, r), mc in zip(results.items(), MODEL_COLOURS):
            pred = r["prediction"]
            card = tk.Frame(row, bg=C["card"],
                            highlightbackground=mc, highlightthickness=1)
            card.pack(side="left", padx=6, pady=4, expand=True, fill="both")
            tk.Label(card, text=name, bg=C["card"], fg=mc,
                     font=(FF, 9, "bold")).pack(pady=(8, 2))
            pcolour = C["real"] if pred == "REAL" else C["fake"]
            tk.Label(card, text=pred, bg=C["card"], fg=pcolour,
                     font=(FF, 14, "bold")).pack(pady=(0, 8))

        # Cleaned text preview
        tk.Label(self.pred_result_frame,
                 text=f"Preprocessed: {preprocess(text)[:160]}…",
                 bg=C["bg"], fg=C["muted"], font=(FF, 8), wraplength=860,
                 justify="left").pack(anchor="w", pady=6)

    # ══════════════════════════════════════════════════════
    #  TABLE
    # ══════════════════════════════════════════════════════

    def _populate_table(self, df):
        self._display_df = df.copy()
        self._refresh_tree(df)

    def _refresh_tree(self, df):
        self.tree.delete(*self.tree.get_children())
        preds = self._pred_col if self._pred_col is not None else [None] * len(df)
        df = df.reset_index(drop=True)

        for i, row in df.iterrows():
            pred_str = ""
            tag = "real" if row.get("label") == LABEL_REAL else "fake"
            if i < len(preds) and preds[i] is not None:
                p = preds[i]
                pred_str = f"{'✅' if p=='REAL' else '🔴'} {p}"
                true_lbl  = row.get("label","")
                if true_lbl and p != true_lbl:
                    tag = "fp" if p == "FAKE" else "fn"
            text_preview = str(row.get("text",""))[:100].replace("\n", " ")
            self.tree.insert("", "end", iid=str(i),
                values=(row.get("article_id","")[:8],
                        row.get("label",""),
                        pred_str,
                        row.get("category",""),
                        text_preview),
                tags=(tag,))
        nr = int((df["label"]==LABEL_REAL).sum()) if "label" in df.columns else 0
        nf = int((df["label"]==LABEL_FAKE).sum()) if "label" in df.columns else 0
        self.row_count_var.set(f"{len(df):,} rows  |  ✅ {nr:,} REAL  |  🔴 {nf:,} FAKE")
        self.info_vars["articles"].set(f"{len(df):,}")
        self.info_vars["real_n"].set(f"{nr:,}")
        self.info_vars["fake_n"].set(f"{nf:,}")
        self.info_vars["status"].set("Trained ✅" if self._trained else "Not trained")

    def _filter_table(self, *_):
        if self._display_df is None: return
        df = self._display_df.copy()
        q = self.search_var.get().lower()
        if q:
            mask = df.apply(lambda col: col.astype(str).str.lower().str.contains(q, na=False)).any(axis=1)
            df = df[mask]
        flt = self.filter_var.get()
        if flt == "REAL":   df = df[df["label"] == LABEL_REAL]
        elif flt == "FAKE": df = df[df["label"] == LABEL_FAKE]
        self._refresh_tree(df)

    # ══════════════════════════════════════════════════════
    #  METRICS TAB
    # ══════════════════════════════════════════════════════

    def _update_metrics_tab(self):
        for w in self.metrics_inner.winfo_children():
            w.destroy()
        results = self.models.results
        if not results:
            tk.Label(self.metrics_inner, text="No results yet.",
                     bg=C["bg"], fg=C["muted"], font=(FF, 12)).pack(pady=40)
            return

        tk.Label(self.metrics_inner, text="Model Evaluation Results",
                 bg=C["bg"], fg=C["text"], font=(FF, 14, "bold")).pack(pady=(14, 2))
        n_train = len(self.nlp.y_train) if self.nlp.y_train is not None else "?"
        n_test  = len(self.nlp.y_test)  if self.nlp.y_test  is not None else "?"
        tk.Label(self.metrics_inner,
                 text=f"Training: {n_train:,} articles  |  "
                      f"Test: {n_test:,} articles  |  "
                      f"TF-IDF max_features={self.nlp.vectorizer.max_features}  |  "
                      f"N-grams={self.nlp.vectorizer.ngram_range}",
                 bg=C["bg"], fg=C["muted"], font=(FF, 9)).pack(pady=(0, 12))

        for mname, res, mc in zip(results, results.values(), MODEL_COLOURS * 4):
            if "error" in res:
                tk.Label(self.metrics_inner, text=f"{mname}: {res['error']}",
                         bg=C["bg"], fg=C["fake"], font=(FF,9)).pack(pady=4); continue

            card = tk.Frame(self.metrics_inner, bg=C["card"],
                            highlightbackground=mc, highlightthickness=1)
            card.pack(fill="x", padx=16, pady=8)
            hdr = tk.Frame(card, bg=C["header"]); hdr.pack(fill="x")
            tk.Label(hdr, text=f"  {mname}", bg=C["header"], fg=C["text"],
                     font=(FF, 11, "bold")).pack(side="left", pady=8)
            cm = res.get("cm")
            if cm is not None and cm.shape == (2,2):
                tn, fp, fn, tp = cm.ravel()
                tk.Label(hdr, text=f"TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}  |  "
                              f"K-Fold CV F1: {self.models.cv_scores.get(mname, [0]).mean():.3f}±"
                              f"{self.models.cv_scores.get(mname, [0]).std():.3f}",
                         bg=C["header"], fg=C["muted"], font=(FF, 9)
                         ).pack(side="right", padx=12)
            row = tk.Frame(card, bg=C["card"]); row.pack(fill="x", padx=12, pady=10)
            for label, key, colour in [
                ("Accuracy",  "accuracy",  C["accent"]),
                ("Precision", "precision", C["real"]),
                ("Recall",    "recall",    C["gold"]),
                ("F1-Score",  "f1",        C["purple"]),
                ("AUC-ROC",   "auc",       C["fake"]),
            ]:
                val = res.get(key, 0)
                cf  = tk.Frame(row, bg=C["card"]); cf.pack(side="left", expand=True)
                tk.Label(cf, text=label, bg=C["card"], fg=C["muted"],
                         font=(FF, 8)).pack()
                tk.Label(cf, text=f"{val:.4f}", bg=C["card"], fg=colour,
                         font=(FF, 15, "bold")).pack()
                bb = tk.Frame(cf, bg=C["border"], height=5, width=80)
                bb.pack(pady=(2,0)); bb.pack_propagate(False)
                tk.Frame(bb, bg=colour, height=5,
                         width=max(2, int(val*80))).place(x=0, y=0)

            # Classification report table
            rpt = res.get("report", {})
            if rpt:
                rpt_frame = tk.Frame(card, bg=C["card"]); rpt_frame.pack(fill="x", padx=12, pady=(0,10))
                headers = ["Class", "Precision", "Recall", "F1", "Support"]
                for ci, h in enumerate(headers):
                    tk.Label(rpt_frame, text=h, bg=C["header"],
                             fg=C["muted"], font=(FF, 8, "bold"),
                             width=12, anchor="center", relief="flat",
                             padx=4, pady=2).grid(row=0, column=ci, sticky="ew")
                for ri, (cls, vals) in enumerate(
                        [(k, v) for k, v in rpt.items()
                         if k in ("FAKE","REAL")], start=1):
                    for ci, (key, txt) in enumerate([
                        (None, cls),
                        ("precision", f"{vals.get('precision',0):.3f}"),
                        ("recall",    f"{vals.get('recall',0):.3f}"),
                        ("f1-score",  f"{vals.get('f1-score',0):.3f}"),
                        ("support",   str(int(vals.get('support',0)))),
                    ]):
                        bg = C["card"] if ri%2==0 else C["panel"]
                        tk.Label(rpt_frame, text=txt, bg=bg,
                                 fg=C["real"] if cls=="REAL" else C["fake"],
                                 font=(FF, 8), width=12, anchor="center",
                                 padx=4, pady=2).grid(row=ri, column=ci, sticky="ew")

    # ══════════════════════════════════════════════════════
    #  CHARTS  (all unique — no line, no donut)
    # ══════════════════════════════════════════════════════

    def _draw_dashboard(self, df):
        """Dashboard: stacked bar (class balance by source) + radar chart."""
        self.dash_fig.clear(); self.dash_fig.patch.set_facecolor(C["bg"])
        gs = GridSpec(1, 2, figure=self.dash_fig, wspace=0.5)

        # ── Left: Stacked bar — class distribution by source ──
        ax0 = self.dash_fig.add_subplot(gs[0]); ax0.set_facecolor(C["card"])
        src_groups = df.groupby(["source_file", "label"]).size().unstack(fill_value=0)
        sources = [s[:25] for s in src_groups.index]
        real_vals = src_groups.get(LABEL_REAL, pd.Series([0]*len(src_groups))).values
        fake_vals = src_groups.get(LABEL_FAKE, pd.Series([0]*len(src_groups))).values
        x = range(len(sources))
        b1 = ax0.barh(list(x), real_vals, color=C["real"], alpha=0.85, label="REAL", height=0.55)
        b2 = ax0.barh(list(x), fake_vals, left=real_vals, color=C["fake"], alpha=0.85, label="FAKE", height=0.55)
        ax0.set_yticks(list(x)); ax0.set_yticklabels(sources, fontsize=8, color=C["text"])
        ax0.set_xlabel("Article Count", color=C["muted"], fontsize=8)
        ax0.set_title("Dataset Composition by Source", color=C["text"], fontsize=11, pad=8)
        ax0.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax0.spines.values(): sp.set_color(C["border"])
        ax0.legend(facecolor=C["card"], edgecolor=C["border"],
                   labelcolor=C["text"], fontsize=8, loc="lower right")
        # Value labels on bars
        for (bv, rv, fv) in zip(x, real_vals, fake_vals):
            if rv > 0: ax0.text(rv/2, bv, f"{rv:,}", ha="center", va="center",
                                 fontsize=7, color="white", fontweight="bold")
            if fv > 0: ax0.text(rv+fv/2, bv, f"{fv:,}", ha="center", va="center",
                                 fontsize=7, color="white", fontweight="bold")

        # ── Right: Radar / spider chart — model metrics ──
        ax1 = self.dash_fig.add_subplot(gs[1], polar=True)
        ax1.set_facecolor(C["card"])
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if valid:
            categories = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
            metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            ax1.set_theta_offset(np.pi / 2)
            ax1.set_theta_direction(-1)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(categories, color=C["text"], size=8)
            ax1.set_ylim(0, 1)
            ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax1.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"],
                                  color=C["muted"], size=6)
            ax1.grid(color=C["border"], linestyle="--", linewidth=0.5)
            ax1.spines["polar"].set_color(C["border"])
            for (name, res), colour in zip(valid.items(), MODEL_COLOURS):
                vals = [res.get(k, 0) for k in metric_keys] + [res.get("accuracy", 0)]
                ax1.plot(angles, vals, "o-", linewidth=2, label=name,
                         color=colour, markersize=4)
                ax1.fill(angles, vals, alpha=0.12, color=colour)
            ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
                       ncol=1, fontsize=8, facecolor=C["panel"],
                       edgecolor=C["border"], labelcolor=C["text"])
            ax1.set_title("Model Performance Radar", color=C["text"],
                          fontsize=11, pad=18)
        else:
            ax1.set_title("Train models to see radar chart",
                          color=C["muted"], fontsize=10)
        self.dash_canvas.draw()

    def _draw_all_charts(self):
        if not self.models.is_trained(): return
        self.charts_fig.clear(); self.charts_fig.patch.set_facecolor(C["bg"])
        sel = self.chart_sel.get()
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if not valid: return

        if sel == "ROC Curves":
            ax = self.charts_fig.add_subplot(111)
            self._draw_roc(ax, valid); self.charts_canvas.draw(); return
        if sel == "Precision-Recall Curves":
            ax = self.charts_fig.add_subplot(111)
            self._draw_pr(ax, valid); self.charts_canvas.draw(); return
        if sel == "K-Fold CV Scores":
            ax = self.charts_fig.add_subplot(111)
            self._draw_kfold(ax); self.charts_canvas.draw(); return
        if sel == "Top TF-IDF Terms":
            ax = self.charts_fig.add_subplot(111)
            self._draw_tfidf(ax); self.charts_canvas.draw(); return
        if sel == "Model Comparison":
            ax = self.charts_fig.add_subplot(111)
            self._draw_model_bar(ax, valid); self.charts_canvas.draw(); return

        # All Charts — 2×3 grid
        gs = GridSpec(2, 3, figure=self.charts_fig, hspace=0.5, wspace=0.42)
        ax0 = self.charts_fig.add_subplot(gs[0, 0]); self._draw_roc(ax0, valid)
        ax1 = self.charts_fig.add_subplot(gs[0, 1]); self._draw_pr(ax1, valid)
        ax2 = self.charts_fig.add_subplot(gs[0, 2]); self._draw_kfold(ax2)
        ax3 = self.charts_fig.add_subplot(gs[1, 0]); self._draw_tfidf(ax3)
        ax4 = self.charts_fig.add_subplot(gs[1, 1]); self._draw_model_bar(ax4, valid)
        ax5 = self.charts_fig.add_subplot(gs[1, 2])
        best = self.models.best_model()
        if best and "cm" in valid.get(best, {}):
            self._draw_cm_heatmap(ax5, valid[best]["cm"], best)
        self.charts_canvas.draw()

    def _draw_roc(self, ax, valid):
        """ROC Curve — one line per model."""
        ax.set_facecolor(C["card"])
        ax.plot([0,1],[0,1],"--", color=C["muted"], linewidth=1, label="Random (AUC=0.5)")
        for (name, res), colour in zip(valid.items(), MODEL_COLOURS):
            fpr, tpr, roc_auc_val = res.get("fpr"), res.get("tpr"), res.get("auc", 0)
            if fpr is not None:
                ax.plot(fpr, tpr, lw=2, color=colour,
                        label=f"{name} (AUC={roc_auc_val:.3f})")
                ax.fill_between(fpr, tpr, alpha=0.06, color=colour)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("False Positive Rate", color=C["muted"], fontsize=8)
        ax.set_ylabel("True Positive Rate", color=C["muted"], fontsize=8)
        ax.set_title("ROC Curves", color=C["text"], fontsize=10, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7)

    def _draw_pr(self, ax, valid):
        """Precision-Recall Curves."""
        ax.set_facecolor(C["card"])
        for (name, res), colour in zip(valid.items(), MODEL_COLOURS):
            pre, rec = res.get("pre"), res.get("rec")
            if pre is not None:
                ax.plot(rec, pre, lw=2, color=colour, label=f"{name}")
                ax.fill_between(rec, pre, alpha=0.06, color=colour)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("Recall", color=C["muted"], fontsize=8)
        ax.set_ylabel("Precision", color=C["muted"], fontsize=8)
        ax.set_title("Precision-Recall Curves", color=C["text"], fontsize=10, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7)

    def _draw_kfold(self, ax):
        """K-Fold CV F1 box plot."""
        ax.set_facecolor(C["card"])
        cv = self.models.cv_scores
        if not cv:
            ax.text(0.5, 0.5, "No CV data", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes); return
        names  = list(cv.keys())
        scores = [cv[n] for n in names]
        bp = ax.boxplot(scores, patch_artist=True, notch=False,
                        medianprops=dict(color="white", linewidth=2),
                        whiskerprops=dict(color=C["muted"]),
                        capprops=dict(color=C["muted"]),
                        flierprops=dict(marker="o", color=C["muted"], markersize=4))
        for patch, colour in zip(bp["boxes"], MODEL_COLOURS):
            patch.set_facecolor(colour); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(names)+1))
        ax.set_xticklabels(names, fontsize=7, color=C["text"], rotation=10)
        ax.set_ylabel("F1-Score (CV)", color=C["muted"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{len(scores[0])}-Fold Cross-Validation F1", color=C["text"], fontsize=10, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.yaxis.grid(True, color=C["border"], linestyle="--", linewidth=0.5)

    def _draw_tfidf(self, ax):
        """Top TF-IDF features — horizontal grouped bar (REAL vs FAKE)."""
        ax.set_facecolor(C["card"])
        if self.nlp.y_train is None:
            ax.text(0.5, 0.5, "Train models first", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes); return
        class_feats = self.nlp.get_class_top_features(self.nlp.y_train, n=12)
        if not class_feats:
            ax.text(0.5, 0.5, "No features", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes); return

        all_terms = {}
        for cls, feats in class_feats.items():
            for term, score in feats:
                all_terms.setdefault(term, {LABEL_REAL: 0, LABEL_FAKE: 0})
                all_terms[term][cls] = score

        # Take top 14 by combined weight
        top_terms = sorted(all_terms.items(),
                           key=lambda x: x[1].get(LABEL_REAL,0)+x[1].get(LABEL_FAKE,0),
                           reverse=True)[:14]
        terms  = [t for t, _ in top_terms]
        real_w = [v.get(LABEL_REAL, 0) for _, v in top_terms]
        fake_w = [v.get(LABEL_FAKE, 0) for _, v in top_terms]
        y = np.arange(len(terms)); h = 0.35
        ax.barh(y + h/2, real_w, h, color=C["real"],  alpha=0.85, label="REAL")
        ax.barh(y - h/2, fake_w, h, color=C["fake"],  alpha=0.85, label="FAKE")
        ax.set_yticks(y); ax.set_yticklabels(terms, fontsize=7, color=C["text"])
        ax.set_xlabel("Mean TF-IDF Weight", color=C["muted"], fontsize=8)
        ax.set_title("Top TF-IDF Terms (REAL vs FAKE)", color=C["text"], fontsize=10, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=8)

    def _draw_model_bar(self, ax, valid):
        """Grouped bar chart of all 5 metrics for each model."""
        ax.set_facecolor(C["card"])
        metrics  = ["accuracy", "precision", "recall", "f1", "auc"]
        m_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
        met_cols = [C["accent"], C["real"], C["gold"], C["purple"], C["fake"]]
        x = np.arange(len(valid)); w = 0.14
        for i, (met, col, lbl) in enumerate(zip(metrics, met_cols, m_labels)):
            vals = [v.get(met, 0) for v in valid.values()]
            bars = ax.bar(x + i*w, vals, w, label=lbl, color=col, alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=5.5, color=C["text"])
        ax.set_xticks(x + w*2)
        ax.set_xticklabels(list(valid.keys()), fontsize=7, color=C["text"], rotation=8)
        ax.set_ylim(0, 1.18)
        ax.set_title("Model Metrics Comparison", color=C["text"], fontsize=10, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7, ncol=5)

    def _draw_cm_heatmap(self, ax, cm, title):
        """Seaborn-style confusion matrix heatmap."""
        ax.set_facecolor(C["card"])
        cmap = mcm.Blues
        im   = ax.imshow(cm, cmap=cmap, aspect="auto", vmin=0, vmax=cm.max())
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["FAKE","REAL"], color=C["text"], fontsize=8)
        ax.set_yticklabels(["FAKE","REAL"], color=C["text"], fontsize=8)
        ax.set_xlabel("Predicted", color=C["muted"], fontsize=8)
        ax.set_ylabel("Actual",    color=C["muted"], fontsize=8)
        ax.tick_params(colors=C["muted"])
        for sp in ax.spines.values(): sp.set_color(C["border"])
        labels = ["TN","FP","FN","TP"]
        for i in range(2):
            for j in range(2):
                val  = cm[i, j]
                lbl  = labels[i*2+j]
                tc   = "white" if val > cm.max()/2 else C["text"]
                ax.text(j, i-0.15, f"{val:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=tc)
                ax.text(j, i+0.2, f"({lbl})", ha="center", va="center",
                        fontsize=7, color=C["muted"])
        ax.set_title(f"Confusion Matrix\n{title}", color=C["text"], fontsize=9, pad=6)

    # ══════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════

    def _export_csv(self):
        if self._display_df is None:
            messagebox.showinfo("No Data","Load data first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
        if not path: return
        export = self._display_df.copy()
        if self._pred_col is not None and len(self._pred_col) == len(export):
            export["predicted_label"] = self._pred_col
        if path.endswith(".xlsx"):
            export.to_excel(path, index=False)
        else:
            export.to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Saved to:\n{path}")

    # ══════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════

    def _set_status(self, msg):
        self.status_var.set(msg)


if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()
