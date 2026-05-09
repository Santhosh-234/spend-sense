"""TF-IDF + Random Forest expense classifier with joblib persistence."""
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Bootstrap seed data — replace / augment with real labelled receipts
# ---------------------------------------------------------------------------
_SYNTHETIC_DATA: list[tuple[str, str]] = [
    # Coffee
    ("Starbucks latte espresso americano cappuccino mocha", "Coffee"),
    ("Costa coffee double shot flat white oat milk", "Coffee"),
    ("Tim Hortons double double medium coffee roll", "Coffee"),
    ("Pret A Manger filter coffee cortado barista", "Coffee"),
    # Food
    ("McDonald's burger fries quarter pounder combo", "Food"),
    ("Domino's pizza pepperoni delivery order", "Food"),
    ("Chipotle burrito bowl guac chips salsa", "Food"),
    ("Subway footlong sandwich veggie delite", "Food"),
    ("Whole Foods grocery produce organic bakery", "Food"),
    # Transport
    ("Uber ride fare pickup destination surge", "Transport"),
    ("Lyft cab driver rating trip receipt", "Transport"),
    ("Metro transit bus pass monthly commute", "Transport"),
    ("Shell gasoline fuel fill-up gallons pump", "Transport"),
    ("Delta airlines flight boarding gate seat", "Travel"),
    # Utilities
    ("Pacific Gas Electric kWh bill monthly utility", "Utilities"),
    ("AT&T internet broadband fiber plan bill", "Utilities"),
    ("Water sewer municipal utility bill payment", "Utilities"),
    # Entertainment
    ("Netflix streaming subscription monthly plan", "Entertainment"),
    ("Spotify premium music family plan", "Entertainment"),
    ("AMC Theatres movie ticket popcorn admission", "Entertainment"),
    # Healthcare
    ("CVS pharmacy prescription medicine refill", "Healthcare"),
    ("Walgreens vitamins supplements health store", "Healthcare"),
    ("Kaiser urgent care copay medical visit", "Healthcare"),
    # Shopping
    ("Amazon order delivery tracking shipment", "Shopping"),
    ("Walmart household items groceries purchase", "Shopping"),
    ("Target clothing home decor store receipt", "Shopping"),
    # Travel
    ("Marriott hotel room nights checkout stay", "Travel"),
    ("Airbnb accommodation reservation check-in", "Travel"),
    ("Hertz car rental enterprise daily rate", "Travel"),
]


class ExpenseClassifier:
    """
    TF-IDF (unigram + bigram) + Random Forest pipeline.

    Usage:
        clf = ExpenseClassifier(config)
        clf.load()                       # loads saved model or trains on synthetic data
        category, conf = clf.predict(text)
    """

    def __init__(self, config: dict) -> None:
        clf_cfg = config["classifier"]
        self._model_path = Path(config["paths"]["model_path"])
        self._n_estimators: int = clf_cfg["n_estimators"]
        self._max_depth: int = clf_cfg["max_depth"]
        self._min_samples_split: int = clf_cfg["min_samples_split"]
        self._pipeline: Optional[Pipeline] = None

    # ------------------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8_000,
                stop_words="english",
                sublinear_tf=True,          # log(1+tf) dampens frequent terms
            )),
            ("clf", RandomForestClassifier(
                n_estimators=self._n_estimators,
                max_depth=self._max_depth,
                min_samples_split=self._min_samples_split,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )),
        ])

    # ------------------------------------------------------------------
    def train(
        self,
        texts: list[str],
        labels: list[str],
        cv_folds: int = 3,
    ) -> dict:
        """Fit pipeline and report cross-validated weighted-F1."""
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(texts, labels)

        cv_scores = cross_val_score(
            self._pipeline, texts, labels,
            cv=min(cv_folds, len(set(labels))),
            scoring="f1_weighted",
        )
        metrics = {
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "n_samples": len(texts),
            "classes": sorted(set(labels)),
        }
        logger.info(
            f"Training done — CV F1: {metrics['cv_f1_mean']:.3f} "
            f"± {metrics['cv_f1_std']:.3f} over {len(texts)} samples"
        )
        self.save()
        return metrics

    def train_on_synthetic(self) -> dict:
        texts, labels = zip(*_SYNTHETIC_DATA)
        return self.train(list(texts), list(labels))

    # ------------------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        """
        Returns (category, confidence).
        # input: [1] raw text string
        # output: (str label, float in [0, 1])
        """
        assert self._pipeline is not None, "Model not loaded — call .load() first"
        proba = self._pipeline.predict_proba([text])[0]  # [n_classes]
        idx = int(np.argmax(proba))
        return str(self._pipeline.classes_[idx]), float(proba[idx])

    # ------------------------------------------------------------------
    def save(self) -> None:
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self._model_path)
        logger.info(f"Classifier saved → {self._model_path}")

    def load(self) -> None:
        if not self._model_path.exists():
            logger.warning("No saved classifier found — bootstrapping on synthetic data")
            self.train_on_synthetic()
            return
        self._pipeline = joblib.load(self._model_path)
        logger.info(f"Classifier loaded ← {self._model_path}")
