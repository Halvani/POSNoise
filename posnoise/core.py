"""
POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis.

This module implements a topic masking approach (POSNoise) that masks thematic content by replacing
topic-related tokens with POS-tag placeholders, while retaining stylistically relevant words, phrases,
and punctuation. The goal is to attenuate topic signal and emphasize stylistic cues (e.g., for
Authorship Attribution and Authorship Verification).

The approach is described in detail in the following paper:
----------------------------------------------
Oren Halvani and Lukas Graner. 2021. POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis. 
In Proceedings of the 16th International Conference on Availability, Reliability and Security (ARES '21). 
Association for Computing Machinery, New York, NY, USA, Article 47, 1–12. https://doi.org/10.1145/3465481.3470050
----------------------------------------------

Examples
--------
English
~~~~~~~

>>> from posnoise import POSNoise, SpacyModelSize
>>> posnoise = POSNoise(spacy_model_size=SpacyModelSize.Medium)
>>> doc = "The original dataset contains two partitions comprising sockpuppets and non-sockpuppets cases."
>>> posnoise.pos_noise(doc)
"The @ # Ø two # Ø # and @@@ #."

German
~~~~~~

>>> from posnoise import POSNoise, SpacyLanguage, SpacyModelSize
>>> posnoise = POSNoise(language=SpacyLanguage.German, spacy_model_size=SpacyModelSize.Medium)
>>> doc = "Schöneberg ist ein Ortsteil im Berliner Bezirk Tempelhof-Schöneberg."
>>> posnoise.pos_noise(doc)
"§ ist ein # im @ # §."
"""

from __future__ import annotations

import re
from pathlib import Path
from enum import Enum
from typing import Callable, Iterable, Optional, Union

import numpy as np
import spacy
from spacy.language import Language
from tqdm.auto import tqdm

try:
    # Python 3.9+
    from functools import cache
except ImportError:  # pragma: no cover
    from functools import lru_cache as cache  # type: ignore


class SpacyLanguage(Enum):
    English = "en"
    German = "de"


class SpacyModelSize(Enum):
    """
    spaCy English model presets used for tokenization and POS tagging.
    
    Small (lightweight, fast, no vectors)
    Medium (word vectors)
    Large (larger vectors, enhanced coverage)
    Neural (transformer-based pipeline)
    """
    Small = "sm"
    Medium = "md"
    Large = "lg"
    Neural = "trf"


SPACY_MODEL_IDS = {
    SpacyLanguage.English: {
        SpacyModelSize.Small: "en_core_web_sm",
        SpacyModelSize.Medium: "en_core_web_md",
        SpacyModelSize.Large: "en_core_web_lg",
        SpacyModelSize.Neural: "en_core_web_trf",
    },
    SpacyLanguage.German: {
        SpacyModelSize.Small: "de_core_news_sm",
        SpacyModelSize.Medium: "de_core_news_md",
        SpacyModelSize.Large: "de_core_news_lg",
        SpacyModelSize.Neural: "de_dep_news_trf",
    },
}


class POSNoise:    
    """
    Topic masking through POS-tag substitution.

    This class implements the POSNoise method (Halvani et al. 2021) to mask
    thematic content in documents by replacing content words with symbolic
    POS placeholders (e.g., NOUN → "#", VERB → "Ø"), while retaining
    punctuation marks as well as words and phrases that are stylistically
    relevant. The result is a "skeleton" of the text emphasizing style
    rather than topic.

    POSNoise supports multiple languages through language-specific spaCy
    models and safe-pattern lists. Currently, English and German are supported.

    Parameters
    ----------
    nlp_model : spacy.language.Language, optional
        A pre-initialized spaCy pipeline. If omitted, one is loaded internally.

    abbrev_pos_tags : dict, optional
        Mapping from POS tags (e.g. "NOUN") to placeholder symbols.

    language : SpacyLanguage or str, default=SpacyLanguage.English
        Language of the input documents. Determines which spaCy model family
        and safe-pattern list are used. Supported values are
        ``SpacyLanguage.English`` and ``SpacyLanguage.German``.

    spacy_model_size : SpacyModelSize or str, default=SpacyModelSize.Large
        Size of the spaCy model to load when ``nlp_model`` is not supplied.
        Depending on the selected language, one of the corresponding spaCy
        model variants (Small, Medium, Large, or Neural) is loaded.

    disable : Iterable[str], default=("parser", "ner")
        Components of the spaCy pipeline to disable for efficiency.

    verbose : bool, default=False
        If True, prints progress messages during model loading and downloading.

    log_fn : Callable[[str], None], optional
        Custom logger callback for progress messages. If provided, overrides
        printing. Useful for GUIs or notebooks that capture output.

    Examples
    --------
    English
    ~~~~~~~~
    >>> posnoise = POSNoise(language=SpacyLanguage.English, spacy_model_size=SpacyModelSize.Medium)
    >>> posnoise.pos_noise("The dataset contains sockpuppets.")
    'The # Ø #.'

    German
    ~~~~~~
    >>> posnoise = POSNoise(language=SpacyLanguage.German, spacy_model_size=SpacyModelSize.Medium)
    >>> posnoise.pos_noise("Ich liebe Python!")
    'Ich Ø §!'

    Notes
    -----
    The exact masking output may vary slightly across spaCy model versions,
    as tokenization and POS tagging are determined by the underlying spaCy
    pipeline.
    """

    @cache
    def safe_patterns(self):
        """
        Load language-specific token/phrase-level safe patterns.
        """
        pattern_files = {
            SpacyLanguage.English: "posnoise/pattern_list/POSNoise_PatternList_En_v2.1.txt",
            SpacyLanguage.German: "posnoise/pattern_list/POSNoise_PatternList_De_v3.0.txt"}

        patterns_filepath = Path(pattern_files[self.language])

        return [
            [t.text.lower() for t in self.nlp_model(p)]
            for p in patterns_filepath.read_text(encoding="utf-8").splitlines()
            if p.strip()
        ]

    def __init__(
        self,
        nlp_model: Optional["spacy.language.Language"] = None,
        abbrev_pos_tags: Optional[dict] = None,
        language: Union[SpacyLanguage, str] = SpacyLanguage.English,
        spacy_model_size: Union[SpacyModelSize, str] = SpacyModelSize.Large,
        safe_patterns_path: Optional[Union[str, Path]] = None,
        disable: Iterable[str] = ("parser", "ner"),
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None):

        self.verbose = bool(verbose)
        self._log_fn = log_fn
        self.language = (language if isinstance(language, SpacyLanguage) else SpacyLanguage(str(language)))
        self.spacy_model_size = (spacy_model_size if isinstance(spacy_model_size, SpacyModelSize) else SpacyModelSize(str(spacy_model_size)))

        self._disable = tuple(disable)
        model_id = SPACY_MODEL_IDS[self.language][self.spacy_model_size]
        self.nlp_model = nlp_model or self.get_spacy_nlp(model_id)

        if safe_patterns_path is None:
            if self.language == SpacyLanguage.English:
                safe_patterns_path = "posnoise/pattern_list/POSNoise_PatternList_Ver.2.1.txt"
            elif self.language == SpacyLanguage.German:
                safe_patterns_path = "posnoise/pattern_list/POSNoise_PatternList_DE.txt"

        self.safe_patterns_path = Path(safe_patterns_path)
        self.abbrev_pos_tags = abbrev_pos_tags or {
            "NOUN": "#",
            "PROPN": "§",
            "VERB": "Ø",
            "AUX": "Ø",
            "ADJ": "@",
            "ADV": "©",
            "NUM": "µ",
            "SYM": "$",
            "X": "¥"}

    def _log(self, msg: str) -> None:
        """Log progress messages either via custom logger or stdout."""
        
        if self._log_fn is not None:
            try:
                self._log_fn(msg)
            except Exception:
                # Fall back to print if custom logger fails
                print(msg, flush=True)
        elif self.verbose:
            print(msg, flush=True)

    @cache
    def get_spacy_nlp(self, model_id: str) -> "spacy.language.Language":
        """
        Load or auto-install a spaCy pipeline by ID, with user-visible feedback.

        Parameters
        ----------
        model_id : str
            The model name (e.g. "en_core_web_lg").

        Returns
        -------
        spacy.language.Language
            The loaded spaCy model.

        Raises
        ------
        RuntimeError
            If installation or loading fails.
        """
        
        self._log(f"[POSNoise] Loading spaCy model '{model_id}' (disabled: {', '.join(self._disable) or 'none'})...")
        try:
            nlp = spacy.load(model_id, disable=list(self._disable))
            self._log(f"[POSNoise] Loaded '{model_id}'.")
            return nlp
        except OSError:
            self._log(f"[POSNoise] Model '{model_id}' not found. Attempting to download...")
            try:
                from spacy.cli import download as spacy_download
            except Exception as import_err:
                raise RuntimeError(
                    f"spaCy model '{model_id}' is not installed and the download utility "
                    f"could not be imported. Original error: {import_err}")

            try:
                self._log(f"[POSNoise] Downloading '{model_id}' — this may take a few minutes.")
                spacy_download(model_id)
                self._log(f"[POSNoise] Download complete. Installing/validating '{model_id}'...")
            except SystemExit as e:
                # Note, spacy.cli.download may call sys.exit on failure
                self._log(f"[POSNoise] Download failed for '{model_id}'.")
                raise RuntimeError(
                    f"Failed to auto-install spaCy model '{model_id}'. Detail: {e}")
            except Exception as e:
                self._log(f"[POSNoise] Download failed for '{model_id}'.")
                raise RuntimeError(f"Failed to auto-install spaCy model '{model_id}': {e}")

            try:
                nlp = spacy.load(model_id, disable=list(self._disable))
                self._log(f"[POSNoise] Successfully installed and loaded '{model_id}'.")
                return nlp
            except Exception as e:
                self._log(f"[POSNoise] Installed '{model_id}', but loading failed.")
                raise RuntimeError(
                    f"Installed spaCy model '{model_id}', but failed to load it: {e}")

    def pos_noise_(self, text: str):
        """
        Compute which tokens to preserve vs. replace with placeholders.

        Tokens are preserved if they appear in the safe patterns list, are certain
        contractions, punctuation marks or are numerals (but not purely digits). 
        All other tokens represent candidates for topic masking replacement.

        Parameters
        ----------
        text : str
            Input document.

        Returns
        -------
        tuple[numpy.ndarray, list]
            - bitmask: Boolean array (True = preserve, False = replace)
            - tokens: List of spaCy tokens
        """
        
        tokens = list(self.nlp_model(text))
        bitmask = np.zeros(len(tokens), dtype=bool)
        
        ENGLISH_CONTRACTIONS = {"'m", "'d", "'s", "'t", "'ve", "'ll", "'re", "'ts", "'em", "'Tis"}
        GERMAN_STYLE_TOKENS = {"'s", "’s"}

        for safe_pattern in self.safe_patterns():
            i = 0
            pattern_index = 0
            while i < len(tokens):
                if tokens[i].text.lower() == safe_pattern[pattern_index]:
                    pattern_index += 1
                    if pattern_index == len(safe_pattern):
                        for j in range(i - len(safe_pattern) + 1, i + 1):
                            bitmask[j] = True
                        pattern_index = 0
                else:
                    i -= pattern_index
                    pattern_index = 0
                i += 1

        for i, token in enumerate(tokens):            
            if self.language == SpacyLanguage.English and token.text in ENGLISH_CONTRACTIONS:
                bitmask[i] = True
                
            if self.language == SpacyLanguage.German and token.text in GERMAN_STYLE_TOKENS:
                bitmask[i] = True
                
            if token.pos_ == "NUM" and not re.fullmatch(r"\d+", token.text):
                bitmask[i] = True

        return bitmask, tokens

    def pos_noise(self, text: str) -> str:
        """
        Apply topic masking to text by replacing topic-related tokens with POS placeholders. 
        Stylistically relevant words and phrases contained in safe_patterns() remain unchanged.

        Parameters
        ----------
        text : str
            Input document to transform.

        Returns
        -------
        str
            Topic-masked output: topic-affected words and phrases replaced by POS placeholders.

        Examples
        --------
        >>> posnoise = POSNoise(spacy_model_size=SpacyModelSize.Medium)
        >>> posnoise.pos_noise("The dataset contains sockpuppets.")
        "The # Ø #."
        """
        
        bitmask, tokens = self.pos_noise_(text)
        for m, token in reversed(list(zip(bitmask, tokens))):
            if not m:
                replace_token = self.abbrev_pos_tags.get(token.pos_, token.text)
                text = text[: token.idx] + replace_token + text[token.idx + len(token.text) :]
        return text

    def pos_noise_corpus(
        self,
        corpus_path: Union[str, Path],
        overwrite: bool = False,
        destination_path: Optional[Union[str, Path]] = None,
        files_to_exclude: Optional[Iterable[str]] = None,
        file_extensions: Optional[Iterable[str]] = (".txt",),
        encoding: str = "utf-8",
        errors: str = "strict") -> None:
        """
        Apply POS-based masking to a text corpus on disk.

        Recursively traverses `corpus_path` and applies `pos_noise()` to files matching the given file extensions.
        By default, ground-truth metadata files {"contents.json", "truth.txt", "meta.csv"} commonly found in well-known 
        PAN Authorship Verification corpora [1] are not masked and are either copied unchanged (when `overwrite=False`) 
        or left untouched (when `overwrite=True`).
        
        [1] PAN Authorship Verification corpora --> https://pan.webis.de/data.html

        Args:
            corpus_path: Root directory of the source corpus.
            overwrite: If True, overwrite files in place under `corpus_path`.
                If False, mirror the directory structure into `destination_path`
                and write masked files there.
            destination_path: Output root directory, required when `overwrite` is False.
            files_to_exclude: Filenames to exclude from masking (matched against `Path.name`).
                Defaults to {"contents.json", "truth.txt"}.
            file_extensions: File extensions to process (e.g. [".txt", ".md"]).
                - Default: (".txt",)
                - If None: process all files except excluded meta-files.
            encoding: Text encoding used for reading and writing.
            errors: Encoding error handling strategy ("strict", "replace", "ignore").

        Raises:
            ValueError: If `corpus_path` is not an existing directory.
            ValueError: If `overwrite` is False and `destination_path` is not provided.
        """
    
        src_root = Path(corpus_path)
        if not src_root.exists() or not src_root.is_dir():
            raise ValueError(f"Corpus path must be an existing directory, got: {src_root}")

        excluded_names = set(files_to_exclude) if files_to_exclude is not None else {
            "contents.json", "truth.txt", "meta.csv"}

        normalized_exts = None
        if file_extensions is not None:
            normalized_exts = {ext.lower() for ext in file_extensions}

        if overwrite:
            dst_root = None
        else:
            if destination_path is None:
                raise ValueError("Destination path must be provided when overwrite=False")
            dst_root = Path(destination_path)
            dst_root.mkdir(parents=True, exist_ok=True)

        all_files = [p for p in src_root.rglob("*") if p.is_file()]

        for src_path in tqdm(all_files, desc="POSNoise corpus", unit="file"):
            rel_path = src_path.relative_to(src_root)
            dst_path = src_path if overwrite else dst_root / rel_path  # type: ignore[operator]
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Always skip excluded meta-files
            if src_path.name in excluded_names:
                if not overwrite:
                    dst_path.write_bytes(src_path.read_bytes())
                continue

            # Extension-based filtering
            if normalized_exts is not None and src_path.suffix.lower() not in normalized_exts:
                if not overwrite:
                    dst_path.write_bytes(src_path.read_bytes())
                continue

            text = src_path.read_text(encoding=encoding, errors=errors)
            masked = self.pos_noise(text)
            dst_path.write_text(masked, encoding=encoding, errors=errors)
