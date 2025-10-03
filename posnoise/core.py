"""
POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis.

This module implements a topic masking approach (POSNoise) that masks
thematic content by replacing topic-bearing tokens with POS-tag placeholders,
while retaining punctuation, stylistic markers as well as syntactic structures.
The goal is to attenuate topic signal and emphasize stylistic cues (e.g.,
for Authorship Attribution and Authorship Verification). 

The approach is described in:
----------------------------------------------
Oren Halvani and Lukas Graner. 2021. POSNoise: An Effective Countermeasure Against 
Topic Biases in Authorship Analysis. In Proceedings of the 16th International Conference 
on Availability, Reliability and Security (ARES '21). Association for Computing Machinery, 
New York, NY, USA, Article 47, 1–12. https://doi.org/10.1145/3465481.3470050
----------------------------------------------

Examples
--------
>>> posnoise = POSNoise(spacy_model_size=SpacyModelSize.Medium)
>>> doc = "The original dataset contains two partitions comprising sockpuppets and non-sockpuppets cases."
>>> posnoise.pos_noise(doc)
"The # # Ø µ # Ø # and #-# #."
"""


from __future__ import annotations

import re
from pathlib import Path
from enum import Enum
from typing import Callable, Iterable, Optional, Union

import numpy as np
import spacy

try:
    # Python 3.9+
    from functools import cache
except ImportError:  # pragma: no cover
    from functools import lru_cache as cache  # type: ignore


class SpacyModelSize(Enum):
    """
    spaCy English model presets used for tokenization and POS tagging.

    Small
        "en_core_web_sm" (lightweight, fast, no vectors).
    Medium
        "en_core_web_md" (word vectors).
    Large
        "en_core_web_lg" (larger vectors, enhanced coverage).
    Neural
        "en_core_web_trf" (transformer-based pipeline).
    """

    Small = "en_core_web_sm"
    Medium = "en_core_web_md"
    Large = "en_core_web_lg"
    Neural = "en_core_web_trf"


class POSNoise:
    """
    Topic masking through POS-tag substitution.

    This class implements the POSNoise method (Halvani et al. 2021) to mask
    thematic content in documents by replacing content words with symbolic
    POS placeholders (e.g., NOUN → "#", VERB → "Ø"), while retaining punctuation marks 
    as well as words and phrases that are stylisticaly relevant. The result is
    a “skeleton” of the text emphasizing style rather than topic.

    Parameters
    ----------
    nlp_model : spacy.language.Language, optional
        A pre-initialized spaCy pipeline. If omitted, one is loaded internally.
    abbrev_pos_tags : dict, optional
        Mapping from POS tags (e.g. "NOUN") to placeholder symbols.
    spacy_model_size : SpacyModelSize or str, default=SpacyModelSize.Small
        Identifier of spaCy model to load when `nlp_model` is not supplied.
    disable : Iterable[str], default=("parser", "ner")
        Components of the spaCy pipeline to disable for efficiency.
    verbose : bool, default=False
        If True, prints progress messages during model load/download.
    log_fn : Callable[[str], None], optional
        Custom logger callback for progress messages. If provided, overrides
        printing. Useful for GUIs or notebooks that capture output.

    Examples
    --------
    >>> posnoise = POSNoise(spacy_model_size=SpacyModelSize.Medium, verbose=True)
    >>> posnoise.pos_noise("The dataset contains sockpuppets.")
    "The # Ø #."
    """

    @cache
    def safe_patterns(self):
        """
        Load a list of token/phrase-level safe patterns that should remain unchanged.

        Returns
        -------
        list[list[str]]
            Each pattern is tokenized and lowercased; tokens matching these
            patterns will *not* be masked by POSNoise.
        """
        #patterns_filepath = Path(os.getcwd(), "pattern_list", "POSNoise_PatternList_Ver.2.1.txt")
        patterns_filepath = Path(r"posnoise/pattern_list/POSNoise_PatternList_Ver.2.1.txt")
                                
        return [
            [t.text.lower() for t in self.nlp_model(p)]
            for p in patterns_filepath.read_text(encoding="utf-8").splitlines()
            if p.strip()
        ]

    def __init__(
        self,
        nlp_model: Optional["spacy.language.Language"] = None,
        abbrev_pos_tags: Optional[dict] = None,
        spacy_model_size: Union[SpacyModelSize, str] = SpacyModelSize.Small,
        disable: Iterable[str] = ("parser", "ner"),
        verbose: bool = False,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the POSNoise masker, setting up the NLP pipeline and placeholder map.

        If `nlp_model` is absent, loads or auto-installs the spaCy model given
        by `spacy_model_size`. When `verbose=True` (or a `log_fn` is provided),
        the user is informed about download/installation steps.
        """
        self.verbose = bool(verbose)
        self._log_fn = log_fn
        model_id = (
            spacy_model_size.value
            if isinstance(spacy_model_size, SpacyModelSize)
            else str(spacy_model_size)
        )
        self._disable = tuple(disable)
        self.nlp_model = nlp_model or self.get_spacy_nlp(model_id)
        self.abbrev_pos_tags = abbrev_pos_tags or {
            "NOUN": "#",
            "PROPN": "§",
            "VERB": "Ø",
            "ADJ": "@",
            "ADV": "©",
            "NUM": "µ",
            "SYM": "$",
            "X": "¥",
        }

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
            The model name (e.g. "en_core_web_md").

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
                    f"could not be imported. Original error: {import_err}"
                )

            try:
                self._log(f"[POSNoise] Downloading '{model_id}' — this may take a few minutes.")
                spacy_download(model_id)  # May print its own progress; we also announce start/finish.
                self._log(f"[POSNoise] Download complete. Installing/validating '{model_id}'...")
            except SystemExit as e:
                # spacy.cli.download may call sys.exit on failure
                self._log(f"[POSNoise] Download failed for '{model_id}'.")
                raise RuntimeError(
                    f"Failed to auto-install spaCy model '{model_id}'. Detail: {e}"
                )
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
                    f"Installed spaCy model '{model_id}', but failed to load it: {e}"
                )

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
            if token.text in {"'m", "'d", "'s", "'t", "'ve", "'ll", "'re", "'ts", "'em", "'Tis"}:
                bitmask[i] = True
            if token.pos_ == "NUM" and not re.fullmatch(r"\d+", token.text):
                bitmask[i] = True

        return bitmask, tokens

    def pos_noise(self, text: str) -> str:
        """
        Apply topic masking to text by replacing tokens with POS placeholders.

        For each token not marked for preservation, substitute the token text
        with its POS placeholder. Preserved tokens (e.g. punctuation, safe
        patterns, certain numerals, contractions) remain unchanged.

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

