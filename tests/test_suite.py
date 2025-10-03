import os
import sys
import inspect
import pytest
import unittest
import spacy

# Import CTL from the parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import posnoise
from posnoise import POSNoise
#from posnoise.errors import *


class TestPOSNoise(unittest.TestCase):
    #nlp = None
    #defect_nlp = None

    def test_posnoise_masking_1_large_model(self):
        posnoise_instance = POSNoise(spacy_model_size=posnoise.SpacyModelSize.Large)
        document = "I love python !"
        posnoised_doc = posnoise_instance.pos_noise(document) 
        
        assert posnoised_doc == "I Ø # !"

   