# POSNoise
POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis


## Description
POSNoise is a preprocessing method that systematically masks topic-related content from text documents. The motivation for this becomes clear when we take a closer look at the field of authorship analysis, particularly **Authorship Attribution** (**AA**) and **Authorship Verification** (**AV**).  

AA is concerned with determining which of several possible authors wrote a particular anonymous text. AV, on the other hand, deals with the so-called fundamental problem of whether two documents were written by the same person. To accomplish both tasks, we rely on the assumption that every person has his or her own writing style, which differs to a certain extent from the writing styles of other people. This is exactly where the topic can pose a serious challenge.

Consider for example the following scenario: *D1* and *D2* represent two documents that deal with the same topic but were written by two different people. An AV method (which aims to answer the question of whether two documents were written by the same author) could mistakenly conclude that *D1* and *D2* were written by the same author, as both documents are very similar in terms of topic. However, the goal of AA and AV is **not** to predict whether authorship is the same or different based on the **topic**, but rather based on the actual **writing style**. 

To prevent AA and AV methods from focusing on the topic and instead force them to concentrate on linguistic patterns such as function words or punctuation marks (which are closely related to writing style), topic masking approaches can be used, where POSNoise is one such approach... 


## Installation
The easiest way to install POSNoise is to use pip, where you can choose between (1) the PyPI repository and (2) this repository. 

- (1) ```pip install posnoise```

- (2) ```pip install git+https://github.com/Halvani/POSNoise.git ```

The latter will pull and install the latest commit from this repository as well as the required Python dependencies. 


## Quickstart
```python
import posnoise
from posnoise import POSNoise

posnoise_instance = POSNoise(spacy_model_size=posnoise.core.SpacyModelSize.Large)

document = "I love python !"
posnoised_doc = posnoise_instance.pos_noise(document)

print(document)
print(posnoised_doc)
 
# I love python !
# I Ø # !
```

A detailed explanation of the **POS placeholders** (Ø, #, etc.) used in POSNoise can be found in Table 2 in the paper. Here, the arXiv version for open access: https://arxiv.org/abs/2005.06605

## Features
- Effective way to mask topic-relatd content with **custom POS placeholders**
- **Automatic NLP pipeline creation** (loads and installs the spaCy models on demand, while providing feedback)
- **No API dependency** (after downloading the spaCy models, POSNoise can be used completely **offline**)
- **Documented** source code

## Citation
If you find this library helpful, please invest a few minutes and cite it in your paper/project:
```bibtex
@inproceedings{HalvaniGranerPOSNoise:2021,
  author       = {Oren Halvani and Lukas Graner},
  editor       = {Delphine Reinhardt and Tilo M{\"{u}}ller},
  title        = {{POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis}},
  booktitle    = {{ARES} 2021: The 16th International Conference on Availability, Reliability and Security, Vienna, Austria, August 17-20, 2021},
  pages        = {47:1--47:12},
  publisher    = {{ACM}},
  year         = {2021},
  url          = {https://doi.org/10.1145/3465481.3470050},
  doi          = {10.1145/3465481.3470050},
  timestamp    = {Sun, 04 Aug 2024 19:40:49 +0200},
  biburl       = {https://dblp.org/rec/conf/IEEEares/HalvaniG21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License
The POSNoise package is released under the Apache-2.0 license. See <a href="https://github.com/Halvani/posnoise/blob/main/LICENSE">LICENSE</a> for further details.


## Last Remarks
As is usual with open source projects, we developers do not earn any money with what we do, but are primarily interested in giving something back to the community with fun, passion and joy. Nevertheless, we would be very happy if you rewarded all the time that has gone into the project with just a small star 🤗