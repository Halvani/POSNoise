# POSNoise
POSNoise: An Effective Countermeasure Against Topic Biases in Authorship Analysis


## Description
POSNoise is a preprocessing method that systematically masks topic-related content from text documents. The motivation for this becomes clear when we take a closer look at the field of authorship analysis, particularly **Authorship Attribution** (**AA**) and **Authorship Verification** (**AV**).  

AA is concerned with determining which of several possible authors wrote a particular anonymous text. AV, on the other hand, deals with the so-called fundamental problem of whether two documents were written by the same person. To accomplish both tasks, we rely on the assumption that every person has his or her own writing style, which differs to a certain extent from the writing styles of other people. This is exactly where the topic can pose a serious challenge.

Consider for example the following scenario: *D1* and *D2* represent two documents that deal with the same topic but were written by two different people. An AV method (which aims to answer the question of whether two documents were written by the same author) could mistakenly conclude that *D1* and *D2* were written by the same author, as both documents are very similar in terms of topic. However, the goal of AA and AV is **not** to predict whether authorship is the same or different based on the **topic**, but rather based on the actual **writing style**. 

To prevent AA and AV methods from focusing on the topic and instead force them to concentrate on linguistic patterns such as function words or punctuation marks (which are closely related to writing style), topic masking approaches can be used, where POSNoise is one such approach... 


