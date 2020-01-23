# Hierarchical_BiLSTM-CRF_Encoder

This repository is an attempt to debug and make functional the existing code base from [this](https://github.com/YanWenqiang/HBLSTM-CRF/blob/master/HBLSTM-CRF.py) repo. The code is an implementation of a paper titlted [Dialogue Act Recognition via CRF-Attentive Structured Network](https://arxiv.org/pdf/1711.05568.pdf). The paper claims a nearly SOTA solution to Dialogue Act classification from text sequences. 

To run the code, clone the repository and run 'python model.py'. This will train your model on the Switchboard Dialogue Act dataset referenced in the paper.

The Switchboard Dialogue Act data has been taken from the following [repo](https://github.com/cgpotts/swda). The repo provides the raw data as well as a preprocessing script to make the data more usable. I have already done this and am including the processed data.

You will notice another file in this repository, 'processed_data.py'. I wrote this because the model requires the data to be further processed in order to be fed into its architecture. This is automatically called in 'model.py'.

For the project, I have used:
- Python 3
- tensorflow == 1.15.0
- pandas == 0.25.3
- numpy == 1.17.5

### NOTE:

_January 23, 2020_
- This code still has bugs. The model trains but consistantly breaks on the second validation scoring loop in 'model.py'. 
