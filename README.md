# Electronic Health Record graph completion using RAGAT: Relation Aware Graph Attention Network.
<!-- ## Overview
![The architecture of RAGAT.](model.png)

We propose a Relation Aware Graph ATtention network (RAGAT) that constructs separate message functions for different relations. Speciﬁcally, we introduce relation speciﬁc parameters to augment the expressive capability of message functions, which enables the model to extract relational information in parameter space. To validate the effect of relation aware mechanism, RAGAT is implemented with a variety of relation aware message functions. Experiments show RAGAT outperforms state-of-the-art link prediction baselines on standard FB15k-237 and WN18RR datasets. -->
## Data
The encoded data is uploaded in the ehr folder in the form of subject,relation,object triples:
- Diagnosis(PheWAS code), associated_with, Diagnosis(PheWAS code)
- Diagnosis(PheWAS code), treated_with, Drug(ATC code)
- Procedure(CPT code), performed_for, Diagnosis(PheWAS code)
- Drug(ATC code), causes, Side Effect(UMLS code)

Although the original data source from which these triples have been extracted is private, the links captured happened in real life patient encounters.

## Dependencies
- Pytorch 1.5

## Datasets
- FB15k-237
- WN18RR

## Training model
```bash
# FB15k-237
python run.py -epoch 1500 -name test_fb -model ragat -score_func
interacte -opn cross -gpu 0 -data FB15k-237 -gcn_drop 0.4 -ifeat_drop 0.4 
-ihid_drop 0.3 -batch 1024 -iker_sz 9 -attention True -head_num 2
# WN18RR
python run.py -epoch 1500 -name test_wn -model ragat -score_func
interacte -opn cross -gpu 0 -data WN18RR -gcn_drop 0.4 -ifeat_drop 0.2 
-ihid_drop 0.3 -batch 256 -iker_sz 11 -iperm 4 -attention True -head_num 1
# EHR
#pretrained
python run.py -epoch 1 -name ehr_ragat_17_03_2022_12:29:23 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.4 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 256 -iker_sz 11 -iperm 4 -attention True -head_num 1 -restore
python run.py -epoch 1 -name extended_ehr_29_06_2022_07:15:16 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.2 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 128 -iker_sz 11 -iperm 4 -attention True -head_num 1 -lbl_smooth 0.125 -lr 0.0005 -restore
#Training with Hyperparamter Optimization using Optuna
python run.py -epoch 800 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.4 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 256 -iker_sz 11 -iperm 4 -attention True -head_num 1
python run.py -epoch 5 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.4 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 256 -iker_sz 11 -iperm 4 -attention True -head_num 1
python run.py -epoch 800 -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.2 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 128 -iker_sz 11 -iperm 4 -attention True -head_num 1 -lbl_smooth 0.125 -lr 0.0005
python run.py -epoch 1000 -name extended_ehr -model ragat -score_func interacte -opn cross -gpu 0 -gcn_drop 0.2 -ifeat_drop 0.2 -ihid_drop 0.3 -batch 128 -iker_sz 11 -iperm 4 -attention True -head_num 1 -lbl_smooth 0.125 -lr 0.0005

```

## Acknowledgement
This code has been forked from the original RAGAT repo.[RAGAT](https://github.com/liuxiyang641/RAGAT)
The project is built upon [COMPGCN](https://github.com/malllabiisc/CompGCN)
