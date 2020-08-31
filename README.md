# TransferDet

This is the repository containing team OverFeat's submission to CVPPP 2020's Wheat Detection Challenge. 

Final Ranking: 3/2245 (Solo Gold)

Leaderboard: https://www.kaggle.com/c/global-wheat-detection/leaderboard

Solution Journal: https://www.kaggle.com/c/global-wheat-detection/discussion/175961

Submission Notebook: https://www.kaggle.com/alexanderliao/effdet-d6-pl-s-bn-r-bb-a3-usa-eval-94-13-db?scriptVersionId=40133294

Pre-processed Jigsaw Data: https://www.kaggle.com/alexanderliao/wheatfullsize

Pseudo-labelled SPIKE Dataset: https://www.kaggle.com/alexanderliao/wheatspike

Private/Public mAP [0.5:0.75:0.05] : 0.6879/0.7700

Steps to reproduce leaderboard performance:

1. Download train data
2. Prepare jigsaw data by running `jigsaw/jigsaw_{0-6}.ipynb`
3. Train baseline model using `train_baseline.py`
4. Train 2-nd level models using `train_STAC.py` and `train_SplitBN.py`
5. Run `effdet-d6-pl-s-bn-r-bb-a3-usa-eval-94-13-db.ipynb` for pseudo-labeling and final inference; or fork my notebook on Kaggle