### Predict Future Sales ###
Solution to the Final project for "How to win a data science competition" Coursera course

Public leaderboard score:  0.893321

Private leaderboard score: 0.889371


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# HARDWARE / SOFTWARE SPECIFICATIONS

Hardware specifications:
2,3 GHz Intel Core i5 - 4 cores

RAM 8Go

Software specifications:

MacOS Mojave 10.14.6

python 3.7.0

numpy 1.17.2

pandas 0.25.1

sklearn 0.21.3

xgboost 0.90

matplotlib 3.1.2

seaborn 0.9.0


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# QUICK DESCRIPTION

This solution is mostly based on
- splitting of dataset into three classes of realisations (month,shop,item): item new in the dataset this month (seniority = 0) / item not new but never sold in this shop before (seniority = 1) / item sold in this shop in the past(seniority = 2)
- validation and fitting of three separate regressions, each using xgbregressor from xgboost, for each of the three classes of realisations defined above
- main features for (shop,item) of seniority 2: past sale quantities for same (shop,item)
- main features for (shop,item) of seniority 1: typical past sale quantities for the item averaged over shops where it was sold, and typical sale quantities in this shop for items of the same category and released for the same amount of time 
- main features for (shop,item) of seniority 0: typical past sale quantities for items of the same category on the month where they were initially released


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# DETAILS OF THE SOLUTION

We aim at predicting the quantities sold for each sample (shop, item) on the next month. For each item, the quantities sold vary through time (from month to month) and space (from shop to shop). Besides, similarity between items is embedded in the category they belong to. The 3 dimensions to account for when forming a prediction are thus time, space, and the specificity of each item. We form features to account for variability of the sales in each of these 3 directions (from month to month, from shop to shop, from item to item).

- Additional features are extracted from the item categories to enhance the understanding of item similarity. In particular, supercategories are defined that encompass several categories of items of the same type (such as consoles, video games, accessories, books, music, ...).

- The spatial variability of the sales appear to be mostly orthogonal to the temporal variations (similar temporal variations across all shops). Spatial trends are accounted for through target encoding of the shops.

- This solution is mostly based on the temporal separation of the samples (shop, item) depending on so-called 'seniority' levels. Each month, we analyse separately samples made of items that are new in the dataset this month (seniority = 0), samples (shop, item) made of items that are not new but have never been sold in this specific shop before (seniority = 1), and samples (shop, item) made of items that have already been sold in the past in this very shop (seniority = 2).

The main idea behind this separation is that the dataset on a given month is made of the cartesian product between the whole global catalogue of items and the whole set of shops open, whereas in practice the local catalogue of items available in a given shop only includes a fraction of the global catalogue. Being able to discriminate items that are in fact available in a given shop from items that are actually not should ease the predictions, as samples (shop, item) with items not even available in the shop in question cannot possibly be sold there in the future. 

Samples (shop, item) of seniority 2 are made of items that have previously been sold in this very shop, and so they are in the local catalogue of this shop for sure. For such samples, our predictions will be mostly based on the quantities sold for this item in this shop in the past.

Samples (shop, item) of seniority 1 are made of items that have never been sold in this shop, but have been sold in other shops before. Such a sample (shop, item) is likely to be made of an item that is not actually available in this shop, and the likelihood that this item will be sold in this shop in the future is very low. In fact, data analysis proves that only 5% of these samples contribute to the sales (compared to 20-25% of samples of seniority 2). Those few samples of seniority 1 that will be sold are likely made of items that are always sold in low quantities, possibly less than one per month. Here again, data analysis shows that the average sales for items of seniority 1 is around 0.05 (against 0.4 in seniority 2) and the likelihood that an item of seniority 1 is sold in quantities superior to 3 is less than 0.1%. Samples of seniority 1 yielding sale quantities above 8 are so rare that they may be considered noise. For seniority 1, the critical features are the number of months since it has been released in the catalogue without being sold in this shop, and the typical sale quantities of this item in shops where it has been sold in the past. If an item is typically sold in large quantities but have never been sold in a given shop, it is unlikely that it will suddenly start being sold next month. Similarly, if the item has never been sold in a given shop in spite having been released for many months in the catalogue, it is likely that it will never be sold in this shop, or only in very small quantities. Conversely, if the item has only been released last month, it is possible that it just hasn't add the chance to be sold last month but it may be sold in low quantities next month. In particular, we find that the most relevant features are target encodings of the past typical sale quantities of samples of seniority 1 in this shop, in the same category of items, and with the same number of months since it has been released, as well as the quantities sold for this item in other shops where it has been sold in the past.

Finally, samples (shop, item) of seniority 0 are totally new in the catalogue. No past data is available for these pairs. Besides, the typical sale quantities of the new items vary a lot from month to month and their temporal evolution appears much noisier than older items. For these items, we base our predictions mostly on past typical sale quantities of similar items on the month where they were released. Similar items are defined as items of the same category, or same supercategory (like some other game, or some other book).

- Apart from the separation between seniority levels, the influence of temporality is threefold: the quantities sold for a given item in a given shop depend on the absolute period of the year (more sales in December due to Christmas), on how many months since the item has been released (older items are less popular than newer ones), and also on the month where it has been released (items released in December are sold more all year round than items released in February). The sensitivity of the items to each of these three temporality vary from category to category and depending on the seniority level of the sample: samples of seniority 1 are almost insensitive to the absolute period of the year but very sensitive to the number of months since they have been released, while deliveries are sold in similar quantities regardless of how long the service has been offered, etc... Thus, the effect of temporality is accounted for jointly with other features through target encoding.

- Three separate models are built using the XGBRegressor algorithm from xgboost. Different hyperparameters are used to account for the different distributions of target variable, different problem complexities, and different tendencies to overfit



# SPLITTING TRAIN-VALIDATION-TEST SET
- The first 18 months are excluded from the training set. Indeed, the past data of samples close to the beginning of the dataset is not known, and the first 18 months have to be excluded in order for the features generated to be insensitive to the proximity to the beginning of the dataset.
- The validation set is chosen as made of months 32 and 33. For validation purposes, the training set includes months 18 to 31.
- The test set is month 34, and the training set for prediction purposes includes months 18 to 33.




- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# PROCEDURE TO GENERATE SUBMISSION FILE FROM RAW DATA

*First untar the sales_train.csv.gz and test.csv.gz data files in data/raw.

This solution uses the XGBRegressor from xgboost. The code can be found in the 'code' directory, all in notebooks. In order to reproduce the submission files from the raw data, all notebooks may be ran in the following order:

- 0_data_preparation : this notebook includes data cleaning, monthly aggregation, and exploratory data analysis

- 1_target_encoding : this notebook performs the target encoding of several quantities with respect to each month, with different levels of feature interactions

- 2_feature_aggregation : this notebook includes the splitting of the dataset according to the seniority level of the realisations, and the aggregation of relevant features to each of the three dataframes thus formed

- (4_training_for_predictions-seniority_0,  4_training_for_predictions-seniority_1,  4_training_for_predictions_2) : in these three notebooks, we manually discard the features that actually do not improve the quality of the model (based on manual testing), and fit the xgbregressor to the training set, respectively for each seniority level. The test dataset and the xgbregressor model are then serialized to disk to allow predictions to be formed later on.

- 5_assemble_predictions : in this notebook, we import the test dataset and xgbregressor models fit for each seniority level, then form a prediction for each and assemble then to form the global prediction on the whole test set.


Feature selection and hyperparameter optimization for each of the three models is performed manually, based on results on the validation set. These test were performed in notebooks (3_validation-seniority_0,  3_validation-seniority_1,  3_validation-seniority_2).

All the output data from each of the notebooks is provided in the 'data' directory (including raw data and final submission file), so that any of the aforementionned notebooks may be run directly without having to run the previous ones.

The submission file can be found at 'data/predictions/xgb_submission.csv' .



- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# TIME

With the specifications mentionned above, the time for fitting the models were approximately:
- ~ 100s   (1-2 mins)   for seniority 0
- ~ 2200s  (35-40 mins) for seniority 1
- ~ 14700s (~4 hours)   for seniority 2

