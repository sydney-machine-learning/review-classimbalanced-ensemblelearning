# review-classimbalanced-ensemblelearning
A review of class imbalanced problems using data augumentation and ensemble learning 

### Usage Guidelines and Examples
`main.py` is the python script that can be used to evaluate all ensemble learning and data augmentation techniques except CT-GAN for which we have created a seperate `CT-GAN.ipynb`.

More information on datasets is provided in `Datasets/readme.txt`.

We have benchmarked the datasets and combinations of data augmentation and ensemble learning techniques.

```
!python3 main.py --data Ecoli4 --augmentation SMOTE --ensemble XGBoost
```
The above code can be used to evaluate combination of SMOTE and XGBoost on Ecoli4 dataset. Similarly, other options can be explored. 

```
!python3 main.py --help
```
Above code can be used for help.
