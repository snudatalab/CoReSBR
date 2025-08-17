# Context-aware Sequential Bundle Recommendation via User-specific Representations

This project is a pytorch implementation of 'Context-aware Sequential Bundle Recommendation via User-specific Representations'.
We provide executable source code with adjustable arguments and preprocessed datasets used in the paper.
We also open-source a newely constructed dataset crawled from AllRecipes.com.

## Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)

## Usage

There are 3 folders and each consists of:
- datasets: preprocessed datasets
- runs: pre-trained models for each dataset
- src: source codes

You can run a demo script 'demo.sh' to check the performance of CoReSBR on Movielens-10M dataset.
The result looks as follows:
```
dataset: ml-10m
name: demo
average sequence length: 9.88
user num: 15683
bundle num: 8811
item num: 1245
100%|█████████████████████████████████████████| 15683/15683 [03:25<00:00, 76.31it/s]

----- original (k=5, 20) ----- 

recall: [0.0591, 0.1504]
nDCG:   [0.0377, 0.0633]

```

You can also train the model by running 'main.py'.
You can control following arguments:
- path (any string, default is 'run1'): the path to save the trained model and training log.
- dataset ('allrecipes', 'foods', or 'ml-10m')
- gamma (any number): controls the sharpness of preference emphasis (see Equation(1))
- lmb (=lambda, any number): controls the recency weight (see Equation (3))
- num_strategies (any int): number of bundling strategies to model (see Section 3.4)
- dropout_rate (any number between 0 and 1): dropout rate

For example, you can train the model for allrecipes dataset with lambda of 10, gamma of 0.5, and 4 strategies:
```
python src/main.py --dataset allrecipes --lmb 10 --gamma 0.5 --num_strategies 4
```

You can also test the trained_model by running 'main.py' with the argument 'test' as True:
```
python src/main.py --dataset allrecipes --lmb 10 --gamma 0.5 --num_strategies 4 --test True
```

## Datasets
Preprocessed data are included in the data directory.
AllRecipes dataset is a newly constructed dataset crawled from AllRecipes.com.
| Dataset | Users | Bundles | Items | Interactions |
| --- | ---: | ---: | ---: | ---: |
| AllRecipes | 21,270 | 9,860 | 4,619 | 961,410 |
| Foods | 4,876 | 7,278 | 2,941 | 203,073 |
| Movielens 10M | 15,683 | 8,811 | 12,450 | 134,119 |

The original datasets are available at:
- Foods: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
- Movielens 10M: https://grouplens.org/datasets/movielens
