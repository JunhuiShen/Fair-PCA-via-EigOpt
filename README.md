# Fair PCA via Eigenvalue Optimization

This section describes the implementation of a novel fair PCA algorithm using eigenvalue optimization presented in the paper "Fair and Efficient: Hidden Convexity of Fair PCA and  Fast Solution via Eigenvalue Optimization".

## Algorithm Implementation

- The algorithm, **Fair PCA via Eigenvalue Optimization**, is implemented in [FairPCAviaEigOpt.m](./FairPCAviaEigOpt.m).
- Main function: [main.m](./main.m).
- Average reconstruction loss function: [rloss.m](./rloss.m).
- Examples of synthetic data: [synthetic_data.m](./synthetic_data.m).

## Datasets

Here are the datasets we consider:

- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing): 
  - $M \in \mathbb{R}^{45211 \times 16}$
  - $A \in \mathbb{R}^{44401 \times 16}$
  - $B \in \mathbb{R}^{810 \times 16}$
  - [Script](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/bank_marketing.m)

- [Crop Mapping](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set):
  - $M \in \mathbb{R}^{325834 \times 173}$
  - $A \in \mathbb{R}^{39162 \times 173}$
  - $B \in \mathbb{R}^{286672 \times 173}$
  - [Script](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/crop_mapping.m)

- [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients):
  - $M \in \mathbb{R}^{30000 \times 23}$
  - $A \in \mathbb{R}^{10599 \times 23}$
  - $B \in \mathbb{R}^{19401 \times 23}$
  - [Script](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/default_credit.m)

- [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/):
  - $M \in \mathbb{R}^{13232 \times 1764}$
  - $A \in \mathbb{R}^{2962 \times 1764}$
  - $B \in \mathbb{R}^{10270 \times 1764}$
  - [Script](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/LFW.m)

Note: The data matrices are all centered. That is, the mean value of every matrix $A$ and $B$ is set to $0$.

## Reference Paper

The code for the Fair PCA via LP algorithm is introduced in the paper "[The Price of Fair PCA: One Extra Dimension](https://arxiv.org/abs/1811.00103)" by Samadi S., Tantipongpipat U., Morgenstern J., Singh M., and Vempala S., NeurIPS, 2018. Our implementation, [FairPCAviaLP.m](./FairPCAviaLP.m), is adapted from their code by integrating all the related functions and set the tolerance condition.

You can access the code repository [here](https://github.com/samirasamadi/Fair-PCA?tab=readme-ov-file).

## Data Folder

The data folder needed to run the code may be accessed [here](https://drive.google.com/drive/u/1/folders/1xmdlEYPJDS7nwMQqbOoEuG3TCWLCBkUJ).
