# Fair PCA via Eigenvalue Optimization

This repository contains the implementation of the Fair PCA algorithm using eigenvalue optimization, as presented in the paper _"Fair and Efficient: Hidden Convexity of Fair PCA and Fast Solution via Eigenvalue Optimization."_ This work was conducted by Junhui Shen (University of California, Davis), Aaron Davis (University of Kentucky), Ding Lu (University of Kentucky), and Zhaojun Bai (University of California, Davis).

## Algorithm Implementation

The Fair PCA via Eigenvalue Optimization (**FPCA via EigOpt**) algorithm is implemented and can be found in the following files:

- **FPCA via EigOpt**: [`FPCAviaEigOpt.m`](./FPCAviaEigOpt.m)
- **Comparison of FPCA via EigOpt and FPCA via SDP**: [`main.m`](./main.m)
- **Comparison of FPCA via EigOpt and c-FPCA**: [`main2.m`](./main2.m)
- **Average Reconstruction Loss Function**: [`rloss.m`](./rloss.m)
- **Example of Synthetic Data**: [`synthetic_data.m`](./synthetic_data.m)

## Datasets

The following datasets are used in the experiments:

- **[Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)**:
  - $M \in \mathbb{R}^{45211 \times 16}$
  - $A \in \mathbb{R}^{44401 \times 16}$
  - $B \in \mathbb{R}^{810 \times 16}$
  - Dataset script: [`bank_marketing.m`](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/bank_marketing.m)

- **[Crop Mapping](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)**:
  - $M \in \mathbb{R}^{325834 \times 173}$
  - $A \in \mathbb{R}^{39162 \times 173}$
  - $B \in \mathbb{R}^{286672 \times 173}$
  - Dataset script: [`crop_mapping.m`](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/crop_mapping.m)

- **[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**:
  - $M \in \mathbb{R}^{30000 \times 23}$
  - $A \in \mathbb{R}^{10599 \times 23}$
  - $B \in \mathbb{R}^{19401 \times 23}$
  - Dataset script: [`default_credit.m`](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/default_credit.m)

- **[Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/)**:
  - $M \in \mathbb{R}^{13232 \times 1764}$
  - $A \in \mathbb{R}^{2962 \times 1764}$
  - $B \in \mathbb{R}^{10270 \times 1764}$
  - Dataset script: [`LFW.m`](https://github.com/JunhuiShen/Fair-PCA-Eigenvalue-Optimization/blob/main/LFW.m)

Note: All data matrices are centered, meaning the mean of each matrix $A$ and $B$ is set to 0.

## Reference Paper

The **Fair PCA via SDP** algorithm is based on the paper _"[The Price of Fair PCA: One Extra Dimension](https://papers.nips.cc/paper_files/paper/2018/hash/cc4af25fa9d2d5c953496579b75f6f6c-Abstract.html)"_ by Samadi et al., presented at NeurIPS 2018. The implementation [`FPCAviaSDR.m`](./FPCAviaSDR.m) is adapted from their code, with all related functions integrated. You can access their code repository [here](https://github.com/samirasamadi/Fair-PCA?tab=readme-ov-file).

The **c-FPCA** algorithm is introduced in the paper _"[A novel approach for Fair Principal Component Analysis based on eigendecomposition](https://ieeexplore.ieee.org/document/10192331)"_ by Pelegrina and Duarte. The implementation [`c_FPCA.m`](./c_FPCA.m) is adapted from their code for constrained Fair PCA. You can access their code repository [here](https://github.com/GuilhermePelegrina/FPCA).

## Data Folder

The data folder needed to run the code may be accessed [here](https://drive.google.com/drive/u/1/folders/1xmdlEYPJDS7nwMQqbOoEuG3TCWLCBkUJ).
