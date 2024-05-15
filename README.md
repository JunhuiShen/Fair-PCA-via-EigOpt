## Fair PCA Algorithm

The work proposes a novel fair PCA algorithm using eigenvalue optimization:

- The algorithm, **Fair PCA via Eigenvalue Optimization**, is implemented in [`fpca_Eigenvalue_Optimization.m`](./fpca_Eigenvalue_Optimization.m).
- Main function: [`main.m`](./main.m).
- Reconstruction loss function: [`rloss.m`](./rloss.m).
- Examples of synthetic data: [`synthetic_data.m`](./synthetic_data.m) and [`synthetic_data2.m`](./synthetic_data2.m).

## Datasets

Here are the datasets we consider:

- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing): Matrix sizes are $M \in \mathbb{R}^{45211 \times 16}$, $A \in \mathbb{R}^{44401 \times 16}$, and $B \in \mathbb{R}^{810 \times 16}$.
- [Crop mapping using fused optical-radar data set](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set): Matrix sizes are $M \in \mathbb{R}^{325834 \times 173}$, $A \in \mathbb{R}^{39162 \times 173}$, and $B \in \mathbb{R}^{286672 \times 173}$.
- [Default of credit card clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): Matrix sizes are $M \in \mathbb{R}^{30000 \times 23}$, $A \in \mathbb{R}^{10599 \times 23}$, and $B \in \mathbb{R}^{19401 \times 23}$.
- [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/): Matrix sizes are $M \in \mathbb{R}^{13232 \times 1764}$, $A \in \mathbb{R}^{2962 \times 1764}$, and $B \in \mathbb{R}^{10270 \times 1764}$.

Note: The mean value of every matrix $A$ and $B$ is set to $0$.

## Reference Paper

The code for the Fair PCA via LP algorithm is introduced in the paper "[The Price of Fair PCA: One Extra Dimension](https://arxiv.org/abs/1811.00103)" by Samadi S., Tantipongpipat U., Morgenstern J., Singh M., and Vempala S., NeurIPS, 2018. Our implementation, `fpca_LP.m`, is adapted from their code.

You can access the code repository [here](https://github.com/samirasamadi/Fair-PCA?tab=readme-ov-file).

## Data Folder

The data folder needed to run the code may be accessed [here](https://drive.google.com/drive/u/1/folders/1xmdlEYPJDS7nwMQqbOoEuG3TCWLCBkUJ).
