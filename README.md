The work proposes a novel fair PCA algorithm using eigenvalue optimization. 


Here are the data sets we consider: <br>
- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing): The size of the matrix is $M \in \mathbb{R}^{45211 \times 16}$, $A \in \mathbb{R}^{44401 \times 16}$, and $B \in \mathbb{R}^{810 \times 16}$. <br>
- [Crop mapping using fused optical-radar data set](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set): The size of the matrix is $M \in \mathbb{R}^{325834 \times 173}$, $A \in \mathbb{R}^{39162 \times 173}$, and $B \in \mathbb{R}^{286672 \times 173}$..<be>
- [Default of credit card clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients): The size of the matrix is $M \in \mathbb{R}^{30000 \times 23}$, $A \in \mathbb{R}^{10599 \times 23}$, and $B \in \mathbb{R}^{19401 \times 23}$. <br>
- [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/): The size of the matrix is $M \in \mathbb{R}^{13232 \times 1764}$, $A \in \mathbb{R}^{2962 \times 1764}$, and $B \in \mathbb{R}^{10270 \times 1764}$. <be>

Note that the mean value of every matrix $A$ and $B$ is all set to $0$. <br>

The [code](https://github.com/samirasamadi/Fair-PCA?tab=readme-ov-file) for the Fair PCA via LP algorithm is introduced in the paper "[The Price of Fair PCA: One Extra Dimension](https://arxiv.org/abs/1811.00103)" by Samadi S., Tantipongpipat U., Morgenstern J., Singh M., and Vempala S., NeurIPS, 2018. Our function "fpca_LP.m" is a mixture of their code.


The data folder needed to run the code may be accessed [here](https://drive.google.com/drive/u/1/folders/1xmdlEYPJDS7nwMQqbOoEuG3TCWLCBkUJ).
