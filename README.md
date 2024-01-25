The work proposes a novel fair PCA algorithm using convex optimization.


Here are the data sets we consider: <br>
- Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing. The size of the matrix is $M \in \mathbb{R}^{45211 \times 16}$, $A \in \mathbb{R}^{44401 \times 16}$, and $B \in \mathbb{R}^{816 \times 16}$. <br>
- Crop mapping using fused optical-radar data set: https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set. The size of the matrix is $M \in \mathbb{R}^{325834 \times 173}$, $A \in \mathbb{R}^{39162 \times 173}$, and $B \in \mathbb{R}^{286672 \times 173}$..<be>
- Default of credit card clients: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. The size of the matrix is $M \in \mathbb{R}^{30000 \times 23}$, $A \in \mathbb{R}^{10599 \times 23}$, and $B \in \mathbb{R}^{19401 \times 23}$. <br>
- Labeled Faces in the Wild: https://vis-www.cs.umass.edu/lfw/. The size of the matrix is $M \in \mathbb{R}^{13232 \times 1764}$, $A \in \mathbb{R}^{2962 \times 1764}$, and $B \in \mathbb{R}^{10270 \times 1764}$.
If you would like to try our algorithm on a new data set, please note that our algorithm requires that the average value of matrix $A$ and $B$ be $0$.

The data folder needed to run the code may be accessed through: https://drive.google.com/drive/u/1/folders/1xmdlEYPJDS7nwMQqbOoEuG3TCWLCBkUJ
