
# HDEmb: High-Dimensional Distance Embedding

HDEmb is a Python module designed for dimensionality reduction on high-dimensional data. It employs iterative triangular adjustments based on high dimensional distances to project the data into 2D space. It sets out to embed the high dimensional distances without skewing or exaggerating values.

## Features

- Dimensionality reduction on high-dimensional data.
- Produces visual plots for initial and final embeddings as well as a the mean adjustment step size throughout the embedding process as a loss function.
- Produces visual plots for initial and final embeddings as well as the mean adjustment step size throughout the embedding process as a loss function.

- Option to generate a GIF animation of the embedding process.
- Customizable parameters: 
-- distance metric: for example cosine, euclidean, manhattan, minkowski. 
-- iterations, can be adjusted based on the loss function
-- entropy variable, to avoid the embedding getting stuck in local minima. 

## Installation

Ensure you have the required libraries installed:
```bash
pip install matplotlib seaborn numpy pandas scikit-learn imageio PIL
```

Clone the repository and navigate to its directory:
```bash
git clone https://github.com/Clostridioides/HDEmb
cd HDEmb
```

## Usage

Here's a basic usage example:

```python
from hdemb import HDEmb

embeddings = HDEmb(
    csv_path="Data.csv", 
    output_folder="Output", 
    label_columns=6,
    iterations=5000000,
    generate_gif=True,
    delimiter=';',
    decimal=',',
    graph_label_column=5
)
```

For detailed function parameters and their explanations, refer to the comments in `hdemb.py`.

## Author

Adrian Fehn - 04.09.2023

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
