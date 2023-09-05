import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from sklearn.metrics import pairwise_distances
from collections import deque
import imageio
import os
from io import BytesIO
from PIL import Image

def HDEmb(csv_path, output_folder, label_columns=1, data_columns=None, 
                                    distance_metric='cosine', iterations=5000000, 
                                    entropy_variable=1.0001, s=0.0000001, generate_gif=False, 
                                    delimiter=',', decimal='.', graph_label_column=0):
    """
    Perform dimensionality reduction on high-dimensional data.
    
    Parameters:
    - csv_path: Path to the .csv file containing the high-dimensional data.
    - output_folder: Path to the folder where output files will be saved.
    - label_columns: Number of columns in the DataFrame that are label columns.
    - data_columns: Number of columns in the DataFrame that are data columns.
    - distance_metric: Metric used for distance calculations.
    - iterations: Number of iterations for the optimization.
    - entropy_variable: Entropy parameter to control randomness.
    - s: Mean step value beneath which embedding should stop.
    - generate_gif: Whether to generate a GIF of the embedding process.
    - delimiter: Delimiter used in the CSV file. Default ','
    - decimal: Decimal used in the CSV file. Default '.'
    - graph_label_column. Default 0 - use the first label column for graphing

    Returns:
    - numpy array containing 2D embeddings.
    - Plots for initial and final embedding, and loss function.
    - Optional GIF.
    """
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read CSV file into a DataFrame and then convert to NumPy array
    df = pd.read_csv(csv_path, delimiter=delimiter, decimal=decimal)
    data_array = df.to_numpy()
    
    # Separate labels and features into different NumPy arrays
    labels_array = data_array[:, :label_columns]
    data_array = data_array[:, label_columns:data_columns]
    
    # Calculate distance matrix
    distance_matrix = pairwise_distances(data_array, metric=distance_metric)
    min_val, max_val = np.min(distance_matrix), np.max(distance_matrix)
    distance_matrix = (distance_matrix - min_val) / (max_val - min_val)

    # Initialize random 2D embeddings
    embeddings = 2 * np.random.rand(data_array.shape[0], 2) - 1

    # Extract labels for plotting (assuming labels are categorical)
    labels = pd.Series(labels_array[:, graph_label_column]).astype('category')

    # Initial Embedding Plot
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels)
    plt.title('Initial random Embedding')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{output_folder}/initial_embedding.png")
    plt.show()

    # Initialize optimization variables
    num_index = embeddings.shape[0] - 1
    control = deque(maxlen=1000)
    delta_values = []
    frames = []
    counter = 0

    # Optimization Loop
    for i in range(iterations):
        num1, num2, num3 = random.sample(range(num_index + 1), 3)
        coord1, coord2, coord3 = embeddings[num1], embeddings[num2], embeddings[num3]
        
        T = distance_matrix[num1, num2] / distance_matrix[num1, num3]
        E12, E13 = np.linalg.norm(coord2 - coord1), np.linalg.norm(coord3 - coord1)
        
        if E12 == 0 or E13 == 0: continue
        
        E = E12 / E13
        v = np.random.uniform(1, entropy_variable)
        
        if E != T:

            factor = 1 - E / T if E < T else 1 - T / E
            direction = -1 if E < T else 1
            delta = direction * v * 0.01 * factor * (coord2 - coord1) / E12
            embeddings[num1] += delta
        
        # Update loss and check for early stopping
        delta_norm = np.linalg.norm(delta)
        control.append(delta_norm)
        mc = np.mean(control)
        delta_values.append((i, mc))
        if mc < s: break
        
        # Periodic plotting and optional GIF generation
        counter += 1
        if counter == 10000:
            sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels)
            plt.title(f'Ongoing embedding with stepsize {mc:.5f}')
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            
            progress = (i / iterations) * 100
            print(f"Progress: {progress:.2f}%")
            
            if generate_gif:
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                im = Image.open(buf)
                frames.append(np.array(im))
            
            plt.show()
            counter = 0

    # Final Embedding Plot
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels)
    plt.title('Final Embedding')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{output_folder}/final_embedding.png")
    plt.show()

    # Loss Function Plot
    iterations, mean_step_size = zip(*delta_values)
    plt.plot(iterations, mean_step_size)
    plt.xlabel('Iteration')
    plt.ylabel('Mean Step Size')
    plt.savefig(f"{output_folder}/loss_function_plot.png")
    plt.show()

    # Optional GIF generation
    if generate_gif:
        gif_path = f"{output_folder}/embedding_animation.gif"
        imageio.mimsave(gif_path, frames, duration=0.07)

    return embeddings
