# Neural Forge

Neural Forge is a Python package that makes it easy to train and use transformer models using TensorFlow. The package provides a simple and intuitive API for defining, training, and evaluating transformer models, as well as saving and loading trained models.

## Installation

To install Neural Forge, simply run the following command:

```shell
!pip install NeuralForge
```

## Usage

Here's an example of how to use Neural Forge to train a transformer model:

```python
from neuralforge import Transformer

# Define the model architecture
model = Transformer(num_layers=6, d_model=512, dff=2048, num_heads=8)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(input_data, target_data, epochs=10)

# Save the trained model
model.save('transformer.h5')
```
You can also use Neural Forge to load a pre-trained model:
```python
from neuralforge import Transformer

# Load the trained model
model = Transformer.load('transformer.h5')

# Evaluate the model
model.evaluate(test_data)
```
## API Reference

The package provides the following functions and classes:

- Transformer(num_layers, d_model, dff, num_heads): This class is used to create a transformer model with the given number of layers, model dimension, feed-forward dimension, and number of heads.

- compile(optimizer, loss): This method is used to compile the model with the given optimizer and loss function.

- fit(input_data, target_data, epochs): This method is used to train the model on the input data and target data for the given number of epochs.

- evaluate(test_data): This method is used to evaluate the model on the test data.

- save(filepath): This method is used to save the trained model to the given file path.

- load(filepath): This method is used to load a pre-trained model from the given file path.
## Contributing
We welcome contributions to Neural Forge! Please open an issue or pull request on GitHub if you would like to contribute.

## License
### Apache 2.0 License
[![License](https://img.shields.io/badge/License-Apache_2.0-yellowgreen.svg)](https://opensource.org/licenses/Apache-2.0)  

Please note that this is just an example readme.md and the package and the functions might not exist and the package is not tested.
