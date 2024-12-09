# Crime GPT LLM Project

This project is based on building a Crime GPT Large Language Model (LLM) inspired by the "Build a Large Language Model (From Scratch)" workshop by Sebastian Raschka. The project involves various stages including data preparation, model architecture design, pretraining, and weight loading.

## Project Structure

### Notebooks

**02.ipynb**  
This notebook focuses on loading and preparing the crime novel dataset for training the GPT model.

**03.ipynb**  
This notebook covers the design and implementation of the GPT model architecture.

**04.ipynb**  
This notebook is dedicated to pretraining the GPT model on the prepared dataset.

**05_part-1.ipynb and 05_part-2.ipynb**  
These notebooks handle the loading of pretrained weights into the GPT model and demonstrate how to use the model for text generation.

### Scripts

**gpt_download.py**  
This script contains functions to download and extract GPT-2 model weights from OpenAI's servers.

**supplementary.py**  
Each supplementary.py file in the respective directories contains helper functions and classes used in the notebooks.

## Usage

1. **Data Preparation**: Run the notebook `02.ipynb` to load and preprocess the dataset.
2. **Model Architecture**: Use the notebook `03.ipynb` to design and implement the GPT model architecture.
3. **Pretraining**: Execute the notebook `04.ipynb` to pretrain the model on the dataset.
4. **Weight Loading**: Follow the steps in `05_part-1.ipynb` and `05_part-2.ipynb` to load pretrained weights and generate text samples.

## Requirements

- Python 3.x
- PyTorch
- TensorFlow
- NumPy
- Matplotlib
- tqdm
- transformers

## Installation

Install the required packages using pip:

```bash
pip install torch tensorflow numpy matplotlib tqdm transformers
```

## Acknowledgements

This project is inspired by the "Build a Large Language Model (From Scratch)" workshop by Sebastian Raschka. Special thanks to OpenAI for providing the GPT-2 model weights.

## License

This project is licensed under the MIT License. See the LICENSE file for details.



























































