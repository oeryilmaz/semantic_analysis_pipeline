# Semantic Analysis Pipeline

A robust Natural Language Processing (NLP) pipeline that analyzes semantic relationships in text corpora using word embeddings and econometric validation techniques. This project stands out by applying rigorous statistical methods from economics to validate and interpret semantic patterns, going beyond traditional NLP approaches.

## Why This Project?
I built parts of the script while following NLP courses at ETH Zurich during the first year of my PhD in Economics. This is a polished version that is highly customizable. 

Traditional NLP approaches to semantic analysis often lack rigorous statistical validation. This pipeline bridges that gap by:
- Quantifying effect sizes for semantic relationships
- Implementing additional validation strategies (t-tests + standardized differences)
- Providing comprehensive distribution analysis
- Enabling interpretable evaluation of semantic patterns

[📚 Read the detailed methodology here](methodology.md)

## Prerequisites

- Python 3.8 or higher
- At least 3GB of free disk space for models and processed data
- Basic understanding of NLP concepts
- Text corpus in .txt format (sample corpus available [here](https://drive.google.com/file/d/1uPlm0JtJd9VipdUH7f-l1LjQ93VYRyAt/view))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/oeryilmaz/semantic_analysis_pipeline.git
   cd semantic_analysis_pipeline
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Prepare Your Data**
   - Place your text files in `data/Project_Gutenberg/txt/`
   - The pipeline supports both UTF-8 and Latin-1 encoded files
   - Required directories are created automatically

2. **Configure Analysis**
   - Review and adjust semantic categories in `config.py`
   - Default categories include: governance, rights, conflict, democracy, economy
   - Modify embedding parameters if needed (vector size, window size, etc.)

3. **Run the Pipeline**
   ```bash
   python main.py
   ```

## Example Output

After running the pipeline, you'll see analysis results like:

```
Category: democracy
Effect size: 0.842
Semantic distinctiveness: 0.234

Within-category exemplar pairs:
  strongest: citizen-vote (0.876)
  weakest: election-people (0.543)

Cross-category exemplar pairs:
  strongest: citizen-rights (0.654)
  weakest: vote-market (0.123)
```

## Project Structure

```
semantic-analysis-pipeline/
├── config.py               # Configuration and parameters
├── embedding_training.py   # Word2Vec training module
├── main.py                # Pipeline orchestration
├── paths.py               # Path management
├── semantic_analysis.py   # Statistical analysis
├── text_processing.py     # Corpus preprocessing
└── requirements.txt       # Dependencies
```

## Customization

The pipeline is highly configurable through `config.py`:

- `SEMANTIC_CATEGORIES`: Define your own word groups for analysis
- `MODEL_TRAINING_PARAMS`: Adjust Word2Vec training parameters
- `PREPROCESSING_PARAMS`: Modify text processing settings
- `SEMANTIC_ANALYSIS_PARAMS`: Configure analysis thresholds

## Troubleshooting

Common issues and solutions:

1. **Memory Errors**
   - Reduce `sample_size` in `PREPROCESSING_PARAMS`
   - Decrease `vector_size` in `MODEL_TRAINING_PARAMS`

2. **Runtime Warnings**
   - Ensure minimum word occurrences with `min_count` parameter
   - Check corpus size meets minimum requirements

3. **Missing Directories**
   - Directories are created automatically
   - Check write permissions if errors occur

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Onur Eryilmaz
- GitHub: [@oeryilmaz](https://github.com/oeryilmaz)
- LinkedIn: [Onur Eryilmaz](https://linkedin.com/in/onureryilmaz/)

## Acknowledgments

- Project Gutenberg corpus from Nagaraj, A. and Kejriwal, M. (2022), "Dataset for studying gender disparity in English literary texts", Data in Brief, 41, p.107905
- Word2Vec implementation based on Mikolov et al. (2013), "Efficient estimation of word representations in vector space", arXiv:1301.3781