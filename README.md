# ECG Lightweight Anomaly Detector

A comprehensive deep learning framework for ECG anomaly detection using multiple state-of-the-art neural network architectures. This project implements various models for classifying ECG signals into different arrhythmia types based on the MIT-BIH Arrhythmia Database.

## 🚀 Features

- **Multiple Model Architectures**: LiteECGNet, ECGNet, SE_ECGNet, BiRCNN, ResNet, DeepECGNet, LDCNN
- **Stratified Dataset Support**: Both stratified and random dataset splitting strategies
- **Comprehensive Evaluation**: Model metrics including parameters, FLOPs, inference latency, and memory usage
- **Flexible Configuration**: Easy-to-modify hyperparameters and model configurations
- **Caching System**: Preprocessed data caching for faster training
- **Reproducible Results**: Deterministic training with fixed random seeds

## 📋 Requirements

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn
pip install wfdb
pip install tqdm
pip install psutil
pip install thop
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

## 📁 Project Structure

```
LiteECG-Net-git/
├── data/
│   └── mitbih/                    # MIT-BIH Arrhythmia Database files
│       ├── *.dat                  # ECG signal data files
│       ├── *.atr                  # Annotation files
│       ├── *.hea                  # Header files
│       └── cache_*.npz            # Preprocessed data cache files
├── models/                        # Neural network model implementations
│   ├── __init__.py
│   ├── LiteECGNet.py             # Lightweight ECG network with attention
│   ├── ECGNet.py                 # Original ECGNet architecture
│   ├── SE_ECGNet.py              # Squeeze-and-Excitation ECGNet
│   ├── BiRCNN.py                 # Bidirectional RNN with CNN
│   ├── ResNet.py                 # 1D ResNet for ECG signals
│   ├── DeepECGNet.py             # Deep ECG network with transformer
│   └── LDCNN.py                  # Lightweight Deep CNN
├── config.py                     # Model and training configurations
├── dataset.py                    # Random dataset splitting implementation
├── stratified_dataset.py         # Stratified dataset splitting implementation
├── trainer.py                    # Main training and evaluation script
├── utils.py                      # Utility functions for data processing
├── run_all_models.bat            # Batch script to run all models
├── run_all_models-nostra.bat     # Batch script without stratified sampling
└── quick_test.bat                # Quick test script (1 epoch per model)
```

## 🏃‍♂️ Quick Start

### 1. Data Preparation

Download the MIT-BIH Arrhythmia Database and place the files in `data/mitbih/` directory. The required files are:
- `*.dat` - ECG signal data
- `*.atr` - Annotation files  
- `*.hea` - Header files

**Note**: Cache files (`.npz`) and trained models (`.pt`) are not included in this repository due to size limitations. These will be automatically generated during the first run with the `--cache` flag.

### 2. Basic Usage

#### Train a single model:
```bash
python trainer.py --model liteecgnet --epochs 30 --batch-size 256 --lr 3e-3 --cache
```

#### Train all models (Windows):
```bash
run_all_models.bat
```

#### Quick test all models (1 epoch each):
```bash
quick_test.bat
```

### 3. Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `liteecgnet` | Model to train (`liteecgnet`, `ecgnet`, `se_ecgnet`, `bircnn`, `resnet`, `deepecgnet`, `ldcnn`) |
| `--root` | str | `./data/mitbih` | Path to MIT-BIH directory |
| `--epochs` | int | `30` | Number of training epochs |
| `--batch-size` | int | `256` | Batch size |
| `--lr` | float | `3e-3` | Learning rate |
| `--weight-decay` | float | `1e-4` | Weight decay for regularization |
| `--segment-sec` | float | `2.0` | ECG segment duration in seconds |
| `--fs` | int | `360` | Sampling frequency (Hz) |
| `--cache` | flag | `False` | Enable data caching |
| `--class-limit` | int | `None` | Per-class sample limit for training |
| `--patience` | int | `7` | Early stopping patience |
| `--focal` | flag | `False` | Use Focal Loss instead of CrossEntropy |
| `--no-stratified` | flag | `False` | Disable stratified sampling |
| `--device` | str | `cuda` | Device to use (`cuda` or `cpu`) |


## 📊 Dataset Information

### MIT-BIH Arrhythmia Database
- **Total Records**: 48 ECG recordings
- **Sampling Rate**: 360 Hz
- **Duration**: 30 minutes per recording
- **Classes**: 4-class classification
  - **N**: Normal beats (N, L, R, e, j)
  - **SVEB**: Supraventricular ectopic beats (A, a, J, S)
  - **VEB**: Ventricular ectopic beats (V, E)
  - **Other**: Other beats (P, F, /, f, Q, x, etc.)

### Data Splitting Strategies
1. **Stratified Sampling**: Maintains class distribution across splits
2. **Random Sampling**: Random record-based splitting (7:1.5:1.5 ratio)

## 🔧 Configuration

### Model Configurations
Edit `config.py` to modify model architectures:

```python
LITEECGNET_CONFIG = {
    'num_classes': 4,
    'fs': 360,
    'segment_len': 720,
    'base_channels': 32  # Adjust for model size
}
```

### Training Parameters
Modify default values in `trainer.py` or use command line arguments.

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Classification Metrics**: Accuracy, F1-score (macro), Precision, Recall
- **Model Metrics**: Parameter count, model size, FLOPs, inference latency
- **Training Metrics**: Loss curves, epoch times, memory usage
- **Confusion Matrix**: Detailed class-wise performance

## 📁 Output Files

Training results are saved in `data/records/`:
- `ecg_lightweight_{model}_{timestamp}.pt` - Trained model weights
- `eval_report_{model}_{timestamp}.json` - Detailed evaluation report

## 🛠️ Customization

### Adding New Models
1. Create model class in `models/` directory
2. Add configuration in `config.py`
3. Import and add to model selection in `trainer.py`

### Custom Loss Functions
Implement custom loss functions in `trainer.py` and modify the criterion selection logic.

### Data Augmentation
Add augmentation techniques in the dataset classes (`dataset.py`, `stratified_dataset.py`).

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 128`
   - Use CPU: `--device cpu`

2. **Data Loading Errors**
   - Ensure MIT-BIH files are in correct format
   - Check file permissions and paths

3. **Model Import Errors**
   - Verify all dependencies are installed
   - Check Python path and environment

### Performance Tips

1. **Enable Caching**: Use `--cache` flag for faster subsequent runs
2. **GPU Memory**: Monitor GPU usage and adjust batch size accordingly
3. **Data Loading**: Use appropriate `--num-workers` for your system

## 📚 References

- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/1.0.0/
- ECGNet: https://arxiv.org/abs/1804.00712
- Squeeze-and-Excitation Networks: https://arxiv.org/abs/1709.01507

## 📄 License

This project is open source. Please cite appropriately if used in research.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

---

**Note**: This framework is designed for research and educational purposes. For clinical applications, please ensure proper validation and regulatory compliance.
