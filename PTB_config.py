# config.py

# =============================================================================
# Data Path Configuration
# =============================================================================
DATA_ROOT = './data/ptb'  # 默认指向PTB数据集目录
DATA_MITBIH_ROOT = './data/mitbih'

# =============================================================================
# Model Configurations (Optimized for PTB Dataset: 187 points, 2 classes)
# =============================================================================

# 基础常量
PTB_NUM_CLASSES = 2
PTB_INPUT_LEN = 187
PTB_FS = 125  # PTB CSV版本的常用等效频率

ECGNET_CONFIG = {
    'struct': [15, 17, 19, 21],
    'in_channels': 8,
    'fixed_kernel_size': 17,
    'num_classes': PTB_NUM_CLASSES,
}

SE_ECGNET_CONFIG = {
    'struct': [(1, 3), (1, 5), (1, 7)],
    'num_classes': PTB_NUM_CLASSES,
}

BIRCNN_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
}

RESNET_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
}

LITEECGNET_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'base_channels': 32
}

DEEPECGNET_CONFIG = {
    'in_ch': 1,
    'base_ch': 64,
    'd_model': 128,
    'n_transformer_layers': 2,
    'nhead': 4,
    'dim_ff': 256,
    'dropout': 0.1,
    'num_classes': PTB_NUM_CLASSES,
    'hidden': 128,
    'layers': 2,
    'heads': 4,
    'ff': 256
}

LDCNN_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'input_len': PTB_INPUT_LEN,  # 关键修改：对应PTB的187个点
}

# =============================================================================
# IMBECGNET 系列配置
# =============================================================================

IMBECGNET_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'attn_heads': 4,
}

IMBECGNET_IMPROVED_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'attn_heads': 4,
}

IMBECGNET_ENHANCED_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'attn_heads': 8,
}

IMBECGNET_CGHC_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'attn_heads': 4,
}

# =============================================================================
# 可选：轻量化变体配置 (如果需要微调)
# =============================================================================

PTB_SMALL_CONFIG = {
    'num_classes': PTB_NUM_CLASSES,
    'fs': PTB_FS,
    'segment_len': PTB_INPUT_LEN,
    'attn_heads': 2,
    'sinc_out_channels': 4,
    'sinc_kernel_size': 21,
    'conv7_out_channels': 8,
    'conv7_kernel_size': 5,
    'pw32_out_channels': 16,
    'stage1_expand': 1,
    'stage1_kernel': 3,
    'stage2_expand': 1,
    'stage2_kernel': 3,
    'stage3_expand': 1,
    'stage3_kernel': 3,
    'tag_kernel': 5,
    'dropout_rate': 0.05
}