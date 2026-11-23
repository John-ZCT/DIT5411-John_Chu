# Results (32x32 px, 240 training images per character)

Training images: 3,135,600. Test images: 162,550 (not augmented, also used as validation data).
GPU: Tesla T4. Subset: none (all 13,065 classes).

| Model | Parameters | Epochs run (best) | Training time | Test top-1 | Test top-5 |
|---|---:|---:|---:|---:|---:|
| mlp | 7,489,801 | 3 (3) | 3.3 min | 58.47% | 82.48% |
| cnn_simple | 7,844,105 | 4 (4) | 11.6 min | 85.56% | 97.58% |
| **cnn_vgg** | **7,058,665** | **5 (5)** | **30.8 min** | **96.65%** | **99.73%** |

Best model: **cnn_vgg**
