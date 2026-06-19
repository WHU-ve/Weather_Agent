# WeatherBench Baseline Results (pyiqa, RGB channel)

All PSNR/SSIM computed using `pyiqa.create_metric`, default settings (RGB). Images loaded from each method's `results/` directory.

## All-in-One Methods

### WGWS-Net S2 (Rain1400 + RESIDE + Snow100K)
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| rain        | 25.16    | 0.7934 | 200     |
| haze        | 13.29    | 0.5891 | 200     |
| snow        | 21.73    | 0.7799 | 200     |
| **TOTAL**   | **20.06** | **0.7208** | **600** |

### WGWS-Net S3 (SPA+ + REVIDE + RealSnow)
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| rain        | 29.59    | 0.9033 | 200     |
| haze        | 12.07    | 0.5535 | 200     |
| snow        | 20.60    | 0.7315 | 200     |
| **TOTAL**   | **20.75** | **0.7294** | **600** |

### MWFormer_L
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| rain        | 26.43    | 0.8317 | 200     |
| haze        | 11.94    | 0.5542 | 200     |
| snow        | 21.98    | 0.7808 | 200     |
| **TOTAL**   | **20.12** | **0.7223** | **600** |

### MWFormer_real
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| rain        | 26.54    | 0.8349 | 200     |
| haze        | 12.75    | 0.5715 | 200     |
| snow        | 21.60    | 0.7779 | 200     |
| **TOTAL**   | **20.30** | **0.7281** | **600** |

## Task-Specific Methods

### DRSformer Rain200H (Derain only)
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| rain        | 28.40    | 0.8786 | 200     |

### DehazeFormer-m (Dehaze only)
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| haze        | 13.76    | 0.5962 | 200     |

### SnowFormer CSD (Desnow only)
| Weather Type | PSNR(dB) | SSIM   | Samples |
|-------------|----------|--------|---------|
| snow        | 17.81    | 0.6762 | 200     |

## Comparison Table
| Weather Type | All-in-One | | Task-Specific | |
|-------------|------------|------------|--------------|----|
|             | WGWS-Net S2 | MWFormer_L | DRSformer R200H | DehazeFormer-m / SnowFormer CSD |
| rain        | 25.16 / 0.7934 | 26.43 / 0.8317 | **28.40 / 0.8786** | — |
| haze        | 13.29 / 0.5891 | 11.94 / 0.5542 | — | **13.76 / 0.5962** |
| snow        | 21.73 / 0.7799 | **21.98 / 0.7808** | — | 17.81 / 0.6762 |
| **TOTAL**   | 20.06 / 0.7208 | 20.12 / 0.7223 | — | — |

---

*Evaluation: pyiqa (PSNR + SSIM, RGB channel, default settings)*
