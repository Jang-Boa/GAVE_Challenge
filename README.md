# GAVE Challenge – BOA Inference Pipeline

This repository provides the **inference code and models** used to reproduce the preliminary submission results for the GAVE Challenge.  
The models are exactly the same as those used in the official preliminary submission.

---

## 📦 Requirements

- Python >= 3.8  
- PyTorch >= 1.11 (CUDA 11.x support)  
- segmentation-models-pytorch  
- numpy, opencv-python, pillow, tqdm, matplotlib  

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 0. Download model weights
Model weights are stored in Google Drive. Please download them from the following link and place them in the working directory before running inference:

[Google Drive – Model Weights](https://drive.google.com/drive/folders/12JnGXZCmm7C03nVB8ELtBqIhezatnD8g?usp=sharing)

### 1. Select GPU (optional)
```bash
export CUDA_VISIBLE_DEVICES=0    # Use only GPU 0
```

### 2. Run inference
```bash
python infer.py \
  --input "/path/to/validation/images" \
  --output "./BOA" \
  --vessel-weight "Vessel_model.pth" \
  --vessel-support-weight "M-FunFound_Vessel.pth" \
  --av-weight "AV_model.pth" \
  --optic-weight "optic_mask_1024.pth" \
  --use-cuda --device 0
```

### Main arguments
- `--input` : Directory containing input images (searched recursively)  
- `--output` : Output directory (created automatically if missing)  
- `--vessel-weight`, `--vessel-support-weight`, `--av-weight`, `--optic-weight` : Model checkpoint paths  
- `--use-cuda` : Use GPU (default = CPU)  
- `--device` : GPU index to use (e.g., 0, 1, …)  

---

## 📂 Output Structure

After inference, the following structure will be created under `--output` (e.g., `./BOA`):

```
BOA/
 ├─ Task1_2/          # Intermediate vessel-related outputs
 ├─ Task3/AVR.txt     # Final AVR values (filename + value)
```

> Temporary folders (`Optic`, `Contour`, `Vessel`, `Vis`) are automatically removed after AVR computation.  
> The final result is stored in `Task3/AVR.txt`.

---

## 📝 Notes

1. The models provided here are **exactly those used in the official preliminary submission**. The same models must be used for the final stage according to the rules.  
2. All paths are configurable. If models are too large to submit directly, a cloud storage link and SHA256 checksum will be provided instead.  
3. `requirements.txt` lists all dependencies with versions used in our environment.  

---

## 📧 Contact

- Team: **BOA**  
- Contact: **boajang97@gmail.com**
