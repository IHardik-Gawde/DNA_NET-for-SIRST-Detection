# Infrared Small Target Detection and Segmentation

This repository presents a deep learning–based framework for **infrared small target detection and segmentation**.  
The work evaluates multiple pretrained checkpoints on standard **SIRST datasets**, focusing on the trade-off between **segmentation accuracy, detection probability, and false alarm rate**.

The project is structured for **clarity, reproducibility, and extensibility**, making it suitable for both research and practical deployment scenarios.

---

## Key Contributions

- Evaluation of multiple pretrained SIRST checkpoints on a common dataset
- Quantitative comparison using **Mean IoU**, **Probability of Detection (PD)**, and **False Alarm Rate (FA)**
- Analysis of detection–false alarm trade-offs
- A final model optimized for **significantly reduced false alarms**
- Modular and reproducible project structure

---

## Project Structure
DNA_NET_SIRST_Detection/
│
├── checkpoints/
│ ├── DNANet_model.py
│ ├── pretrain_DNANet_model.tar
│ ├── mIoU_DNANet_NUAA-SIRST_.tar
│ ├── mIoU_DNANet_NUST-SIRST_.tar
│ ├── mIoU_DNANet_NUDT-SIRST_*.tar
│ └── pycache/
│
├── data_256/
│ ├── data1/
│ ├── data2/
│ └── data3/
│
├── results/
│ └── results.png
│
├── src/
│ ├── compute_metric.py
│ ├── DNANet_model.py
│ ├── load_model.py
│ ├── new_inference.py
│ ├── resize_images_mask.py
│ ├── shape.py
│ └── visualize.py
│
├── README.md


