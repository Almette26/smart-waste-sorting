# Smart Waste Sorting - Computer Vision 3

Proiect realizat în cadrul laboratorului de Computer Vision 3.  
Sistem de clasificare multi-clasă a deșeurilor reciclabile (6 clase: cardboard, glass, metal, paper, plastic, trash) folosind rețele neuronale convoluționale și augmentare generativă.

## Cerințe de sistem și mediu
- **Python:** 3.10 / 3.11
- **Hardware testat:** AMD Ryzen 7 7700X, NVIDIA GeForce RTX 4060 (8 GB VRAM), 32 GB RAM

## Instalare și configurare mediu virtual
```bash
# Creare mediu virtual
python -m venv venv

# Activare mediu virtual (Windows)
.\venv\Scripts\activate
# Activare mediu virtual (Linux / macOS)
source venv/bin/activate

# Instalare dependențe cu versiuni fixate
pip install --upgrade pip
pip install -r requirements.txt