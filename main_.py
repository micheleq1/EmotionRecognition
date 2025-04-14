import os

print("🔥 AVVIO PIPELINE COMPLETA 🔥")

os.system("python segmentazione_ottimizzata.py")
os.system("python estrazione_feature_ottimizzata.py")
os.system("python train_model.py")

print("🚀 PIPELINE COMPLETATA.")
