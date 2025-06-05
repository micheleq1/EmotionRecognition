import os

print("🔥 AVVIO PIPELINE CNN  🔥")


os.system("python segmentazione_ottimizzata.py")

os.system("python estrazione_feature_ottimizzata.py")

os.system("python train_model_cnn.py")

print("🚀 PIPELINE COMPLETATA con CNN 1D.")
