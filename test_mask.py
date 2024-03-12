import torch

# Liste de longueurs k pour chaque ligne
k = [3, 5, 2, 4]
max_k = max(k)

# Conversion de k en tenseur PyTorch
k_tensor = torch.tensor(k)[:, None]  # Transforme en colonne pour le broadcasting

# Création d'un tenseur d'indices
indices = torch.arange(max_k)[None, :]  # Transforme en ligne pour le broadcasting

# Création du masque par comparaison, puis conversion en tenseur de 1 et 0
mask = (indices < k_tensor).int()

print(mask)