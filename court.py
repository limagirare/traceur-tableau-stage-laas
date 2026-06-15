from PIL import Image
import numpy as np
import random

img = Image.open("laas.png").convert("L")
pixels = np.array(img)

hauteur = pixels.shape[0]
seuil = 128
points = []

for y in range(hauteur):
    for x in range(pixels.shape[1]):
        if pixels[y, x] < seuil:
            points.append((x, hauteur - 1 - y))

nombre_points = 500

if len(points) > nombre_points:
    points = random.sample(points, nombre_points)

with open("points.txt", "w", encoding="utf-8") as fichier:
    for x, y in points:
        fichier.write(f"{x},{y}\n")

print(f"{len(points)} points enregistrés dans points.txt")