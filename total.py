import cv2
import numpy as np
import contourpy
import os

NOM_IMAGE = "laas.png"  
NB_POINTS_TOTAL = 900  # On augmente à 900 pour une précision parfaite de toutes les lettres

DOSSIER_DU_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CHEMIN_COMPLET_IMAGE = os.path.join(DOSSIER_DU_SCRIPT, NOM_IMAGE)

if not os.path.exists(CHEMIN_COMPLET_IMAGE):
    print(f"ERREUR : Image introuvable dans {DOSSIER_DU_SCRIPT}")
    exit()

# 1. CHARGEMENT ET AJOUT D'UNE BORDURE BLANCHE DE SÉCURITÉ
img_gris = cv2.imread(CHEMIN_COMPLET_IMAGE, cv2.IMREAD_GRAYSCALE)

# On ajoute 10 pixels blancs tout autour pour décoller le logo des bords
BORDURE = 10
img_blanche = cv2.copyMakeBorder(img_gris, BORDURE, BORDURE, BORDURE, BORDURE, cv2.BORDER_CONSTANT, value=255)
hauteur_image, largeur_image = img_blanche.shape

# 2. SEUILLAGE ET NETTOYAGE
_, img_binaire = cv2.threshold(img_blanche, 220, 255, cv2.THRESH_BINARY_INV)

# Épaississement très léger pour lier le "S" s'il est trop fin
noyau = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
img_propre = cv2.morphologyEx(img_binaire, cv2.MORPH_CLOSE, noyau)

# 3. EXTRACTION DES CONTOURS
generateur = contourpy.contour_generator(z=img_propre)
toutes_les_formes = generateur.lines(127)

perimetres = [np.sum(np.sqrt(np.sum(np.diff(f, axis=0)**2, axis=1))) for f in toutes_les_formes]
perimetre_total = sum(perimetres)

# 4. ÉCHANTILLONNAGE ET CORRECTION DE L'AXE Y
points_geogebra = []
for forme, perim in zip(toutes_les_formes, perimetres):
    nb_pts_forme = int(round((perim / perimetre_total) * NB_POINTS_TOTAL))
    if nb_pts_forme < 3: 
        nb_pts_forme = 3
    
    for j in range(nb_pts_forme):
        frac = (j / nb_pts_forme) * len(forme)
        idx = int(frac)
        reste = frac - idx
        pt = forme[idx % len(forme)] + reste * (forme[(idx + 1) % len(forme)] - forme[idx % len(forme)])
        
        # On enlève la bordure pour retrouver les vraies coordonnées de l'image
        x_reel = int(round(pt[0] - BORDURE))
        y_reel = int(round(pt[1] - BORDURE))
        
        # Inversion de l'axe Y pour GeoGebra
        y_inverse = (hauteur_image - 2 * BORDURE) - y_reel
        
        points_geogebra.append((x_reel, y_inverse))

# 5. AFFICHAGE DE SÉCURITÉ DANS LA CONSOLE
print("\n" + "="*60)
print(f" COPIE LES {len(points_geogebra)} POINTS CI-DESSOUS POUR GEOGEBRA :")
print("="*60 + "\n")

for pt in points_geogebra:
    print(f"{pt[0]}, {pt[1]}")

print("\n" + "="*60)
print(" SÉLECTIONNE TOUT LE BLOC DE CHIFFRES, FAIS Ctrl+C ET METS-LES DANS A1")
print("="*60)
input()