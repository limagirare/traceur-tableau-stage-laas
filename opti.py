import cv2
import numpy as np
import os

# ==============================================================================
# RÉGLAGES
# ==============================================================================
NOM_IMAGE = "laas.png"

# TOLERANCE : Plus ce chiffre est grand, plus on supprime de points.
# 1.0 ou 1.5 est idéal pour un traceur (on garde les angles mais on vire les points inutiles en ligne droite).
TOLERANCE_SIMPLIFICATION = 1.2 

DOSSIER_DU_SCRIPT = os.path.dirname(os.path.abspath(__file__))
CHEMIN_IMAGE = os.path.join(DOSSIER_DU_SCRIPT, NOM_IMAGE)

if not os.path.exists(CHEMIN_IMAGE):
    print(f"ERREUR : Place 'laas.png' dans : {DOSSIER_DU_SCRIPT}")
    exit()

# 1. CHARGEMENT ET BORDURE DE SÉCURITÉ
img_gris = cv2.imread(CHEMIN_IMAGE, cv2.IMREAD_GRAYSCALE)
BORDURE = 10
img_blanche = cv2.copyMakeBorder(img_gris, BORDURE, BORDURE, BORDURE, BORDURE, cv2.BORDER_CONSTANT, value=255)
hauteur_image, _ = img_blanche.shape

# 2. SEUILLAGE POUR CAPTURER TOUTES LES COULEURS (MÊME LES VAGUES CLAIRES)
_, img_binaire = cv2.threshold(img_blanche, 220, 255, cv2.THRESH_BINARY_INV)

# 3. EXTRACTION DES CONTOURS NATIFS D'OPENCV (Idéal pour suivre l'ordre d'un tracé)
# RETR_EXTERNAL permet de ne prendre que l'extérieur des lignes sans faire de doublons aller-retour.
contours, _ = cv2.findContours(img_binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

points_arduino = []
points_geogebra = []
lignes_index = []
index_courant = 0

# 4. SIMPLIFICATION FORME PAR FORME
for i, c in enumerate(contours):
    # Algorithme de Douglas-Peucker pour réduire le nombre de points
    points_simplifies = cv2.approxPolyDP(c, TOLERANCE_SIMPLIFICATION, closed=True)
    
    nb_pts_forme = len(points_simplifies)
    if nb_pts_forme < 2:
        continue
        
    # Stockage de l'index pour savoir où commence et s'arrête chaque forme (Utile pour lever le feutre !)
    virgule = "," if i < len(contours) - 1 else ""
    lignes_index.append(f"  {{ {index_courant}, {nb_pts_forme} }}{virgule} // Forme {i}\n")
    
    for pt in points_simplifies:
        x_reel = int(pt[0][0] - BORDURE)
        y_reel = int(pt[0][1] - BORDURE)
        
        # Points normaux (Arduino)
        points_arduino.append((x_reel, y_reel))
        
        # Points inversés (GeoGebra)
        y_inverse = (hauteur_image - 2 * BORDURE) - y_reel
        points_geogebra.append((x_reel, y_inverse))
        
    index_courant += nb_pts_forme

# ==============================================================================
# 5. ÉCRITURE DU FICHIER ARDUINO (C++)
# ==============================================================================
code_c = "// --- CONFIGURATION POUR TRACEUR ARDUINO ---\n"
code_c += f"// Nombre de formes indépendantes (permet de savoir quand lever le stylo)\n"
code_c += f"const int NOMBRE_FORMES = {len(lignes_index)};\n"
code_c += "const int indexFormes[NOMBRE_FORMES][2] = {\n"
code_c += "".join(lignes_index)
code_c += "};\n\n"

code_c += f"// Liste des coordonnées réduites à envoyer aux moteurs\n"
code_c += f"const int TOTAL_POINTS = {len(points_arduino)};\n"
code_c += "const int listePoints[TOTAL_POINTS][2] = {\n"
for k, pt in enumerate(points_arduino):
    virgule = "," if k < len(points_arduino) - 1 else ""
    code_c += f"  {{ {pt[0]}, {pt[1]} }}{virgule}\n"
code_c += "};\n"

with open(os.path.join(DOSSIER_DU_SCRIPT, "code_arduino.txt"), "w", encoding="utf-8") as f_cpp:
    f_cpp.write(code_c)

# ==============================================================================
# 6. ÉCRITURE DU FICHIER GEOGEBRA (Pour vérification visuelle)
# ==============================================================================
texte_geogebra = ""
for pt in points_geogebra:
    texte_geogebra += f"{pt[0]}, {pt[1]}\n"

with open(os.path.join(DOSSIER_DU_SCRIPT, "points_geogebra.txt"), "w", encoding="utf-8") as f_geo:
    f_geo.write(texte_geogebra)

print("==========================================================")
print("             SIMPLIFICATION EFFECTUÉE !                   ")
print("==========================================================")
print(f"-> Nombre de points générés : {len(points_arduino)} (parfait pour la mémoire de l'Arduino)")
print(f"-> Nombre de formes (levées de stylo) : {len(lignes_index)}")
print("\nFichiers 'code_arduino.txt' et 'points_geogebra.txt' prêts.")