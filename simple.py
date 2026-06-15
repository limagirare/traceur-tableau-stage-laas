import cv2
import numpy as np
import contourpy
import os
import matplotlib.pyplot as plt

# ==============================================================================
# RÉGLAGES DU SCRIPT
NOM_IMAGE = "laas.png"  # nom exact de image ici
SEUIL = 127                     # Sensibilité (0 à 255)
NB_POINTS_TOTAL = 500           # Nombre de points 

# On vérifie que l'image est bien dans le dossier
if not os.path.exists(NOM_IMAGE):
    print(f"ERREUR : fichier '{NOM_IMAGE}' introuvable.")
    print(f"Vérifier qu'il est bien rangé dans le dossier : {os.getcwd()}")
    input("\nAppuyer sur Entrée pour quitter...")
    exit()

# ==============================================================================
# 1. RECHERCHE DES LIGNES AVEC CONTOURPY
# 
# On charge la photo en noir, blanc et gris
img_gris = cv2.imread(NOM_IMAGE, cv2.IMREAD_GRAYSCALE)

# On démarre le moteur de calcul de ContourPy
generateur = contourpy.contour_generator(z=img_gris)

# ContourPy extrait toutes les courbes qui suivent la valeur du SEUIL
toutes_les_formes = generateur.lines(SEUIL)

# ==============================================================================
# 2. MESURE DES FORMES (THÉORÈME DE PYTHAGORE)

perimetres = []
for forme in toutes_les_formes:
    # On calcule l'écart entre chaque pixel pour mesurer la longueur de la forme
    ecarts = np.diff(forme, axis=0)
    longueur = np.sum(np.sqrt(np.sum(ecarts**2, axis=1)))
    perimetres.append(longueur)

# On fait la somme de toutes les longueurs pour avoir le périmètre total
perimetre_total = sum(perimetres)

# ==============================================================================
# 3. CRÉATION DES POINTS RÉGULIERS

index_courant = 0
points_calcules = []
lignes_index = []

for i, (forme, perim) in enumerate(zip(toutes_les_formes, perimetres)):
    # plus une forme est grande, plus on lui donne de points
    nb_pts_forme = int(round((perim / perimetre_total) * NB_POINTS_TOTAL))
    if nb_pts_forme < 3: 
        nb_pts_forme = 3  # Minimum 3 points pour dessiner une forme fermée
    
    # On prépare la ligne de code C++ pour l'index de cette forme
    virgule = "," if i < len(toutes_les_formes) - 1 else ""
    lignes_index.append(f"  {{ {index_courant}, {nb_pts_forme} }}{virgule} // Forme {i}\n")
    
    # Calcul des coordonnées exactes le long de la ligne
    for j in range(nb_pts_forme):
        fraction_distance = (j / nb_pts_forme) * len(forme)
        idx_sommet = int(fraction_distance)
        reste = fraction_distance - idx_sommet
        
        # Interpolation
        pt1 = forme[idx_sommet % len(forme)]
        pt2 = forme[(idx_sommet + 1) % len(forme)]
        pt_final = pt1 + reste * (pt2 - pt1)
        
        # On arrondit pour obtenir des numéros de pixels entiers [X, Y]
        points_calcules.append([int(round(pt_final[0])), int(round(pt_final[1]))])
        
    index_courant += nb_pts_forme

# ==============================================================================
# 4. CRÉATION DU FICHIER TEXTE POUR L'ARDUINO
# On assemble le code C++ dans une grosse variable textuelle
code_texte = "// --- CODE ARDUINO GÉNÉRÉ AUTOMATIQUEMENT ---\n"
code_texte += f"const int NOMBRE_FORMES = {len(toutes_les_formes)};\n\n"
code_texte += "const int indexFormes[NOMBRE_FORMES][2] = {\n"
code_texte += "".join(lignes_index)
code_texte += "};\n\n"

code_texte += f"const int TOTAL_POINTS = {len(points_calcules)};\n"
code_texte += "const int listePoints[TOTAL_POINTS][2] = {\n"
for k, pt in enumerate(points_calcules):
    virgule = "," if k < len(points_calcules) - 1 else ""
    code_texte += f"  {{ {pt[0]}, {pt[1]} }}{virgule}\n"
code_texte += "};\n"

# On enregistre ce texte dans un vrai fichier sur l'ordinateur
with open("code_arduino.txt", "w", encoding="utf-8") as fichier:
    fichier.write(code_texte)

print("[SUCCÈS] Le fichier 'code_arduino.txt' a été créé à côté du script !")

# ==============================================================================
# 5. DESSIN DU NUAGE DE POINTS À L'ÉCRAN
# ==============================================================================
# On convertit notre liste pour que Matplotlib puisse la lire facilement
pts_tableau = np.array(points_calcules)

plt.figure(figsize=(10, 6))

# On trace les points bleus
plt.scatter(pts_tableau[:, 0], pts_tableau[:, 1], color='blue', s=12, label='Points pour l\'Arduino')

# Titres et étiquettes des axes
plt.title(f"Aperçu du Nuage de Points ({len(points_calcules)} points)", fontsize=12, weight='bold')
plt.xlabel("Axe X (Pixels)")
plt.ylabel("Axe Y (Pixels)")

# En informatique, le point (0,0) est en haut à gauche. 
# On inverse l'axe Y pour éviter que le logo s'affiche à l'envers.
plt.gca().invert_yaxis()

# On affiche une grille grise de repère
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.axis('equal')  # Garde les vraies proportions géométriques (pas de déformation)

# On ouvre la fenêtre d'affichage
print("[INFO] Ouverture de la fenêtre graphique...")
plt.show()