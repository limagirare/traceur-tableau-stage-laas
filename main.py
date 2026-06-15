import cv2
import numpy as np

def extraire_points_contour(image_path, n_points=50):
    # 1. Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erreur : Impossible de charger l'image.")
        return None

    # 2. Binariser l'image (Seuillage pour avoir du noir et blanc pur)
    # Ajustez le seuil (127) selon votre image (fond clair ou foncé)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 3. Détection des contours les plus simples possibles
    # RETR_EXTERNAL : uniquement le contour le plus externe
    # CHAIN_APPROX_SIMPLE : compresse les segments horizontaux, verticaux et diagonaux
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Aucun contour trouvé.")
        return None

    # Prendre le plus grand contour trouvé
    contour_principal = max(contours, key=cv2.contourArea)

    # 4. Échantillonner uniformément les points le long du contour
    # On utilise la longueur du contour pour répartir les points
    points_resamples = []
    perimetre = cv2.arcLength(contour_principal, True)
    
    # On parcourt le contour à intervalles réguliers
    for i in range(n_points):
        distance = (i * perimetre) / n_points
        # Trouver le point correspondant à cette distance cumulative
        # OpenCV stocke les contours sous forme de segments, on approxime ici
        pt = distribuer_point_uniforme(contour_principal, distance)
        points_resamples.append(pt)

    return points_resamples

def distribuer_point_uniforme(contour, distance_cible):
    """Trouve un point à une distance spécifique le long du contour"""
    distance_cumulee = 0.0
    for i in range(len(contour)):
        pt1 = contour[i][0]
        pt2 = contour[(i + 1) % len(contour)][0]
        dist_segment = np.linalg.norm(pt2 - pt1)
        
        if distance_cumulee + dist_segment >= distance_cible:
            # Interpolation linéaire entre pt1 et pt2
            if dist_segment == 0:
                return pt1.tolist()
            ratio = (distance_cible - distance_cumulee) / dist_segment
            pt_interp = pt1 + ratio * (pt2 - pt1)
            return [int(round(pt_interp[0])), int(round(pt_interp[1]))]
        
        distance_cumulee += dist_segment
    return contour[0][0].tolist()

# --- CONFIGURATION ET EXÉCUTION ---
# Remplacez par le chemin de votre image et le nombre de points souhaités
IMAGE_PATH = "votre_image.png" 
NOMBRE_DE_POINTS = 30 

points = extraire_points_contour(IMAGE_PATH, n_points=NOMBRE_DE_POINTS)

if points:
    print(f"// --- COPIER CE CODE DANS VOTRE SCRIPT ARDUINO ---")
    print(f"const int NOMBRE_POINTS = {len(points)};")
    print("const int pointsContour[NOMBRE_POINTS][2] = {")
    
    for i, pt in enumerate(points):
        virgule = "," if i < len(points) - 1 else ""
        print(f"  {{{pt[0]}, {pt[1]}}}{virgule} \t// Point {i}")
        
    print("};")

# --- AJOUTEZ CETTE LIGNE ICI ---
input("\nAppuyez sur ENTRÉE pour fermer le programme...")