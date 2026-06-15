# IMPORTATION DES BIBLIOTHÈQUES (Les outils nécessaires au programme)
import cv2  #traitement d'images
import numpy as np  # mathématique pour coordonnées
import tkinter as tk  #interfaces graphiques
from tkinter import filedialog, ttk  #explorateur de fichiers
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # graphiques
import matplotlib.pyplot as plt 

# CLASSE PRINCIPALE DE L'APPLICATION
class AppMultiContoursArduino:
    def __init__(self, root):
        """exécute automatiquement à l'ouverture du programme """
        self.root = root  # On sauvegarde la fenêtre principale dans une variable interne
        self.root.title("Extracteur multi-contours pour Arduino")  # On définit le titre de la fenêtre supérieure
        self.root.geometry("1200x750")  # On fixe la taille par défaut de la fenêtre (Largeur x Hauteur) en pixels
        
        # --- INITIALISATION DES VARIABLES DU PROGRAMME ---
        self.image_path = None  # Contiendra le chemin d'accès du fichier image sur l'ordinateur
        self.img_gray = None  # Contiendra l'image convertie en niveaux de gris (indispensable pour les contours)
        self.thresh = None  # Contiendra l'image binarisée (uniquement du noir pur et du blanc pur)
        self.tous_les_contours = []  # Liste brute de toutes les formes géométriques détectées par OpenCV
        self.points_par_contour = []  # Liste de listes : contiendra les points finaux échantillonnés par forme
        
        # --- VARIABLES DYNAMIQUES LIÉES AUX CURSEURS ET CASES (TKINTER) ---
        self.n_points_var = tk.IntVar(value=100)  # Variable Tkinter contenant le nombre total de points souhaité (Défaut : 100)
        self.seuil_var = tk.IntVar(value=127)  # Variable contenant la valeur du seuil de binarisation (Défaut : 127)
        self.taille_min_var = tk.IntVar(value=20)  # Variable pour éliminer les formes plus petites que X pixels (Bruit)

        # Appel de la fonction qui va dessiner visuellement les boutons et les zones de texte
        self.creer_interface()

    def creer_interface(self):
        """ Cette fonction construit l'agencement visuel (le design) de notre logiciel """
        
        # ----------------------------------------------------------------------
        # 1. PANNEAU DE CONTRÔLE (À GAUCHE)
        # ----------------------------------------------------------------------
        # Création d'un bloc rectangulaire gris clair à gauche de 250 pixels de large
        panneau_gauche = tk.Frame(self.root, width=250, bg="#f0f0f0", padx=10, pady=10)
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y)  # On l'accroche à gauche et il s'étire sur toute la hauteur (Y)

        # Bouton vert pour charger l'image
        btn_charger = tk.Button(panneau_gauche, text="1. Importer une Photo", command=self.charger_image, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'), pady=5)
        btn_charger.pack(fill=tk.X, pady=10)  # On l'affiche en lui demandant d'occuper toute la largeur du panneau

        # Texte indicatif pour le curseur des points
        tk.Label(panneau_gauche, text="Nombre total de points :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        # Case numérique (Spinbox) pour choisir précisément le nombre de points de 10 à 5000
        spin_points = tk.Spinbox(panneau_gauche, from_=10, to=5000, textvariable=self.n_points_var, command=self.recalculer)
        spin_points.pack(fill=tk.X, pady=5)

        # Texte indicatif pour le curseur de seuil
        tk.Label(panneau_gauche, text="Seuil de détection (0-255) :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        # Glissière horizontale (Scale) pour ajuster la sensibilité de la binarisation en direct
        slider_seuil = tk.Scale(panneau_gauche, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.seuil_var, command=lambda x: self.recalculer())
        slider_seuil.pack(fill=tk.X, pady=5)

        # Texte indicatif pour le curseur de filtrage de taille
        tk.Label(panneau_gauche, text="Taille minimale forme (px) :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        # Glissière pour filtrer les impuretés de l'image (les points isolés ou les pixels bizarres)
        slider_taille = tk.Scale(panneau_gauche, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.taille_min_var, command=lambda x: self.recalculer())
        slider_taille.pack(fill=tk.X, pady=5)

        # ----------------------------------------------------------------------
        # 2. ZONE CENTRALES DE VISUALISATION (LES ONGLETS)
        # ----------------------------------------------------------------------
        # Création du système d'onglets au centre
        self.zone_onglets = ttk.Notebook(self.root)
        self.zone_onglets.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Création de 3 zones de graphiques vides avec Matplotlib (taille 5x5 pouces virtuels)
        self.fig1, self.ax1 = plt.subplots(figsize=(5, 5))
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 5))
        self.fig3, self.ax3 = plt.subplots(figsize=(5, 5))

        # Création des 3 cadres (Frames) Tkinter qui serviront de pages pour nos onglets
        self.tab1 = ttk.Frame(self.zone_onglets)
        self.tab2 = ttk.Frame(self.zone_onglets)
        self.tab3 = ttk.Frame(self.zone_onglets)

        # Attribution des titres textuels visibles sur chaque onglet
        self.zone_onglets.add(self.tab1, text="1. Original & Gris")
        self.zone_onglets.add(self.tab2, text="2. Tous les Contours")
        self.zone_onglets.add(self.tab3, text="3. Points Uniformes")

        # Insertion physique des graphiques Matplotlib à l'intérieur des onglets de Tkinter (le "Canvas")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------------------------
        # 3. PANNEAU LATÉRAL DU CODE SOURCE (À DROITE)
        # ----------------------------------------------------------------------
        # Création d'un panneau gris foncé à droite de 350 pixels de large pour accueillir le texte Arduino
        panneau_droite = tk.Frame(self.root, width=350, bg="#e8e8e8", padx=10, pady=10)
        panneau_droite.pack(side=tk.RIGHT, fill=tk.Y)

        # Étiquette de titre au dessus du bloc de texte
        tk.Label(panneau_droite, text="Code Arduino généré :", bg="#e8e8e8", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Zone de texte multicolonne éditable (Text) utilisant une police à pas fixe ("Consolas") idéale pour le code
        self.texte_arduino = tk.Text(panneau_droite, width=42, font=('Consolas', 9))
        self.texte_arduino.pack(fill=tk.BOTH, expand=True, pady=5)

    def charger_image(self):
        """ Ouvre une boîte de dialogue pour que l'utilisateur sélectionne son image """
        # Cette ligne ouvre l'explorateur Windows/Mac standard filtré sur les fichiers images connus
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:  # Si l'utilisateur n'a pas cliqué sur "Annuler"
            self.image_path = file_path  # On enregistre le chemin de l'image sélectionnée
            # OpenCV charge l'image en mémoire et la convertit immédiatement en nuances de gris (IMREAD_GRAYSCALE)
            self.img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.recalculer()  # On lance automatiquement l'analyse mathématique

    def distribuer_point_uniforme(self, contour, distance_cible):
        """ ALGORITHME : Parcourt les segments d'une forme pour poser un point exact à une distance donnée """
        distance_cumulee = 0.0  # Suivi de la distance parcourue le long de la ligne de contour
        
        # On parcourt chaque segment reliant le point 'i' au point suivant 'i+1' du contour
        for i in range(len(contour)):
            pt1 = contour[i][0]  # Coordonnée de départ du segment [X, Y]
            pt2 = contour[(i + 1) % len(contour)][0]  # Coordonnée d'arrivée (le symbole % permet de boucler du dernier point au premier)
            dist_segment = np.linalg.norm(pt2 - pt1)  # Calcul mathématique de la distance entre pt1 et pt2 (Théorème de Pythagore)
            
            # Si le point recherché se trouve précisément sur ce segment
            if distance_cumulee + dist_segment >= distance_cible:
                if dist_segment == 0: return pt1.tolist()  # Sécurité si le segment fait 0 pixel de long
                # Règle de trois (interpolation) pour trouver la position exacte du point entre pt1 et pt2
                ratio = (distance_cible - distance_cumulee) / dist_segment
                pt_interp = pt1 + ratio * (pt2 - pt1)
                return [int(round(pt_interp[0])), int(round(pt_interp[1]))]  # On renvoie les coordonnées sous forme d'entiers
                
            distance_cumulee += dist_segment  # Sinon, on ajoute la longueur du segment parcouru et on continue
        return contour[0][0].tolist()  # Sécurité par défaut : renvoie le tout premier point du contour

    def recalculer(self):
        """ LE CŒUR DE L'APPLICATION : Traite l'image, extrait les formes et calcule les points """
        if self.img_gray is None:
            return  # Sécurité : On annule l'opération si aucune image n'est actuellement importée

        # ÉTAPE A : LE SEUILLAGE (Binarisation)
        # Transforme les nuances de gris en Noir ou Blanc pur selon le réglage de la glissière. 
        # cv2.THRESH_BINARY_INV inverse le résultat pour que les formes d'intérêt deviennent blanches sur fond noir.
        _, self.thresh = cv2.threshold(self.img_gray, self.seuil_var.get(), 255, cv2.THRESH_BINARY_INV)

        # ÉTAPE B : RECHERCHE DES CONTOURS
        # RETR_EXTERNAL ignore l'intérieur des trous (ex: l'intérieur d'un 'O') pour ne capturer que la silhouette extérieure.
        # CHAIN_APPROX_SIMPLE compresse les lignes droites pour optimiser l'espace mémoire.
        tous_contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # FILTRAGE : On recrée la liste en excluant toutes les formes minuscules (bruit/poussières) via la fonction cv2.contourArea
        self.tous_les_contours = [c for c in tous_contours if cv2.contourArea(c) >= self.taille_min_var.get()]

        self.points_par_contour = []  # On vide la liste des anciens points calculés avant de recalculer
        
        if self.tous_les_contours:  # Si au moins une forme valide a survécu au filtrage
            # On calcule la longueur géométrique de chaque forme via cv2.arcLength(..., True signifie fermé)
            perimetres = [cv2.arcLength(c, True) for c in self.tous_les_contours]
            perimetre_total = sum(perimetres)  # Somme cumulée de toutes les longueurs de toutes les formes réunies
            
            if perimetre_total > 0:
                n_points_total = self.n_points_var.get()  # On récupère l'objectif global de points (ex: 500)
                
                # ÉTAPE C : RÉPARTITION ÉQUITABLE DU NOMBRE DE POINTS
                for c, perim in zip(self.tous_les_contours, perimetres):
                    # Règle de trois : Plus une forme est grande, plus elle reçoit de points.
                    n_points_forme = int(round((perim / perimetre_total) * n_points_total))
                    if n_points_forme < 3: n_points_forme = 3  # Sécurité géométrique : il faut au moins 3 points pour faire une forme fermée
                    
                    points_forme = []  # Liste de points dédiée à cette forme précise
                    # On appelle notre algorithme d'échantillonnage uniforme
                    for i in range(n_points_forme):
                        distance = (i * perim) / n_points_forme  # Découpage régulier de la forme
                        pt = self.distribuer_point_uniforme(c, distance)
                        points_forme.append(pt)  # Ajout du point trouvé dans la liste de la forme
                    
                    self.points_par_contour.append(points_forme)  # On sauvegarde cette forme de points dans la grande liste

            self.generer_code_arduino()  # On écrit textuellement la traduction en langage C++ pour l'Arduino
        else:
            # Si aucune forme n'existe, on efface le texte à droite et on affiche un avertissement
            self.texte_arduino.delete("1.0", tk.END)
            self.texte_arduino.insert(tk.END, "// Aucun contour détecté.\n// Ajustez le seuil ou la taille minimale.")

        self.mettre_a_jour_graphiques()  # On actualise le rendu visuel dans les 3 onglets centraux

    def mettre_a_jour_graphiques(self):
        """ Gère l'actualisation visuelle des fenêtres graphiques de l'interface """
        
        # --- ONGLET 1 : IMAGE ORIGINALE ---
        self.ax1.clear()  # Efface l'ancien dessin
        self.ax1.imshow(self.img_gray, cmap='gray')  # Affiche la matrice d'image brute en mode noir et blanc ('gray')
        self.ax1.set_title("Image originale (Gris)")  # Donne un titre au sous-graphique
        self.ax1.axis('off')  # Masque la graduation des axes X et Y (0, 100, 200 pixels) pour faire propre
        self.canvas1.draw()  # Demande à Tkinter de redessiner physiquement le canevas numéro 1

        # --- ONGLET 2 : LE MASQUE DE SEUILLAGE ET CONTOURS ROUGES ---
        self.ax2.clear()
        self.ax2.imshow(self.thresh, cmap='gray')  # Affiche l'image binarisée (Masque Noir et Blanc)
        for c in self.tous_les_contours:
            c_poly = c.reshape(-1, 2)  # Reformate la structure OpenCV complexe en liste simple de coordonnées [[x,y], [x,y]]
            if len(c_poly) > 0:
                # Dessine une ligne rouge continue reliant les coordonnées du contour, en forçant la fermeture (premier point rattaché)
                self.ax2.plot(np.append(c_poly[:, 0], c_poly[0, 0]), 
                              np.append(c_poly[:, 1], c_poly[0, 1]), color='red', linewidth=2)
        self.ax2.set_title(f"Contours Détectés ({len(self.tous_les_contours)} formes)")
        self.ax2.axis('off')
        self.canvas2.draw()

        # --- ONGLET 3 : RENDER VECTORIEL DES POINTS DE SORTIE ARDUINO ---
        self.ax3.clear()
        self.ax3.set_facecolor('white')  # On choisit un fond blanc immaculé pour bien évaluer le tracé mathématique
        
        total_points_affiches = 0  # Compteur pour informer l'utilisateur du nombre réel final de points générés
        for idx, pts in enumerate(self.points_par_contour):
            if not pts: continue
            pts_array = np.array(pts)  # Transformation en tableau mathématique NumPy
            total_points_affiches += len(pts)
            
            # Dessine des pointillés verts fins reliant les futurs points de l'Arduino pour simuler la trajectoire
            self.ax3.plot(np.append(pts_array[:, 0], pts_array[0, 0]), 
                          np.append(pts_array[:, 1], pts_array[0, 1]), 'g--', alpha=0.4)
            # Superpose des points ronds bleus (scatter) pour localiser précisément chaque coordonnée exportée
            self.ax3.scatter(pts_array[:, 0], pts_array[:, 1], s=15, zorder=5)
            # Ajoute un libellé textuel violet (ex: F0 pour "Forme 0", F1 pour "Forme 1") juste à côté du premier point
            self.ax3.text(pts_array[0, 0], pts_array[0, 1], f" F{idx}", color="purple", fontsize=8, weight='bold')

        self.ax3.set_title(f"Répartition sur l'image ({total_points_affiches} points au total)")
        # ATTENTION CRITIQUE : En informatique, l'origine (0,0) d'une image est en haut à gauche. 
        # Pour éviter que Matplotlib n'inverse l'image à l'envers, on force l'axe Y à aller du maximum (en bas) vers 0 (en haut).
        self.ax3.set_ylim(self.img_gray.shape[0], 0)
        self.ax3.set_xlim(0, self.img_gray.shape[1])
        self.canvas3.draw()

    def generer_code_arduino(self):
        """ Génère le texte structuré en C++ pour l'IDE Arduino et l'injecte à l'écran """
        self.texte_arduino.delete("1.0", tk.END)  # Vide totalement la zone de texte à droite
        if not self.points_par_contour: return  # S'il n'y a rien à exporter, on s'arrête là
            
        total_pts = sum(len(pts) for pts in self.points_par_contour)  # Somme des points réels générés
        
        # --- ÉCRITURE DE L'ENTÊTE ---
        code = f"// Code Multi-Formes Généré\n"
        code += f"const int NOMBRE_FORMES = {len(self.points_par_contour)};\n"
        code += f"const int TOTAL_POINTS  = {total_pts};\n\n"
        
        # --- TEXTE DU STRUCTURE 1 : INDEX DE SÉCURITÉ ---
        code += "// Index de début et taille (nombre de points) de chaque forme\n"
        code += "const int indexFormes[NOMBRE_FORMES][2] = {\n"
        index_courant = 0  # Permet de calculer le décalage (offset) de chaque forme dans le tableau à plat
        for i, pts in enumerate(self.points_par_contour):
            virgule = "," if i < len(self.points_par_contour) - 1 else ""  # Syntaxe C++ : pas de virgule sur le dernier élément
            code += f"  {{{index_courant}, {len(pts)}}}{virgule} // Forme {i} (Début, Taille)\n"
            index_courant += len(pts)  # La forme suivante commencera juste après la fin de la forme actuelle
        code += "};\n\n"
        
        # --- TEXTE DU STRUCTURE 2 : TABLEAU PLAT DES COORDONNÉES ---
        code += "// Liste plate de toutes les coordonnées {X, Y}\n"
        code += "const int listePoints[TOTAL_POINTS][2] = {\n"
        
        compteur = 0  # Compteur absolu pour suivre la totalité des points imprimés
        for idx_f, pts in enumerate(self.points_par_contour):
            code += f"  // --- Début Forme {idx_f} ---\n"
            for i, pt in enumerate(pts):
                virgule = "," if compteur < total_pts - 1 else ""  # Détermine s'il faut séparer par une virgule en C++
                code += f"  {{{pt[0]}, {pt[1]}}}{virgule} // F{idx_f} Pt {i}\n"
                compteur += 1
                
        code += "};"
        self.texte_arduino.insert(tk.END, code)  # Injecte la chaîne de caractères construite dans le widget textuel de droite

# ==============================================================================
# POINT D'ENTRÉE STANDARD DU SCRIPT
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()  # Initialisation de l'instance du moteur graphique Tkinter
    app = AppMultiContoursArduino(root)  # Instanciation de notre application personnalisée
    root.mainloop()  # Lancement de la boucle infinie d'écoute des événements (clics, mouvements de souris)