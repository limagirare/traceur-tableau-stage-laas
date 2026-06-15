import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class AppMultiContoursArduino:
    def __init__(self, root):
        self.root = root
        self.root.title("Extracteur multi-contours pour Arduino")
        self.root.geometry("1200x750")
        
        self.image_path = None
        self.img_gray = None
        self.thresh = None
        self.tous_les_contours = []
        self.points_par_contour = [] # Liste de listes de points
        
        self.n_points_var = tk.IntVar(value=100)
        self.seuil_var = tk.IntVar(value=127)
        self.taille_min_var = tk.IntVar(value=20) # Pour filtrer le bruit/pixels parasites

        self.creer_interface()

    def creer_interface(self):
        # ---- PANNEAU DE CONTRÔLE (GAUCHE) ----
        panneau_gauche = tk.Frame(self.root, width=250, bg="#f0f0f0", padx=10, pady=10)
        panneau_gauche.pack(side=tk.LEFT, fill=tk.Y)

        btn_charger = tk.Button(panneau_gauche, text="1. Importer une Photo", command=self.charger_image, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold'), pady=5)
        btn_charger.pack(fill=tk.X, pady=10)

        # Nombre total de points
        tk.Label(panneau_gauche, text="Nombre total de points :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        spin_points = tk.Spinbox(panneau_gauche, from_=10, to=5000, textvariable=self.n_points_var, command=self.recalculer)
        spin_points.pack(fill=tk.X, pady=5)

        # Réglage du seuil de binarisation
        tk.Label(panneau_gauche, text="Seuil de détection (0-255) :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        slider_seuil = tk.Scale(panneau_gauche, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.seuil_var, command=lambda x: self.recalculer())
        slider_seuil.pack(fill=tk.X, pady=5)

        # Filtrage des petites formes (bruit)
        tk.Label(panneau_gauche, text="Taille minimale forme (px) :", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,0))
        slider_taille = tk.Scale(panneau_gauche, from_=0, to=500, orient=tk.HORIZONTAL, variable=self.taille_min_var, command=lambda x: self.recalculer())
        slider_taille.pack(fill=tk.X, pady=5)

        # ---- ZONE DE VISUALISATION (CENTRE) ----
        self.zone_onglets = ttk.Notebook(self.root)
        self.zone_onglets.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig1, self.ax1 = plt.subplots(figsize=(5, 5))
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 5))
        self.fig3, self.ax3 = plt.subplots(figsize=(5, 5))

        self.tab1 = ttk.Frame(self.zone_onglets)
        self.tab2 = ttk.Frame(self.zone_onglets)
        self.tab3 = ttk.Frame(self.zone_onglets)

        self.zone_onglets.add(self.tab1, text="1. Original & Gris")
        self.zone_onglets.add(self.tab2, text="2. Tous les Contours")
        self.zone_onglets.add(self.tab3, text="3. Points Uniformes")

        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- PANNEAU CODE ARDUINO (DROITE) ----
        panneau_droite = tk.Frame(self.root, width=350, bg="#e8e8e8", padx=10, pady=10)
        panneau_droite.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(panneau_droite, text="Code Arduino généré :", bg="#e8e8e8", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.texte_arduino = tk.Text(panneau_droite, width=42, font=('Consolas', 9))
        self.texte_arduino.pack(fill=tk.BOTH, expand=True, pady=5)

    def charger_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.recalculer()

    def distribuer_point_uniforme(self, contour, distance_cible):
        distance_cumulee = 0.0
        for i in range(len(contour)):
            pt1 = contour[i][0]
            pt2 = contour[(i + 1) % len(contour)][0]
            dist_segment = np.linalg.norm(pt2 - pt1)
            
            if distance_cumulee + dist_segment >= distance_cible:
                if dist_segment == 0: return pt1.tolist()
                ratio = (distance_cible - distance_cumulee) / dist_segment
                pt_interp = pt1 + ratio * (pt2 - pt1)
                return [int(round(pt_interp[0])), int(round(pt_interp[1]))]
            distance_cumulee += dist_segment
        return contour[0][0].tolist()

    def recalculer(self):
        if self.img_gray is None:
            return

        # 1. Seuillage
        _, self.thresh = cv2.threshold(self.img_gray, self.seuil_var.get(), 255, cv2.THRESH_BINARY_INV)

        # 2. Trouver TOUS les contours externes
        tous_contours, _ = cv2.findContours(self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer par taille minimale pour éviter de prendre des pixels isolés ou du bruit
        self.tous_les_contours = [c for c in tous_contours if cv2.contourArea(c) >= self.taille_min_var.get()]

        self.points_par_contour = []
        
        if self.tous_les_contours:
            # Calculer le périmètre total cumulé de toutes les formes réunies
            perimetres = [cv2.arcLength(c, True) for c in self.tous_les_contours]
            perimetre_total = sum(perimetres)
            
            if perimetre_total > 0:
                n_points_total = self.n_points_var.get()
                
                # Échantillonner chaque contour
                for c, perim in zip(self.tous_les_contours, perimetres):
                    # Nombre de points proportionnel à la taille de la forme
                    n_points_forme = int(round((perim / perimetre_total) * n_points_total))
                    if n_points_forme < 3: n_points_forme = 3 # Minimum de points pour fermer une forme
                    
                    points_forme = []
                    for i in range(n_points_forme):
                        distance = (i * perim) / n_points_forme
                        pt = self.distribuer_point_uniforme(c, distance)
                        points_forme.append(pt)
                    
                    self.points_par_contour.append(points_forme)

            self.generer_code_arduino()
        else:
            self.texte_arduino.delete("1.0", tk.END)
            self.texte_arduino.insert(tk.END, "// Aucun contour détecté.\n// Ajustez le seuil ou la taille minimale.")

        self.mettre_a_jour_graphiques()

    def mettre_a_jour_graphiques(self):
        # Onglet 1 : Original
        self.ax1.clear()
        self.ax1.imshow(self.img_gray, cmap='gray')
        self.ax1.set_title("Image originale (Gris)")
        self.ax1.axis('off')
        self.canvas1.draw()

        # Onglet 2 : Masque et Contours (Tous dessinés en rouge)
        self.ax2.clear()
        self.ax2.imshow(self.thresh, cmap='gray')
        for c in self.tous_les_contours:
            c_poly = c.reshape(-1, 2)
            # Fermer le tracé du contour en ajoutant le premier point à la fin
            if len(c_poly) > 0:
                self.ax2.plot(np.append(c_poly[:, 0], c_poly[0, 0]), 
                              np.append(c_poly[:, 1], c_poly[0, 1]), color='red', linewidth=2)
        self.ax2.set_title(f"Contours Détectés ({len(self.tous_les_contours)} formes)")
        self.ax2.axis('off')
        self.canvas2.draw()

        # Onglet 3 : Points Uniformes de toutes les formes
        self.ax3.clear()
        self.ax3.set_facecolor('white')
        
        total_points_affiches = 0
        for idx, pts in enumerate(self.points_par_contour):
            if not pts: continue
            pts_array = np.array(pts)
            total_points_affiches += len(pts)
            
            # Dessiner les lignes de liaison (vertes)
            self.ax3.plot(np.append(pts_array[:, 0], pts_array[0, 0]), 
                          np.append(pts_array[:, 1], pts_array[0, 1]), 'g--', alpha=0.4)
            # Dessiner les points (bleus)
            self.ax3.scatter(pts_array[:, 0], pts_array[:, 1], s=15, zorder=5)
            # Marquer le début de chaque forme autonome
            self.ax3.text(pts_array[0, 0], pts_array[0, 1], f" F{idx}", color="purple", fontsize=8, weight='bold')

        self.ax3.set_title(f"Répartition sur l'image ({total_points_affiches} points au total)")
        self.ax3.set_ylim(self.img_gray.shape[0], 0)
        self.ax3.set_xlim(0, self.img_gray.shape[1])
        self.canvas3.draw()

    def generer_code_arduino(self):
        self.texte_arduino.delete("1.0", tk.END)
        if not self.points_par_contour: return
            
        total_pts = sum(len(pts) for pts in self.points_par_contour)
        
        code = f"// Code Multi-Formes Généré\n"
        code += f"const int NOMBRE_FORMES = {len(self.points_par_contour)};\n"
        code += f"const int TOTAL_POINTS  = {total_pts};\n\n"
        
        # 1. Génération des index de début et de fin pour chaque forme
        code += "// Index de début et taille (nombre de points) de chaque forme\n"
        code += "const int indexFormes[NOMBRE_FORMES][2] = {\n"
        index_courant = 0
        for i, pts in enumerate(self.points_par_contour):
            virgule = "," if i < len(self.points_par_contour) - 1 else ""
            code += f"  {{{index_courant}, {len(pts)}}}{virgule} // Forme {i} (Début, Taille)\n"
            index_courant += len(pts)
        code += "};\n\n"
        
        # 2. Tableau à plat contenant absolument toutes les coordonnées à la suite
        code += "// Liste plate de toutes les coordonnées {X, Y}\n"
        code += "const int listePoints[TOTAL_POINTS][2] = {\n"
        
        compteur = 0
        for idx_f, pts in enumerate(self.points_par_contour):
            code += f"  // --- Début Forme {idx_f} ---\n"
            for i, pt in enumerate(pts):
                virgule = "," if compteur < total_pts - 1 else ""
                code += f"  {{{pt[0]}, {pt[1]}}}{virgule} // F{idx_f} Pt {i}\n"
                compteur += 1
                
        code += "};"
        self.texte_arduino.insert(tk.END, code)

if __name__ == "__main__":
    root = tk.Tk()
    app = AppMultiContoursArduino(root)
    root.mainloop()