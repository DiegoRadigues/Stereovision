# **Stéréovision Lab 3 et 4**

---

## **Introduction**
La stéréovision est une méthode de vision artificielle permettant de reconstruire la géométrie 3D d'une scène ou d'un objet à partir de plusieurs images prises sous différents angles. Dans ce projet, l'objectif est d'utiliser des images issues de deux caméras pour reconstruire en 3D un objet scanné à l'aide d'un plan laser. 

Les consignes complètes et les documents sources sont disponibles sur le site :  
[https://quentin.lurkin.xyz/courses/imageprocessing/lab34/](https://quentin.lurkin.xyz/courses/imageprocessing/lab34/)

Ce processus repose sur plusieurs étapes :
1. Calibration des caméras pour connaître leurs paramètres intrinsèques et extrinsèques.
2. Utilisation de la géométrie épipolaire pour trouver des correspondances entre les points des deux images.
3. Application de la triangulation pour calculer les coordonnées 3D des points correspondants.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c112e147-8348-41a5-b796-64e1d15bb60e">
  <br>
  <i>Figure 1 Objet à scanner.</i>
</p>

---


## **Objectifs**
- **Calibrer les caméras** : Utiliser des images d'échiquier pour déterminer les matrices intrinsèques et extrinsèques.
- **Reconstituer un objet scanné en 3D** : Identifier les points correspondant à un laser projeté sur l'objet dans les deux images.
- **Visualiser le résultat** : Afficher les points reconstruits dans un espace 3D.

---

## **Fonctionnement**


### 1. Calibration d’une caméra (mono-caméra)

#### 1.1 Objectif de la calibration
Le but de la calibration est de déterminer la matrice intrinsèque (aussi appelée “camera matrix” ou `mtx` dans le code), et les coefficients de distorsion optique.

- **Matrice intrinsèque** : encode les paramètres internes de la caméra (focale `fx`, `fy`, et centre optique `cx`, `cy`).
- **Distorsion** : corrige la déformation de l’image (effet “fisheye” léger) due aux lentilles.

#### 1.2 Méthode employée : utilisation d’un damier (chessboard)
Dans le script, on utilise `cv2.findChessboardCorners(...)` pour trouver les coins d’un motif en damier (7x7 dans l’exemple). Pourquoi un damier ?

- Parce que c’est un motif simple et régulier qui permet de retrouver facilement des points de référence bien localisés et identifiés dans l’espace image.

Ensuite, on utilise :
- `objp = np.zeros((7*7, 3), np.float32)` : on crée des coordonnées 3D `(x, y, z)` correspondant aux points du damier dans le monde réel. Ici, `z=0`, `x` et `y` vont de `0 à 6`.
- On lit nos images `glob.glob(pathToCalibrate)`.
- Pour chaque image :
  - On trouve les coins du damier `cv2.findChessboardCorners(...)`.
  - Si trouvés, on affine la détection avec `cv2.cornerSubPix(...)` pour plus de précision.
  - On stocke :
    - `objpoints` (points 3D correspondants au damier dans le monde réel)
    - `imgpoints` (points 2D détectés dans l’image)

#### 1.3 Résultat de la calibration
La fonction `cv2.calibrateCamera(...)` retourne :
- `ret` : indicateur de réussite,
- `mtx` : la matrice intrinsèque,
- `dist` : les coefficients de distorsion,
- `rvecs`, `tvecs` : les rotations et translations estimées pour chaque image servant à la calibration.

Dans le code, c’est la fonction `cameraCalibrate(...)` qui encapsule `calibrateCamera`.

---

### 2. Calibration stéréo

Après avoir calibré chaque caméra séparément (on obtient donc `mtxL`, `distL` pour la gauche et `mtxR`, `distR` pour la droite), il faut calibrer le système stéréo pour déterminer :
- La matrice de rotation `R` entre les caméras (comment la caméra de droite est orientée par rapport à la gauche).
- Le vecteur de translation `T` entre les deux caméras (combien la caméra de droite est décalée en X, Y, Z par rapport à la gauche).
- L’Essential Matrix (`E`) et la Fundamental Matrix (`F`).

Ici, la ligne suivante effectue cette calibration stéréo :
```python
_, _, _, _, _, R, T, E, FundamentalMatrix = cv2.stereoCalibrate(
    objPoints, imgPointsLeft, imgPointsRight, 
    mtxL, distL, mtxR, distR, 
    gray.shape[::-1],
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)
```

#### 2.1 Fundamental Matrix (F) vs Essential Matrix (E)
- **Essential Matrix (E)** : Lie les points 3D d’une scène et leurs projections entre deux caméras si on connaît les paramètres intrinsèques.
- **Fundamental Matrix (F)** : Relation entre deux vues même si on ne connaît pas les matrices intrinsèques (caméras non calibrées).

Dans notre cas, on a calibré les caméras, donc on peut avoir accès à `E` et `F`.

---

### 3. Équations épipolaires et épipolarité

#### 3.1 Épipoles et épilignes
Pour une paire de points correspondants `(x_l, x_r)` (un point observé à gauche et à droite) :
- **Lignes épipolaires** : lignes dans l’image droite (ou gauche) sur lesquelles doit se trouver la correspondance du point de l’autre vue.
- La `FundamentalMatrix` (`F`) permet de calculer ces lignes épipolaires via :
```python
epilines_right = cv2.computeCorrespondEpilines(points_left, 2, FundamentalMatrix)
```
- `2` indique qu’on calcule les épilignes pour la vue de droite en se basant sur des points dans la vue gauche.

#### 3.2 Intersection sur la ligne rouge
Dans le code, on a des images où une ligne rouge apparaît (scan d’un objet en projetant un laser rouge). On récupère les points de cette ligne rouge :
```python
mask = cv2.inRange(image, (0, 0, 40), (0, 0, 255))  # on isole le rouge
### ...
redPoints.append([first_non_zero_pixel, i])  # on prend le pixel rouge
```

Ensuite, avec `plotEpipolar(...)`, on dessine les épilignes à droite et on cherche l’endroit où la ligne épipolaire intersecte la ligne rouge. Ce point d’intersection correspond à la corrélation spatiale du même point laser vu par la caméra droite/gauche.

---

### 4. Triangulation des points (3D)

#### 4.1 Principe de la triangulation
Une fois qu’on sait où est un point dans l’image gauche et dans l’image droite, on peut calculer sa position 3D.
- On dispose de deux rayons (un par caméra), chacun passant par le centre optique de sa caméra et le pixel détecté.
- L’intersection ou la combinaison de ces rayons nous donne la coordonnée 3D du point dans l’espace.
- La fonction `cv2.triangulatePoints(P_l, P_r, left_point, right_point)` opère cette triangulation mathématique.

#### 4.2 Sortie : coordonnées homogènes
`cv2.triangulatePoints` renvoie des coordonnées en `[X, Y, Z, W]`. Il faut donc diviser par `W` pour avoir les coordonnées 3D `(X/W, Y/W, Z/W)`.

Dans le code :
```python
points_3d_homogeneous = cv2.triangulatePoints(P_l, P_r, left_point, right_point)
points_3d = points_3d_homogeneous[:3] / points_3d_homogeneous[3]
```

---

### 5. Visualisation 3D

#### 5.1 Matplotlib
Dans `plotTriangulatedPoints`, on utilise un subplot 3D :
```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
```
Pour afficher des points 3D (ex : coins du damier, points d’intérêt, etc.).

#### 5.2 Plotly
La fonction `plotTriangulatedPointsInteractive` utilise Plotly pour faire une visualisation interactive dans le navigateur :
```python
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=4,
        color=z, 
        colorscale='Viridis',
        opacity=0.8
    )
))
```

### 6. Logique globale du script

- **findPoints(...)** :
  - Extrait les coins du damier dans différentes images pour calibrer chaque caméra.
  - Retourne `objpoints`, `imgpoints`, et la dernière image grise pour la dimension.

- **cameraCalibrate(...)** :
  - Calcule la calibration mono-caméra (matrices intrinsèques, distorsions).
  - Construit la matrice de projection `P = mtx * [R | t]`.
  - Calcule le centre de la caméra en coordonnées monde.

- **cv2.stereoCalibrate(...)** :
  - Avec les points 2D de la gauche et de la droite (et la connaissance a priori du motif 3D `objpoints`), on obtient `R`, `T`, `E`, `F`.

- **triangulateChessboardCorners(...)** :
  - Pour chaque coin correspondant (gauche/droite), calcule le point 3D via `cv2.triangulatePoints()`.

- **plotTriangulatedPoints(...)** :
  - Affiche quelques points 3D du damier par Matplotlib.

- **getRedImages(...)** :
  - Récupère toutes les images `left` et `right` d’un scan laser (ligne rouge).
  - Extrait la ligne rouge (masque binaire).

- **getRedLine(...)** :
  - Pour chaque scan (en 2D), récupère la coordonnée `(x, y)` des pixels rouges.

- **plotEpipolar(...)** :
  - Calcule et affiche les épilignes, repère les intersections entre la ligne épipolaire et le pixel rouge.

- **triangulateMask(...)** :
  - Récupère ces points gauche/droite (intersections), puis appelle la triangulation.
  - Retourne un nuage de points 3D.

- **Fusion de tous les nuages** :
  - On concatène (`np.concatenate`) les points 3D de chaque scan.
  - On les visualise en 3D (Matplotlib + Plotly).


## **Résultats**
### **Calibration**
- Les matrices obtenues permettent de corriger la distorsion des images et de modéliser précisément les caméras.
- Exemple de matrice intrinsèque (\( \mathbf{K} \)) :
\[
\mathbf{K} = \begin{bmatrix}
fx & 0 & cx \\
0 & fy & cy \\
0 & 0 & 1
\end{bmatrix}
\]

### **Reconstruction 3D**
Les points reconstruits forment une représentation cohérente de l’objet scanné. Les épilignes alignent correctement les points correspondants, et la triangulation produit une structure réaliste.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9b0e29ac-c32b-44bf-8726-5f0a00b6bdc1" alt="Capture d'écran 2025-01-13 210757">
   <br>
  <i>Figure 1 : Image avec épilignes et correspondances triangulées.</i>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/74843ca6-d203-475e-ac75-83f127f680aa" alt="Capture d'écran 2025-01-13 210842">
  <br>
  <i>Figure 2 : Reconstruction 3D des points scannés.</i>
</p>

---

## **Utilisation**

### **Pré-requis**
- Python 3.x
- Bibliothèques nécessaires :
  ```bash
  pip install opencv-python matplotlib numpy plotly

### Arborescence du projet

```
Stereovision/
│
├── .ipynb_checkpoints/       # Fichiers temporaires pour Jupyter Notebook
├── chessboards/              # Contient les images des damiers pour la calibration
├── env/                      # Environnement virtuel Python
├── scanLeft/                 # Images capturées par la caméra gauche
├── scanRight/                # Images capturées par la caméra droite
│
├── epipolar.png              # Exemple d'image avec lignes épipolaires
├── main.png                  # Image principale pour la calibration
├── README.md                 # Documentation du projet
├── stereovision.ipynb        # Script principal sous forme de Jupyter Notebook
├── triangulated_points.txt   # Résultats des points triangulés
```

### Instructions pour lancer le programme

1. **Accéder au dossier du projet**  
   Ouvrez une fenêtre de terminal et placez-vous dans le répertoire du projet, par exemple :
   ```bash
   cd "C:\Users\diego\Documents\Stereovision"
   ```

2. **Activer l'environnement virtuel Python**  
   Activez l'environnement virtuel pour installer et exécuter les dépendances nécessaires :
   ```bash
   .\env\Scripts\activate
   ```

3. **Lancer Jupyter Notebook**  
   Démarrez Jupyter Notebook pour ouvrir et exécuter le fichier `stereovision.ipynb` :
   ```bash
   jupyter notebook
   ```



