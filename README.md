# 🩺 Diabetes Prediction — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Status](https://img.shields.io/badge/Status-Complete-green)

> **Projet Machine Learning End-to-End** — Prédiction du diabète à partir de données médicales  
> Module : Machine Learning

---

## Table des Matières

- [Contexte et Objectifs](#-contexte-et-objectifs)
- [Dataset](#-dataset)
- [Structure du Projet](#-structure-du-projet)
- [Pipeline ML](#-pipeline-ml)
- [Résultats](#-résultats)
- [Application Interactive](#-application-interactive)
- [Installation](#-installation)
- [Auteur](#-auteur)

---

##  Contexte et Objectifs

Le diabète est une maladie chronique qui touche plus de **537 millions** de personnes dans le monde. Un diagnostic précoce peut réduire considérablement les complications médicales.

**Objectif :** Développer un modèle de Machine Learning capable de **prédire automatiquement** si un patient est diabétique ou non, à partir de 8 mesures médicales simples.

**Type de problème :** Classification binaire (0 = Non diabétique, 1 = Diabétique)

---

## Dataset

**Source :** [Pima Indians Diabetes Database — Kaggle/UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

> **Justification du choix :** Ce dataset provient de l'UCI Machine Learning Repository, une plateforme reconnue internationalement. Il est idéal pour un problème de classification médicale binaire, avec 768 entrées et 8 features médicales pertinentes, dépassant le seuil minimum requis de 500 lignes.

| Caractéristique | Valeur |
|----------------|--------|
| Nombre de lignes | 768 |
| Nombre de features | 8 (+ 3 créées) |
| Variable cible | `Outcome` (0/1) |
| % Diabétiques | 34.9% |
| % Non diabétiques | 65.1% |

### Variables du Dataset

| Feature | Description | Type |
|---------|-------------|------|
| `Pregnancies` | Nombre de grossesses | int |
| `Glucose` | Taux de glucose sanguin (mg/dL) | int |
| `BloodPressure` | Pression artérielle (mmHg) | int |
| `SkinThickness` | Épaisseur du pli cutané (mm) | int |
| `Insulin` | Taux d'insuline (µU/mL) | int |
| `BMI` | Indice de masse corporelle (kg/m²) | float |
| `DiabetesPedigreeFunction` | Historique familial de diabète | float |
| `Age` | Âge du patient (ans) | int |
| `Outcome` | **Cible** : 1=Diabétique, 0=Non diabétique | int |

---

## Structure du Projet

```
diabetes-ml-project/
│
├──  data/
│   └── diabetes.csv                  # Dataset original
│
├──  notebooks/
│   ├── draft_exploration.ipynb       # Notebook brouillon (expérimentations)
│   └── diabetes_analysis.ipynb       # Notebook final soigné
│
├──  models/
│   ├── svm_model.pkl                 # Meilleur modèle sauvegardé (SVM)
│   └── scaler.pkl                    # StandardScaler sauvegardé
│
├──  app/
│   └── index.html                    # Application web interactive
│
├── requirements.txt                  # Dépendances Python
└── README.md                         # Documentation du projet
```

---

##  Pipeline ML

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE ML                          │
├─────────────────────────────────────────────────────────┤
│  1. CHARGEMENT        → diabetes.csv (768 lignes)       │
│  2. EDA               → Visualisations, corrélations    │
│  3. NETTOYAGE         → Zéros impossibles → Médiane     │
│  4. FEATURE ENG.      → Obese, HighGlucose, AgeGroup    │
│  5. SPLIT             → 80% Train / 20% Test            │
│  6. NORMALISATION     → StandardScaler                  │
│  7. MODÈLES           → LR, RF, SVM, XGBoost            │
│  8. ÉVALUATION        → Accuracy, F1, Confusion Matrix  │
│  9. SÉLECTION         → SVM (75.97%)                    │
│ 10. SAUVEGARDE        → svm_model.pkl                   │
└─────────────────────────────────────────────────────────┘
```

### Nettoyage des Données

Certaines colonnes contenaient des **zéros biologiquement impossibles** (ex: Glucose=0, BMI=0). Ces valeurs ont été remplacées par la **médiane** de chaque colonne.

| Colonne | Zéros détectés | % |
|---------|---------------|---|
| Insulin | 374 | 48.7% |
| SkinThickness | 227 | 29.6% |
| BloodPressure | 35 | 4.6% |
| BMI | 11 | 1.4% |
| Glucose | 5 | 0.7% |

### Feature Engineering

3 nouvelles features créées à partir des features existantes :

| Nouvelle Feature | Définition | Justification |
|-----------------|------------|---------------|
| `Obese` | BMI > 30 → 1, sinon 0 | L'obésité est un facteur de risque majeur |
| `HighGlucose` | Glucose > 125 → 1, sinon 0 | Seuil clinique du prédiabète |
| `AgeGroup` | 0-30→0, 30-45→1, >45→2 | Le risque augmente avec l'âge |

---

##  Résultats

### Comparaison des Modèles

| Rang | Modèle | Accuracy | Precision | Recall | F1-Score |
|------|--------|----------|-----------|--------|----------|
| 1 | **SVM** | **75.97%** | 0.69 | 0.57 | 0.63 |
| 2 | XGBoost | 73.38% | 0.63 | 0.59 | 0.61 |
| 3 | Logistic Regression | 72.73% | 0.63 | 0.54 | 0.58 |
| 4️ | Random Forest | 72.73% | 0.62 | 0.59 | 0.60 |

###  Meilleur Modèle : SVM (Support Vector Machine)

**Pourquoi SVM ?**
-  Meilleure accuracy globale : **75.97%**
-  Meilleure précision sur les non-diabétiques (0.79)
-  Robuste aux données médicales avec classes déséquilibrées
-  Fonctionne bien avec des features normalisées

### Feature Importance (XGBoost)

```
Glucose          ████████████████████████████  0.28
BMI              ████████████████              0.16
Age              ███████████████               0.15
Pregnancies      ████████████                  0.12
Insulin          ███████████                   0.11
DiabetesPedigree ██████████                    0.10
BloodPressure    █████████                     0.09
SkinThickness    ████████                      0.08
```

> **Conclusion :** Le **Glucose** est de loin la feature la plus prédictive, suivi du **BMI** et de l'**âge** — ce qui est cohérent avec la littérature médicale.

---

## 🌐 Application Interactive

Une application web interactive a été développée permettant de :
- Entrer les données médicales d'un patient via des sliders
- Obtenir une prédiction en temps réel (Diabétique / Non diabétique)
- Visualiser le niveau de risque en pourcentage
- Comparer les performances des 4 modèles

**Lancer l'application :**
1. Ouvrir `app/index.html` dans VS Code
2. Cliquer sur **"Go Live"** (extension Live Server)
3. L'app s'ouvre sur `http://127.0.0.1:5500/app/`

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/[votre-username]/diabetes-ml-project.git
cd diabetes-ml-project

# 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le notebook
code notebooks/diabetes_analysis.ipynb
```

### Dépendances principales

```
pandas==3.0.3
numpy==2.4.6
matplotlib==3.11.0
seaborn==0.13.2
scikit-learn==1.9.0
xgboost==3.2.0
```

---

## 👤 Auteur

**mohamed A mmeri** — Étudiant(e) en 4ème année  
📧 Contact : mohamed.amari@polytechnicien.tn  
🎓 Module : Machine Learning | 2025-2026

---

*Projet réalisé dans le cadre du module Machine Learning — Examen Pratique End-to-End*
