# Scikit-Learn-Breast-Cancer-Wisconsin-Diagnostic-Data-Set

### Google Colab: https://colab.research.google.com/gist/keepitfoxy/06efb17f94816291793050e109fb9f2f/mlp.ipynb?authuser=1#scrollTo=nqDkMVFy5bdH

# Sprawozdanie z Badania Klasyfikatora Perceptronu Wielowarstwowego (MLP) // ENG BELOW

### 1. Cel i Metodologia

Celem ćwiczenia było zbadanie wpływu kluczowych hiperparametrów (architektury, funkcji aktywacji, solwera) modelu Perceptronu Wielowarstwowego (MLP) na zadanie klasyfikacji binarnej.

* **Zbiór Danych:** Breast Cancer Wisconsin (Diagnostic) Data Set (569 instancji, 30 cech).
* **Model:** `sklearn.neural_network.MLPClassifier`.
* **Metoda Oceny:** Walidacja krzyżowa (CV=5) w celu zapewnienia statystycznej wiarygodności wyników.
* **Metryka Wydajności:** **Balanced Accuracy ($\bar{A}_{bal}$)**, wybrana ze względu na potencjalną niezbalansowaną dystrybucję klas w zbiorach medycznych.

### 2. Przygotowanie Danych (Pre-processing)

1.  Usunięto kolumny `ID` oraz zbędną pustą kolumnę.
2.  Zmienną docelową `diagnosis` zakodowano numerycznie: Złośliwy (M) $\rightarrow$ **1**, Łagodny (B) $\rightarrow$ **0**.
3.  Wszystkie cechy wejściowe przeskalowano do zakresu $[0, 1]$ za pomocą **`MinMaxScaler`**. Skalowanie jest fundamentalne dla stabilności numerycznej i efektywnej zbieżności optymalizatorów gradientowych.

### 3. Wyniki Badań Eksperymentalnych

Badania przeprowadzono sekwencyjnie. Optymalny parametr z poprzedniego kroku był używany jako stała w kolejnym badaniu.

#### 3.1. Wpływ Architektury (`hidden_layer_sizes`)

Badano zarówno szerokość, jak i głębokość sieci (1-warstwowe vs. 2-warstwowe).

| Architektura (HLS) | $\bar{A}_{bal}$ | Wnioski (Krytyka) |
| :--- | :--- | :--- |
| **(30, 30)** | **0.9756** | Najwyższy wynik. Sugeruje, że dwie warstwy o średniej szerokości są optymalne dla uchwycenia nieliniowych relacji. |
| (100,) | 0.9755 | Wynik niemal identyczny, co wskazuje na redundancję. Prosta sieć (100,) jest efektywna, ale (30, 30) minimalnie lepsza. |
| (60, 60) | 0.9647 | Spadek wydajności w porównaniu do prostszych opcji, co może wskazywać na nadmierną złożoność modelu względem rozmiaru zbioru danych. |
| (20,), (50, 50, 50) | $\approx$ 0.9647 | Modele zbyt płytkie (20,) i zbyt głębokie (50, 50, 50) wypadły gorzej. |

**Optymalna Architektura dla dalszych badań:** $(30, 30)$.

#### 3.2. Wpływ Funkcji Aktywacji (`activation`)

Badania przeprowadzono dla optymalnej architektury $HLS=(30, 30)$.

| Funkcja Aktywacji | $\bar{A}_{bal}$ | Wnioski (Krytyka) |
| :--- | :--- | :--- |
| **`relu`** | **0.9756** | Potwierdza swoją stabilność i efektywność w zapobieganiu zanikającym gradientom. Osiągnęła najwyższą dokładność. |
| `logistic` | 0.9719 | Niska różnica względem ReLU. Jest to sygnał, że zbieżność nie była drastycznie utrudniona. |
| `identity` | 0.9704 | Wynik niższy. Sieć neuronowa bez nieliniowej aktywacji sprowadza się do regresji liniowej, co jest niewystarczające. |

**Optymalna Funkcja Aktywacji dla dalszych badań:** `relu`.

#### 3.3. Wpływ Solvera (Optymalizatora)

Badania przeprowadzono dla $HLS=(30, 30)$ i $Activation='relu'$.

| Solver | $\bar{A}_{bal}$ | Wnioski (Krytyka) |
| :--- | :--- | :--- |
| **`adam`** | **0.9756** | Adaptacyjny algorytm gradientowy, najlepszy wybór. Wykazał największą zdolność do szybkiego znalezienia optimum globalnego. |
| `lbfgs` | 0.9517 | Znacznie gorszy wynik. Choć często efektywny dla małych zbiorów, w tym przypadku nie osiągnął wydajności `adam`. |
| `sgd` | 0.9413 | Najsłabszy wynik. **Krytyka:** Wersja `sgd` bez dostosowanego współczynnika uczenia się i momentum jest nieskuteczna. Wymaga dogłębnego tuningu. |

**Optymalny Solver:** `adam`.

### 4. Podsumowanie i Wnioski Końcowe

Ostatecznie, najlepszy model klasyfikatora MLP osiągnął średnią **Balanced Accuracy na poziomie 0.9756** przy następującej konfiguracji:

$$
\text{MLPClassifier}(\text{hidden\_layer\_sizes}=(30, 30), \text{activation}='relu', \text{solver}='adam')
$$

**Kluczowe Wnioski Metodologiczne:**

1.  **Łatwość Klasyfikacji:** Stosunkowo wysokie wyniki osiągnięto dla większości testowanych architektur, co sugeruje, że cechy zbioru Breast Cancer Wisconsin są **silnie dyskryminacyjne**.
2.  **Rola Optymalizatora:** Badanie solverów wykazało **największy kontrast** w wydajności. Potwierdza to, że efektywność algorytmu optymalizacyjnego (`adam`) jest czynnikiem krytycznym, przewyższającym marginalne różnice wynikające ze zmian architektury.
3.  **Następne Kroki:** W celu potencjalnej poprawy wyniku, należałoby zbadać wpływ **współczynnika regularyzacji** ($\alpha$) oraz przeprowadzić dokładniejszy Grid Search wokół najlepszych parametrów.

---
---

# English Version (Technical Report)

## Multilayer Perceptron (MLP) Classifier Hyperparameter Study Report

### 1. Objective and Methodology

The objective of this exercise was to investigate the influence of key hyperparameters (architecture, activation function, solver) of the Multilayer Perceptron (MLP) model on a binary classification task.

* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Data Set (569 instances, 30 features).
* **Model:** `sklearn.neural_network.MLPClassifier`.
* **Evaluation Method:** Cross-validation (CV=5) to ensure statistical reliability of results.
* **Performance Metric:** **Balanced Accuracy ($\bar{A}_{bal}$)**, chosen due to the potential for class imbalance in medical datasets.

### 2. Data Preparation (Pre-processing)

1.  The `ID` column and an extraneous empty column were dropped.
2.  The target variable `diagnosis` was numerically encoded: Malignant (M) $\rightarrow$ **1**, Benign (B) $\rightarrow$ **0**.
3.  All input features were scaled to the range $[0, 1]$ using **`MinMaxScaler`**. Scaling is fundamental for numerical stability and efficient convergence of gradient-based optimizers.

### 3. Experimental Results

The studies were conducted sequentially, with the optimal parameter from each step being used as a constant in the subsequent analysis.

#### 3.1. Influence of Architecture (`hidden_layer_sizes`)

Both network width and depth (1-layer vs. 2-layer) were examined.

| Architecture (HLS) | $\bar{A}_{bal}$ | Findings (Critique) |
| :--- | :--- | :--- |
| **(30, 30)** | **0.9756** | Highest score. Suggests that two moderately wide layers are optimal for capturing non-linear relationships. |
| (100,) | 0.9755 | Nearly identical score, indicating redundancy. A simple (100,) network is effective, but (30, 30) is marginally superior. |
| (60, 60) | 0.9647 | Performance drop compared to simpler options, possibly indicating excessive model complexity relative to the dataset size (risk of overfitting, mitigated by CV). |
| (20,), (50, 50, 50) | $\approx$ 0.9647 | Models too shallow (20,) and too deep (50, 50, 50) performed worse. |

**Optimal Architecture for further study:** $(30, 30)$.

#### 3.2. Influence of Activation Function (`activation`)

The study was conducted using the optimal architecture $HLS=(30, 30)$.

| Activation Function | $\bar{A}_{bal}$ | Findings (Critique) |
| :--- | :--- | :--- |
| **`relu`** | **0.9756** | Confirms its stability and efficiency in preventing vanishing gradients. Achieved the highest accuracy. |
| `logistic` | 0.9719 | Low difference relative to ReLU. This suggests that convergence was not severely hampered in this shallow network. |
| `identity` | 0.9704 | Lower score. A neural network without a non-linear activation function defaults to linear regression, which is insufficient. |

**Optimal Activation Function for further study:** `relu`.

#### 3.3. Influence of Solver (Optimizer)

The study was conducted using $HLS=(30, 30)$ and $Activation='relu'$.

| Solver | $\bar{A}_{bal}$ | Findings (Critique) |
| :--- | :--- | :--- |
| **`adam`** | **0.9756** | Adaptive gradient algorithm, the superior choice. Demonstrated the greatest ability to quickly find the global optimum. |
| `lbfgs` | 0.9517 | Significantly poorer result. Although often effective for small datasets, it did not match `adam`'s performance in this case. |
| `sgd` | 0.9413 | The weakest result. **Critique:** The default `sgd` implementation, lacking tuned learning rate and momentum, is uncompetitive and requires in-depth manual tuning. |

**Optimal Solver:** `adam`.

### 4. Summary and Final Conclusions

The final best MLP classifier model achieved an average **Balanced Accuracy of 0.9756** with the following configuration:

$$
\text{MLPClassifier}(\text{hidden\_layer\_sizes}=(30, 30), \text{activation}='relu', \text{solver}='adam')
$$

**Key Methodological Conclusions:**

1.  **Ease of Classification:** The consistently high scores across most architectures suggest that the features of the Breast Cancer Wisconsin dataset are **highly
