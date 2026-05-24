# Analiza ryzyka akcji XTB S.A.

Projekt z przedmiotu **Zarządzanie Ryzykiem w Przedsiębiorstwie** — kompleksowa analiza ryzyka spółki **XTB S.A.** (`XTB.WA`) notowanej na GPW w okresie **2018–2025**.

Repozytorium obejmuje cztery etapy pracy projektowej (notebooki `prez3`–`prez5`), hybrydowe modele **LSTM + VaR/FHS**, optymalizację portfela akcji GPW oraz podsumowanie wyników w formie notebooka i raportu PDF.

---

## Spis treści

- [Kontekst i cel](#kontekst-i-cel)
- [Główne wnioski](#główne-wnioski)
- [Struktura repozytorium](#struktura-repozytorium)
- [Metody i zakres analizy](#metody-i-zakres-analizy)
- [Wymagania](#wymagania)
- [Instalacja i uruchomienie](#instalacja-i-uruchomienie)
- [Dane](#dane)
- [Raporty i wykresy](#raporty-i-wykresy)
- [Autor](#autor)

---

## Kontekst i cel

XTB S.A. to globalny broker instrumentów finansowych, którego akcje w badanym okresie przeszły transformację od mniejszego emitenta do składnika indeksu **WIG20**. Wzrost zmienności (roczne σ ≈ 47% vs. ≈ 22% dla WIG20), ujemna skośność (−0,47) i wysoka kurtoza (excess ≈ 25) wymuszają stosowanie metod dynamicznych uwzględniających **klastrowanie zmienności** i **grube ogony** rozkładu zwrotów.

Celem projektu jest:

1. Charakterystyka ryzyka XTB.WA i powiązanych zmiennych (FX, CFD).
2. Porównanie metod **VaR**, **ES** i **EVaR** z backtestingiem regulacyjnym.
3. Modelowanie ekstremów (**EVT**: GEV, GPD).
4. Optymalizacja portfela akcji GPW (**Markowitz**, model jednoskładnikowy).
5. Hybrydowy model **LSTM + Filtered Historical Simulation (FHS)** prognozujący dynamiczny VaR.

---

## Główne wnioski

| Obszar | Wynik |
|--------|-------|
| Rozkład zwrotów | Żaden rozkład parametryczny nie przechodzi testu K-S; najlepsze dopasowanie: Johnson SU, t-Student |
| VaR statyczny | Normalny **niedoszacowuje** VaR 99% o ~20% względem metod t i HS |
| Backtesting (rolling, W=500) | **FHS GARCH** i **EWMA + Hill** przechodzą testy Kupieca i Christoffersena; Param Normal generuje nadmierne naruszenia (strefa czerwona Basel) |
| EVT | GPD (POT) lepiej estymuje skrajne kwantyle (99,9%) niż Block Maxima (GEV) |
| LSTM-FHS | Najlepsza kalibracja w testach Kupieca/Christoffersena (p ≈ 1,0 na poziomach 95% i 99%) |
| Portfel GPW | MVP (Sharpe ≈ 0,79) vs. portfel rynkowy (Sharpe ≈ 1,26); dywersyfikacja 13 spółek >> portfel 2-składnikowy |

**Rekomendowany stack ryzyka dla XTB:**

- Raportowanie dzienne → **FHS GARCH**
- Wymóg kapitałowy (IMA) → **LSTM-FHS** / **EWMA + Hill**
- Stress test (Filar II) → **GPD / POT**
- Alokacja strategiczna → **Markowitz + SIM**

---

## Struktura repozytorium

```
ZRWP/
├── prez3.ipynb              # Projekt 1: miary ryzyka, portfele FX/CFD, EVT
├── prez4.ipynb              # Projekt 2: VaR/EVaR, backtesting, FHS GARCH, EWMA+Hill
├── prez5.ipynb              # Projekt 3: optymalizacja portfela Markowitza (13 spółek GPW)
├── prez5v1.ipynb            # Wcześniejsza wersja prez5 (portfel 2-składnikowy VOTUM/KRUK)
├── podsumowanie.ipynb       # Integracja wyników + wykresy do raportu
├── lstm_var_xtb.py          # LSTM + parametryczny VaR (t-Student)
├── lstm_fhs_xtb.py          # LSTM + FHS (empiryczny kwantyl reszt) — model końcowy
├── wig20_d.csv              # Dzienne notowania WIG20 (2021–2025)
├── figures_raport/          # Wykresy wygenerowane do raportu (fig_01–fig_09)
├── raport_podsumowanie.tex  # Źródło LaTeX raportu podsumowującego
├── raport_podsumowanie.pdf  # Raport końcowy (PDF)
├── raport_analiza_ryzyka_XTB.pdf
└── raport_prez4.pdf
```

---

## Metody i zakres analizy

### `prez3.ipynb` — Klasyczne miary ryzyka i EVT

- **Zmienne:** EUR/PLN, USD/PLN, GBP/PLN, XTB.WA
- Miary zmienności (σ, wariancja, IQR, MAD), kwantyle VaR (empiryczny i parametryczny)
- Portfel walutowy (50/30/20) i portfel CFD (złoto, S&P 500, NASDAQ-100)
- **EVT:** Block Maxima (GEV) i Peaks Over Threshold (GPD)
- Backtesting: testy Kupieca i Christoffersena

### `prez4.ipynb` — VaR, EVaR i backtesting regulacyjny

- **7 metod VaR:** Param Normal, Param t, HS zwykła, HS ważona (BRW), FHS GARCH, EWMA + Hill, MC t / ARMA-GARCH
- **EVaR** i **Expected Shortfall (ES)**
- Backtesting rolling window (W = 500 dni, okres 2020–2025)
- Testy: **Kupiec**, **Christoffersen**, **Berkowitz**, **Basel Traffic Light**
- Bootstrapowe przedziały ufności dla VaR

### `prez5.ipynb` — Optymalizacja portfela Markowitza

- Alokacja **1 mln PLN** w 13 akcjach GPW (PKO BP, PKN Orlen, KGHM, CD Projekt, Dino, VOTUM, KRUK, PZU, Enter Air, Develia, Dom Development, Allegro, Asseco)
- Portfel minimalnej wariancji (MVP), portfel rynkowy (max Sharpe), efektywna granica
- Chmura 10 000 losowych portfeli (Monte Carlo, rozkład Dirichleta)
- Model jednoskładnikowy (SIM) z β względem WIG20
- Ograniczenie VaR 99% ≤ 20% (rocznie)

### `lstm_var_xtb.py` / `lstm_fhs_xtb.py` — Hybrydowy model LSTM

Oba skrypty implementują pipeline:

```
VaR_α,t+1 = −σ̂_LSTM(t+1) · Q_α(z)
```

gdzie LSTM(32) prognozuje log|zwrot| z 6 cech (log-zwroty, wolumen, EWMA, σ_close itd.), a kwantyl reszt `Q_α(z)` pochodzi z:

| Plik | Kwantyl reszt |
|------|---------------|
| `lstm_var_xtb.py` | Parametryczny t-Student (fit na resztach walidacyjnych) |
| `lstm_fhs_xtb.py` | Empiryczny kwantyl FHS (lepszy dla ujemnej skośności) |

Cechy implementacji: rolling refit co 90 dni, grid search (WINDOW × LAMBDA_EWMA × SIGMA_FLOOR), gradient clipping, deterministyczny seed, wsparcie **Apple MPS** / CPU.

### `podsumowanie.ipynb` — Integracja

Notebook łączy wyniki wszystkich projektów, generuje wykresy w jednolitej stylistyce (paleta kolorów XTB) i stanowi źródło figur do raportu LaTeX.

---

## Wymagania

- Python **3.10+**
- Jupyter Notebook / JupyterLab

Główne biblioteki:

| Pakiet | Zastosowanie |
|--------|--------------|
| `numpy`, `pandas` | Obliczenia, szeregi czasowe |
| `yfinance` | Pobieranie danych GPW/Yahoo Finance |
| `matplotlib` | Wizualizacja |
| `scipy` | Statystyka, optymalizacja, rozkłady |
| `scikit-learn` | Skalowanie cech (LSTM) |
| `torch` | Sieci LSTM |
| `arch` | Modele GARCH (prez4, podsumowanie) |

---

## Instalacja i uruchomienie

```bash
# Klonowanie repozytorium
git clone https://github.com/michalmarchwiak/ZRWP.git
cd ZRWP

# Wirtualne środowisko (zalecane)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Instalacja zależności
pip install -r requirements.txt
```

> **PyTorch:** na macOS z Apple Silicon pakiet `torch` z PyPI zwykle wykrywa MPS automatycznie. W razie problemów z instalacją sprawdź [pytorch.org](https://pytorch.org/get-started/locally/).

### Notebooki

```bash
jupyter notebook podsumowanie.ipynb   # pełne podsumowanie (zalecany start)
jupyter notebook prez4.ipynb          # VaR i backtesting
jupyter notebook prez5.ipynb          # optymalizacja portfela
```

### Modele LSTM

```bash
python lstm_fhs_xtb.py    # model końcowy LSTM + FHS
python lstm_var_xtb.py    # wariant z t-Studentem
```

> **Uwaga:** Skrypty LSTM pobierają dane z Yahoo Finance i trenują sieć — pierwsze uruchomienie może zająć kilka minut (GPU/MPS przyspiesza trening).

---

## Dane

| Źródło | Zawartość | Okres |
|--------|-----------|-------|
| Yahoo Finance (`yfinance`) | XTB.WA, akcje GPW, kursy FX, indeksy | 2018–2025 |
| `wig20_d.csv` (lokalny) | Dzienne OHLCV WIG20 | 2021–2025 |

Główny zbiór analityczny: **n = 2030** obserwacji dziennych log-zwrotów XTB.WA (2018-01-01 – 2025-12-30).

---

## Raporty i wykresy

| Plik | Opis |
|------|------|
| `raport_podsumowanie.pdf` | Formalne podsumowanie wszystkich projektów (LaTeX → PDF) |
| `raport_analiza_ryzyka_XTB.pdf` | Wcześniejszy raport analityczny |
| `raport_prez4.pdf` | Raport z projektu VaR/EVaR |
| `figures_raport/fig_01.png` – `fig_09.png` | Wykresy do raportu (charakterystyka, K-S, VaR, backtest, EVT, LSTM, Markowitz) |

Kompilacja raportu LaTeX (opcjonalnie):

```bash
pdflatex raport_podsumowanie.tex
```

---

## Autor

**Michał Marchwiak**  
Semestr letni 2025/2026 — Zarządzanie Ryzykiem w Przedsiębiorstwie

---

## Licencja

Projekt akademicki — kod i materiały udostępniane wyłącznie w celach edukacyjnych. Dane rynkowe pochodzą z publicznych źródeł (Yahoo Finance, GPW) i podlegają warunkom tych serwisów.
