# Analiza ryzyka akcji XTB S.A.

Projekt z przedmiotu **Zarządzanie Ryzykiem w Przedsiębiorstwie** — kompleksowa analiza ryzyka spółki **XTB S.A.** (`XTB.WA`) notowanej na GPW w okresie **2018–2025**.

Repozytorium obejmuje pięć etapów pracy projektowej (notebooki `prez3`–`prez6`), hybrydowy model **LSTM + FHS**, optymalizację portfela akcji GPW, strategię zabezpieczenia ryzyka walutowego oraz podsumowanie wyników w `podsumowanie.ipynb`.

---

## Spis treści

- [Kontekst i cel](#kontekst-i-cel)
- [Główne wnioski](#główne-wnioski)
- [Struktura repozytorium](#struktura-repozytorium)
- [Metody i zakres analizy](#metody-i-zakres-analizy)
- [Wymagania](#wymagania)
- [Instalacja i uruchomienie](#instalacja-i-uruchomienie)
- [Dane](#dane)
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
6. Zabezpieczenie ekspozycji walutowej **EUR/PLN** i **USD/PLN** kontraktami terminowymi **FEUR** i **FUSD** na GPW.

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
| Hedging FX | Kontrakty GPW redukują VaR 99% (1 dzień) z ~4,0 mln PLN do ~0,014 mln PLN; CFD na platformie XTB **nie stanowi hedżu** (net zero w grupie) |

**Rekomendowany stack ryzyka dla XTB:**

- Raportowanie dzienne → **FHS GARCH**
- Wymóg kapitałowy (IMA) → **LSTM-FHS** / **EWMA + Hill**
- Stress test (Filar II) → **GPD / POT**
- Alokacja strategiczna → **Markowitz + SIM**
- Ekspozycja walutowa → **FEUR / FUSD na GPW** (KDPW, fixing NBP)

---

## Struktura repozytorium

```
Enterprise-Risk-Management/
├── prez3.ipynb              # Projekt 1: miary ryzyka, portfele FX/CFD, EVT
├── prez4.ipynb              # Projekt 2: VaR/EVaR, backtesting, FHS GARCH, EWMA+Hill
├── prez5.ipynb              # Projekt 3: optymalizacja portfela Markowitza (13 spółek GPW)
├── prez6.ipynb              # Projekt 4: hedging FX (FEUR/FUSD), Delta, VaR strategii
├── podsumowanie.ipynb       # Integracja wyników + wykresy raportowe
├── lstm_fhs_xtb.py          # LSTM + FHS (empiryczny kwantyl reszt)
├── wig20_d.csv              # Dzienne notowania WIG20 (stooq.pl, 2021–2025)
├── requirements.txt
├── .gitignore
└── README.md
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
- Model jednoskładnikowy (SIM) z β względem WIG20 (dane z `wig20_d.csv`)
- Ograniczenie VaR 99% ≤ 20% (rocznie)

### `prez6.ipynb` — Zabezpieczenie ryzyka walutowego

- Ekspozycja netto: **50 mln EUR** i **30 mln USD** (struktura przychodów z `prez3`)
- Instrumenty: kontrakty terminowe **FEUR** i **FUSD** na GPW (fixing NBP, clearing KDPW)
- **Hedge ratio** minimum wariancji, liczba kontraktów, depozyt zabezpieczający i variation margin
- Miary wrażliwości: **Delta**, P&L przy ruchu ±1%
- **VaR 99%** (1 dzień) strategii przed i po hedżu: HS, parametryczna, t-Student + **ES**
- Analiza kosztów rollowania, basis risk i scenariuszy stress (±3σ)
- Uzasadnienie wykluczenia **CFD na platformie XTB** jako narzędzia hedgingu

### `lstm_fhs_xtb.py` — Hybrydowy model LSTM

Pipeline:

```
VaR_α,t+1 = −σ̂_LSTM(t+1) · Q_α(z)
```

LSTM(32) prognozuje log|zwrot| z 6 cech (log-zwroty, wolumen, EWMA, σ_close itd.), a `Q_α(z)` to empiryczny kwantyl standaryzowanych reszt FHS. Rolling refit co 90 dni, grid search (WINDOW × LAMBDA_EWMA × SIGMA_FLOOR), wsparcie **Apple MPS** / CPU.

### `podsumowanie.ipynb` — Integracja

Notebook łączy wyniki projektów `prez3`–`prez5` i modelu LSTM, generuje wykresy w jednolitej stylistyce (paleta kolorów XTB).

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

Pełna lista w [`requirements.txt`](requirements.txt).

---

## Instalacja i uruchomienie

```bash
# Klonowanie repozytorium
git clone https://github.com/michalmarchwiak/Enterprise-Risk-Management.git
cd Enterprise-Risk-Management

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
jupyter notebook prez6.ipynb          # hedging walutowy FEUR/FUSD
```

### Model LSTM

```bash
python lstm_fhs_xtb.py
```

> **Uwaga:** Skrypt LSTM pobiera dane z Yahoo Finance i trenuje sieć — pierwsze uruchomienie może zająć kilka minut (GPU/MPS przyspiesza trening).

---

## Dane

| Źródło | Zawartość | Okres |
|--------|-----------|-------|
| Yahoo Finance (`yfinance`) | XTB.WA, akcje GPW, kursy FX (EUR/PLN, USD/PLN) | 2018–2025 |
| `wig20_d.csv` (lokalny, [stooq.pl](https://stooq.pl)) | Dzienne OHLCV WIG20 | 2021–2025 |
| GPW | Specyfikacja kontraktów FEUR / FUSD (fixing NBP) | — |

Główny zbiór analityczny: **n = 2030** obserwacji dziennych log-zwrotów XTB.WA (2018-01-01 – 2025-12-30).

---

## Autor

**Michał Marchwiak**  
Semestr letni 2025/2026 — Zarządzanie Ryzykiem w Przedsiębiorstwie

---

## Licencja

Projekt akademicki — kod i materiały udostępniane wyłącznie w celach edukacyjnych. Dane rynkowe pochodzą z publicznych źródeł (Yahoo Finance, GPW, stooq.pl) i podlegają warunkom tych serwisów.
