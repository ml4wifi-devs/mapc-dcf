# IEEE 802.11 Multi-AP Distributed Coordination Function (DCF) Simulator

# Installation

1. Najlepiej jakby python był w jednej z wersji 3.8-3.10, na tych testowaliśmy kod
2. Należy sobie stworzyć środowisko wirtualne: `python -m venv some-cool-venv-name`
3. Aktywować to środowisko: `source ./some-cool-venv-name/bin/activate`
4. Sprawdzić czy wersja pythona się zgadza: `python --version` (u mnie jest 3.10.4)
5. Teraz pobieramy repo mapc-mab, instalujemy je, i ta instalacja automatycznie zaciągnie też mapc-sim z PyPI  
    i) `git clone https://github.com/ml4wifi-devs/mapc-mab.git`  
    ii) `cd mapc-mab`  
    iii) `pip install .`
6. Teraz pobieramy repo mapc-dcf i je instalujemy  
    i) `cd ..`  
    ii) `git clone https://github.com/ml4wifi-devs/mapc-dcf.git` (Domyślnie będziemy w main, to dobrze, bo tam wrzuciłem zaktualizowane requirements)  
    iii) `cd mapc-dcf`  
    iV) `pip install -e .` (Tutaj dodatkowa flaga -e, która pomaga przy zmianach kodu. Warto ją dodać)
7. W zasadzie to tyle, teraz uruchamiamy:  
    i) Najpierw przechodzimy do interesującego nas brancha, polecam `joblib`, ale `dcf` też da rade  
    ii) Możemy wywołać run z flagą help, która podpowie nam jakie są wymagane argumenty `python mapc_dcf/run.py -h`  
    iii) Tworzymy plik konfiguracyjny symulację. Tu tak jak zauważyłaś, jest pewna rozbieżność w loggerach między branchem joblib, a resztą, postaram się to ujednolicić w przyszłości ale to chyba nie jest priorytet  
    iV) Ale jeśli mamy stworzony config, to możemy uruchomić symulację: `python mapc_dcf/run.py -c configs/debug_dense_n2.json`
