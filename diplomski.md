title: Neuralne mreže u izdvajanju govornog signala iz zvučnog zapisa
author: Stjepan Henc
##Neuralne mreže u izdvajanju govornog signala iz zvučnog zapisa

# Kratice

GPGPU (engl. General Purpouse Graphic Processor Unit - grafički procesora opće namjene)
GPU
BSS
BLSTM
LSTM
RNN
CUDA

# Uvod

Zadatak ovog diplomskog rada je pronaći prikladan programski paket za 
izdvajanje govornog signala u zvučnim zapisima u kojima su prisutne 
smetnje, jeka, žamor i slično. 
Izdvajanje tj. pročišćavanje govora je ključan korak u velikom broju praktičnih
primjena računalnog prepoznavanja govora. To je zato jer je mnogo lakše doći
do nekvalitetnih nego do kvalitetnih snimki govora.
Također, većini sustava za računalno prepoznavanje govora performanse naglo
padaju ako se ulaznom govornom signalu dodaje smetnja. [ referenca? ]

Dokaz da je problem moguće riješiti je činjenica da ljudska bića mogu 
razumjeti govor i kada je signal smetnje višestruko jači od korisnog signala. [ referenca? ]
Proučavanje ljudskog slušnog sustava dalo je mnoge ideje za rješavanje ovog problema,
između ostalih i primjenu neuronskih mreža za izvlačenje značajki (engl. feature extraction).

Tip neuronske mreže koji se pokazao najprikladniji za ovaj problem je
rekurzivna neuronska mreža (RNN) sa dvosmjernom dugom-kratkom memorijom (BLSTM).
Iako je taj pristup već poznat duže vrijeme [ reference ? ], pojava pristupačnih
GPGPU-a u zadnjih nekoliko godina i vrtoglavi rast računalne snage koji je to
uzrokovalo omogućio je i njihovu praktičnu primjenu.
To je dio većeg trenda u strojnom učenju poznatog pod imenom Deep Learning, tj. uže Deep Neural Networks.
[citirat nešto od Andrew Ng-a o algoritamskoj efikasnosti i poboljšavanju performansi strojnog učenja]
Budući da je to u zadnje vrijeme vrlo popularno područje za istraživanje,
pojavili su se mnogi programski paketi koji bi bili prikladni za tu namjenu 



što je zadatak (problem)
zašto ga rješavamo (motivacija)
kako ga rješavamo (pristup)
opis organizacije ostatka dokumenta (najčešće, po poglavljima)
za ovo obično ne treba više od 1 stranice


# Zaključak


# Literatura
