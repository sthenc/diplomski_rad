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

U praksi je jako često da zvučne zapisi govora koje treba pretočiti u 
tekstualne zapise (engl. speech-to-text) sadrže i razne smetnje 
koje su prisutne jer govor nije bio sniman u studijskim uvjetima.
Smetnje mogu biti razni šumovi, buka, muzika ili najgorem slučaju čak
i drugi govor.
Sve takve smetnje uzrokuju jako veliki pad točnosti računalnog prepoznavnja
govora. [ referenca? ]
Zadatak ovog diplomskog rada je pronaći odgovarajući algoritam i programski
paket koji bi omogućio izdvajanje što čišćeg govora iz takvih zvučnih zapisa.
Između mnogih opcija kao algoritam je odabrana RNN-BLSTM neuronska mreža [ referenca? ]
implementiran u programskom paketu CURRENNT, koji ubrzava treniranje modela [ referenca? ]
pomoću GPU-a.

Efikasnost ovog pristupa je ispitana pomoću skupa podataka "CHiME 2nd Challenge"  [ referenca? ]
i pripadnih alata.

[stavit brojeve]
U poglavlju "Pregled literature" opisan je kratak pregled literature i stanja
istraživanja na ovom području, s posebnim naglaskom na ono što je korišteno u ovom radu.

U poglavlju "Metodologija" opisan je rad RNN-BLSTM algoritma, kako je tehnički
izveden CURRENNT i detalji o "CHiME 2nd Challenge" skupu podataka.

U poglavlju "Primjena" opisano je što je bilo potrebno napraviti da bi se
istrenirala neuronska mreža i koliko je poboljšanje u prepoznavanju govora
postignuto.

U poglavlju "Rezultati i diskusija" dani su rezultati i što oni znače za
računalno prepoznavanje govora u praksi.




{Upute:
što je zadatak (problem)
zašto ga rješavamo (motivacija)
kako ga rješavamo (pristup)
opis organizacije ostatka dokumenta (najčešće, po poglavljima)
za ovo obično ne treba više od 1 stranice/
}



----- Pregled literature ---------

Opis problema


Izdvajanje tj. pročišćavanje govora je ključan korak u velikom broju praktičnih
primjena računalnog prepoznavanja govora. U mnogim primjenama je nemoguće
Također, većini sustava za računalno prepoznavanje govora performanse naglo
padaju ako se ulaznom govornom signalu dodaje smetnja. [ referenca? ]

Dokaz da je problem moguće riješiti je činjenica da ljudska bića mogu 
razumjeti govor i kada je signal smetnje višestruko jači od korisnog signala. [ referenca? ]
Proučavanje ljudskog slušnog sustava dalo je mnoge ideje za rješavanje ovog problema,
između ostalih i primjenu neuronskih mreža za izvlačenje značajki (engl. feature extraction).


CHiME

Odabir strategije

Tip neuronske mreže koji se pokazao najprikladniji za ovaj problem je
rekurzivna neuronska mreža (RNN) sa dvosmjernom dugom-kratkom memorijom (BLSTM).
Iako je taj pristup već poznat duže vrijeme [ reference ? ], pojava pristupačnih
GPGPU-a u zadnjih nekoliko godina i vrtoglavi rast računalne snage koji je to
uzrokovalo omogućio je i njihovu praktičnu primjenu.
To je dio većeg trenda u strojnom učenju poznatog pod imenom Deep Learning, tj. uže Deep Neural Networks.
[citirat nešto od Andrew Ng-a o algoritamskoj efikasnosti i poboljšavanju performansi strojnog učenja]
Budući da je to u zadnje vrijeme vrlo popularno područje za istraživanje,
pojavili su se mnogi programski paketi koji bi bili prikladni za tu namjenu 

što onda točno radimo
[slika]

Odabir programskog paketa

----- Pregled literature ---------




--------- Metodologija ------

Opis BLSTM-RNN

Opis CURRENNT

Opis kriterija testiranja kod CHiME

--------- Metodologija ------





---------- Primjena -----------

Priprema podataka
  - pomoćne skripte, openSMILE ?

Uvježbavanje algoritma, opis radne okoline i stroja, komentar na trajanje

Nešto o validation i test setu

Rezultati - dobivena točnost, u usporedbi sa očekivanom

OpenSMILE - mjerenje brzine real-time izvršavanja mreže ?

---------- Primjena -----------




# Rezultati i diskusija
	- obrada rezultata, što se iz njih može zaključiti




# Zaključak
	- koliko je super, ali da poboljšanje na small-vocabulary tasku ne znači
	nužno da će biti toliko na mid i big vocabulary task




# Literatura

[ubacit iz literatura.md]
