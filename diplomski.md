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
SDR signal-to-distortion ratio
SNR
CHiME


# Uvod

U praksi je jako često da zvučne zapisi govora koje treba pretočiti u 
tekstualne zapise (engl. speech-to-text) sadrže i razne smetnje 
koje su prisutne jer govor nije bio sniman u studijskim uvjetima.
Smetnje mogu biti razni šumovi, buka, muzika ili najgorem slučaju čak
i drugi govor.
Sve takve smetnje uzrokuju jako veliki pad točnosti računalnog prepoznavnja
govora. [ book_articulation ]
Zadatak ovog diplomskog rada je pronaći odgovarajući algoritam i programski
paket koji bi omogućio izdvajanje što čišćeg govora iz takvih zvučnih zapisa.
Između mnogih opcija kao algoritam je odabrana RNN-BLSTM neuronska mreža 
implementiran u programskom paketu CURRENNT, koji ubrzava treniranje modela 
pomoću GPU-a.

Efikasnost ovog pristupa je ispitana pomoću skupa podataka "CHiME 2nd Challenge"  
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

Stanje istraživanja

U svim primjenama koje se bave govorom, prisutnost smetnji je neizbježna.
Bilo da se radi o snimanju zvuka, telekomunikacijama ili ljudsko-računalnim
sučeljima (engl. human–machine interfaces), mikrofon koji snima govor uglavnom
će snimiti i smetnje. Zbog toga se snimljeni govorni signal treba pročistiti
digitalnom obradom signala prije svoje upotrebe.
Taj proces pročišćavanja govora obično se naziva ili suzbijanje
buke (engl. noise reduction) ili poboljšavanje govora (engl. speech enhancement)
ili izdvajanje govora (engl. speech separation). 
To je područje koje se intenzivno proučava već nekoliko desetljeća [book_springer],
no problem još uvijek nije riješen.

Posebno važna primjena izdvajanja govora je računalno prepoznavanje govora (engl. ASR).
Iz inžinjerske perspektive ljudsko uho radi nevjerojatno dobro.
Kod ljudskih bića razumijevanje govora, ovisno o primjeni, počinje padati
kada je SNR od -6 do 0 dB, a tek se od -25 do -20 dB sasvim gubi razumljivost.
Računalno prepoznavanje govora počinje gubiti na točnosti več oko +20 dB,
a na 0 dB (jednaka snaga signala govora i smetnje) se već približava nasumičnom
pogađanju. [book_articulation]

Činjenica da su ljudi tako sposobni u obavljanju tog zadatka potaknula je mnoge
istraživače da u proučavanju ljudskog slušnog sustava pokušaju naći inspiraciju
za nova tehnološka rješenja [book_asa][book_casa].

Postoje tri glavna pristupa za izdvajanje govornog signala:

1. Tehnologija nizova mikrofona (engl. microphone-array technology)
2. Slijepo razdvajanje signala
3. Razdvajanje temeljeno na modelu govora

Prvi od tih pristupa, temeljen na akustičnoj tehnologiji, je tradicionalan
i već u prisutan u praksi. No, iako radi dovoljno dobro u određenim uvjetima,
kompaktnija rješenja su daleko od ljudske fleksibilnosti i sposobnosti.
Drugi pristup se temelji na teoriji informacija, i trenutno je glavni trend
na polju istraživanja razdvajanja signala.
Treći pristup pokušava donekle emulirati ljudski slušni sustav koristeći
apriorno znanje tj. model ciljnih signala kao najvažniji faktor.
U usporedbi sa slijepim razdvajanjem signala ovaj pristup je još u povojima,
no njegovu opravdanost potvrđuju mnoga otrkića na području psihologije
[book_nn_sp].

U zadnjih nekoliko godina veoma je značajno CHiME natjecanje u razdvajanju
i prepoznavanju govora (engl. CHiME Speech Separation and Recognition Challenge)
kao platforma gdje različite istraživačke skupine iz akademskog svijeta i industrije
mogu usporediti svoja rješenja na prilično realističnom skupu podataka. [chime_data]

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

što onda točno radimo - ubacit malo formalizma čisto iz fore
[slika]

Odabir programskog paketa

- usporedba svih živih

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
