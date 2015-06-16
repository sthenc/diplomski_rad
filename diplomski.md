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

CHiME


### Motivacija
U zadnjih nekoliko godina veoma je značajno CHiME natjecanje u razdvajanju
i prepoznavanju govora (engl. CHiME Speech Separation and Recognition Challenge)
kao platforma gdje različite istraživačke skupine iz akademskog svijeta i industrije
mogu usporediti svoja rješenja na skupu podataka koji daje dobar pokazatelj koliko
dobro bi ta rješenja mogla raditi na stvarnim podacima. [chime_data]

Iako je u tijeku već treća iteracija CHiME natjecanja, u ovom radu će se
koristiti podaci za drugo CHiME natjecanje (engl. 2nd CHiME challenge),
budući da u ovom trenutku već postoji mnogo objavljenih rezultata za taj skup podataka,
a napravljena su i neka poboljšanja u odnosu na prvu verziju. [chime_overview]

Cilj CHiME natjecanja je dobiti što veću točnost prepoznavanja govora
izobličenog sa realističnim izvorima smetnji.
Problem pročišćivanja govora za automatsko prepoznavanje je različit od običnog pročišćivanja
govora, zato jer velik broj tehnika pročišćivanja govora samo poboljšava doživljaj
kvalitete govora, ali ne povećava i njegovu razumljivost. [book_speech_enhancement]

Računalno prepoznavanje govora čak i u uvjetima savršeno čistog govornog signala je težak problem.
Faktori koji taj problem mogu dodatno otežati su promjenjivost položaja i udaljenost govornika u odnosu na mikrofon, 
veličina rječnika i prirodnost govora. 
Budući da je fokus CHiME natjecanja izdvajanje govora, autori su odlučili napraviti
set podataka sa realnim signalima smetnje, snimljenim u pravoj dnevnoj sobi.
Tako je govoru superponirana izrazito realna smetnja, no kako bi zadatak ostao
rješiv, govor kojeg treba prepoznati je nerealno jednostavan. [chime_data]


###Grid corpus 
Izvor čistog govora je Grid korpus govora, koji se sastoji
od zvučnih zapisa jednostavnih komandi od 34 različita govornika engleskog jezika. [chime_grid_cite]
Zvučni zapisi su rečenice od šest riječi u obliku 
<naredba:4><boja:4><prijedlog.:4><slovo:25><znamenka:10><prilog:4>,
brojevi u uglatim zagradama označuju koliko u svakoj kategoriji ima mogućih riječi.
Zadatak je prepoznati slovo i znamenku, i točnost prepoznavanja se mjeri 
samo na te dvije riječi.

Dakle govor kojeg treba prepoznati se sastoji od malog rječnika i jednostavne gramatike,
nije prirodan i govornik je uglavnom na istom položaju, što prepoznavanje
čistog govora čini vrlo laganim. Točnost za čisti govor je 97.25% [chime_data], usporedivim sa 
ljudskom točnosti koja je u tom slučaju oko 98.3% (točnost za slova je 99.05%, a za brojke 99.3%) [chime_grid_cite].

###Smetnje
Kako bi se dobio željeni raspon odnosa signala prema smetnji (engl. SNR),
izgovorene rečenice su tako pozicionirane u odnosu na pozadinsku buku da se dobiju željenne
vrijednost: -6, -3, 0, 3, 6 i 9 dB. To je učinjeno tako da je pozadinska buka
nasumično pretražena i odabran je onaj vremenski interval koji ima željeni SNR
za tu rečenicu. Na 9dB, najpoboljnijem odnosu željenog i neželjenog signala,
smetnje su uglavnom kvazi-stacionarni šumovi, dok su oni na -6dB uglavnom
iznenadni nestacionarni zvučni događaji.
Da bi zadatak bio još realističniji, napravljena je konvolucija čiste izgovorene rečenice 
sa binauralnim impolsnim odzivima sobe [ BRIR-om] koji simuliraju jeku i ograničeno pomicanje govornika. 
[chime_data]

###Podaci
Sve snimke su u 16-bitnom WAV formatu uzorkovanom na 16kHz.
Set podataka za uvježbavanje (engl. training set) sarži 17000 rečenica, 500 za svakog od 34 govornika.
Razvojni set podataka (engl. development set) i ispitni set podataka (engl. test set)
sadrže 600 rečenica na 6 različitih SNR-a. [chime_website]

###Ocjenjivanje točnosti
Osim ispitnih podataka, u sklopu CHiME 2 natjecanja dostupni su i alati za
provođenje mjerenja točnosti baziranih na sustavu za prepoznavanje baziranom
na besplatnom i standardnom HTK programskom paketu [book_htk].
[chime_readme] Iako CHiME dozvoljava korištenje vlastitog
rješenja za prepoznavanje govora, na raspolaganje je stavljen osnovni (engl. baseline)
sustav sa nekoliko unaprijed istreniranih modela.
To je učinjeno kako bi se moglo odrediti koji dio poboljšanja točnosti se može
pripisati pročišćavanju govora, a koji sustavu za prepoznavanje govora 
i olakšala usporedba između različitih sustava. [chime_readme]

Osnovni sustav za prepoznavanje je prilagođen sintaksi rečenica u Grid korpusu.
Sustav je baziran na skrivenim markovljevim modelima. Svaka od 51 riječi
prisutna u Grid korpusu modelirana pomoću skrivenog markovljevog modela sa 
2 stanja po fonemu. Vjerojatnost izostavljanja svakog stanja je predstavljena pomoću
mješovitog Gaussovog modela sa 7 komponenti i dijagonalnom kovarijancom. [chime_data]

Govor je parametriziran kao niz standardnih MFCC značajki.
Svaki vektor značajke sadrži 39 parametara:
12 mel-kepstralnih koeficijenata koji su normalizirani po srednjoj vrijednosti
(ali ne i standardnoj devijaciji) (engl. CMN), zatim logaritamska energija okvira,
ta zatim 13 differencijalnih koeficijenata prvog reda i 13 drugog reda 
(engl. delta and acceleration coefficients). 
Standardna HTK šifra za te značajke je MFCC_E_D_A_Z i detaljno (ali ne i jednoznačno)
je opisana u literaturi [book_htk][book_opensmile]
MFCC značajke se računaju na vremenskim okvirima od 25 ms, a korak je 10 ms.
Budući da su zvučni podaci dani u stereo formatu, signal je pretvoren u
mono signal uzimanjem srednje vrijednosti oba kanala. [chime_data]


Odabir strategije

- po mogućnosti rezultati na CHiME-u [chime_overview - proletit prije spavanja]

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

	-12 dana, 211 epoha po 4850 sekundi (oko 1h 21 min)

Nešto o validation i test setu

Rezultati - dobivena točnost, u usporedbi sa očekivanom

OpenSMILE - mjerenje brzine real-time izvršavanja mreže ?

---------- Primjena -----------




# Rezultati i diskusija
	- obrada rezultata, što se iz njih može zaključiti




# Zaključak
	- koliko je super, ali da poboljšanje na small-vocabulary tasku ne znači
	nužno da će biti toliko na mid i big vocabulary task

Indeed, ASR systems can be surpris-
ingly sensitive to speaker location and it is well known that
systems optimized for small vocabulary read speech often fail
to scale to larger vocabulary spontaneous speech.[chime_data]


# Literatura

[ubacit iz literatura.md]
