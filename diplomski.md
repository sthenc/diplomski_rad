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
govora. [ book_articulation:2]
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
To je područje koje se intenzivno proučava već nekoliko desetljeća ,
no problem još uvijek nije riješen [book_springer:843-845].

Posebno važna primjena izdvajanja govora je računalno prepoznavanje govora (engl. ASR).
Iz inžinjerske perspektive ljudsko uho radi nevjerojatno dobro.
Kod ljudskih bića razumijevanje govora, ovisno o primjeni, počinje padati
kada je SNR od -6 do 0 dB, a tek se od -25 do -20 dB sasvim gubi razumljivost.
Računalno prepoznavanje govora počinje gubiti na točnosti več oko +20 dB,
a na 0 dB (jednaka snaga signala govora i smetnje) se već približava nasumičnom
pogađanju. [book_articulation:2]

Činjenica da su ljudi tako sposobni u obavljanju tog zadatka potaknula je mnoge
istraživače da u proučavanju ljudskog slušnog sustava pokušaju naći inspiraciju
za nova tehnološka rješenja [book_asa][book_casa][book_human_machine].

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
[book_nn_sp: 184].

CHiME


### Motivacija
U zadnjih nekoliko godina veoma je značajno CHiME natjecanje u razdvajanju
i prepoznavanju govora (engl. CHiME Speech Separation and Recognition Challenge)
kao platforma gdje različite istraživačke skupine iz akademskog svijeta i industrije
mogu usporediti svoja rješenja na skupu podataka koji daje dobar pokazatelj koliko
dobro bi ta rješenja mogla raditi na stvarnim podacima. [chime_data][chime_cite2]

Iako je u tijeku već treća iteracija CHiME natjecanja, u ovom radu će se
koristiti podaci za drugo CHiME natjecanje (engl. 2nd CHiME challenge),
budući da u ovom trenutku već postoji mnogo objavljenih rezultata za taj skup podataka,
a napravljena su i neka poboljšanja u odnosu na prvu verziju. [chime_overview]

Cilj CHiME natjecanja je dobiti što veću točnost prepoznavanja govora
izobličenog sa realističnim izvorima smetnji.
Problem pročišćivanja govora za automatsko prepoznavanje je različit od običnog pročišćivanja
govora, zato jer velik broj tehnika pročišćivanja govora samo poboljšava doživljaj
kvalitete govora, ali ne povećava i njegovu razumljivost. [book_speech_enhancement:609]

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
mješovitog Gaussovog modela sa 7 komponenti i dijagonalnom kovarijancom.
Dana su tri unaprijed uvježbana modela za prepoznavanje:
1. "Čisti" model (eng. clean - treniran na čistom govoru)
2. "Jeka" model (eng. reverberated - treniran na govoru izobličenom jekom)
3. "Buka" model (eng. noisy - treniran na govoru izobličenom bukom)

Osim ta tri modela, dostupni su alati za lako uvježbavanje tj. prilagođavanje modela (eng. retraining)
na skupu podataka za treniranje izobličenom algoritmom za pročišćavanje [chime_data].

Govor je parametriziran kao niz standardnih MFCC značajki.
Svaki vektor značajke sadrži 39 parametara:
12 mel-kepstralnih koeficijenata koji su normalizirani po srednjoj vrijednosti
(ali ne i standardnoj devijaciji) (engl. CMN), zatim logaritamska energija okvira,
ta zatim 13 differencijalnih koeficijenata prvog reda i 13 drugog reda 
(engl. delta and acceleration coefficients). 
Standardna HTK šifra za te značajke je MFCC_E_D_A_Z i detaljno (ali ne i jednoznačno)
je opisana u literaturi [book_htk:80][book_opensmile:32]
MFCC značajke se računaju na vremenskim okvirima od 25 ms, a korak je 10 ms.
Budući da su zvučni podaci dani u stereo formatu, signal je pretvoren u
mono signal uzimanjem srednje vrijednosti oba kanala. [chime_data]


Odabir strategije

Zanimljiva povijesna činjenica je da su neuronske mreže u području slijepog
razdvajanja signala prisutne od samog početka istraživanja na tom području
80-ih godina prošlog stoljeća.
Prvi algoritam koji je korišten je analiza principalnih komponenata (eng. PCA),
gdje se parametrizirana reprezentacija signala (najčešće spektar) pokušavala
razdvojiti na komponente koje odgovaraju pojedinim izvorima signala pomoću
određenih statističkih svojstava tih signala.
Analiza neovisnih komponenata (eng. ICA) još je jedna metoda koja se može
svrstati u metode strojnog učenja pomoću neuronskih mrežam, a nastala je
kao poboljšanje originalnog PCA algoritma [book_bss_ica:7-9] [book_nn_sp:180].

U literaturi se mogu naći stvarno brojni i nerijetko vrlo složeni pristupi ovoj problematici, 
no valja izdvojiti dva koja su se pokazala posebno uspješnima i popularnima u posljednjih
nekoliko godina.

To su nenegativna faktorizacija matrica (eng. NMF) [book_bss_ica:515] i 
duboke neuronske mreže (eng. DNN).
Oba pristupa su relativno jednostavna, no NMF ima nekoliko nedostataka u
usporedbi sa DNN.
NMF je isključivo linearan model, dok DNN (ovisno o konkretnoj izvedbi)
u pravilu može modelirati i nelinearno preslikavanje iz izvora signala 
u mješavinu.
Također, kod primjene istreniranog NMF modela mora se provoditi iterativni
postupak koji uključuje operacije množenja matrica, što je jako zahtjevno
po računalne resurse.
S druge strane, DNN-ovi se u pravilu duže treniraju, ali se zato primjena
istreniranog modela sastoji samo od množenja nekoliko matrica, što ih
čini pogodnima sa primjenu u stvarnom vremenu.
Svi ti faktori čine DNN-ove moćnijim i bržim modelom (jednom kada ga se uspije istrenirati) [dnn_faster_nmf].

No, zanimljivo je da je na CHiME 2nd challenge pobijedio sustav koji, između ostalih,
koristi oba ova pristupa [wen_chime_pobjednik], te su u literaturi poznate 
razne kombinacije ovih pristupa [dnn_nmf][dnn_vs_nmf_novo][deep_nmf].

Uglavnom svi visokorangirani sustavi koriste kombinaciju nekoliko složenih pristupa,
i za razliku od ovog rada nije im cilj doći do sustava koji bi bio dovoljno brz za
primjenu u praksi, već pod svaku cijenu dobiti čim veće performanse na testnim podacima [chime_overview].

Duboke neuronske mreže su dio jednog većeg pokreta na području umjetne inteligencije
pod nazivom duboko učenje.
Ideja vodilja tog pokreta je da su korištenje veće količine podataka [ang_banko_brill_scale]
i većih modela [ang_coates_model_size] glavni motori povećanja performansi.
Cilje je iskoristiti sve veću raspoloživu računalna moć kako bi se u osnovi stari
algoritmi iskoristili za rješavanje dosad nezamislivih problema.
Računalni resursi koji se koriste mogu biti tisuće servera u nekoj od velikih
internetskih kompanija [ang_large_dnn], ili pak grafički procesori [ang_cudnn] koji 
danas i običnim studentima čine dostupnom računalnu moć u rangu nekadašnjih 
superračunala.
No, najmoćnija je kombinacija više servera sa nekoliko grafičkih procesora,
što omogućuje brzo treniranje neuronskih mreža sa nekoliko milijardi parametara [ang_cots_hpc].

Povećani intenzitet istraživanja na ovom području doveo je i do novih
algoritama i ideja, između ostalog i obećavajućih postignuća na području
računalnog prepoznavanja govora [ang_deep_speech][graves14].


Dosad je u ovom poglavlju pojam DNN korišten kao da se radi o jednom pristupu,
no samo u području pročišćavanja govora može se odnositi na mnogo različitih
tipova mreža sa različitim svojstvima[dnn_turci][dnn_wang_ss_frontend][dnn_rnn_smaragdis]
[dnn_kinezi][dnn_multitalker][dnn_music].

Povećani interes za i šira primjena neuronskih mreža dovela 

Tip duboke neuronske mreže koji se pokazao najprikladniji za ovaj problem je
rekurzivna neuronska mreža (RNN) sa dvosmjernom dugom-kratkom memorijom (BLSTM),
i to će biti daljnji fokus ovog rada [wen_chime13][wen_chime14][wen_sdr_lstm][wen_chime1].


Odabir programskog paketa

Budući da su duboke neuronske mreže u zadnje vrijeme vrlo popularno područje za istraživanje,
pojavili su se mnogi programski paketi koji olakšavaju njihovu upotrebu.

Budući da je odabran BLSTM tip rekurzivne neuronske mreže, u obzir dolaze samo paketi
koji podržavaju takve slojeve. 
Odabrani paket također mora podržavati ubrzavanje izvođenja na grafičkim 
procesorima i biti općenito dovoljno učinkovit, jer bi u suprotnom treniranje
mreže moglo premašiti trajanje ljetnog semestra.

Tablica [broj N] daje usporedbu dostupnih paketa otvorenog koda i neke
njihove karakteristike [pybrain_cite][theano_cite1][theano_cite2][torch7_cite][wen_currennt_cite][rnnlib]. 

			pybrain		    torch7 	      theano          rnnlib           CURRENNT
GPU 		ne                 da             da              ne               da
BLSTM		da                 ne             ne              da               da
jezik       python             lua/c          python          c++              c++

Jedino programski paket CURRENNT zadovoljava sve potrebne kriterije,
podržava BLSTM-RNN neuronske mreže, ubrzavanje izvršavanja na grafičkim procesorima
korištenjem biblioteke CUDA [cuda-cite] i napisan je u programskom jeziku C++,
što vjerojatno znači da će zadovoljavajuće brzo obaviti treniranje mreže.

----- Pregled literature ---------

--------- Metodologija ------

Rekurzivne neuronske mreže

Naziv "rekurzivna neuronska mreža" u užem smislu odnosi se na nadogradnju višeslojnog
perceptrona [graves_blstm : 20]. U najčešćoj varijanti sloju se uz uobičajenu
pobudu daje i njegov izlaz iz prethodnog trenutka (pod trenutak se podrazumijeva
pozicija na vremenskoj ili prostornoj osi).
Slika [rnn.png] daje primjer jedne takve mreže.

Ovaj tip mreže najčešće se koristi kada je u problemu potrebno iskoristiti
kontekst, npr. prepoznavanja rukopisa, zato jer ova nadogradnja omogućava
mreži da u svojem internom stanju pohrani informaciju o prethodnim ulazima.

Prolaz unaprijed kod rekurzivne neuronske mreže izgleda isto kao kod 
višeslojnog perceptrona, no kod prolaza unazad koristi se BPTT algoritam
(engl. backpropagation through time).
Ideja algoritma je da se mreža "razmota", tako da se mreži na ulaz odjednom
da cijeli ulazni niz. Izračuna se izlaz mreže za cijeli niz, i izračunaju
se greške za svaki korak. Budući da se zbog razmotavanja mreže težine ponavljaju,
za svaku težinu se zbroje sve pripadne greške i s tom vrijednošću se osvježi 
težina te veze.

[opcionalno rnn 3.30 - 3.35 (str. 20)]

### Dvosmjerna rekurzivna neuronska mreža

Budući da je u mnogim primjena osim konteksta koji prethodi danom
trenutku korisno uzeti u obzir i ono što slijedi nakon njega,
uvedene su i dvosmjerne rekurzivne neuronske mreže.

To je nadogradnja rekurzivne neuronske mreže gdje jedna polovina rekurzivnog
skrivenog sloja analizira ulazni niz u pozitivnom smjeru, a druga u negativnom,
kao u primjeru na slici za dvosmjernu rekurzivnu mrežu sa jednim skrivenim slojem [brnn.png].

Kako bi se izbjegli ciklusi u neuronskoj mreži ta dva dijela sloja nisu 
direktno međusobno povezani, već njihov izlaz služi kao ulaz višim slojevima.

Rad mreže je u osnovi isti kao kod obične rekurzivne neuronske mreže,
no potrebno je malo prilagoditi algoritam za izračunavanje izlaza mreže
i prolaz unatrag, algoritam [1.1] i algoritam [1.2].

za t = 1 do T 
	Prolaz unaprijed za skriveni sloj koji računa unaprijed, za svaki korak se spremaju
	izlazi 
za t = T do 1 
	Prolaz unaprijed za skriveni sloj koji računa unazad, za svaki korak se spremaju
	izlazi 
za sve t, bilo kojim redoslijedom
	Prolaz unaprijed za izlazni sloj, koristeći spremljene izlaze iz oba skrivena sloja
Algorithm 3.1: BRNN prolaz unaprijed


za sve t, bilo kojim redoslijedom
	Prolaz unazad za izlazni sloj, spremajući δ članove za svaki korak
za t = T do 1 
	BPTT prolaz unazad za skriveni sloj koji računa unazad, koristeći δ članove
	iz izlaznog sloja
za t = 1 do T 
	BPTT prolaz unazad za skriveni sloj koji računa unaprijed, koristeći δ članove
	iz izlaznog sloja
Algorithm 3.2: BRNN prolaz unazad [graves_blstm str.21.]

### Dugotrajna-kratkotrajna memorija

Rekurzivne neuronske mreže imaju boljku da kod treniranja pate od "eksplodirajućeg"
ili "iščezavajućeg" gradijenta, tj. greška pri prolazu unatrag kroz mrežu ili
naglo raste sa svakim korakom ili se naglo smanjuje.
Problem eksplodirajućeg gradijenta može dovesti do nestabilnosti mreže,
pa je jedan način da se smanji stopa učenja, što kao i iščezavajući gradijent
vodi sporijem učenju mreže. Posljedica toga je da rekurzivne neuronske mreže
teško pamte kontekst duže od nekoliko desetaka koraka.

Nadogradnja na rekurzivne mreže koja rješava te probleme je dugotrajna-kratkotrajna
memorija (eng. LSTM) [lstm]. Na slici [lstm.png] je prikazana arhitektura LSTM
ćelije.

Slika [lstm.png] LSTM memorijski blok sa jednom ćelijom. Troja vrata koja su prikazana
su nelinearne sume koje skupljaju pobude izvan i unutar bloka, i kontroliraju aktivnost
ćelije preko množenja (mali crni krugovi). Ulazna i izlazna vrata množe ulaz i izlaz ćelije,
dok vrata za brisanje množe ćelijino prethodno stanje. Unutar ćelije nema aktivacijske
funkcije. Aktivacijska funkcija vrata 'f' je obično sigmoidna funkcija, tako da joj 
je izlaz između 0 (vrata zatvorena) i 1 (vrata otvorena).
Ulazna i izlazna aktivacijska funkcija ćelije ('g' i 'h') su obično tangens hiperbolni
i sigmoidna funkcija, iako 'h' nekada može biti i funkcija identiteta.
Veze od memorijske ćelije prema vratima (eng. peephole connections) su prikazane
isprekidanim strelicama, i one za razliku od ostalih veza unutar bloka imaju težinu. [graves_blstm]
Blok ima četiri ulaza i samo jedan izlaz. Tako svaki LSTM blok ima sedam parametara,
tri unutarnje veze sa težinom, te još četiri pomaka (eng. bias) za svaki od ulaza.
Izlaz svakog od N neurona na koji je ovaj blok spojen spaja se na sva
četiri ulaza, tako da je broj ulaznih težina 4 * N.


[Jednadžbe - prilagodit ?  4.1 - 4.16 ? (str. 37-38)]

### Arhitektura sustava

U ovom radu je korištena dvosmjerna LSTM mreža (eng. BLSTM), koja je zapravo
obična dvosmjerna rekurzivna mreža samo su neuroni zamijenjeni sa LSTM blokovima.

Na slici [arhitektura.png] je nacrtana arhitektura mreže koja je korištena.
Arhitektura je preuzeta iz drugog rada [wen_chime13] jer zbog ograničenih
računalnih resursa nije bilo vremena da se empirijski odredi optimalna
arhitektura.

Svaki BLSTM blok je povezan sa svim blokovima u slojevima ispod i iznad.
BLSTM blok se sastoji od dva nepovezana LSTM bloka.
Jedan je povezan sa izlazom iz prošlog koraka od svih LSTM blokova koji računaju unaprijed u tom sloju.
Drugi je povezan sa izlazom iz idućeg koraka od svih LSTM blokova koji računaju unazad u tom sloju.
Tako je ukupni broj parametara za ovu mrežu 582339.

Mreža ima 39 neurona u ulaznom sloju jer toliko parametara ima standardni 
MFCC_E_D_A_Z vektor značajki koji se koristi u osnovnom prepoznavaču govora
koji je referentan na CHiME natjecanju [chime_data].

Slika [sustav.png] prikazuje shemu sustava. Ulazni stereo zvučni zapis se
usrednjavanjem oba kanala prebacuje u mono zapis. Zatim se na temelju
tog zapisa izračunavaju MFCC značajke na način opisan u poglavlju [broj poglavlja].
Dobivene značajke se normiraju sa vrijednostima izračunatim na cijelom 
skupu podataka za treniranje, tako da je srednja vrijednost svakog koeficijenta
0 i standardna devijacija 1. Na taj način se ne gubi nikakva informacija, 
ali bi se mreža trebala brže trenirati [wen_chime13].
Zatim se izračunava izlaz mreže za cijeli zapis tj. niz značajki.
Izlazne značajke iz mreže su također približno normirane tako da bi ih se za korištenje
u uobičajenim sustavima za prepoznavanje govora treba pomnožiti sa standardnom
devijacijom i dodati srednju vrijednost izračunatu na training setu.
CHiME osnovni sustav za prepoznavanje govora baziran na HTK paketu koristi normalizirane značajke,
no one su po svemu sudeći nekako drugačije normalizirane. Stoga se izlazne MFCC značajke
moraju normalizirati tako da im statistička svojstva odgovaraju značajkama na kojima je treniran
model koji se koristi u sustavu za prepoznavanje, jer će u suprotnom doći do pada točnosti prepoznavanja.

###Metoda treniranja neuronske mreže

Treniranje i izvršavanje neuronske mreže obavljeno je korištenjem programskog
paketa CURRENNT, jedinog koji podržava treniranje BLSTM mreža pomoću grafičkih
procesora i time omogućuje ubrzavanje treniranja i do 20 puta u nekim scenarijima [wen_currennt_cite].

Kako bi se ostvarilo ubrzanje u treniranju CURRENNT obavlja treniranje na više 
ulaznih sekvenci paralelno i tako izračunava gradijent greške na cijelom
podskupu ulaznih podataka. Zatim se nakon svake mini-serije osvježavaju težine,
dakle riječ je o stohastičkom hibridnom online-batch treniranju.
Kod dubokih neuronskih mreža ključna je dobra početna inicjalizacija, pa 
CURRENNT podržava podešavanje parametara distribucija za slučajnu inicijalizaciju. [wen_currennt_README]

Kod dubokih neuronskih mreža također je veliki problem i pretreniranje (eng. overfitting).

CURRENNT može koristiti sve tri uobičajene metode [graves_blstm: 26][wen_currennt_README]
da bi smanjio problem pretreniranja:
uranjeno zaustavljanje (eng. early stopping),
zašumljavanje ulaza (eng. input noise),
zašumljavanje težina (eng. weight noise), i u ovom radu su i korištene.

Zašumljvanje ulaza i težina se provodi tako da se jednostavno pri treniranju
svakom ulazu ili težini pribroji mala slučajna vrijednost da bi se poboljšala
sposobnost generalizacije kod mreže. Kod testiranja se te vrijednosti ne dodaju.

Za uranjeno zaustavljanje je osim uobičajenog skupa za treniranje i skupa
za testiranje potrebno imati i skup za validaciju.
Kod uranjenog zaustavljanja se mreža trenira na skupu za treniranje i 
računa se greška na skupu za treniranje i skupu za validaciju.
U jednom trenutku će greška na skupu za validaciju prestati padati iako će
greška na skupu za treniranje i dalje padati. Obično se treniranje nastavi još
nekoliko epoha da bi se osiguralo da je to stvarno minimum, ali se kao najbolja
mreža odabire ona koja daje najbolji rezultat na skupu za validaciju.
U tu svrhu bi se mogao koristiti i skup za testiranje, no to bi bio oblik 
indirektnog treniranja na skupu za testiranje. To ne bi imalo smisla jer 
nam skup za testiranje služi kako bismo dobili procjenu kako će se mreža 
ponašati ako na ulaz dobije još neviđene podatke [test_val].

CHiME skup podataka je već podijeljen na skup za treniranje sa 17000 zapisa (500 za svakog od 34 govornika),
skup za testiranje sa 3600 zapisa i skup sa validaciju sa 3600 zapisa (600 za svaku od 6 SNR vrijednosti) [chime_data].
Skup za testiranje se ne može koristiti za treniranje ili validaciju jer snimke sa čistim signalom nisu dostupne.

--------- Metodologija ------


---------- Primjena -----------

###Priprema podataka

Prije treniranja mreže potrebno je pripremiti podatke, što je u mnogim 
primjenama strojnog učenja, a tako i ovdje, velik dio posla.

Za generiranje značajki korišten je paket otvorenog koda [wen_opensmile_cite] tvrtke
audEERING UG (haftungsbeschränkt), koji podržava generiranje HTK-kompatibilnih
značajki. No budući da openSMILE ne podržava njihovo normiranje na način koji je potreban,
generirane su MFCC_E_D_A značajke, a normalizacija je provedena naknadno.

CURRENNT podatcima pristupa isključivo preko NetCDF znanstvenog formata sa
razmjenu podataka, što znači da je sve podatke potrebno prebaciti u taj format [wen_currennt_README]. 
Za to je korišten program 'htk2nc' koji je dio programskog paketa CURRENNT [wen_currennt_tools_README].

Normalizacija se obavlja nakon što se skupovi za treniranje, testiranje i validaciju
obrade i pospreme i zasebne NetCDF datoteke.
U sklopu programskog paketa CURRENNT dostupan je 'nc-standardize' alat
koji izračunava srednje vrijednosti i standardne devijacije za ulazne i izlazne podatke u NetCDF datoteci,
sprema ih u istu datoteku, te normalizira nizove s tim vrijednostima.
Normalizacija je provedena tako da su skup podataka za validaciju i testiranje
normalizirani sa srednjim vrijednostima i standardnim devijacijama skupa podataka
za treniranje.
  
Skup podataka za treniranje i validaciju se sastoje se od ulaznih nizova značajki dobivenih
od zašumljenog signala i očekivanih nizova značajki koji odgovaraju signalu
koji je izobličen sa jekom. Skup podataka za testiranje sadrži samo zašumljene signale.
Jeka u ovom slučaju ne utječe značajno na točnost prepoznavanja,
a pokusno treniranje je pokazalo da ova neuronska mreža ima problema sa konvergiranjem
ako joj se dade zadatak da nauči i poništavanje utjecaja jeke.

Sve Python skripte koje su razvijene za pripremanje podataka su javno dostupne [github_nc_packer].

###Treniranje mreže

Korišteno je računalo sa procesorom AMD Athlon II X3 450, sa 3 jezgre i radnim taktom od 3.2GHz, te 8 GB radne memorije.
Kao grafička kartica korištena je kineska kopija "Nvidia GeForce GT 630" ili sličnog modela kartice
sa 1 GB grafičke radne memorije i 96 procesnih jedinica.
Iako je kartica nelegitimnog porijekla, podržava naredbe CUDA 2.1 arhitekture, što
znači da bez problema može izvršavati sve algoritme za treniranje mreže.
Za treniranje na grafičkim procesorima CURRENNT treba biblioteku CUDA verzije 5 ili više,
a korištena je verzija 6.5 [wen_currennt_README].
Korišten je operativni sustav Lubuntu Linux 14.04, na kojem su lako dostupni i besplatni
svi potrebni paketi za pripremanje izvršnih verzija CURRENNT (verzija 0.2-rc1) i OpenSMILE (verzija 2.1) paketa
iz izvornog koda.

Jedna epoha na ovom računalu i u tom programskom okruženju trajala je oko 4850 sekundi,
tj. oko 1 sat i 20 minuta. Za treniranje finalne mreže trebalo je 211 epoha, tj. oko 12 dana.
No ukupno vrijeme treniranja, s neuspješnim pokušajima je bilo oko 24 dana.
Zbog dugog vremena treniranja mreže i kratkog trajanja semestra nije bilo dovoljno vremena
da se eksperimentira s arhitekturom mreže i parametrima treniranja, već su uzete
već provjerene vrijednosti iz literature [wen_chime13].

Arhitektura mreže je već opisana u poglavlju [Arhitektura sustava],
stopa učenja učenja iznosi [ni = 10e-5], a moment [m = 0.9].
Težinama i ulazima se dodaju slučajne vrijednosti iz distribucije sa
srednjom vrijednosti [mi=0] i standardnom devijacijom [sigma=0.1].
Veličina mini-serije (eng. mini-batch) koja se paralelno obrađuje je
100 ulaznih nizova.
Treniranje se zaustavlja kada nakon 30 epoha više ne dođe do smanjenja greške
na skupu za validaciju.
CURRENNT je također bio konfiguriran da sprema težine mreže nakon svake epohe,
što omogućava i naknadno proučavanje svojstava mreže.

Obično se kod korištenja stohastičke inicijalizacije mreža nekoliko puta 
trenira ispočetka, pa se odabire mreža koja postigne najmanju grešku na validacijskom
skupu podataka, no ni to nije napravljeno zbog vremenskih ograničenja.

Na slici [training_colors.png] prikazana je krivulja učenja.
Prikazane greške treniranja i validacije su kvadratne sredine greške između izlaznog
vektora značajki i očekivanog vektora značajki. 
Formula za kvadratnu sredinu greške je:

[RMSE]

Treniranje mreže sa stopom učenja [ni = 10e-5] daje najmanju grešku na validacijskom skupu za epohu 182.
Eksperiment je pokazao da smanjivanje stope učenja na [ni = 10e-6] i nastavljanje treniranja na mreži
iz epohe 180 dodatno smanjuje grešku na skupu za treniranje i skupu za validaciju.

[komentar na sliku: najbolje vrijednosti za sve tri krivulje su posebno označene]

Na slici [training_colors.png] prikazane je i točnost prepoznavanja (na obrnutoj skali)
na validacijskom skupu
podataka korištenjem modela treniranog na govoru izobličenom jekom [chime_data].
Tijekom provjeravanja uspješnosti rada mreže uočeno je da smanjivanje greške
između izlaznih i očekivanih značajki ne odgovara sasvim smanjivanju pogreške
prepoznavanja. 

Stoga je finalni kriterij za odabir najbolje mreže točnost prepoznavanja
na validacijskom skupu.

###Rezultati

Rezultati sa odabranu mrežu (epoha 197) prikazani su u Tablici [tab1].
Za usporedbu su dani i rezultati prepoznavanja na nepročišćenom govoru.
"Odjek" (eng. reverberated) model je uvježban na govoru iz skupa za treniranje
koji je izobličen samo bukom. "Buka" (eng. noisy) model je uvježban na govoru
sa bukom, koji se koristi kao ulaz mreže kod treniranja, tako da je optimalno
prilagođen takvom govoru. Točnost prepoznavanja čistog govora izobličenog jekom
na podacima sa validaciju iznosi 93.8%, a na podacima za testiranje je vjerojatno
1-2% veča, no to nije moguće provjeriti. Taj podatak je dan kao procjena gornje
granice točnosti koju može postići teoretski idealni sustav za pročišćavanje govora.
[rezultati tab1]


U tablici [tab2] dani su rezultati iz rada u kojem je korištena ista strategija
za pročišćavanje govora [wen_chime13] i dobiveni rezultati su samo apsolutno 1 do 1.5 % lošiji.
[rezultati tab2] 

U tablici [tab3] je prikaz prosječnog trajanje epohe za treniranje mreže na
običnom i na grafičkom procesoru.
[rezultati tab3] 

U tablici [tab4] prikazani su rezultati mjerenja brzine obrade 18 minuta zvučnih
zapisa pomoću dobivene mreže i CURRENNT paketa. 
[rezultati tab4]

---------- Primjena -----------

# Diskusija

Dobiveni rezultati su očekivani. Kod neprilagođenog sustav za prepoznavanje
dobiveno je apsolutno poboljšanje točnosti prepoznavanja od 26.5% dodavanjem
predkoraka sa pročišćavanjem govora.
Najveće poboljšanje je naravno dobiveno u slučaju kada je signal govora za 6dB
tiši od smetnje, gdje je postignuto poboljšanje od apsolutnih 39.1%.
Ovaj rezultat nam govori kakav utjecaj bi dodavanje ovakvog pročišćavanje
govora trebalo imati na performanse sustava za automatsko prepoznavanje govora
bez prilagođavanja njegovog modela.

Za slučaj kada je prilagođavanje modela moguće, imamo druge rezultate.
U usporedbi sa prepoznavačem koji je već uvježban na govoru sa smetnjom,
prosječno poboljšanje korištenjem pročišćavanja govora i prilagodbom modela
je apsolutnih 14.3% bolje, a u najgorim uvjetima prepoznavanja čak 24% bolje.

Rezultati su 1 do 1.5% lošiji od onih koji su dani u literaturi [wen_chime13],
vjerojatno zato jer je vjerojatno bilo potrebno nekoliko puta trenirati mrežu kako bi se
dobro inicijalizirala i konvergirala bliže globalnom minimumu.
Pristup većim računalnim resursima omogućio bi više eksperimentiranja u istom
vremenu i sasvim sigurno i jednake, a možda i bolje, rezultate.
	
Na slici [usporedba.png] dana je ilustracija izlaza iz mreže i pročišćavanja
značajki za jedan slučaj signala govora 6dB tišeg od smetnje.
Dane su ilustracije kako jeka i dodavanje buke izobličavaju obični, spektralni
i MFCC prikaz signala. Također je vidljivo kako mreža uspijeva proizvesti
nešto što je slično MFCC značajkama govora sa jekom, iako je prikaz dosta
glađi od originala. To po mišljenju autora može značiti ili da model dobro
generalizira ili da bi složeniji model mogao dati još bolje performanse.

[usporedba.png]

Ideja koja stoji iza ovog diplomskog rada bila je istražiti postoji li metoda
koja bi omogućila da se iskoristi govor iz brojnih radio i televizijskih 
emisija na hrvatskom jeziku za razvoj prepoznavanja govora na našem jeziku.

U emitiranim emisijama često govor prati muzika, a povremeno slabiji šumovi
(ako je govor snimljen van studija) i govor više osoba koje govore istovremeno.
No, zato je snimljeni govor uglavnom jako dobre kvalitete, glasniji je od
svih pozadinskih smetnji i utjecaj jeke je puno manji jer je mikrofon u 
pravilu blizu govornika. Pročišćavanje takvog govora trebalo bi biti lakši
problem od pročišćavanja govora u CHiME skupu podataka, gdje je jeka prisutna
u svim snimkama i odnos snage govora i smetnje mnogo nepovoljniji.

Sa druge strane, u literaturi se iznosi činjenica da dobar rezultat u pročišćavanju
i prepoznavanju govora sa malim rječnikom ne mora nužno biti prenosiv na prepoznavanje
govora sa srednjim i velikim rječnicima.
CHiME natjecanje u svojoj drugoj verziji uključuje i prepoznavanje govora
sa srednjim riječnikom (Wall Street Journal korpus čitanog govora sa rječnikom od 5000 riječi)[chime_data],
no ti podaci nisu javno dostupni [chime_website2].

Svakako je zanimljiva i mogućnost pročišćavanja i prepoznavanja govora u stvarnom vremenu,
i stoga je izmjeren RT faktor. Zanimljivo je da iako primjena grafičkih procesora
ubrzava treniranje mreže za 2.6 puta (Tablica [tab 3]), to ne bi bio slučaj kod primjene mreže uživo.
U scenariju kada se mreža primjenjuje samo na jednom ulaznom podatku usporenje je 3 puta (Tablica [tab 4]),
vjerojatno zbog puno veće latencije prebacivanja podataka iz glavne memorije u memoriju grafičke kartice.

Zanimljiv je i podatak da OpenSMILE podržava obradu zvučnog signala uživo pomoću
rekurzivnih mreža, no trenutno ne podržava dvosmjerne rekurzivne mreže.
Vjerojatno se korištenjem LSTM-RNN može dobiti nešto lošije, ali još uvijek
dobre performanse.

Za istražiti još uvijek preostaje i utjecaj korištenja drugih značajki
na ulazu neuronske mreže. Pokazano je da primjena jednostavnijih značajki
koje imitiraju svojstva ljudske pužnice (eng. log-filterbank) 
daje još bolje rezultate jer omogućava mreži da nauči bolju parametrizaciju signala [wen_chime14].

Još jedan relativno novi doprinos polju dubokog učenja je algoritam za 
vremensku klasifikaciju pomoću neuronskih mreža (eng. Connectionist Temporal Classification - CTC)
[graves14] [graves_blstm], koji omogućava treniranje neuronskih mreža koje obavljaju cijeli proces
od pročišćavanja govora do njegove transkripcije. To bi rješilo i problem koji je bio prisutan u ovom 
radu, da funkcija greške koja se koristi za treniranje mreže ne odgovara točno željenim performansama.

Glavna i vjerojatno neizbježna boljka svih spomenutih pristupa temeljenih 
na dubokom učenju je što zahtijevaju velike računalne resurse.
Velike svjetske kompanije koje imaju pristup upravo spomenutim velikim
računalnim resursima postižu zavidne rezultate [ang_deep_speech].
Složenost i računalna moć ljudskog slušnog sustava koji je svojevrsni ideal kojemu se teži,
svakako opravdava korištenje barem dijela tih resursa, no postavlja se
pitanje kako umješnim korištenjem umjerene količine računalne moći
postići slične rezultate i ostvariti upotrebljivo prepoznavanje govora
na hrvatskom jeziku.

# Zaključak - max 350 riječi

U ovom radu dan je uvod u i definicija problema računalnog izdvajanja govora.
Dana je usporedba osjetljivosti ljudskog i računalnog prepoznavanja govora
na točnost prepoznavanja kao motivacija za računalno izdvajanje govora.
Nakon kratkog pregleda povijesnih pristupa izdvajanju govora, među
najuspješnijim novijim pristupima odabran je pristup korištenjem dubokih neuronskih
mreža. BLSTM-RNN tip neuronske mreže je pokazana kao trenutno najbolji izbor za
arhitekturu mreže u ovoj primjeni.
Od pet javno dostupnih paketa za rad sa rekurzivnim mrežama odabran je CURRENNT,
koji podržava odabrani tip mreže i ubrzavanje treniranja korištenjem grafičkih procesora.

Za ispitivanje uspješnosti izdvajanja govora odabrani su javno dostupni podaci
i alati drugog CHiME natjecanja (eng. CHiME 2nd challenge Task 1).

Mreža je istrenirana i izmjerena je uspješnost prepoznavanja.
Najvažniji rezultat je koliko poboljšanje se dobiva u prepoznavanju govora 
sa smetnjama korištenjem sustava koji je specijaliziran za prepoznavanje čistog govora.
U tom scenariju je dobiveno apsolutno poboljšanje točnosti prepoznavanja od 26% za sve
odnose govora i smetnje i čak 39.1% za nepovoljniji odnos(6dB u korist smetnje).

Dana je analiza koliki je značaj tog rezultata za povećanje otpornosti na smetnje
kod sustava za prepoznavanja govora sa velikim rječnikom
i koje su mogući pravci daljnjeg istraživanja.

Demonstrirana je učinkovitost dubokih neuronskih mreža u izdvajanju govornih signala iz zvučnih zapisa.
Isto tako je pokazano da je za stvaranje praktičnog sustava za izdvajanje govora iz snimki snimljenih u
realnim uvjetima potrebno još ulagati.

# Sažetak - max. 100 riječi

U ovom radu dan je pregled trenutnog stanja područja izdvajanja govora
i pregled najuspješnijih strategija s posebnim naglaskom na dubokim neuronskim mrežama.
Odabrana je BLSTM-RNN arhitektura i CURRENNT kao programski paket za rad
s tom mrežom. Uspješnost izdvajanja govora je ispitana na skupu podataka prvog zadatka
drugog CHiME natjecanja (eng. CHiME 2nd challenge Task 1) i postignuto je apsolutno poboljšanje
od 25% u točnosti prepoznavanja govora sa smetnjama koristeći sustav za prepoznavanje
čistog govora.

ključne riječi: izdvajanje govora, duboke neuronske mreže, RNN, BLSTM, CUDA

# Summary - max. 100 riječi

A overview of recent developments in the speech extraction field is given,
including the survey of most succesfull strategies with a special emphassis
on deep neural networks. The BLSTM-RNN architecture is chosen along with the
CURRENNT software package for working with the network.
Successfullness of speech extraction is evaluated on Task 1 of the CHiME 2nd challenge, 
and an absolute improvement in word accuracy of 25% is achieved on a noisy speech
recognition task using an ASR system specialised for clean speech.

keywords: speech extraction, deep neural networks, RNN, BLSTM, CUDA

# Literatura

[ubacit iz literatura.md]
