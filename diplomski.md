title: Neuralne mreže u izdvajanju govornog signala iz zvučnog zapisa
author: Stjepan Henc

##Neuralne mreže u izdvajanju govornog signala iz zvučnog zapisa

# Kratice

engl. - engleski
GPU - grafički procesor opće namjene (engl. Graphic Processor Unit)
SNR - omjer korisnog signala i signala smetnje (engl. Signal to Noise Ratio)
CHiME - natjecanje u računalnom prepoznavanju govora u okruženjima gdje je prisutno više
izvora signala(engl. Computational Hearing in Multisource Environments)
CHiME2 - druga iteracija CHiME natjecanja (engl. CHiME 2nd challenge)
PCA - analiza principalnih komponenata (engl. PCA - Principal Component Analysis)
ICA - analiza neovisnih komponenata (engl. PCA - Principal Component Analysis)
NMF - nenegativna faktorizacija matrica (engl. Non-negative Matrix Factorization)
DNN - duboka neuronska mreža (engl. Deep Neural Network)
RNN - rekurzivna neuronska mreža (engl. Recursive Neural Network)
BRNN - dvosmjerna rekurzivna neuronska mreža (engl. Bidirectional Recursive Neural Network)
LSTM - dugotrajno-kratkotrajna memorija (engl. Long-Short Term Memory)
BLSTM - dvosmjerna dugotrajno-kratkotrajna memorija (engl. Bidirectional Long-Short Term Memory)
BPTT - nadogradnja backpropagation algoritma za treniranje RNN (engl. Backpropagation Through Time)
CTC - klasifikacija vremenskih nizova pomoću neuronskih mreža (engl. Connectionist Temporal Classification)

# Uvod

U praksi je jako često da zvučni zapisi govora koje treba pretočiti u 
tekstualne zapise (engl. speech-to-text) sadrže i razne smetnje 
koje su prisutne jer govor nije bio sniman u idealnim uvjetima.
Smetnje mogu biti razni šumovi, buka, muzika ili istovremeni govor i žamor.
Sve takve smetnje uzrokuju jako veliki pad točnosti računalnog prepoznavnja
govora [ book_articulation:2].
Zadatak ovog diplomskog rada je pronaći odgovarajuću metodu i programski
paket koji bi omogućio izdvajanje najčišćeg mogućeg govora iz takvih zvučnih zapisa
s ciljem povećanja točnosti prepoznavanja.
Među mnogim opcijama kao metoda je odabrana RNN-BLSTM neuronska mreža,
a od programskih paketa CURRENNT, koji podržava ubrzavanje treniranja pomoću
GPU-a.
Efikasnost ovog pristupa je ispitana pomoću skupa podataka CHiME2
i pripadnih alata.

[stavit brojeve]
U poglavlju "Pregled literature" iznesen je pregled literature i stanja
istraživanja na ovom području, s posebnim naglaskom na ono što je korišteno u ovom radu.
Opisan je CHiME2 skup podataka kao dobra platforma za usporedbu rješenja za izdvajanje govornog signala.

U poglavlju "Metodologija" opisan je rad RNN-BLSTM algoritma, odabrana arhitektura mreže
i metoda treniranja mreže koje podržava programski paket CURRENNT.

U poglavlju "Primjena" opisana je priprema podataka za treniranje, postupak treniranja mreže
i dobiveni rezultati na CHiME2 skupu podataka.

U poglavlju "Diskusija" opisano je što bi ti rezultati mogli značiti za računalno prepoznavanje govora u praksi,
te mogući daljnji pravci istraživanja.


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
To je područje koje se intenzivno proučava već nekoliko desetljeća,
no problem još uvijek nije riješen [book_springer:843-845].

Posebno važna primjena izdvajanja govora je računalno prepoznavanje govora.
U usporedbi s računalnim sustavima za prepoznavanje govora, ljudsko uho radi nevjerojatno dobro.
Kod ljudskih bića razumijevanje govora, ovisno o primjeni, počinje padati
kada je SNR od -6 do 0 dB, a tek se od -25 do -20 dB sasvim gubi razumljivost.
Računalno prepoznavanje govora počinje gubiti na točnosti već oko +20 dB,
a na 0 dB (jednaka snaga signala govora i smetnje) se već približava nasumičnom
pogađanju [book_articulation:2].

Činjenica da su ljudi tako sposobni u obavljanju tog zadatka potaknula je mnoge
istraživače da u proučavanju ljudskog slušnog sustava pokušaju pronaći inspiraciju
za nova tehnološka rješenja [book_asa][book_casa][book_human_machine].

Postoje tri glavna pristupa izdvajanju govornog signala:

1. Tehnologija nizova mikrofona (engl. Microphone-Array Technology)
2. Slijepo razdvajanje signala (engl. Blind Signal Separation)
3. Razdvajanje temeljeno na modelu govora

Prvi od tih pristupa, temeljen na akustičnoj tehnologiji, je tradicionalan
i već u prisutan u praksi. No, iako radi dovoljno dobro u određenim uvjetima,
manji sustavi su, u usporedbi sa čovjekom, nefleksibilni i loši.
Drugi pristup se temelji na teoriji informacija, i trenutno je glavni trend
u polju istraživanja razdvajanja signala.
Treći pristup pokušava donekle emulirati ljudski slušni sustav koristeći
apriorno znanje, tj. model ciljnih signala, kao najvažniji faktor.
U usporedbi sa slijepim razdvajanjem signala ovaj pristup je još u povojima,
no njegovu opravdanost potvrđuju mnoga otrkića na području psihologije
[book_nn_sp: 184].

CHiME

### Motivacija
U zadnjih nekoliko godina veoma je značajno CHiME natjecanje u razdvajanju
i prepoznavanju govora (engl. CHiME Speech Separation and Recognition Challenge)
kao platforma gdje različite istraživačke skupine iz akademskog svijeta i industrije
mogu usporediti svoja rješenja na skupu podataka koji je dobar pokazatelj koliko
dobro bi ta rješenja mogla raditi na stvarnim podacima [chime_data][chime_cite2].

Iako je u tijeku već treća iteracija CHiME natjecanja, u ovom radu će se
koristiti podaci za drugo CHiME natjecanje ili CHiME2,
budući da u ovom trenutku već postoji mnogo objavljenih rezultata za taj skup podataka,
pa je moguće usporediti dobivene rezultate s već objavljenima [chime_overview].

Cilj CHiME natjecanja je dobiti što veću točnost prepoznavanja govora
izobličenog realističnim izvorima smetnji.
Problem izdvajanja govora za automatsko prepoznavanje je različit od običnog pročišćivanja
govora, zato jer veliki broj tehnika pročišćivanja govora samo poboljšava doživljaj
kvalitete govora, ali ne povećava i njegovu razumljivost (bilo za ljudske ili računalne slušače) [book_speech_enhancement].

Računalno prepoznavanje govora je težak problem i u uvjetima savršeno čistog govornog signala.
Faktori koji taj problem mogu dodatno otežati su: 1. promjenjivost položaja i udaljenost govornika u odnosu na mikrofon, 
2. veličina rječnika i 3. prirodnost govora. 
Budući da je fokus CHiME natjecanja na izdvajanju govora, autori su odlučili napraviti
skup podataka s realnim signalima smetnje, snimljenim u pravoj dnevnoj sobi.
Tako je govoru superponirana izrazito realna smetnja, no kako bi zadatak ostao
rješiv, govor kojeg treba prepoznati je nerealno jednostavan [chime_data].


###Grid corpus 
Izvor čistog govora je Grid korpus govora, koji se sastoji
od zvučnih zapisa jednostavnih komandi.
Korištene su snimke 34 različita izvorna govornika engleskog jezika [chime_grid_cite].
Zvučni zapisi su rečenice od šest riječi u obliku 
<naredba:4><boja:4><prijedlog.:4><slovo:25><znamenka:10><prilog:4>,
pri čemu brojevi u uglatim zagradama označuju koliko opcija postoji za svaku kategoriju.
Zadatak je prepoznati slovo i znamenku pa se točnost prepoznavanja mjeri 
samo na te dvije riječi.

Dakle, govor kojeg treba prepoznati se sastoji od malog rječnika i jednostavne gramatike,
nije prirodan i govornik je uglavnom na istom položaju, što prepoznavanje
čistog govora čini laganim. Točnost za čisti govor je 97.25% [chime_data], što je usporedivio s 
ljudskom točnošću koja je u tom slučaju oko 98.3% (točnost za slova je 99.05%, a za brojke 99.3%) [chime_grid_cite].

###Smetnje

Kako bi se dobio željeni raspon SNR,
izgovorene rečenice su pozicionirane u odnosu na pozadinsku buku tako da se dobiju željenne
vrijednosti: -6, -3, 0, 3, 6 i 9 dB. Pozadinska buka
je nasumično pretraživana i odabrani su oni vremenski interval koji imaju željeni SNR
u odnosu na zadanu rečenicu. Na 9dB, najpovoljnijem odnosu željenog i neželjenog signala,
smetnje su uglavnom kvazi-stacionarni šumovi (npr. šum ventilatora), dok su oni na -6dB uglavnom
iznenadni nestacionarni zvučni događaji (npr. dječje vrištanje).
Kako bi zadatak bio još realističniji, napravljena je konvolucija čiste izgovorene rečenice 
s vremenski promjenjivim binauralnim impulsnim odzivima sobe (engl. Binaural Room Impulse Response)
koji simuliraju ograničeno pomicanje govornika i odjek prostorije [chime_data].

###Podaci

Sve snimke u CHiME2 skupu podataka su u 16-bitnom WAV formatu uzorkovanom na 16kHz.
Skup podataka za treniranje (engl. training set) sarži 17000 rečenica, 500 za svakog od 34 govornika.
Skup podataka za validaciju (engl. validation/development set) i skup podataka za testiranje (engl. test set)
sadrže 600 rečenica na 6 različitih SNR-a [chime_website].

###Ocjenjivanje točnosti

Osim ispitnih podataka, u sklopu CHiME2 dostupni su i alati za
mjerenje točnosti prepoznavanja govora. 
Ti alati su temeljeni na sustavima za prepoznavanje govora koji su dio
HTK programskog paketa [book_htk][chime_readme].
Iako CHiME dozvoljava korištenje vlastitog
rješenja za prepoznavanje govora, na raspolaganje je stavljen osnovni
sustav (engl. baseline recognizer) sa nekoliko unaprijed istreniranih modela.
To je učinjeno kako bi se moglo odrediti koji dio poboljšanja točnosti se može
pripisati izdvajanju govora, a koji sustavu za prepoznavanje govora.
Time se olakšava i usporedba različitih sustava [chime_readme].

Osnovni sustav za prepoznavanje je prilagođen sintaksi rečenica u Grid korpusu.
Sustav je baziran na skrivenim markovljevim modelima. Svaka od 51 riječi
prisutna u Grid korpusu modelirana je pomoću skrivenog markovljevog modela s
2 stanja po fonemu. Vjerojatnost izostavljanja svakog stanja je predstavljena pomoću
mješovitog Gaussovog modela s 7 komponenti i dijagonalnom kovarijancom.

Dana su tri unaprijed uvježbana modela za prepoznavanje:
1. "Čisti" model (engl. clean - treniran na čistom govoru)
2. "Odjek" model (engl. reverberated - treniran na govoru izobličenom jekom)
3. "Buka" model (engl. noisy - treniran na govoru izobličenom bukom)

Osim ta tri modela, dostupni su alati za lako uvježbavanje tj. prilagođavanje modela (engl. retraining)
svog vlastitog modela. To se radi tako da se skup podataka za treniranje propusti kroz algoritam
za izdvajanje govora i model za prepoznavanje se onda trenira na tom govoru. Cilj je poništiti 
utjecaj izobličenja govora koja nastaju njegovim izdvajanjem [chime_data].

Govor je parametriziran kao niz standardnih MFCC značajki.
Svaki vektor značajke sadrži 39 parametara.
Prvi parametri u vektoru su 13 mel-kepstralnih koeficijenata koji su normalizirani po srednjoj vrijednosti
(ali ne i standardnoj devijaciji). Ta metoda se zova kepstralna metoda normalizacije
srednje vrijednosti (engl. CMN - Cepstral Mean Normalisation) i smanjuje utjecaj
razlika u obliku vokalnog trakta na točnost prepoznavanja govora različitih govornika.
Umjesto nultog MFCC parametra koristi se logaritamska energija okvira.
Na tih 13 parametara dodaje se 13 differencijalnih koeficijenata prvog reda
i 13 drugog reda (engl. delta and acceleration coefficients). 
Standardna HTK šifra za te značajke je MFCC_E_D_A_Z i detaljno
je opisana u literaturi [book_htk][book_opensmile].
MFCC značajke se standardno računaju na vremenskim okvirima od 25 ms, a korak je 10 ms.
Budući da su zvučni podaci dani u stereo formatu, signal je pretvoren u
mono signal uzimanjem srednje vrijednosti oba kanala [chime_data].


Odabir strategije

Zanimljiva povijesna činjenica je da su neuronske mreže u području slijepog
razdvajanja signala prisutne od samog početka njegovog 80-ih godina prošlog stoljeća.

Prvi algoritam koji je korišten je analiza principalnih komponenata ili PCA,
gdje se parametrizirana reprezentacija signala (najčešće spektar) pokušavala
razdvojiti na komponente koje odgovaraju pojedinim izvorima signala korištenjem
određenih statističkih svojstava tih signala.
Analiza neovisnih komponenata ili kraće ICA još je jedna metoda koja se može
svrstati u neuronske mreže, a nastala je
nadogradnjom originalnog PCA algoritma [book_bss_ica:7-9] [book_nn_sp:180].

U literaturi se mogu naći stvarno brojni i nerijetko vrlo složeni pristupi ovoj problematici, 
no valja izdvojiti dva koja su se pokazala posebno uspješnima i popularnima u posljednjih
nekoliko godina.

To su nenegativna faktorizacija matrica ili NMF [book_bss_ica:515] i 
duboke neuronske mreže ili DNN.
Oba pristupa su relativno jednostavna, no NMF ima nekoliko nedostataka u
usporedbi s DNN.
NMF je isključivo linearan model, dok DNN (ovisno o konkretnoj izvedbi)
u pravilu može modelirati i nelinearno preslikavanje iz izvora signala 
u mješavinu.
Također, kod primjene istreniranog NMF modela mora se provoditi iterativni
postupak koji uključuje množenje nekoliko velikih matrica, što je jako računski zahtjevno.

S druge strane, duboke neuronske mreže se u pravilu duže treniraju, ali se zato primjena
istreniranog modela sastoji samo od jednokratnog množenja nekoliko matrica, što ih
čini pogodnima sa primjenu u stvarnom vremenu jer imaju u osnovi linearnu složenost.
Svi ti faktori čine duboke neuronske mreže moćnijim i bržim modelom (jednom kada ih se uspije istrenirati) [dnn_faster_nmf].

No, zanimljivo je da je na CHiME2 pobijedio sustav koji, među ostalima,
koristi i DNN i NMF [wen_chime_pobjednik], te su u literaturi poznate 
razne kombinacije ovih pristupa [dnn_nmf][dnn_vs_nmf_novo][deep_nmf].

Uglavnom svi visokorangirani sustavi koriste kombinaciju nekoliko složenih pristupa i,
za razliku od ovog rada, nije im cilj doći do sustava koji bi bio dovoljno brz za
primjenu u praksi, već pod cijenu brzine i jednostavnosti dobiti čim veće performanse
na skupu podataka za testiranje [chime_overview].

Duboke neuronske mreže su dio jedne šire paradigme na području umjetne inteligencije
pod nazivom duboko učenje.
Glavna ideja te paradigme je da su korištenje veće količine podataka [ang_banko_brill_scale]
i većih modela [ang_coates_model_size] glavni motori povećanja performansi.
Cilj je iskoristiti sve veću raspoloživu računalna moć i sve veća količina podataka kako bi se u svojoj osnovi stari
algoritmi iskoristili za rješavanje dosad nezamislivih problema.
Računalni resursi koji se koriste mogu biti tisuće servera u nekoj od velikih
internetskih kompanija [ang_large_dnn], ili pak grafički procesori [ang_cudnn] koji 
danas i običnim studentima čine dostupnom računalnu moć u rangu nekadašnjih 
superračunala.

No, najmoćnija je kombinacija više servera sa nekoliko grafičkih procesora,
što omogućuje treniranje neuronskih mreža sa nekoliko milijardi parametara
u roku nekoliko dana [ang_cots_hpc].

Povećani intenzitet istraživanja na ovom području doveo je i do novih
algoritama i ideja, između ostalog i obećavajućih postignuća na području
računalnog prepoznavanja govora [ang_deep_speech][graves14].

Dosad je u ovom poglavlju pojam DNN korišten kao da se radi o jednom pristupu,
no samo u području izdvajanja govora može se odnositi na mnogo različitih
tipova mreža s različitim svojstvima [dnn_turci][dnn_wang_ss_frontend][dnn_rnn_smaragdis]
[dnn_kinezi][dnn_multitalker][dnn_music].

Tip duboke neuronske mreže koji se pokazao najprikladniji za ovaj problem je
rekurzivna neuronska mreža ili RNN  s dvosmjernim dugom-kratkom memorijskim ćelijama ili BLSTM [wen_chime13][wen_chime14][wen_sdr_lstm][wen_chime1].
Taj pristup će biti daljnji fokus ovog rada.


Odabir programskog paketa

Budući da su duboke neuronske mreže u zadnje vrijeme vrlo popularno područje istraživanja,
pojavili su se mnogi programski paketi koji olakšavaju njihovu upotrebu.

Budući da je odabran BLSTM tip rekurzivne neuronske mreže, u obzir dolaze samo paketi
koji podržavaju takve slojeve. 
Odabrani paket također mora podržavati ubrzavanje izvođenja na grafičkim 
procesorima i biti općenito dovoljno učinkovit, jer bi u suprotnom treniranje
mreže moglo premašiti trajanje ljetnog semestra.

Tablica [broj N] daje usporedbu dostupnih paketa otvorenog koda i neke
njihove karakteristike [pybrain_cite][theano_cite1][theano_cite2][torch7_cite][wen_currennt_cite][rnnlib]. 

[prebacit u tablicu]
			pybrain		    torch7 	      theano          rnnlib           CURRENNT
GPU 		ne                 da             da              ne               da
BLSTM		da                 ne             ne              da               da
jezik       python             lua/c          python          c++              c++

Jedino programski paket CURRENNT zadovoljava sve zadane kriterije:
podržava BLSTM-RNN neuronske mreže i ubrzavanje izvršavanja na grafičkim procesorima.
Programski paket koristi biblioteku CUDA za rad sa GPU-om [cuda-cite] i napisan je u programskom jeziku C++,
što mu omogućava da obavi treniranje mreže zadovoljavajućom brzinom.

----- Pregled literature ---------

--------- Metodologija ------

Rekurzivne neuronske mreže

Naziv rekurzivna neuronska mreža u užem smislu odnosi se na nadogradnju višeslojnog
perceptrona [graves_blstm]. U najčešćoj varijanti RNN-a, sloju se uz uobičajenu
pobudu daju i izlazi iz tog sloja u prethodnom trenutku (pod trenutak se podrazumijeva
pozicija na vremenskoj ili prostornoj osi).
Slika [rnn.png] daje primjer jedne takve mreže.

Ovaj tip mreže najčešće se koristi kada je u problemu klasifikacije nekog niza
podataka potrebno iskoristiti kontekst, npr. prepoznavanja rukopisa.
Ova nadogradnja višeslojnog perceptrona omogućava
mreži da u svojem internom stanju pohrani informaciju o prethodnim ulazima
i na taj način pamti što je bilo na ulazu u prethodnim koracima.

Prolaz unaprijed kod rekurzivne neuronske mreže izgleda isto kao kod 
višeslojnog perceptrona, no kod prolaza unazad koristi se BPTT algoritam.

Ideja algoritma je da se mreža "razmota", tako da se mreži na ulaz odjednom
postavi cijeli ulazni niz. Izračuna se izlaz mreže za cijeli niz, i izračunaju
se greške za svaki korak. Budući da se zbog razmotavanja mreže težine ponavljaju,
za svaku težinu se zbroje sve pripadne greške i s tom vrijednošću se osvježi njezina
težina.

[opcionalno rnn 3.30 - 3.35 (str. 20)]

### Dvosmjerna rekurzivna neuronska mreža

Budući da je u mnogim primjenama osim konteksta koji prethodi danom
korako korisno uzeti u obzir i ono što slijedi nakon njega,
uvedene su i dvosmjerne rekurzivne neuronske mreže.

To je nadogradnja rekurzivne neuronske mreže gdje jedna polovina rekurzivnog
skrivenog sloja analizira ulazni niz u pozitivnom smjeru, a druga u negativnom,
kao u primjeru na slici [brnn.png], gdje je prikazana dvosmjerna rekurzivna mreža sa jednim skrivenim slojem .

Kako bi se izbjegli ciklusi u neuronskoj mreži ta dio koji računa unaprijed
i dio koji računa unazad u istom sloju ne smiju biti međusobno povezani,
već njihov izlaz služi kao ulaz višim slojevima.

Rad mreže je u osnovi isti kao kod obične rekurzivne neuronske mreže,
no potrebno je malo prilagoditi algoritam za izračunavanje izlaza mreže
i prolaz unatrag. Te izmjene su prikazane pseudokodom [1.1] i pseudokodom [1.2] [graves_blstm].

za t = 1 do T 
	Prolaz unaprijed za skriveni sloj koji računa unaprijed, za svaki korak se spremaju
	izlazi 
za t = T do 1 
	Prolaz unaprijed za skriveni sloj koji računa unazad, za svaki korak se spremaju
	izlazi 
za sve t, bilo kojim redoslijedom
	Prolaz unaprijed za izlazni sloj, koristeći spremljene izlaze iz oba skrivena sloja
Pseudokod [1.1]: BRNN prolaz unaprijed


za sve t, bilo kojim redoslijedom
	Prolaz unazad za izlazni sloj, spremajući δ članove za svaki korak
za t = T do 1 
	BPTT prolaz unazad za skriveni sloj koji računa unazad, koristeći δ članove
	iz izlaznog sloja
za t = 1 do T 
	BPTT prolaz unazad za skriveni sloj koji računa unaprijed, koristeći δ članove
	iz izlaznog sloja
Pseudokod [1.2]: BRNN prolaz unazad

### Dugotrajno-kratkotrajna memorija

Rekurzivne neuronske mreže imaju boljku da kod treniranja pate od ili "eksplodirajućeg"
ili "iščezavajućeg" gradijenta (engl. exploding and vanishing gradient),
tj. greška pri prolazu unatrag kroz mrežu ili naglo raste sa svakim korakom ili se naglo smanjuje.
Problem eksplodirajućeg gradijenta može dovesti do nestabilnosti mreže,
pa je jedan način da se tome doskoči smanjiti stopu učenja, što u skoro svim
slučajevima vodi u drugu krajnost.
Iščezavajući gradijent ima posljedicu da je treniranje mreže za pamćenje dužih
vremenskih ovisnosti jako sporo.
Posljedica toga je da rekurzivne neuronske mreže
teško pamte kontekst duže od nekoliko desetaka koraka.

Nadogradnja na rekurzivne mreže koja rješava te probleme je dugotrajno-kratkotrajna
memorija ili LSTM [lstm]. Na slici [lstm.png] je prikazana arhitektura LSTM
ćelije.

Slika [lstm.png] LSTM blok sa jednom ćelijom. Troja vrata koja su prikazana
su nelinearne sume koje skupljaju pobude izvan i unutar bloka, i kontroliraju aktivnost
ćelije preko množenja (mali crni krugovi). Ulazna i izlazna vrata množe redom ulaz i izlaz ćelije,
dok vrata za brisanje množe prethodno stanje ćelije. Sama ćelija nema aktivacijsku
funkciju već pamti nepromijenjeno ono što dobije na ulaz.
Aktivacijska funkcija vrata 'f' je obično sigmoidna funkcija, tako da joj 
je izlaz između 0 (vrata zatvorena) i 1 (vrata otvorena).
Ulazna i izlazna aktivacijska funkcija ćelije ('g' i 'h') su obično tangens hiperbolni
ili sigmoidna funkcija, iako 'h' nekada može biti i funkcija identiteta.
Veze od memorijske ćelije prema vratima (engl. peephole connections) su prikazane
isprekidanim strelicama, i one za razliku od ostalih veza unutar bloka imaju težinu [graves_blstm].
Blok ima četiri ulaza i samo jedan izlaz. Tako svaki LSTM blok ima sedam parametara : 
tri unutarnje veze sa težinama, te još četiri pomaka (engl. bias) za svaki od ulaza.
Izlaz svakog od N neurona na koji je ovaj blok spojen spaja se na sva
četiri ulaza, tako da je broj ulaznih težina 4 * N.


[Jednadžbe - prilagodit ?  4.1 - 4.16 ? (str. 37-38)]

### Arhitektura sustava

U ovom radu je korištena dvosmjerna LSTM mreža ili BLSTM mreža, koja je zapravo
obična dvosmjerna rekurzivna mreža, samo su neuroni zamijenjeni sa LSTM blokovima.

Na slici [arhitektura.png] je nacrtana arhitektura mreže koja je korištena.
Arhitektura je preuzeta iz drugog rada [wen_chime13] jer zbog ograničenih
računalnih resursa nije bilo vremena da se empirijski odredi optimalna
veličina i broj slojeva.

Svaki BLSTM blok je povezan sa svim blokovima u slojevima ispod i iznad.
BLSTM blok se sastoji od dva nepovezana LSTM bloka (zbog izbjegavanja ciklusa).
Jedan je povezan sa izlazom svih LSTM blokova koji računaju unaprijed tog sloja u prošlom koraku.
Drugi je povezan sa izlazom svih LSTM blokova koji računaju unazad tog sloja u idućem koraku.
Iz toga slijedi da je ukupni broj parametara za ovu mrežu 582339.

Mreža ima 39 neurona u ulaznom sloju jer toliko parametara ima standardni 
MFCC_E_D_A_Z vektor značajki koji se koristi u osnovnom prepoznavaču govora
koji je referentan na CHiME2 [chime_data].

Slika [sustav.png] prikazuje shemu sustava. Ulazni stereo zvučni zapis se
usrednjavanjem oba kanala prebacuje u mono zapis. Zatim se na temelju
tog zapisa izračunavaju MFCC značajke metodom opisanom u poglavlju [broj poglavlja].
Dobivene značajke se normaliziraju s vrijednostima izračunatim na cijelom 
skupu podataka za treniranje. Na taj način se ne gubi nikakva informacija, 
ali se mreža brže trenira [wen_chime13].
Zatim se izračunava izlaz mreže za cijeli zapis tj. niz značajki.
Izlazne značajke iz mreže su također približno normalizirane tako da ih se za korištenje
u uobičajenim sustavima za prepoznavanje govora treba pomnožiti sa standardnom
devijacijom i dodati srednju vrijednost izračunatu na skupu za treniranje.
CHiME2 osnovni prepoznavač koristi normalizirane značajke,
no one su normalizirane na način koji nije sasvim jednoznačno objašnjen u dokumentaciji i drugačiji
je od normalizacije korištene za treniranje mreže. Stoga se izlazne MFCC značajke
moraju normalizirati tako da im statistička svojstva odgovaraju značajkama na kojima je treniran
model za prepoznavanje. To se radi da se izbjegne pad performansi uslijed razlike
između podataka na kojima je obavljeno treniranje i onima na kojima se ispituje točnost.

###Metoda treniranja neuronske mreže

Treniranje i izvršavanje neuronske mreže obavljeno je korištenjem programskog
paketa CURRENNT, jedinog koji podržava treniranje BLSTM mreža pomoću grafičkih
procesora. Korištenje GPU-a u nekim scenarijima omogućuje ubrzavanje treniranja i do 20 puta [wen_currennt_cite].

Kako bi se ubrzalo treniranje, CURRENNT obavlja treniranje na više 
ulaznih sekvenci paralelno i tako izračunava gradijent greške na tom
podskupu (engl. mini-batch) ulaznih podataka.
Zatim se nakon izračuna greške na svakom podskupu osvježavaju težine.
Ta metoda se naziva stohastičko hibridno online-batch treniranje.
Kod dubokih neuronskih mreža ključna je dobra početna inicjalizacija, pa 
CURRENNT podržava podešavanje parametara distribucija za slučajnu inicijalizaciju. [wen_currennt_README]

Kod dubokih neuronskih mreža također je veliki problem i pretreniranje (engl. overfitting).

CURRENNT može koristiti sve tri uobičajene metode [graves_blstm: 26][wen_currennt_README]
da bi smanjio problem pretreniranja:
uranjeno zaustavljanje (engl. early stopping),
zašumljavanje ulaza (engl. input noise),
zašumljavanje težina (engl. weight noise).
U ovom radu su korištene sve tri navedene metode.

Zašumljvanje ulaza i težina provodi se tako da se pri treniranju
svakom ulazu ili težini pribroji mala slučajna vrijednost.
Ideja je da će to smanjiti osjetljivost mreže na nebitne detalje u ulaznim podacima
i poboljšati sposobnost generalizacije mreže. Testiranje se provodi bez dodavanja
tih slučajnih vrijednosti, jer bi to dovelo do pada performansi.

Za uranjeno zaustavljanje je osim uobičajenog skupa za treniranje i skupa
za testiranje potrebno imati i skup za validaciju.
Kod uranjenog zaustavljanja mreža se, naravno, trenira na skupu za treniranje.
Tijekom treniranja računa se greška na skupu za treniranje i skupu za validaciju.
Skup podataka za validaciju nam omogućava da detektiramo kada je došlo do pretreniranja.
Kada mreža počne biti pretrenirana, greška na primjerima za treniranje i dalje pada
dok na neviđenim ispitnim primjerima (skupu za validaciju) stagnira ili počinje rasti.
Obično se treniranje nastavi još
nekoliko epoha nakon toga da bi se osiguralo da je to stvarno minimum,
ali se prekida nakon unaprijed određenog broja epoha.

Kao najbolja mreža odabire ona koja daje najbolji rezultat na skupu za validaciju.

Za spriječavanje pretreniranja na taj način u ovom slučaju se ne koristi skup za testiranje,
jer je odabir najbolje mreže ovisno o greški na skupu za testiranje ekvivalentno treniranju
jednog parametra na tom skupu podataka. Skup podataka za testiranje služi da se dobije
procjena kako će se mreža ponašati ako na ulaz dobije još neviđene podatke [test_val].

CHiME skup podataka je već podijeljen na skup za treniranje sa 17000 zapisa (500 za svakog od 34 govornika),
skup za testiranje sa 3600 zapisa i skup za validaciju sa 3600 zapisa (600 za svaku od 6 SNR vrijednosti) [chime_data].
Skup za testiranje i ne bi se mogao koristiti za treniranje ili validaciju
jer nisu dostupne snimke čistog govora za taj dio podataka.

--------- Metodologija ------


---------- Primjena -----------

###Priprema podataka

Prije treniranja mreže potrebno je pripremiti podatke, što je u mnogim 
primjenama strojnog učenja, a tako i ovdje, velik dio posla.

Za generiranje značajki korišten je openSMILE paket otvorenog koda [wen_opensmile_cite] tvrtke
audEERING UG (haftungsbeschränkt), koji podržava generiranje HTK-kompatibilnih
značajki. No budući da openSMILE ne podržava njihovo normiranje na način koji je potreban,
generirane su MFCC_E_D_A značajke, a normalizacija je provedena naknadno.

CURRENNT podatcima pristupa isključivo preko NetCDF znanstvenog formata za
razmjenu podataka, što znači da je sve podatke potrebno prebaciti u taj format [wen_currennt_README]. 
Za to je korišten program 'htk2nc' koji je dio programskog paketa CURRENNT [wen_currennt_tools_README].

Normalizacija se obavlja nakon što se skupovi za treniranje, testiranje i validaciju
obrade i pospreme u zasebne NetCDF datoteke.
U sklopu programskog paketa CURRENNT dostupan je i 'nc-standardize' alat [wen_currennt_tools_README]
koji izračunava srednje vrijednosti i standardne devijacije za ulazne i izlazne podatke u NetCDF datoteci,
te ih sprema istu datoteku. Ulazne i izlazne nizove u NetCDF datoteci može normalizirati s
vrijednostima izračunatima na njima samima ili sa srednjim vrijednostima i standardnim devijacijama
drugog skupa podataka.
Normalizacija je provedena tako da su skup podataka za validaciju i testiranje
normalizirani sa srednjim vrijednostima i standardnim devijacijama skupa podataka
za treniranje.
  
Skupovi podataka za treniranje i validaciju se sastoje se od ulaznih nizova značajki dobivenih
od zašumljenog signala i očekivanih nizova značajki koji odgovaraju signalu
koji je izobličen samo simuliranim odjekom. Skup podataka za testiranje sadrži samo zašumljene signale.
Simulirani odjek prostorije u ovom slučaju ne utječe značajno na točnost prepoznavanja,
a probno treniranje je pokazalo da ova neuronska mreža ima problema sa konvergiranjem
ako joj se dade zadatak da nauči i poništavanje utjecaja jeke.

Budući da je priprema podataka bila toliko zahtjevna, u sklopu ovog rada je razvijeno nekoliko
skripti koje automatiziraju taj proces.
Razvijene Python skripte su javno dostupne [github_nc_packer].

###Treniranje mreže

Za treniranje mreže korišteno je računalo sa procesorom AMD Athlon II X3 450,
s tri jezgre i radnim taktom od 3.2GHz, te 8 GB radne memorije.
Kao grafička kartica korištena je kineska kopija "Nvidia GeForce GT 630" ili sličnog modela kartice
sa 1 GB grafičke radne memorije i 96 CUDA procesnih jedinica.
Iako je kartica nelegitimnog porijekla, podržava naredbe CUDA 2.1 arhitekture, što
znači da bez problema može izvršavati sve algoritme za treniranje mreže.
Za treniranje na grafičkim procesorima CURRENNT treba biblioteku CUDA verzije 5 ili više,
a korištena je verzija 6.5 [wen_currennt_README].
Korišten je operativni sustav Lubuntu Linux 14.04, na kojem su lako dostupni i besplatni
svi potrebni paketi za pripremanje izvršnih verzija CURRENNT (verzija 0.2-rc1) i OpenSMILE (verzija 2.1) paketa
iz izvornog koda.

Jedna epoha na ovom računalu i u tom programskom okruženju trajala je oko 4850 sekundi,
tj. oko 1 sat i 20 minuta. Za treniranje finalne mreže trebalo je 211 epoha, tj. oko 12 dana.
No ukupno vrijeme treniranja, zajedno s neuspješnim pokušajima je bilo oko 24 dana.
Zbog dugog vremena treniranja mreže i kratkog trajanja semestra nije bilo dovoljno vremena
da se eksperimentira s arhitekturom mreže i parametrima treniranja, pa su uzete
vrijednosti iz literature [wen_chime13], što garantira da će se mreža
dobro istrenirati i u prvom pokušaju.

Arhitektura mreže je već opisana u poglavlju [Arhitektura sustava],
stopa učenja učenja iznosi [ni = 10e-5], a moment [m = 0.9].
Težinama i ulazima se dodaju slučajne vrijednosti iz distribucije sa
srednjom vrijednosti [mi=0] i standardnom devijacijom [sigma=0.1].
Veličina mini-serije (engl. mini-batch) koja se paralelno obrađuje je
100 ulaznih nizova.
Treniranje se zaustavlja kada nakon 30 epoha više ne dođe do smanjenja greške
na skupu za validaciju.
CURRENNT je također bio konfiguriran da sprema težine mreže nakon svake epohe,
što omogućava naknadno proučavanje svojstava mreže.

Obično se kod korištenja stohastičke inicijalizacije mreža nekoliko puta 
trenira ispočetka, pa se odabire mreža koja postigne najmanju grešku na validacijskom
skupu podataka. To nije napravljeno zbog vremenskih ograničenja.

Na slici [training_colors.png] prikazana je krivulja učenja.
Prikazane greške treniranja i validacije su kvadratne sredine razlike između izlaznog
vektora značajki i očekivanog vektora značajki. 

Treniranje mreže sa stopom učenja [ni = 10e-5] daje najmanju grešku na validacijskom skupu za epohu 182.
Eksperiment je pokazao da smanjivanje stope učenja na [ni = 10e-6] i nastavljanje treniranja
od epohe 180 dodatno smanjuje grešku na skupu za treniranje i skupu za validaciju.

[komentar na sliku: najbolje vrijednosti za sve tri krivulje su posebno označene]

Na slici [training_colors.png] prikazana je i točnost prepoznavanja (na obrnutoj skali)
na validacijskom skupu podataka korištenjem "Odjek" osnovnog modela.
Tijekom provjeravanja uspješnosti rada mreže uočeno je da smanjivanje mjere razlike
između izlaznih i očekivanih značajki ne odgovara u svakom slučaju smanjivanju pogreške
prepoznavanja. 

Stoga je kao finalni kriterij za odabir najbolje mreže uzeta točnost prepoznavanja
na validacijskom skupu.

###Rezultati

Rezultati sa odabranu mrežu (epoha 197) prikazani su u Tablici [tab1].
Za usporedbu su dani i rezultati prepoznavanja na zašumljenom govoru.
Točnost prepoznavanja čistog govora izobličenog jekom
na podacima za validaciju iznosi 93.8%, a na podacima za testiranje je vjerojatno
1 do 2% veča, no to nije moguće provjeriti jer čisti govor nije javno dostupan.
Točnost prepoznavanja na čistom govoru je gornja
granica točnosti koju može postići teoretski idealni sustav za pročišćavanje govora.
[rezultati tab1]


U tablici [tab2] dani su rezultati iz rada u kojem je korištena ista strategija
za pročišćavanje govora [wen_chime13]. U usporedbi s tim rezultatima dobiveni rezultati su samo 1 do 1.5 % lošiji.
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
koje imitiraju svojstva ljudske pužnice (engl. log-filterbank) 
daje još bolje rezultate jer omogućava mreži da nauči bolju parametrizaciju signala [wen_chime14].

Još jedan relativno novi doprinos polju dubokog učenja je algoritam za 
vremensku klasifikaciju pomoću neuronskih mreža ili CTC algoritam
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
i alati drugog CHiME natjecanja (engl. CHiME 2nd challenge).

Mreža je istrenirana i izmjerena je uspješnost prepoznavanja.
Najvažniji rezultat je koliko poboljšanje se dobiva u prepoznavanju govora 
sa smetnjama korištenjem sustava koji je specijaliziran za prepoznavanje čistog govora.
U tom scenariju je dobiveno apsolutno poboljšanje točnosti prepoznavanja od 26% za sve
odnose govora i smetnje i čak 39.1% za nepovoljniji odnos( 6dB u korist smetnje).

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
drugog CHiME natjecanja (engl. CHiME 2nd challenge) i postignuto je apsolutno poboljšanje
od 25% u točnosti prepoznavanja govora sa smetnjama koristeći sustav za prepoznavanje
čistog govora.

ključne riječi: izdvajanje govora, duboke neuronske mreže, RNN, BLSTM, CUDA, CHiME, CURRENNT

# Summary - max. 100 riječi

A overview of recent developments in the speech extraction field is given,
including the survey of most succesfull strategies with a special emphassis
on deep neural networks. The BLSTM-RNN architecture is chosen along with the
CURRENNT software package for working with the network.
Successfullness of speech extraction is evaluated on Task 1 of the CHiME 2nd challenge, 
and an absolute improvement in word accuracy of 25% is achieved on a noisy speech
recognition task using an ASR system specialised for clean speech.

keywords: speech extraction, deep neural networks, RNN, BLSTM, CUDA, CHiME, CURRENNT

# Literatura

[ubacit iz literatura.md]
