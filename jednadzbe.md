
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

### Arhitektura mreže

U ovom radu je korištena dvosmjerna LSTM mreža (eng. BLSTM), koja je zapravo
obična dvosmjerna rekurzivna mreža samo su neuroni zamijenjeni sa LSTM blokovima.

Na slici [arhitektura.png] 

Jednadžbe - prilagodit ? 
4.1 - 4.16 ? (str. 37-38)
