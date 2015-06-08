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

Opis problema

U svim primjenama koje se bave govorom, prisutnost smetnji je neizbježna.
Bilo da se radi o snimanju zvuka, telekomunikacijama ili ljudsko-računalnim
sučeljima (engl. human–machine interfaces), mikrofon koji snima govor uglavnom
će snimiti i smetnje. Zbog toga se snimljeni govorni signal prije svoje upotrebe
mora pročistiti. Taj proces pročišćavanja govora obično se naziva ili suzbijanje
buke (engl. noise reduction) ili 
The existence of noise is inevitable. In all ap-
plications that are related to voice and speech,
from sound recording, telecommunications, and
telecollaborations, to human–machine inter-
faces, the signal of interest that is picked up
by a microphone is generally contaminated by
noise. As a result, the microphone signal has
to be cleaned up with digital signal-processing
tools before it is stored, analyzed, transmit-
ted, or played out. The cleaning process, which
is often referred to as either noise reduction
or speech enhancement, has attracted a con-
siderable amount of research and engineering
attention for several decades. Remarkable ad-
vances have already been made, and this area
is continuing to progress, with the aim of cre-
ating processors that can extract the desired
speech signal as if there is no noise. This chap-
ter presents a methodical overview of the state
of the art of noise-reduction algorithms. Based
on their theoretical origin, the algorithms are
categorized into three fundamental classes:
filtering techniques, spectral restoration, and
model-based methods. We outline the ba-
sic ideas underlying these approaches, discuss
their characteristics, explain their intrinsic re-
lationships, and review their advantages and
disadvantages [book_springer]



Izdvajanje tj. pročišćavanje govora je ključan korak u velikom broju praktičnih
primjena računalnog prepoznavanja govora. U mnogim primjenama je nemoguće
Također, većini sustava za računalno prepoznavanje govora performanse naglo
padaju ako se ulaznom govornom signalu dodaje smetnja. [ referenca? ]

As we shall see, human recognition performance drops to chance levels at a
wideband SNR somewhere between −20 and −25 dB and saturates at a minimum
error rate when the SNR is between −6 and 0 dB. The actual numbers depend
on the complexity (i.e., the entropy) of the recognition task. Machine performance
starts degrading by +20 dB SNR and approaches chance level near 0 dB SNR.
This large discrepancy in performance is called the robustness problem, which refers
to the sensitivity of the score to the SNR. What is needed is some insight into
why this large difference between human performance and present day machine
performance exists. I believe we can answer this question, and this is one of the
goals of this monograph [book_articulation]





An important area of application is the automatic recognition of sounds by computers. Many of the methods that
are currently in use, for example, in speech recognition, deteriorate badly when extraneous sounds are present.
Incorporating a primitive scene-analysis stage into the recognition process might allow the systems to resist being
derailed by these sounds. Some beginnings have been made by using a few features of a voice, such as its
fundamental or its spatial location, to track it through a mixture, but no computational system has so far attempted
to implement the full range of heuristics described in the earlier chapters. [book_asa]

There are three major approaches to speech signal separation: (1) an approach based on microphone-
array technology [48], (2) an approach based on blind signal separation [2, 9, 26], and (3) an approach
based on model-based separation [33, 42, 64]. The first approach, i.e., acoustic-technology-based, is
the most traditional and orthodox, and is already used in real-world recognizers. However, although it
works sufficiently under some limited conditions, it cannot emulate the high flexibility and capability
of humans in separating a target speech signal with a reasonably small-size implementation. The
second approach is based on the information theory, and it presently represents a main research trend
in the field of signal separation. The third approach attempts to emulate the human hearing process
to some extent by assuming the importance of model-based (top-down or a priori) knowledge about
target signals. Compared to the second approach, blind signal separation, this model-based approach
is still at the beginning stages of research. Nonetheless, the validity and prospects of this approach
are supported by many psychological findings [5].
This chapter focuses on two recent trends, i.e., blind separation and model-based separation. From
the standpoint of speech recognition, the techniques of these two separation approaches have not yet
been fully developed. They also do not necessarily have a direct link to ANN technologies. Only a
few links can be observed in the structural similarities between the ANN and the problem formalism
of signal separation. However, studying these technical topics is clearly worthwhile because math-
ematics and ultimate research goals, i.e., the development of speech recognition technologies that
successfully emulate humans, are common to both the ANN and signal separation.

[book_nn_sp]



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
