# autoresearcher2 v1 Validation Checklist

Doel van dit document: objectief vaststellen of v1 **werkt op het niveau dat v1 belooft**.

v1 claimt nog **niet**:
- beter te zijn dan echte `autoresearch` op een GPT-trainingspipeline
- beter te zijn dan GP-UCB / ASHA
- echte cross-campaign transfer te hebben bewezen

v1 claimt **wel**:
- dat de architectuur coherent werkt
- dat structured Bayesian experiment selection op synthetic taken iets zinnigs leert
- dat appraisal belief-shifting resultaten kan markeren
- dat het systeem beter kan presteren dan simpele baselines zoals random en greedy

---

## 1. Technische validatie

### 1.1 Test suite
- [ ] `uv run pytest -v --tb=short` geeft volledig groen
- [ ] `uv run pytest --cov=autoresearcher2 --cov-report=term-missing` geeft acceptabele coverage
- [ ] Alle tests zijn reproduceerbaar met vaste seeds
- [ ] Geen flaky tests over meerdere runs

### 1.2 Architectuurcontracten
- [ ] `Appraisal` muteert het model niet
- [ ] `ResearchLoop` doet echt: select -> run -> snapshot_before -> update -> snapshot_after -> appraise -> store
- [ ] `Controller` gebruikt seed-injectable RNG
- [ ] `Environment` gebruikt seed-injectable RNG
- [ ] `Baseline` agents gebruiken seed-injectable RNG
- [ ] `Toy` agent / POMDP zijn reproduceerbaar

### 1.3 Basale invarianten
- [ ] Voor alle cells geldt: `config_to_cell(cell_to_config(i)) == i`
- [ ] Posterior variantie daalt na observaties
- [ ] Predictive mean verschuift richting observatie
- [ ] `score_all_cells()` bevat altijd `pragmatic`, `epistemic`, `total`
- [ ] Toy EFE decompositie voldoet exact aan `total = pragmatic + epistemic`

---

## 2. Synthetic validatie

Gebruik minimaal **3 synthetic environments** met bekende ground truth.

### Verplichte synthetic omgevingen

#### Env A — main effects dominant
Voorbeeld:
- optimizer: adam = +0.3, sgd = -0.3
- lr: low = -0.1, high = +0.1
- lage noise

- [ ] Gedefinieerd
- [ ] Seeded
- [ ] Ground truth opgeslagen in evaluatienotities

#### Env B — interactie aanwezig
Voorbeeld:
- zelfde main effects
- extra interactie: `(adam, high_lr) = +0.2`

- [ ] Gedefinieerd
- [ ] Seeded
- [ ] Ground truth opgeslagen

#### Env C — hoge noise / lastigere wereld
Voorbeeld:
- zelfde structuur
- hogere `noise_std`

- [ ] Gedefinieerd
- [ ] Seeded
- [ ] Ground truth opgeslagen

Optioneel later:
- [ ] misleidend lokaal optimum
- [ ] sparse effects
- [ ] schema waarin 1 factor nauwelijks uitmaakt

---

## 3. Experimenteel protocol

Voor elke synthetic environment:

- [ ] Run `autoresearcher2`
- [ ] Run `random`
- [ ] Run `greedy`
- [ ] Gebruik identiek experimentbudget per agent
- [ ] Gebruik meerdere seeds per environment

### Minimum
- [ ] `n_seeds >= 20`
- [ ] `n_experiments >= 50` per run
  (mag lager voor mini-schema's, maar leg dat expliciet vast)

### Vast te loggen per run
- [ ] outcomes per experiment
- [ ] best-so-far curve
- [ ] cumulative regret
- [ ] epistemic score / variance over tijd
- [ ] factor importances aan het eind
- [ ] top appraisal events
- [ ] aantal unieke cells bezocht
- [ ] seed / env-config / schema-config

---

## 4. Validatie van het model

### 4.1 Leert het model echte structuur?
Per environment:

- [ ] Factor met grootste ground-truth effect krijgt meestal hoogste importance
- [ ] Zwakkere factor krijgt lagere importance
- [ ] Bij voldoende data daalt epistemische onzekerheid in bezochte regio's
- [ ] Predicties op verwante / unseen cells verbeteren t.o.v. prior

### 4.2 Minimale acceptatiecriteria
Voor Env A:

- [ ] In >= 80% van de seeds geldt: `importance(optimizer) > importance(lr)`

Voor Env B:

- [ ] Model presteert beter dan een puur tabulaire benadering of random sampling op unseen combinaties
  *(mag handmatig of via extra benchmarkscript)*

Voor Env C:

- [ ] Model blijft stabiel en divergeert niet numeriek
- [ ] Onzekerheid daalt langzamer dan in Env A, zoals verwacht

---

## 5. Validatie van de controller

### 5.1 Exploreert vroeg, exploiteert later?
- [ ] In vroege fase bezoekt controller meerdere cells / regio's
- [ ] In latere fase concentreert controller zich vaker op goede regio's
- [ ] Epistemische score/variance neemt gemiddeld af over tijd
- [ ] Best-so-far neemt gemiddeld toe over tijd

### 5.2 Minimale acceptatiecriteria
Per environment:

- [ ] `autoresearcher2` verslaat `random` op gemiddelde cumulative regret
- [ ] `autoresearcher2` is niet slechter dan `greedy` op eenvoudige main-effects omgeving
- [ ] `autoresearcher2` is beter dan `greedy` op minstens een omgeving waar generalisatie helpt
  *(bijv. interactie-omgeving of setting met beperkte data)*

### 5.3 Over meerdere seeds
- [ ] Prestatieverschil is robuust over seeds
- [ ] Geen afhankelijkheid van een lucky run

---

## 6. Validatie van appraisal

Doel: aantonen dat appraisal niet zomaar "mooie signalen" produceert, maar belief-changing events markeert.

### 6.1 Te controleren eigenschappen
- [ ] `surprise` is hoger voor onverwachte uitkomsten dan voor verwachte
- [ ] `theory_conflict` is hoger wanneer het model **confident wrong** was
- [ ] `prediction_impact_breadth` stijgt wanneer een update meer voorspellingen verandert
- [ ] `learntropy` piekt vooral bij interpreteerbare belief shifts, niet bij triviale bevestigingen

### 6.2 Handmatige inspectie
Neem per environment de top-10 events op `learntropy`.

Voor elk event handmatig beoordelen:
- [ ] Was dit echt onverwacht?
- [ ] Was het model vooraf relatief zeker?
- [ ] Veranderde hierdoor het model merkbaar?
- [ ] Is dit een "wow" / belief-restructuring moment?
- [ ] Of is het eigenlijk alleen noise?

### 6.3 Minimale acceptatiecriteria
- [ ] Top appraisal events zijn overwegend belief-shifting events
- [ ] Lage appraisal events zijn overwegend bevestigend / saai
- [ ] Appraisal correleert zichtbaar met posterior change, niet alleen met ruwe outcome-grootte

---

## 7. Validatie van memory

Voor v1 is memory nog eenvoudig. Valideer alleen wat v1 echt doet.

- [ ] Alle experimenten worden opgeslagen
- [ ] `has_tried()` werkt correct
- [ ] Retrieval per cell werkt correct
- [ ] `top_by_appraisal()` geeft daadwerkelijk hoog-appraisal events terug
- [ ] `summary()` klopt met de rungeschiedenis

### Praktisch
- [ ] Controleer dat appraisal-rijke events makkelijker terug te halen zijn dan triviale events
- [ ] Controleer dat dedup / tried-cell tracking nuttig is in analyse

---

## 8. Toy validation (theoretische referentie)

Dit valideert de **canonieke referentie**, niet direct de praktische controller.

### Te valideren
- [ ] Toy POMDP initialiseert correct
- [ ] ActiveInferenceAgent kiest geldige acties
- [ ] Beliefs veranderen na observaties
- [ ] Gemiddelde epistemische component daalt over tijd
- [ ] EFE decompositie klopt exact

### Interpretatie
- [ ] Resultaten van de toy environment worden niet overdreven als bewijs voor de praktische v1
- [ ] Toy-resultaten worden expliciet gelabeld als theoretische referentie

---

## 9. Rapportageformat

Na validatie moet er minimaal een evaluatiedocument komen:

`docs/eval/v1-synthetic-evaluation.md`

Met daarin:

### 9.1 Setup
- [ ] Welke schema's gebruikt zijn
- [ ] Welke synthetic environments gebruikt zijn
- [ ] Welke seeds gebruikt zijn
- [ ] Welk experimentbudget gebruikt is

### 9.2 Resultaten
- [ ] Tabel met mean +/- std van best outcome
- [ ] Tabel met mean +/- std van cumulative regret
- [ ] Plots van best-so-far curves
- [ ] Plots van epistemische afname
- [ ] Samenvatting van factor importances per environment
- [ ] Top appraisal examples met korte duiding

### 9.3 Eerlijke conclusie
- [ ] Wat werkt aantoonbaar?
- [ ] Wat werkt nog niet?
- [ ] Welke claims zijn nu gerechtvaardigd?
- [ ] Welke claims zijn nog te vroeg?

---

## 10. Exit criteria: wanneer zeggen we "v1 werkt"?

v1 geldt als **geslaagd** als aan al het volgende is voldaan:

### Technisch
- [ ] test suite volledig groen
- [ ] geen architectuurcontracten geschonden
- [ ] reproduceerbaar over seeds

### Model
- [ ] leert op synthetic taken de dominante factorstructuur terug

### Controller
- [ ] verslaat random robuust
- [ ] is minstens competitief met greedy op eenvoudige taken
- [ ] laat zinnig explore->exploit gedrag zien

### Appraisal
- [ ] markeert belief-changing events beter dan triviale confirmaties

### Rapportage
- [ ] resultaten zijn gedocumenteerd in `docs/eval/v1-synthetic-evaluation.md`

Als een van deze blokken faalt, dan is v1 **nog niet geslaagd**.

---

## 11. Niet concluderen op basis van v1

Als v1 slaagt, mag je **nog niet** concluderen:

- [ ] dat het beter is dan echte `autoresearch`
- [ ] dat het beter is dan GP-UCB / ASHA
- [ ] dat transfer over campagnes werkt
- [ ] dat active inference "bewezen" is voor de applied setting
- [ ] dat learntropy al een directe control law is

Dat zijn expliciet **post-v1** vragen.

---

## 12. Volgende stap na succesvolle v1

Pas starten als alle exit criteria gehaald zijn.

- [ ] Koppelen aan echte proxy workload (`train.py` -> `val_bpb`)
- [ ] Head-to-head met autoresearch-stijl baseline
- [ ] GP-UCB / ASHA toevoegen
- [ ] Evalueren op echte compute-efficiency
- [ ] Later pas: transfer, rijkere memory dynamics, regime variable
