# Theory

## Tips Luca:
- % Churn?
- Come hai gestito il fatto il dataset fosse fortemente sbilanciato?
- Insight generali?
- Modello scelto? xgboost
- metrica? precision recall?
- Come sono state scelte le feature? Che features hai generato?
- Che tool? librerie hai usato?
- Come si capisce la feature con più potere predittivo?


churn definizione difficile, in eon scindi il contrattto e hai churnato, in sisal non è così banale perché l'account resta semplicemente non logghi.

diminuzione frequenza di gioco rispetto alla media normale (def di sisal)

se uno gioca 1 volta a settimana, se per 3 settimane non gioca è un indizio di abbandono. ci sono soglie di tolleranza (almeno 15 giorni di assenza)

Si prende lo storico cliente, si fa uno snapshot
numero giocate, frequenza ricarica, spacchettato per prodotto, ammontare medio giocate (alto valore - basso valore), data ultima giocata. variabili rfm (recently frequency monetary) spacchettato per prodotto, anagrafica (regione, età, ecc), variabili sulle campagne (bonus), ticket (in eon si usava il crm, se un cliente aveva una bolletta sbagliata).

Sullo sport sono almeno 30 giorni, 14 giorni sul bingo.

prendo feature ad una certa data, predico se sarà churn a 28 giorni.
Se lo fai con contratti (eon) ti chiedi se tra 28 giorni sarà scisso.

tu non vuoi sapere se ci saranno 28 giorni di inattività, tu guardi se supera la media di inattività (se gioca 1 volta al mese devi avere 3 mesi di inattività), quindi vuol dire che in cosa predici c'è una parte di passato e questo causa un data leakage.

Si potrebbe fare una regola matematica che calcoli la frequenza di gioco e determini quali giocatori non possono andare in churn a 28 giorni (ma ad esempio a 35). Sugli altri ti basi su frequenze poissoniane e tiri fuori la probabilità.

Il problema è questa regola complicata sulla variabile target, senza la regola si poteva usare ml meglio.

xgboost ha parametri che ribilanciano i pesi delle classi automaticamente.
Non genera dati sintetici, fa solo ribilanciamento dei pesi e funziona quando la % è grande ma non enorme.

20% churna

resampling funziona ma ti limiti tantissimo il dataset. in più questo funziona per classificazione binaria, qui noi volevamo invece dare una percentuale di probabilità di churn. resampling nel nostro caso avrebbe deformato le probabilità
generare dati sintetici invece non funziona perché (nelle immagini ci sono simmetrie e constrain che sono utili da generare sinteticamente, se ruoti foto cane è ancora cane), qui invece se ruoto feature (x diventa y e viceversa) sui dati di un cliente non ha senso fisico.
xgboost classifier, ha una logistica al fondo (foglia finale) e quindi da lì tiri fuori la probabilità. predict_proba è il metodo che si usa.

ultimi 3 anni di cliente, ogni mese ho la foto del cliente con variabili rfm.
essendo dati nel passato (training) vado a mettere la label 1 0, alleno modello.

crm è stato imposto, ma meno del 2% dei clienti apre un ticket quindi è inutile.

radar chart per ogni cliente 

confusion matrix funziona sul classificatore vero e proprio --> devi ottimizzare soglia di probabilità.

l'area sotto la curva della precision recall non si può trattare come quella della roc perché dipende dallo sbilanciamento.
Diverse precision recall di modelli diversi modelli si possono paragonare solo a parità di dataset, qui 0.2 che è sbilanciamento classi è male.

in eon 0.01% churna, ma chi è perso è perso, qua invece a volte ritornano (soprattutto per fine campionato, è una regola un poì stupida)

fare data engineer ti aiuta a capire cosa è successo e che ha causato la sporcizia sul dataset, challenge per dati prezzo carburante e sono dati sporchi, per ogni giorno anno c'è un file con prezzo e data comunicazione prezzo.
ci sono dti di 3 anni prima dell'estrazione e questo capita perché ogni giorno il sistema prende il dato più recente e ci sono dei rami che non sono stati aggiornati.

in eon hanno fatto forecast della customer base (cresceva - scendeva) + mix acquisitivo (da dove proveniva il cliente).
Con il mix acquisitivo che stiamo pianificando come varia la customer base? come churneranno?
cluster basati su aging cliente + canale acquisitivo, il cliente tramite internet che probabilità aveva di passare da mese 1 a churn? il cliente web ha più probabilità churn dopo 1 anno.
sono sorte di regole di markov. usare un modello di serie storica non li avrebbe aiutati a capire il motivo. quindi anche evitare approccio ml a volte funziona, devi fare approccio modellistico descrivendo equazione alla base. ml si usa quando non puoi creare il modello preciso

## Churn in Sisal
Design an in-house churnk risk model based on transactional data and crm data.

Constrain
- Non si voleva un modello ricorsivo perché questo rischia di far "scoppiare gli errori".
- Vista a 360 gradi sul clienti, sia sulle attività online che su quelle offline (registrate tramite la carta loyalty)
- Customizzato sulla base delle principali attività del cliente
- Aggiornare lo score del churn giornalmente
- Il churn si calcola solo sui clienti active o riattivati

as is: 

Regole del CLC:
- If customer doesn’t take 1 action in the last year ‘Dorman/No Deposito’. This customers will be excluded by the model. 
- If inactivity days >90 then ‘Churn’ 
- If first activity in 104 days was take before than 14 days ago then ‘New/Reactivated’ 
- If customer takes less than 3 action days in 14 day (based on the last 104 days) then ‘Churn’ 
- If Churn Factor >3 and Inactivity days >=30 and main product in (‘Multi’, ‘Sport’, ‘Poker’, ‘Virtual Sport Fantasy Mister’) then ‘Churn’ 
- If Churn Factor >3 and Inactivity days >=21 and main product in (‘Skill’, ‘Casino & Quick’, ‘Lottery LOL Scratch’) then ‘Churn’ 
- If Churn Factor >3 and Inactivity days >=14 and main product is ‘Bingo’ then ‘Churn’ 
- Otherwise ‘Active’

Per il calcolo del churn ogni giocatore ha un suo churn factor che indica quanto di frequente gioca (per evidenziare il calo di frequenza nelle giocate):
$$
churn\ factor = \frac{(\#action - days\ in\ 1\ yeat\ from\ last\ action -1)*inactivity\ days}{days\ between\ first\ and\ last\ action\ in\ 1\ year\ from\ last\ action}
$$
![[rule.PNG]]

per capire il prodotto più giocato si è normalizzato l'amount speso nella rolling window e si è vista cche percentuale è stata spesa su ogni gioco. Se nessun gioco superava il 75% allora il giocatore era considerato un "multi".

proposed:

Every day a query on aep fills the dataset with a snapshot, the model fetch the most recent snapshot and computes the churn score.

variabili:
- Bingo, scommesse sul bingo negli ultimi 90 giorni
- casinoquick, scommesse sul casino negli ultimi 90 giorni
- poker, scommesse sul poker negli ultimi 90 giorni (per ogni gioco...)
- churn factor
- clc, attuale stato del cliente
- attività più vecchia negli ultimi 104 giorni
- prodotto principale del cliente
- numero di giorni di attività nell'ultimo anno
- numero di ticket aperti al supporto nell'ultimo anno (poco utile perché solo il 20% apre un ticket, ma era una feature chiesta dal business) (si contano il numero di ticket per tutte le categorie di ticket che ci sono)
- numero di giorni passati dal primo giorno di attività dell'ultimo anno e l'ultimo giorno di attività

metriche:
- auc roc
- auc pr
- recall first decile, si prendono i churner del decile con la probabilità di churn più alto e si divide per il numero di churner totali
- precision first decile, churner del decile con p di churn più alto e si divide per tutti i clienti appartnenti al decile
- accyracy first decile, poi si fa anche per secondo, terzo e quarto decile
## Understanding Churn
- https://medium.com/adobetech/gaining-a-deeper-understanding-of-churn-using-data-science-workspace-18a2190e0cf3

**L'articolo non si riferisce a Sisal**

Highly competitive markets (i.e. gaming, hotel, casino chains operate in) needs to understand customers, their behavior (CLC) and the features which lead to churn.
The goal was to understand which customer were highly unlikely to make a booking over a six month period, then sending personalized offers to those customers.
![[aep_theory.jpg]]

Over a 18 month window 95% of customers re-book within six months, therefore with only 6 months we could understand one complete purchase cycle for customers

**Our data included half a million customers and with a churn rate of 53%.**

- Useful features were:
	- Amount of plays
	- Days since last bet (compared with median bet cycle for that customer)
- Insight:
	- A customer who made frequent bookings in a one-month period was more likely to churn out. Consistent engagement of the customer over longer durations can help prevent churn. (It is not a good pattern if someone starts to play massively in a short period, consistency is the key)

The model was trained over one cycle, 6 months, using this as a benchmark it was studied if the customer would return to make a booking in the next 6 months.
Also, the data across time periods was studied to make sure there was no major seasonality.

- Evaluation parameters:
	- Recall of the model: # of churners the model is able to identify
	- Precision of the model: # of churners over the total predicted churns
	- Stability of the metrics over time