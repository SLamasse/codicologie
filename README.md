# Codicologie


Il s'agit d'un ensemble de scripts que nous maintenons pour le projet **Robocodico**. 

Dans cette phase nous avons testé **DHSegment** et **Kraken** pour en extraire les informations nécessaires pour un article portant sur les réseaux de facture de manuscrits à partir de l'ensemble des manuscrits de la BNF. 

Sur cette phase du projet les scripts ont été produits par Stéphane Lamassé et Pierre Lebec.



### Structure générale 
```mermaid
|-- img/ 
|
|             |-- model/
|-- dhSegment |-- polylines/
|             |-- log/ 
|
|-- resultats/
```


#### Processus

> Recopie les images haute résolution ou pdf de la BnF à partir d'une liste de ark
> correction de l'orientation et des images (*crop.py*)
> on passe au traitement avec dhSegment ou Kraken 



--> recopie les images haute résiolution  ou pdf de la
