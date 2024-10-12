# Self-attention classique

L'approche classique d'attention, souvent utilisée dans des modèles comme le **seq2seq** avec LSTM ou GRU, calcule un vecteur de contexte qui pondère les différentes parties d'une séquence en fonction de leur importance relative à une tâche donnée: 
- Un seul vecteur d'attention est appris pour pondérer les différents mots dans une séquence.
- Les poids d'attention sont calculés via une fonction softmax, normalisant les scores d'importance pour chaque mot.
- Les poids d'attention sont souvent calculés en fonction d'une requête (comme un état caché dans un LSTM), ce qui signifie que l'attention est dirigée en fonction du contexte de cette requête
- La séquence est résumée en un seul vecteur de contexte, qui est une somme pondérée des vecteurs de mots

Limites : 
- Puisqu'un seul vecteur d'attention est utilisé, l’attention globale capturée est limitée. Si plusieurs informations pertinentes sont dispersées dans la séquence, elles risquent de ne pas toutes être représentées correctement.
- L'attention classique peut se concentrer sur un petit nombre de tokens importants, négligeant d'autres informations utiles dans la séquence.

# Self-attention auto-structurée (Yang et al., A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING)
L'approche introduite par Lin et al. propose des modifications clés qui améliorent la capture de l'information contextuelle.

- Au lieu d'un seul vecteur d'attention, plusieurs vecteurs d'attention sont appris, ce qui permet au modèle de capturer différentes perspectives et nuances à travers la séquence. Chaque tête d'attention génère un ensemble différent de poids pour la séquence, permettant de se concentrer sur plusieurs parties importantes de la phrase simultanément.

- Pour éviter que les différents vecteurs d'attention apprennent des informations redondantes (c'est-à-dire, qu'ils se concentrent tous sur les mêmes mots), une régularisation est introduite pour maximiser la diversité entre les vecteurs d'attention. Cela permet d'obtenir des informations plus variées de la séquence.

- Plutôt que de produire un seul vecteur résumant la séquence, l'approche agrège plusieurs vecteurs attentionnés, chacun représentant une sous-partie différente de la séquence. Cela capture davantage de nuances dans les relations entre les mots.

- Contrairement à l'attention classique qui nécessite une requête (comme l'état caché dans les seq2seq), cette approche ne dépend pas d’une requête externe. L'attention est calculée uniquement en fonction des représentations internes de la séquence.

# Transformer (Attention is All You Need)
Les auteurs proposent de remplacer les réseaux récurrents (RNN) ou convolutifs (CNN) par un mécanisme d'attention uniquement, supprimant ainsi la nécessité de traiter les séquences de manière ordonnée.

- Le cœur du Transformer repose sur l'attention "auto-dirigée" (self-attention), qui permet à chaque mot d'une séquence d'analyser sa relation avec tous les autres mots, indépendamment de leur distance dans la séquence. Ce mécanisme calcule des poids d'attention qui mesurent l'importance relative entre chaque paire de mots.

- Au lieu d'utiliser un seul vecteur d'attention, l'article propose le concept de "multi-head attention", où plusieurs ensembles de poids d'attention sont calculés simultanément. Cela permet au modèle de capturer différents types de relations entre les mots, améliorant la richesse de l'information capturée.

- Comme le Transformer ne traite pas les séquences de manière récurrente, il est nécessaire de fournir des informations sur l'ordre des mots. Pour cela, des encodages positionnels sont ajoutés aux représentations des mots afin d'intégrer leur position dans la séquence.

- Le modèle Transformer est composé de blocs empilés comprenant des mécanismes d'attention multi-tête et des couches de feed-forward. Ces blocs sont utilisés à la fois dans l'encodeur (qui traite la séquence d'entrée) et le décodeur (qui génère la séquence de sortie).

- Contrairement aux RNN et LSTM, le Transformer traite toutes les positions de la séquence en parallèle, ce qui permet d'accélérer les calculs en tirant parti des GPU et d'augmenter l'efficacité de l'apprentissage.

Le **remplacement des opérations récurrentes par le mécanisme de self-attention** dans les modèles de traitement du langage naturel (NLP) présente plusieurs avantages significatifs :

- Les réseaux de neurones récurrents (RNN) traitent les données séquentiellement, ce qui rend difficile la parallélisation. Chaque étape dépend de la précédente, ce qui limite la capacité à exploiter pleinement les architectures modernes basées sur les GPU. En revanche, le mécanisme de self-attention permet de traiter toutes les positions d'une séquence simultanément, augmentant ainsi l'efficacité de l'entraînement et réduisant le temps de calcul.

- Les RNN ont souvent du mal à capturer les dépendances à long terme dans les séquences en raison du problème de l'atténuation des gradients. Le mécanisme de self-attention permet de relier directement des mots qui peuvent être éloignés dans la séquence, en fournissant des poids d'attention qui mesurent l'importance relative entre tous les mots, quelle que soit leur distance.

- Dans un RNN, les poids des connexions sont partagés entre toutes les étapes de temps, ce qui peut limiter la capacité du modèle à apprendre des relations spécifiques entre les mots. Avec la self-attention, des poids d'attention uniques peuvent être calculés pour chaque paire de mots, ce qui permet de capturer des interactions plus complexes.

- Les modèles basés sur la self-attention, comme les Transformers, ont une architecture plus simple et plus modulaire. Cela facilite l'ajout de mécanismes supplémentaires (comme la normalisation ou la régularisation) et l'expérimentation avec différentes configurations.

- Le mécanisme d'attention fournit une façon d'interpréter les décisions du modèle. En visualisant les poids d'attention, il est possible de comprendre quelles parties de la séquence influencent le plus les prédictions, offrant ainsi une meilleure transparence par rapport aux décisions du modèle.

- Les résultats expérimentaux montrent que les modèles basés sur la self-attention surpassent souvent les architectures RNN traditionnelles sur des tâches de traduction et d'autres tâches de traitement du langage, notamment en termes de qualité des résultats et de rapidité d'entraînement.

# HAN : Hierarchical Attention Network

*Ce modèle a été introduit par Yang et al. dans leur article intitulé "Hierarchical Attention Networks for Document Classification" en 2016. L'idée clé derrière HAN est de capturer la structure hiérarchique naturelle des documents, en traitant d'abord les mots au sein des phrases, puis les phrases au sein des documents, en utilisant des mécanismes d'attention à chaque niveau.*

**Architecture du HAN**

a. Niveau des Mots (Word-Level)

- Chaque mot du texte est converti en un vecteur dense (embedding) à l'aide d'une couche d'embedding (par exemple, Word2Vec, GloVe).
Cela transforme les mots en représentations numériques permettant au réseau de traiter les informations textuelles.

- Un BiGRU (Gated Recurrent Unit bidirectionnel) est utilisé pour capturer les dépendances contextuelles des mots au sein de chaque phrase.
Le BiGRU lit la séquence de mots dans les deux directions (de gauche à droite et de droite à gauche), permettant de capturer des informations contextuelles complètes.

- Une couche d'attention est appliquée sur les sorties du BiGRU. Ce mécanisme d'attention apprend à pondérer l'importance de chaque mot dans la phrase, en attribuant des coefficients d'attention qui reflètent la pertinence des mots pour la tâche spécifique (par exemple, classification).

- La sortie de l'attention au niveau des mots est une représentation vectorielle de chaque phrase, pondérée par l'importance des mots.

b. Niveau des Phrases (Sentence-Level)

- Chaque représentation de phrase obtenue au niveau des mots est passée à travers un TimeDistributed (ou traitement séquentiel) qui applique le même encodeur (AttentionBiGRU) à chaque phrase du document. Cela permet de traiter chaque phrase indépendamment tout en maintenant la structure séquentielle du document.

- Un autre BiGRU est utilisé pour capturer les dépendances contextuelles entre les phrases du document. Cela permet de comprendre comment les phrases interagissent et contribuent au sens global du document. Une deuxième couche d'attention est appliquée sur les sorties du BiGRU des phrases. Ce mécanisme attribue des poids aux phrases, identifiant celles qui sont les plus pertinentes pour la tâche (par exemple, identifier les phrases clés pour la classification).

- La sortie de l'attention au niveau des phrases est une représentation vectorielle du document entier, pondérée par l'importance des phrases.

c. Couche Finale de Prédiction

- La représentation globale du document est passée à travers une couche linéaire suivie d'une fonction d'activation (par exemple, Sigmoid pour la classification binaire ou Softmax pour la classification multi-classes). Cela génère les prédictions finales du modèle.

**Avantages du HAN** 
 
- En traitant séparément les niveaux des mots et des phrases, HAN reflète mieux la structure naturelle des documents.

- L'attention permet au modèle de se concentrer sur les parties les plus pertinentes du texte, améliorant ainsi les performances en se basant sur l'importance contextuelle.

- Les coefficients d'attention fournissent une certaine interprétabilité, montrant quelles phrases et quels mots ont été jugés importants par le modèle pour prendre une décision.