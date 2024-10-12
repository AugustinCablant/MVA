# Self-attention classique

L'approche classique d'attention, souvent utilisée dans des modèles comme le **seq2seq** avec LSTM ou GRU, calcule un vecteur de contexte qui pondère les différentes parties d'une séquence en fonction de leur importance relative à une tâche donnée: 
- Un seul vecteur d'attention est appris pour pondérer les différents mots dans une séquence.
- Les poids d'attention sont calculés via une fonction softmax, normalisant les scores d'importance pour chaque mot.
- Les poids d'attention sont souvent calculés en fonction d'une requête (comme un état caché dans un LSTM), ce qui signifie que l'attention est dirigée en fonction du contexte de cette requête
- La séquence est résumée en un seul vecteur de contexte, qui est une somme pondérée des vecteurs de mots

Limites : 
- Puisqu'un seul vecteur d'attention est utilisé, l’attention globale capturée est limitée. Si plusieurs informations pertinentes sont dispersées dans la séquence, elles risquent de ne pas toutes être représentées correctement.
- L'attention classique peut se concentrer sur un petit nombre de tokens importants, négligeant d'autres informations utiles dans la séquence.

# Self-attention auto-structurée (Yang et al.)
L'approche introduite par Lin et al. propose des modifications clés qui améliorent la capture de l'information contextuelle.

- Au lieu d'un seul vecteur d'attention, plusieurs vecteurs d'attention sont appris, ce qui permet au modèle de capturer différentes perspectives et nuances à travers la séquence. Chaque tête d'attention génère un ensemble différent de poids pour la séquence, permettant de se concentrer sur plusieurs parties importantes de la phrase simultanément.

- Pour éviter que les différents vecteurs d'attention apprennent des informations redondantes (c'est-à-dire, qu'ils se concentrent tous sur les mêmes mots), une régularisation est introduite pour maximiser la diversité entre les vecteurs d'attention. Cela permet d'obtenir des informations plus variées de la séquence.

- Plutôt que de produire un seul vecteur résumant la séquence, l'approche agrège plusieurs vecteurs attentionnés, chacun représentant une sous-partie différente de la séquence. Cela capture davantage de nuances dans les relations entre les mots.

- Contrairement à l'attention classique qui nécessite une requête (comme l'état caché dans les seq2seq), cette approche ne dépend pas d’une requête externe. L'attention est calculée uniquement en fonction des représentations internes de la séquence.

