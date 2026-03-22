# refraction_market_trend_detection

# Refraction Market

## À propos

Ce projet est une expérimentation conceptuelle : l’idée était de regarder l’information de marché autrement, en essayant de la **compresser sous une forme géométrique et optique** plutôt que de la traiter uniquement comme une suite de prix ou de rendements.

L’intuition de départ était simple : s’inspirer des **lois de Snell** et d’un cadre proche de l’optique pour représenter le marché comme une succession de **milieux** traversés par une trajectoire.  
Dans cette lecture, le régime n’est plus défini seulement par des variations de prix, mais par des **angles**, des **segments** et des **paramètres du milieu** (volatilité locale, viscosité, indice effectif, etc.).

## Idée du projet

Au lieu de poser directement la question :

> “Quel sera le prochain prix ?”

ce projet essaie plutôt de poser une autre question :

> “Dans quel type de milieu ou de segment le marché est-il en train d’évoluer, et quand change-t-il de régime ?”

L’objectif n’était donc pas forcément de faire un modèle prédictif classique, mais plutôt de construire une autre **représentation du marché**, plus structurée, plus compressée, et potentiellement réutilisable dans un cadre plus large.

## Ce que ce projet cherche à faire

- transformer la dynamique de marché en une lecture par **angles** et **transitions**
- représenter les phases de marché comme des **segments causaux**
- décrire un régime à partir de **paramètres de milieu**
- fournir une base exploitable plus tard dans un autre moteur ou un autre modèle

## Ce que ce projet n’est pas

Ce dépôt n’a pas été pensé comme une vérité finale ni comme un système de trading “prouvé”.  
C’était avant tout une **idée exploratoire**, une façon différente de regarder le problème.

Qu’elle fonctionne parfaitement ou non n’est pas le point principal ici.  
L’intérêt du projet est surtout dans le **cadre conceptuel** qu’il propose, et dans le fait qu’il pourra servir plus tard de brique, d’inspiration ou de sous-module dans une approche plus robuste.

## Statut

Projet expérimental / prototype conceptuel.

Il s’agit surtout d’un terrain de recherche personnel autour de :

- la segmentation de marché
- la détection de transition de régime
- la représentation géométrique des séries financières
- la réutilisation future de ces variables dans d’autres modèles

## Suite possible

Plus tard, cette idée pourra être :

- réutilisée telle quelle dans un autre pipeline
- simplifiée en ne gardant que certaines variables utiles
- enrichie par d’autres méthodes de segmentation ou d’inférence
- intégrée comme couche de features dans un modèle plus classique

## Note

En résumé, ce projet est surtout une tentative de **voir le marché autrement** :  
non pas seulement comme une courbe de prix, mais comme une propagation dans des milieux successifs, définis par des angles, des structures locales et des transitions de régime.