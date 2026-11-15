#!/bin/bash

# Definir l'adresse du serveur sur lequel effectuer le ping

SERVEUR=192.168.1.254

#Envoyer 2 pings au serveur correspondant a la variable $SERVEUR
ping -c 2 $SERVEUR > /dev/null

#Determiner si le serveur est en ligne ou pas

if [ $? -ne 0 ]
then 
	echo " Erreur - Le serveur $SERVEUR n'a pas repondu au ping ! "

else

	echo " OK - Le serveur $SERVEUR est en ligne. "

fi 
