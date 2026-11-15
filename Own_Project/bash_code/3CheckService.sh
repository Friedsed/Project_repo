#!/bin/bash

# Verify is SSh is working activate or non

ServiceSSH=$(systemctl is-active ssh) 

# Etat du service Apache2

ServiceApache2=$(systemctl is-active apache2)

#Determiner si les services sont actifs

if [ $ServiceSSH = "active" ] && [ $ServiceApache2 = "active" ]
then

	echo " Les deux services sont en cours d'execution"

else

	echo " ERREUR ! Statut SSH : $ServiceSSH - Statut Apache2 : $ServiceApache2"


fi

# need to be improve not well understande find in the second session video on youtube of 
# IT-Connect -Florian ,,,, Script Bash - Maitriser les boucles For

# last modif wednesday 11/12026 at 11.06 am
