#!/bin/bash

#Commentaire

################one way of using the for


#for i in {1..5}
#do
#	echo "Step number $i "

#done

###################Other way to code with the for

#for (( i=1 ; i<=5 ; i++ ))
#do
#	echo "Step number $i "
#done


#################### other way of using for in a code

# why this code
# due to      cat $file | wc -l   , it will return the number of ligne in each file
#   la cmd a executer se trouve dans $(.......)
for file in /home/friedly/Documents/study/scan_file/*.pdf
do
	echo "$file : $(cat $file | wc -l ) ligne(s) "
done


############################### other way to use the boucle for

#com
#com

# Boucle sur tous les fichiers PDF du dossier
for file in /home/friedly/Documents/study/scan_file/*.pdf
do
    # On extrait uniquement le nom du fichier (ex: cours.pdf au lieu du chemin complet)
    filename=$(basename "$file")

    # On renomme en gardant le fichier dans son dossier d'origine
    mv "$file" "/home/friedly/Documents/study/scan_file/scan_$filename"
done


