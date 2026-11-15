#!/bin/bash

#Commentaire

################

echo " Enter your first name then a space and your family name "
# if the user don't enter his name the programm will keep asking him his name 
while [ -z $username ]
do
	echo " Please enter your login: "
	read username
done

echo " You Login is : $username "
