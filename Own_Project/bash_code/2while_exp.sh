#!/bin/bash

#com1
#com2

nb=5
nbb=2
while [ $nb -le 10  ] && [ $nbb -le 8 ]
do
	echo "Number is : $nb - and number nbb is : $nbb "
	(( nb++ ))
	(( nbb++ ))

done

echo " Done with commande "
