#!/bin/bash

# number to verify 
hour=20

# Check if the hour corresponds to the time of day

if [[ $hour -lt 13 ]]
then
    echo " Good morning Sophie ! love you so much"

elif [[ $hour -lt 17 ]]
then
    echo "Good afternoon Sophie ! My love is intensive! "

else
    echo " Good evening! Just waiting for you to join me"
fi
