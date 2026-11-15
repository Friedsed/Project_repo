#!/bin/bash

# number to verify 

#check if the number is more than 10

file="/home/friedly/Documents/coding_file/terminal_usage/Sophie_my_love.txt"

if [ -e "$file" ]

then

	echo " Sophie loves you of course ! "

else

	echo " She is just waiting for your approval "


fi
