#!/bin/bash

#Commentaire

################

PS3=" What do you wanna execute? "
options=("See the system version" "See the SSD memory " "See the configuration of the IP " " See the last date you restart your computer" "Exit")

select choix in "${options[@]}"; do
        case $REPLY in
                1)
                        echo "The system version is : "
                        uname -a
                        ;;
                2)
                        echo " The SSD memory is : "
                        df -h
                        ;;
                3)
                        echo "Configuration of the IP"
                        ip a
                        ;;
                4)
                        echo " Last date the computer was restart "
                        who -b
                        ;;
                5)
                        echo " Exit"
                        break
                        ;;
                *)
                        echo "Option is not valide, please take it again ! "
                        ;;
        esac
done
