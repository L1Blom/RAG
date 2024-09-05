#!/usr/bin/bash
# for help use option -h

while getopts "useh" option
do 
    case "${option}" in
        u)do_update=1;;
        s)do_stop=1;;
        e)do_enable=1;;
        h)do_help=1;;
    esac

done

if [ "$do_help" = 1 ]; then
    echo 'Run this script from the directory with the source service files (*.service)'
    echo ' Options are:'
    echo ' -u -> this will cp and overwrite the service files to /etc/systemd/system and reload daemon'
    echo ' -s -> `systemctl stop` for any service found with the name of the service files without extension'
    echo ' -e -> `systemctl enable and start` for any service found with the name of the service files without extension'
    echo ' -h -> show this message'
    echo 'Combinations are possible, -h will exit without perform any actions.'
    echo 'if no option is given, it will perform update -> stop -> enable -> start in this sequence.'
    echo ''
    exit
fi

if [ "$do_update$do_stop$do_enable" = "" ]; then
    echo "No options given, executing 'update', 'stop' and 'enable and start'"
fi

if [ "$do_update$do_stop$do_enable" = "" ] ||  [ "$do_update" = 1 ]; then
    #move or update the services files
    echo "Updating services files: " `ls *.service`   
    sudo cp -u *.service /etc/systemd/system
    #reload daemon
    sudo systemctl daemon-reload
fi


if [ "$do_update$do_stop$do_enable" = "" ] ||  [ "$do_stop" = 1 ]; then
    #stop the services
    for service in *.service; do
        filename=$(basename -- "$service")
        sname="${filename%.*}"
        echo "Stopping and disbling service: " $sname
        up=`sudo systemctl |grep $sname | wc -l` 
        if [ $up = 1 ]; then
            sudo systemctl stop $sname
            sudo systemctl disable $sname
        fi
    done
fi

#start the services
if [ "$do_update$do_stop$do_enable" = "" ] ||  [ "$do_enable" = 1 ]; then
    for service in *.service; do
        filename=$(basename -- "$service")
        sname="${filename%.*}"
        echo "Enabling and starting: " $sname
        sudo systemctl enable $sname
        sudo systemctl start $sname
    done
fi