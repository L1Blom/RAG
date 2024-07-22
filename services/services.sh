#!/usr/bin/bash
while getopts "use" option
do 
    case "${option}"
        in
        u)do_update=1;;
        s)do_stop=1;;
        e)do_enable=1;;
    esac

done

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