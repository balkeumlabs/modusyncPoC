#!/bin/bash

SESSION="fl-session"
tmux new-session -d -s $SESSION

tmux rename-window -t $SESSION 'Server'
tmux send-keys -t $SESSION "python3 server.py" C-m

sleep 2

for i in 0 1 2
do
    tmux new-window -t $SESSION -n "Client$i"
    tmux send-keys -t $SESSION "echo $i | python3 clients.py" C-m
    sleep 1
done

tmux attach -t $SESSION
