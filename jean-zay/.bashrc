
############# spython specific macros #####################
alias sl='python $HOME/logs.py'
alias spython='python ~/jean-zay.py'

saccelerate() {
    echo $@
    python ~/jean-zay.py $@ --command accelerate
}

############# spython specific macros #####################
