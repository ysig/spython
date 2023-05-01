
############# spython specific macros #####################
alias spython='python ~/jean-zay.py'

saccelerate() {
    echo $@
    python ~/jean-zay.py $@ --command accelerate
}
############# spython specific macros #####################
