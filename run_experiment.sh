for i in {1..5}; do
    for a in {dqn,qlearning}; do
        for c in {config1,config2,config3}; do
            python3 main.py "$c" aio "$a" 2>&1 >> experiment.logs
            echo
        done
    done
done
