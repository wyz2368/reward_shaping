from simulator import scan_and_sim
import datetime

def run(method_list):
    payoff_matrix_att, payoff_matrix_def, child_partition = scan_and_sim(method_list)




if __name__ == '__main__':
    print("...Begin Payoff Matrix Simulation")
    print("Current time is ", datetime.datetime.now())
    method_list = ["reward_shaping", "DO_selfplay"]
    run(method_list)
    print("Current time is ", datetime.datetime.now())
    print("...Done Payoff Matrix Simulation...")