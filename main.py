import numpy as np
import math
#import matplotlib.pyplot as plt
from matspy import spy
from datetime import datetime

def kacans_power_flow_solver(cdf_path):

    start=datetime.now()

    with open(cdf_path) as fil:
     lines = [line.rstrip('\n') for line in fil]

    count_999 = 0
    indexes = []
    for i in range(len(lines)):
      if lines[i][0:3]=="BUS" or lines[i][0:4] =="-999" or lines[i][0:6]=="BRANCH":
       indexes.append(i)
       count_999 += 1
       if count_999>3:
           break

    bus_data_follows = lines[(indexes[0]+1):indexes[1]]
    branch_data_follows = lines[(indexes[2]+1):(indexes[3])]

    #Precreating the bus data arrays to prevent dynamic array creation
    bus_number= np.zeros((len(bus_data_follows),1))
    bus_type=np.zeros((len(bus_data_follows),1))
    bus_voltage=np.zeros((len(bus_data_follows),1))
    bus_angle=np.zeros((len(bus_data_follows),1))
    bus_load_mw=np.zeros((len(bus_data_follows),1))
    bus_load_mvar=np.zeros((len(bus_data_follows),1))
    bus_gen_mw=np.zeros((len(bus_data_follows),1))
    bus_gen_mvar=np.zeros((len(bus_data_follows),1))
    bus_max_limit=np.zeros((len(bus_data_follows),1))
    bus_min_limit=np.zeros((len(bus_data_follows),1))
    bus_shunt_G=np.zeros((len(bus_data_follows),1))
    bus_shunt_B=np.zeros((len(bus_data_follows),1))

    for bus in range(len(bus_data_follows)):
        bus_number[bus] = bus_data_follows[bus][0:4]
        bus_type[bus] = bus_data_follows[bus][24:26]
        bus_voltage[bus] = bus_data_follows[bus][27:33]
    #    bus_angle[bus] = bus_data_follows[bus][33:40]
        bus_load_mw[bus] = bus_data_follows[bus][40:49]
        bus_load_mvar[bus] = bus_data_follows[bus][49:59]
        bus_gen_mw[bus] = bus_data_follows[bus][59:67]
        bus_gen_mvar[bus] = bus_data_follows[bus][67:75]
        bus_max_limit[bus] = bus_data_follows[bus][90:98]
        bus_min_limit[bus] = bus_data_follows[bus][98:106]
        bus_shunt_G[bus] = bus_data_follows[bus][106:114]
        bus_shunt_B[bus] = bus_data_follows[bus][114:122]


    bus_number = np.int_(bus_number)
    bus_type = np.int_(bus_type)
    bus_voltage = np.float_(bus_voltage)
    bus_angle = np.float_(bus_angle)
    bus_load_mw = np.float_(bus_load_mw)
    bus_load_mvar = np.float_(bus_load_mvar)
    bus_gen_mw = np.float_(bus_gen_mw)
    bus_gen_mvar = np.float_(bus_gen_mvar)
    bus_max_limit = np.float_(bus_max_limit)
    bus_min_limit = np.float_(bus_min_limit)
    bus_shunt_G = np.float_(bus_shunt_G)
    bus_shunt_B = np.float_(bus_shunt_B)


    #Precreating the branch data arrays to prevent dynamic array creation
    from_bus = np.zeros((len(branch_data_follows),1))     #For tap Xformer, 'a' is here other is unity
    to_bus =  np.zeros((len(branch_data_follows),1))
    branch_type = np.zeros((len(branch_data_follows),1))
    branch_r = np.zeros((len(branch_data_follows),1))
    branch_x = np.zeros((len(branch_data_follows),1))
    branch_b = np.zeros((len(branch_data_follows),1))  #Total
    transformer_ratio = np.zeros((len(branch_data_follows),1))
    transformer_angle = np.zeros((len(branch_data_follows),1))

    for bra in range(len(branch_data_follows)):
        from_bus[bra] = branch_data_follows[bra][0:4]
        to_bus[bra] = branch_data_follows[bra][5:9]
        branch_type[bra] = branch_data_follows[bra][18:19]
        branch_r[bra] = branch_data_follows[bra][19:29]
        branch_x[bra] = branch_data_follows[bra][29:40]
        branch_b[bra] = branch_data_follows[bra][40:50]
        transformer_ratio[bra] = branch_data_follows[bra][76:82]
        transformer_angle[bra] = branch_data_follows[bra][83:90]

    from_bus = np.int_(from_bus)
    to_bus = np.int_(to_bus)
    branch_type = np.int_(branch_type)
    branch_r = np.float_(branch_r)
    branch_x = np.float_(branch_x)*1j
    branch_series = branch_r+branch_x
    branch_b = np.float_(branch_b)*1j
    transformer_ratio = np.float_(transformer_ratio)
    transformer_angle = np.float_(transformer_angle)

    from_bus_indexed = [np.where(bus_number == element)[0][0] for element in from_bus]
    to_bus_indexed = [np.where(bus_number == element)[0][0] for element in to_bus]

    number_of_buses=len(bus_number)
    bus_numbers_consecutive = list(range(number_of_buses))


    Y_bus = np.zeros((number_of_buses,number_of_buses))*1j


    for e in range(len(bus_number)):
        Y_bus[e, e] = (bus_shunt_G[e,0]+(bus_shunt_B[e,0]*1j))

    for entry in range(len(from_bus)):
        branch_y = (1/(branch_series[entry,0]))
        line_b = branch_b[entry, 0]*0.5
        if branch_type[entry, 0] == 0:
            Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += branch_y + line_b
            Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y + line_b
            Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y
            Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y

        if branch_type[entry] == 1 or branch_type[entry] == 2 or branch_type[entry] == 3:
            Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += (branch_y / (transformer_ratio[entry,0]**2)) + line_b
            Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y + line_b
            Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y / transformer_ratio[entry,0]
            Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y / transformer_ratio[entry,0]

        if branch_type[entry] == 4:
            Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += (branch_y / (transformer_ratio[entry,0]**2)) + line_b
            Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y + line_b
            rad = math.radians(transformer_angle[entry,0])
            ratio = transformer_ratio[entry,0]*(math.cos(rad)+(math.sin(rad)*1j))
            conj_ratio = transformer_ratio[entry,0]*(math.cos(rad)-(math.sin(rad)*1j))
            Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y / conj_ratio
            Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y / ratio

    spy(Y_bus)
    """
    plt.title("Y_bus Sparsity Plot")
    plt.spy(Y_bus)
    plt.show()
    """
    G_bus = Y_bus.real
    B_bus = Y_bus.imag

    start=datetime.now()
    slack_bus_index = np.where(bus_type == 3)
    slakiana = bus_numbers_consecutive.pop(slack_bus_index[0][0])
    bus_numbers_with_slack_consecutive = bus_numbers_consecutive[:]
    bus_numbers_with_slack_consecutive.insert(slakiana, slakiana)
    aug_bus_load_mw = np.delete(bus_load_mw, slack_bus_index[0])
    aug_bus_gen_mw = np.delete(bus_gen_mw, slack_bus_index[0])
    aug_bus_voltages = np.ones((number_of_buses-1,1))
    aug_bus_voltages_with_slack = np.insert(aug_bus_voltages,slack_bus_index[0][0],1)
    aug_bus_angles = np.zeros((number_of_buses-1,1))
    aug_bus_angles_with_slack = np.insert(aug_bus_angles,slack_bus_index[0][0],0)


    bus_count_pq = np.sum(bus_type == 0) + np.sum(bus_type == 1)
    bus_count_pv = np.sum(bus_type == 2)
    missmatch_len = bus_count_pq+bus_count_pv+bus_count_pq

    pq_buses_voltage = np.empty(shape=[bus_count_pq,1])
    pq_buses_names = np.empty(shape=[bus_count_pq,1])


    x_counter = 0
    for pq in range(len(bus_data_follows)):
       if bus_type[pq] == 0 or bus_type[pq] == 1:
           pq_buses_voltage[x_counter, 0] = 1
           pq_buses_names[x_counter, 0] = bus_number[pq,0]
           x_counter += 1

    x = np.vstack((aug_bus_angles, pq_buses_voltage))
    Jacobian = np.zeros((len(x),len(x)))

    pq_index = [np.where(bus_number == element)[0][0] for element in pq_buses_names]
    pq_index = np.array(pq_index)

    given_power_vector = np.zeros([missmatch_len,1])
    calculated_power_vector = np.zeros([missmatch_len,1])
    for g in range((len(bus_number)-1)):
        given_power_vector[g, 0] = aug_bus_gen_mw[g] - aug_bus_load_mw[g]

    g += 1
    for m in range((len(bus_number))):
        if (bus_type[m, 0] == 0) or (bus_type[m, 0] == 1):
            given_power_vector[g, 0] = bus_gen_mvar[m, 0] - bus_load_mvar[m, 0]
            g += 1

    #Flat start adjustment
    for volt in range((len(bus_voltage))):
        bus_voltage[volt, 0] = 1

    step = 0
    best_fit = 1e6
    best_fit_mean = 1e6
    mismatch_vector = np.array([1])
    while (abs(np.mean(mismatch_vector))>0.001):
        calculated_power_vector = np.zeros([missmatch_len, 1])
        Jacobian = np.zeros((len(x), len(x)))

        fit = abs(mismatch_vector.max())
        fit_mean = abs(np.mean(mismatch_vector))

        if fit < best_fit:
            bus_voltages_best = aug_bus_voltages_with_slack
            bus_angles_best = aug_bus_angles_with_slack
            best_fit = fit
            fit_step = step


        if fit_mean < best_fit_mean:

            bus_voltages_mean_best = aug_bus_voltages_with_slack
            bus_angles_mean_best = aug_bus_angles_with_slack
            best_fit_mean = fit_mean
            fit_mean_step = step

        for i in range((len(bus_number)-1)):
            for k in range((len(bus_number))):
                calculating_bus = bus_numbers_consecutive[i]
                angle = (aug_bus_angles_with_slack[calculating_bus] - aug_bus_angles_with_slack[k])
                cos = math.cos(angle)
                sin = math.sin(angle)
                calculated_power_vector[i, 0] += aug_bus_voltages_with_slack[calculating_bus]*aug_bus_voltages_with_slack[k]*((G_bus[calculating_bus,k]*cos)+(B_bus[calculating_bus,k]*sin))

        calculated_bus_q = np.zeros((number_of_buses,1))
        for z in range(len(bus_number)):
            for m in range(len(bus_number)):
                angle = (aug_bus_angles_with_slack[z] - aug_bus_angles_with_slack[m])
                cos = math.cos(angle)
                sin = math.sin(angle)
                calculated_bus_q[z,0] += aug_bus_voltages_with_slack[z]*aug_bus_voltages_with_slack[m]*((G_bus[z,m]*sin)-(B_bus[z,m]*cos))

        for l in range((len(bus_number))): #hataaaa burada
            if (bus_type[l, 0] == 0) or (bus_type[l, 0] == 1):
                i += 1
                calculated_power_vector[i, 0] += calculated_bus_q[l, 0]

        if (abs(np.mean(mismatch_vector))<0.01):
            break

        mismatch_vector = calculated_power_vector - (given_power_vector/100)



        for p in range(len(bus_number)-1):
            for delta_theta in range(len(bus_number)-1):
                which_p_bus = bus_numbers_consecutive[p]
                deriv_theta = bus_numbers_consecutive[delta_theta]
                theta_diff = aug_bus_angles_with_slack[which_p_bus] - aug_bus_angles_with_slack[deriv_theta]
                cos = math.cos(theta_diff)
                sin = math.sin(theta_diff)
                if which_p_bus != deriv_theta:
                    Jacobian[p, delta_theta] = aug_bus_voltages_with_slack[which_p_bus]*aug_bus_voltages_with_slack[deriv_theta]*((G_bus[which_p_bus,deriv_theta]*sin) - (B_bus[which_p_bus,deriv_theta]*cos))
                if which_p_bus == deriv_theta:
                    Jacobian[p, delta_theta] = -calculated_bus_q[which_p_bus,0]-(B_bus[which_p_bus, deriv_theta]*(aug_bus_voltages_with_slack[which_p_bus]**2))
                    #print(p, delta_theta, -calculated_bus_q[which_p_bus,0]-(B_bus[which_p_bus, deriv_theta]*(aug_bus_voltages_with_slack[which_p_bus]**2)))

        for q in range(bus_count_pq):
            for delta_theta in range(len(bus_number) - 1):
                which_q_bus = pq_index[q]
                derive_theta = bus_numbers_consecutive[delta_theta]
                theta_diff = aug_bus_angles_with_slack[which_q_bus] - aug_bus_angles_with_slack[derive_theta]
                cos = math.cos(theta_diff)
                sin = math.sin(theta_diff)
                if which_q_bus != derive_theta:
                    Jacobian[p+q+1, delta_theta] -= pq_buses_voltage[q, 0]*aug_bus_voltages[delta_theta,0]*((G_bus[which_q_bus, derive_theta]*cos)+(B_bus[which_q_bus, derive_theta]*sin))
                if which_q_bus == derive_theta:
                    Jacobian[p+q+1, delta_theta] = calculated_power_vector[which_q_bus, 0] - (G_bus[which_q_bus, which_q_bus]*(aug_bus_voltages_with_slack[which_q_bus]**2))

        for pp in range(len(bus_number)-1):
            for delta_v in range(bus_count_pq):
                which_bus_v = bus_numbers_with_slack_consecutive[pq_index[delta_v]]
                which_p = bus_numbers_consecutive[pp]
                theta_diff = aug_bus_angles_with_slack[which_p] - aug_bus_angles_with_slack[which_bus_v]
                cos = math.cos(theta_diff)
                sin = math.sin(theta_diff)
                if which_bus_v != which_p:
                    Jacobian[pp, delta_theta + delta_v+1] = (aug_bus_voltages_with_slack[which_p]*G_bus[which_p, which_bus_v]*cos)+(aug_bus_voltages_with_slack[which_p]*B_bus[which_p, which_bus_v]*sin)
                if which_bus_v == which_p:
                    Jacobian[pp, delta_theta + delta_v + 1] = (calculated_power_vector[which_p, 0] / aug_bus_voltages_with_slack[which_p])+(G_bus[which_p, which_bus_v]*aug_bus_voltages_with_slack[which_p])



        for qq in range(bus_count_pq):
            for delta_v_v in range(bus_count_pq):
                which_bus_v = bus_numbers_with_slack_consecutive[pq_index[delta_v_v]]
                which_q = bus_numbers_with_slack_consecutive[pq_index[qq]]
                theta_diff = aug_bus_angles_with_slack[which_q] - aug_bus_angles_with_slack[which_bus_v]
                cos = math.cos(theta_diff)
                sin = math.sin(theta_diff)

                if which_bus_v != which_q:
                    Jacobian[pp + qq + 1, delta_theta + delta_v_v + 1] = (aug_bus_voltages_with_slack[which_q]*G_bus[which_q, which_bus_v]*sin)-(aug_bus_voltages_with_slack[which_q]*B_bus[which_q, which_bus_v]*cos)
                if which_bus_v == which_q:
                    Jacobian[pp + qq + 1, delta_theta + delta_v_v + 1] = (calculated_bus_q[which_q,0] / aug_bus_voltages_with_slack[which_q])-(B_bus[which_q, which_q]*aug_bus_voltages_with_slack[which_q])


        Jacobian_inverse = np.linalg.inv(Jacobian)
        x = x - np.matmul(Jacobian_inverse,mismatch_vector)
        aug_bus_angles = x[:len(bus_number)-1,:]
        aug_bus_angles = aug_bus_angles % (math.pi*2)
        aug_bus_angles_with_slack = np.insert(aug_bus_angles,slack_bus_index[0][0],0)



        x_counter = 0
        for pq in range(len(bus_number)):
           if bus_type[pq] == 0 or bus_type[pq] == 1:
               aug_bus_voltages_with_slack[pq] = x[len(bus_number) + x_counter -1 , 0]
               x_counter += 1
        aug_bus_voltages = aug_bus_voltages_with_slack[~np.isin(np.arange(aug_bus_voltages_with_slack.size), slack_bus_index[0][0])]
        aug_bus_voltages = aug_bus_voltages.reshape(len(bus_number)-1,1)
        step += 1


        if step==250:
            print("Max iteration has been reached. Could not solve!")
            break


    #System loss calculation
    total_gen_mw = 0
    total_gen_mvar = 0

    total_load_mw = sum(bus_load_mw)/100
    total_load_mvar = sum(bus_load_mvar)/100

    pq_bus_now = np.zeros((number_of_buses,1))

    for y in range(len(bus_type)):
        if bus_type[y] == 2 or bus_type[y] == 3:
            for w in range(len(bus_type)):
                theta_difference = bus_angles_mean_best[y] - bus_angles_mean_best[w]
                cos = math.cos(theta_difference)
                sin = math.cos(theta_difference)
                G_yw = G_bus[y,w]
                B_yw = B_bus[y,w]
                total_gen_mw += bus_voltages_mean_best[y]*bus_voltages_mean_best[w]*((G_yw*cos)+(B_yw*sin))
                pq_bus_now[y,0] += bus_voltages_mean_best[y]*bus_voltages_mean_best[w]*((G_yw*sin)-(B_yw*cos))

    total_gen_mvar = sum(pq_bus_now)
    buses_that_did_not_stuck_to_limits = []
    for e in range(len(pq_bus_now)):
        if bus_type[e] == 2:
            if bus_min_limit[e,0] <= pq_bus_now[e,0]*100 <= bus_max_limit[e,0]:
                buses_that_did_not_stuck_to_limits.append(bus_number[e,0])




    loss_mw = abs(total_gen_mw - total_load_mw)
    loss_mvar = abs(total_gen_mvar - total_load_mvar)


    print("Bus voltages (p.u.):", bus_voltages_mean_best)
    print("Bus Angles (degrees):", bus_angles_mean_best*180/math.pi)
    print("Solution time(h:m:s):", datetime.now()-start)
    print("Number of iterations:", step+1)
    print("Active power loss in the system (mw):", loss_mw[0])
    print("Reactive power loss in the system (mwar):", loss_mvar[0])
    print("PV bus numbers that stuck to Q limits:", buses_that_did_not_stuck_to_limits)


kacans_power_flow_solver(r'C:\Users\Msi\Desktop\class_4\EE472\proje_1\ieee300cdf.txt')