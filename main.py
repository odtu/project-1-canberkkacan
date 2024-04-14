import pandas as pd
import numpy as np
import math

with open(r'C:\Users\Msi\Desktop\class_4\EE472\proje_1\ieee300cdf.txt') as fil:
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
    bus_angle[bus] = bus_data_follows[bus][33:40]
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

Y_bus = np.zeros((number_of_buses,number_of_buses))*1j


for e in range(len(bus_number)):
    Y_bus[e, e] = (bus_shunt_G[e,0]+bus_shunt_B[e,0]*1j)

for entry in range(len(from_bus)):
    branch_y = (1/(branch_series[entry,0]))+branch_b[entry,0]
    if branch_type[entry,0] == 0:
        Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += branch_y
        Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y
        Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y
        Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y

    if branch_type[entry] == 1 or branch_type[entry] == 2 or branch_type[entry] == 3:
        Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += branch_y / (transformer_ratio[entry,0]**2)
        Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y
        Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y / transformer_ratio[entry,0]
        Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y / transformer_ratio[entry,0]

    if branch_type[entry] == 4:
        Y_bus[from_bus_indexed[entry], from_bus_indexed[entry]] += branch_y / (transformer_ratio[entry,0]**2)
        Y_bus[to_bus_indexed[entry], to_bus_indexed[entry]] += branch_y
        rad = math.radians(transformer_angle[entry,0])
        ratio = transformer_ratio[entry,0]*(math.cos(rad)+(math.sin(rad)*1j))
        conj_ratio = transformer_ratio[entry,0]*(math.cos(rad)-(math.sin(rad)*1j))
        Y_bus[from_bus_indexed[entry], to_bus_indexed[entry]] -= branch_y / conj_ratio
        Y_bus[to_bus_indexed[entry], from_bus_indexed[entry]] -= branch_y / ratio
