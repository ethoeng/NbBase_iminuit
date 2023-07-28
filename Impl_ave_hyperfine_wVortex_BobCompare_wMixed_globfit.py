#!/usr/bin/python3

"""
Fit the 8Li SLR rate in the Nb (baseline) sample.
"""

import json
import sys
from glob import glob
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.pdg_format import pdg_format, latex
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

from scipy import constants, integrate, interpolate, special

from sklearn import linear_model

# sys.path.append(
#     "/home/rmlm/git/github/rmlmcfadden/hyperfine"
# )  # Adds higher directory to Python modules path.
sys.path.append("/mnt/c/Users/thoen/Desktop/RMLM-Global-Fit/ImplAveCode/hyperfine")

from hyperfine.minuit import minuit2json, json2minuit, LeastSquares3D
from hyperfine.GL_averaging import GL_Quad,averaged_fields_GL
from hyperfine.bnmr.meissner_wVortex_BobCompare_globfit import DepthAveragingCalculator


sys.path.append("/mnt/c/Users/thoen/Desktop/RevSciInst/rsi/figures/Python")
plt.style.use('tex')
from my_plot import set_size

# sys.path.append("../")
# from DepthAveragingCalculator import *

# create a single DepthAveragingCalculator
dac = DepthAveragingCalculator("../8Li_stopping_Nb2O5_5nm_Nb_fitpars.csv")

# redefine dac functor for temp_array (i.e., T-scan)
def fcn(
    temperatures_K: float,
    energy_keV: float,
    applied_field_T: float,
    dead_layer_nm: float,
    lambda_L_100mT_nm: float,
    lambda_L_110mT_nm: float,
    lambda_L_125mT_nm: float,
    lambda_L_150mT_nm: float,
    lambda_L_200mT_nm: float,
    a:float,#dipolar_field_T: float,
    b:float,#correlation_time_s: float,
    # temperature_K: float,
    critical_temperature_K: float,
    critical_field_2_T: float,
    enhance_fact_100mT:float,
    enhance_fact_110mT:float,
    enhance_fact_125mT:float,
    enhance_fact_150mT:float,
    enhance_fact_200mT:float,
    # B_noscreen_T:float,
    # enhance_fact_mixed:float,
    # B_mixed_T:float,
) -> float:
    temperature_K = np.asarray(temperatures_K)
    if temperature_K.size == 0:
        return dac(
            energy_keV,
            applied_field_T,
            dead_layer_nm,
            lambda_L_100mT_nm,
            lambda_L_110mT_nm,
            lambda_L_125mT_nm,
            lambda_L_150mT_nm,
            lambda_L_200mT_nm,
            a,#dipolar_field_T: float,
            b,#correlation_time_s: float,
            temperature_K,
            critical_temperature_K,
            critical_field_2_T,
            enhance_fact_100mT,
            enhance_fact_110mT,
            enhance_fact_125mT,
            enhance_fact_150mT,
            enhance_fact_200mT,
            # B_noscreen_T,
            # enhance_fact_mixed,
            # B_mixed_T,
        )
    else:
        results = np.empty(temperature_K.size)
        for i, t in enumerate(temperatures_K):
            results[i] = dac(
                energy_keV,
                applied_field_T,
                dead_layer_nm,
                lambda_L_100mT_nm,
                lambda_L_110mT_nm,
                lambda_L_125mT_nm,
                lambda_L_150mT_nm,
                lambda_L_200mT_nm,
                a,#dipolar_field_T: float,
                b,#correlation_time_s: float,
                t,
                critical_temperature_K,
                critical_field_2_T,
                enhance_fact_100mT,
                enhance_fact_110mT,
                enhance_fact_125mT,
                enhance_fact_150mT,
                enhance_fact_200mT,
                # B_noscreen_T,
                # enhance_fact_mixed,
                # B_mixed_T,
                    )
        return results


def fcn_b(
    applied_fields_T: float,
    temperature_K: float,
    energy_keV: float,
    dead_layer_nm: float,
    lambda_L_100mT_nm: float,
    lambda_L_110mT_nm: float,
    lambda_L_125mT_nm: float,
    lambda_L_150mT_nm: float,
    lambda_L_200mT_nm: float,
    a:float,#dipolar_field_T: float,
    b:float,#correlation_time_s: float,
    critical_temperature_K: float,
    critical_field_2_T: float,
    enhance_fact_100mT:float,
    enhance_fact_110mT:float,
    enhance_fact_125mT:float,
    enhance_fact_150mT:float,
    enhance_fact_200mT:float,
    # B_noscreen_T:float,
    # enhance_fact_mixed:float,
    # B_mixed_T:float,
) -> float:
    applied_field_T = np.asarray(applied_fields_T)
    if applied_field_T.size == 0:
        return dac(
            energy_keV,
            applied_field_T,
            dead_layer_nm,
            lambda_L_100mT_nm,
            lambda_L_110mT_nm,
            lambda_L_125mT_nm,
            lambda_L_150mT_nm,
            lambda_L_200mT_nm,
            a,#dipolar_field_T: float,
            b,#correlation_time_s: float,
            temperature_K,
            critical_temperature_K,
            critical_field_2_T,
            enhance_fact_100mT,
            enhance_fact_110mT,
            enhance_fact_125mT,
            enhance_fact_150mT,
            enhance_fact_200mT,
            # B_noscreen_T,
            # enhance_fact_mixed,
            # B_mixed_T,
        )
    else:
        results = np.empty(applied_field_T.size)
        for i, b in enumerate(applied_field_T):
            results[i] = dac(
                energy_keV,
                b,
                dead_layer_nm,
                lambda_L_100mT_nm,
                lambda_L_110mT_nm,
                lambda_L_125mT_nm,
                lambda_L_150mT_nm,
                lambda_L_200mT_nm,
                a,#dipolar_field_T: float,
                b,#correlation_time_s: float,
                temperature_K,
                critical_temperature_K,
                critical_field_2_T,
                enhance_fact_100mT,
                enhance_fact_110mT,
                enhance_fact_125mT,
                enhance_fact_150mT,
                enhance_fact_200mT,
                # B_noscreen_T,
                # enhance_fact_mixed,
                # B_mixed_T,  
                )
        return results


def fcn3d(
    temperature_K: float,
    energy_keV: float,
    applied_field_T: float,
    dead_layer_nm: float,
    lambda_L_100mT_nm: float,
    lambda_L_110mT_nm: float,
    lambda_L_125mT_nm: float,
    lambda_L_150mT_nm: float,
    lambda_L_200mT_nm: float,
    a:float,#dipolar_field_T: float,
    b:float,#correlation_time_s: float,
    critical_temperature_K: float,
    critical_field_2_T: float,
    enhance_fact_100mT:float,
    enhance_fact_110mT:float,
    enhance_fact_125mT:float,
    enhance_fact_150mT:float,
    enhance_fact_200mT:float, 
    # B_noscreen_T:float,
    # enhance_fact_mixed:float,
    # B_mixed_T:float,
) -> float:
    return dac(
        energy_keV,
        applied_field_T,
        dead_layer_nm,
        lambda_L_100mT_nm,
        lambda_L_110mT_nm,
        lambda_L_125mT_nm,
        lambda_L_150mT_nm,
        lambda_L_200mT_nm,
        a,#dipolar_field_T: float,
        b,#correlation_time_s: float,
        temperature_K,
        critical_temperature_K,
        critical_field_2_T,
        enhance_fact_100mT,
        enhance_fact_110mT,
        enhance_fact_125mT,
        enhance_fact_150mT,
        enhance_fact_200mT,
        # B_noscreen_T,
        # enhance_fact_mixed,
        # B_mixed_T,
    )


"""
#####################
GLOBAL VARIABLES
###################
"""
N_FIT_POINTS = 50

"""
Max
"""
B_max_mT_plot = 240.0
x_max_nm_plot = 160.0#215.0
E_max_keV_plot = 21.0

E_max_keV_plotRelax = 21.0
E_max_keV_fit = 20.0 

Relax_max_Hz_plot = 5.0

energy_keV_cutoff_vortex = 4.5

"""
Min
"""
# Plotting range purpose
B_min_mT_plot = 0.0
x_min_nm_plot = 0.0
E_min_keV_plot = 0.0

E_min_keV_plotRelax = 2.0 # min plot range for Relax Rate

# Fit E-array purpose
E_min_keV_fit = 0.1 # below this -> rho(E) interp. error

Relax_min_Hz_plot = 0.0


"""
Convert Range (nm) to Energy (keV) for dead layer
"""
def range_to_erg(range_nm, a=2.98666643,b=6.7216862):
    return (range_nm-b)/a

"""
#######################################
"""





"""
#########################
#####################
LPF DATA
####################
#########################
"""
df_LPF = pd.read_csv("M1963_Oct2019_NbBaseline_SCDS_24mT.csv", delimiter=",", comment="#")
df_LPF.rename(
                    columns={
                    'Error- 1_T1':'Error- 1_T1_0',
                    'Error+ 1_T1':'Error+ 1_T1_0',
                    '1_T1':'1_T1_0'}, 
                    inplace=True
                    )

df_LPF['Error_ave 1_T1_0'] = df_LPF[['Error- 1_T1_0','Error+ 1_T1_0']].mean(axis=1)
Temps_K = np.sort(pd.unique(df_LPF["Temperature (K)"].round(decimals=2)))
# print('24mT Temps:',Temps_K)

"""
#############
SCDS: 24 mT
############
"""
data_24mT_SC_Depthscan = df_LPF[
                            np.isclose(df_LPF["Temperature (K)"],4.0,atol=0.5)
                        ].sort_values(by=['Impl. Energy (keV)'])

"""
############
SCFS: 4 keV
###########
"""

"""
#################
#################
HPF DATA
#################
#################
"""
# alldata = pd.read_csv("M1963_Oct2021_NbBaseline_50mT.csv", delimiter=",", comment="#")
df_all = pd.read_csv("NbBase_2023_AllFit.csv", delimiter=",", comment="#")
# df_all.loc[df_all["Year"]>2022,"Year"] = 2022

df_all['Error_ave 1_T1_0'] = df_all[['Error- 1_T1_0','Error+ 1_T1_0']].mean(axis=1)

# Fields_T =  np.sort(pd.unique(df_all["B0 Field (T)"].round(decimals=3)))
# print('All Fields:{} mT'.format(Fields_T*1e3))

# Energies_keV =  np.sort(pd.unique(df_all["Impl. Energy (keV)"].round(decimals=3)))
# print('All Energies:{} keV'.format(Energies_keV))

# Temps_K = np.sort(pd.unique(df_all["Temperature (K)"].round(decimals=0)))
# print('All Temps: {} K'.format(Temps_K))

# print(np.sort(pd.unique(df_all['Year'])))

"""
# Oct 2021 (50 mT) data filter
"""
df1 = df_all[
            np.isclose(df_all["B0 Field (T)"],0.050, rtol=0.01)
            ]# pd.read_csv("M1963_Oct2021_NbBaseline_50mT.csv", delimiter=",", comment="#")

# df1['Error_ave 1_T1_0'] = df1[['Error- 1_T1_0','Error+ 1_T1_0']].mean(axis=1)
# data_Oct21 = df1

# data_Oct21 = df1.drop(df1[
#                                 df1["Temperature (K)"] < 10.0
#                                 ].index)
# data_Oct21 = data_Oct21[
                        # np.isclose(data_Oct21["Temperature (K)"],15,atol=1.5) & #NSFS_12keV
                        # np.isclose(data_Oct21['Impl. Energy (keV)'],11, atol=1) #NSFS_12keV
                        # ]                 


"""
# Oct 2022 Data Filter - Field Scans (100 - 200 mT)
"""
df2 = df_all[
             (df_all["B0 Field (T)"] > 0.07) 
             & (df_all["Run Number"] >= 45238)
            ]
            #pd.read_csv("M1963_Oct2022_NbBaseline.csv", delimiter=",", comment="#")
# df2=df2.drop(df2[df2["B0 Field (T)"] < 0.075].index) # drop the 50 mT fit 
# df2['Error_ave 1_T1_0'] = df2[['Error- 1_T1_0','Error+ 1_T1_0']].mean(axis=1)


# Fields_T =  np.sort(pd.unique(df2["B0 Field (T)"].round(decimals=3)))
# print('All Fields:{} mT'.format(Fields_T*1e3))

# Energies_keV =  np.sort(pd.unique(df2["Impl. Energy (keV)"].round(decimals=3)))
# print('All Energies:{} keV'.format(Energies_keV))

# Temps_K = np.sort(pd.unique(df2["Temperature (K)"].round(decimals=3)))
# print('All Temps: {} K'.format(Temps_K))


"""
# Aug 2022 Data Filter - 100 mT
"""
df3 = df_all[
            np.isclose(df_all["B0 Field (T)"],0.10, rtol=0.01)
             & (df_all["Run Number"] >= 45074)
             & (df_all["Run Number"] <= 45115)
            ]

"""
##########################
FILTER & PLOT BY SCANS
##########################
"""

def df_inspect(df_dict):
    fig, ax = plt.subplots()
    # for df_dict in df_dict_all:

    df = df_dict["df-obj"]
    df_label = df_dict["label"]

    Fields_T =  np.sort(pd.unique(df["B0 Field (T)"].round(decimals=3)))
    Energies_keV =  np.sort(pd.unique(df["Impl. Energy (keV)"].round(decimals=3)))
    Temps_K = np.sort(pd.unique(df["Temperature (K)"].round(decimals=3)))

    print("Type of Scans:{} \n Fields: {} mT \n Energies: {} keV \n Temps: {} K\n".format(df_label,Fields_T*1e3, Energies_keV, Temps_K))


    if "sortby-key" in df_dict:
        sortby_val = np.sort(pd.unique(df[df_dict["sortby-key"]].round(decimals=3)))
        for i,val in enumerate(sortby_val): # plot dataset for each e.g. magnetic field
            """
            #change tolerance below 
            #if filter by temp, e.g. atol=0.5, 
            # need rtol=0.0 for filter by B-field
            """
            df_filtered = df[
                                np.isclose(df[df_dict["sortby-key"]],val,rtol=0.01) 
                                # & np.isclose(df["Temperature (K)"],temps[1],rtol=0.5)
                            ].sort_values(by=[df_dict["x-key"]])

            if ("normalizeby-key" in  df_dict) & ("normalizeby-val" in  df_dict):

                # print('sortby idx:',i)
                
                

                df_filtered_norm_fact = df_filtered.loc[
                                            np.isclose(
                                                df_filtered[df_dict["normalizeby-key"]],
                                                df_dict["normalizeby-val"],
                                                atol = 0.5
                                                    )
                                            , "1_T1_0"                                                 ]
                df_filtered_norm_fact_x = df_filtered.loc[
                                            np.isclose(
                                                df_filtered[df_dict["normalizeby-key"]],
                                                df_dict["normalizeby-val"],
                                                atol = 0.5
                                                    )
                                            , df_dict["normalizeby-key"]                                                 ]
                # print(df_filtered_norm_fact.values)
                # ax.plot(df_filtered_norm_fact_x,df_filtered_norm_fact,'ko',zorder=3)

                df_norm = df_filtered["1_T1_0"]/df_filtered_norm_fact.values
                
                ax.errorbar(
                            df_filtered[df_dict["x-key"]], 
                            df_norm,#df_filtered["1_T1_0"],
                            yerr=df_filtered["Error_ave 1_T1_0"],
                            fmt='o--',
                            label='1/T1 {}:{} Norm-to {}:{}'.format(df_dict["sortby-key"],val, df_dict["normalizeby-key"], df_dict["normalizeby-val"])
                            )
            
            else: 
                ax.errorbar(
                                df_filtered[df_dict["x-key"]], 
                                df_filtered["1_T1_0"],
                                yerr=df_filtered["Error_ave 1_T1_0"],
                                fmt='o--',
                                label='{}:{}'.format(df_dict["sortby-key"],val)
                                )

    elif "labelby-key" in df_dict:
        ax.errorbar(
                        df[df_dict["x-key"]], 
                        df["1_T1_0"],
                        yerr=df["Error_ave 1_T1_0"],
                        fmt='o--',
                        label=df[df_dict["labelby-key"]]
                    )
    else:
        ax.errorbar(df[df_dict["x-key"]], df["1_T1_0"],yerr=df["Error_ave 1_T1_0"],fmt='o--')


    ax.legend(loc='best')
    ax.set_title(str(df_label))
    ax.set_xlabel(df_dict["x-key"])
    ax.set_ylabel(r"1/T1 (sec$^{-1}$)")
    
df_scan = dict()

"""
NSFS Filter (50 - 200 mT)
"""
data_NSFS = pd.concat([
                    df1[
                            np.isclose(df1["Temperature (K)"],15,atol=1.5) & #NSFS_12keV
                            np.isclose(df1['Impl. Energy (keV)'],11, atol=1) #NSFS_12keV
                            ],
                    df2[
                        np.isclose(df2["Temperature (K)"],14,atol=1.5) & #NSFS_12keV
                        np.isclose(df2['Impl. Energy (keV)'],11.15, atol=0.75) #& #NSFS_12keV
                        ]     
                    ])

df_scan = dict()
df_scan["df-obj"] = data_NSFS
df_scan["label"] = "NSFS 50-200 mT"
df_scan["x-key"] = "B0 Field (T)"
df_scan["y-key"] = "1_T1_0"

# df_inspect(df_scan)

# """
# T-scan Filter: Fixed at 50 mT 
# """
# #### Normal State T-scan ####
# Bc2_T = 404e-3

# Tc_50mT = dac.critical_temperature2(50e-3, Bc2_T, 9.25)
# Energy_Tscan_keV = 20

# # print('Tc at 50 mT:',Tc_50mT)

# data_50mT_Normal_Tscan = df1[
#                             np.isclose(df1['Impl. Energy (keV)'],Energy_Tscan_keV, atol=1)
#                             & (df1["Temperature (K)"] > Tc_50mT)  
#                             ].sort_values(by=['Temperature (K)'])

# # df_scan = dict()
# # df_scan["df-obj"] = data_50mT_Normal_Tscan
# # df_scan["label"] = "Normal T-scan 50 mT"
# # df_scan["x-key"] = "Temperature (K)"
# # df_scan["y-key"] = "1_T1_0"

# # df_inspect(df_scan)

# #### SC State T-scan ####
# data_50mT_SC_Tscan = df1[
#                             np.isclose(df1['Impl. Energy (keV)'],Energy_Tscan_keV, atol=1)
#                             & (df1["Temperature (K)"] < Tc_50mT)  
#                             ].sort_values(by=['Temperature (K)'])

# # df_scan = dict()
# # df_scan["df-obj"] = data_50mT_SC_Tscan
# # df_scan["label"] = "SC T-scan 50 mT"
# # df_scan["x-key"] = "Temperature (K)"
# # df_scan["y-key"] = "1_T1_0"

# # df_inspect(df_scan)

# """
# All T-scans: 50 & 100 mT
# """

# """
# T-scans
# """

Bc2_T = 404e-3

# fig_Tscans, ax_Tscans = plt.subplots()

# fig_Tscans_50, ax_Tscans_50 = plt.subplots()
# fig_Tscans_100, ax_Tscans_100 = plt.subplots()



# """
# B = 50 mT
# """
Tc_50mT = dac.critical_temperature2(50e-3, Bc2_T, 9.25)
Energy_Tscan_keV_50mT = [20]
data_50mT_Tscan_20keV = df1[
                        np.isclose(df1['Impl. Energy (keV)'],20.0, atol=1)
                        # & (df1["Run Number"] >= 45175)
                        # #     or
                        # #     (df1["Run Number"] < 45167)
                        # # ) 
                        
                        # # & 

                        ].sort_values(by=['Temperature (K)'])

# Remove duplicate SCDS data from Tscan
data_50mT_Tscan_20keV = data_50mT_Tscan_20keV.drop(data_50mT_Tscan_20keV[(data_50mT_Tscan_20keV["Run Number"] < 45167) & (data_50mT_Tscan_20keV["Run Number"] >= 45162)].index)

# # print('Tc at 50 mT:',Tc_50mT)

# for erg in Energy_Tscan_keV_50mT:
#     data_50mT_Tscan = df1[
#                             np.isclose(df1['Impl. Energy (keV)'],erg, atol=1)
#                             ].sort_values(by=['Temperature (K)'])

#     df_scan = dict()
#     df_scan["df-obj"] = data_50mT_Tscan
#     df_scan["label"] = "T-scan 50 mT at {:.0f} keV".format(erg)
#     df_scan["x-key"] = "Temperature (K)"
#     df_scan["y-key"] = "1_T1_0"

#     # df_inspect(df_scan)

#     ax_Tscans.errorbar(
#                         data_50mT_Tscan["Temperature (K)"],
#                         data_50mT_Tscan["1_T1_0"], 
#                         yerr=data_50mT_Tscan["Error_ave 1_T1_0"],
#                         label='50 mT at {:.0f} keV'.format(erg)
#                         )

#     ax_Tscans_50.errorbar(
#                         data_50mT_Tscan["Temperature (K)"],
#                         data_50mT_Tscan["1_T1_0"], 
#                         yerr=data_50mT_Tscan["Error_ave 1_T1_0"],
#                         label='50 mT at {:.0f} keV'.format(erg)
#                         )

# # #### Normal State T-scan ####
# #     data_50mT_Normal_Tscan = df_50mT_Nb120[
# #                                 np.isclose(df_50mT_Nb120['Impl. Energy (keV)'],erg, atol=1)
# #                                 & (df_50mT_Nb120["Temperature (K)"] >= Tc_50mT)  
# #                                 ].sort_values(by=['Temperature (K)'])

# #     df_scan = dict()
# #     df_scan["df-obj"] = data_50mT_Normal_Tscan
# #     df_scan["label"] = "Normal T-scan 50 mT"
# #     df_scan["x-key"] = "Temperature (K)"
# #     df_scan["y-key"] = "1_T1_0"

# #     df_inspect(df_scan)

# # #### SC State T-scan ####
# #     data_50mT_SC_Tscan = df_50mT_Nb120[
# #                                 np.isclose(df_50mT_Nb120['Impl. Energy (keV)'],erg, atol=1)
# #                                 & (df_50mT_Nb120["Temperature (K)"] < Tc_50mT)  
# #                                 ].sort_values(by=['Temperature (K)'])

# #     df_scan = dict()
# #     df_scan["df-obj"] = data_50mT_SC_Tscan
# #     df_scan["label"] = "SC T-scan 50 mT"
# #     df_scan["x-key"] = "Temperature (K)"
# #     df_scan["y-key"] = "1_T1_0"

# #     df_inspect(df_scan)


# """
# B = 100 mT
# """
Tc_100mT = dac.critical_temperature2(100e-3, Bc2_T, 9.25)
Energy_Tscan_keV_100mT = [8,12,20]
data_100mT_Tscan_8keV = df3[
                                    np.isclose(df3['Impl. Energy (keV)'],8.0, atol=1)
                                    ].sort_values(by=['Temperature (K)'])

data_100mT_Tscan_12keV = df3[
                                    np.isclose(df3['Impl. Energy (keV)'],12.0, atol=1)
                                    ].sort_values(by=['Temperature (K)'])

data_100mT_Tscan_20keV = df3[
                                    np.isclose(df3['Impl. Energy (keV)'],20.0, atol=1)
                                    ].sort_values(by=['Temperature (K)'])

data_100mT_Tscan = [
                                data_100mT_Tscan_8keV,
                                data_100mT_Tscan_12keV,
                                data_100mT_Tscan_20keV,
]

# # print('Tc at 50 mT:',Tc_50mT)

# for erg in Energy_Tscan_keV_100mT:
#     data_100mT_Tscan = df3[
#                                     np.isclose(df3['Impl. Energy (keV)'],erg, atol=1)
#                                     ].sort_values(by=['Temperature (K)'])

#     df_scan = dict()
#     df_scan["df-obj"] = data_100mT_Tscan
#     df_scan["label"] = "T-scan 100 mT at {:.0f} keV".format(erg)
#     df_scan["x-key"] = "Temperature (K)"
#     df_scan["y-key"] = "1_T1_0"

#     # df_inspect(df_scan)

    
#     ax_Tscans.errorbar(
#                         data_100mT_Tscan["Temperature (K)"],
#                         data_100mT_Tscan["1_T1_0"], 
#                         yerr=data_100mT_Tscan["Error_ave 1_T1_0"],
#                         label='100 mT at {:.0f} keV'.format(erg)
#                         )

#     ax_Tscans_100.errorbar(
#                         data_100mT_Tscan["Temperature (K)"],
#                         data_100mT_Tscan["1_T1_0"], 
#                         yerr=data_100mT_Tscan["Error_ave 1_T1_0"],
#                         label='100 mT at {:.0f} keV'.format(erg)
#                         )

# ax_Tscans.legend()
# ax_Tscans.set_ylim([2.0,14.0])
# fig_Tscans.savefig('NbBaseline_50-100mT_Tscan.pdf')

# ax_Tscans_50.legend()
# ax_Tscans_50.set_ylim([2.0,14.0])
# fig_Tscans_50.savefig('NbBaseline_50mT_Tscan.pdf')

# ax_Tscans_100.legend()
# ax_Tscans_100.set_ylim([2.0,14.0])
# fig_Tscans_100.savefig('NbBaseline_100mT_Tscan.pdf')
# # #### Normal State T-scan ####
# #     data_100mT_Normal_Tscan = df_100mT_Nb120[
# #                                 np.isclose(df_100mT_Nb120['Impl. Energy (keV)'],erg, atol=1)
# #                                 & (df_100mT_Nb120["Temperature (K)"] >= Tc_100mT)  
# #                                 ].sort_values(by=['Temperature (K)'])

# #     df_scan = dict()
# #     df_scan["df-obj"] = data_100mT_Normal_Tscan
# #     df_scan["label"] = "Normal T-scan 50 mT"
# #     df_scan["x-key"] = "Temperature (K)"
# #     df_scan["y-key"] = "1_T1_0"

# #     df_inspect(df_scan)

# # #### SC State T-scan ####
# #     data_100mT_SC_Tscan = df_100mT_Nb120[
# #                                 np.isclose(df_100mT_Nb120['Impl. Energy (keV)'],erg, atol=1)
# #                                 & (df_100mT_Nb120["Temperature (K)"] < Tc_100mT)  
# #                                 ].sort_values(by=['Temperature (K)'])

# #     df_scan = dict()
# #     df_scan["df-obj"] = data_100mT_SC_Tscan
# #     df_scan["label"] = "SC T-scan 50 mT"
# #     df_scan["x-key"] = "Temperature (K)"
# #     df_scan["y-key"] = "1_T1_0"

# #     df_inspect(df_scan)


"""
#############################
NORMAL STATE DEPTH SCAN: 100 mT (14 K)
##############################
"""
# data_100mT_Normal_Depthscan = df2[
#                         np.isclose(df2["Temperature (K)"],14,atol=1.5)
#                         & np.isclose(df2["B0 Field (T)"], 0.100, rtol=0.01)  
#                     ] 
data_100mT_Normal_Depthscan = df3[
                        np.isclose(df3["Temperature (K)"],14,atol=1.5)
                        & np.isclose(df3["B0 Field (T)"], 0.100, rtol=0.01)  
                    ] 


# df_scan["df-obj"] = data_100mT_Normal_Depthscan#data_50mT_SC_Depthscan#
# df_scan["label"] = "Normal E-scan"
# df_scan["x-key"] = "Impl. Energy (keV)"
# df_scan["y-key"] = "1_T1_0"
# df_scan["sortby-key"] = "B0 Field (T)"

# df_inspect(data_100mT_Normal_Depthscan)

"""
########################################
SUPERCONDUCTING DEPTH SCANS: 24-200 mT

SC
- 50 mT: Filter T = 4.6 K
- 100-110 mT: Filter T 
########################################
"""


#####
# ENHANCE THE DEAD LAYER BY REDUCING THE UNCERTAINTY FOR 2 and 4 keV
####
# data_50mT_SC_Depthscan = data_50mT_SC_Depthscan.drop(data_50mT_SC_Depthscan[
#                         np.isclose(
#                                     data_50mT_SC_Depthscan["Impl. Energy (keV)"],
#                                     8.0, 
#                                     atol=0.5,
#                         )
#                        ].index)


# data_100_mT_SC_Depthscan_old["label"] = '2022-08'

# """
# ENHANCE DEAD LAYER
# """
# ENHANCEMENT_FACTOR_DEADLAYER = 1

# # print(data_50mT_SC_Depthscan.loc[data_50mT_SC_Depthscan["Impl. Energy (keV)"]<5.0,"Error_ave 1_T1_0"])

# data_50mT_SC_Depthscan.loc[data_50mT_SC_Depthscan["Impl. Energy (keV)"]<5,"Error_ave 1_T1_0"] = data_50mT_SC_Depthscan.loc[df_all["Impl. Energy (keV)"]<5,"Error_ave 1_T1_0"].multiply(1/ENHANCEMENT_FACTOR_DEADLAYER)

# # print(data_50mT_SC_Depthscan.loc[data_50mT_SC_Depthscan["Impl. Energy (keV)"]<5.0,"Error_ave 1_T1_0"])

# ##############

# # print(data_100_mT_SC_Depthscan_old.loc[data_100_mT_SC_Depthscan_old["Impl. Energy (keV)"]<5.0,"Error_ave 1_T1_0"])

# data_100_mT_SC_Depthscan_old.loc[data_100_mT_SC_Depthscan_old["Impl. Energy (keV)"]<5,"Error_ave 1_T1_0"] = data_100_mT_SC_Depthscan_old.loc[df_all["Impl. Energy (keV)"]<5,"Error_ave 1_T1_0"].multiply(1/ENHANCEMENT_FACTOR_DEADLAYER)

# # print(data_100_mT_SC_Depthscan_old.loc[data_100_mT_SC_Depthscan_old["Impl. Energy (keV)"]<5.0,"Error_ave 1_T1_0"])

# """
# ENHANCE ALL SC E-SCANS
# """
# ENHANCEMENT_FACTOR_MEISSNERSCANS = 1


# data_50mT_SC_Depthscan["Error_ave 1_T1_0"] = data_50mT_SC_Depthscan["Error_ave 1_T1_0"].multiply(1/ENHANCEMENT_FACTOR_MEISSNERSCANS)

# data_100_mT_SC_Depthscan_old["Error_ave 1_T1_0"] = data_100_mT_SC_Depthscan_old["Error_ave 1_T1_0"].multiply(1/ENHANCEMENT_FACTOR_MEISSNERSCANS)


###########################################################################


data_50mT_SC_Depthscan = df1[
                            np.isclose(df1["Temperature (K)"],4.5,rtol=0.01)
                        ].sort_values(by=['Impl. Energy (keV)'])

data_100_mT_SC_Depthscan = df2[
                                 np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
                                & np.isclose(df2["B0 Field (T)"], 0.100, rtol=0.01) #& #SCDS < 125 mT
                                # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
                                ]

data_100_mT_SC_Depthscan_old = df3[np.isclose(df3["Temperature (K)"],4.6,atol=0.5)].sort_values(by=['Impl. Energy (keV)'])

data_110_mT_SC_Depthscan = df2[
                                 np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
                                & np.isclose(df2["B0 Field (T)"], 0.110, rtol=0.01) #& #SCDS < 125 mT
                                # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
                                ]

data_125_mT_SC_Depthscan = df2[
                                 np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
                                & np.isclose(df2["B0 Field (T)"], 0.125, rtol=0.01) #& #SCDS < 125 mT
                                # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
                                ]

data_150_mT_SC_Depthscan = df2[
                                 np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
                                & np.isclose(df2["B0 Field (T)"], 0.150, rtol=0.01) #& #SCDS < 125 mT
                                # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
                                ]


data_200_mT_SC_Depthscan = df2[
                                 np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
                                & np.isclose(df2["B0 Field (T)"], 0.20, rtol=0.01) #& #SCDS < 125 mT
                                # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
                                ]



# data_100_110_mT_SC_Depthscan = df2[
#                                  np.isclose(df2["Temperature (K)"],4.6,atol=0.5)
#                                 & (df2["B0 Field (T)"] < 0.120) #& #SCDS < 125 mT
#                                 # & np.isclose(df2["B0 Field (T)"], 0.11, rtol=0.01)
#                                 ].sort_values(by=[
#                                                     'B0 Field (T)',
#                                                     # 'Impl. Energy (keV)'
#                                                     ])

data_100_110_mT_SC_Depthscan = pd.concat([
  data_100_mT_SC_Depthscan,
#   data_100_mT_SC_Depthscan_old,
  data_110_mT_SC_Depthscan,  
])
                

# data_100_110_mT_SC_Depthscan = data_100_110_mT_SC_Depthscan[
#                                                             np.isclose(data_100_110_mT_SC_Depthscan["Temperature (K)"],4.6,atol=0.5)
#                                                         ].sort_values(by=['B0 Field (T)','Impl. Energy (keV)'])


##### > 10 keV ########
data_150_mT_SC_Depthscan_gt10keV = data_150_mT_SC_Depthscan[
    data_150_mT_SC_Depthscan["Impl. Energy (keV)"]>10.0
]

data_200_mT_SC_Depthscan_gt10keV = data_200_mT_SC_Depthscan[
    data_200_mT_SC_Depthscan["Impl. Energy (keV)"]>10.0
]

##### 12 keV only ######
data_200_mT_SC_Depthscan_12keV = data_200_mT_SC_Depthscan[
    np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],12.0,atol=0.5)
]

data_150_mT_SC_Depthscan_12keV = data_150_mT_SC_Depthscan[
    np.isclose(data_150_mT_SC_Depthscan["Impl. Energy (keV)"],12.0,atol=0.5)
]

######
data_50_to_200mT_SC_Depthscan = pd.concat([
                                            data_50mT_SC_Depthscan,
                                            data_100_mT_SC_Depthscan,
                                            # data_100_mT_SC_Depthscan_old,
                                            data_110_mT_SC_Depthscan,   
                                            # data_100_110_mT_SC_Depthscan,
                                            data_125_mT_SC_Depthscan,
                                            data_150_mT_SC_Depthscan,
                                            data_200_mT_SC_Depthscan,
                                                ])

data_24_to_200mT_SC_Depthscan = pd.concat([
                                                data_24mT_SC_Depthscan,
                                                data_50_to_200mT_SC_Depthscan,
                                            ])

df_scan["df-obj"] = data_24_to_200mT_SC_Depthscan#data_50mT_SC_Depthscan#
df_scan["label"] = "SC E-scan"
df_scan["x-key"] = "Impl. Energy (keV)"
df_scan["y-key"] = "1_T1_0"
df_scan["sortby-key"] = "B0 Field (T)"
# df_scan["labelby-key"] = "label"


# # # uncomment below to normalize multi-field E scans to 4keV 1/T1 value
# # df_scan["normalizeby-val"] = 4.0
# # df_scan["normalizeby-key"] = "Impl. Energy (keV)"

# df_inspect(df_scan)

# # # print('50mT NSTS:',data_50mT_Normal_Tscan[["Temperature (K)","Run Number","Impl. Energy (keV)"]])


# # # print('NSFS:',data_NSFS[["B0 Field (T)","Run Number","Impl. Energy (keV)"]])


"""
##############################################
SC Field-scan Filter: B = 50-110 mT, E = 4 keV.
T-dep params = slope_s_K
##############################################
"""

data_SCFS_4keV = pd.concat([
                        # data_50mT_SC_Depthscan,
                        # data_100_mT_SC_Depthscan,
                        data_100_mT_SC_Depthscan_old,
                        data_110_mT_SC_Depthscan,
                        # data_125_mT_SC_Depthscan,
                        # np.average(
                                    data_150_mT_SC_Depthscan["1_T1_0"],
                        #             weights = data_150_mT_SC_Depthscan["Error_ave 1_T1_0"].pow(-2)),
                        # np.average(
                                    data_200_mT_SC_Depthscan["1_T1_0"],
                                    # weights = data_200_mT_SC_Depthscan["Error_ave 1_T1_0"].pow(-2)),
                   ])

data_SCFS_4keV = data_SCFS_4keV[
                                    np.isclose(
                                        data_SCFS_4keV["Impl. Energy (keV)"],
                                        4.0,
                                        atol = 0.5
                                    )
                                ]

df_SC_NS_FieldScans = pd.concat([
                        data_SCFS_4keV,
                        data_NSFS,
                   ])


"""
DATASET DEFINITIONS
"""



#  Cost (master): All(100,110,125,150,200)
data_df_B_plot = [
                    data_100_mT_SC_Depthscan,
                    data_110_mT_SC_Depthscan,
                    data_125_mT_SC_Depthscan, #not working
                    data_150_mT_SC_Depthscan.loc[
                                                    data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
                    ], #weird
                    data_200_mT_SC_Depthscan.loc[
                                                    data_200_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
                    ], #norm value is crazy
                ]

"""
##########################
WORKING COST FUNCTION
######################
"""
# # Cost (master): All(100,110,125,150,200). exclude (slave): None
# cost_glob_1 = data_df_B_plot

# # Cost (master): All(100,110,125,150) + 8keV(200). exclude (slave): None
# cost_glob_2 = [
#                     data_100_mT_SC_Depthscan,
#                     data_110_mT_SC_Depthscan,
#                     data_125_mT_SC_Depthscan, #not working
#                     data_150_mT_SC_Depthscan.loc[
#                                                     data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
#                     ], #weird
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
#                                             ],
#                 ]

# # Cost (master): All(100,110,125,150) + 8,12 keV(200). exclude (slave): None
# cost_glob_3 = [
#                     data_100_mT_SC_Depthscan,
#                     data_110_mT_SC_Depthscan,
#                     data_125_mT_SC_Depthscan, #not working
#                     data_150_mT_SC_Depthscan.loc[
#                                                     data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
#                     ], #weird
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
#                                             ],
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],12.0,atol=0.5)
#                                             ],
#                 ]

"""
##########################
NOT WORKING COST FUNCTION
#########################
"""
# # Cost (master): All(100,110,125,150) + 20 keV(200). exclude (slave): None
# cost_glob_4 = [
#                     data_100_mT_SC_Depthscan,
#                     data_110_mT_SC_Depthscan,
#                     data_125_mT_SC_Depthscan, #not working
#                     data_150_mT_SC_Depthscan.loc[
#                                                     data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
#                     ], #weird
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],20.0,atol=0.5)
#                                             ],
#                 ]

# # # Cost (master): All(100,110,125,150) + 20,12 keV(200). exclude (slave): None
# cost_glob_5 = [
#                     data_100_mT_SC_Depthscan,
#                     data_110_mT_SC_Depthscan,
#                     data_125_mT_SC_Depthscan, #not working
#                     data_150_mT_SC_Depthscan.loc[
#                                                     data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
#                     ], #weird
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],20.0,atol=0.5)
#                                             ],
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],12.0,atol=0.5)
#                                             ],
#                 ]

# """
# #########################################
# GLOBAL PARALLEL FIT
# #########################################
# """

alldata = pd.concat(cost_glob_5)
# alldata = pd.concat(data_df_B_plot)
# alldata = pd.concat(
#                 [
#                     data_100_mT_SC_Depthscan, #best #minimal for d,LPD

#                     data_110_mT_SC_Depthscan[
#                 np.isclose(data_110_mT_SC_Depthscan["Impl. Energy (keV)"],4.0,atol=0.5)
#                                             ],

#                     data_125_mT_SC_Depthscan[
#                 np.isclose(data_125_mT_SC_Depthscan["Impl. Energy (keV)"],4.0,atol=0.5)
#                                             ],
                    
#                     data_150_mT_SC_Depthscan[
#                 np.isclose(data_150_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
#                                             ],
                    
#                     data_200_mT_SC_Depthscan[
#                 np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
#                                             ],
#                 ]
#                 )

# print(alldata['Run Number'])

global_chi2 = LeastSquares3D(
    model=fcn3d,
    x=alldata["Temperature (K)"],
    y=alldata["Impl. Energy (keV)"],
    z=alldata["B0 Field (T)"],
    value=alldata['1_T1_0'],
    error=alldata[['Error_ave 1_T1_0']],
    verbose=True,
)




"""
Load From json
"""

# with open("Bobs_ET-ReFit_CSTDemag_Bob-SCDS_Global.json", "r") as file_handle:
# with open("Bobs_ET-ReFit_CSTDemag_ET-SCDS_Global.json", "r") as file_handle:
# with open("BobCompare_ETCSTVal_GlobFits_StandAlone_GlobalFits_cost-glob3(200-8,12keV).json", "r") as file_handle:
with open("BobCompare_ETCSTVal_GlobFits_StandAlone_GlobalFits_cost-glob5.json", "r") as file_handle:

    old_results = json.load(file_handle)

# initial_params = dict()
# initial_params["dead_layer_nm"] = 22.6

# initial_params["lambda_L_100mT_nm"] = 40.1
# initial_params["lambda_L_110mT_nm"] = 58.9
# initial_params["lambda_L_125mT_nm"] = 174.2
# initial_params["lambda_L_150mT_nm"] = 226.3
# initial_params["lambda_L_200mT_nm"] = 1559.0

# initial_params["a"] = 59000
# initial_params["b"] = 11700

# initial_params["critical_temperature_K"] = 9.25
# initial_params["critical_field_2_T"] = 410.0

# initial_params["enhance_fact_100mT"] = 1/0.93
# initial_params["enhance_fact_110mT"] = 1/0.93
# initial_params["enhance_fact_125mT"] = 1/0.998
# initial_params["enhance_fact_150mT"] = 1.0
# initial_params["enhance_fact_200mT"] = 1.0

# # initial_params["B_noscreen_T"] = 140e-3
# # initial_params["enhance_fact_mixed"] = 1/0.98
# # initial_params["B_mixed_T"] = 115e-3



# m = Minuit(
#     global_chi2,
#     **initial_params,# **old_fit_results["values"],
# )
m = Minuit(global_chi2, **old_results["values"])

# json2minuit(m, "Bobs_ET-ReFit_CSTDemag_Bob-SCDS_Global.json")
# json2minuit(m, "Bobs_ET-ReFit_CSTDemag_ET-SCDS_Global.json")
json2minuit(m, "BobCompare_ETCSTVal_GlobFits_StandAlone_GlobalFits_cost-glob5.json")



"""
Set value manually: Only for the very first step of fit, afterwards only load via json2minuit
# """
# m.values["critical_temperature_K"] = 9.25
# m.values["critical_field_2_T"] = 410.0
# m.values["B_noscreen_T"] = 0.14
# m.values["enhance_fact_mixed"] = 1/0.98
# m.values["B_mixed_T"] = 115e-3



# m.values["enhance_fact_100mT"] = 1.081 #1/0.93
# m.values["enhance_fact_110mT"] = 1.081#1/0.93
# m.values["enhance_fact_125mT"] = 1.0
# m.values["enhance_fact_150mT"] = 1.0
# m.values["enhance_fact_200mT"] = 1.0

# m.values["dead_layer_nm"] = 23.494104399699008 #22.6#23#21.9

# m.values["lambda_L_100mT_nm"] = 37.52809507189834#40.1
# m.values["lambda_L_110mT_nm"] = 55.172643834316695#58.9
# m.values["lambda_L_125mT_nm"] = 157.80569887876345#174.2
# m.values["lambda_L_150mT_nm"] = 212.42537232198242#226.3

# m.values["lambda_L_200mT_nm"] = np.inf

# m.values["a"] = 59924 #59000
# m.values["b"] = 11661 #11700.0

"""
ALWAYS FIX
"""

m.fixed["critical_temperature_K"] = True
m.fixed["critical_field_2_T"] = True

# m.fixed["B_noscreen_T"] = True
# m.fixed["enhance_fact_mixed"] = True
# m.fixed["B_mixed_T"] = True

m.fixed["lambda_L_200mT_nm"] = True
m.fixed["enhance_fact_100mT"] = True

"""
FIX = VARY
"""

m.fixed["enhance_fact_200mT"] = False
m.fixed["lambda_L_100mT_nm"] = False

m.fixed["enhance_fact_150mT"] = False
m.fixed["enhance_fact_125mT"] = False
m.fixed["enhance_fact_110mT"] = False

m.fixed["dead_layer_nm"] = False
m.fixed["lambda_L_110mT_nm"] = False
m.fixed["lambda_L_125mT_nm"] = False
m.fixed["lambda_L_150mT_nm"] = False

m.fixed["a"] = False
m.fixed["b"] = False


"""
Set limit manually
"""

m.limits["critical_temperature_K"] = (0.0,15.0)
m.limits["critical_field_2_T"] = (0.0,None)

m.limits["enhance_fact_100mT"] = (1.0,1.081)
m.limits["enhance_fact_110mT"] = (1.000,1.081)
m.limits["enhance_fact_125mT"] = (1.000,1.081)
m.limits["enhance_fact_150mT"] = (1.000,1.081)
m.limits["enhance_fact_200mT"] = (1.000,1.081)

m.limits["dead_layer_nm"] = (0.0,None)

m.limits["lambda_L_100mT_nm"] = (0.0,None)
m.limits["lambda_L_110mT_nm"] = (0.0,None)
m.limits["lambda_L_125mT_nm"] = (0.0,None)
m.limits["lambda_L_150mT_nm"] = (0.0,None)
m.limits["lambda_L_200mT_nm"] = (0.0,None)

m.limits["a"] = (0.0,None)
m.limits["b"] = (0.0,None)

# m.limits["B_noscreen_T"] = (0.0,None)
# m.limits["enhance_fact_mixed"] = (0.0,2.0)
# m.limits["B_mixed_T"] = (0.0,None)

FIT_THE_DATA = False

if FIT_THE_DATA:
    m.migrad()
    # m.hesse()
    # m.minos()

print(m)

minuit2json(m, "BobCompare_ETCSTVal_GlobFits_StandAlone_GlobalFits_cost-glob5.json")

"""
#########################################
!!!! SLICE AND PLOT DATA + FIT !!!!
#########################################
"""

E_EXTRAPOLATE_KEV = 2.0

# """
# #####################################
# Energy Scans: using mean(field, temp)
# #####################################
# """

fig_SCDS_Escan, ax_SCDS_Escan = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=set_size(),#(10, 8),
    constrained_layout=True,
) 

Temp_SCDS_Escan = np.array([4.5])

data_SCDS_plot = data_df_B_plot
                # [
                #     data_100_mT_SC_Depthscan, #best #minimal for d,LPD
                #     data_110_mT_SC_Depthscan,
                #     data_125_mT_SC_Depthscan,
                #     data_150_mT_SC_Depthscan,
                #     data_200_mT_SC_Depthscan,

                # #     data_110_mT_SC_Depthscan[
                # # np.isclose(data_110_mT_SC_Depthscan["Impl. Energy (keV)"],4.0,atol=0.5)
                # #                             ],

                # #     data_125_mT_SC_Depthscan[
                # # np.isclose(data_125_mT_SC_Depthscan["Impl. Energy (keV)"],4.0,atol=0.5)
                # #                             ],
                    
                # #     data_150_mT_SC_Depthscan[
                # # np.isclose(data_150_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
                # #                             ],
                    
                # #     data_200_mT_SC_Depthscan[
                # # np.isclose(data_200_mT_SC_Depthscan["Impl. Energy (keV)"],8.0,atol=0.5)
                # #                             ],

                # ]

c_Escan = [
            'blue',
            'brown',
            'grey',
            'orange',
            'darkturquoise',
            ]
#cm.rainbow(np.linspace(0,1,len(data_SCDS_plot)))#['r','b','k','purple']#


regr = linear_model.LinearRegression()

for i, data_SCDS in enumerate(data_SCDS_plot):
    data_SCDS = data_SCDS.sort_values(by="Impl. Energy (keV)")
    mean_temperature_K_SCDS = data_SCDS["Temperature (K)"].mean()
    mean_field_T_SCDS = data_SCDS["B0 Field (T)"].mean()

    energies_keV_SCDS = np.linspace(
        np.min(data_SCDS["Impl. Energy (keV)"]),
        np.max(data_SCDS["Impl. Energy (keV)"]),
        num=N_FIT_POINTS,
    )
    # np.linspace(
    #     np.min(data_50mT_SC_Depthscan["Impl. Energy (keV)"]) - 1,
    #     np.max(data_50mT_SC_Depthscan["Impl. Energy (keV)"]) + 1,
    #     num=N_FIT_POINTS,
    # )

    ax_SCDS_Escan.errorbar(
        data_SCDS["Impl. Energy (keV)"],
        data_SCDS['1_T1_0'],
        # xerr=data_Escan["Energy Error (keV)"],
        yerr=data_SCDS['Error_ave 1_T1_0'],
        fmt="o",
        zorder=2,
        c= c_Escan[i],
        ecolor=lighten_color(c_Escan[i],0.4),
        # label='Data ({:.1f} mT)'.format(mean_field_T_SCDS*1e3)
        label='{:.0f} mT'.format(np.round(mean_field_T_SCDS*1e3),decimals=0),
        markersize=3.0,
    )

    """
    Plot Extrapolated 1/T1 vs E from fit parameters
    """    
    # par_SCDS = [
    #     mean_field_T_SCDS,
    #     m.values["dead_layer_nm"],
    #     m.values["lambda_L_100mT_nm"],
    #     m.values["lambda_L_110mT_nm"],
    #     m.values["lambda_L_125mT_nm"],
    #     m.values["lambda_L_150mT_nm"],
    #     m.values["lambda_L_200mT_nm"],
    #     m.values["a"],
    #     m.values["b"],
    #     mean_temperature_K_SCDS,
    #     m.values["critical_temperature_K"],
    #     m.values["critical_field_2_T"],
    #     m.values["enhance_fact_100mT"],
    #     m.values["enhance_fact_110mT"],
    #     m.values["enhance_fact_125mT"],
    #     m.values["enhance_fact_150mT"],
    #     m.values["enhance_fact_200mT"],
    #     # m.values["B_noscreen_T"],
    #     # m.values["enhance_fact_mixed"],
    #     # m.values["B_mixed_T"],
    # ]

    # ax_SCDS_Escan.plot(
    #     energies_keV_SCDS,

    #     dac(energies_keV_SCDS, *par_SCDS),
    #     "-",
    #     zorder=1,
    #     color = c_Escan[i],
    #     # label='SCDS Fit at {:.1f} mT'.format(mean_field_T_SCDS*1e3)
    #     # label='Fit ({:.0f} mT)'.format(mean_field_T_SCDS*1e3)
    # )

    """
    Plot Linear Regression Instead
    """
    nrow =  data_SCDS["Impl. Energy (keV)"].shape[0]

    regr.fit(
                data_SCDS["Impl. Energy (keV)"].values.reshape(nrow,1),
                data_SCDS['1_T1_0'].values.reshape(nrow,1)
            )
    
    # y_reg = regr.predict(energies_keV_SCDS.reshape(energies_keV_SCDS.size,1))
    # print(np.shape(energies_keV_SCDS), np.shape(y_reg))
    
    ax_SCDS_Escan.plot(
        energies_keV_SCDS,
        regr.predict(
                        energies_keV_SCDS.reshape(
                                    energies_keV_SCDS.size,1
                                    )
                    ),
        "--",
        zorder=1,
        color = c_Escan[i],
    )

ax_SCDS_Escan.legend(
                    title=r'$B_{app}$[mT]:',
                    bbox_to_anchor=(1.0,1.0),
                    # loc='best'
                    )

# ax_SCDS_Escan.set_xlabel('Impl. Energy [keV]', fontsize=12)
# ax_SCDS_Escan.set_ylabel(r'$\frac{1}{T_1} [s^{-1}]$', fontsize=12)

ax_SCDS_Escan.set_xlabel('Impl. Energy [keV]')#, fontsize=12)
ax_SCDS_Escan.set_ylabel(r'$1/T_1$ [s$^{-1}$]')#, fontsize=12)


# fig_SCDS_Escan.savefig('NbBase_RelaxRate-vs-E.pdf',dpi=600)
# fig_SCDS_Escan.savefig('NbBase_RelaxRate-vs-E_Flat200_GlobFits.pdf',dpi=600)















# # # """
# # # ###########################################
# # # ## Plot B(z) from London Model and <B>(E)
# # # ########################################
# # # """

# """
# Averaging Parameter
# """

# E_fit_arr = np.linspace(E_min_keV_fit, E_max_keV_fit, N_FIT_POINTS)
# z_max = dac.z_max_2(E_max_keV_fit)

# x_GL_nm_arr, w_GL_arr = GL_Quad(N_FIT_POINTS, 0.0, x_max_nm_plot)#z_max) 


# # data_df_B_plot = [
# #                     data_100_mT_SC_Depthscan,
# #                     data_110_mT_SC_Depthscan,
# #                     data_125_mT_SC_Depthscan, #not working
# #                     data_150_mT_SC_Depthscan.loc[
# #                                                     data_150_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
# #                     ], #weird
# #                     data_200_mT_SC_Depthscan.loc[
# #                                                     data_200_mT_SC_Depthscan["Impl. Energy (keV)"] > energy_keV_cutoff_vortex
# #                     ], #norm value is crazy
# #                 ]

# field_mT_B_plot = np.array([
#                     100,
#                     110,
#                     125,
#                     150,
#                     200,
#                 ])

# c_B_plot = [
#             'blue',
#             'brown',
#             'grey',
#             'orange',
#             'darkturquoise',                             
# ]

# """
# ######################################
# ## Plot B(z) from London Model from fit params
# ######################################
# """

# fig_Bz_London, ax_Bz_London = plt.subplots(
#                                     nrows=1,
#                                     ncols=1,
#                                     figsize=set_size(
#                                                         # width='thesis',   
#                                                         subplots=(1,1),
#                                                         ),
#                                     constrained_layout=True,
#                                     # sharex='col',
#                                 )

# fig_BaveVsE, ax_BaveVsE = plt.subplots(
#                                     nrows=1,
#                                     ncols=1,
#                                     figsize=set_size(
#                                                         # width='thesis',   
#                                                         subplots=(1,1),
#                                                         ),
#                                     constrained_layout=True,
#                                     # sharex='col',
#                                 )

# # fig_RelaxVsBave, ax_RelaxVsBave = plt.subplots(
# #                                     nrows=1,
# #                                     ncols=1,
# #                                     figsize=set_size(
# #                                                         # width='thesis',   
# #                                                         subplots=(1,1),
# #                                                         ),
# #                                     constrained_layout=True,
# #                                     # sharex='col',
# #                                 )

# # ax_Lorentz_recheck, ax_BaveVsE = ax_BaveVsE

# # fig_Lorentz_recheck, ax_Lorentz_recheck = plt.subplots(
# #                                     figsize=set_size(),
# #                                     constrained_layout=True,
# #                                 )


# B_z_London_T_all = []
# B_ave_B_arr = []
# A_minuit_param = []
# B_minuit_param = []

# for idx_b, data_df in enumerate(data_df_B_plot): 

#     mean_temperature_K_SCDS = data_df["Temperature (K)"].mean()
#     mean_field_T_SCDS = data_df["B0 Field (T)"].mean()

#     mean_depth_nm = []

#     """
#     Load json files for each SCDS
#     """
#     global_chi2_indiv = LeastSquares3D(
#         model=fcn3d,
#         x=data_df["Temperature (K)"],
#         y=data_df["Impl. Energy (keV)"],
#         z=data_df["B0 Field (T)"],
#         value=data_df['1_T1_0'],
#         error=data_df[['Error_ave 1_T1_0']],
#         verbose=True,
#     )

#     # with open("Bobs_ET-ReFit_CSTDemag_Bob-SCDS_{:.0f}mT.json".format(field_mT_B_plot[idx_b]), "r") as file_handle:
#     with open("Bobs_ET-ReFit_CSTDemag_ET-SCDS_Global.json", "r") as file_handle:
#         old_results = json.load(file_handle)

#     m = Minuit(global_chi2_indiv, **old_results["values"])

#     # json2minuit(m, "Bobs_ET-ReFit_CSTDemag_Bob-SCDS_{:.0f}mT.json".format(field_mT_B_plot[idx_b]))
#     json2minuit(m, "Bobs_ET-ReFit_CSTDemag_ET-SCDS_Global.json")


#     A_minuit_param.append(m.values['a'])
#     B_minuit_param.append(m.values['b'])


#     """
#     Recreate B(z) London from fit parameters
#     """
#     B_z_London_mT = dac.london_model(
#                                             x_GL_nm_arr,
#                                             # field_mT_B_plot[idx_b]*m.values["enhance_fact"],# convert to Tesla for B-averaging
#                                             field_mT_B_plot[idx_b]*m.values["enhance_fact_%imT" %(field_mT_B_plot[idx_b])],# convert to Tesla for B-averaging

#                                             m.values["dead_layer_nm"],
#                                             m.values["lambda_L_%imT_nm" %(field_mT_B_plot[idx_b])],
#                                         )

#     B_z_London_T_all.append(B_z_London_mT)

#     ax_Bz_London.plot(
#                     x_GL_nm_arr, 
#                     B_z_London_mT,
#                     "-",
#                     zorder=1,
#                     color = c_B_plot[idx_b],
#                     label='{:.0f}'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
#                     )
#     ax_Bz_London.axvspan(
#                             xmin=0,
#                             xmax=m.values["dead_layer_nm"],
#                             color='grey',
#                             alpha=0.1,
#                             )
    

#     """
#     #############################
#     Plot B-ave vs E
#     #############################
#     """

#     # B_data_ave_mT = []
#     B_data_ave_mT = np.sqrt((m.values["a"]/data_df["1_T1_0"]) - m.values["b"])
#     B_fit_ave_mT = np.empty(E_fit_arr.size)

#     #### for plotting B-ave vs E ####
#     for idx_e,e_keV in enumerate(E_fit_arr):
#         def B_x_rho(z: float) -> float:
#             # B = np.full_like(z,B_z_London_mT[0])
#             B = dac.london_model(
#                                             z,
#                                             field_mT_B_plot[idx_b]*m.values["enhance_fact_%imT" %field_mT_B_plot[idx_b]],# convert to Tesla for B-averaging
#                                             m.values["dead_layer_nm"],
#                                             m.values["lambda_L_%imT_nm" %field_mT_B_plot[idx_b]],
#                                         )

#             rho = dac.stopping_distribution_e(z, e_keV)
#             return B * rho

#         # do the numeric integration using adaptive Gaussian quadrature
#         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
#         # B_fit_ave_mT[idx_e] = 
#         B_ave = integrate.quad(
#             B_x_rho,
#             0.0,  # lower integration limit
#             max(  # upper integration limit
#                 np.max(dac.z_max_1(e_keV)), np.max(dac.z_max_2(e_keV))
#             ),
#             epsabs=np.sqrt(np.finfo(float).eps),  # absolute error tolerance
#             epsrel=np.sqrt(np.finfo(float).eps),  # relative error tolerance
#             limit=np.iinfo(np.int32).max,  # maximum number of subintervals
#             points=[  # potential singularities/discontinuities in the integrand
#                 0.0,  #
#                 5.0,  # 5 nm Nb2O5 surface oxide layer
#                 dac.z_max_1(e_keV),
#                 dac.z_max_2(e_keV),
#                 m.values["dead_layer_nm"],
#             ],
#         )
#         B_fit_ave_mT[idx_e] = B_ave[0]
        
#     ax_BaveVsE.plot(
#                     data_df["Impl. Energy (keV)"],
#                     B_data_ave_mT,
#                     'o',
#                     color = c_B_plot[idx_b],
#                     markersize=3.0,
#                     label='{:.0f}'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
#                     )    
    
#     ax_BaveVsE.plot(
#                     E_fit_arr,
#                     B_fit_ave_mT,
#                     '-',
#                     color = c_B_plot[idx_b],
#                     # label='{:.0f} mT'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
#                     )    
    
#     ax_BaveVsE.axvspan(
#                             xmin=0,
#                             xmax=range_to_erg(m.values["dead_layer_nm"]),
#                             color='grey',
#                             alpha=0.1,
#                             )
    
#     """
#     ########################################
#     #### Plot 1/T1 vs B-ave ####
#     ########################################
#     """
#     # E_data_arr = data_df["Impl. Energy (keV)"]#np.sort(pd.unique(data_df["Impl. Energy (keV)"].round(decimals=0)))
#     # B_minuit_ave_mT = np.empty(E_data_arr.size)

#     # for idx_e,e_keV in enumerate(E_data_arr):
#     #     ### Calculate B-ave ###
#     #     def B_x_rho(z: float) -> float:
#     #         # B = np.full_like(z,B_z_London_mT[0])
#     #         B = dac.london_model(
#     #                                         z,
#     #                                         field_mT_B_plot[idx_b]*m.values["enhance_fact"],# convert to Tesla for B-averaging
#     #                                         m.values["dead_layer_nm"],
#     #                                         m.values["lambda_L_nm"],
#     #                                     )

#     #         rho = dac.stopping_distribution_e(z, e_keV)
#     #         return B * rho

#     #     # do the numeric integration using adaptive Gaussian quadrature
#     #     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html
#     #     # B_fit_ave_mT[idx_e] = 
#     #     B_ave = integrate.quad(
#     #         B_x_rho,
#     #         0.0,  # lower integration limit
#     #         max(  # upper integration limit
#     #             np.max(dac.z_max_1(e_keV)), np.max(dac.z_max_2(e_keV))
#     #         ),
#     #         epsabs=np.sqrt(np.finfo(float).eps),  # absolute error tolerance
#     #         epsrel=np.sqrt(np.finfo(float).eps),  # relative error tolerance
#     #         limit=np.iinfo(np.int32).max,  # maximum number of subintervals
#     #         points=[  # potential singularities/discontinuities in the integrand
#     #             0.0,  #
#     #             5.0,  # 5 nm Nb2O5 surface oxide layer
#     #             dac.z_max_1(e_keV),
#     #             dac.z_max_2(e_keV),
#     #             m.values["dead_layer_nm"],
#     #         ],
#     #     )
#     #     B_minuit_ave_mT[idx_e] = B_ave[0]

#     # # print(
#     # #         'size check:',
#     # #       np.shape(data_df['1_T1_0']), 
#     # #       np.shape(B_minuit_ave_mT),
#     # #       np.shape(E_data_arr),
#     # #       )

#     # #### Plot 1/T1 vs E from 'a' and 'b'
#     # ax_RelaxVsBave.errorbar(
#     #                             B_minuit_ave_mT,
#     #                             data_df['1_T1_0'],
#     #                             # xerr=data_Escan["Energy Error (keV)"],
#     #                             yerr=data_df['Error_ave 1_T1_0'],
#     #                             fmt="o",
#     #                             zorder=2,
#     #                             c= c_B_plot[idx_b],
#     #                             ecolor=lighten_color(c_B_plot[idx_b],0.4),
#     #                             # label='Data ({:.1f} mT)'.format(mean_field_T_SCDS*1e3)
#     #                             label='{:.0f} mT'.format(np.round(mean_field_T_SCDS*1e3),decimals=0),
#     #                             markersize=3.0,
#     #                         )

    
    

    
#     """
#     Backpropagate obtained B_ave from int{rho*B_London}dz to Lorentzian
#     Don't forget B in Tesla
#     """
#     # lor_recheck = dac.simpler_lorentzian(
#     #                                         B_fit_ave_mT*1e-3,
#     #                                         m.values["a"],
#     #                                         m.values["b"]
#     #                                     )
#     # print('dim B_ave:{}, dim lor_from_B_ave:{}'.format(
#     #                                                     np.size(B_fit_ave_mT),np.size(lor_recheck)))

    
#     # ax_Lorentz_recheck.errorbar(
#     #                 data_df["Impl. Energy (keV)"],
#     #                 data_df["1_T1_0"],
#     #                 yerr=data_df["Error_ave 1_T1_0"],
#     #                 fmt='o',
#     #                 color = c_B_plot[idx_b],
#     #                 markersize=3.0,
#     #                 label='{:.0f} mT'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
#     #                 )   
    
#     # ax_Lorentz_recheck.plot(
#     #                 E_fit_arr,
#     #                 lor_recheck,
#     #                 '-',
#     #                 color = c_B_plot[idx_b],
#     #                 label='{:.0f} mT'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
#     #                 )    
    

# # Plot <B> vs E from data points + minuit param
# # B_ave_fit_arr = np.linspace(0,250,100)
# # lor_vs_Bave = dac.simpler_lorentzian(
# #                                         B_ave_fit_arr*1e-3,
# #                                         A_minuit_param[0],#m.values["a"],
# #                                         B_minuit_param[0],#m.values["b"]
# #                                     )

# # ax_RelaxVsBave.plot(
# #                 B_ave_fit_arr,
# #                 lor_vs_Bave,
# #                 '-',
# #                 color = 'k',#c_B_plot[idx_b],
# #                 # label='{:.0f} mT'.format(np.round(field_mT_B_plot[idx_b]),decimals=0),
# #                 )    

# """
# ###########
# PLOT B-AVE
# ############
# """
# ax_BaveVsE.legend( title=r'$B_{app}$[mT]:',
#                bbox_to_anchor=(1.0,1.0))

# # ax_Lorentz_recheck.set_ylabel(r'$1/T_1 [s^{-1}]$')

# ax_BaveVsE.set_xlabel('Impl. Energy [keV]')
# ax_BaveVsE.set_xlim([E_min_keV_plot, E_max_keV_plot])
# ax_BaveVsE.set_ylim([B_min_mT_plot, B_max_mT_plot])

# ax_BaveVsE.set_ylabel(r'$\langle B \rangle$ [mT]')
# # fig_BaveVsE.suptitle('Average Analysis for B_London')

# # fig_BaveVsE.savefig('NbBase_BAve-vs-E.pdf',dpi=600)
# fig_BaveVsE.savefig('NbBase_BAve-vs-E_wDead.pdf',dpi=600)


# """
# ##################
# PLOT 1/T1 vs B-AVE
# ##################
# """
# # ax_RelaxVsBave.legend( title=r'$B_{app}$[mT]:',
# #                bbox_to_anchor=(1.0,1.0))

# # # ax_Lorentz_recheck.set_ylabel(r'$1/T_1 [s^{-1}]$')

# # # ax_RelaxVsBave.set_xlabel('Impl. Energy [keV]')
# # # ax_RelaxVsBave.set_ylim([])

# # ax_RelaxVsBave.set_xlim([B_min_mT_plot, B_max_mT_plot])
# # ax_RelaxVsBave.set_xlabel(r'$\langle B \rangle [mT]$')
# # ax_RelaxVsBave.set_ylabel(r'$1/T_1 [s^{-1}]$')


# # # fig_BaveVsE.suptitle('Average Analysis for B_London')

# # # fig_BaveVsE.savefig('B_Average_analysis_fixed.pdf')

    
# """
# #############
# PLOT B-LONDON
# ############
# """
# B_z_London_T_all=np.array(B_z_London_T_all)

# ax_Bz_London.legend(
#                                     title=r'$B_{app}$[mT]:',
#                                     bbox_to_anchor=(1.0,1.0),
#                                     # loc='best'
#                                     )

# ax_Bz_London.set_xlim([x_min_nm_plot,x_max_nm_plot])
# ax_Bz_London.set_ylim([B_min_mT_plot,B_max_mT_plot])#210.0])

# ax_Bz_London.set_xlabel('x [nm]')#, fontsize=12)
# ax_Bz_London.set_ylabel('B(x) [mT]')#, fontsize=12)

# # fig_Bz_London.savefig('NbBase_BLondon_vs_x.pdf',dpi=600)

plt.show(block=True)