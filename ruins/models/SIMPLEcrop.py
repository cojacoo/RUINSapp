# SIMPLEcrop in python 3
# RUINS Project (cc) c.jackisch@tu-braunschweig.de

# This is a python implementation based on Zhao et al. 2019 and other crop, soilwater and evapotranspiration models
# The intention is a to port the SIMPLE crop model to a fast and modular modelling system
# Zhao, C. et al. (2019), A SIMPLE crop model, EUROPEAN JOURNAL OF AGRONOMY, 104, 97–106, doi:10.1016/j.eja.2019.01.009.

import numpy as np
import pandas as pd
import numba

def dTT(T,Tbase):
    #Temperature with reference to a base for degree-day method after Ritchie et al. 1985 (CERES-Wheat)
    return np.fmax(T-Tbase,0.)

def dBM(fSolar, SRAD, F_CO2, F_Temp, F_Water, F_Heat, RUE):
    #change in biomass similar to Williams et al. 1989 (EPIC)
    dBiomass = 10. * RUE * fSolar * SRAD * F_CO2 * F_Temp * np.fmin(F_Water, F_Heat)
    return dBiomass

def fsolar(TT,F_Water,fSolar_water,Tsum,I50maxW,I50maxH,fsolmax=0.95):
    #Beer-Lambert’s law of light attenuation
    I50B = TT.copy() * 0.
    fSolar = TT.copy() * 0.
    for i in np.arange(len(TT))[1:]:
        fSol1 = np.min([1.,fsolmax/(1+np.exp((TT.iloc[i]-I50B.iloc[i-1])*-0.01))])
        fSol2 = np.min([1.,fsolmax / (1 + np.exp((TT.iloc[i]-(Tsum - I50B.iloc[i-1])) * 0.01))])
        fSolar.iloc[i] = np.min([fSol1,fSol2])*np.min([fSolar_water.iloc[i-1],1.])

        I50B.iloc[i] = np.max([np.max([I50B.iloc[i - 1] + I50maxW*(1.-F_Water.iloc[i-1]),I50B.iloc[i - 1] + I50maxH*(1.-F_Water.iloc[i-1])]),0.])
        if TT.iloc[i] > Tsum:
            maturityday = TT.iloc[i]
            #print('crop maturity')
            break
        if (fSolar.iloc[i]<fSolar.iloc[i-1]) & (fSolar.iloc[i]<=0.005):
            maturityday = TT.index[i]
            #print('crop maturity due to senecence')
            break
    return fSolar,I50B


def T_resp(tmean,Tbase,Topt):
    Tr = np.fmax((tmean-Tbase)/(Topt-Tbase),0.)
    Tr[tmean>=Topt] = 1.
    return Tr

def heat_resp(tmax,maxT,extremeT):
    #heat stress factor based on  Asseng et al., 2011 (APSIM-Nwheat)
    F_Heat = np.fmax(1. - (tmax - maxT) / (extremeT - maxT), 0.)
    F_Heat[tmax > extremeT] = 0.
    F_Heat[tmax <= maxT] = 1.
    return F_Heat

def water_resp(arid,s_water):
    #water balance approach
    return np.fmax(0.,1.-s_water*arid)

def arid(ETo,PAW):
    #water balance aridity index after Woli et al. (2012)
    return 1. - np.min([ETo,0.096*PAW])/ETo

def CO2_resp(CO2, CO2_RUE):
    if (CO2 >= 700.):
        return (1 + CO2_RUE * 350 / 100)
    else:
        return np.max([(CO2_RUE * CO2 * 0.01 + 1 - 0.01 * 350. * CO2_RUE), 1.])

def I50Bp1(I50B,Imaxwater,arid,swater):
    fwater = water_resp(arid, swater)
    return I50B + Imaxwater * (1. - fwater)


### EVAPOTRANSPIRATION ###
# the ETo function can refer to different models.
# I provide the daily ET Szilagyi Jozsa model as first entry

@numba.jit
def Te_opt(T_e,gammax, vabar):
    maxdeltaT_e = 1.
    maxit = 9999999
    itc = 0
    while maxdeltaT_e < 0.01:
        v_e = 0.6108 * np.exp(17.27 * T_e / (T_e + 237.3))  # saturated vapour pressure at T_e (S2.5)
        T_enew = gammax * (v_e - vabar)  # rearranged from S8.8
        deltaT_e = T_enew - T_e
        T_e = T_enew
        maxdeltaT_e = np.abs(np.max(deltaT_e))
        if itc > maxit:
            break
        itc += 1
    return T_e



def ET_SzilagyiJozsa(data, Elev, lat, windfunction_ver='1948', alpha=0.23, zerocorr=True):
    # Taken from R package Evapotranspiration >> Danlu Guo <danlu.guo@adelaide.edu.au>
    # Daily Actual Evapotranspiration after Szilagyi, J. 2007, doi:10.1029/2006GL028708
    # data is assumed to be a pandas data frame with at least:
    # T or Tmin/max - daily temperature in degree Celcius,
    # RH or RHmin/max - daily  relative humidity in percentage,
    # u2 - daily wind speed in meters per second
    # Rs - daily solar radiation in Megajoule per square meter.
    # Result in [mm/day] for EToSJ, EToPM, and EToPT reference ET

    # update alphaPT according to Szilagyi and Jozsa (2008)
    alphaPT = 1.31
    lat_rad = lat * np.pi / 180.
    alphaPT = 1.31
    sigma = 4.903e-09
    Gsc = 0.082
    lambdax = 2.45

    # add julian Days
    J = data.index.dayofyear

    # Calculating mean temperature
    if ('Ta' in data.columns):
        Ta = data['Ta']
    elif ('T' in data.columns):
        Ta = data['T']
    elif ('tas' in data.columns):
        Ta = data['tas']
    else:
        if ('Tmax' in data.columns):
            Ta = (data['Tmax'] + data['Tmin']) / 2  
        else:
            Ta = (data['tasmax'] + data['tasmin']) / 2  
            # Equation S2.1 in Tom McMahon's HESS 2013 paper, which in turn was based on Equation 9 in Allen et al, 1998.


    # Saturated vapour pressure
    if ('Tmax' in data.columns):
        vs_Tmax = 0.6108 * np.exp(17.27 * data['Tmax'] / (data['Tmax'] + 237.3))  # Equation S2.5
        vs_Tmin = 0.6108 * np.exp(17.27 * data['Tmin'] / (data['Tmin'] + 237.3))  # Equation S2.5
        vas = (vs_Tmax + vs_Tmin) / 2.  # Equation S2.6
    else:
        vs_Tmax = 0.6108 * np.exp(17.27 * data['tasmax'] / (data['tasmax'] + 237.3))  # Equation S2.5
        vs_Tmin = 0.6108 * np.exp(17.27 * data['tasmin'] / (data['tasmin'] + 237.3))  # Equation S2.5
        vas = (vs_Tmax + vs_Tmin) / 2.  # Equation S2.6
    # Vapour pressure
    if 'RHmax' in data.columns:
        vabar = (vs_Tmin * data['RHmax'] / 100. + vs_Tmax * data['RHmin'] / 100.) / 2.  # Equation S2.7
        # mean relative humidity
        RHmean = (data['RHmax'] + data['RHmin']) / 2

    else:
        if ('RH' in data.columns):
            vabar = (vs_Tmin + vs_Tmax) / 2 * data['RH'] / 100.
            RHmean = data['RH']
        elif ('huss' in data.columns):
            vabar = (vs_Tmin + vs_Tmax) / 2 * data['huss'] / 100.
            RHmean = data['huss']
        elif ('hurs' in data.columns):
            vabar = (vs_Tmin + vs_Tmax) / 2 * data['hurs'] / 100.
            RHmean = data['hurs']

        # Calculations from data and constants for Penman


    P = 101.3 * ((293. - 0.0065 * Elev) / 293.) ** 5.26  # atmospheric pressure (S2.10)
    delta = 4098 * (0.6108 * np.exp((17.27 * Ta) / (Ta + 237.3))) / (
                (Ta + 237.3) ** 2)  # slope of vapour pressure curve (S2.4)
    gamma = 0.00163 * P / lambdax  # psychrometric constant (S2.9)
    d_r2 = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)  # dr is the inverse relative distance Earth-Sun (S3.6)
    delta2 = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)  # solar dedication (S3.7)
    w_s = np.arccos(-1. * np.tan(lat_rad) * np.tan(delta2))  # sunset hour angle (S3.8)
    N = 24 / np.pi * w_s  # calculating daily values
    R_a = (1440 / np.pi) * d_r2 * Gsc * (
                w_s * np.sin(lat_rad) * np.sin(delta2) + np.cos(lat_rad) * np.cos(delta2) * np.sin(
            w_s))  # extraterristrial radiation (S3.5)
    R_so = (0.75 + (2 * 10 ** - 5) * Elev) * R_a  # clear sky radiation (S3.4)

    if 'Rs' in data.columns:
        R_s = data['Rs']
    elif 'rsds' in data.columns:
        R_s = data['rsds']
    else:
        print('Radiation data missing')
        return

    if ('Tmax' in data.columns):
        R_nl = sigma * (0.34 - 0.14 * np.sqrt(vabar)) * ((data['Tmax'] + 273.2) ** 4 + (data['Tmin'] + 273.2) ** 4) / 2 * (
                1.35 * R_s / R_so - 0.35)  # estimated net outgoing longwave radiation (S3.3)
    else:
        R_nl = sigma * (0.34 - 0.14 * np.sqrt(vabar)) * ((data['tasmax'] + 273.2) ** 4 + (data['tasmin'] + 273.2) ** 4) / 2 * (
                1.35 * R_s / R_so - 0.35)  # estimated net outgoing longwave radiation (S3.3)
    # For vegetated surface
    R_nsg = (1 - alpha) * R_s  # net incoming shortwave radiation (S3.2)
    R_ng = R_nsg - R_nl  # net radiation (S3.1)

    if 'u2' in data.columns:
        u2 = data['u2']
        if windfunction_ver == "1948":
            f_u = 2.626 + 1.381 * u2  # wind function Penman 1948 (S4.11)
        elif windfunction_ver == "1956":
            f_u = 1.313 + 1.381 * u2  # wind function Penman 1956 (S4.3)

        Ea = f_u * (vas - vabar)  # (S4.2)

        Epenman_Daily = delta / (delta + gamma) * (R_ng / lambdax) + gamma / (
                    delta + gamma) * Ea  # Penman open-water evaporation (S4.1)
    
    elif 'sfcWind' in data.columns:
        u2 = data['sfcWind']
        if windfunction_ver == "1948":
            f_u = 2.626 + 1.381 * u2  # wind function Penman 1948 (S4.11)
        elif windfunction_ver == "1956":
            f_u = 1.313 + 1.381 * u2  # wind function Penman 1956 (S4.3)

        Ea = f_u * (vas - vabar)  # (S4.2)

        Epenman_Daily = delta / (delta + gamma) * (R_ng / lambdax) + gamma / (
                    delta + gamma) * Ea  # Penman open-water evaporation (S4.1)
    
    else:

        Epenman_Daily = 0.047 * R_s * np.sqrt(Ta + 9.5) - 2.4 * (R_s / R_a) ** 2 + 0.09 * (Ta + 20) * (
                    1 - RHmean / 100)  # Penman open-water evaporation without wind data by Valiantzas (2006) (S4.12)

    # Iteration for equilibrium temperature T_e
    T_e = Ta
    gammax = Ta - 1 / gamma * (1 - R_ng / (lambdax * Epenman_Daily))

    T_e = Te_opt(T_e.values, gammax.values, vabar.values)
    T_e = pd.Series(T_e, index=Ta.index)

    deltaT_e = 4098 * (0.6108 * np.exp((17.27 * T_e) / (T_e + 237.3))) / (
                (T_e + 237.3) ** 2)  # slope of vapour pressure curve (S2.4)
    E_PT_T_e = alphaPT * (deltaT_e / (deltaT_e + gamma) * R_ng / lambdax)  # Priestley-Taylor evapotranspiration at T_e
    E_SJ_Act_Daily = 2 * E_PT_T_e - Epenman_Daily  # actual evapotranspiration by Szilagyi and Jozsa (2008) (S8.7)

    if zerocorr:
        ET_Daily = np.fmax(E_SJ_Act_Daily, 0.)
    else:
        ET_Daily = E_SJ_Act_Daily

    return ET_Daily,Epenman_Daily,E_PT_T_e


### INFILTRATION ###
# the function for available soil water content can be set to different solvers here
# if there is no time series provided (which might be the common case)
# for now, I stick to the original ARIDITY estimate


def fARID(data, para, ETfu='SJ'):
    '''
    Simple soil water update based on original SIMPLE code
    :param AWC: available water capacity (Vol/Vol)
    :param DDC: deep drainage coefficient (-)
    :param RCN: runoff curve number (-)
    :param RZD: rootzone depth (mm)
    :param WUC: water uptake coefficient (-)
    :switch ETfu: function for ET can be SJ, PM or PT
    '''

    #caluclate ETo if not existing
    if 'EToSJ' in data.columns:
        if ETfu == 'PT':
            ETO = data.EToPT
        elif ETfu == 'PM':
            ETO = data.EToPM2
        else:
            ETO = data.EToSJ
    elif 'Rs' in data.columns:
        EToSJ,EToPM,EToPT = ET_SzilagyiJozsa(data, para['Elev'], para['lat'])
        if ETfu == 'PT':
            ETO = EToPT
        elif ETfu == 'PM':
            ETO = EToPM
        else:
            ETO = EToSJ
    else:
        print('NO RADIATION DATA IN DATAFRAME! CANNOT RUN CROP MODEL.')

    #calculate potential direct runoff
    RO = (data['Prec'] - 0.2 * (25400 / para['RCN'] - 254)) ** 2 / (data['Prec'] + 0.8 * (25400 / para['RCN'] - 254))
    RO[data['Prec'] <= 0.2 * (25400 / para['RCN'] - 254)] = 0.
    CWBD = data['Prec'] - RO

    WBD = CWBD.cumsum() + (para['RZD'] * para['AWC'])
    WBD[WBD / para['RZD'] > para['AWC']] -= para['RZD'] * para['DDC'] * (
                WBD[WBD / para['RZD'] > para['AWC']] / para['RZD'] - para['AWC'])

    WAT = CWBD.copy() * 0.
    ARID = WAT.copy() * 0.
    WAT.iloc[0] = para['RZD'] * para['AWC']
    for i in np.arange(len(CWBD))[1:]:
        WAT.iloc[i] = WAT.iloc[i - 1] + CWBD.iloc[i]
        if (WAT.iloc[i] / para['RZD'] > para['AWC']):
            WAT.iloc[i] -= np.min([para['RZD'] * para['DDC'] * (WAT.iloc[i] / para['RZD'] - para['AWC']), WAT.iloc[i]])
        TR = np.min([para['WUC'] * para['RZD'] * WAT.iloc[i] / para['RZD'], ETO.iloc[i]])
        WAT.iloc[i] -= TR
        if (TR > 0.) & (ETO.iloc[i] > 0.):
            ARID.iloc[i] = 1. - TR / ETO.iloc[i]
        else:
            ARID.iloc[i] = 1. - TR

    return [ARID, ETO]


def SIMPLE(data, sowing, stopday, para, CO2):
    '''
    SIMPLE crop model after Zhao et al. 2019
    :para[' data']: data frame of input weather data
    :para[' sowing']: time stamp of sowing
    :para[' stopday']: time stamp of end of simulation
    :para[' para'][' dictionary'] of para['eters']
    :return: time series of crop development
    '''

    ARID, ETO = fARID(data.loc[sowing:stopday], para, ETfu='SJ')
    F_Heat = heat_resp(data.loc[sowing:stopday, 'Tmax'], para['maxT'], para['extremeT'])
    F_Temp = T_resp(data.loc[sowing:stopday, 'T'], para['Tbase'], para['Topt'])
    F_CO2 = CO2_resp(CO2, para['CO2_RUE'])

    TT = dTT(data.loc[sowing:stopday, 'T'], para['Tbase']).cumsum() + para['InitialTT']

    F_Water = water_resp(ARID, para['s_water'])
    fSolar_water = F_Water
    fSolar_water[fSolar_water > 0.1] = 1.
    fSolar_water[
        fSolar_water <= 0.1] += 0.9  # radiation interception is affected when the drought stress becomes severe enough after Steduto et al. (2009) (AquaCrop model)

    fSolar, I50B = fsolar(TT, F_Water, fSolar_water, para['Tsum'], para['I50maxW'], para['I50maxH'], fsolmax=0.95)
    BM = (10. * para['RUE'] * fSolar * data.loc[sowing:stopday, 'Rs'] * F_CO2 * F_Temp * np.fmin(F_Water,
                                                                                               F_Heat)).cumsum() + para['IniBio']  # change in biomass similar to Williams et al. 1989 (EPIC)
    Yield = BM * para['HIp']

    res = pd.concat([Yield, ARID, ETO, fSolar, I50B], axis=1)
    res.columns = ['Yield', 'ARID', 'ETO', 'fSolar', 'I50B']
    return res
