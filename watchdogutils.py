# ------- Load dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors
import matplotlib.pyplot as plt

import os


# parseAndReadMetData:
# Summary: Parses file name to gather metadata, appends to read in pandas dataframe
#
# Inputs  - fname (full file path of met data file)
# Returns - df    (pandas dataframe with logger location and station ID)
def parseAndReadMetData(dataDir, fname):
    location       = fname.split('_')[0]
    stationNum     = fname.split('_')[1]
    df             = pd.read_csv(dataDir + fname, sep = '\t', skiprows=2)
    df['Locale']   = location
    df['LoggerID'] = int(stationNum)
    return df

# prepareTimeStamps:
# Summary: generates time pandas date-time timestamps from time column, renames and adds
# time variables.
#
# Inputs  - df (pandas dataframe with logger location and station ID
# Returns - df (pandas dataframe with appended time stamp information)
def prepareTimeStamps(df):
    df.rename(columns = {df.columns[0]:'Timestamp'}, inplace = True)
    df.index    = pd.to_datetime(df['Timestamp'])
    df['doy']   = df.index.dayofyear
    df['month'] = df.index.month
    df['year']  = df.index.year
    df['hour']  = df.index.hour
    return df

# getMonthLabels:
# Summary: Finds out what unique months there are in the dataframe, and
# returns a correctly orderd list of those months, and the one letter
# abbreviation for that month. This is strictly for plotting purposes.

# Inputs  - df (pandas dataframe with complete timestamps)
# Returns - monthsInDF, monthLabels
def getMonthLabels(df):
    months = ['J','F','M','A','M','J','J','A','S','O','N','D']
    indexes = np.unique(df.month, return_index=True)[1]
    monthsInDF = np.array([df.month[index] for index in sorted(indexes)])
    monthLabels = [months[i] for i in monthsInDF-1]
    return monthsInDF, monthLabels

# rawSummaryPlots:
# Summary: Creates seven subplots for the main variables output by the Watchdog 2000
# series loggers. Saves the plot with a site and station ID specific file name.

# Inputs  - df (pandas dataframe with complete timestamps)
#         - outDir (/path/where/output/will/be/saved/)
# Returns - null
def rawSummaryPlots(df, outDir):
    # Setup plot axes
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4,2, figsize = (20,22))
    f.delaxes(ax8)

    # Soil temperature (sensor ports A and B)
    ax1.plot(df.index, df['TMPA'], lw = 3, color = 'gray', label = 'Temp A')
    ax1.plot(df.index, df['TMPB'], lw = 3, color = 'green', label = 'Temp B')
    ax1.set_title(df['Locale'][0] + ' ' + str(df['LoggerID'][0]))
    ax1.set_ylim([-5, 20])
    ax1.set_ylabel('Soil Temperature (deg C)')
    ax1.set_xticklabels([])
    ax1.legend()

    # Soil VWC (sensor ports C and D)
    ax2.plot(df.index, df['VWCC'], lw = 3, color = 'gray', 
             label = 'VWC C')
    ax2.plot(df.index, df['VWCD'], lw = 3, color = 'green', 
             label = 'VWC D')
    ax2.set_ylim([0, 40])
    ax2.set_ylabel('Volumetric Water Content (%)')
    ax2.set_xticklabels([])
    ax2.legend()

    # rH 
    ax3.plot(df.index, df['HMD'], lw = 2, color = 'black')
    ax3.set_ylabel('Relative Humidity (%)')
    ax3.set_xticklabels([])

    # TA 
    ax4.plot(df.index, df['TMP'], lw = 2, color = 'black')
    ax4.set_ylabel('Air Temperature (deg C)')
    ax4.set_xticklabels([])
    ax4.legend()

    # Wind velocity 
    ax5.plot(df.index, df['WNG'], lw = 1, color = 'gray', 
             label = 'Gusts', alpha = 0.5)
    ax5.plot(df.index, df['WNS'], lw = 1, color = 'black', 
             label = 'Wind Speed')
    ax5.set_ylabel('Wind Speed (km h$^{-1}$)')
    ax5.set_xticklabels([])

    # Wind direction
    ax6.plot(df.index, df['WND'], lw = 2, color = 'black')
    ax6.set_ylabel('Wind Direction (deg)')
    ax6.set_ylim([0,360])
    plt.setp(ax6.get_xticklabels(), rotation = 45)

    # Precip
    ax7.plot(df.index, df['RNF'], lw = 2, color = 'black')
    ax7.set_ylabel('Precipitation (mm)')
    plt.setp(ax7.get_xticklabels(), rotation = 45)

    sns.despine()

    # Create the file name and save the figure
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig(outDir + '/stationTS/' + plotStationName + 'RawSummary.tif')
    plt.close()
    
# plotWindRose:
# plotWindRose:
# Summary: Creates a single windrose plot for a given metdf.
# Saves the plot with a site and station ID specific file name.

# Inputs  - df (pandas dataframe with complete timestamps)
#         - outDir (/path/where/output/will/be/saved/)
# Returns - null
def plotWindRose(df, outDir):
    from windrose import WindroseAxes
    
    ax = WindroseAxes.from_ax()
    ax.bar(df.WND, df.WNS, normed=True, 
           opening=0.9, bins = np.arange(0,35, 5), edgecolor='white')

    lgd = ax.legend(loc = (0.9,0.9), fontsize = 15, title = 'Windspeed (kph)')
    ax.set_xlabel(df['Locale'][0] + ' ' + str(df['LoggerID'][0]), fontsize = 18)
    
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig(outDir + '/stationTS/' + plotStationName + 'Windrose.tif', 
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


# cleanVWC:
# Summary: Parses file name to gather metadata, appends to read in pandas dataframe
#
# Inputs  - df (pandas dataframe with complete timestamps)
#         - outDir (/path/where/output/will/be/saved/)
# Returns - df
def cleanVWC(df, outDir):
    # Specify the threshold and windowsize for the filter. Thresh is in
    # units of % (VWC), and windowsize counts the number of 15 minute
    # intervals over which to assess the threshold. 
    thresh = 30
    windowsize = 96
    
    # We're going to output the results of the outlier detection to a 
    # summary figure for review. Setup the plot structure outside the loop.
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15,10))
    plt.subplots_adjust(hspace=0.5)
    kw = dict(marker='o', linestyle='none', color='r', markersize = 10)
    idx = 0
    
    # Iterate over the VWC columns in the metstation DF
    for VWCvar in ['VWCC','VWCD']:
        
        # Create column identifiers for the threshold and filtered values
        filtThresh = VWCvar + '_fT'
        filteredVWC = VWCvar + '_f'
        
        # Rolling median filter with specified window size
        df[filtThresh] = pd.rolling_median(df[VWCvar], window=windowsize, 
            center=True).fillna(method='bfill').fillna(method='ffill')
        
        # Tese filtered values against specified threshold
        difftest = np.abs(df[VWCvar] - df[filtThresh])
        
        # Boolean for values that do not pass the test
        outlier_pos = difftest > thresh
        
        # Replace filtered values with NaN, so long as there are identified outliers
        df[filteredVWC] = df[VWCvar]
        if outlier_pos[outlier_pos == True].size > 0:
            df[filteredVWC][outlier_pos] = np.nan
    
        # populate the plot
        df[VWCvar].plot(ax = f.axes[idx], color = 'gray')
        if outlier_pos[outlier_pos == True].size > 0:
            df[VWCvar][outlier_pos].plot(ax = f.axes[idx], **kw)
            f.axes[idx].set_ylim([0,105])
        df[filteredVWC].plot(ax = f.axes[idx+2], color = 'gray')
        f.axes[idx].set_title(VWCvar)
        idx += 1
        
    for ax in f.axes:
        ax.set_xlabel('')
    ax1.set_ylabel('Soil VWC (%)')
    ax3.set_ylabel('Soil VWC \nfiltered (%)')
   
    plt.tight_layout()
    sns.despine()
    
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig(outDir + '/QAQC/' + plotStationName + 'VWC_Filt.tif')
    plt.close()
    return df

# cleanTMP:
# Summary: Parses file name to gather metadata, appends to read in pandas dataframe
#
# Inputs  - fname (full file path of met data file)
# Returns - df    (pandas dataframe with TMP data )
def cleanTMP(df, outDir):
    # Specify the threshold and windowsize for the filter. Thresh is in
    # units of degrees (C), and windowsize counts the number of 15 minute
    # intervals over which to assess the threshold. 
    thresh = 5
    windowsize = 30
    
    # We're going to output the results of the outlier detection to a 
    # summary figure for review. Setup the plot structure outside the loop.
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (15,10))
    plt.subplots_adjust(hspace=0.5)
    kw = dict(marker='o', linestyle='none', color='r', markersize = 10)
    idx = 0
    
    # Iterate over the TMP columns in the metstation DF
    for TMPvar in ['TMPA','TMPB']:
        
        # Create column identifiers for the threshold and filtered values
        filtThresh = TMPvar + '_fT'
        filteredTMP = TMPvar + '_f'
        
        # Rolling median filter with specified window size
        df[filtThresh] = pd.rolling_median(df[TMPvar], window=windowsize, 
            center=True).fillna(method='bfill').fillna(method='ffill')
        
        # Tese filtered values against specified threshold
        difftest = np.abs(df[TMPvar] - df[filtThresh])
        
        # Boolean for values that do not pass the test
        outlier_pos = difftest > thresh
        
        # Replace filtered values with NaN, so long as there are identified outliers
        df[filteredTMP] = df[TMPvar]
        if outlier_pos[outlier_pos == True].size > 0:
            df[filteredTMP][outlier_pos] = np.nan
    
        # populate the plot
        df[TMPvar].plot(ax = f.axes[idx], color = 'gray')
        if outlier_pos[outlier_pos == True].size > 0:
            df[TMPvar][outlier_pos].plot(ax = f.axes[idx], **kw)
        df[filteredTMP].plot(ax = f.axes[idx+2], color = 'gray')
        f.axes[idx].set_title(TMPvar)
        idx += 1
        
    for ax in f.axes:
        ax.set_xlabel('')
    ax1.set_ylabel('Soil Temp (degrees C)')
    ax3.set_ylabel('Soil Temp \nfiltered (degrees C)')
    
    plt.tight_layout()
    sns.despine()
    
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig(outDir + '/QAQC/' + plotStationName + 'TMP_Filt.tif')
    plt.close()
    return df

def precipSummaryPlot(df):
    monthsInDF, monthLabels = getMonthLabels(df)

    f, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (15,5))

    df.RNF.plot(ax = ax1, color = 'gray')
    ax1.set_xlabel('')
    ax1.set_ylabel('Rainfall (mm)')

    df.RNF.cumsum().plot(ax = ax2, color = 'gray')
    ax2.set_xlabel('')

    monthlySum = df.groupby([df.index.year, df.index.month]).sum().reset_index()

    ax3.bar(monthlySum.index, monthlySum.RNF, align='center', color = 'gray')
    ax3.set_xticks([0,1,2,3,4,5])
    ax3.set_xticklabels(monthLabels)
    sns.despine()
    
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig('Z:/JFSP_2015/Weather Stations/Data/Vis/Summary/' + plotStationName + 'P_Summary.tif')
    plt.close()

def tempSummaryPlot(df):
    months = ['J','F','M','A','M','J','J','A','S','O','N','D']

    indexes = np.unique(df.month, return_index=True)[1]
    monthsInDF = np.array([df.month[index] for index in sorted(indexes)])
    monthLabels = [months[i] for i in monthsInDF-1]

    df['TimeOfDay'] = 'daytime'
    df['TimeOfDay'][df.hour > 18] = 'nighttime'
    
    f, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (15,5))
    sns.boxplot(x="month", y="TMPA", data=df, palette=['white','gray'], 
                hue = 'TimeOfDay', ax = ax1, order = monthsInDF)
    ax1.set_xticklabels(monthLabels)
    ax1.set_title('Soil Temp Open')
    ax1.legend(loc = 2)
    
    sns.boxplot(x="month", y="TMPB", data=df, palette=['white','gray'], 
                hue = 'TimeOfDay', ax = ax2, order = monthsInDF)
    ax2.set_xticklabels(monthLabels)
    ax2.set_title('Soil Temp Shrub')
    ax2.legend_.remove()

    sns.boxplot(x="month", y="TMP", data=df, palette=['white','gray'], 
                hue = 'TimeOfDay', ax = ax3, order = monthsInDF)
    ax3.set_xticklabels(monthLabels)
    ax3.set_title('Air Temp')
    ax3.legend_.remove()

    for ax in f.axes:
        ax.set_ylabel('')
        ax.set_ylim(-20,30)
        ax.set_xlabel('')
    ax1.set_ylabel('Temperature (deg C)')
    ax2.set_xlabel('Month')
    sns.despine()
    plt.tight_layout()
    
    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig('Z:/JFSP_2015/Weather Stations/Data/Vis/Summary/' + plotStationName + 'TMP_Summary.tif')
    plt.close()
    
def VWCSummaryPlot(df):
    monthsInDF, monthLabels = getMonthLabels(df)

    df['TimeOfDay'] = 'daytime'
    df['TimeOfDay'][df.hour > 18] = 'nighttime'

    f, ((ax1, ax2, ax3)) = plt.subplots(1,3, figsize = (15,5))
    sns.boxplot(x="month", y="VWCC_f", data=df,
                ax = ax1, order = monthsInDF)
    ax1.set_xticklabels(monthLabels)
    ax1.set_title('Soil VWC Open')
    ax1.legend(loc = 2)

    sns.boxplot(x="month", y="VWCD_f", data=df, 
                ax = ax2, order = monthsInDF, color = 'white')
    ax2.set_xticklabels(monthLabels)
    ax2.set_title('Soil VWC Shrub')

    df.RNF.plot(ax = ax3, color = 'gray')
    ax1.set_xlabel('')
    ax1.set_ylabel('Rainfall (mm)')

    # Strange workaround, completely allows control of all 2d line elements
    # that make up sns boxplots. From stackoverflow question #36874697
    for ax in f.axes:
        ax.set_ylabel('')
        ax.set_xlabel('')

        for i,artist in enumerate(ax.artists):
            # Set the linecolor on the artist to the facecolor, and set the facecolor to None
            # col = artist.get_facecolor()
            artist.set_edgecolor('black')
            artist.set_facecolor('None')
            for j in range(i*6,i*6+6):
                    line = ax2.lines[j]
                    line.set_color('black')
                    line.set_mfc('black')
                    line.set_mec('black')

    ax1.set_ylabel('VWC (%)')
    ax2.set_xlabel('Month')
    
    # Set both VWC lims to same max value
    ymax = max(df.VWCC_f.max(), df.VWCD_f.max()) + 2
    
    ax1.set_ylim(0,ymax)
    ax2.set_ylim(0,ymax)

    sns.despine()
    plt.tight_layout()

    plotStationName = df['Locale'][0] + '_' + str(df['LoggerID'][0]) + '_'
    plt.savefig('Z:/JFSP_2015/Weather Stations/Data/Vis/Summary/' + plotStationName + 'VWC_Summary.tif')
    plt.close()