'''
NAME:   plt_filter.py
PURPOSE:    Python script for filtering and plotting NPD (Raw mode) data
Main Functions:     NPD_raw_read (see NPD_raw_readv1.py for details): Access NPD (RAW mode) Module, NPD_raw_readv1.py
                    RawFilter.filter_data():    Filters raw data via coincidence flags, coins = 0 (obsolete)
                    RawFilter.mask():   Masks raw data via coincidence flags (mainly used for data processing)
                    RawPlot.dir_sep():  Direction separation, dir0, dir1, dir2
                    RawPlot.plt_tph():  Plots Time Series Pulse Height (Ph)
                    RawPlot.tof_ph()    Time of Flight vs. Ph
                    RawPlot.store():    Stores filtered Ph and TOF data to 'npd?rawYYYY.json' file (text file)
                                        (See json documentation in python)
                    json File Format:
                        1) each line is a dictionary type of variable containing 'keys'
                        2) keys are the following:
                            Filename, Date, Start, End, dt, dir0_ph, dir0_tof, dir0_time, dir1_ph,
                            dir1_tof, dir1_time, dir2_ph, dir2_tof, dir2_time
                        3) Filename: name of the netCDF file from which the data taken from
                            f_date: date in UTC based on the starting time on which the data was taken (YYYY-MM-DD)
                            dir#_ph: filtered Pulse height values
                            dir#_tof: filtered TOF values
                            dir#_time: corresponding time in sec for every Ph/TOF value
                            cnt_st: Start counter
                            cnt_stp0: Stop0 counter
                            cnt_stp1: Stop1 counter
                            cnt_stp2: Stop2 counter
                            cnt_tof: TOF counter
                            cnt_raw: Raw counter

PROGRAMMER: Emmanuel Nunez (emmnun-4@student.ltu.se)
Last Updated: 2016-03-24
Revision: 2016-05-10    - dt[sec] bug fixed. dt value will no longer negative
Revision: 2016-06-23    - counters added in json file


REFERENCES: Yoshifumi Futaana
            Xiao-Dong Wang

PARAMS:
	    id: data filename (either netCDF file or file list containing '.nc' file names)
	    -v: optional argument, print information in netCDF file (not recommended on file list input)
	    -dinfo: optional argument, displays date information of the file
	    -cnts: optional argument, prints Pulse height counts
	    -ttof: optional argument, plots time series Time-of-Flight (TOF)
	    -tofph: optional argument, plots TOF - Pulse height
	    -tph: optional argument for time series plotting (not recommended on file list input)
	    -save: optional argument, stores filtered data to '.json' file. (See above description)
	    -savefig: optional argument, saves figures

Caution! Input file list must have a '.txt' trailing file name
         Using of file list as input is not recommended for direct plotting! Use netCDF file instead.
         File list as input file is designed mainly for storing filtered data
         Zero and max values (255) ph values are filtered at RawPlot.comp_tph()

Note:   json file contains time-to-ph/tof array-pairs in each file for plotting purposes. Thus, time values
        may be repeating and not necessarily the same size for each direction and cannot be used for time-stamp purposes.

Sample Usage: python3 plt_filter.py npd2raw20110426000341.nc -tph (To plot Pulse height time series)
              python3 plt_filter.py npd1raw2010.txt -save (To store data to 'npd1raw2010.json'
                                                        where 'npd1raw2010.txt' containing lists of netCDF files)
'''

# !/usr/local/bin/python3

import numpy as np
import argparse, os, datetime, time, json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import epoch2num
import NPD_raw_readv1 as raw
import matplotlib.colors as colors
import irfpy.vexpvat.vexspice as vspice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

'''========================= Filtering Class Object==========================
'''


class RawFilter(object):
    # Constructor
    def __init__(self, data):
        self.data = data

    def var(self):
        # Access key info
        print('Data Variables:')
        for key in self.data:
            print ('Name: ', key)
            print ('shape: ', self.data[key].shape)


    @property
    def filter_data(self):
        coins = self.data['coins']
        tofs = self.data['TOFs']
        phs = self.data['PHs']
        dirs = self.data['dirs']

        # Determine indices of coincidence flags
        val_bool = (coins == 0)  # bool index arr [True, False,...]
        val_coins = coins[val_bool]  # arr of valid coins
        inv_coins = coins[coins != 0]  # arr of invalid coins

        # Array of valid variables
        val_tofs = tofs[val_bool]  # arr valid TOFs 0.5 ns resolution
        val_phs = phs[val_bool]
        val_dirs = dirs[val_bool]

        print ('No. of valid coins=0: ', len(val_coins))
        print ('No. of invalid coins(coins !=0): ', len(inv_coins))
        fltr_data = {'ST': self.data['ST'], 'ET': self.data['ET'], 'Time': self.data['Time'], 'DT': self.data['DT'],
                     'Cnts': self.data['Cnts'], 'Calib': self.data['Calib'], 'Regs': self.data['Regs'],
                     'TOFs': val_tofs, 'dirs': val_dirs, 'coins': val_coins, 'PHs': val_phs,
                     'discard': self.data['discard']}

        return fltr_data

    @property
    def mask(self):
        coins = self.data['coins']
        tofs = self.data['TOFs']
        phs = self.data['PHs']
        dirs = self.data['dirs']


        # Determine indices of coincidence flags (method of masking verified correct)
        coin_bool = (coins != 0)  # bool index arr [True, False,...]

        # Masking data
        m_coins = np.ma.array(coins, mask=coin_bool)
        m_tofs = np.ma.array(tofs, mask=coin_bool)
        m_dirs = np.ma.array(dirs, mask=coin_bool)
        m_phs = np.ma.array(phs, mask=coin_bool)

        '''
                 #Test Masked data
                 test_coin = np.ma.compressed(m_coins)
                 test_dir = np.ma.compressed(m_dirs)
                 test_PHs = np.ma.compressed(m_PHs)
                 print(test_coin)
                 print(test_dir)
                 print(test_PHs)
                 '''

        masked_data = {'ST': self.data['ST'], 'ET': self.data['ET'], 'Time': self.data['Time'], 'DT': self.data['DT'],
                       'Cnts': self.data['Cnts'], 'Calib': self.data['Calib'], 'Regs': self.data['Regs'],
                       'TOFs': m_tofs, 'dirs': m_dirs, 'coins': m_coins, 'PHs': m_phs, 'discard': self.data['discard']}
        return masked_data


'''============================Plotting Class Object ==================================
'''


class RawPlot(object):
    _phbin = [150, 256]
    _tofbin = [256, 16]

    # Constructor
    def __init__(self, data, name):
        self.data = data
        self.filename = name

    def info(self, disp = False):  # General info of the data

        f_name = self.filename
        st = self.data['ST']
        et = self.data['ET']
        d_format = '%H:%M:%S'

        # start date and time formatting
        s_day = epoch2num(st)  # num of days since epoch
        s_date = datetime.date.fromordinal(s_day)  # returns formatted date YYYY-MM-DD
        s_time = time.gmtime(st)  # returns struct_time of date and time values
        e_time = time.gmtime(et)

        # Range in seconds
        st_sec = s_time.tm_hour * 3600 + s_time.tm_min * 60 + s_time.tm_sec
        et_sec = e_time.tm_hour * 3600 + e_time.tm_min * 60 + e_time.tm_sec
        dsec = abs(et_sec - st_sec)

        start_str = time.strftime(d_format, s_time)
        end_str = time.strftime(d_format, e_time)
        # print(s_date)

        info_d = {'Filename': f_name, 'Date': s_date, 'Start': start_str, 'End': end_str, 'dt': dsec}
        if disp:
            print('=============================================')
            print('Filename: ', f_name)
            print('Date: ', s_date, ', epoch[day]: ', s_day[0])
            print('GMT:')
            print('Start Time : ', time.strftime(d_format, s_time), ', End Time: ', time.strftime(d_format, e_time))
            print('=============================================')
            print('Dictionary Keys, Values:')

        return info_d

    def time_conv(self, t):

        # start date and time formatting
        t_struct = time.gmtime(t)  # returns struct_time of date and time values

        return t_struct

    def rho(self, coord):
        return np.sqrt(coord[1] ** 2 + coord[2] ** 2)

    def dctime_conv(self, datadict):

        dcx, dcr = [], []

        dctime = datadict['Time']
        for t in range(len(dctime)):
            t_struct = self.time_conv(dctime[t])
            temp = vspice.get_position(
                        datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                          t_struct.tm_min, t_struct.tm_sec))
            dcx.append(temp[0])
            rho = self.rho(temp)  # calculate rho
            dcr.append(rho)

        return dcx, dcr

    def ph_dirsep(self, cnts = False):  # Direction separation function for Ph

        phs = self.data['PHs']
        dirs = self.data['dirs']

        # direction/sectors separation
        dir0_phs = np.ma.masked_where(dirs != 0, phs)  # mask elements in PHs not equal to 0 in dirs
        dir1_phs = np.ma.masked_where(dirs != 1, phs)
        dir2_phs = np.ma.masked_where(dirs != 2, phs)

        if cnts:
            # masking check
            print('\n Valid coincidence each direction: ')
            print('dir0: ', dir0_phs[~np.ma.getmask(dir0_phs)].size)
            print('dir1: ', dir1_phs[~np.ma.getmask(dir1_phs)].size)
            print('dir2: ', dir2_phs[~np.ma.getmask(dir2_phs)].size)

        return dir0_phs, dir1_phs, dir2_phs

    def cnter_chek(self, i):    # performance check

        counter = self.data['Cnts']
        ct = counter[i, 1] + counter[i, 2] + counter[i, 3]
        if counter[i, 1] == 0 and counter[i, 2] == 0 and counter[i, 3] == 0:
            # print(counter[i])
            return True
        if ct < 512:
            return True

        else:
            return False

    def tof_dirsep(self):   # Direction separation function for TOF

        tofs = self.data['TOFs']
        dirs = self.data['dirs']

        # direction/sectors separation
        dir0_tofs = np.ma.masked_where(dirs != 0, tofs)  # mask elements in TOFs not equal to 0 in dirs
        dir1_tofs = np.ma.masked_where(dirs != 1, tofs)
        dir2_tofs = np.ma.masked_where(dirs != 2, tofs)

        # masking check
        # print(dir0_tofs[~np.ma.getmask(dir0_tofs)])

        return dir0_tofs, dir1_tofs, dir2_tofs

    def duty_cycle(self, counter): # Compute duty cycle

        low_cnt = 0.
        # Format [Start Stop0, Stop1, Stop2, TOF, Raw] (Grigoriev, 2007)
        dclist = []
        stop0 = counter[:, 1]
        stop1 = counter[:, 2]
        stop2 = counter[:, 3]
        for i in range(len(stop0)):
            ct = stop0[i] + stop1[i] + stop2[i]
            dc = float(512)/ct
            #print(dc)
            if ct < 512:
                low_cnt += 1    # number of low counts
                dc = 0. # disregard the record
            if np.isinf(dc) or np.isnan(dc):
                #cnt_err = {'Filename':self.filename, 'Cnts': counter[i, :].tolist(), 'Index': counter[:i].shape, 'Time': self.data['Time'].tolist()}
                #self.err_store(cnt_err)
                dc = 0.

            dclist.append(dc)

        lcfrac = float(low_cnt)/len(stop0) # fraction of low counts over the entire records

        return dclist, lcfrac  # duty cycle

    def raw_eventplt(self, dir, Rp, bins, savefig=False):

        counters = self.data['Cnts']

        # Call for general info
        info = self.info()

        # Compute Duty cycle
        dc, low_cnt = self.duty_cycle(counters)

        # Call Time to Coordinates Conversion
        dcx, dcr = self.dctime_conv(self.data)

        # ==== TOF VALUES ======
        datadict = self.test_comp(dir)

        x, r = [], []
        dir_time = datadict[str(dir) + '_time']
        dir_tof = datadict[str(dir) + '_tof']

        for t in range(len(dir_time)):
            t_struct = self.time_conv(dir_time[t])
            temp = vspice.get_position(
                        datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                          t_struct.tm_min, t_struct.tm_sec))
            x.append(temp[0])
            rho = self.rho(temp)  # calculate rho
            r.append(rho)


        # List to Array Conversion
        x_arr = np.array(x)
        r_arr = np.array(r)

        dcx_arr = np.array(dcx)
        dcr_arr = np.array(dcr)


        # Divide by Rp
        x_arr = np.divide(x_arr, Rp)
        r_arr = np.divide(r_arr, Rp)

        dcx_arr = np.divide(dcx_arr, Rp)
        dcr_arr = np.divide(dcr_arr, Rp)

        print(dcr_arr[0], dcr_arr[-1])
        #print(dc[-5:])

        # ========Plot Setup ============

        # set bin arrays
        bin_xrange = np.arange(-8, 5, bins[0])
        bin_yrange = np.arange(0, 12.5, bins[1])

        plt.close('all')
        fig1 = plt.figure(figsize=(12, 9))
        fig1.suptitle('No. of Valid Events, Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'] + ', dir: ' + str(dir), fontsize=11,
                     y=0.95)
        ax1 = fig1.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax1.set_xlim([5, -8])
        ax1.set_ylim([0, 12.5])
        ax1.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax1.set_ylabel(r'$r = \sqrt{y^2 + x^2}$', fontsize=11)

        fig2 = plt.figure(figsize=(12, 9))
        fig2.suptitle('Accumulation Time, Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End']+ ', dir: ' + str(dir), fontsize=11,
                     y=0.95)
        ax2 = fig2.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax2.set_xlim([5, -8])
        ax2.set_ylim([0, 12.5])
        ax2.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax2.set_ylabel(r'$r = \sqrt{y^2 + x^2}$', fontsize=11)

        fig3 = plt.figure(figsize=(12, 9))
        fig3.suptitle('Count Rate, Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'] + ', dir: ' + str(dir), fontsize=11,
                     y=0.95)
        ax3 = fig3.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax3.set_xlim([5, -8])
        ax3.set_ylim([0, 12.5])
        ax3.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax3.set_ylabel(r'$r = \sqrt{y^2 + x^2}$', fontsize=11)

        # events
        hist1, xedges, yedges = np.histogram2d(x_arr, r_arr, bins=[bin_xrange, bin_yrange])

        # duty time
        hist2, xedges, yedges = np.histogram2d(dcx_arr, dcr_arr, weights= dc, bins=[bin_xrange, bin_yrange])

        ma_hist2 = np.ma.masked_where(hist2 == 0., hist2)

        ma_ratio = hist1/ma_hist2

        print('No. of Events: ', x_arr.size)
        print('Accumulated Time: ', dcx_arr.size)

        ma_hist1 = np.ma.masked_where(hist1 == 0., hist1)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="2%", pad=0.05)

        ma_hist2 = np.ma.masked_where(hist2 == 0., hist2)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="2%", pad=0.05)

        div3 = make_axes_locatable(ax3)
        cax3 = div3.append_axes("right", size="2%", pad=0.05)

        # ========= Event Image =================
        # ax1.scatter(x_arr, r_arr)
        im1 = ax1.imshow(np.rot90(ma_hist1), interpolation='nearest', vmin= ma_hist1.min(), vmax=ma_hist1.max(),
                            #norm=colors.LogNorm(vmin=ma_hist1.min(), vmax=10. ** 6),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        cb1 = fig1.colorbar(im1, cax=cax1)
        cb1.set_label('# of Events', labelpad=2.)

        # ========= Accumulated Time Image =======
        im2 = ax2.imshow(np.rot90(ma_hist2), interpolation='nearest', vmin= ma_hist2.min(), vmax=ma_hist2.max(),
                            #norm=colors.LogNorm(vmin=ma_hist2.min(), vmax=10. ** 6),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        cb2 = fig2.colorbar(im2, cax=cax2)
        cb2.set_label('sec', labelpad=2.)

        # ====== Ratio ========
        im3 = ax3.imshow(np.rot90(ma_ratio), interpolation='nearest', vmin= ma_ratio.min(), vmax=ma_ratio.max(),
                            #norm=colors.LogNorm(vmin=ma_ratio.min(), vmax=10. ** 3),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        cb3 = fig1.colorbar(im3, cax=cax3)
        cb3.set_label('cnts/sec', labelpad=2.)

        self.plt_venus(ax1)
        self.plt_bsicb(ax1)
        self.plt_venus(ax2)
        self.plt_bsicb(ax2)
        self.plt_venus(ax3)
        self.plt_bsicb(ax3)
        if savefig:
            fig1.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_Raw/figs/npd2/cntrate/'+self.filename[:-3]+'_event.png')
            fig2.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_Raw/figs/npd2/cntrate/'+self.filename[:-3]+'_accutime.png')
            fig3.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_Raw/figs/npd2/cntrate/'+self.filename[:-3]+'_cntrate.png')
        plt.show()

    def plt_venus(self, subplot):

        self.dual_half_circ((0, 0), radius=1.0, angle=90, ax=subplot, colors=('k', 'goldenrod'))

    def dual_half_circ(self, center, radius, angle=0, ax=None, colors=('w', 'k'), **kwargs):
        """
        Add two half circles to the axes *ax* (or the current axes) with the
        specified facecolors *colors* rotated at *angle* (in degrees).
        """
        if ax is None:
            ax = plt.gca()
        theta1, theta2 = angle, angle + 180
        w1 = patches.Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
        w2 = patches.Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
        for wedge in [w1, w2]:
            ax.add_artist(wedge)
        return [w1, w2]

    def plt_bsicb(self, subplot):

        # Initialize
        theta = np.linspace(0, 5*np.pi/6.0, 25)

        # Bow Shock Variables (MartinenCZ et. al. 2008)
        L = 1.303   # semi-latus rectum
        epsilon = 1.056 # eccentricity
        r_tsd = 1.984   # terminator shock distance

        r_bs = float(L)/(1 + (epsilon*np.cos(theta)))

        y_bs = r_bs * np.sin(theta)
        x_bs = (r_bs * np.cos(theta)) + L/2

        # ICB
        k = -0.097
        d = 1.109
        r_icb = 1.09    # radius dayside

        # dayside ICB
        h_theta = np.linspace(0, np.pi/2, 25)
        xd = r_icb * np.cos(h_theta)
        yd = r_icb * np.sin(h_theta)
        # nightside ICB
        xn = np.linspace(0, -8, 25)
        yn = (k*xn) + d

        # terminator shock
        y_tsd = r_tsd * np.sin(h_theta)
        x_tsd = r_tsd * np.cos(h_theta)

        # Plotting
        subplot.plot(x_bs, y_bs, 'k-', label = 'Bow Shock')
        subplot.plot(xd, yd, 'k--')
        subplot.plot(xn, yn, 'k--', label = 'IMB')
        subplot.plot(x_tsd, y_tsd, 'k--', label = 'Terminator Shock')

    def plt_tcounter(self, ax):

        counter = self.data['Cnts']
        time = self.data['Time']
        stop0 = counter[:, 1]
        stop1 = counter[:, 2]
        stop2 = counter[:, 3]

        dum_counter = np.ones(len(stop0))

        for i in range(len(stop0)):
            ct = counter[i, 1] + counter[i, 2] + counter[i, 3]
            if stop0[i] == 0 and stop1[i] == 0 and stop2[i] == 0:
                dum_counter[i] = 0
            if ct < 512:
                dum_counter[i] = 0

        ma_stop0 = np.ma.masked_where(dum_counter == 0, stop0)
        ma_stop1 = np.ma.masked_where(dum_counter == 0, stop1)
        ma_stop2 = np.ma.masked_where(dum_counter == 0, stop2)
        ma_time = np.ma.masked_where(dum_counter == 0, time)

        stop0 = np.ma.compressed(ma_stop0)
        stop1 = np.ma.compressed(ma_stop1)
        stop2 = np.ma.compressed(ma_stop2)
        time = np.ma.compressed(ma_time)

        ax.scatter(time, stop0, color='lime', label ='stop0', lw= 0.01)
        ax.scatter(time, stop1, color='cyan', label ='stop1', lw = 0.01)
        ax.scatter(time, stop2, color='violet', label ='stop2', lw = 0.01)
        ax.set_ylabel('Counter')
        ax.set_yscale('log')
        ax.set_xlim([time.min(), time.max()])
        ax.legend(prop ={'size':8}, loc='upper left')

        return time

    def test_comp(self, dir):   # reduced memory allocation

        t_data = self.data['Time']

        # Direction Separation
        dir0_phs, dir1_phs, dir2_phs = self.ph_dirsep()
        dir0_tofs, dir1_tofs, dir2_tofs = self.tof_dirsep()

        if dir == 'dir0':

            # ======= dir PH, TOF, TIME Allocation =========

            dir0_phs_h = np.array([], dtype='int32')
            dir0_tofs_h = np.array([], dtype='int32')
            dir0_time = np.array([], dtype='float64')

            for i in range(len(t_data)):

                dir0_phs_row = np.ma.compressed(dir0_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
                dir0_tofs_row = np.ma.compressed(dir0_tofs[i, :])

                cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
                if cnt_check:
                    # print(dir0_tofs[i-1].shape)
                    # print('dir0', dir0_tofs_row)
                    dir0_phs_row = [0]
                    dir0_tofs_row = [0]

                # Check num dim matched
                # print(dir0_phs_row.shape, dir0_tofs_row.shape)

                # time arr allocation
                time_row = np.zeros(len(dir0_phs_row))  # number of zeros of time_row equal to number of phs per row
                time_row = time_row + t_data[i]
                dir0_time = np.concatenate((dir0_time, time_row), axis=0)  # consider append list, reshape in the future?

                dir0_phs_h = np.concatenate((dir0_phs_h, dir0_phs_row), axis=0)
                dir0_tofs_h = np.concatenate((dir0_tofs_h, dir0_tofs_row), axis=0)

            '''
            # ============== Masking 0 and 255 PH values ===================
            dir0_phs_h = np.ma.masked_where(dir0_phs_h == 0, dir0_phs_h)
            dir0_tofs_h = np.ma.masked_where(dir0_phs_h == 0, dir0_tofs_h)   # use ph values to mask tofs and time
            dir0_time = np.ma.masked_where(dir0_phs_h == 0, dir0_time)

            dir0_phs_h = np.ma.masked_where(dir0_phs_h == 255, dir0_phs_h)
            dir0_tofs_h = np.ma.masked_where(dir0_phs_h == 255, dir0_tofs_h)
            dir0_time = np.ma.masked_where(dir0_phs_h == 255, dir0_time)

            dir0_phs_h = np.ma.compressed(dir0_phs_h)
            dir0_tofs_h = np.ma.compressed(dir0_tofs_h)
            dir0_time = np.ma.compressed(dir0_time)
            #print(dir0_phs_h[dir0_phs_h == 0], dir0_phs_h[dir0_phs_h == 255])'''

            return {'dir0_ph':dir0_phs_h, 'dir0_tof':dir0_tofs_h, 'dir0_time':dir0_time}

        elif dir == 'dir1':
            # ============== dir1 PH, TOF, TIME Allocation =================
            dir1_phs_h = np.array([], dtype='int32')
            dir1_tofs_h = np.array([], dtype='int32')
            dir1_time = np.array([], dtype='float64')

            for i in range(len(t_data)):
                dir1_phs_row = np.ma.compressed(dir1_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
                dir1_tofs_row = np.ma.compressed(dir1_tofs[i, :])
                # if len(dir0_PHs_row) != 0:

                cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
                if cnt_check:
                    # print('dir1', dir1_tofs_row)
                    dir1_phs_row = [0]
                    dir1_tofs_row = [0]

                # time arr allocation
                time_row = np.zeros(len(dir1_phs_row))  # zeros of time_row equal to number of phs per row
                time_row = time_row + t_data[i]
                dir1_time = np.concatenate((dir1_time, time_row), axis=0)  # consider append list, reshape in the future?

                dir1_phs_h = np.concatenate((dir1_phs_h, dir1_phs_row), axis=0)
                dir1_tofs_h = np.concatenate((dir1_tofs_h, dir1_tofs_row), axis=0)

            '''
            dir1_phs_h = np.ma.masked_where(dir1_phs_h == 0, dir1_phs_h)
            dir1_tofs_h = np.ma.masked_where(dir1_phs_h == 0, dir1_tofs_h)   # use ph values to mask tofs and time
            dir1_time = np.ma.masked_where(dir1_phs_h == 0, dir1_time)

            dir1_phs_h = np.ma.masked_where(dir1_phs_h == 255, dir1_phs_h)
            dir1_tofs_h = np.ma.masked_where(dir1_phs_h == 255, dir1_tofs_h)
            dir1_time = np.ma.masked_where(dir1_phs_h == 255, dir1_time)

            dir1_phs_h = np.ma.compressed(dir1_phs_h)
            dir1_tofs_h = np.ma.compressed(dir1_tofs_h)
            dir1_time = np.ma.compressed(dir1_time)'''

            return {'dir1_ph':dir1_phs_h, 'dir1_tof':dir1_tofs_h, 'dir1_time': dir1_time}

        elif dir == 'dir2':
            # =============== dir2 PH, TOF, TIME Allocation =================
            dir2_phs_h = np.array([], dtype='int32')
            dir2_tofs_h = np.array([], dtype='int32')
            dir2_time = np.array([], dtype='float64')

            for i in range(len(t_data)):
                dir2_phs_row = np.ma.compressed(dir2_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
                dir2_tofs_row = np.ma.compressed(dir2_tofs[i, :])

                cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
                if cnt_check:
                    #print('dir2', dir2_tofs_row)
                    dir2_phs_row = [0]
                    dir2_tofs_row = [0]

                # time arr allocation
                time_row = np.zeros(len(dir2_phs_row))  # zeros of time_row equal to number of phs per row
                time_row = time_row + t_data[i]
                dir2_time = np.concatenate((dir2_time, time_row), axis=0)  # consider append list, reshape in the future?

                dir2_phs_h = np.concatenate((dir2_phs_h, dir2_phs_row), axis=0)
                dir2_tofs_h = np.concatenate((dir2_tofs_h, dir2_tofs_row), axis=0)
            '''
            dir2_phs_h = np.ma.masked_where(dir2_phs_h == 0, dir2_phs_h)
            dir2_tofs_h = np.ma.masked_where(dir2_phs_h == 0, dir2_tofs_h)   # use ph values to mask tofs and time
            dir2_time = np.ma.masked_where(dir2_phs_h == 0, dir2_time)

            dir2_phs_h = np.ma.masked_where(dir2_phs_h == 255, dir2_phs_h)
            dir2_tofs_h = np.ma.masked_where(dir2_phs_h == 255, dir2_tofs_h)
            dir2_time = np.ma.masked_where(dir2_phs_h == 255, dir2_time)

            dir2_phs_h = np.ma.compressed(dir2_phs_h)
            dir2_tofs_h = np.ma.compressed(dir2_tofs_h)
            dir2_time = np.ma.compressed(dir2_time)'''

            return {'dir2_ph':dir2_phs_h, 'dir2_tof':dir2_tofs_h, 'dir2_time':dir2_time}

    def comp_tofph(self, cnts = False):  # Time series PH, TOF data manipulation/concatenation

        t_data = self.data['Time']

        # Direction Separation
        if cnts:
            dir0_phs, dir1_phs, dir2_phs = self.ph_dirsep(cnts=True)
        else:
            dir0_phs, dir1_phs, dir2_phs =  self.ph_dirsep()
        dir0_tofs, dir1_tofs, dir2_tofs = self.tof_dirsep()

        # ======= dir0 PH, TOF, TIME Allocation =========

        dir0_phs_h = np.array([], dtype='int32')
        dir0_tofs_h = np.array([], dtype='int32')
        dir0_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir0_phs_row = np.ma.compressed(dir0_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            dir0_tofs_row = np.ma.compressed(dir0_tofs[i, :])

            cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
            if cnt_check:
                dir0_phs_row = [0]
                dir0_tofs_row = [0]

            # Check num dim matched
            # print(dir0_phs_row.shape, dir0_tofs_row.shape)

            # time arr allocation
            time_row = np.zeros(len(dir0_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir0_time = np.concatenate((dir0_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir0_phs_h = np.concatenate((dir0_phs_h, dir0_phs_row), axis=0)
            dir0_tofs_h = np.concatenate((dir0_tofs_h, dir0_tofs_row), axis=0)

        # ============== dir1 PH, TOF, TIME Allocation =================
        dir1_phs_h = np.array([], dtype='int32')
        dir1_tofs_h = np.array([], dtype='int32')
        dir1_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir1_phs_row = np.ma.compressed(dir1_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            dir1_tofs_row = np.ma.compressed(dir1_tofs[i, :])
            # if len(dir0_PHs_row) != 0:

            cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
            if cnt_check:
                dir1_phs_row = [0]
                dir1_tofs_row = [0]


            # time arr allocation
            time_row = np.zeros(len(dir1_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir1_time = np.concatenate((dir1_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir1_phs_h = np.concatenate((dir1_phs_h, dir1_phs_row), axis=0)
            dir1_tofs_h = np.concatenate((dir1_tofs_h, dir1_tofs_row), axis=0)

        # =============== dir2 PH, TOF, TIME Allocation =================
        dir2_phs_h = np.array([], dtype='int32')
        dir2_tofs_h = np.array([], dtype='int32')
        dir2_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir2_phs_row = np.ma.compressed(dir2_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            dir2_tofs_row = np.ma.compressed(dir2_tofs[i, :])

            cnt_check = self.cnter_chek(i)  # Check for valid counters, i.e stop0 and stop1 and stop2 != 0
            if cnt_check:
                dir2_phs_row = [0]
                dir2_tofs_row = [0]

            # time arr allocation
            time_row = np.zeros(len(dir2_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir2_time = np.concatenate((dir2_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir2_phs_h = np.concatenate((dir2_phs_h, dir2_phs_row), axis=0)
            dir2_tofs_h = np.concatenate((dir2_tofs_h, dir2_tofs_row), axis=0)

        if cnts:
            # dir 0 and 255 values checking
            dir0_255 = [dir0_phs_h == 255]
            dir1_255 = [dir1_phs_h == 255]
            dir2_255 = [dir2_phs_h == 255]
            print('\n 255 PH-value counts: ')
            print('dir0: ', len(dir0_255))
            print('dir1: ', len(dir1_255))
            print('dir2: ', len(dir2_255))

        # ============== Masking 0 and 255 PH values ===================

        # dir0
        dir0_phs_h = np.ma.masked_where(dir0_phs_h == 0, dir0_phs_h)
        dir0_tofs_h = np.ma.masked_where(dir0_phs_h == 0, dir0_tofs_h)   # use ph values to mask tofs and time
        dir0_time = np.ma.masked_where(dir0_phs_h == 0, dir0_time)

        dir0_phs_h = np.ma.masked_where(dir0_phs_h == 255, dir0_phs_h)
        dir0_tofs_h = np.ma.masked_where(dir0_phs_h == 255, dir0_tofs_h)
        dir0_time = np.ma.masked_where(dir0_phs_h == 255, dir0_time)
        dir0_phs_h = np.ma.compressed(dir0_phs_h)
        dir0_tofs_h = np.ma.compressed(dir0_tofs_h)
        dir0_time = np.ma.compressed(dir0_time)

        # dir1
        dir1_phs_h = np.ma.masked_where(dir1_phs_h == 0, dir1_phs_h)
        dir1_tofs_h = np.ma.masked_where(dir1_phs_h == 0, dir1_tofs_h)   # use ph values to mask tofs and time
        dir1_time = np.ma.masked_where(dir1_phs_h == 0, dir1_time)

        dir1_phs_h = np.ma.masked_where(dir1_phs_h == 255, dir1_phs_h)
        dir1_tofs_h = np.ma.masked_where(dir1_phs_h == 255, dir1_tofs_h)
        dir1_time = np.ma.masked_where(dir1_phs_h == 255, dir1_time)
        dir1_phs_h = np.ma.compressed(dir1_phs_h)
        dir1_tofs_h = np.ma.compressed(dir1_tofs_h)
        dir1_time = np.ma.compressed(dir1_time)

        # dir2
        dir2_phs_h = np.ma.masked_where(dir2_phs_h == 0, dir2_phs_h)
        dir2_tofs_h = np.ma.masked_where(dir2_phs_h == 0, dir2_tofs_h)   # use ph values to mask tofs and time
        dir2_time = np.ma.masked_where(dir2_phs_h == 0, dir2_time)

        dir2_phs_h = np.ma.masked_where(dir2_phs_h == 255, dir2_phs_h)
        dir2_tofs_h = np.ma.masked_where(dir2_phs_h == 255, dir2_tofs_h)
        dir2_time = np.ma.masked_where(dir2_phs_h == 255, dir2_time)
        dir2_phs_h = np.ma.compressed(dir2_phs_h)
        dir2_tofs_h = np.ma.compressed(dir2_tofs_h)
        dir2_time = np.ma.compressed(dir2_time)

        if cnts:
            # Check counts
            print('\n Remaining PH Counts after 0 and 255 Masking: ')
            print('dir0: ', dir0_phs_h.size)
            print('dir1: ', dir1_phs_h.size)
            print('dir2: ', dir2_phs_h.size)

        dir_tofph = {'dir0_ph': dir0_phs_h, 'dir0_tof': dir0_tofs_h, 'dir0_time': dir0_time, 'dir1_ph': dir1_phs_h, 'dir1_tof': dir1_tofs_h, 'dir1_time': dir1_time,
                    'dir2_ph': dir2_phs_h, 'dir2_tof': dir2_tofs_h, 'dir2_time': dir2_time}

        return dir_tofph     # time, ph, tof values for plotting!

    def xticks(self, ax, mintick, maxtick, dtick, n):  # Tick Customization

        d_format = '%H:%M:%S'

        # Enable Minor Ticks
        minorlocator = MultipleLocator(5)
        ax.xaxis.set_minor_locator(minorlocator)

        # Tick formatting to UTC
        utc_maxtick = time.gmtime(maxtick)
        utc_mintick = time.gmtime(mintick)

        # Range in seconds
        st_sec = utc_mintick.tm_hour * 3600 + utc_mintick.tm_min * 60 + utc_mintick.tm_sec
        et_sec = utc_maxtick.tm_hour * 3600 + utc_maxtick.tm_min * 60 + utc_maxtick.tm_sec
        dsec = et_sec - st_sec

        # print(st_sec/60)
        # print(dsec/60)

        if dtick == 'min':
            m_range = np.arange(0, dsec / 60, 6)  # dx for minute range

            # make new division of original array
            new_range = np.zeros(len(m_range))
            dx = (maxtick - mintick) / len(m_range)
            for i in range(len(m_range)):
                new_range[i] = new_range[i] + mintick  # use for new tick range
                mintick += dx
            # print(new_range)

            # UTC conversion of new tick range
            new_utc = [time.gmtime(t) for t in new_range]

            # Convert to utc,readable date string
            tick_label = []
            for x in new_utc:
                tick_label.append(time.strftime(d_format, x))  # new labels for ticks
            # print(tick_label)

            return {'major_tick': new_range, 'tick_label': tick_label}

        elif dtick == 'hour':

            h_range = np.arange(0, dsec / 3600, 0.25)

            # make new division of original array
            new_range = np.zeros(len(h_range))
            dx = (maxtick - mintick) / len(h_range)
            for i in range(len(h_range)):
                new_range[i] = new_range[i] + mintick  # use for new tick range
                mintick += dx
            # print(new_range)

            # UTC conversion of new tick range
            new_utc = [time.gmtime(t) for t in new_range]

            # Convert to utc,readable date string
            tick_label = []
            for x in new_utc:
                tick_label.append(time.strftime(d_format, x))  # new labels for ticks
            # print(tick_label)

            return {'major_tick': new_range, 'tick_label': tick_label}

        elif dtick == 'n_hours':

            # Enable Minor Ticks
            minorlocator = MultipleLocator(50)
            ax.xaxis.set_minor_locator(minorlocator)
            d_range = np.arange(0, dsec / (3600 * n), 0.5)
            # print(dsec)

            # make new division of original array
            new_range = np.zeros(len(d_range))
            dx = (maxtick - mintick) / len(d_range)
            for i in range(len(d_range)):
                new_range[i] = new_range[i] + mintick  # use for new tick range
                mintick += dx
            # print(new_range)

            # UTC conversion of new tick range
            new_utc = [time.gmtime(t) for t in new_range]

            # Convert to utc,readable date string
            tick_label = []
            for x in new_utc:
                tick_label.append(time.strftime(d_format, x))  # new labels for ticks
            # print(tick_label)

            return {'major_tick': new_range, 'tick_label': tick_label}

    def phdist(self, binning=False):  # Ph histogram plot(uses direct filtering)
        f_name = self.filename  # filename
        PHs = self.data['PHs']
        dirs = self.data['dirs']

        # direction/sectors separation
        dir0_PHs = PHs[dirs == 0]
        dir0_cnts = len(dir0_PHs)  # bin number for dir0
        dir1_PHs = PHs[dirs == 1]
        dir1_cnts = len(dir1_PHs)
        dir2_PHs = PHs[dirs == 2]
        dir2_cnts = len(dir2_PHs)
        ph_max = 300
        n_bins = 100

        # Plotting
        f, dirarr = plt.subplots(3)
        if binning:  # if raw data
            n, bins, patches = dirarr[0].hist(dir0_PHs, n_bins, normed=1, label='dir0')
            dirarr[0].set_title('PH Distribution raw data, binning =  %s' % str(n_bins))
            dirarr[0].text(ph_max / 2, np.amax(n), f_name[:-3], ha='center', va='center')
            dirarr[1].hist(dir1_PHs, bins, normed=1, label='dir1')
            dirarr[2].hist(dir2_PHs, bins, normed=1, label='dir2')
        else:  # if screened
            n, bins, patches = dirarr[0].hist(dir0_PHs, dir0_cnts, normed=1, label='dir0')  # no binning
            dirarr[0].set_title('PH Distribution, coinflag = 0, no binning')
            dirarr[0].text(ph_max / 2, np.amax(n), f_name[:-3], ha='center', va='center')
            dirarr[1].hist(dir0_PHs, dir1_cnts, normed=1, label='dir1')  # no binning
            dirarr[2].hist(dir0_PHs, dir2_cnts, normed=1, label='dir2')  # no binning

        for i in range(dirarr.size):
            dirarr[i].legend()
            dirarr[i].set_yscale('log')
            dirarr[i].set_ylabel('Counts')
            dirarr[i].set_xlim(0, ph_max)

        dirarr[2].set_xlabel('PH', fontsize=14)

        plt.show()

    def t_phd(self, phd_bin):  # Plots only time series phd without histogram (obsolete)

        if not phd_bin:  # if bin is empty
            phd_bin = [100, 255]

        phs = self.data['PHs']
        dirs = self.data['dirs']
        time = self.data['Time']

        # direction/sectors separation
        dir0_phs = np.ma.masked_where(dirs != 0, phs)  # mask elements in PHs not equal to 0 in dirs
        dir1_phs = np.ma.masked_where(dirs != 1, phs)
        dir2_phs = np.ma.masked_where(dirs != 2, phs)

        # masking check
        # print(dir0_phs[~np.ma.getmask(dir0_phs)])


        fig, axarr = plt.subplots(3, sharex=True, figsize=(13, 7))

        axarr[0].set_ylabel('PHs\ndir0', multialignment='center')

        dir0_phs_h = np.array([], dtype='int32')
        dir0_time = np.array([], dtype='float64')

        for i in range(len(time)):
            dir0_phs_row = np.ma.compressed(dir0_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            # if len(dir0_PHs_row) != 0:

            # time arr allocation
            time_row = np.zeros(len(dir0_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + time[i]
            dir0_time = np.concatenate((dir0_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir0_phs_h = np.concatenate((dir0_phs_h, dir0_phs_row), axis=0)

        d0_h, d0_xedges, d0_yedges = np.histogram2d(dir0_time, dir0_phs_h, bins=phd_bin)

        d0X, d0Y = np.meshgrid(d0_xedges, d0_yedges)
        im0 = axarr[0].pcolormesh(d0X, d0Y, d0_h.T)
        # axarr[0].axis([d0X.min(), d0X.max(), d0Y.min(), d0Y.max()])

        # axarr[0].set_aspect(2)
        # plt.gca().set_aspect(2)

        axarr[0].set_ylim([d0Y.min() - 5, d0Y.max() + 5])

        plt.colorbar(im0, ax=axarr[0])

        axarr[1].set_ylabel('PHs\ndir1', multialignment='center')
        dir1_phs_h = np.array([], dtype='int32')
        dir1_time = np.array([], dtype='float64')

        for i in range(len(time)):
            dir1_phs_row = np.ma.compressed(dir1_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            # if len(dir0_PHs_row) != 0:

            # time arr allocation
            time_row = np.zeros(len(dir1_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + time[i]
            dir1_time = np.concatenate((dir1_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir1_phs_h = np.concatenate((dir1_phs_h, dir1_phs_row), axis=0)

        d1_h, d1_xedges, d1_yedges = np.histogram2d(dir1_time, dir1_phs_h, bins=phd_bin)

        d1X, d1Y = np.meshgrid(d1_xedges, d1_yedges)
        im1 = axarr[1].pcolormesh(d1X, d1Y, d1_h.T)
        axarr[1].set_ylim([d1Y.min() - 5, d1Y.max() + 5])

        # axarr[1].set_aspect(2)
        # plt.gca().set_aspect(2)

        # plt.scatter(dir1_time, dir1_phs_h)
        plt.colorbar(im1, ax=axarr[1])

        axarr[2].set_ylabel('PHs\ndir2', multialignment='center')
        dir2_phs_h = np.array([], dtype='int32')
        dir2_time = np.array([], dtype='float64')

        for i in range(len(time)):
            dir2_phs_row = np.ma.compressed(dir2_phs[i, :])  # compress unmasked phs values every row (1 sec interval)

            # time arr allocation
            time_row = np.zeros(len(dir2_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + time[i]
            dir2_time = np.concatenate((dir2_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir2_phs_h = np.concatenate((dir2_phs_h, dir2_phs_row), axis=0)

        d2_h, d2_xedges, d2_yedges = np.histogram2d(dir2_time, dir2_phs_h, bins=phd_bin)

        d2X, d2Y = np.meshgrid(d2_xedges, d2_yedges)
        im2 = axarr[2].pcolormesh(d2X, d2Y, d2_h.T)
        # axarr[2].set_aspect(2)

        plt.colorbar(im2, ax=axarr[2])

        # plt.gca().set_aspect(2)

        # axarr[2].axis([d2X.min(), d2X.max(), d2Y.min(), d2Y.max()])

        axarr[2].set_xlim([d2X.min(), d2X.max()])
        axarr[2].set_ylim([d2Y.min() - 5, d2Y.max() + 5])

        # Setting ticks and tick labels for all subplots
        xticks = self.xticks(axarr[2], mintick=d2X.min(), maxtick=d2X.max(), dtick='min')

        plt.setp(axarr, xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])

        axarr[2].set_xlabel('Time')

        # plt.scatter(dir2_time, dir2_phs_h)

        plt.show()

    def plt_tph(self, phd_bin, savefig=False, cnts = False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Call Time series PH Distribution function
        if cnts:
            dir_phs_h = self.comp_tofph(cnts = True)
        else:
            dir_phs_h = self.comp_tofph()
        # dir_phs_h = self.append_method()  # Plan B for efficiency

        # Call for general info
        info = self.info()

        # Setup Figure and Subplots
        plt.close('all')

        fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(16, 8))

        fig.suptitle('Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'], fontsize=11,
                     y=0.04)

        axarr[0, 0].set_ylabel('PH\ndir0', multialignment='center', fontsize=10)
        axarr[1, 0].set_ylabel('PH\ndir1', multialignment='center', fontsize=10)
        axarr[2, 0].set_ylabel('PH\ndir2', multialignment='center', fontsize=10)

        axarr[0, 1].set_ylabel('Counts\ndir0', multialignment='center', fontsize=10)
        axarr[1, 1].set_ylabel('Counts\ndir1', multialignment='center', fontsize=10)
        axarr[2, 1].set_ylabel('Counts\ndir2', multialignment='center', fontsize=10)

        # Plotting dir0
        d0_h, d0_xedges, d0_yedges = np.histogram2d(dir_phs_h['dir0_time'], dir_phs_h['dir0_ph'], bins=phd_bin)
        d0X, d0Y = np.meshgrid(d0_xedges, d0_yedges)
        im0 = axarr[0, 0].pcolormesh(d0X, d0Y, d0_h.T)
        axarr[0, 0].set_ylim([d0Y.min(), d0Y.max()])
        plt.colorbar(im0, ax=axarr[0, 0])
        axarr[0, 0].set_xlim([d0X.min(), d0X.max()])

        # Plotting dir1
        d1_h, d1_xedges, d1_yedges = np.histogram2d(dir_phs_h['dir1_time'], dir_phs_h['dir1_ph'], bins=phd_bin)
        d1X, d1Y = np.meshgrid(d1_xedges, d1_yedges)
        im1 = axarr[1, 0].pcolormesh(d1X, d1Y, d1_h.T)
        axarr[1, 0].set_ylim([d1Y.min(), d1Y.max()])
        plt.colorbar(im1, ax=axarr[1, 0])
        axarr[1, 0].set_xlim([d1X.min(), d1X.max()])

        # Plotting dir2
        d2_h, d2_xedges, d2_yedges = np.histogram2d(dir_phs_h['dir2_time'], dir_phs_h['dir2_ph'], bins=phd_bin)
        d2X, d2Y = np.meshgrid(d2_xedges, d2_yedges)
        im2 = axarr[2, 0].pcolormesh(d2X, d2Y, d2_h.T)
        plt.colorbar(im2, ax=axarr[2, 0])

        axarr[2, 0].set_xlim([d2X.min(), d2X.max()])
        axarr[2, 0].set_ylim([d2Y.min(), d2Y.max()])

        # Setting ticks and tick labels for all first column subplots
        t_diff = info['dt']

        val_time = self.plt_tcounter(axarr[3, 0])

        # Identify time duration
        if t_diff <= 1800:
            xticks = self.xticks(axarr[3, 0], mintick=val_time.min(), maxtick=val_time.max(), dtick='min', n=0)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        elif t_diff <= 3600:
            xticks = self.xticks(axarr[3, 0], mintick=val_time.min(), maxtick= val_time.max(), dtick='hour', n=0)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        else:
            n_hours = int(float(t_diff)/3600)
            xticks = self.xticks(axarr[3, 0], mintick=val_time.min(), maxtick=val_time.max(), dtick='n_hours', n=n_hours)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])

        axarr[3, 0].set_xlabel('Time', fontsize=10)

        # Histogram Plot
        '''axarr[0, 1].hist(dir_phs_h['dir0_ph'], bins=phd_bin[1], normed=0, label='dir0', histtype='stepfilled',
                         facecolor='w')
        axarr[1, 1].hist(dir_phs_h['dir1_ph'], bins=phd_bin[1], normed=0, label='dir1', histtype='stepfilled',
                         facecolor='w')
        axarr[2, 1].hist(dir_phs_h['dir2_ph'], bins=phd_bin[1], normed=0, label='dir2', histtype='stepfilled',
                         facecolor='w')'''

        hist0, bins0 = np.histogram(dir_phs_h['dir0_ph'], bins=phd_bin[1])
        hist1, bins1 = np.histogram(dir_phs_h['dir1_ph'], bins=phd_bin[1])
        hist2, bins2 = np.histogram(dir_phs_h['dir2_ph'], bins=phd_bin[1])

        # Normalize with time
        hist0 = hist0/float(len(val_time))
        hist1 = hist1/float(len(val_time))
        hist2 = hist2/float(len(val_time))

        axarr[0, 1].plot(bins0[:-1], hist0, '-', color='k')
        axarr[1, 1].plot(bins1[:-1], hist1, '-', color='k')
        axarr[2, 1].plot(bins2[:-1], hist2, '-', color='k')

        # Enable Minor Ticks
        minorlocator = MultipleLocator(10)

        for i in range(axarr.size - 4):
            axarr[i, 1].xaxis.set_minor_locator(minorlocator)

            # Determining log scale (crude way)
            if dir_phs_h['dir0_ph'].max() - dir_phs_h['dir0_ph'].min() > 100:
                axarr[i, 1].set_yscale('log')
            axarr[i, 1].set_xlim(0, 255)

        axarr[2, 1].set_xlabel('PH', fontsize=10)
        fig.subplots_adjust(hspace=0.2)

        plt.tight_layout()
        if savefig:
            print('Saving figure..')
            # Uncomment next lines and provide the right path
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/'+self.filename[:-3]+'_tph.png')
            # plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/data/figs/'+self.filename[5:-3]+'.png')
        plt.show()

    def plt_ttof(self, tof_bin, savefig=False):

        if not tof_bin:  # if bin is empty
            tof_bin = self.tofbin

        # Call Time series PH Distribution function
        dir_tofs_h = self.comp_tofph()

        # Call for general info
        info = self.info()

        # Setup Figure and Subplots
        plt.close('all')

        fig, axarr = plt.subplots(nrows=4, ncols=2, figsize=(16, 8))

        fig.suptitle('Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'], fontsize=11,
                     y=0.04)

        axarr[0, 0].set_ylabel('TOF\ndir0', multialignment='center', fontsize=10)
        axarr[1, 0].set_ylabel('TOF\ndir1', multialignment='center', fontsize=10)
        axarr[2, 0].set_ylabel('TOF\ndir2', multialignment='center', fontsize=10)

        axarr[0, 1].set_ylabel('cnts/s/ns \ndir0', multialignment='center', fontsize=10)
        axarr[1, 1].set_ylabel('cnts/s/ns \ndir1', multialignment='center', fontsize=10)
        axarr[2, 1].set_ylabel('cnts/s/ns \ndir2', multialignment='center', fontsize=10)

        # Plotting dir0
        d0_h, d0_xedges, d0_yedges = np.histogram2d(dir_tofs_h['dir0_time'], dir_tofs_h['dir0_tof'], bins=tof_bin)
        d0X, d0Y = np.meshgrid(d0_xedges, d0_yedges)
        im0 = axarr[0, 0].pcolormesh(d0X, d0Y, d0_h.T)
        axarr[0, 0].set_ylim([0, 2048])
        plt.colorbar(im0, ax=axarr[0, 0])
        axarr[0, 0].set_xlim([d0X.min(), d0X.max()])

        # Plotting dir1
        d1_h, d1_xedges, d1_yedges = np.histogram2d(dir_tofs_h['dir1_time'], dir_tofs_h['dir1_tof'], bins=tof_bin)
        d1X, d1Y = np.meshgrid(d1_xedges, d1_yedges)
        im1 = axarr[1, 0].pcolormesh(d1X, d1Y, d1_h.T)
        axarr[1, 0].set_ylim([0, 2048])
        plt.colorbar(im1, ax=axarr[1, 0])
        axarr[1, 0].set_xlim([d1X.min(), d1X.max()])

        # Plotting dir2
        d2_h, d2_xedges, d2_yedges = np.histogram2d(dir_tofs_h['dir2_time'], dir_tofs_h['dir2_tof'], bins=tof_bin)
        d2X, d2Y = np.meshgrid(d2_xedges, d2_yedges)
        im2 = axarr[2, 0].pcolormesh(d2X, d2Y, d2_h.T)
        plt.colorbar(im2, ax=axarr[2, 0])

        axarr[2, 0].set_xlim([d2X.min(), d2X.max()])
        axarr[2, 0].set_ylim([0, 2048])

        # Setting ticks and tick labels for all first column subplots
        t_diff = info['dt']

        val_time = self.plt_tcounter(axarr[3, 0])

        # Identify time duration
        if t_diff <= 1800:
            xticks = self.xticks(axarr[3, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='min', n=0)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        elif t_diff <= 3600:
            xticks = self.xticks(axarr[3, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='hour', n=0)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        else:
            n_hours = int(float(t_diff)/3600)
            xticks = self.xticks(axarr[3, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='n_hours', n=n_hours)
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])

        axarr[3, 0].set_xlabel('Time', fontsize=10)

        hist0, bins0 = np.histogram(dir_tofs_h['dir0_tof'], bins=tof_bin[1])
        hist1, bins1 = np.histogram(dir_tofs_h['dir1_tof'], bins=tof_bin[1])
        hist2, bins2 = np.histogram(dir_tofs_h['dir2_tof'], bins=tof_bin[1])

        # Normalize with time
        hist0 = hist0/float(len(val_time))
        hist1 = hist1/float(len(val_time))
        hist2 = hist2/float(len(val_time))

        axarr[0, 1].plot(bins0[:-1], hist0, '-', color='k')
        axarr[1, 1].plot(bins1[:-1], hist1, '-', color='k')
        axarr[2, 1].plot(bins2[:-1], hist2, '-', color='k')


        '''# Histogram Plot
        axarr[0, 1].hist(dir_tofs_h['dir0_tof'], bins=tof_bin[1], normed=0, label='dir0', histtype='stepfilled',
                         facecolor='w')
        axarr[1, 1].hist(dir_tofs_h['dir1_tof'], bins=tof_bin[1], normed=0, label='dir1', histtype='stepfilled',
                         facecolor='w')
        axarr[2, 1].hist(dir_tofs_h['dir2_tof'], bins=tof_bin[1], normed=0, label='dir2', histtype='stepfilled',
                         facecolor='w')'''

        # Enable Minor Ticks
        minorlocator = MultipleLocator(50)
        majorlocator = MultipleLocator(250)

        for i in range(axarr.size - 4):
            axarr[i, 1].xaxis.set_minor_locator(minorlocator)
            axarr[i, 1].xaxis.set_major_locator(majorlocator)
            axarr[i, 1].set_xlim(0, 2048)
            #axarr[i, 1].set_yscale('log')

        axarr[3, 1].set_xlabel('TOF[ns]', fontsize=10)
        fig.subplots_adjust(hspace=0.2)

        #plt.tight_layout()
        if savefig:
            print('Saving figure..')
            # Uncomment next lines and provide the right path
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/'+self.filename[:-3]+'_ttof.png')
            # plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/data/figs/'+self.filename[5:-3]+'.png')
        plt.show()

    def plt_tofph(self, tof_bin, savefig=False):  # PH -TOF plotting

        if not tof_bin:  # if bin is empty
            tof_bin = self.tofbin
        # Call Time series PH, TOF computation
        dir_tofph = self.comp_tofph()


        # Call for general info
        info = self.info()

        # Setup Figure and Subplots
        plt.close('all')

        fig, axarr = plt.subplots(nrows=3, sharex= True, figsize=(15, 9))

        fig.suptitle('Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'], fontsize=11,
                     y=0.04, x= 0.85)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.08, hspace=0.1)

        axarr[0].set_ylabel('PHs\ndir0', multialignment='center', fontsize=10)
        axarr[1].set_ylabel('PHs\ndir1', multialignment='center', fontsize=10)
        axarr[2].set_ylabel('PHs\ndir2', multialignment='center', fontsize=10)


        # Plotting dir0
        d0_h, d0_xedges, d0_yedges = np.histogram2d(dir_tofph['dir0_tof'], dir_tofph['dir0_ph'], bins=tof_bin)
        d0X, d0Y = np.meshgrid(d0_xedges, d0_yedges)

        # normalize
        d0_norm = d0_h/float(len(dir_tofph['dir0_tof']))

        im0 = axarr[0].pcolormesh(d0X, d0Y, d0_norm.T)
        axarr[0].set_ylim([d0Y.min(), d0Y.max()])
        cb0 = fig.colorbar(im0, ax=axarr[0])
        cb0.formatter.set_powerlimits((0, 0))
        cb0.update_ticks()
        axarr[0].set_xlim([0, 2048])
        #axarr[0].set_xlim([d0X.min(), d0X.max()])

        # Plotting dir1
        d1_h, d1_xedges, d1_yedges = np.histogram2d(dir_tofph['dir1_tof'], dir_tofph['dir1_ph'], bins=tof_bin)
        d1X, d1Y = np.meshgrid(d1_xedges, d1_yedges)

        # normalize
        d1_norm = d1_h/float(len(dir_tofph['dir1_tof']))

        im1 = axarr[1].pcolormesh(d1X, d1Y, d1_norm.T)
        axarr[1].set_ylim([d1Y.min(), d1Y.max()])
        cb1 = fig.colorbar(im1, ax=axarr[1])
        cb1.formatter.set_powerlimits((0, 0))
        cb1.update_ticks()
        #axarr[1].set_xlim([d1X.min(), d1X.max()])
        axarr[1].set_xlim([0, 2048])

        # Plotting dir2
        d2_h, d2_xedges, d2_yedges = np.histogram2d(dir_tofph['dir2_tof'], dir_tofph['dir2_ph'], bins=tof_bin)
        d2X, d2Y = np.meshgrid(d2_xedges, d2_yedges)

        # normalize
        d2_norm = d2_h/float(len(dir_tofph['dir2_tof']))

        im2 = axarr[2].pcolormesh(d2X, d2Y, d2_norm.T)
        cb2 = fig.colorbar(im2, ax=axarr[2])
        cb2.formatter.set_powerlimits((0, 0))
        cb2.update_ticks()
        axarr[2].set_xlim([0, 2048])
        #axarr[2].imshow(np.rot90(d2_norm), interpolation='nearest', norm = colors.PowerNorm(gamma=1./2.),
                                            #extent=[0, 2048, 0, 255],aspect='auto')
        #axarr[2].set_xlim([d2X.min(), d2X.max()])
        axarr[2].set_ylim([d2Y.min(), d2Y.max()])

         # Enable Minor Ticks
        minorlocator = MultipleLocator(50)
        majorlocator = MultipleLocator(250)

        for i in range(axarr.size):
            axarr[i].xaxis.set_minor_locator(minorlocator)
            axarr[i].xaxis.set_major_locator(majorlocator)

        axarr[2].set_xlabel('TOF [ns]', fontsize=10)

        if savefig:
            print('Saving figure..')
            # Uncomment next lines and provide the right path
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/'+self.filename[:-3]+'_tofph.png')
            # plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/data/figs/'+self.filename[5:-3]+'.png')
        plt.show()

    def test_store(self, path, file):

        # Call Time series PH, TOF function
        dir0 = self.test_comp('dir0')
        dir1 = self.test_comp('dir1')
        dir2 = self.test_comp('dir2')

        # Call Duty Cycle Computation
        dclist, low_cnt = self.duty_cycle(self.data['Cnts'])      # duty cycle list


        # Call info
        info = self.info()
        f_name = info['Filename']  # for id
        path += '/json/'  # new path including sub directory 'json'
        # print (path)

        # Convert to list
        dir0_phs_h = dir0['dir0_ph'].tolist()
        dir0_tofs_h = dir0['dir0_tof'].tolist()
        dir0_time = dir0['dir0_time'].tolist()

        dir1_phs_h = dir1['dir1_ph'].tolist()
        dir1_tofs_h = dir1['dir1_tof'].tolist()
        dir1_time = dir1['dir1_time'].tolist()

        dir2_phs_h = dir2['dir2_ph'].tolist()
        dir2_tofs_h = dir2['dir2_tof'].tolist()
        dir2_time = dir2['dir2_time'].tolist()

        f_date = str(info['Date'])
        st = info['Start']
        et = info['End']
        dt = str(info['dt'])

        # Dictionary with data info (Note: Use Start and End for observation time, trimmed data of valid events)
        data_file = {'Filename': f_name, 'Date': f_date, 'dclist': dclist, 'Time': self.data['Time'].tolist(), 'Start': st, 'End': et, 'dt': dt, 'dir0_ph': dir0_phs_h, 'dir0_tof': dir0_tofs_h, 'dir0_time': dir0_time,
                     'dir1_ph': dir1_phs_h, 'dir1_tof': dir1_tofs_h, 'dir1_time': dir1_time, 'dir2_ph': dir2_phs_h, 'dir2_tof': dir2_tofs_h, 'dir2_time': dir2_time, 'low_cnt': low_cnt
                     }

        if file.lower().endswith('.txt'):  # if text file containing list
            # Write to file
            if os.path.isfile(path + file[:-4] + '.json'):  # if file exists,

                print('File exists!')
                with open(path + file[:-4] + '.json', 'a') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)  # append each dict to the file
                    print('appending...')
                    f.write(os.linesep)  # 1 dict per line in the file
                f.close()

            else:
                with open(path + file[:-4] + '.json', 'w') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    f.write(os.linesep)
                    # print(open(path + file[:-4] + '.json', 'r').read())

        elif file.lower().endswith('.nc'):  # if file is a single netcdf file
            # Write to file
            if os.path.isfile(path + f_name[:11] + '.json'):  # if file exists,

                print('File exists!')
                with open(path + f_name[:11] + '.json', 'a') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    print('appending...')

                    f.write(os.linesep)
                f.close()

            else:
                with open(path + f_name[:11] + '.json', 'w') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    f.write(os.linesep)

                    # print(open(path + f_name[:-3] + '.json', 'r').read())

    def err_store(self, datadict):


        # Call info
        info = self.info()
        f_name = info['Filename']  # for id
        dirpath = os.path.dirname(os.path.abspath(f_name))  # file path
        dirpath += '/data/'  # new path including sub directory 'json'

        file = f_name[0:7] + '_ErrCnt'

        # Store files
        if os.path.isfile(dirpath + file + '.json'):  # if file exists,

            print('File exists!')
            with open(dirpath + file + '.json', 'a') as f:
                json.dump(datadict, f, separators=(',', ':'), sort_keys=True)  # append each dict to the file
                print('appending...')
                f.write(os.linesep)  # 1 dict per line in the file
            f.close()

        else:
            with open(dirpath + file + '.json', 'w') as f:
                json.dump(datadict, f, separators=(',', ':'), sort_keys=True)
                f.write(os.linesep)

    @property
    def phbin(self):
        return self._phbin

    def set_phbin(self, phbin):
        self._phbin = phbin

    @property
    def tofbin(self):
        return self._tofbin

    def set_tofbin(self, tofbin):
        self._tofbin = tofbin


def raw_filter(args):  # sub-function
    # Parsing Arguments
    parser = argparse.ArgumentParser(prog='python3 plt_filter.py')
    parser.add_argument("id", type=argparse.FileType('r'), help="input filename")
    parser.add_argument("-v", "--verbose", action="store_true", help = "Display variable names in netCDF file")
    parser.add_argument("-dinfo", "--dinfo", action="store_true", help="Display date information")
    parser.add_argument("-cnts", "--counts", action="store_true", help = "Display counts information")
    parser.add_argument("-tph", "--time_ph", action="store_true", help="Time series PH plotting")
    parser.add_argument("-ttof", "--time_tof", action ="store_true", help="Time series TOF plotting")
    parser.add_argument("-tofph", "--tof_ph", action = "store_true", help="TOF - PH plotting")
    parser.add_argument("-phdist", "--ph_dist", action="store_true", help="PHs Distribution through filtering")
    parser.add_argument("-save", "--store", action="store_true", help="Saves filtered data to json file")
    parser.add_argument("-savefig", "--sfig", action="store_true", help="Saves figure to directory")
    parser.add_argument("-cntplt", "--cntplt", action="store_true", help="Plot Couters 0,1 and 2")
    parser.add_argument("-dir", "--dir", type=str, action="store", help="input which direction, options: dir0, dir1, dir2")
    parser.add_argument("-rawevent", "--rawevent", action = "store_true", help="Plot valid events, accumulated time and ratio in VSO")

    args = parser.parse_args()

    dirpath = os.path.dirname(os.path.abspath(args.id.name))  # file path
    # print (path)

    nc_f = os.path.basename(args.id.name)  # filename
    # print(nc_f)
    full_path = os.path.abspath(args.id.name)

    vspice.init()

    # Pre-defined Var
    R_v = 6.051850 * 10 ** 3
    binning = [0.25, 0.25]


    phd_bin = [150, 256]
    tof_bin = [256, 500]

    # File Checking
    if nc_f.lower().endswith('.txt'):
        f_error = []

        # Read file containing list
        with open(full_path, 'r') as f:
            contents = [line.rstrip('\n') for line in f]
            success = 0
            failed = 0
            for content in contents:
                try:
                    full_path = dirpath + '/' + content
                    raw_data = raw.NPD_raw_read(full_path, args.verbose)  # reading netCDF file
                    flt_data = RawFilter(raw_data)  # new filtered data object 'flt_data'
                    if args.verbose:
                        # Print netCDF info and new variables, ie. PHs, TOFs
                        print (flt_data.var())

                    plot_flt = RawPlot(flt_data.filter_data,
                                       content)  # filtered 'flt_data' object, assign to 'plot_flt'
                    if args.ph_dist:
                        # Plotting PH histogram of filtered data (coinflag == 0)
                        plot_flt.phdist()

                    # masking
                    mask_data = flt_data.mask  # masking data dict only
                    plot_mask = RawPlot(mask_data, content)  # masking data, assign to plot_mask object

                    if args.time_ph:
                        if args.sfig:
                            # Plotting Time series PHD
                            plot_mask.plt_tph(phd_bin, savefig=True)  # Concatenate method (slower but stable)
                            # plot_mask.t_phd(phd_bin)  # Obsolete
                        else:
                            plot_mask.plt_tph(phd_bin)

                    if args.time_tof:
                        if args.sfig:
                            # Plotting Time series TOF
                            plot_mask.plt_ttof(tof_bin, savefig=True)
                        else:
                            plot_mask.plt_ttof(tof_bin)

                    if args.tof_ph:
                        if args.sfig:
                            # Plotting TOF - PH
                            plot_mask.plt_tofph(phd_bin, savefig=True)

                        else:
                            plot_mask.plt_tofph(phd_bin)

                    if args.store:
                        plot_mask.test_store(dirpath, nc_f)
                        # plot_mask.store(path, nc_f)  # copying
                        success += 1
                except (ValueError, RuntimeError, IndexError) as err:
                    print('Error: ', err)
                    f_error.append(content)
                    failed += 1

            with open(dirpath + '/' + 'err' + nc_f, 'a') as file:
                for item in f_error:
                    file.write('%s\n' % item)
            print('No. of Files saved: ', success)
            print('No. failed: ', failed)

    elif nc_f.lower().endswith('.nc'):
        raw_data = raw.NPD_raw_read(full_path, args.verbose)  # reading netCDF file
        flt_data = RawFilter(raw_data)  # new filtered data object 'flt_data'
        if args.verbose:
            # Print netCDF info and new variables, ie. PHs, TOFs
            print (flt_data.var())

        plot_flt = RawPlot(flt_data.filter_data, nc_f)  # filtered 'flt_data' object, assign to 'plot_flt'
        if args.ph_dist:
            # Plotting PH histogram of filtered data (coinflag == 0)
            plot_flt.phdist()

        if args.cntplt:
            plot_flt.plt_tcounter()

        if args.dinfo:
            # Display date information of File
            print(plot_flt.info(disp=True))
        # masking
        mask_data = flt_data.mask  # masking data dict only
        plot_mask = RawPlot(mask_data, nc_f)  # masking data, assign to plot_mask object

        if args.time_ph:
            if args.sfig:
                if args.counts:
                    # Plotting Time series PHD
                    plot_mask.plt_tph(phd_bin, savefig=True, cnts = True)  # Concatenate method (slower but stable)
                    # plot_mask.t_phd(phd_bin)  # Obsolete
                else:
                    plot_mask.plt_tph(phd_bin, savefig=True)
            else:
                if args.counts:
                    plot_mask.plt_tph(phd_bin, cnts= True)
                else:
                    plot_mask.plt_tph(phd_bin)
        if args.time_tof:
            if args.sfig:
                # Plotting Time series TOF
                plot_mask.plt_ttof(tof_bin, savefig=True)
            else:
                plot_mask.plt_ttof(tof_bin)

        if args.tof_ph:
            if args.sfig:
                # Plotting TOF - PH
                plot_mask.plt_tofph(tof_bin, savefig=True)

            else:
                plot_mask.plt_tofph(tof_bin)

        if args.rawevent:
            if args.sfig:
                plot_mask.raw_eventplt(args.dir, R_v, binning, savefig=True)
            else:
                plot_mask.raw_eventplt(args.dir, R_v, binning)

        if args.store:
            plot_mask.test_store(dirpath, nc_f)
            # plot_mask.store(path, nc_f)  # copying

if __name__ == "__main__":  # Main function
    import sys
    raw_filter(sys.argv[1:])
