'''
NAME: plt_filter.py
PURPOSE: Python script for filtering and plotting NPD (Raw mode) data
        Access NPD (RAW mode) Module, NPD_raw_read.py

PROGRAMMER: Emmanuel Nunez (emmnun-4@student.ltu.se)

REFERENCES: Yoshifumi Futaana
            Xiao-Dong Wang

PARAMS:
	id: data filename (either netCDF file or file list containing '.nc' file names)
	-v: optional argument, print information in netCDF file
	-tphs: optional argument for time series plotting

	Sample Usage: python3 plt_filter.py npd2raw20110426000341.nc -tphs
'''

# !/usr/local/bin/python3

import numpy as np
import argparse, os, glob, datetime, time, json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import epoch2num
import NPD_raw_readv1 as raw

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

        # Determine indices of coincidence flags
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
    _phbin = [100, 255]
    _data_cp = []

    # Constructor
    def __init__(self, data, name):
        self.data = data
        self.filename = name

    def info(self):  # General info of the data

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
        dsec = et_sec - st_sec

        start_str = time.strftime(d_format, s_time)
        end_str = time.strftime(d_format, e_time)

        info_d = {'Filename': f_name, 'Date': s_date, 'Start': start_str, 'End': end_str, 'dt': dsec}
        '''
        print('Filename: ', f_name)
        print('Date: ', s_date, ', epoch[day]: ', s_day[0])
        print('GMT:')
        print('Start Time : ', time.strftime(d_format, s_time), ', End Time: ', time.strftime(d_format, e_time))
        '''
        return info_d

    def dir_sep(self):  # Direction separation function

        phs = self.data['PHs']
        dirs = self.data['dirs']

        # direction/sectors separation
        dir0_phs = np.ma.masked_where(dirs != 0, phs)  # mask elements in PHs not equal to 0 in dirs
        dir1_phs = np.ma.masked_where(dirs != 1, phs)
        dir2_phs = np.ma.masked_where(dirs != 2, phs)

        # masking check
        # print(dir0_phs[~np.ma.getmask(dir0_phs)])

        return dir0_phs, dir1_phs, dir2_phs

    def comp_tph(self):  # Time series PH data manipulation/concatenation

        t_data = self.data['Time']

        # Direction Separation
        dir0_phs, dir1_phs, dir2_phs = self.dir_sep()

        # dir0 PH Distribution Computation
        dir0_phs_h = np.array([], dtype='int32')
        dir0_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir0_phs_row = np.ma.compressed(dir0_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            # if len(dir0_PHs_row) != 0:

            # 0 and 255 values screening
            dir0_phs_row = dir0_phs_row[dir0_phs_row != 0]
            dir0_phs_row = dir0_phs_row[dir0_phs_row != 255]

            # time arr allocation
            time_row = np.zeros(len(dir0_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir0_time = np.concatenate((dir0_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir0_phs_h = np.concatenate((dir0_phs_h, dir0_phs_row), axis=0)

        # dir1 PH Distribution Computation
        dir1_phs_h = np.array([], dtype='int32')
        dir1_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir1_phs_row = np.ma.compressed(dir1_phs[i, :])  # compress unmasked phs values every row (1 sec interval)
            # if len(dir0_PHs_row) != 0:

            # 0 and 255 values screening
            dir1_phs_row = dir1_phs_row[dir1_phs_row != 0]
            dir1_phs_row = dir1_phs_row[dir1_phs_row != 255]

            # time arr allocation
            time_row = np.zeros(len(dir1_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir1_time = np.concatenate((dir1_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir1_phs_h = np.concatenate((dir1_phs_h, dir1_phs_row), axis=0)

        # dir2 PH Distribution Computation
        dir2_phs_h = np.array([], dtype='int32')
        dir2_time = np.array([], dtype='float64')

        for i in range(len(t_data)):
            dir2_phs_row = np.ma.compressed(dir2_phs[i, :])  # compress unmasked phs values every row (1 sec interval)

            # 0 and 255 values screening
            dir2_phs_row = dir2_phs_row[dir2_phs_row != 0]
            dir2_phs_row = dir2_phs_row[dir2_phs_row != 255]

            # time arr allocation
            time_row = np.zeros(len(dir2_phs_row))  # zeros of time_row equal to number of phs per row
            time_row = time_row + t_data[i]
            dir2_time = np.concatenate((dir2_time, time_row), axis=0)  # consider append list, reshape in the future?

            dir2_phs_h = np.concatenate((dir2_phs_h, dir2_phs_row), axis=0)

        dir_ph_h = {'dir0_phs_h': dir0_phs_h, 'dir0_time': dir0_time, 'dir1_phs_h': dir1_phs_h, 'dir1_time': dir1_time,
                    'dir2_phs_h': dir2_phs_h, 'dir2_time': dir2_time}

        return dir_ph_h

    def xticks(self, ax, mintick, maxtick, dtick):  # Tick Customization

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

        elif dtick == 'day':

            # Enable Minor Ticks
            minorlocator = MultipleLocator(50)
            ax.xaxis.set_minor_locator(minorlocator)
            d_range = np.arange(0, dsec / (3600 * 24), 0.02)
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

    def plt_tph(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Call Time series PH Distribution function
        dir_phs_h = self.comp_tph()
        # dir_phs_h = self.append_method()  # Plan B for efficiency

        # Call for general info
        info = self.info()

        # Setup Figure and Subplots
        plt.close('all')

        fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))

        fig.suptitle('Filename: ' + self.filename + '\nTime: ' + info['Start'] + ' - ' + info['End'], fontsize=11,
                     y=0.04)

        axarr[0, 0].set_ylabel('PHs\ndir0', multialignment='center', fontsize=10)
        axarr[1, 0].set_ylabel('PHs\ndir1', multialignment='center', fontsize=10)
        axarr[2, 0].set_ylabel('PHs\ndir2', multialignment='center', fontsize=10)

        axarr[0, 1].set_ylabel('Counts\ndir0', multialignment='center', fontsize=10)
        axarr[1, 1].set_ylabel('Counts\ndir1', multialignment='center', fontsize=10)
        axarr[2, 1].set_ylabel('Counts\ndir2', multialignment='center', fontsize=10)

        # Plotting dir0
        d0_h, d0_xedges, d0_yedges = np.histogram2d(dir_phs_h['dir0_time'], dir_phs_h['dir0_phs_h'], bins=phd_bin)
        d0X, d0Y = np.meshgrid(d0_xedges, d0_yedges)
        im0 = axarr[0, 0].pcolormesh(d0X, d0Y, d0_h.T)
        axarr[0, 0].set_ylim([d0Y.min() - 5, d0Y.max() + 5])
        plt.colorbar(im0, ax=axarr[0, 0])
        axarr[0, 0].set_xlim([d0X.min(), d0X.max()])

        # Plotting dir1
        d1_h, d1_xedges, d1_yedges = np.histogram2d(dir_phs_h['dir1_time'], dir_phs_h['dir1_phs_h'], bins=phd_bin)
        d1X, d1Y = np.meshgrid(d1_xedges, d1_yedges)
        im1 = axarr[1, 0].pcolormesh(d1X, d1Y, d1_h.T)
        axarr[1, 0].set_ylim([d1Y.min() - 5, d1Y.max() + 5])
        plt.colorbar(im1, ax=axarr[1, 0])
        axarr[1, 0].set_xlim([d1X.min(), d1X.max()])

        # Plotting dir2
        d2_h, d2_xedges, d2_yedges = np.histogram2d(dir_phs_h['dir2_time'], dir_phs_h['dir2_phs_h'], bins=phd_bin)
        d2X, d2Y = np.meshgrid(d2_xedges, d2_yedges)
        im2 = axarr[2, 0].pcolormesh(d2X, d2Y, d2_h.T)
        plt.colorbar(im2, ax=axarr[2, 0])

        axarr[2, 0].set_xlim([d2X.min(), d2X.max()])
        axarr[2, 0].set_ylim([d2Y.min() - 5, d2Y.max() + 5])

        # Setting ticks and tick labels for all first column subplots
        t_diff = info['dt']

        # Identify time duration
        if t_diff <= 1800:
            xticks = self.xticks(axarr[2, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='min')
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        elif t_diff <= 3600:
            xticks = self.xticks(axarr[2, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='hour')
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])
        else:
            xticks = self.xticks(axarr[2, 0], mintick=d2X.min(), maxtick=d2X.max(), dtick='day')
            plt.setp(axarr[:, 0], xticks=xticks['major_tick'], xticklabels=xticks['tick_label'])

        axarr[2, 0].set_xlabel('Time', fontsize=10)

        # Histogram Plot
        axarr[0, 1].hist(dir_phs_h['dir0_phs_h'], bins=phd_bin[0], normed=0, label='dir0', histtype='stepfilled',
                         facecolor='w')
        axarr[1, 1].hist(dir_phs_h['dir1_phs_h'], bins=phd_bin[0], normed=0, label='dir1', histtype='stepfilled',
                         facecolor='w')
        axarr[2, 1].hist(dir_phs_h['dir2_phs_h'], bins=phd_bin[0], normed=0, label='dir2', histtype='stepfilled',
                         facecolor='w')

        # Enable Minor Ticks
        minorlocator = MultipleLocator(5)

        for i in range(axarr.size - 3):
            axarr[i, 1].xaxis.set_minor_locator(minorlocator)

            # Determining log scale (crude way)
            if dir_phs_h['dir0_phs_h'].max() - dir_phs_h['dir0_phs_h'].min() > 100:
                axarr[i, 1].set_yscale('log')
            axarr[i, 1].set_xlim(0, 255)

        axarr[2, 1].set_xlabel('PH', fontsize=10)
        fig.subplots_adjust(hspace=0.15)

        plt.tight_layout()
        if savefig:
            print('Saving figure..')
            # Uncomment next lines and provide the right path
            # plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/'+self.filename[5:-3]+'.png')
            # plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/data/figs/'+self.filename[5:-3]+'.png')
            plt.show()

        return

    def tof_ph(self):  # PH spectra for three sectors
        pass
        PHs = self.data['val_PHs']
        dirs = self.data['val_dirs']
        TOFs = self.data['val_TOFs']
        time = self.data['Time']
        dir0_PHs = PHs[dirs == 0]  # dir0
        dir1_PHs = PHs[dirs == 1]
        dir2_PHs = PHs[dirs == 2]
        print (dir0_PHs.shape, time.shape, self.data['DT'][:6])

        # Plotting
        plt.figure(1)
        plt.subplot(311)
        plt.plot(time, dir0_PHs, '.')
        plt.subplot(312)
        plt.plot(time, dir1_PHs, '.')
        plt.subplot(313)
        plt.plot(time, dir2_PHs, '.')

        plt.figure(2)
        plt.xlim(1, 2048, 0.5)
        plt.xlabel('TOF')
        plt.ylabel('PHs')
        plt.title('PHs vs TOFs')
        plt.plot(TOFs, PHs, '.')
        plt.show()

    def store(self, path, file):

        # Call Time series PH Distribution function
        dir_phs_h = self.comp_tph()

        # Call info
        info = self.info()
        f_name = info['Filename']  # for id
        path += '/json/'  # new path (sub directory 'json')
        # print (path)

        # Convert to list
        dir0_phs_h = dir_phs_h['dir0_phs_h'].tolist()
        dir0_time = dir_phs_h['dir0_time'].tolist()
        dir1_phs_h = dir_phs_h['dir1_phs_h'].tolist()
        dir1_time = dir_phs_h['dir1_time'].tolist()
        dir2_phs_h = dir_phs_h['dir2_phs_h'].tolist()
        dir2_time = dir_phs_h['dir2_time'].tolist()
        info = str(info['Date'])

        # Dictionary with data info
        data_file = {'Filename': f_name, 'Info':info, 'dir0_phs_h': dir0_phs_h, 'dir0_time': dir0_time, 'dir1_phs_h': dir1_phs_h, 'dir1_time': dir1_time, 'dir2_phs_h': dir2_phs_h, 'dir2_time': dir2_time}

        if file.lower().endswith('.txt'):   # if text file containing list
            # Write to file
            if os.path.isfile(path + file[:-4] + '.json'):  # if file exists,

                print('File already exists!')
                with open(path + file[:-4] + '.json', 'a') as f:
                    json.dump(data_file,f, separators=(',', ':'), sort_keys=True)
                    print('appending...')
                    f.write(os.linesep)
                f.close()

            else:
                with open(path + file[:-4] + '.json', 'w') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    f.write(os.linesep)
            #print(open(path + file[:-4] + '.json', 'r').read())

        elif file.lower().endswitch('.nc'):
            # Write to file
            if os.path.isfile(path + f_name[:11] + '.json'):  # if file exists,

                print('File already exists!')
                with open(path + f_name[:11] + '.json', 'a') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    print('appending...')

                    f.write(os.linesep)
                f.close()

            else:
                with open(path + f_name[:11] + '.json', 'w') as f:
                    json.dump(data_file, f, separators=(',', ':'), sort_keys=True)
                    f.write(os.linesep)

            #print(open(path + f_name[:-3] + '.json', 'r').read())

    @property
    def phbin(self):
        return self._phbin

    def set_bin(self, phbin):
        self._phbin = phbin

    @property
    def data_cp(self):
        return self._data_cp

    def set_cp(self):
        self._data_cp


def raw_filter(args):
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=argparse.FileType('r'), help="input filename")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-tphs", "--time_PHs", action="store_true", help="Time series PHs through masking")
    parser.add_argument("-phdist", "--ph_dist", action="store_true", help="PHs Distribution through filtering")
    parser.add_argument("-save", "--store", action="store_true", help="Saves filtered data to json file")
    parser.add_argument("-savefig", "--sfig", action="store_true", help="Saves figure to directory")

    args = parser.parse_args()

    path = os.path.dirname(os.path.abspath(args.id.name))  # file path
    # print (path)

    nc_f = os.path.basename(args.id.name)  # filename
    # print(nc_f)
    full_path = os.path.abspath(args.id.name)

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
                    full_path = path + '/' + content
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

                        # Plotting Raw data (all coincidence values)
                        plot_raw = RawPlot(raw_data, content)
                        plot_raw.phdist(binning=True)

                    # masking
                    phd_bin = [150, 255]
                    mask_data = flt_data.mask  # masking data dict only
                    plot_mask = RawPlot(mask_data, content)  # masking data, assign to plot_mask object

                    if args.time_PHs:
                        if args.sfig:
                            # Plotting Time series PHD
                            plot_mask.plt_tph(phd_bin, savefig=True)  # Concatenate method (slower but stable)
                            # plot_mask.t_phd(phd_bin)  # Obsolete
                        else:
                            plot_mask.plt_tph(phd_bin)

                    if args.store:
                        plot_mask.store(path, nc_f)  # copying
                        success += 1
                except ValueError as err:
                    print('Error: ', err)
                    f_error.append(content)
                    failed += 1

            with open(path + '/' + 'Err' + nc_f, 'a') as file:
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

            # Plotting Raw data (all coincidence values)
            plot_raw = RawPlot(raw_data, nc_f)
            plot_raw.phdist(binning=True)

        # masking
        phd_bin = [150, 255]
        mask_data = flt_data.mask  # masking data dict only
        plot_mask = RawPlot(mask_data, nc_f)  # masking data, assign to plot_mask object

        if args.time_PHs:
            if args.sfig:
                # Plotting Time series PHD
                plot_mask.plt_tph(phd_bin, savefig=True)  # Concatenate method (slower but stable)
                # plot_mask.t_phd(phd_bin)  # Obsolete
            else:
                plot_mask.plt_tph(phd_bin)

        if args.store:
            plot_mask.store(path, nc_f)  # copying


if __name__ == "__main__":
    import sys

    raw_filter(sys.argv[1:])