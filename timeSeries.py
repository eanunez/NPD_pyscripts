'''
NAME: Time Series Pulse Height Plot and TOF-PH plot
FUNCTION: - Plots Pulse height distribution function in every direction for long period time series (both 1D and 2D histogram)
            (i.e annually or several years depending on the input file)
          - Plots Time-of-Flight - Pulse height in every direction for long period time series
REQUIREMENTS: Ph data file formatted in json (see plt_filter.py for I/O file formatting)
PARAMETERS: id: filename (either .json file or .txt file containing .json data file lines
                            See plt_filter.py for .json file naming convention)
            -dir#: optional argument, type of direction i.e -dir0, -dir1, -dir2
            -savefig: positional argument for saving figure

SAMPLE USAGE: python3 timeSeries.py npd1raw2006.json -dir0 -save (to generate and save yearly plots in dir0)
              python3 timeSeries.py npd1raw2006.txt -dir0 -save (to generate and save m years of plots in dir0)

PROGRAMMER: Emmanuel Nunez (emmnun-4@student.ltu.se)
LAST UPDATED: 2016-05-11

REFERENCES: Yoshifumi Futaana
            Xiao-Dong Wang
'''

# !/usr/local/bin/python3

import numpy as np
import argparse, os, time, json
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class TimeSeriesPH:
    _phbin = [150, 256]
    _tofbin = [256, 16]

    # ASPERA-4 NPD1 and NPD2 limiting parameters
    _tofH = [258.0, 268.0]  # TOF Mass for Hydrogen npd1, npd2 [ns]
    _tofO = [629.0, 654.0]  # TOF Mass for Oxygen in npd1, npd2 [ns]
    _npd1_phH = [1.16 * 10 ** -2, 8.34 * 10 ** -2,
                 1.27 * 10 ** -2]  # npd1 ph distribution tail for Hydrogen in dir0, dir1, dir2
    _npd2_phH = [4.00 * 10 ** -2, 0.26 * 10 ** -2, 0.06 * 10 ** -2]  # npd2
    _npd1_phO = [3.18 * 10 ** -2, 10.70 * 10 ** -2,
                 5.99 * 10 ** -2]  # npd1 ph distribution tail for Oxygen in dir0, dir1, dir2
    _npd2_phO = [6.00 * 10 ** -2, 4.45 * 10 ** -2, 2.10 * 10 ** -2]  # npd2

    # Constructor
    def __init__(self, f_path, f_name, infile):
        self.f_path = f_path  # full file path
        self.f_name = f_name  # json file
        self.infile = infile  # json or txt file

    def file_len(self):
        with open(self.f_path) as f:
            for i, l in enumerate(f):
                pass
            return i + 1

    def set_info(self, st, et):

        # start date and time formatting
        s_time = time.gmtime(st)  # returns struct_time of date and time values
        e_time = time.gmtime(et)
        d_format = '%H:%M'

        # Range in seconds
        st_sec = s_time.tm_hour * 3600 + s_time.tm_min * 60 + s_time.tm_sec
        et_sec = e_time.tm_hour * 3600 + e_time.tm_min * 60 + e_time.tm_sec
        dsec = et_sec - st_sec

        start_str = time.strftime(d_format, s_time)
        end_str = time.strftime(d_format, e_time)

        return start_str, end_str

    def dir0_hist(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('Probability, Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)

            j, i, count, figcount = 0, 0, 0, 0

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir0_time']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:

                    # Histogram for dir0
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    #hist, bins = np.histogram(data_file['dir0_ph'], bins=phd_bin[1], normed=True)
                    hist, bins = np.histogram(data_file['dir0_ph'], bins=phd_bin[1])

                    # Normalize with time
                    hist = [n / abs(float(data_file['dt'])) for n in hist]

                    # Plot the resulting histogram
                    axarr[j, i].plot(hist, bins[:-1], '-', color='k')
                    axarr[j, i].set_xscale('log')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)
                        j += 1
                        i = 0
                        if count > 39:  # If subplots are filled, open new figure
                            if savefig:
                                figcount += 1
                                print('Saving figure..')
                                # Change path for saving figure
                                plt.savefig(
                                    '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                              :-5] + '_dir0_tph' + str(
                                        figcount) + '.png')
                            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                            fig.suptitle('Probability, Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                            j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir0_tph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year
                plt.show()

    def dir1_hist(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('Probability, Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)

            j, i, count, figcount = 0, 0, 0, 0

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir1_time']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:

                    hist, bins = np.histogram(data_file['dir1_ph'], bins=phd_bin[1])

                    # Normalize with time
                    hist = [n / abs(float(data_file['dt'])) for n in hist]

                    # Plot the resulting histogram
                    axarr[j, i].plot(hist, bins[:-1], '-', color='k')
                    # Plot Histogram for dir1
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    axarr[j, i].set_xscale('log')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)
                        j += 1
                        i = 0
                        if count > 39:  # If subplots are filled, open new figure
                            if savefig:
                                figcount += 1
                                print('Saving figure..')
                                # Change path for saving figure
                                plt.savefig(
                                    '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                              :-5] + '_dir1_tph' + str(
                                        figcount) + '.png')

                            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                            fig.suptitle('Probability, Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                            j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir1_tph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year
                plt.show()

    def dir2_hist(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('Probability, Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.04)

            j, i, count, figcount = 0, 0, 0, 0

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir2_time']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:

                    hist, bins = np.histogram(data_file['dir2_ph'], bins=phd_bin[1])

                    # Normalize with time (dtsec)
                    hist = [n / abs(float(data_file['dt'])) for n in hist]

                    # Plot the resulting histogram
                    axarr[j, i].plot(hist, bins[:-1], color='k')

                    # Plot Histogram for dir2
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    # axarr[j, i].hist(data_file['dir2_ph'], bins=phd_bin[1], normed=0, histtype='stepfilled', facecolor='w', orientation='horizontal')
                    axarr[j, i].set_xscale('log')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)
                        j += 1
                        i = 0
                        if count > 39:  # If subplots are filled, open new figure
                            if savefig:
                                figcount += 1
                                print('Saving figure..')
                                # Change path for saving figure
                                plt.savefig(
                                    '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                              :-5] + '_dir2_tph' + str(
                                        figcount) + '.png')

                            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                            fig.suptitle('Probability, Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                            j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir2_tph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year
                plt.show()

                #  ========================================== 2D Histogram ======================================

    def dir0_hist2d(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
            fig.suptitle('Probability, Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)
            fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir0_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue
                else:

                    # Plotting dir0
                    d0_h, d0_xedges, d0_yedges = np.histogram2d(data_file['dir0_time'], data_file['dir0_ph'],
                                                                bins=phd_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d0_norm = d0_h / float(len(data_file['dir0_ph']))

                    # Masked zero values
                    # d0_h[d0_h == 0.0] = np.nan
                    # cmap = cm.get_cmap('jet')
                    # cmap.set_bad('w', 1.)

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    im = axarr[j, i].imshow(np.rot90(d0_norm), interpolation='nearest',
                                            extent=[d0_xedges[0], d0_xedges[-1], 0, 255],
                                            aspect='auto')
                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    cb.formatter.set_powerlimits((0, 0))
                    cb.update_ticks()

                    # plt.setp([a.get_xticklabels() for a in axarr[j,:]], visible=False)
                    plt.setp(axarr[j, i], xticklabels='')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)
                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir0_2dhist' + str(
                                    figcount) + '.png')
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
                        fig.suptitle('Probability, Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)
                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        cb.formatter.set_powerlimits((0, 0))
                        cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir0_2dhist' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def dir1_hist2d(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
            fig.suptitle('Probability, Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)
            fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir1_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue

                else:

                    # Plotting dir0
                    d_h, d_xedges, d_yedges = np.histogram2d(data_file['dir1_time'], data_file['dir1_ph'],
                                                             bins=phd_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d_norm = d_h / float(len(data_file['dir1_ph']))

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest',
                                            extent=[d_xedges[0], d_xedges[-1], 0, 255],
                                            aspect='auto')

                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    cb.formatter.set_powerlimits((0, 0))
                    cb.update_ticks()

                    # plt.setp([a.get_xticklabels() for a in axarr[j,:]], visible=False)
                    plt.setp(axarr[j, i], xticklabels='')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)
                        # axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)
                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir1_2dhist' + str(
                                    figcount) + '.png')
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
                        fig.suptitle('Probability, Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)
                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        cb.formatter.set_powerlimits((0, 0))
                        cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir1_2dhist' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def dir2_hist2d(self, phd_bin, savefig=False):

        if not phd_bin:  # if bin is empty
            phd_bin = self.phbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0
            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
            fig.suptitle('Probability, Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.05)
            fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir2_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue

                else:
                    # Plotting dir0
                    d_h, d_xedges, d_yedges = np.histogram2d(data_file['dir2_time'], data_file['dir2_ph'],
                                                             bins=phd_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d_norm = d_h / float(len(data_file['dir2_ph']))

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest',
                                            extent=[d_xedges[0], d_xedges[-1], 0, 255],
                                            aspect='auto')

                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    cb.formatter.set_powerlimits((0, 0))
                    cb.update_ticks()

                    plt.setp(axarr[j, i], xticklabels='')
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)

                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir2_2dhist' + str(
                                    figcount) + '.png')
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(16, 8.5), sharey=True)
                        fig.suptitle('Probability, Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.05)
                        fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.06, wspace=0, hspace=0.17)
                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        cb.formatter.set_powerlimits((0, 0))
                        cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir2_2dhist' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def dir0_tofph(self, tof_bin, savefig=False):

        if not tof_bin:  # if bin is empty
            tof_bin = self.tofbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0

            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('TOF [ns], Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir0_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue

                else:
                    # Plotting dir0
                    d_h, d_xedges, d_yedges = np.histogram2d(data_file['dir0_tof'], data_file['dir0_ph'],
                                                             bins=tof_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d_norm = d_h / float(len(data_file['dir0_tof']))

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    '''im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest',
                                            extent=[0, 2048, 0, 255],
                                            aspect='auto')'''
                    im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest', norm = colors.PowerNorm(gamma=1./2.),
                                            extent=[0, 2048, 0, 255],aspect='auto')

                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()

                    axarr[j, i].set_xlim([0,2048])
                    for tick in axarr[j,i].xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)

                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir0_tofph' + str(
                                    figcount) + '.png')
                        # Setup Figure and Subplots
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                        fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                        fig.suptitle('TOF [ns], Ch: dir0\nFilename: ' + self.f_name, fontsize=11, y=0.05)

                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        #cb.formatter.set_powerlimits((0, 0))
                        #cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir0_tofph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def dir1_tofph(self, tof_bin, savefig=False):

        if not tof_bin:  # if bin is empty
            tof_bin = self.tofbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0

            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('TOF [ns], Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir1_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue

                else:
                    # Plotting dir0
                    d_h, d_xedges, d_yedges = np.histogram2d(data_file['dir1_tof'], data_file['dir1_ph'],
                                                             bins=tof_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d_norm = d_h / float(len(data_file['dir1_tof']))

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest', norm = colors.PowerNorm(gamma=1./2.),
                                            extent=[0, 2048, 0, 255],aspect='auto')
                    '''im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest',
                                            extent=[0, 2048, 0, 255],
                                            aspect='auto')'''

                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()

                    axarr[j, i].set_xlim([0,2048])
                    for tick in axarr[j,i].xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)

                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir1_tofph' + str(
                                    figcount) + '.png')
                        # Setup Figure and Subplots
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                        fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                        fig.suptitle('TOF [ns], Ch: dir1\nFilename: ' + self.f_name, fontsize=11, y=0.05)

                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        #cb.formatter.set_powerlimits((0, 0))
                        #cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir1_tofph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def dir2_tofph(self,tof_bin, savefig=False):
        if not tof_bin:  # if bin is empty
            tof_bin = self.tofbin

        # Clean plot
        plt.close('all')

        lines = self.file_len()
        print(lines)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            j, i, count, figcount = 0, 0, 0, 0

            # Setup Figure and Subplots
            fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
            fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
            fig.suptitle('TOF [ns], Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.05)

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir2_time']) == 0:  # if file line is empty
                    print('Empty line.')
                    continue

                else:
                    # Plotting dir0
                    d_h, d_xedges, d_yedges = np.histogram2d(data_file['dir2_tof'], data_file['dir2_ph'],
                                                             bins=tof_bin)

                    # Normalize to 1 (normalize to the total number of counts)
                    d_norm = d_h / float(len(data_file['dir2_tof']))

                    # draw image
                    axarr[j, i].set_title(data_file['Date'] + '\n' + data_file['Start'] + '-' + data_file['End'],
                                          fontsize=8)
                    im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest', norm = colors.PowerNorm(gamma=1./2.),
                                            extent=[0, 2048, 0, 255],aspect='auto')

                    '''im = axarr[j, i].imshow(np.rot90(d_norm), interpolation='nearest',
                                            extent=[0, 2048, 0, 255],
                                            aspect='auto')'''

                    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                    cb = fig.colorbar(im, cax=cbar_ax)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()

                    axarr[j, i].set_xlim([0,2048])
                    for tick in axarr[j,i].xaxis.get_major_ticks():
                        tick.label.set_fontsize(8)
                    axarr[j, 0].set_ylabel('PHs', multialignment='center', fontsize=9)

                    i += 1
                    count += 1
                    if i > 9 and j <= 3:  # If ncols = 10, shift to next row
                        print(j, i)

                        j += 1
                        i = 0

                    if count > 39:  # If subplots are filled, open new figure
                        figcount += 1
                        if savefig:
                            print('Saving figure..')
                            # Change path for saving figure
                            plt.savefig(
                                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                          :-5] + '_dir2_tofph' + str(
                                    figcount) + '.png')
                        # Setup Figure and Subplots
                        fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(17, 8), sharex='col', sharey='row')
                        fig.subplots_adjust(left=0.06, right=0.95, top=0.95, bottom=0.09, wspace=0, hspace=0.17)
                        fig.suptitle('TOF [ns], Ch: dir2\nFilename: ' + self.f_name, fontsize=11, y=0.05)

                        cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
                        cb = fig.colorbar(im, cax=cbar_ax)
                        #cb.formatter.set_powerlimits((0, 0))
                        #cb.update_ticks()
                        j, i, count = 0, 0, 0
            if savefig:
                figcount += 1
                print('Saving figure..')
                # Change path for saving figure
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + self.f_name[
                                                                                                      :-5] + '_dir2_tofph' + str(
                    figcount) + '.png')

            if self.infile.lower().endswith('.json'):  # only show plot for each year

                plt.show()

    def filter_ph(self, data_file, sensor):

        ph_bin = 16

        # Bin mode threshold (dir0, dir1, dir2)
        npd1bin_nth = [7.0, 11.0, 10.0]
        npd2bin_nth = [11.0, 10.0, 5.0]

        if sensor == 'npd1':
            npd1raw_nth = [float(self.phbin[1] / ph_bin) * x for x in npd1bin_nth]  # calculate corresponding threshold channel in raw mode
            dir0_ph_val = [np.array(data_file['dir0_ph']) >= npd1raw_nth[0]]
            dir1_ph_val = [np.array(data_file['dir1_ph']) >= npd1raw_nth[1]]
            dir2_ph_val = [np.array(data_file['dir2_ph']) >= npd1raw_nth[2]]

            # Limiting Probabilities
            dir0_prob = float(len(dir0_ph_val)/len(data_file['dir0_ph']))
            dir1_prob = float(len(dir1_ph_val)/len(data_file['dir1_ph']))
            dir2_prob = float(len(dir2_ph_val)/len(data_file['dir2_ph']))

            if dir0_prob > 0.01:
                return True
            elif dir1_prob > 0.01:
                return True
            elif dir2_prob > 0.01:
                return True
            else:
                return False

        elif sensor == 'npd2':
            npd2raw_nth = [float(self.phbin[1] / ph_bin) * x for x in npd2bin_nth]
            dir0_ph_val = [np.array(data_file['dir0_ph']) >= npd2raw_nth[0]]
            dir1_ph_val = [np.array(data_file['dir1_ph']) >= npd2raw_nth[1]]
            dir2_ph_val = [np.array(data_file['dir2_ph']) >= npd2raw_nth[2]]

            # Limiting Probabilities
            dir0_prob = float(len(dir0_ph_val)/len(data_file['dir0_ph']))
            dir1_prob = float(len(dir1_ph_val)/len(data_file['dir1_ph']))
            dir2_prob = float(len(dir2_ph_val)/len(data_file['dir2_ph']))

            if dir0_prob >= 0.01:
                return True
            elif dir1_prob >= 0.01:
                return True
            elif dir2_prob >= 0.01:
                return True
            else:
                return False

    def filter_tof(self, data_file, sensor):

        # Mass Resolution Filtering
        # ========= TOF CHECK ===============
        # Oxygen NPD1: TOF > 629.0, NPD2: TOF > 654.0
        tofO = self.tofO
        if sensor == 'npd1':
            # ============== TOF Oxygen Check =============
            dir0_tofO = [np.array(data_file['dir0_tof']) > tofO[0]]
            dir1_tofO = [np.array(data_file['dir1_tof']) > tofO[0]]
            dir2_tofO = [np.array(data_file['dir2_tof']) > tofO[0]]

            # ========== PH Oxygen Check =============
            ch_raw = self.filter_ph(data_file, 'npd1')

            if len(dir0_tofO) != 0:
                if ch_raw:
                    return True

            elif len(dir1_tofO) != 0:
                if ch_raw:
                    return True
            elif len(dir2_tofO) != 0:
                if ch_raw:
                    return True
            else:
                return False

        elif sensor == 'npd2':
            # ============== Oxygen Check =============
            dir0_tofO = [np.array(data_file['dir0_tof']) > tofO[1]]
            dir1_tofO = [np.array(data_file['dir1_tof']) > tofO[1]]
            dir2_tofO = [np.array(data_file['dir2_tof']) > tofO[1]]

            # ========== PH Oxygen Check =============
            ch_raw = self.filter_ph(data_file, 'npd2')

            if len(dir0_tofO) != 0:
                if ch_raw:
                    return True
            elif len(dir1_tofO) != 0:
                if ch_raw:
                    return True
            elif len(dir2_tofO) != 0:
                if ch_raw:
                    return True
            else:
                return False

    def list_Oxyf(self, path):

        # Read file containing list
        with open(self.f_path, 'r') as f:

            for line in f:  # reading file per line
                data_file = json.loads(line)  # ph data in dict

                if len(data_file['dir0_time']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    ox = self.filter_tof(data_file, data_file['Filename'][:4])
                    new_file = path +'/' + data_file['Filename'][:4] + '_tofO.list'
                    if ox:
                        with open(new_file, 'a') as file:
                            file.write(data_file['Filename'])
                            file.write(os.linesep)
        f.close()
        print('Append: ', new_file)

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

    @property
    def tofH(self):
        return self._tofH

    def set_tofH(self, tofH):
        self._tofH = tofH

    @property
    def tofO(self):
        return self._tofO

    def set_tofO(self, tofO):
        self._tofO = tofO

    @property
    def npd1_phH(self):
        return self._npd1_phH

    def set_npd1_phH(self, npd1_phH):
        self._npd1_phH = npd1_phH

    @property
    def npd1_phO(self):
        return self._npd1_phO

    def set_npd1_phO(self, npd1_phO):
        self._npd1_phO = npd1_phO

    @property
    def npd2_phH(self):
        return self._npd2_phH

    def set_npd2_phH(self, npd2_phH):
        self._npd2_phH = npd2_phH

    @property
    def npd2_phO(self):
        return self._npd2_phO

    def set_npd2_phO(self, npd2_phO):
        self._npd2_phO = npd2_phO


def timePh(args):  # sub-function
    # Parsing Arguments
    parser = argparse.ArgumentParser(prog='python3 timeSeries.py')

    parser.add_argument("id", type=argparse.FileType('r'), help="input filename")
    parser.add_argument("-dir0", "--dir0", action="store_true", help="Direction to plot (dir0)")
    parser.add_argument("-dir1", "--dir1", action="store_true", help="Direction to plot(dir1)")
    parser.add_argument("-dir2", "--dir2", action="store_true", help="Direction to plot(dir2)")
    parser.add_argument("-hist", "--hist", action="store_true", help="plot time series Pulse Height distribution")
    parser.add_argument("-h2d", "--hist2d", action="store_true", help="plot time series Pulse Height histogram 2d")
    parser.add_argument("-tofph", "--tofph", action="store_true", help = "plot time series TOF-PH")
    parser.add_argument("-savefig", "--savefig", action="store_true", help="Save figure")
    parser.add_argument("-Of", "--Ofilter", action="store_true", help="Filter Oxygen via TOF system")


    args = parser.parse_args()

    # Setting path and filename
    dirpath = os.path.dirname(os.path.abspath(args.id.name))  # file path
    nc_f = os.path.basename(args.id.name)  # filename
    full_path = os.path.abspath(args.id.name)

    phd_bin = [150, 256]
    tof_bin = [256, 16]

    # File Checking
    if nc_f.lower().endswith('.txt'):  # If text file containing list
        with open(full_path, 'r') as f:
            contents = [line.rstrip('\n') for line in f]
            for jsonline in contents:
                try:
                    full_path = dirpath + '/' + jsonline
                    jsonfile = TimeSeriesPH(full_path, jsonline, nc_f)
                    if args.dir0:
                        if args.savefig:
                            if args.hist:
                                jsonfile.dir0_hist(phd_bin, savefig=True)
                            if args.hist2d:
                                jsonfile.dir0_hist2d(phd_bin, savefig=True)
                            if args.tofph:
                                jsonfile.dir0_tofph(tof_bin, savefig=True)
                        else:
                            if args.hist:
                                jsonfile.dir0_hist(phd_bin)
                            if args.hist2d:
                                jsonfile.dir0_hist2d(phd_bin)
                            if args.tofph:
                                jsonfile.dir0_tofph(tof_bin)
                    elif args.dir1:
                        if args.savefig:
                            if args.hist:
                                jsonfile.dir1_hist(phd_bin, savefig=True)
                            if args.hist2d:
                                jsonfile.dir1_hist2d(phd_bin, savefig=True)
                            if args.tofph:
                                jsonfile.dir1_tofph(tof_bin, savefig=True)
                        else:
                            if args.hist:
                                jsonfile.dir1_hist(phd_bin)
                            if args.hist2d:
                                jsonfile.dir1_hist2d(phd_bin)
                            if args.tofph:
                                jsonfile.dir1_tofph(tof_bin)
                    elif args.dir2:
                        if args.savefig:
                            if args.hist:
                                jsonfile.dir2_hist(phd_bin, savefig=True)
                            if args.hist2d:
                                jsonfile.dir2_hist2d(phd_bin, savefig=True)
                            if args.tofph:
                                jsonfile.dir2_tofph(tof_bin, savefig=True)
                        else:
                            if args.hist:
                                jsonfile.dir2_hist(phd_bin)
                            if args.hist2d:
                                jsonfile.dir2_hist2d(phd_bin)
                            if args.tofph:
                                jsonfile.dir2_tofph(tof_bin)
                    elif args.Ofilter:
                            jsonfile.list_Oxyf(dirpath)
                except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                    print('Error: ', err)

    elif nc_f.lower().endswith('.json'):

        file = TimeSeriesPH(full_path, nc_f, nc_f)
        if args.dir0:
            if args.savefig:
                if args.hist:
                    file.dir0_hist(phd_bin, savefig=True)
                if args.hist2d:
                    file.dir0_hist2d(phd_bin, savefig=True)
                if args.tofph:
                    file.dir0_tofph(tof_bin, savefig= True)
            else:
                if args.hist:
                    file.dir0_hist(phd_bin)
                if args.hist2d:
                    file.dir0_hist2d(phd_bin)
                if args.tofph:
                    file.dir0_tofph(tof_bin)
        elif args.dir1:
            if args.savefig:
                if args.hist:
                    file.dir1_hist(phd_bin, savefig=True)
                if args.hist2d:
                    file.dir1_hist2d(phd_bin, savefig=True)
                if args.tofph:
                    file.dir1_tofph(tof_bin, savefig= True)
            else:
                if args.hist:
                    file.dir1_hist(phd_bin)
                if args.hist2d:
                    file.dir1_hist2d(phd_bin)
                if args.tofph:
                    file.dir1_tofph(tof_bin)
        elif args.dir2:
            if args.savefig:
                if args.hist:
                    file.dir2_hist(phd_bin, savefig=True)
                if args.hist2d:
                    file.dir2_hist2d(phd_bin, savefig=True)
                if args.tofph:
                    file.dir2_tofph(tof_bin, savefig= True)
            else:
                if args.hist:
                    file.dir2_hist(phd_bin)
                if args.hist2d:
                    file.dir2_hist2d(phd_bin)
                if args.tofph:
                    file.dir2_tofph(tof_bin)
    else:
        print('Provide a .json file containing ph data')


if __name__ == "__main__":  # Main function
    import sys

    timePh(sys.argv[1:])
