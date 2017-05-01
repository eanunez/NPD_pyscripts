'''
NAME: NPD Raw Vex Spice
FUNCTION: - Generates Vex Spice out from 'filtered' NPD Raw data, stored in .json file
            (See plt_filter.py for I/O File system)
          - Stores new data set with vex spice information, i.e coordinates to json file
            depending on the constraints given (currently stores apoapsis data)
REQUIREMENTS: NPD Raw data file formatted in json (see plt_filter.py for I/O file formatting)
PARAMETERS: id: filename (either .json file or .txt file containing .json data files.
                            See plt_filter.py for .json file naming convention)
            -store: positional argument for storing data to json file

            Json File Format:
                1. each line is a dictionary type of variable containing 'keys'
                2. Keys are the same in 'plt_filter.py' with additional
                    vex ephemeris information for each sensor out from the the time the data is taken

                    Note: json file contains time-to-ph/tof array-pairs in each file for plotting purposes.
                    Thus, time values may be repeating and not necessarily the same size for each direction.
                3. Ephemeris Keys:
                    pos0, pos1, pos2

                4. pos#: contains list of VEX position in Venus-centric Solar Orbit (VSO) measured from Venus

SAMPLE USAGE: python3 spice_npdraw.py data/json/npd1raw2006.json -store
              python3 spice_npdraw.py data/json/npd1raw.txt -store (where 'npd1raw.txt' contains lines of json files)

PROGRAMMER: Emmanuel Nunez (emmnun-4@student.ltu.se)
LAST UPDATED: 2016-05-25

REFERENCES: Yoshifumi Futaana
            Xiao-Dong Wang
'''

# !/usr/local/bin/python3

import numpy as np
import irfpy.vexpvat.vexspice as vspice
import argparse, os, datetime, time, json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.image import AxesImage
from matplotlib.text import Text
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, AutoLocator
import scipy.interpolate as inter
from scipy.stats import maxwell


class Spice_npdraw:
    _R_p = 6.051850 * 10 ** 3  # Planet's radius (venus) [km]
    _tofbin = [256, 16]
    _rhomin = [70000.0]  # limiting position for noise [km]

    def __init__(self, f_path, f_name, infile):
        self.f_path = f_path  # full file path
        self.f_name = f_name  # json file
        self.infile = infile  # json or txt file

    def file_len(self):  # Counts number of lines in the txt file
        with open(self.f_path) as f:
            for i, l in enumerate(f):
                pass
                return i + 1

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

    def ephem_eval(self, datadict):  # Extract position parameters from time stamp (1 sec interval)

        xlist0, xlist1, xlist2 = [], [], []

        rholist0, rholist1, rholist2 = [], [], []

        # Define coordinate - time series relation,
        # dir_time is an array pair to ph and tof for plotting.
        # Thus, values may be repeating and size is different for each dir
        # ============== dir0 =====================
        for t in range(len(datadict['dir0_time'])):
            if t != 0:  # if not 1st element
                if datadict['dir0_time'][t] == datadict['dir0_time'][t - 1]:  # element == equal to previous element
                    xlist0.append(xlist0[t - 1])  # just copy x-axis value from previous element
                    rholist0.append(rholist0[t - 1])

                else:
                    t_struct = self.time_conv(datadict['dir0_time'][t])
                    temp0 = vspice.get_position(
                        datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                          t_struct.tm_min, t_struct.tm_sec))
                    xlist0.append(temp0[0])
                    rho = self.rho(temp0)  # calculate rho
                    rholist0.append(rho)
            else:
                t_struct = self.time_conv(datadict['dir0_time'][t])
                temp0 = vspice.get_position(
                    datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                      t_struct.tm_min, t_struct.tm_sec))
                # Append x-axis
                xlist0.append(temp0[0])
                rho = self.rho(temp0)
                rholist0.append(rho)

        '''# Check size (must be the same)
        print('rholist size: ', len(rholist0))
        print('dir0_time size: ', len(datadict['dir0_time']))
        print('rholist type: ', type(rholist0))'''

        # Do the same for other directions
        # =========== dir1 ==============

        for t in range(len(datadict['dir1_time'])):
            if t != 0:  # if not 1st element
                if datadict['dir1_time'][t] == datadict['dir1_time'][
                            t - 1]:  # if time element == equal to previous t element
                    xlist1.append(xlist1[t - 1])  # just copy coordinates from previous element
                    rholist1.append(rholist1[t - 1])
                else:
                    t_struct = self.time_conv(datadict['dir1_time'][t])
                    temp1 = vspice.get_position(
                        datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                          t_struct.tm_min, t_struct.tm_sec))
                    xlist1.append(temp1[0])
                    rho = self.rho(temp1)  # calculate rho
                    rholist1.append(rho)
            else:
                t_struct = self.time_conv(datadict['dir1_time'][t])
                temp1 = vspice.get_position(
                    datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                      t_struct.tm_min, t_struct.tm_sec))
                xlist1.append(temp1[0])
                rho = self.rho(temp1)
                rholist1.append(rho)

        # ============ dir2 =============

        for t in range(len(datadict['dir2_time'])):
            if t != 0:  # if not 1st element
                if datadict['dir2_time'][t] == datadict['dir2_time'][
                            t - 1]:  # if time element == equal to previous t element
                    xlist2.append(xlist2[t - 1])  # just copy coordinates from previous element
                    rholist2.append(rholist2[t - 1])  # copy previous elem
                else:
                    t_struct = self.time_conv(datadict['dir2_time'][t])
                    temp2 = vspice.get_position(
                        datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                          t_struct.tm_min, t_struct.tm_sec))
                    xlist2.append(temp2[0])
                    rho = self.rho(temp2)  # calculate rho
                    rholist2.append(rho)
            else:
                t_struct = self.time_conv(datadict['dir2_time'][t])
                temp2 = vspice.get_position(
                    datetime.datetime(t_struct.tm_year, t_struct.tm_mon, t_struct.tm_mday, t_struct.tm_hour,
                                      t_struct.tm_min, t_struct.tm_sec))
                xlist2.append(temp2[0])
                rho = self.rho(temp2)
                rholist2.append(rho)

        '''# Convert everything to array
        # Note: Convert datadict to array since data is saved as string list in json!
        dir0_ph = np.array(datadict['dir0_ph'])
        dir0_tof = np.array(datadict['dir0_tof'])
        xarr0 = np.array(xlist0)
        rhoarr0 = np.array(rholist0)

        dir1_ph = np.array(datadict['dir1_ph'])
        dir1_tof = np.array(datadict['dir1_tof'])
        xarr1 = np.array(xlist1)
        rhoarr1 = np.array(rholist1)

        dir2_ph = np.array(datadict['dir2_ph'])
        dir2_tof = np.array(datadict['dir2_tof'])
        xarr2 = np.array(xlist2)
        rhoarr2 = np.array(rholist2)

        # Call for masking of Oxygen
        # ==== dir0 ======
        dir0_pharr, dir0_tofarr, dir0_xarr, dir0_rhoarr = self.mask_oxy(dir0_ph, dir0_tof, xarr0, rhoarr0)

        # ==== dir1 ======
        dir1_pharr, dir1_tofarr, dir1_xarr, dir1_rhoarr = self.mask_oxy(dir1_ph, dir1_tof, xarr1, rhoarr1)

        # ===== dir2 =====
        dir2_pharr, dir2_tofarr, dir2_xarr, dir2_rhoarr = self.mask_oxy(dir2_ph, dir2_tof, xarr2, rhoarr2)

        # Convert everything back to list for json serializable
        dir0_phlist = dir0_pharr.tolist()
        dir0_toflist = dir0_tofarr.tolist()
        dir0_xlist = dir0_xarr.tolist()
        dir0_rholist = dir0_rhoarr.tolist()

        dir1_phlist = dir1_pharr.tolist()
        dir1_toflist = dir1_tofarr.tolist()
        dir1_xlist = dir1_xarr.tolist()
        dir1_rholist = dir1_rhoarr.tolist()

        dir2_phlist = dir2_pharr.tolist()
        dir2_toflist = dir2_tofarr.tolist()
        dir2_xlist = dir2_xarr.tolist()
        dir2_rholist = dir2_rhoarr.tolist()

        new_dict = {'dir0_ph': dir0_phlist, 'dir0_tof': dir0_toflist, 'dir0_x': dir0_xlist, 'dir0_rho': dir0_rholist,
                    'dir1_ph': dir1_phlist, 'dir1_tof': dir1_toflist, 'dir1_x': dir1_xlist, 'dir1_rho': dir1_rholist,
                    'dir2_ph': dir2_phlist, 'dir2_tof': dir2_toflist, 'dir2_x': dir2_xlist, 'dir2_rho': dir2_rholist}'''

        # Call for time to coord conversion for duty cycle
        dcx, dcr = self.dctime_conv(datadict)

        # position data instead of time
        new_dict = {'dir0_x': xlist0, 'dir0_rho': rholist0, 'dir0_ph': datadict['dir0_ph'], 'dir0_tof': datadict['dir0_tof'],
                    'dir1_x': xlist1, 'dir1_rho': rholist1, 'dir1_ph': datadict['dir1_ph'], 'dir1_tof': datadict['dir1_tof'],
                    'dir2_x': xlist2, 'dir2_rho': rholist2, 'dir2_ph': datadict['dir2_ph'], 'dir2_tof': datadict['dir2_tof'],
                    'dcx': dcx, 'dcr': dcr}

        return new_dict

    def ephem_store(self, preformat, datadict):

        file = self.f_name[0:7] + '_spice'

        dirpath = self.f_path[:- len(self.f_name)]
        # print('dirpath: ', dirpath)

        # Merge existing formatted dictionary
        dformat = {**preformat, **datadict}

        '''# Check key values
        for key in dformat:
            print('Check keys: ', key)'''

        # Store files
        if os.path.isfile(dirpath + file + '.json'):  # if file exists,

            print('File exists!')
            with open(dirpath + file + '.json', 'a') as f:
                json.dump(dformat, f, separators=(',', ':'), sort_keys=True)  # append each dict to the file
                print('appending...')
                f.write(os.linesep)  # 1 dict per line in the file
            f.close()

        else:
            with open(dirpath + file + '.json', 'w') as f:
                json.dump(dformat, f, separators=(',', ':'), sort_keys=True)
                f.write(os.linesep)

    def io_ephem(self):

        # Read file containing list
        with open(self.f_path, 'r') as f:

            for line in f:  # reading file per line
                data_file = json.loads(line)  # data in dict

                if len(data_file['dir0_time']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    # Call ephem_eval function to replace time to space dimensions
                    datadict = self.ephem_eval(data_file)


                    # Set additional info format in dictionary
                    preformat = {'Filename': data_file['Filename'], 'Date': data_file['Date'], 'dclist': data_file['dclist'],
                                 'Start': data_file['Start'], 'End': data_file['End'], 'dt': data_file['dt']}

                    # Call for storing data set with position information
                    self.ephem_store(preformat, datadict)
        f.close()

    def mask_oxy(self, pharr, tofarr, xarr, rhoarr):

        tofO = self.get_tofO

        sensor = self.f_name[0:4]

        if sensor == 'npd1':
            # use tofO[0] for masking

            masked_pharr = np.ma.masked_where(tofarr < tofO[0], pharr)
            masked_tofarr = np.ma.masked_where(tofarr < tofO[0], tofarr)
            masked_xarr = np.ma.masked_where(tofarr < tofO[0], xarr)
            masked_rhoarr = np.ma.masked_where(tofarr < tofO[0], rhoarr)

            pharr = np.ma.compressed(masked_pharr)
            tofarr = np.ma.compressed(masked_tofarr)
            xarr = np.ma.compressed(masked_xarr)
            rhoarr = np.ma.compressed(masked_rhoarr)

            # print('Check pharr, tofarr: ', pharr, tofarr)
            return pharr, tofarr, xarr, rhoarr

        elif sensor == 'npd2':
            # use tofO[1] for masking
            masked_pharr = np.ma.masked_where(tofarr < tofO[1], pharr)
            masked_tofarr = np.ma.masked_where(tofarr < tofO[1], tofarr)
            masked_xarr = np.ma.masked_where(tofarr < tofO[1], xarr)
            masked_rhoarr = np.ma.masked_where(tofarr < tofO[1], rhoarr)

            pharr = np.ma.compressed(masked_pharr)
            tofarr = np.ma.compressed(masked_tofarr)
            xarr = np.ma.compressed(masked_xarr)
            rhoarr = np.ma.compressed(masked_rhoarr)

            # print('Check pharr, tofarr: ', pharr, tofarr)
            return pharr, tofarr, xarr, rhoarr

        else:
            print('invalid sensor type')

    @property
    def get_rhomin(self):
        return self._rhomin

    def set_rhomin(self, rhomin):
        self._rhomin = rhomin

    @property
    def get_Rp(self):
        return self._R_p

    def set_Rp(self, Rp):
        self._R_p = Rp


class Spicenpd_plt(Spice_npdraw):
    _subplots = []
    _figcount = 0
    # ASPERA-4 NPD2 estimated Energy Spectra for Oxygen
    npd2_Okev = {'5keV': [180.0, 480.0], '3keV': [480.0, 690.0], '1.3keV': [690.0, 1000.0], '.7keV': [1000.0, 1310.0],
                 '.5keV': [1310.0, 1650.0], '.3keV': [1650.0, 2048]}
    npd2_HOtof = {'overlap': [260., 650.], 'H': [268.0, 1.], 'O': [654.0, 2048]}
    npd2_Htof = [80., 600.]

    def tick_locator(self, ax, minor, major):

        # Enable Minor Ticks
        minorlocator = MultipleLocator(minor)
        majorlocator = MultipleLocator(major)
        ax.xaxis.set_minor_locator(minorlocator)
        ax.xaxis.set_major_locator(majorlocator)

    def vel_tick(self):  # TOF to velocity conversion

        vel_tick = []

        ns_tofticks = np.arange(1., 2048., 2048. / float(5))  # [ns]

        for tof in ns_tofticks:
            vel = self.tofto_vel(tof)
            vel_tick.append(vel)

        vel_ticklabel = ["%.2E" % label for label in vel_tick]
        vel_ticklabel[-1] += '[km/s]'

        # print(vel_ticklabel)
        return vel_tick, vel_ticklabel

    def tofto_vel(self, tof):

        # Define Variables
        L = 8.0  # [cm]
        # k_Hraw = [0.301, 0.248, 0.24, 0.315, 0.375, 0.361]  # Figure 5.44 b.(Griegoriv, 2007)
        k = .34  # Futaana 2006a
        k_factor = 1. / float(1. - k)

        # ===== Calc E0 ========
        eV_kfac = (720. * float(L) / (tof)) ** 2
        eV_amu = eV_kfac * k_factor  # Energy [eV/amu]

        # ====== keV to Joules Conversion =====
        J = 1.60218 * 10 ** -19  # [J/eV]
        kg = 1.66054 * 10 ** -27  # [kg/amu]
        J_kg = eV_amu * J / float(kg)

        # ==== Calculate Velocity =====
        vel = np.sqrt(2. * J_kg) * 10 ** -3  # [km/s]

        return vel

    def eVto_tof(self, eV):
        L = 8.0  # cm
        tof = float(720*L)/np.sqrt(eV*.66)

        return tof

    def tof_toeV(self, tof):
        L = 8.0 # cm
        k = 0.34
        E = (1/float(1-k))*(720.0*L/tof)**2

        return E

    def npd2_kfit(self):

        # Define Variables
        E0 = np.arange(.1, 10., 1.)  # .1 - 10 keV
        k_Hraw = [0.301, 0.248, 0.24, 0.315, 0.375, 0.361]  # Figure 5.44 b.(Griegoriv, 2007)
        E_Hraw = [0.3, 0.5, 0.7, 1.3, 3.0, 5.0]  # keV

        spline = inter.UnivariateSpline(E_Hraw, k_Hraw)
        sy = spline(E0)

        # ==== Plotting ===
        plt.figure()
        plt.xlim([0., 11.])
        plt.xlabel('E [keV]')
        plt.ylim([0., .51])
        plt.ylabel(r'$\Delta E_loss/E$')
        plt.plot(E_Hraw, k_Hraw, 'ko', label='H, raw data')
        lineplot = plt.plot(E0, sy, 'r-', label='H, fit')
        E_values = lineplot[0].get_xdata()
        k_values = lineplot[0].get_ydata()
        plt.legend()
        plt.title('ASPERA-4 NPD2')
        plt.show()

        return E_values, k_values

    def event_plot(self, dir, Rp, bins, lisdate, savefig=False):

        # Define Variables
        x, rho, _date = [], [], []
        dcxlist, dcrlist, dclist = [], [], []
        sensor = self.f_name[0:4]
        _date = []

        # On pick event variables
        var_date = []

        # set bin arrays
        bin_xrange = np.arange(-8, 5, bins[0])
        bin_yrange = np.arange(0, 12.5, bins[1])

        inyear = ['']
        indate = ['']
        default = False
        yrlist = False
        dlist = False

        if not Rp:
            Rp = self.get_Rp

        if isinstance(lisdate, list) and len(lisdate) == 1:
            if len(lisdate[0][:]) == 4:
                inyear[0] = lisdate[0]
                print('inyear: ', inyear[0])
            elif lisdate[0] == 'all':
                default = True
            else:
                indate[0] = lisdate[0]
                print('input date: ', indate[0])
        elif isinstance(lisdate, list) and len(lisdate[0]) == 4:
            yrlist = True
            print('input list: ', yrlist)
        elif isinstance(lisdate, list) and len(lisdate[0]) > 4:
            dlist = True
            print('input list: ', dlist)

        else:
            print('Check date input (-dl date)')

        # ======== Event Plot Setup ============
        plt.close('all')
        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax1.set_xlim([5, -8])
        ax1.set_ylim([0, 12.5])
        ax1.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax1.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)

        # ========= Time Duration Plot Setup ==========
        fig2 = plt.figure(figsize=(12, 9))
        ax2 = fig2.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax2.set_xlim([5, -8])
        ax2.set_ylim([0, 12.5])
        ax2.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax2.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)

        # =========== Event - Time Ratio =============
        fig3 = plt.figure(figsize=(12, 9))
        ax3 = fig3.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax3.set_xlim([5, -8])
        ax3.set_ylim([0, 12.5])
        ax3.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax3.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)


        # Read file containing list
        with open(self.f_path, 'r') as f:

            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict

                if len(datadict[dir + '_ph']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    if dir == 'dir0':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])
                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir0_tof'], datadict['dir0_x'], datadict['dir0_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])

                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir0_tof'], datadict['dir0_x'], datadict['dir0_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])
                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)

                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir0_tof'], datadict['dir0_x'], datadict['dir0_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)
                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)
                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir0_tof'], datadict['dir0_x'], datadict['dir0_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)
                        elif default:
                            _date.append(datadict['Date'])

                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir0_tof'], datadict['dir0_x'], datadict['dir0_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)
                        else:
                            continue

                    elif dir == 'dir1':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])

                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir1_tof'], datadict['dir1_x'], datadict['dir1_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])

                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir1_tof'], datadict['dir1_x'], datadict['dir1_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])
                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)

                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir1_tof'], datadict['dir1_x'], datadict['dir1_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)

                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])
                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)

                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir1_tof'], datadict['dir1_x'], datadict['dir1_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)


                        elif default:
                            if (255 in datadict['dir1_ph']) or (0 in datadict['dir1_ph']):
                                print('True')
                            _date.append(datadict['Date'])
                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir1_tof'], datadict['dir1_x'], datadict['dir1_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        else:
                            continue

                    elif dir == 'dir2':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])

                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir2_tof'], datadict['dir2_x'], datadict['dir2_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])
                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir2_tof'], datadict['dir2_x'], datadict['dir2_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])
                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)

                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir2_tof'], datadict['dir2_x'], datadict['dir2_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)

                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])
                                    dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                                    dcxlist.extend(dcx)
                                    dcrlist.extend(dcr)
                                    dclist.extend(dc)

                                    # limit tof from 100 - 600 ns
                                    sub_x, sub_r = self.tof_filter(datadict['dir2_tof'], datadict['dir2_x'], datadict['dir2_rho'])
                                    x.extend(sub_x)
                                    rho.extend(sub_r)

                        elif default:
                            _date.append(datadict['Date'])
                            dcx, dcr, dc = self.tdc_binning(datadict['dcx'], datadict['dcr'], datadict['dclist'])
                            dcxlist.extend(dcx)
                            dcrlist.extend(dcr)
                            dclist.extend(dc)

                            # limit tof from 100 - 600 ns
                            sub_x, sub_r = self.tof_filter(datadict['dir2_tof'], datadict['dir2_x'], datadict['dir2_rho'])
                            x.extend(sub_x)
                            rho.extend(sub_r)

                        else:
                            continue
        f.close()
        # List to Array Conversion
        x_arr = np.array(x)
        rho_arr = np.array(rho)

        dcx_arr = np.array(dcxlist)
        dcr_arr = np.array(dcrlist)

        # Divide by Rp
        x_arr = np.divide(x_arr, Rp)
        rho_arr = np.divide(rho_arr, Rp)

        dcx_arr = np.divide(dcx_arr, Rp)
        dcr_arr = np.divide(dcr_arr, Rp)

        st_year = _date[0][0:4]
        e_year = _date[-1][0:4]

        # Histogram
        hist, xedges, yedges = np.histogram2d(x_arr, rho_arr, bins=[bin_xrange, bin_yrange])

        hist2, xedges2, yedges2 = np.histogram2d(dcx_arr, dcr_arr, weights= dclist, bins=[bin_xrange, bin_yrange])

        ma_hist2 = np.ma.masked_where(hist2 == 0.0, hist2)

        ratio_hist = hist/ma_hist2

        print('No. of Events: ', x_arr.size)
        print('Accumulated Time: ', dcx_arr.size)

        ma_hist = np.ma.masked_where(hist == 0., hist)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        div = make_axes_locatable(ax1)
        cax = div.append_axes("right", size="2%", pad=0.05)

        #ma_hist2 = np.ma.masked_where(hist2 == 0., hist2)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="2%", pad=0.05)

        ma_hist3 = np.ma.masked_where(ratio_hist == 0., ratio_hist)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        div3 = make_axes_locatable(ax3)
        cax3 = div3.append_axes("right", size="2%", pad=0.05)


        if indate[0] != '':
            im = ax1.imshow(np.rot90(ma_hist), interpolation='nearest', vmin=0., vmax=10.0 ** 3,
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], picker=True)
            cb = fig.colorbar(im, cax=cax)
            # cb.formatter.set_powerlimits((0, 0))
            # cb.update_ticks()
        else:

            im = ax1.imshow(np.rot90(ma_hist), interpolation='nearest',
                            norm=colors.LogNorm(vmin=ma_hist.min(), vmax=10. ** 6),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], picker=True)

            cb = fig.colorbar(im, cax=cax)
        cb.set_label('# of Events', labelpad=2.)

        # ===================================
        if indate[0] != '':
            im2 = ax2.imshow(np.rot90(ma_hist2), interpolation='nearest', vmin=0., vmax=10.0 ** 3,
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], picker=True)
            cb2 = fig2.colorbar(im2, cax=cax2)
            # cb.formatter.set_powerlimits((0, 0))
            # cb.update_ticks()
        else:

            im2 = ax2.imshow(np.rot90(ma_hist2), interpolation='nearest',
                            norm=colors.LogNorm(vmin=ma_hist.min(), vmax=10. ** 6),
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], picker=True)

            cb2 = fig2.colorbar(im2, cax=cax2)
        cb2.set_label('sec', labelpad=2.)

        # ===================================
        if indate[0] != '':
            im2 = ax3.imshow(np.rot90(ma_hist3), interpolation='nearest', vmin=0., vmax=5*10.0 ** 3,
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], picker=True)
            cb2 = fig2.colorbar(im2, cax=cax3)
            # cb.formatter.set_powerlimits((0, 0))
            # cb.update_ticks()
        else:

            im3 = ax3.imshow(np.rot90(ma_hist3), interpolation='nearest', #vmin= 0., vmax=100,
                            norm=colors.LogNorm(vmin=1, vmax=10 ** 2),
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], picker=True)

            cb3 = fig2.colorbar(im3, cax=cax3)
        cb3.set_label('events/sec', labelpad=2.)


        # ax1.scatter(x_arr, rho_arr, lw=0, alpha=0.005)

        self.plt_venus(ax1)
        self.plt_bsicb(ax1)
        self.plt_venus(ax2)
        self.plt_bsicb(ax2)
        self.plt_venus(ax3)
        self.plt_bsicb(ax3)

        ax1.text(-6.0, 12.0, 'bin size: ' + str(bins[0]) + 'x' + str(bins[1]), horizontalalignment='center',
                 verticalalignment='top', transform=ax1.transData)
        ax2.text(-6.0, 12.0, 'bin size: ' + str(bins[0]) + 'x' + str(bins[1]), horizontalalignment='center',
                 verticalalignment='top', transform=ax2.transData)
        ax3.text(-6.0, 12.0, 'bin size: ' + str(bins[0]) + 'x' + str(bins[1]), horizontalalignment='center',
                 verticalalignment='top', transform=ax3.transData)

        if inyear[0] != '':
            ax1.set_title(
                'Events Distribution (coin 0: 1 Start-1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Year: ' + inyear[
                    0], fontsize=12)
            var_date.append(inyear[0])
            ax2.set_title(
                'Accumulated Time Distribution\n' + ', Year: ' + inyear[0], fontsize=12)

            ax3.set_title(
                'Event Rate Distribution (coin 0: 1 Start-1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Year: ' + inyear[
                    0], fontsize=12)

            if savefig:
                fig.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'event_' + dir + '.png')
                fig2.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'accutime_' + dir + '.png')
                fig3.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'etratio_' + dir + '.png')


        elif indate[0] != '':
            ax1.set_title(
                'Events Distribution (coin 0: 1 Start, 1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Date:' + indate[
                    0], fontsize=12)
            var_date.append(indate[0])
            # self.plt_orbit(ax1,indate[0], Rp)
            ax2.set_title( 'Accumulated Time Distribution\n' + 'Year: ' + inyear[0], fontsize=12)

            ax3.set_title(
                'Event Rate Distribution (coin 0: 1 Start-1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Year: ' + inyear[
                    0], fontsize=12)

            if savefig:
                fig.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            indate[0] + 'event_' + dir + '.png')

                fig2.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'accutime_' + dir + '.png')

                fig3.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'etratio_' + dir + '.png')

        elif default:
            ax1.set_title(
                'Events Distribution (coin 0: 1 Start-1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Year: 2005 - 2013',
                fontsize=12)
            var_date.append(st_year)
            var_date.append(e_year)

            ax2.set_title(
                'Accumulated Time Distribution\n' + ', Year:  2005 - 2013', fontsize=12)

            ax3.set_title(
                'Event Rate Distribution (coin 0: 1 Start-1 Stop, TOF: 100-600 ns)\n' + sensor.upper() + ': ' + dir + ', Year: 2005 - 2013', fontsize=12)

            if savefig:
                fig.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            st_year + '_' + e_year + 'event_' + dir + '.png')
                fig2.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'accutime_' + dir + '.png')
                fig3.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'etratio_' + dir + '.png')

        plt.show()

    def rec_plot(self, dir, Rp, bins, lisdate, savefig=False):

        # Define Variables
        x, rho, _date = [], [], []
        sensor = self.f_name[0:4]
        _date = []

        # On pick event variables
        cursor_data = []
        var_date = []

        # set bin arrays
        bin_xrange = np.arange(-8, 5, bins[0])
        bin_yrange = np.arange(0, 12.5, bins[1])

        inyear = ['']
        indate = ['']
        default = False
        yrlist = False
        dlist = False

        if not Rp:
            Rp = self.get_Rp

        if isinstance(lisdate, list) and len(lisdate) == 1:
            if len(lisdate[0][:]) == 4:
                inyear[0] = lisdate[0]
                print('inyear: ', inyear[0])
            elif lisdate[0] == 'all':
                default = True
            else:
                indate[0] = lisdate[0]
                print('input date: ', indate[0])
        elif isinstance(lisdate, list) and len(lisdate[0]) == 4:
            yrlist = True
            print('input list: ', yrlist)
        elif isinstance(lisdate, list) and len(lisdate[0]) > 4:
            dlist = True
            print('input list: ', dlist)

        else:
            print('Check date input (-dl date)')

        # ========Plot Setup ============
        plt.close('all')
        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax1.set_xlim([5, -8])
        ax1.set_ylim([0, 12.5])
        ax1.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax1.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)

        # Read file containing list
        with open(self.f_path, 'r') as f:

            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict

                if len(datadict[dir + '_ph']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    if dir == 'dir0':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])
                            subx, subr = self.rec_count(datadict, 'dir0_x', 'dir0_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])
                            subx, subr = self.rec_count(datadict, 'dir0_x', 'dir0_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir0_x', 'dir0_rho')
                                    x.extend(subx)
                                    rho.extend(subr)
                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir0_x', 'dir0_rho')
                                    x.extend(subx)
                                    rho.extend(subr)
                        elif default:
                            _date.append(datadict['Date'])

                            subx, subr = self.rec_count(datadict, 'dir0_x', 'dir0_rho')
                            x.extend(subx)
                            rho.extend(subr)
                        else:
                            continue

                    elif dir == 'dir1':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])
                            subx, subr = self.rec_count(datadict, 'dir1_x', 'dir1_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])
                            subx, subr = self.rec_count(datadict, 'dir1_x', 'dir1_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir1_x', 'dir1_rho')
                                    x.extend(subx)
                                    rho.extend(subr)

                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir1_x', 'dir1_rho')
                                    x.extend(subx)
                                    rho.extend(subr)

                        elif default:
                            if (255 in datadict['dir1_ph']) or (0 in datadict['dir1_ph']):
                                print('True')
                            _date.append(datadict['Date'])

                            subx, subr = self.rec_count(datadict, 'dir1_x', 'dir1_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        else:
                            continue

                    elif dir == 'dir2':
                        # ========== single input ========
                        if indate[0] == datadict['Date']:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'])
                            subx, subr = self.rec_count(datadict, 'dir2_x', 'dir2_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        elif inyear[0] == datadict['Date'][0:4]:
                            print('Found: ', datadict['Filename'])
                            _date.append(datadict['Date'][0:4])
                            subx, subr = self.rec_count(datadict, 'dir2_x', 'dir2_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        # =========== List input ==========
                        elif dlist:
                            for d in lisdate:
                                print('date: ', d)
                                if d == datadict['Date']:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir2_x', 'dir2_rho')
                                    x.extend(subx)
                                    rho.extend(subr)

                        elif yrlist:
                            for year in lisdate:
                                if year == datadict['Date'][0:4]:
                                    print('Found: ', datadict['Filename'])
                                    _date.append(datadict['Date'])

                                    subx, subr = self.rec_count(datadict, 'dir2_x', 'dir2_rho')
                                    x.extend(subx)
                                    rho.extend(subr)

                        elif default:
                            _date.append(datadict['Date'])

                            subx, subr = self.rec_count(datadict, 'dir2_x', 'dir2_rho')
                            x.extend(subx)
                            rho.extend(subr)

                        else:
                            continue
        f.close()
        # List to Array Conversion
        x_arr = np.array(x)
        rho_arr = np.array(rho)

        # Divide by Rp
        x_arr = np.divide(x_arr, Rp)
        rho_arr = np.divide(rho_arr, Rp)

        st_year = _date[0][0:4]
        e_year = _date[-1][0:4]

        # Histogram
        hist, xedges, yedges = np.histogram2d(x_arr, rho_arr, bins=[bin_xrange, bin_yrange])

        print('No. Records: ', x_arr.size)

        ma_hist = np.ma.masked_where(hist == 0., hist)
        # cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
        div = make_axes_locatable(ax1)
        cax = div.append_axes("right", size="2%", pad=0.05)

        if indate[0] != '':
            im = ax1.imshow(np.rot90(ma_hist), interpolation='nearest', vmin=0., vmax=10.0 ** 3,
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], picker=True)
            cb = fig.colorbar(im, cax=cax)
            # cb.formatter.set_powerlimits((0, 0))
            # cb.update_ticks()
        else:

            im = ax1.imshow(np.rot90(ma_hist), interpolation='nearest',
                            norm=colors.LogNorm(vmin=ma_hist.min(), vmax=10. ** 6),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], picker=True)

            cb = fig.colorbar(im, cax=cax)
        cb.set_label('# of Records', labelpad=2.)

        # ax1.scatter(x_arr, rho_arr, lw=0, alpha=0.005)

        self.plt_venus(ax1)
        self.plt_bsicb(ax1)
        ax1.text(-6.0, 12.0, 'bin size: ' + str(bins[0]) + 'x' + str(bins[1]), horizontalalignment='center',
                 verticalalignment='top', transform=ax1.transData)

        # ======== Button axes ==============
        ax2 = fig.add_axes([0.80, 0.01, 0.01, 0.01], frameon=False)
        ax2.axes.get_yaxis().set_visible(False)
        ax2.axes.get_xaxis().set_visible(False)
        ax2.text(0.5, 0.5, 'Reset', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax2.transAxes,
                 picker=True)

        ax3 = fig.add_axes([0.60, 0.01, 0.01, 0.01], frameon=False)
        ax3.axes.get_yaxis().set_visible(False)
        ax3.axes.get_xaxis().set_visible(False)
        ax3.text(0.5, 0.5, 'Gauss', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax3.transAxes,
                 picker=True)

        ax4 = fig.add_axes([0.70, 0.01, 0.01, 0.01], frameon=False)
        ax4.axes.get_yaxis().set_visible(False)
        ax4.axes.get_xaxis().set_visible(False)
        ax4.text(0.5, 0.5, 'TOF-PH', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax4.transAxes,
                 picker=True)

        ax5 = fig.add_axes([0.50, 0.01, 0.01, 0.01], frameon=False)
        ax5.axes.get_yaxis().set_visible(False)
        ax5.axes.get_xaxis().set_visible(False)
        ax5.text(0.5, 0.5, 'Maxwell', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax5.transAxes,
                 picker=True)

        if inyear[0] != '':
            ax1.set_title(
                'Records Distribution (coin 0: 1 Start, 1 Stop)\n' + sensor.upper() + ': ' + dir + ', Year: ' + inyear[
                    0], fontsize=12)
            var_date.append(inyear[0])

            if savefig:
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            inyear[0] + 'rec_' + dir + '.png')

        elif indate[0] != '':
            ax1.set_title(
                'Records Distribution (coin 0: 1 Start, 1 Stop)\n' + sensor.upper() + ': ' + dir + ', Date:' + indate[
                    0], fontsize=12)
            var_date.append(indate[0])
            # self.plt_orbit(ax1,indate[0], Rp)
            if savefig:
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            indate[0] + 'rec_' + dir + '.png')
        else:
            ax1.set_title(
                'Records Distribution (coin 0: 1 Start, 1 Stop)\n' + sensor.upper() + ': ' + dir + ', Year: ' + st_year + '-' + e_year,
                fontsize=12)
            var_date.append(st_year)
            var_date.append(e_year)

            if savefig:
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            st_year + '_' + e_year + 'rec_' + dir + '.png')

        # plt.tight_layout()

        # ============ Pick Event =================
        def onpick(event):  # pick coordinate to plot TOF-PH plot
            artist = event.artist

            if isinstance(artist, AxesImage):
                mouseevent = event.mouseevent
                xdata = mouseevent.xdata
                ydata = mouseevent.ydata
                print('on pick', xdata, ydata)
                cursor_data.append([xdata, ydata])
            elif isinstance(event.artist, Text):
                text = event.artist
                id = text.get_text()
                if id == 'Reset':
                    cursor_data.clear()
                    print('Reset points: ', cursor_data)
                elif id == 'Gauss' and len(cursor_data) == 1:
                    print('Generating 10 binned tof distribution plots from dir0, dir1, dir2..')
                    self.rad10_gaussian(xedges, yedges, cursor_data, sensor, lisdate, 'dir0', bins, Rp)
                    self.rad10_gaussian(xedges, yedges, cursor_data, sensor, lisdate, 'dir1', bins, Rp)
                    self.rad10_gaussian(xedges, yedges, cursor_data, sensor, lisdate, 'dir2', bins, Rp)

                elif id == 'TOF-PH' and len(cursor_data) == 1:
                    # Call TOF-PH plot
                    print('Generating 5 TOF-PH plots...')
                    self.radial_tofph(hist, xedges, yedges, cursor_data, sensor, lisdate, dir, bins, Rp)

                elif id == 'Maxwell' and len(cursor_data) == 1:
                    print('Generating 10 binned velocity from tof plots from dir0, dir1, dir2..')
                    self.rad10_maxwellian(yedges, yedges, cursor_data, sensor, lisdate, 'dir2', bins, Rp)

                elif len(cursor_data) > 1:
                    print('Reset first to clear data points')

        fig.canvas.mpl_connect('pick_event', onpick)

        plt.show()

    def rec_count(self, datadict, dirx, dirr):

        x, rho = [], []

        for i in range(len(datadict[dirx])):
            if i != 0:
                if (datadict[dirx][i] != datadict[dirx][i - 1]) and \
                        (datadict[dirr][i] != datadict[dirr][i - 1]):  # if coordinates not identical
                    x.append(datadict[dirx][i])
                    rho.append(datadict[dirr][i])
            else:
                x.append(datadict[dirx][0])
                rho.append(datadict[dirr][0])

        return x, rho

    def radial_tofph(self, hist, xedges, yedges, cursor, sensor, lisdate, dir, bins,
                     Rp):  # Generate 5 TOF-PH plots radially

        # 5 TOF-PH plots generated from chosen cursor point.
        # Method, looks at bin edges, 1 increment until 5 max

        # ========Plot Setup ============
        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(5, 5)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.0, hspace=0.2)
        ax5 = fig.add_subplot(gs[4, 0:2])
        ax4 = fig.add_subplot(gs[3, 0:2])
        ax3 = fig.add_subplot(gs[2, 0:2], sharex=ax4)
        ax2 = fig.add_subplot(gs[1, 0:2], sharex=ax4)
        ax1 = fig.add_subplot(gs[0, 0:2], sharex=ax4)
        # mod_ax1 = host_subplot(gs[0, 0:2], sharex= ax4, axes_class=AA.Axes)
        ax6 = fig.add_subplot(gs[:, 3:])

        # ======== ax1-ax5 properties =======
        ax1.set_ylim([0, 255])
        ax1.set_ylabel('PH', multialignment='center', fontsize=9)
        ax2.set_ylim([0, 255])
        ax2.set_ylabel('PH', multialignment='center', fontsize=9)
        ax3.set_ylim([0, 255])
        ax3.set_ylabel('PH', multialignment='center', fontsize=9)
        ax4.set_ylim([0, 255])
        ax4.set_ylabel('PH', multialignment='center', fontsize=9)
        ax5.set_ylim([0, 255])
        ax5.set_ylabel('PH', multialignment='center', fontsize=9)
        ax4.set_xlim([1, 2048])
        ax5.set_xlim([1, 2048])
        ax5.set_xlabel('TOF [ns]', multialignment='center', fontsize=9)
        # Tick locator
        self.tick_locator(ax1, 50, 250)
        self.tick_locator(ax2, 50, 250)
        self.tick_locator(ax3, 50, 250)
        self.tick_locator(ax4, 50, 250)
        self.tick_locator(ax5, 50, 250)
        ax1.axes.xaxis.set_ticklabels([])
        ax2.axes.xaxis.set_ticklabels([])
        ax3.axes.xaxis.set_ticklabels([])
        ax4.axes.xaxis.set_ticklabels([])

        # ==== add top axis(ax1-ax5) subplots ====
        new_ticks_loc = np.array([1., 250., 600.])
        vel_ticks = [self.tofto_vel(tof) for tof in new_ticks_loc]
        vel_ticklabels = ["%.2E" % label for label in vel_ticks]
        vel_ticklabels[-1] += ' [km/s]'
        ax1_top = ax1.twiny()  # responsible for top axis
        ax1_top.set_xlim(ax1.get_xlim())
        ax1_top.set_xticks(new_ticks_loc)
        ax1_top.set_xticklabels(vel_ticklabels, fontsize=9)

        # ====== ax6 properties ======
        plt.grid()
        ax6.set_xlim([5, -8])
        ax6.set_ylim([0, 12.5])
        ax6.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax6.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)
        if lisdate[0] == 'all':
            ax6.set_title(
                'Records Distribution (coin 0: 1 Start, 1 Stop)\n' + sensor.upper() + ': ' + dir + ', Date: 2005-2013',
                fontsize=12)
        else:
            ax6.set_title(
                'Records Distribution (coin 0: 1 Start, 1 Stop)\n' + sensor.upper() + ': ' + dir + ', Date:' + lisdate[
                    0], fontsize=12)

        # ======= Replotting at ax6 ===========

        ma_hist = np.ma.masked_where(hist == 0., hist)
        # cbar_ax = fig.add_axes([0.92, 0.14, 0.01, 0.6])
        div = make_axes_locatable(ax6)
        cax = div.append_axes("right", size="2%", pad=0.05)

        if len(lisdate[0]) == 10:  # if date (yyyy-mm-dd)
            im = ax6.imshow(np.rot90(ma_hist), interpolation='nearest', vmin=0., vmax=10.0 ** 3,
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            cb = fig.colorbar(im, cax=cax)
        else:
            im = ax6.imshow(np.rot90(ma_hist), interpolation='nearest',
                            norm=colors.LogNorm(vmin=ma_hist.min(), vmax=10. ** 6),
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            cb = fig.colorbar(im, cax=cax)

        cb.set_label('# of Records', labelpad=2., size=10)

        # ======= Setting data bounds from bins=========
        xdbin = bins[0] / float(2)
        ydbin = bins[1] / float(2)
        xcursor, ycursor = zip(*cursor)
        xc_bounds = [[x - xdbin, x + xdbin] for x in xcursor]
        yc_bounds = [[y - ydbin, y + ydbin] for y in ycursor]

        # Checking
        '''print('cursor: ', cursor)
        print('xc_bounds: ', xc_bounds)
        print('yc_bounds: ', yc_bounds)'''

        # Find nearest value from histogram edges
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        xbin_bounds = [[find_nearest(xedges, xc_bounds[i][0]), find_nearest(xedges, xc_bounds[i][1])] for i in
                       range(len(xc_bounds))]
        ybin_bounds = [[find_nearest(yedges, yc_bounds[i][0]), find_nearest(yedges, yc_bounds[i][1])] for i in
                       range(len(yc_bounds))]

        # Increment bins radially
        idx0 = yedges[yedges == ybin_bounds[0][0]]
        idx1 = yedges[yedges == ybin_bounds[0][1]]
        ybin_bounds = [[yedges[yedges == idx0 + (i * bins[1])], yedges[yedges == idx1 + (i * bins[1])]] for i in
                       range(5)]

        # Reshape
        ybin_bounds = np.array(ybin_bounds).reshape(len(ybin_bounds), 2)

        '''print('nearest value: ',find_nearest(xedges, xc_bounds[0][0]), find_nearest(xedges, xc_bounds[0][1]))
        print('xedge: ', xedges)'''

        # Frame to data value conversion
        xbounds = np.multiply(xbin_bounds, Rp)
        ybounds = np.multiply(ybin_bounds, Rp)

        # Checking
        '''print('xbounds: ', xbounds, type(xbounds))
        print('ybounds: ', ybounds)'''

        # Define Variables
        axbins = [100, 25]
        xlist1, rlist1, toflist1, phlist1, dc1 = [], [], [], [], []
        xlist2, rlist2, toflist2, phlist2, dc2 = [], [], [], [], []
        xlist3, rlist3, toflist3, phlist3, dc3 = [], [], [], [], []
        xlist4, rlist4, toflist4, phlist4, dc4 = [], [], [], [], []
        xlist5, rlist5, toflist5, phlist5, dc5 = [], [], [], [], []
        rec_cnt1, rec_cnt2, rec_cnt3, rec_cnt4, rec_cnt5 = 0., 0., 0., 0., 0.

        # ========= Matching Data ==================
        # Read file containing list
        with open(self.f_path, 'r') as f:
            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict
                if len(datadict[dir + '_ph']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    if lisdate[0] == datadict['Date']:
                        # Call binned data values
                        x_tile1, r_tile1, tof_tile1, ph_tile1, records1, dctile1 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[0], dir)
                        if x_tile1:
                            xlist1.extend(x_tile1), rlist1.extend(r_tile1), toflist1.extend(tof_tile1), phlist1.extend(
                                ph_tile1), dc1.extend(dctile1)
                            rec_cnt1 += records1
                        x_tile2, r_tile2, tof_tile2, ph_tile2, records2, dctile2 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[1], dir)
                        if x_tile2:
                            xlist2.extend(x_tile2), rlist2.extend(r_tile2), toflist2.extend(tof_tile2), phlist2.extend(
                                ph_tile2), dc2.extend(dctile2)
                            rec_cnt2 += records2
                        x_tile3, r_tile3, tof_tile3, ph_tile3, records3, dctile3 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[2], dir)
                        if x_tile3:
                            xlist3.extend(x_tile3), rlist3.extend(r_tile3), toflist3.extend(tof_tile3), phlist3.extend(
                                ph_tile3), dc3.extend(dctile3)
                            rec_cnt3 += records3
                        x_tile4, r_tile4, tof_tile4, ph_tile4, records4, dctile4 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[3], dir)
                        if x_tile4:
                            xlist4.extend(x_tile4), rlist4.extend(r_tile4), toflist4.extend(tof_tile4), phlist4.extend(
                                ph_tile4), dc4.extend(dctile4)
                            rec_cnt4 += records4
                        x_tile5, r_tile5, tof_tile5, ph_tile5, records5, dctile5 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[4], dir)
                        if x_tile5:
                            xlist5.extend(x_tile5), rlist5.extend(r_tile5), toflist5.extend(tof_tile5), phlist5.extend(
                                ph_tile5), dc5.extend(dctile5)
                            rec_cnt5 += records5

                    elif lisdate[0] == 'all':
                        # Call binned data values
                        x_tile1, r_tile1, tof_tile1, ph_tile1, records1, dctile1 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[0], dir)
                        if x_tile1:
                            xlist1.extend(x_tile1), rlist1.extend(r_tile1), toflist1.extend(tof_tile1), phlist1.extend(
                                ph_tile1), dc1.extend(dctile1)
                            rec_cnt1 += records1
                        x_tile2, r_tile2, tof_tile2, ph_tile2, records2, dctile2 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[1], dir)
                        if x_tile2:
                            xlist2.extend(x_tile2), rlist2.extend(r_tile2), toflist2.extend(tof_tile2), phlist2.extend(
                                ph_tile2), dc2.extend(dctile2)
                            rec_cnt2 += records2
                        x_tile3, r_tile3, tof_tile3, ph_tile3, records3, dctile3 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[2], dir)
                        if x_tile3:
                            xlist3.extend(x_tile3), rlist3.extend(r_tile3), toflist3.extend(tof_tile3), phlist3.extend(
                                ph_tile3), dc3.extend(dctile3)
                            rec_cnt3 += records3
                        x_tile4, r_tile4, tof_tile4, ph_tile4, records4, dctile4 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[3], dir)
                        if x_tile4:
                            xlist4.extend(x_tile4), rlist4.extend(r_tile4), toflist4.extend(tof_tile4), phlist4.extend(
                                ph_tile4), dc4.extend(dctile4)
                            rec_cnt4 += records4
                        x_tile5, r_tile5, tof_tile5, ph_tile5, records5, dctile5 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[4], dir)
                        if x_tile5:
                            xlist5.extend(x_tile5), rlist5.extend(r_tile5), toflist5.extend(tof_tile5), phlist5.extend(
                                ph_tile5), dc5.extend(dctile5)
                            rec_cnt5 += records5
        f.close()
        xlist1 = np.array(xlist1) / float(Rp)
        mean_x1 = np.mean(xlist1)  # plotting purposes (1 coord)
        rlist1 = np.array(rlist1) / float(Rp)
        mean_r1 = np.mean(rlist1)
        xlist2 = np.array(xlist2) / float(Rp)
        mean_x2 = np.mean(xlist2)
        rlist2 = np.array(rlist2) / float(Rp)
        mean_r2 = np.mean(rlist2)
        xlist3 = np.array(xlist3) / float(Rp)
        mean_x3 = np.mean(xlist3)
        rlist3 = np.array(rlist3) / float(Rp)
        mean_r3 = np.mean(rlist3)
        xlist4 = np.array(xlist4) / float(Rp)
        mean_x4 = np.mean(xlist4)
        rlist4 = np.array(rlist4) / float(Rp)
        mean_r4 = np.mean(rlist4)
        xlist5 = np.array(xlist5) / float(Rp)
        mean_x5 = np.mean(xlist5)
        rlist5 = np.array(rlist5) / float(Rp)
        mean_r5 = np.mean(rlist5)

        ax5.set_title(
            'TOF-PH Normed (Total Event Cnts: ' + str(len(toflist1)) + ', binning: ' + str(axbins[0]) + 'x' + str(
                axbins[1]) + ')', fontsize=10)
        ax4.set_title(
            'TOF-PH Normed (Total Event Cnts: ' + str(len(toflist2)) + ', binning: ' + str(axbins[0]) + 'x' + str(
                axbins[1]) + ')', fontsize=10)
        ax3.set_title(
            'TOF-PH Normed (Total Event Cnts: ' + str(len(toflist2)) + ', binning: ' + str(axbins[0]) + 'x' + str(
                axbins[1]) + ')', fontsize=10)
        ax2.set_title(
            'TOF-PH Normed (Total Event Cnts: ' + str(len(toflist4)) + ', binning: ' + str(axbins[0]) + 'x' + str(
                axbins[1]) + ')', fontsize=10)
        # ax1_top.set_title('TOF-PH Normed (Total cnt: '+ str(len(toflist5)) + ', binning: ' + str(axbins[0])+ 'x' + str(axbins[1]) +')', fontsize= 10)
        ax1_top.set_xlabel(
            'TOF-PH Normed (Total Event Cnts: ' + str(len(toflist5)) + ', binning: ' + str(axbins[0]) + 'x' + str(
                axbins[1]) + ')', fontsize=10)
        # ============= Plotting ================
        self.subplot_tofph(ax5, toflist1, phlist1, tof_bin=axbins)
        self.subplot_tofph(ax4, toflist2, phlist2, tof_bin=axbins)
        self.subplot_tofph(ax3, toflist3, phlist3, tof_bin=axbins)
        self.subplot_tofph(ax2, toflist4, phlist4, tof_bin=axbins)
        self.subplot_tofph(ax1, toflist5, phlist5, tof_bin=axbins)
        # self.subplot_tofph(mod_ax1,toflist5, phlist5, tof_bin=axbins)

        ax6.scatter([mean_x1, mean_x2, mean_x3, mean_x4, mean_x5], [mean_r1, mean_r2, mean_r3, mean_r4, mean_r5], c='k',
                    marker='s',
                    label='x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
                        ybin_bounds[0][0]) + ' - ' + str(ybin_bounds[0][1]), s=25.0)
        ax6.text(-6.0, 10.0, 'bin size: ' + str(bins[0]) + 'x' + str(bins[1]), horizontalalignment='center',
                 verticalalignment='top', transform=ax6.transData, fontsize=9)

        # ========= annotate ax ============
        ax6.annotate('x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
            ybin_bounds[0][0]) + ' - ' + str(ybin_bounds[0][1]),
                     xy=(mean_x1, mean_r1), xycoords='data', xytext=(0.52, 0.12), textcoords='figure fraction',
                     arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.1), horizontalalignment='right',
                     verticalalignment='center')
        ax6.annotate('x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
            ybin_bounds[1][0]) + ' - ' + str(ybin_bounds[1][1]),
                     xy=(mean_x2, mean_r2), xycoords='data', xytext=(0.52, 0.30), textcoords='figure fraction',
                     arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.1), horizontalalignment='right',
                     verticalalignment='center')
        ax6.annotate('x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
            ybin_bounds[2][0]) + ' - ' + str(ybin_bounds[2][1]),
                     xy=(mean_x3, mean_r3), xycoords='data', xytext=(0.52, 0.48), textcoords='figure fraction',
                     arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.1), horizontalalignment='right',
                     verticalalignment='center')
        ax6.annotate('x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
            ybin_bounds[3][0]) + ' - ' + str(ybin_bounds[3][1]),
                     xy=(mean_x4, mean_r4), xycoords='data', xytext=(0.52, 0.66), textcoords='figure fraction',
                     arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.1), horizontalalignment='right',
                     verticalalignment='center')
        ax6.annotate('x: ' + str(xbin_bounds[0][0]) + ' - ' + str(xbin_bounds[0][1]) + '\n' + 'r: ' + str(
            ybin_bounds[4][0]) + ' - ' + str(ybin_bounds[4][1]),
                     xy=(mean_x5, mean_r5), xycoords='data', xytext=(0.52, 0.84), textcoords='figure fraction',
                     arrowprops=dict(facecolor='none', edgecolor='black', shrink=0.1), horizontalalignment='right',
                     verticalalignment='center')
        self.plt_venus(ax6)
        # ========= Save Button =================
        ax7 = fig.add_axes([0.85, 0.01, 0.01, 0.01], frameon=False)
        ax7.axes.get_yaxis().set_visible(False)
        ax7.axes.get_xaxis().set_visible(False)
        ax7.text(0.5, 0.5, 'Save', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax7.transAxes,
                 picker=True)

        # ============ Pick Event =================
        def onpick(event):  # pick coordinate to plot TOF-PH plot
            artist = event.artist
            if isinstance(artist, Text):
                self.set_figcount(1)
                figcount = self.get_figcount
                plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                            lisdate[0] + 'rectofph_' + dir + '_' + str(xbin_bounds[0][0]) + str(xbin_bounds[0][1]) +
                            str(ybin_bounds[0][0]) + str(ybin_bounds[0][1]) + '.png')
                print('Save as ' + sensor + '_' + str(lisdate[0]) + 'rectofph_' + dir + '_' + str(
                    xbin_bounds[0][0]) + str(xbin_bounds[0][1]) +
                      str(ybin_bounds[0][0]) + str(ybin_bounds[0][1]) + '.png')

                # ======= Plot and Save Energy Spectra ========
                self.plt_tofspec(xbin_bounds[0], ybin_bounds[0], toflist1, phlist1, rec_cnt1, axbins, sensor, lisdate,
                                 dir, figcount)
                figcount += 1
                '''self.plt_tofspec(xbin_bounds[0], ybin_bounds[1], toflist2, phlist2, rec_cnt2, axbins, sensor, lisdate, dir, figcount)
                figcount += 1
                self.plt_tofspec(xbin_bounds[0], ybin_bounds[2], toflist3, phlist3, rec_cnt3, axbins, sensor, lisdate, dir, figcount)
                figcount += 1
                self.plt_tofspec(xbin_bounds[0], ybin_bounds[3], toflist4, phlist4, rec_cnt4, axbins, sensor, lisdate, dir, figcount)
                figcount += 1
                self.plt_tofspec(xbin_bounds[0], ybin_bounds[4], toflist5, phlist5, rec_cnt5, axbins, sensor, lisdate, dir, figcount)'''

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()

    def rad10_gaussian(self, xedges, yedges, cursor, sensor, lisdate, dir, bins,
                       Rp):  # Generate 10 velocity plots radially

        # 10 radially binned TOF-PH plots generated from chosen cursor point.
        # Method, looks at bin edges, 1 increment until 5 max

        # ======= Setting data bounds from bins=========
        xdbin = bins[0] / float(2)
        ydbin = bins[1] / float(2)
        xcursor, ycursor = zip(*cursor)
        xc_bounds = [[x - xdbin, x + xdbin] for x in xcursor]
        yc_bounds = [[y - ydbin, y + ydbin] for y in ycursor]

        # Checking
        '''print('cursor: ', cursor)
        print('xc_bounds: ', xc_bounds)
        print('yc_bounds: ', yc_bounds)'''

        # Find nearest value from histogram edges
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        xbin_bounds = [[find_nearest(xedges, xc_bounds[i][0]), find_nearest(xedges, xc_bounds[i][1])] for i in
                       range(len(xc_bounds))]
        ybin_bounds = [[find_nearest(yedges, yc_bounds[i][0]), find_nearest(yedges, yc_bounds[i][1])] for i in
                       range(len(yc_bounds))]

        # Increment bins radially
        idx0 = yedges[yedges == ybin_bounds[0][0]]
        idx1 = yedges[yedges == ybin_bounds[0][1]]
        ybin_bounds = [[yedges[yedges == idx0 + (i * bins[1])], yedges[yedges == idx1 + (i * bins[1])]] for i in
                       range(10)]

        # Reshape
        ybin_bounds = np.array(ybin_bounds).reshape(len(ybin_bounds), 2)

        '''print('nearest value: ',find_nearest(xedges, xc_bounds[0][0]), find_nearest(xedges, xc_bounds[0][1]))
        print('xedge: ', xedges)'''

        # Frame to data value conversion
        xbounds = np.multiply(xbin_bounds, Rp)
        ybounds = np.multiply(ybin_bounds, Rp)

        # Checking
        '''print('xbounds: ', xbounds, type(xbounds))
        print('ybounds: ', ybounds)'''

        # Define Variables
        axbins = [100, 25]

        xlist1, rlist1, toflist1, phlist1, dtlist1 = [], [], [], [], []
        xlist2, rlist2, toflist2, phlist2, dtlist2 = [], [], [], [], []
        xlist3, rlist3, toflist3, phlist3, dtlist3 = [], [], [], [], []
        xlist4, rlist4, toflist4, phlist4, dtlist4= [], [], [], [], []
        xlist5, rlist5, toflist5, phlist5, dtlist5 = [], [], [], [], []
        #rec_cnt1, rec_cnt2, rec_cnt3, rec_cnt4, rec_cnt5 = 0., 0., 0., 0., 0.

        xlist6, rlist6, toflist6, phlist6, dtlist6 = [], [], [], [], []
        xlist7, rlist7, toflist7, phlist7, dtlist7 = [], [], [], [], []
        xlist8, rlist8, toflist8, phlist8, dtlist8 = [], [], [], [], []
        xlist9, rlist9, toflist9, phlist9, dtlist9 = [], [], [], [], []
        xlist10, rlist10, toflist10, phlist10, dtlist10 = [], [], [], [], []
        #rec_cnt6, rec_cnt7, rec_cnt8, rec_cnt9, rec_cnt10 = 0., 0., 0., 0., 0.

        # ========= Matching Data ==================
        # Read file containing list
        with open(self.f_path, 'r') as f:
            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict
                if len(datadict[dir + '_ph']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    if lisdate[0] == 'all':
                        # Call binned data values
                        x_tile1, r_tile1, tof_tile1, ph_tile1, dt1 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[0], dir)
                        if x_tile1:
                            xlist1.extend(x_tile1), rlist1.extend(r_tile1), toflist1.extend(tof_tile1), phlist1.extend(
                                ph_tile1), dtlist1.append(dt1)
                            #rec_cnt1 += records1

                        x_tile2, r_tile2, tof_tile2, ph_tile2, dt2 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[1], dir)
                        if x_tile2:
                            xlist2.extend(x_tile2), rlist2.extend(r_tile2), toflist2.extend(tof_tile2), phlist2.extend(
                                ph_tile2), dtlist2.append(dt2)
                            #rec_cnt2 += records2

                        x_tile3, r_tile3, tof_tile3, ph_tile3, dt3 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[2], dir)
                        if x_tile3:
                            xlist3.extend(x_tile3), rlist3.extend(r_tile3), toflist3.extend(tof_tile3), phlist3.extend(
                                ph_tile3), dtlist3.append(dt3)
                            #rec_cnt3 += records3

                        x_tile4, r_tile4, tof_tile4, ph_tile4, dt4 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[3], dir)
                        if x_tile4:
                            xlist4.extend(x_tile4), rlist4.extend(r_tile4), toflist4.extend(tof_tile4), phlist4.extend(
                                ph_tile4), dtlist4.append(dt4)
                            #rec_cnt4 += records4

                        x_tile5, r_tile5, tof_tile5, ph_tile5, dt5 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[4], dir)

                        if x_tile5:
                            xlist5.extend(x_tile5), rlist5.extend(r_tile5), toflist5.extend(tof_tile5), phlist5.extend(
                                ph_tile5), dtlist5.append(dt5)
                            #rec_cnt5 += records5

                        x_tile6, r_tile6, tof_tile6, ph_tile6, dt6 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[5], dir)
                        if x_tile6:
                            xlist6.extend(x_tile6), rlist6.extend(r_tile6), toflist6.extend(tof_tile6), phlist6.extend(
                                ph_tile6), dtlist6.append(dt6)
                            #rec_cnt6 += records6

                        x_tile7, r_tile7, tof_tile7, ph_tile7, dt7 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[6], dir)
                        if x_tile7:
                            xlist7.extend(x_tile7), rlist7.extend(r_tile7), toflist7.extend(tof_tile7), phlist7.extend(
                                ph_tile7), dtlist7.append(dt7)
                            #rec_cnt7 += records7

                        x_tile8, r_tile8, tof_tile8, ph_tile8, dt8 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[7], dir)
                        if x_tile8:
                            xlist8.extend(x_tile8), rlist8.extend(r_tile8), toflist8.extend(tof_tile8), phlist8.extend(
                                ph_tile8), dtlist8.append(dt8)
                            #rec_cnt8 += records8

                        x_tile9, r_tile9, tof_tile9, ph_tile9, dt9 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[8], dir)
                        if x_tile9:
                            xlist9.extend(x_tile9), rlist9.extend(r_tile9), toflist9.extend(tof_tile9), phlist9.extend(
                                ph_tile9), dtlist9.append(dt9)
                            #rec_cnt9 += records9


                        x_tile10, r_tile10, tof_tile10, ph_tile10, dt10 = self.binned_data(datadict, xbounds[0],
                                                                                                ybounds[9], dir)

                        if x_tile10:
                            xlist10.extend(x_tile10), rlist10.extend(r_tile10), toflist10.extend(tof_tile10), phlist10.extend(ph_tile10), dtlist10.append(dt10)
                            #rec_cnt10 += records10
        f.close()

        xlist1 = np.array(xlist1) / float(Rp)
        mean_x1 = np.mean(xlist1)  # plotting purposes (1 coord)
        rlist1 = np.array(rlist1) / float(Rp)
        mean_r1 = np.mean(rlist1)

        xlist2 = np.array(xlist2) / float(Rp)
        mean_x2 = np.mean(xlist2)
        rlist2 = np.array(rlist2) / float(Rp)
        mean_r2 = np.mean(rlist2)

        xlist3 = np.array(xlist3) / float(Rp)
        mean_x3 = np.mean(xlist3)
        rlist3 = np.array(rlist3) / float(Rp)
        mean_r3 = np.mean(rlist3)

        xlist4 = np.array(xlist4) / float(Rp)
        mean_x4 = np.mean(xlist4)
        rlist4 = np.array(rlist4) / float(Rp)
        mean_r4 = np.mean(rlist4)

        xlist5 = np.array(xlist5) / float(Rp)
        mean_x5 = np.mean(xlist5)
        rlist5 = np.array(rlist5) / float(Rp)
        mean_r5 = np.mean(rlist5)

        xlist6 = np.array(xlist6) / float(Rp)
        mean_x6 = np.mean(xlist6)
        rlist6 = np.array(rlist6) / float(Rp)
        mean_r6 = np.mean(rlist6)

        xlist7 = np.array(xlist7) / float(Rp)
        mean_x7 = np.mean(xlist7)
        rlist7 = np.array(rlist7) / float(Rp)
        mean_r7 = np.mean(rlist7)

        xlist8 = np.array(xlist8) / float(Rp)
        mean_x8 = np.mean(xlist8)
        rlist8 = np.array(rlist8) / float(Rp)
        mean_r8 = np.mean(rlist8)

        xlist9 = np.array(xlist9) / float(Rp)
        mean_x9 = np.mean(xlist9)
        rlist9 = np.array(rlist9) / float(Rp)
        mean_r9 = np.mean(rlist9)

        xlist10 = np.array(xlist10) / float(Rp)
        mean_x10 = np.mean(xlist10)
        rlist10 = np.array(rlist10) / float(Rp)
        mean_r10 = np.mean(rlist10)

        # ================ Plot Setup ==============
        fig, axarr = plt.subplots(nrows=2, ncols=5, figsize=(17, 8))
        fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
        fig.suptitle('TOF Spectra of ENA (coin 0: 1 Start, 1 Stop), x: ' + str(xbin_bounds[0][0]) + '-' + str(
            xbin_bounds[0][1]) + r'$R_V$' + ', r: ' + str(ybin_bounds[0][0]) +
                     '-' + str(
            ybin_bounds[9][1]) + r'$R_V$' + '\n' + 'VSO, ' + sensor.upper() + ', Ch: ' + dir + ', Date: 2005-2013',
                     fontsize=11, y=0.98)

        # ======= Normalize with Duty Time, Fit the data, slice and Plot ===========
        tof_bin = 512
        mtof1, mcnt1, hwhmtof1 = self.dt_norm(axarr[0, 0], toflist1, phlist1, tof_bin, dtlist1)
        mtof2, mcnt2, hwhmtof2 = self.dt_norm(axarr[0, 1], toflist2, phlist2, tof_bin, dtlist2)
        mtof3, mcnt3, hwhmtof3 = self.dt_norm(axarr[0, 2], toflist3, phlist3, tof_bin, dtlist3)
        mtof4, mcnt4, hwhmtof4 = self.dt_norm(axarr[0, 3], toflist4, phlist4, tof_bin, dtlist4)
        mtof5, mcnt5, hwhmtof5 = self.dt_norm(axarr[0, 4], toflist5, phlist5, tof_bin, dtlist5)
        mtof6, mcnt6, hwhmtof6 = self.dt_norm(axarr[1, 0], toflist6, phlist6, tof_bin, dtlist6)
        mtof7, mcnt7, hwhmtof7 = self.dt_norm(axarr[1, 1], toflist7, phlist7, tof_bin, dtlist7)
        mtof8, mcnt8, hwhmtof8 = self.dt_norm(axarr[1, 2], toflist8, phlist8, tof_bin, dtlist8)
        mtof9, mcnt9, hwhmtof9 = self.dt_norm(axarr[1, 3], toflist9, phlist9, tof_bin, dtlist9)
        mtof10, mcnt10, hwhmtof10 = self.dt_norm(axarr[1, 4], toflist10, phlist10, tof_bin, dtlist10)


        # =========== Stack Plot ==============
        stacktoflist = []
        dtlist = []
        stacktoflist.append(toflist1)
        stacktoflist.append(toflist2)
        stacktoflist.append(toflist3)
        stacktoflist.append(toflist4)
        stacktoflist.append(toflist5)
        stacktoflist.append(toflist6)
        stacktoflist.append(toflist7)
        stacktoflist.append(toflist8)
        stacktoflist.append(toflist9)
        stacktoflist.append(toflist10)

        dtlist.append(dtlist1)
        dtlist.append(dtlist2)
        dtlist.append(dtlist3)
        dtlist.append(dtlist4)
        dtlist.append(dtlist5)
        dtlist.append(dtlist6)
        dtlist.append(dtlist7)
        dtlist.append(dtlist8)
        dtlist.append(dtlist9)
        dtlist.append(dtlist10)

        # uncomment for rowstack
        #self.subplot_rowstack(stacktoflist, dtlist, tof_bin, xbin_bounds, ybin_bounds, sensor, dir)
        #self.gauss_fit(stacktoflist, dtlist, tof_bin, xbin_bounds, ybin_bounds, sensor, dir)


        for i in range(5):
            axarr[1, i].set_xlabel('TOF [ns]')
        axarr[0, 0].set_ylabel(r'$cnts \ s^{-1}ns^{-1}$', multialignment='center', fontsize = 10)
        axarr[1, 0].set_ylabel(r'$cnts \ s^{-1}ns^{-1}$', multialignment='center', fontsize = 10)

        # ========= Store Button =================
        ax7 = fig.add_axes([0.85, 0.01, 0.01, 0.01], frameon=False)
        ax7.axes.get_yaxis().set_visible(False)
        ax7.axes.get_xaxis().set_visible(False)
        ax7.text(0.5, 0.5, 'Store', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax7.transAxes,
                 picker=True)

        ax8 = fig.add_axes([0.65, 0.01, 0.01, 0.01], frameon=False)
        ax8.axes.get_yaxis().set_visible(False)
        ax8.axes.get_xaxis().set_visible(False)
        ax8.text(0.5, 0.5, 'Save', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax8.transAxes,
                 picker=True)

        # ============ Pick Event =================
        def onpick(event):  # pick coordinate to plot TOF-PH plot
            artist = event.artist
            id = artist.get_text()
            if isinstance(artist, Text):
                if id == 'Store':
                    # ===== Storing datadict file =======
                    if mtof1 != 0:
                        ybin_bound0 = ybin_bounds[0].astype(float)
                        print(ybin_bound0)
                        tofdict1 = {'dir': dir, 'mean_x': float(mean_x1), 'mean_r': float(mean_r1),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound0),
                                    'mean_tof': float(mtof1), 'hwhm': float(hwhmtof1), 'mean_cnt': float(mcnt1)}
                        self.store_meanHtof(tofdict1)
                    if mtof2 !=0:
                        ybin_bound1 = ybin_bounds[1].astype(float)
                        tofdict2 = {'dir': dir, 'mean_x': float(mean_x2), 'mean_r': float(mean_r2),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound1),
                                    'mean_tof': float(mtof2), 'hwhm': float(hwhmtof2), 'mean_cnt': float(mcnt2)}
                        self.store_meanHtof(tofdict2)
                    if mtof3 != 0:
                        ybin_bounds2 = ybin_bounds[2].astype(float)
                        tofdict3 = {'dir': dir, 'mean_x': float(mean_x3), 'mean_r': float(mean_r3),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds2),
                                    'mean_tof': float(mtof3), 'hwhm': float(hwhmtof3), 'mean_cnt': float(mcnt3)}
                        self.store_meanHtof(tofdict3)
                    if mtof4 != 0:
                        ybin_bound3 = ybin_bounds[3].astype(float)
                        tofdict4 = {'dir': dir, 'mean_x': float(mean_x4), 'mean_r': float(mean_r4),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound3),
                                    'mean_tof': float(mtof4), 'hwhm': float(hwhmtof4), 'mean_cnt': float(mcnt4)}
                        self.store_meanHtof(tofdict4)
                    if mtof5 != 0:
                        ybin_bound4 = ybin_bounds[4].astype(float)
                        tofdict5 = {'dir': dir, 'mean_x': float(mean_x5), 'mean_r': float(mean_r5),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound4),
                                    'mean_tof': float(mtof5), 'hwhm': float(hwhmtof5), 'mean_cnt':float(mcnt5)}
                        self.store_meanHtof(tofdict5)
                    if mtof6 != 0:
                        ybin_bound5 = ybin_bounds[5].astype(float)
                        tofdict6 = {'dir': dir, 'mean_x': float(mean_x6), 'mean_r': float(mean_r6),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound5),
                                    'mean_tof': float(mtof6), 'hwhm': float(hwhmtof6), 'mean_cnt': float(mcnt6)}
                        self.store_meanHtof(tofdict6)
                    if mtof7 != 0:
                        ybin_bound6 = ybin_bounds[6].astype(float)
                        tofdict7 = {'dir': dir, 'mean_x': float(mean_x7), 'mean_r': float(mean_r7),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound6),
                                    'mean_tof': float(mtof7), 'hwhm': float(hwhmtof7), 'mean_cnt': float(mcnt7)}
                        self.store_meanHtof(tofdict7)
                    if mtof8 != 0:
                        ybin_bound7 = ybin_bounds[7].astype(float)
                        tofdict8 = {'dir': dir, 'mean_x': float(mean_x8), 'mean_r': float(mean_r8),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound7),
                                    'mean_tof': float(mtof8), 'hwhm': float(hwhmtof8), 'mean_cnt':float(mcnt8)}
                        self.store_meanHtof(tofdict8)
                    if mtof9 != 0:
                        ybin_bound8 = ybin_bounds[8].astype(float)
                        tofdict9 = {'dir': dir, 'mean_x': float(mean_x9), 'mean_r': float(mean_r9),
                                    'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound8),
                                    'mean_tof': float(mtof9), 'hwhm': float(hwhmtof9), 'mean_cnt':float(mcnt9)}
                        self.store_meanHtof(tofdict9)
                    if mtof10 != 0:
                        ybin_bound9 = ybin_bounds[9].astype(float)
                        tofdict10 = {'dir': dir, 'mean_x': float(mean_x10), 'mean_r': float(mean_r10),
                                     'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bound9),
                                     'mean_tof': float(mtof10), 'hwhm': float(hwhmtof10), 'mean_cnt': float(mcnt10)}
                        self.store_meanHtof(tofdict10)
                elif id == 'Save':
                    plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/npd2/tofspec/' + sensor + '_'
                                + '10binavgtof_' + str(xbin_bounds[0][0]) + str(xbin_bounds[0][1]) + str(ybin_bounds[0][0])
                                + str(ybin_bounds[-1][1]) + dir + '.png')
                    print('saving image...')


        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()

    def dt_norm(self, ax, tof, ph, tof_bin, dtlist):

        # Enable Minor Ticks
        minorlocator = MultipleLocator(100)
        majorlocator = MultipleLocator(500)

        # ========= For hwhm ============
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx


        '''def find_nearest(tof_arr, half_max, mean_tof):
            less_tof = tof_arr[tof_arr < mean_tof]
            less_idx = (np.abs(less_tof - half_max)).argmin()

            if tof_arr[less_idx] < 100: # disregard if lower tof half max
                less_idx = tof_arr[min(tof_arr)]

            gr_tof = tof_arr[tof_arr > mean_tof]
            gr_idx = (np.abs(gr_tof - half_max)).argmin()

            if tof_arr[gr_idx] > 600:
                gr_idx = tof_arr[max(tof_arr)]

            return less_idx, gr_idx'''

        npd2_Htof = self.get_npd2_Htof  # tof range boundaries 80 - 600 ns

        ylimax = 0.

        if np.sum(dtlist) == 0:
            normf = 0.
        else:
            normf = float(1)/np.sum(dtlist)

        tof_hist, tofbins = np.histogram(tof, bins=tof_bin)

        #m = tofbins[:-1].size
        #s = m - np.sqrt(2*m)        # smoothing function

        xx = np.linspace(1., 2048, 512)

        # normalize with duty time
        norm_tof = tof_hist*normf/float(4.0)        # 1 bin/4 points

        # ====== Apply Noise subtraction =========
        norm_tof = self.noise_subtract(norm_tof, tofbins)

        ax.plot(tofbins[:-1], norm_tof, '-', color='k', label='data')

        ax.xaxis.set_minor_locator(minorlocator)
        ax.xaxis.set_major_locator(majorlocator)
        ax.set_xlim([1, 2048])

        ph_hist, phbins = np.histogram(ph, bins=tof_bin)
        norm_ph = ph_hist*normf

        # fit data
        spline = inter.UnivariateSpline(tofbins[:-1], norm_tof, s=0)

        # plot mean, fwhm
        lineplot = ax.plot(xx, spline(xx), 'r-', label='line fit')
        xvalues = lineplot[0].get_xdata()
        yvalues = lineplot[0].get_ydata()

        ax.set_ylim([0, max(yvalues) + 0.01])

        # ========= slice using tof range ========
        ma_x = np.ma.masked_outside(xvalues, npd2_Htof[0], npd2_Htof[1])
        ma_y = np.ma.masked_where(ma_x.mask, yvalues)
        subx = np.ma.compressed(ma_x)
        suby = np.ma.compressed(ma_y)

        idx = np.where(suby == max(suby))   # find index of max value

        mean_tof = subx[idx]    # tof of max value
        mean_cnt = suby[idx]

        r1 = 0.

        if mean_tof.size > 1:

            mean_tof, mtof_idx = find_nearest(mean_tof, np.mean(mean_tof))
            mean_cnt = mean_cnt[mtof_idx]

        if mean_cnt > ylimax:
                ylimax = mean_cnt
                print(ylimax)

        if 100.0 < mean_tof <= 600.0:

            try:
                ax.plot(mean_tof, mean_cnt, 'o', color='b', label='max')
                halfmax, half_idx = find_nearest(suby, mean_cnt / float(2.))
                r1 += subx[half_idx]
                r2 = -r1
                ax.axvspan(r1, mean_tof, facecolor='g', alpha=0.5, label='hwhm')

                '''less_idx, gtr_idx = find_nearest(suby, mean_cnt / float(2.), mean_cnt)
                r1 = subx[less_idx]
                r2 = subx[gtr_idx]
                ax.axvspan(r1, r2, facecolor='g', alpha=0.1, label='fwhm')'''

                #ax.axvspan(r1, r2, facecolor='g', alpha=0.5, label='fwhm')
                #ax.set_yscale('log')
                #ax.set_title('Normalized to Duty Time: ' + str(sumdt), fontsize=9)
                ax.set_ylim([0, ylimax + (ylimax*0.1)])
                ax.set_title('Normalized with Duty Time: ', fontsize=9)
                plt.legend(prop={'size': 8})
            except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                print('Error: ', err)
                mean_tof, mean_cnt, r1 = 0., 0., 0.

        return mean_tof, mean_cnt, r1

    def noise_subtract(self, tof_hist, tofbins):

        ma_bins = np.ma.masked_where(tofbins[:-1] < 700, tofbins[:-1])
        ma_tofhist = np.ma.masked_where(ma_bins.mask, tof_hist)
        tofhist = np.ma.compressed(ma_tofhist)
        const = np.mean(tofhist)

        #print(const)

        calib_tofhist = tof_hist - const

        return calib_tofhist

    def gauss_fit(self, toflist, dtlist, tof_bin, xbin_bounds, ybin_bounds, sensor, dir,):

        tof_bin = 800
        max_ylim = 0.
        stacklist = []
        xstack = []

        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx

        #npd2_Htof = self.get_npd2_Htof  # tof range boundaries 80 - 600 ns

        fig = plt.figure(figsize=(12, 9))
        ax1 = fig.add_subplot(111)
        ax1.set_xlim([0, 700])
        #ax1.set_yscale('log')
        ax1.set_xlabel('TOF [ns]', fontsize = 9)
        ax1.set_ylabel(r'$cnts \ s^{-1}ns^{-1}$', multialignment='center', fontsize=9)
        ax1.set_title('H ENA Energy Spectra (coin 0: 1 Start, 1 Stop), x: ' + str(xbin_bounds[0][0]) + '-' + str(
                xbin_bounds[0][1]) + '\n' + sensor.upper() + ', dir: ' + dir)

        for i in range(len(toflist)):

            # ===== normalize each list with duty time =====
            if np.sum(dtlist[i]) == 0:
                normf = 0.
            else:
                normf = float(1)/np.sum(dtlist[i])

            tof_hist, tofbins = np.histogram(toflist[i], bins=tof_bin)
            bin_centers = (tofbins[:-1] + tofbins[1:])/float(2)

            norm_tof = tof_hist*normf

            # === slice tof 100 - 700 ns ====

            #ma_tofbins = np.ma.masked_outside(bin_centers, 100., 700.)

            ma_tofbins = np.ma.masked_where(bin_centers > 700., bin_centers)
            ma_tof = np.ma.masked_where(ma_tofbins.mask, norm_tof)
            subtofbins = np.ma.compressed(ma_tofbins)
            subtof = np.ma.compressed(ma_tof)

            max_idx = np.where(subtof == max(subtof))   # find index of max value
            max_cnt = subtof[max_idx]
            max_tofbin = subtofbins[max_idx]

            # rebinning
            hist2, tofbins2 = np.histogram(subtof, bins= 200)
            bin_centers = (tofbins2[:-1] + tofbins2[1:])/2

            if max_cnt[0] > max_ylim: # max ylim determination
                max_ylim = max_cnt[0]

            print(hist2.shape, bin_centers.shape)

            stacklist.append(hist2.tolist())
            xstack.append(bin_centers.tolist())
            '''
            ax1.plot(subtofbins, subtof, '-', color='k', label=str(ybin_bounds[i][0]) + '-' + str(ybin_bounds[i][1]))
            ax1.set_ylim([0, max(subtof)])

            # ======== Gaussian fit ========
            #mean = sum(subtofbins*subtof)/len(subtofbins)     # note this correction
            #sigma = sum(subtof*(subtofbins - mean)**2)/len(subtofbins)  # note this correction

            def gaussian(x, amp, cen, wid):

                return (amp/np.sqrt(2*np.pi)*wid) * np.exp(-(x-cen)**2/(2.*wid**2))

            # popt, pcov = curve_fit(gaus,subtofbins,subtof, p0=[1, 0., 1])
            #lineplot = ax1.plot(subtofbins, gaus(subtofbins, *popt), 'r-')

            gmod = Model(gaussian)
            halfmax, half_idx = find_nearest(subtof, max_cnt / float(2.))
            wid = subtofbins[half_idx]
            result = gmod.fit(subtof, x = subtofbins, amp = max_cnt[0], cen= max_tofbin[0], wid = wid)

            lineplot = ax1.plot(subtofbins, result.best_fit, 'r-')

            xvalues = lineplot[0].get_xdata()
            yvalues = lineplot[0].get_ydata()

            idx = np.where(yvalues == max(yvalues))   # find index of max value

            mean_tof = xvalues[idx]    # tof of max value
            mean_cnt = yvalues[idx]

            if mean_tof.size > 1:
                mean_tof, mtof_idx = find_nearest(mean_tof, np.mean(mean_tof))
                mean_cnt = mean_cnt[mtof_idx]

            if 100.0 < mean_tof <= 600.0:
                try:
                    ax1.plot(mean_tof, mean_cnt, 'o', color='b', label='max')
                    halfmax, half_idx = find_nearest(yvalues, mean_cnt / float(2.))
                    r1 = xvalues[half_idx]
                    r2 = -r1

                    #ax2 = ax1.twiny()

                except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                    print('Error: ', err)'''

        ax1.stackplot(xstack[0], stacklist, 'k-')
        ax1.set_ylim([0, max_ylim])
        ax8 = fig.add_axes([0.65, 0.01, 0.01, 0.01], frameon=False)
        ax8.axes.get_yaxis().set_visible(False)
        ax8.axes.get_xaxis().set_visible(False)
        ax8.text(0.5, 0.5, 'Save', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax8.transAxes,
                 picker=True)

        # ============ Pick Event =================
        def onpick(event):  # pick coordinate to plot TOF-PH plot
            artist = event.artist
            id = artist.get_text()
            if isinstance(artist, Text):
                if id == 'Save':
                    plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/npd2/tofspec/subsolar' + sensor + '_'
                                + '10binrowstack_' + str(xbin_bounds[0][0]) + str(xbin_bounds[0][1]) + str(ybin_bounds[0][0])
                                + str(ybin_bounds[-1][1]) + dir + '.png')
                    print('saving image...')

        fig.canvas.mpl_connect('pick_event', onpick)

        plt.legend(prop={'size': 8})
        plt.show()

    def store_meanHtof(self, datadict):

        file = self.f_name[0:7] + '_meanHtof'

        dirpath = self.f_path[:- len(self.f_name)]
        # print('dirpath: ', dirpath)

        '''# Check key values
        for key in dformat:
            print('Check keys: ', key)'''

        # Store files
        if os.path.isfile(dirpath + file + '.json'):  # if file exists,
            print('File exists!')
            found = False
            with open(dirpath + file + '.json', 'r') as f1:
                for line in f1:  # reading file per line
                    linedict = json.loads(line)  # data in dict

                    if linedict['dir'] == datadict['dir'] and linedict['xbounds'] == datadict['xbounds'] and linedict['ybounds'] == datadict['ybounds']:
                        print('line dictionary already exists..')
                        found = True

            if found == False:
                with open(dirpath + file + '.json', 'a') as f:
                    json.dump(datadict, f, separators=(',', ':'), sort_keys=True)  # append each dict to the file
                    print('appending...')
                    f.write(os.linesep)  # 1 dict per line in the file

        else:
            with open(dirpath + file + '.json', 'w') as f:
                json.dump(datadict, f, separators=(',', ':'), sort_keys=True)
                f.write(os.linesep)

    def binned_data(self, datadict, xbounds, ybounds, dir):  # find data values in chosen bin

        # Define Variables
        dir_rho, dir_x, dir_tof, dir_ph, dir_dc = [], [], [], [], []
        dt = 0.
        #rec_count = 0.

        # Create tiles
        tof_tile, ph_tile, x_tile, r_tile = [], [], [], []

        if dir == 'dir0':
            # list to array conversion
            dir_ph = np.array(datadict['dir0_ph'])
            dir_tof = np.array(datadict['dir0_tof'])
            dir_x = np.array(datadict['dir0_x'])
            dir_rho = np.array(datadict['dir0_rho'])
            
        elif dir == 'dir1':
            # list to array conversion
            dir_ph = np.array(datadict['dir1_ph'])
            dir_tof = np.array(datadict['dir1_tof'])
            dir_x = np.array(datadict['dir1_x'])
            dir_rho = np.array(datadict['dir1_rho'])
            
        elif dir == 'dir2':
            # list to array conversion
            dir_ph = np.array(datadict['dir2_ph'])
            dir_tof = np.array(datadict['dir2_tof'])
            dir_x = np.array(datadict['dir2_x'])
            dir_rho = np.array(datadict['dir2_rho'])

        # ============== Masking ===============

        # 1st Dimension masking
        ma_x = np.ma.masked_outside(dir_x, xbounds[0], xbounds[1])
        # use ma_x as mask
        ma_r = np.ma.masked_where(ma_x.mask, dir_rho)
        ma_tof = np.ma.masked_where(ma_x.mask, dir_tof)
        ma_ph = np.ma.masked_where(ma_x.mask, dir_ph)
        
        # Compress
        x = np.ma.compressed(ma_x)
        r = np.ma.compressed(ma_r)
        tof = np.ma.compressed(ma_tof)
        ph = np.ma.compressed(ma_ph)

        if x.size:  # if not empty
            try:
                # Checking
                '''print('x bounds: ', x[0], x[-1])
                print('xbounds: ', xbounds[0], xbounds[1])'''

                # 2nd Dimension masking
                ma_r = np.ma.masked_outside(r, ybounds[0], ybounds[1])
                # use ma_r as mask
                ma_x = np.ma.masked_where(ma_r.mask, x)
                ma_tof = np.ma.masked_where(ma_r.mask, tof)
                ma_ph = np.ma.masked_where(ma_r.mask, ph)

                # Compress
                r = np.ma.compressed(ma_r)
                x = np.ma.compressed(ma_x)
                tof = np.ma.compressed(ma_tof)
                ph = np.ma.compressed(ma_ph)

                if x.size:  # if not empty
                    x_tile.append(np.mean(x))  # for plotting purpose
                    r_tile.append(np.mean(r))
                    tof_tile.extend(tof)
                    ph_tile.extend(ph)

                    # Bin the duty cycle and coordinates to find sum of dc
                    dcx, dcr, dcsum = self.dc_binning(datadict, xbounds[0], xbounds[1], ybounds[0], ybounds[1])
                    dt += dcsum

                    # record counting
                    '''for i in range(x.size):
                        if i != 0:
                            if (x[i] != x[i-1]) and (r[i] != r[i - 1]):  # if coordinates not identical
                                rec_count += 1
                    else:
                        rec_count += 1'''
                    #arrcount = np.unique(x)
                    #rec_count += arrcount.size

            except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                print('Error: ', err)

            # checking
            '''print('tof_tiles size: ', len(tof_tile))
            print('ph_tiles size: ', len(ph_tile))'''

        return x_tile, r_tile, tof_tile, ph_tile, dt

    def tdc_binning(self, dcx, dcr, dc):

        ma_dc = np.ma.masked_where(dc == 0.0, dc)
        ma_dcx = np.ma.masked_where(ma_dc.mask, dcx)
        ma_dcr = np.ma.masked_where(ma_dc.mask, dcr)

        subdc = np.ma.compressed(ma_dc)
        subdcx = np.ma.compressed(ma_dcx)
        subdcr = np.ma.compressed(ma_dcr)

        return subdcx, subdcr, subdc

    def dc_binning(self, datadict, x1, x2, r1, r2):

        # ========= Masking =======
        ma_dcx = np.ma.masked_outside(datadict['dcx'], x1, x2)
        ma_dcr = np.ma.masked_where(ma_dcx.mask, datadict['dcr'])
        ma_dc = np.ma.masked_where(ma_dcx.mask, datadict['dclist'])

        # ======= Compress ========
        dcx = np.ma.compressed(ma_dcx)
        dcr = np.ma.compressed(ma_dcr)
        dc = np.ma.compressed(ma_dc)

        if dc.size:
            # ==== second dimension masking ======
            ma_dcr = np.ma.masked_outside(dcr, r1, r2)
            ma_dcx = np.ma.masked_where(ma_dcr.mask, dcx)
            ma_dc = np.ma.masked_where(ma_dcr.mask, dc)

            dcr = np.ma.compressed(ma_dcr)
            dcx = np.ma.compressed(ma_dcx)
            dc = np.ma.compressed(ma_dc)

            if dc.size:
                dcsum = np.sum(dc)
            else:
                dcsum = 0

            return  dcx, dcr, dcsum

    def slice_tofspec(self, tof, bins, tof_range):

        # ===== Masking =====
        ma_bins = np.ma.masked_outside(bins[:-1], tof_range[0], tof_range[1])
        ma_tof = np.ma.masked_where(ma_bins.mask, tof)

        new_bins = np.ma.compressed(ma_bins)
        new_tof = np.ma.compressed(ma_tof)
        
        return new_tof, new_bins

    def tof_filter(self, tof, x, r):    # limits valid events to tof only 100-600 ns

        # range limits
        tof_min = 100.
        tof_max = 600.
        tof = np.array(tof)

        ma_tof = np.ma.masked_outside(tof, tof_min, tof_max)
        ma_x = np.ma.masked_where(ma_tof.mask, x)
        ma_r = np.ma.masked_where(ma_tof.mask, r)

        new_tof = np.ma.compressed(ma_tof)
        new_x = np.ma.compressed(ma_x)
        new_r = np.ma.compressed(ma_r)

        return new_x, new_r

    def subplot_rowstack(self, toflist, dtlist, tof_bin, xbin_bounds, ybin_bounds, sensor, dir):

        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx], idx

        '''def find_nearest(tof_arr, half_max, mean_tof):
            less_tof = tof_arr[tof_arr < mean_tof]
            less_idx = (np.abs(less_tof - half_max)).argmin()

            if tof_arr[less_idx] < 100: # disregard if lower tof half max
                less_idx = tof_arr[min(tof_arr)]

            gr_tof = tof_arr[tof_arr > mean_tof]
            gr_idx = (np.abs(gr_tof - half_max)).argmin()

            if tof_arr[gr_idx] > 600:
                gr_idx = tof_arr[max(tof_arr)]

            return less_idx, gr_idx'''

        npd2_Htof = self.get_npd2_Htof  # tof range boundaries 80 - 600 ns

        ylimax = 0.

        # ================ Plot Setup ==============

        # Enable Minor Ticks
        xminorlocator = MultipleLocator(100)
        xmajorlocator = MultipleLocator(500)

        fig, axarr = plt.subplots(nrows=5, ncols=2, figsize=(15, 8))
        fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.1, hspace=0.05, wspace= 0.30)
        fig.suptitle(
            'TOF Spectra of ENA (coin 0: 1 Start, 1 Stop), x: ' + str(xbin_bounds[0][0]) + '-' + str(
                xbin_bounds[0][1]) + r'$R_V$' + ', r: ' + str(ybin_bounds[0][0]) +
            '-' + str(
                ybin_bounds[-1][1]) + r'$R_V$' + '\n' + 'VSO, ' + sensor.upper() + ', Ch: ' + dir + ', Date: 2005-2013',
            fontsize=11, y=0.98)

        for j in range(2):
            for i in range(4):
                axarr[i, j].xaxis.set_minor_locator(xminorlocator)
                axarr[i, j].xaxis.set_major_locator(xmajorlocator)

                #axarr[i, j].set_xlim([1, 700])
                #axarr[i, j].axes.xaxis.set_ticklabels([])
                axarr[i, j].set_ylabel(r'$cnt \ s^{-1}ns^{-1}$', multialignment='center')
                [tick.label.set_fontsize(9) for tick in axarr[i, j].yaxis.get_major_ticks()]
                # axarr[0, j].set_title('eV', fontsize=9)
                axarr[4, j].set_ylabel(r'$cnt \ s^{-1}ns^{-1}$', multialignment='center')
                axarr[4, j].set_xlabel('TOF [ns]')

        tof_ticks = np.array([1., 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0])
        e_ticks = [round(self.tof_toeV(tof), 1) for tof in tof_ticks]
        ax1 = axarr[0, 0].twiny()
        ax2 = axarr[0, 1].twiny()
        ax1.set_xlim(axarr[0, 0].get_xlim())
        ax2.set_xlim(axarr[0, 1].get_xlim())
        ax1.set_xlabel('eV')
        ax2.set_xlabel('eV')
        ax1.set_xticks(tof_ticks)
        ax2.set_xticks(tof_ticks)
        ax1.axes.xaxis.set_ticklabels(e_ticks, fontsize = 9)
        ax2.axes.xaxis.set_ticklabels(e_ticks, fontsize = 9)

        row = 4
        col = 0
        for i in range(len(toflist)):
            if row < 0:
                col += 1
                row = 4

            if np.sum(dtlist[i]) == 0:
                normf = 0.
            else:
                normf = float(1)/np.sum(dtlist[i])

            xx = np.linspace(1., 2048, 512)
            #xx = np.linspace(1., 2048, 256)

            tof_hist, tofbins = np.histogram(toflist[i], bins=tof_bin)
            #tof_hist, tofbins = np.histogram(toflist[i], bins=256)

            # normalize with duty time
            norm_tof = tof_hist*normf/float(4.0)    # 1 bin/4 points
            #norm_tof = tof_hist*normf/float(8.0)    # 1 bin/8 points

            # =========== noise reduction ===========
            norm_tof = self.noise_subtract(norm_tof, tofbins)

            spline = inter.UnivariateSpline(tofbins[:-1], norm_tof, s=0)

            axarr[row, col].plot(tofbins[:-1], norm_tof, '-', color='k', label='data')
            axarr[row, col].axvline(x = 224.2, color = 'k', linestyle='--')
            ax3 = axarr[row, col].twinx()
            ax3.set_ylabel('r: '+ str(ybin_bounds[i][0]) + '-' + str(ybin_bounds[i][1]), rotation = 'vertical')
            ax3.axes.yaxis.set_ticklabels([])

            # plot mean, fwhm
            lineplot = axarr[row, col].plot(xx, spline(xx), 'r-', label='line fit')
            xvalues = lineplot[0].get_xdata()
            yvalues = lineplot[0].get_ydata()

            # ========= slice using tof range ========
            ma_x = np.ma.masked_outside(xvalues, npd2_Htof[0], npd2_Htof[1])
            ma_y = np.ma.masked_where(ma_x.mask, yvalues)
            subx = np.ma.compressed(ma_x)
            suby = np.ma.compressed(ma_y)

            idx = np.where(suby == max(suby))   # find index of max value

            mean_tof = subx[idx]    # tof of max value
            mean_cnt = suby[idx]

            if mean_tof.size > 1:
                mean_tof, mtof_idx = find_nearest(mean_tof, np.mean(mean_tof))
                mean_cnt = mean_cnt[mtof_idx]

            if mean_cnt > ylimax:
                ylimax = mean_cnt
                print(ylimax)

            if 100.0 < mean_tof <= 600.0:
                try:
                    axarr[row, col].plot(mean_tof, mean_cnt, 'o', color='b', label='max')
                    half_max, idx = find_nearest(suby, mean_cnt / float(2.))
                    r1 = subx[idx]
                    r2 = subx[idx]
                    axarr[row,col].axvspan(r1, mean_tof, facecolor='g', alpha=0.5, label='hwhm')

                    '''less_idx, gtr_idx = find_nearest(suby, mean_cnt / float(2.), mean_cnt)
                    r1 = subx[less_idx]
                    r2 = subx[gtr_idx]
                    axarr[row,col].axvspan(r1, r2, facecolor='g', alpha=0.5, label='fwhm')'''
                    #axarr[row,col].set_xlim([1, 700])
                    #axarr[row,col].set_yscale('log')

                    #axarr[row,col].text(100.0, 7.0*10** -2, 'No. of Records: ' + str(dclist[i]), horizontalalignment='center',
                    #  verticalalignment='top', transform=axarr[row,col].transData, fontsize = 8)

                except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                    print('Error: ', err)
            row -= 1
            plt.legend(prop={'size': 8})

        print('ylim: ', ylimax)
        try:
            ymax_majorlocator = MultipleLocator(np.round(ylimax/4.0, 1))
            ymajorlocator = MultipleLocator(np.round(0.02/4.0, 2))
        except ValueError as err:
            print(err, ': Setting tick to default formatter')
            ymajorlocator = AutoLocator()
            ymax_majorlocator = AutoLocator()

        for j in range(2):
            for i in range(4):
                axarr[i, j].set_ylim([0, 0.02])
                #axarr[i, j].yaxis.set_major_locator(ymajorlocator)
                #axarr[4, j].yaxis.set_major_locator(ymajorlocator)
                axarr[i, j].set_xlim([1, 700])
        #axarr[4, 0].set_ylim([0, 0.02])
        axarr[4, 0].yaxis.set_major_locator(ymajorlocator)
        axarr[4, 0].set_ylim([0, ylimax])
        #axarr[3, 0].yaxis.set_major_locator(ymax_majorlocator)
        #axarr[0, 1].yaxis.set_major_locator(ymax_majorlocator)
        #axarr[3, 0].set_ylim([0, ylimax])
        #axarr[0, 1].set_ylim([0, ylimax])
        axarr[4, 1].set_ylim([0, 0.02])
        axarr[4, 0].set_xlim([1, 700])
        axarr[4, 1].set_xlim([1, 700])
        [tick.label.set_fontsize(9) for tick in axarr[4, 0].yaxis.get_major_ticks()]
        [tick.label.set_fontsize(9) for tick in axarr[4, 1].yaxis.get_major_ticks()]

        ax8 = fig.add_axes([0.65, 0.01, 0.01, 0.01], frameon=False)
        ax8.axes.get_yaxis().set_visible(False)
        ax8.axes.get_xaxis().set_visible(False)
        ax8.text(0.5, 0.5, 'Save', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax8.transAxes,
                 picker=True)

        # ============ Pick Event =================
        def onpick(event):  # pick coordinate to plot TOF-PH plot
            artist = event.artist
            id = artist.get_text()
            if isinstance(artist, Text):
                if id == 'Save':
                    plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_'
                                + '10binrowstack2_' + str(xbin_bounds[0][0]) + str(xbin_bounds[0][1]) + str(ybin_bounds[0][0])
                                + str(ybin_bounds[-1][1]) + dir + '.png')
                    print('saving image...')

        fig.canvas.mpl_connect('pick_event', onpick)

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
        #subplot.plot(x_tsd, y_tsd, 'k--', label = 'Terminator Shock')

    def plt_orbit(self, subplot, date, Rp):

        print('Generating orbit params...')

        tuplerange = []
        st = []
        et = []
        dt = []

        # Find start and end dates from file
        with open(self.f_path, 'r') as f:
            for line in f:
                datadict = json.loads(line)
                if len(datadict['dir0_ph']) == 0:
                    continue
                else:
                    if datadict['Date'] == date:
                        st.append(datadict['Start'])
                        et.append(datadict['End'])
                        dt.append(datadict['dt'])
                        '''
                        print(st)
                        print(et)
                        '''

        # Generate set of orbit params depending on number of data and duration of observation
        str_dformat = '%Y-%m-%d'
        str_tformat = '%H:%M:%S'
        tdiv = datetime.timedelta(minutes=10)  # time division

        for i in range(len(st)):
            st_dtime = datetime.datetime.strptime(st[i], str_tformat)  # start time
            # print('start time: ', st_dtime)

            et_dtime = st_dtime + datetime.timedelta(seconds=int(dt[i]))  # end time

            st_date = datetime.datetime.strptime(date, str_dformat)  # start date
            if et_dtime.hour < st_dtime.hour:  # if next day
                next_day = st_date + datetime.timedelta(days=1)
                et_date = time.strftime('%Y-%m-%d', next_day.timetuple())  # date of next day
                et_tuple = datetime.datetime.strptime(et_date + ' ' + et[i],
                                                      str_dformat + ' ' + str_tformat)  # date and time tuple
                # print('end tuple: ', et_tuple)

            else:  # otherwise, use input date
                et_tuple = datetime.datetime.strptime(date + ' ' + et[i], str_dformat + ' ' + str_tformat)
                # print('end tuple: ', et_tuple)

            # Create start Date and Time tuple
            st_tuple = datetime.datetime.strptime(date + ' ' + st[i], str_dformat + ' ' + str_tformat)
            # print('start tuple: ', st_tuple)

            # Increment by tdiv = 10 mins
            x_tuple = st_tuple + tdiv
            tuplerange.append(x_tuple)

            while x_tuple < et_tuple:
                x_tuple += tdiv  # keep incrementing
                tuplerange.append(x_tuple)
                # print(x_tuple, et_tuple)
        '''print(tuplerange)
        print(st)
        print(et)'''

        # generate orbit data
        xlist, rholist = [], []

        for i in range(len(tuplerange)):
            coord = vspice.get_position(tuplerange[i])
            rho = self.rho(coord)
            xlist.append(coord[0] / float(Rp))
            rholist.append(rho / float(Rp))

        # Plot orbit
        subplot.plot(xlist, rholist, 'r+')

    def tof_cover(self, dir, savefig = False):

        sensor = self.f_name[0:4]
        x, r, tof, cnt, fwhm = [], [], [], [], []

        # Read file containing list
        with open(self.f_path, 'r') as f:

            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict

                if not datadict:  # if file line is empty
                    print('Empty line.')
                    continue
                else:
                    if dir == datadict['dir']:

                        px = datadict['xbounds'][0] + (datadict['xbounds'][1] - datadict['xbounds'][0])/float(2)
                        pr = datadict['ybounds'][0] + (datadict['ybounds'][1] - datadict['ybounds'][0])/float(2)
                        x.append(px)
                        r.append(pr)
                        fwhm.append(datadict['hwhm'])
                        #fwhm.append(datadict['fwhm'])
                        tof.append(datadict['mean_tof'])
                        cnt.append(datadict['mean_cnt'])

        fwhm = np.array(fwhm)
        cnt = np.array(cnt)
        tof = np.array(tof)

        ma_cnt = np.ma.masked_where(cnt == 0., cnt)
        ma_fwhm = np.ma.masked_where(ma_cnt.mask, fwhm)
        ma_tof = np.ma.masked_where(ma_cnt.mask, tof)
        ma_r = np.ma.masked_where(ma_cnt.mask, r)
        ma_x = np.ma.masked_where(ma_cnt.mask, x)

        ma_fwhm = np.ma.masked_where(ma_fwhm == 0., ma_fwhm)
        ma_cnt = np.ma.masked_where(ma_fwhm.mask, ma_cnt)
        ma_tof = np.ma.masked_where(ma_fwhm.mask, ma_tof)
        ma_r = np.ma.masked_where(ma_fwhm.mask, ma_r)
        ma_x = np.ma.masked_where(ma_fwhm.mask, ma_x)

        fwhm = np.ma.compressed(ma_fwhm)
        cnt = np.ma.compressed(ma_cnt)
        tof = np.ma.compressed(ma_tof)
        r = np.ma.compressed(ma_r)
        x = np.ma.compressed(ma_x)

        # Enable Minor Ticks
        minorlocator = MultipleLocator(50)
        majorlocator = MultipleLocator(250)

        # ========Plot TOF PEAK ============
        plt.close('all')
        fig1 = plt.figure(figsize=(12, 9))
        ax1 = fig1.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax1.set_xlim([5, -8])
        ax1.set_ylim([0, 12.5])
        ax1.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax1.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)
        ax1.set_title('ENA Peak of TOF Distribution, binning: 0.25 x 0.25, ' + sensor.upper() + ': ' + dir + ', Date: 2005-2013', fontsize=12)


        cmap = plt.get_cmap('jet')

        '''# set colors for colorbars
        deV = 0.5*10**3
        eVbounds = np.arange(0.5*10**3, 5.5*10**3, deV) # 10 elements
        dtofbound = self.eVto_tof(eVbounds)
        dtofbound = np.append(dtofbound, [80])
        bounds = np.around(dtofbound[::-1], decimals=2)
        bounds = np.append(bounds, 600.0)
        #bounds = [80.0, 100.269, 112.10, 129.45, 158.54, 224.21, 317.1]
        cmap = colors.ListedColormap(['k','darkred','r','orange', 'y','lime', 'g', 'c', 'b', 'darkblue', 'blueviolet'])
        norm = colors.BoundaryNorm(bounds, cmap.N)

        im = ax1.scatter(x, r, c = tof, cmap = cmap, norm=norm, marker = 's', s = 102.0, lw = 0)'''

        cmap_r = plt.get_cmap('jet_r')

        im = ax1.scatter(x, r, c = tof, cmap = cmap_r, norm = colors.LogNorm(), marker = 's', s = 103.0, lw = 0, vmin = 90., vmax = 600.)

        cbar_ax = fig1.add_axes([0.87, 0.1, 0.017, 0.8])
        #cb = plt.colorbar(im, cax=cbar_ax, cmap = cmap, norm = norm, boundaries = bounds, ticks= bounds)

        tofticks = self.eVto_tof(np.array([100., 500., 1000., 1500., 2000., 2500., 3000., 3500., 4000., 4500., 5000., 5500.]))
        tofticks = np.round(tofticks, 1)
        cb = fig1.colorbar(im, cax = cbar_ax)
        cb.set_ticks(tofticks)
        cb.set_ticklabels(tofticks)
        #cb.set_label('Peak of TOF', labelpad=2., size=10)
        ax2 = cb.ax.twinx()
        ax2.set_ylim(cb.ax.get_ylim())
        ax2.set_yticks(cb.ax.get_yticks())
        #evlabels = ['5.0', '4.5', '4.0', '3.5', '3.0', '2.5', '2.0', '1.5', '1.0', '.5', '.1']
        evlabels = ['.1', '.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0', '5.5']

        ax2.set_yticklabels(evlabels)
        ax3 = cb.ax.twiny()
        ax3.axes.xaxis.set_ticklabels([])
        ax3.set_xlabel('[Peak of TOF]     [keV]', fontsize = 9)


        '''hist2, xedges2, yedges2 = np.histogram2d(x, r, weights= tof, bins=[bin_xrange, bin_yrange])



        im1 = ax1.imshow(np.rot90(hist2), interpolation='nearest', #vmin= 0., vmax=100,
                            norm=colors.LogNorm(vmin=1, vmax=10 ** 2),
                            extent=[xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]], picker=True)

        cb3 = fig2.colorbar(im3, cax=cax3)
        cb3.set_label('events/sec', labelpad=2.)'''

        self.plt_venus(ax1)
        self.plt_bsicb(ax1)

        if savefig:
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' + 'all_peaktof_' + dir + '.png')

        # ======= PLOT Count Rate ========
        fig2 = plt.figure(figsize=(12, 9))
        ax4 = fig2.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax4.set_xlim([5, -8])
        ax4.set_ylim([0, 12.5])
        ax4.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax4.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)
        ax4.set_title('ENA Maximum Count Rate Distribution, binning: 0.25 x 0.25, ' + sensor.upper() + ': ' + dir + ', Date: 2005-2013', fontsize=12)

        # set colors for colorbars

        #bounds = np.logspace(-3, 0, 12)
        #cmap = colors.ListedColormap(['violet', 'blueviolet', 'darkblue', 'b', 'c', 'g', 'lime', 'y', 'orange', 'r', 'darkred'])
        #norm = colors.BoundaryNorm(bounds, cmap.N)

        im2 = ax4.scatter(x, r, c = cnt, cmap = cmap, norm = colors.LogNorm(), marker = 's', s = 103.0, lw= 0, vmin = 0.001, vmax = 1. )

        div = make_axes_locatable(ax4)
        cax = div.append_axes("right", size="2%", pad=0.05)
        #cb2 = fig2.colorbar(im, cax=cax, cmap= cmap, norm = norm, boundaries =bounds, ticks =bounds)
        cb2 = fig2.colorbar(im2, cax=cax)
        cb2.set_label(r'$cnt \ s^{-1}ns^{-1}$', labelpad=2., size=12)

        self.plt_venus(ax4)
        self.plt_bsicb(ax4)

        if savefig:
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' + 'all_maxcountr_' + dir + '.png')

        # ======= PLOT HWHM =========
        fig3 = plt.figure(figsize=(12, 9))
        ax5 = fig3.add_subplot(111)
        plt.axes().set_aspect('equal')
        plt.grid()
        ax5.set_xlim([5, -8])
        ax5.set_ylim([0, 12.5])
        ax5.set_xlabel(r'$R_V = 6.051850 \cdot 10^3$', fontsize=11)
        ax5.set_ylabel(r'$r = \sqrt{y^2 + z^2}$', fontsize=11)
        ax5.set_title('ENA Energy (Half-Width at Half-Maximum), binning: 0.25 x 0.25, ' + sensor.upper() + ': ' + dir + ', Date: 2005-2013', fontsize=12)

        '''l_fwhm = np.ma.masked_where(fwhm > tof, fwhm)
        ma_x = np.ma.masked_where(l_fwhm.mask, x)
        ma_r = np.ma.masked_where(l_fwhm.mask, r)'''

        fwhm_e = self.tof_toeV(fwhm)
        #fwhm_e = self.tof_toeV(l_fwhm)

        etof = self.tof_toeV(tof)

        de = abs(etof - fwhm_e)

        # set colors for colorbars

        #debounds = np.linspace(min(de), 1000, 12) # 12 elements
        '''eVbound = self.tof_toeV(tofbounds)'''
        #new_bounds = np.around(debounds, decimals=2)
        #cmap = colors.ListedColormap(['k','darkred','r','orange', 'y','lime', 'g', 'c', 'b', 'darkblue', 'blueviolet'])
        #norm = colors.BoundaryNorm(new_bounds, cmap)

        im3 = ax5.scatter(x, r, c = de, cmap = cmap, norm = colors.LogNorm() , marker = 's', s = 103.0, lw = 0, vmin = 1.0, vmax = 10000)

        div = make_axes_locatable(ax5)
        cax = div.append_axes("right", size="2%", pad=0.05)
        #cb = fig3.colorbar(im, cax=cax, cmap= cmap, norm = norm, boundaries =new_bounds, ticks =bounds)
        cb3 = fig3.colorbar(im3, cax=cax)

        #cbar_ax = fig3.add_axes([0.87, 0.1, 0.017, 0.8])
        #cb = fig3.colorbar(im, cax=cbar_ax, cmap = cmap, norm = norm, boundaries = new_bounds, ticks= new_bounds)
        cb3.set_label(r'$\delta E$ [eV]', fontsize = 12)

        '''ax2 = cb.ax.twinx()
        ax2.set_ylim(cb.ax.get_ylim())
        ax2.set_yticks(cb.ax.get_yticks())
        evlabels = [round(eV, 1) for eV in eVbound]

        ax2.set_yticklabels(evlabels)
        ax3 = cb.ax.twiny()
        ax3.axes.xaxis.set_ticklabels([])
        ax3.set_xlabel('[Width of TOF]     [eV]', fontsize = 9)'''

        self.plt_venus(ax5)
        self.plt_bsicb(ax5)

        if savefig:
            plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' + 'all_fwhm_' + dir + '.png')

        plt.show()

    def vel_plt(self, xrange, savefig=False):

        # Define Variables
        sensor = self.f_name[0:4]
        tof0, tof1, tof2 = [], [], []
        fwhm0, fwhm1, fwhm2 = [], [], []
        r0, r1, r2 = [], [], []
        xbounds0, ybounds0 = [], []
        xbounds1, ybounds1 = [], []
        xbounds2, ybounds2 = [], []
        xrange = [float(i) for i in xrange]

        dir = ['dir0', 'dir1', 'dir2']

        # Read file containing list
        with open(self.f_path, 'r') as f:
            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict
                if not datadict:  # if file is empty
                    print('Empty line.')
                    continue
                else:

                    if datadict['dir'] == dir[0] and xrange == datadict['xbounds']:
                        tof0.append(datadict['mean_tof'])
                        fwhm0.append(datadict['fwhm'])
                        r0.append(datadict['mean_r'])
                        xbounds0.append(datadict['xbounds'])
                        ybounds0.append(datadict['ybounds'])
                    elif datadict['dir'] == dir[1] and xrange == datadict['xbounds']:
                        tof1.append(datadict['mean_tof'])
                        fwhm1.append(datadict['fwhm'])
                        r1.append(datadict['mean_r'])
                        xbounds1.append(datadict['xbounds'])
                        ybounds1.append(datadict['ybounds'])
                    elif datadict['dir'] == dir[2] and xrange == datadict['xbounds']:
                        tof2.append(datadict['mean_tof'])
                        fwhm2.append(datadict['fwhm'])
                        r2.append(datadict['mean_r'])
                        xbounds2.append(datadict['xbounds'])
                        ybounds2.append(datadict['ybounds'])

        print(tof0)
        # Compute Velocity
        v0 = [self.tofto_vel(tof) for tof in tof0]
        v1 = [self.tofto_vel(tof) for tof in tof1]
        v2 = [self.tofto_vel(tof) for tof in tof2]

        # fwhm tof to velocity
        vel_yerr0 = [self.tofto_vel(fwhm) for fwhm in fwhm0]
        vel_yerr1 = [self.tofto_vel(fwhm) for fwhm in fwhm1]
        vel_yerr2 = [self.tofto_vel(fwhm) for fwhm in fwhm2]

        yerr0 = np.divide(vel_yerr0, float(2.35))  # stdv for gaussian
        yerr1 = np.divide(vel_yerr1, float(2.35))
        yerr2 = np.divide(vel_yerr2, float(2.35))

        v_set = []
        v_set.extend(v0)
        v_set.extend(v1)
        v_set.extend(v2)
        # ======== Setup figure and Subplot ============
        fig = plt.figure(figsize=(15, 9))
        ax1 = fig.add_subplot(111)
        ax1.set_title(sensor.upper() + ' H ENA Velocity as Function of Venus Radius  at x range: ' + str(
            xbounds0[0][0]) + '-' + str(xbounds0[0][1]) + r'$\cdot R_v$', fontsize=12)
        ax1.set_xlim([np.amin(ybounds0), np.amax(ybounds0) + .5])
        ax1.set_xlabel('r' + r'$\cdot R_v = 6.051850 \cdot 10^3$')
        ax1.set_ylim([min(v_set) - 100., max(v_set) + 200.])
        ax1.set_ylabel('v(r) [km/s]')
        ax1.errorbar(r0, v0, yerr=yerr0, fmt='o', color='red', label='dir0')
        ax1.errorbar(r1, v1, yerr=yerr1, fmt='o', color='blue', label='dir1')
        ax1.errorbar(r2, v2, yerr=yerr2, fmt='o', color='c', label='dir2')
        # ax1.plot(r1, v1, '-', 'b', label = 'dir1')

        # ax1.plot(r2, v2, '-', 'c', label = 'dir2')
        plt.legend(prop={'size': 10})
        if savefig:
            plt.savefig(
                '/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_velplt_' + str(
                    xbounds0[0][0]) + str(xbounds0[0][1]) +
                str(ybounds0[0][0]) + str(ybounds0[-1][1]) + '.png')
        plt.show()

    def rad10_maxwellian(self, xedges, yedges, cursor, sensor, lisdate, dir, bins,
                         Rp):  # Generate 10 velocity plots radially

        # 10 radially binned TOF-PH plots generated from chosen cursor point.
        # Method, looks at bin edges, 1 increment until 5 max

        # ======= Setting data bounds from bins=========
        xdbin = bins[0] / float(2)
        ydbin = bins[1] / float(2)
        xcursor, ycursor = zip(*cursor)
        xc_bounds = [[x - xdbin, x + xdbin] for x in xcursor]
        yc_bounds = [[y - ydbin, y + ydbin] for y in ycursor]

        # Checking
        '''print('cursor: ', cursor)
        print('xc_bounds: ', xc_bounds)
        print('yc_bounds: ', yc_bounds)'''

        # Find nearest value from histogram edges
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return array[idx]

        xbin_bounds = [[find_nearest(xedges, xc_bounds[i][0]), find_nearest(xedges, xc_bounds[i][1])] for i in
                       range(len(xc_bounds))]
        ybin_bounds = [[find_nearest(yedges, yc_bounds[i][0]), find_nearest(yedges, yc_bounds[i][1])] for i in
                       range(len(yc_bounds))]

        # Increment bins radially
        idx0 = yedges[yedges == ybin_bounds[0][0]]
        idx1 = yedges[yedges == ybin_bounds[0][1]]
        ybin_bounds = [[yedges[yedges == idx0 + (i * bins[1])], yedges[yedges == idx1 + (i * bins[1])]] for i in
                       range(10)]

        # Reshape
        ybin_bounds = np.array(ybin_bounds).reshape(len(ybin_bounds), 2)

        '''print('nearest value: ',find_nearest(xedges, xc_bounds[0][0]), find_nearest(xedges, xc_bounds[0][1]))
        print('xedge: ', xedges)'''

        # Frame to data value conversion
        xbounds = np.multiply(xbin_bounds, Rp)
        ybounds = np.multiply(ybin_bounds, Rp)

        # Checking
        '''print('xbounds: ', xbounds, type(xbounds))
        print('ybounds: ', ybounds)'''

        # Define Variables
        axbins = [100, 25]

        xlist1, rlist1, toflist1, phlist1, dc1 = [], [], [], [], []
        xlist2, rlist2, toflist2, phlist2, dc2 = [], [], [], [], []
        xlist3, rlist3, toflist3, phlist3, dc3 = [], [], [], [], []
        xlist4, rlist4, toflist4, phlist4, dc4 = [], [], [], [], []
        xlist5, rlist5, toflist5, phlist5, dc5 = [], [], [], [], []
        rec_cnt1, rec_cnt2, rec_cnt3, rec_cnt4, rec_cnt5 = 0., 0., 0., 0., 0.

        xlist6, rlist6, toflist6, phlist6, dc6 = [], [], [], [], []
        xlist7, rlist7, toflist7, phlist7, dc7 = [], [], [], [], []
        xlist8, rlist8, toflist8, phlist8, dc8 = [], [], [], [], []
        xlist9, rlist9, toflist9, phlist9, dc9 = [], [], [], [], []
        xlist10, rlist10, toflist10, phlist10, dc10 = [], [], [], [], []
        rec_cnt6, rec_cnt7, rec_cnt8, rec_cnt9, rec_cnt10 = 0., 0., 0., 0., 0.

        # ========= Matching Data ==================
        # Read file containing list
        with open(self.f_path, 'r') as f:
            for line in f:  # reading file per line
                datadict = json.loads(line)  # data in dict
                if len(datadict[dir + '_ph']) == 0:  # if file is empty
                    print('Empty line.')
                    continue
                else:
                    if lisdate[0] == 'all':
                        # Call binned data values
                        x_tile1, r_tile1, tof_tile1, ph_tile1, records1, dc_tile1 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[0], dir)
                        if x_tile1:
                            xlist1.extend(x_tile1), rlist1.extend(r_tile1), toflist1.extend(tof_tile1), phlist1.extend(
                                ph_tile1), dc1.extend(dc_tile1)
                            rec_cnt1 += records1

                        x_tile2, r_tile2, tof_tile2, ph_tile2, records2, dc_tile2 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[1], dir)
                        if x_tile2:
                            xlist2.extend(x_tile2), rlist2.extend(r_tile2), toflist2.extend(tof_tile2), phlist2.extend(
                                ph_tile2), dc2.extend(dc_tile2)
                            rec_cnt2 += records2

                        x_tile3, r_tile3, tof_tile3, ph_tile3, records3, dc_tile3 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[2], dir)
                        if x_tile3:
                            xlist3.extend(x_tile3), rlist3.extend(r_tile3), toflist3.extend(tof_tile3), phlist3.extend(
                                ph_tile3), dc3.extend(dc_tile3)
                            rec_cnt3 += records3

                        x_tile4, r_tile4, tof_tile4, ph_tile4, records4, dc_tile4 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[3], dir)
                        if x_tile4:
                            xlist4.extend(x_tile4), rlist4.extend(r_tile4), toflist4.extend(tof_tile4), phlist4.extend(
                                ph_tile4), dc4.extend(dc_tile4)
                            rec_cnt4 += records4

                        x_tile5, r_tile5, tof_tile5, ph_tile5, records5, dc_tile5 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[4], dir)
                        if x_tile5:
                            xlist5.extend(x_tile5), rlist5.extend(r_tile5), toflist5.extend(tof_tile5), phlist5.extend(
                                ph_tile5), dc5.extend(dc_tile5)
                            rec_cnt5 += records5

                        x_tile6, r_tile6, tof_tile6, ph_tile6, records6, dc_tile6 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[5], dir)
                        if x_tile6:
                            xlist6.extend(x_tile6), rlist6.extend(r_tile6), toflist6.extend(tof_tile6), phlist6.extend(
                                ph_tile6), dc6.extend(dc_tile6)
                            rec_cnt6 += records6

                        x_tile7, r_tile7, tof_tile7, ph_tile7, records7, dc_tile7 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[6], dir)
                        if x_tile7:
                            xlist7.extend(x_tile7), rlist7.extend(r_tile7), toflist7.extend(tof_tile7), phlist7.extend(
                                ph_tile7), dc7.extend(dc_tile7)
                            rec_cnt7 += records7

                        x_tile8, r_tile8, tof_tile8, ph_tile8, records8, dc_tile8 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[7], dir)
                        if x_tile8:
                            xlist8.extend(x_tile8), rlist8.extend(r_tile8), toflist8.extend(tof_tile8), phlist8.extend(
                                ph_tile8), dc8.extend(dc_tile8)
                            rec_cnt8 += records8

                        x_tile9, r_tile9, tof_tile9, ph_tile9, records9, dc_tile9 = self.binned_data(datadict, xbounds[0],
                                                                                           ybounds[8], dir)
                        if x_tile9:
                            xlist9.extend(x_tile9), rlist9.extend(r_tile9), toflist9.extend(tof_tile9), phlist9.extend(
                                ph_tile9), dc9.extend(dc_tile9)
                            rec_cnt9 += records9

                        x_tile10, r_tile10, tof_tile10, ph_tile10, records10, dc_tile10 = self.binned_data(datadict, xbounds[0],
                                                                                                ybounds[9], dir)
                        if x_tile10:
                            xlist10.extend(x_tile10), rlist10.extend(r_tile10), toflist10.extend(
                                tof_tile10), phlist10.extend(ph_tile10), dc10.extend(dc_tile10)
                            rec_cnt10 += records10
        f.close()

        xlist1 = np.array(xlist1) / float(Rp)
        mean_x1 = np.mean(xlist1)  # plotting purposes (1 coord)
        rlist1 = np.array(rlist1) / float(Rp)
        mean_r1 = np.mean(rlist1)

        xlist2 = np.array(xlist2) / float(Rp)
        mean_x2 = np.mean(xlist2)
        rlist2 = np.array(rlist2) / float(Rp)
        mean_r2 = np.mean(rlist2)

        xlist3 = np.array(xlist3) / float(Rp)
        mean_x3 = np.mean(xlist3)
        rlist3 = np.array(rlist3) / float(Rp)
        mean_r3 = np.mean(rlist3)

        xlist4 = np.array(xlist4) / float(Rp)
        mean_x4 = np.mean(xlist4)
        rlist4 = np.array(rlist4) / float(Rp)
        mean_r4 = np.mean(rlist4)

        xlist5 = np.array(xlist5) / float(Rp)
        mean_x5 = np.mean(xlist5)
        rlist5 = np.array(rlist5) / float(Rp)
        mean_r5 = np.mean(rlist5)

        xlist6 = np.array(xlist6) / float(Rp)
        mean_x6 = np.mean(xlist6)
        rlist6 = np.array(rlist6) / float(Rp)
        mean_r6 = np.mean(rlist6)

        xlist7 = np.array(xlist7) / float(Rp)
        mean_x7 = np.mean(xlist7)
        rlist7 = np.array(rlist7) / float(Rp)
        mean_r7 = np.mean(rlist7)

        xlist8 = np.array(xlist8) / float(Rp)
        mean_x8 = np.mean(xlist8)
        rlist8 = np.array(rlist8) / float(Rp)
        mean_r8 = np.mean(rlist8)

        xlist9 = np.array(xlist9) / float(Rp)
        mean_x9 = np.mean(xlist9)
        rlist9 = np.array(rlist9) / float(Rp)
        mean_r9 = np.mean(rlist9)

        xlist10 = np.array(xlist10) / float(Rp)
        mean_x10 = np.mean(xlist10)
        rlist10 = np.array(rlist10) / float(Rp)
        mean_r10 = np.mean(rlist10)

        # ====== Slice TOF list ========

        npd2_Htof = self.get_npd2_Htof

        subtof1, subph1, subdc1 = self.slice_tofspec(toflist1, phlist1, dc1, npd2_Htof)
        subtof2, subph2, subdc2 = self.slice_tofspec(toflist2, phlist2, dc2, npd2_Htof)
        subtof3, subph3, subdc3 = self.slice_tofspec(toflist3, phlist3, dc3, npd2_Htof)
        subtof4, subph4, subdc4 = self.slice_tofspec(toflist4, phlist4, dc4, npd2_Htof)
        subtof5, subph5, subdc5 = self.slice_tofspec(toflist5, phlist5, dc5, npd2_Htof)
        subtof6, subph6, subdc6 = self.slice_tofspec(toflist6, phlist6, dc6, npd2_Htof)
        subtof7, subph7, subdc7 = self.slice_tofspec(toflist7, phlist7, dc7, npd2_Htof)
        subtof8, subph8, subdc8 = self.slice_tofspec(toflist8, phlist8, dc8, npd2_Htof)
        subtof9, subph9, subdc9 = self.slice_tofspec(toflist9, phlist9, dc9, npd2_Htof)
        subtof10, subph10, subdc10 = self.slice_tofspec(toflist10, phlist10, dc10, npd2_Htof)

        # =========== Convert tof to velocity =============

        vel1 = self.tofto_vel(subtof1)

        '''# Checking values (confirmed correct)
        print(len(subtof1[subtof1 < 113]), len(subtof1[subtof1 > 567]))
        print(len(vel1[vel1 > 873]), len(vel1[vel1 < 174.82]))'''

        vel2 = self.tofto_vel(subtof2)
        vel3 = self.tofto_vel(subtof3)
        vel4 = self.tofto_vel(subtof4)
        vel5 = self.tofto_vel(subtof5)
        vel6 = self.tofto_vel(subtof6)
        vel7 = self.tofto_vel(subtof7)
        vel8 = self.tofto_vel(subtof8)
        vel9 = self.tofto_vel(subtof9)
        vel10 = self.tofto_vel(subtof10)

        # make tof histograms, if ok, find mean, stdv, store with dir, x,r coords

        # ================ Plot Setup ==============
        fig, axarr = plt.subplots(nrows=2, ncols=5, figsize=(17, 8))
        fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
        fig.suptitle(
            'Velocity Distribution for H ENA (coin 0: 1 Start, 1 Stop), x: ' + str(xbin_bounds[0][0]) + '-' + str(
                xbin_bounds[0][1]) + r'$R_V$' + ', r: ' + str(ybin_bounds[0][0]) +
            '-' + str(
                ybin_bounds[9][1]) + r'$R_V$' + '\n' + 'VSO, ' + sensor.upper() + ', Ch: ' + dir + ', Date: 2005-2013',
            fontsize=11, y=0.98)

        # ============= Plotting ================
        vel_bin = 100.
        self.maxwell_plt(axarr[0, 0], vel1, vel_bin, rec_cnt1)
        self.maxwell_plt(axarr[0, 1], vel2, vel_bin, rec_cnt2)
        self.maxwell_plt(axarr[0, 2], vel3, vel_bin, rec_cnt3)
        self.maxwell_plt(axarr[0, 3], vel4, vel_bin, rec_cnt4)
        self.maxwell_plt(axarr[0, 4], vel5, vel_bin, rec_cnt5)
        self.maxwell_plt(axarr[1, 0], vel6, vel_bin, rec_cnt6)
        self.maxwell_plt(axarr[1, 1], vel7, vel_bin, rec_cnt7)
        self.maxwell_plt(axarr[1, 2], vel8, vel_bin, rec_cnt8)
        self.maxwell_plt(axarr[1, 3], vel9, vel_bin, rec_cnt9)
        self.maxwell_plt(axarr[1, 4], vel10, vel_bin, rec_cnt10)

        for i in range(5):
            axarr[1, i].set_xlabel('v [km/s]')
        axarr[0, 0].set_ylabel('Cnts/records', multialignment='center', fontsize=9)
        axarr[1, 0].set_ylabel('Cnts/records', multialignment='center', fontsize=9)

        # ========= Store Button =================
        ax7 = fig.add_axes([0.85, 0.01, 0.01, 0.01], frameon=False)
        ax7.axes.get_yaxis().set_visible(False)
        ax7.axes.get_xaxis().set_visible(False)
        ax7.text(0.5, 0.5, 'Store', color='k', bbox=dict(facecolor='red', edgecolor='black'), transform=ax7.transAxes,
                 picker=True)

        '''# ============ Pick Event =================
        def onpick(event):     # pick coordinate to plot TOF-PH plot
            artist = event.artist
            id = artist.get_text()
            if isinstance(artist, Text):
                if id == 'Store':
                    tofdict1 = {'dir': dir, 'mean_x': float(mean_x1), 'mean_r': float(mean_r1), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[0]),
                                 'mean_tof': float(mtof1), 'fwhm': float(fwhmtof1)}
                    tofdict2 = {'dir': dir, 'mean_x': float(mean_x2), 'mean_r': float(mean_r2), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[1]),
                                 'mean_tof': float(mtof2), 'fwhm': float(fwhmtof2)}
                    tofdict3 = {'dir': dir, 'mean_x': float(mean_x3), 'mean_r': float(mean_r3), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[2]),
                                 'mean_tof': float(mtof3), 'fwhm': float(fwhmtof3)}
                    tofdict4 = {'dir': dir, 'mean_x': float(mean_x4), 'mean_r': float(mean_r4), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[3]),
                                 'mean_tof': float(mtof4), 'fwhm': float(fwhmtof4)}
                    tofdict5 = {'dir': dir, 'mean_x': float(mean_x5), 'mean_r': float(mean_r5), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[4]),
                                 'mean_tof': float(mtof5), 'fwhm': float(fwhmtof5)}
                    tofdict6 = {'dir': dir, 'mean_x': float(mean_x6), 'mean_r': float(mean_r6), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[5]),
                                 'mean_tof': float(mtof6), 'fwhm': float(fwhmtof6)}
                    tofdict7 = {'dir': dir, 'mean_x': float(mean_x7), 'mean_r': float(mean_r7), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[6]),
                                 'mean_tof': float(mtof7), 'fwhm': float(fwhmtof7)}
                    tofdict8 = {'dir': dir, 'mean_x': float(mean_x8), 'mean_r': float(mean_r8), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[7]),
                                 'mean_tof': float(mtof8), 'fwhm': float(fwhmtof8)}
                    tofdict9 = {'dir': dir, 'mean_x': float(mean_x9), 'mean_r': float(mean_r9), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[8]),
                                 'mean_tof': float(mtof9), 'fwhm': float(fwhmtof9)}
                    tofdict10 = {'dir': dir, 'mean_x': float(mean_x10), 'mean_r': float(mean_r10), 'xbounds': list(xbin_bounds[0]), 'ybounds': list(ybin_bounds[9]),
                                 'mean_tof': float(mtof10), 'fwhm': float(fwhmtof10)}
                    # ===== Storing datadict file =======
                    self.store_meanHtof(tofdict1)
                    self.store_meanHtof(tofdict2)
                    self.store_meanHtof(tofdict3)
                    self.store_meanHtof(tofdict4)
                    self.store_meanHtof(tofdict5)
                    self.store_meanHtof(tofdict6)
                    self.store_meanHtof(tofdict7)
                    self.store_meanHtof(tofdict8)
                    self.store_meanHtof(tofdict9)
                    self.store_meanHtof(tofdict10)

        fig.canvas.mpl_connect('pick_event', onpick)'''
        plt.show()

    def subplot_tofph(self, ax, dir_tof, dir_ph, tof_bin):
        if not tof_bin:
            tof_bin = self._tofbin

        # Plotting dir0
        hist, xedges, yedges = np.histogram2d(dir_tof, dir_ph, bins=tof_bin)

        # normalize (sum of heights equal to one)
        h_norm = hist / float(len(dir_tof))

        ma_hist = np.ma.masked_where(h_norm == 0., h_norm)

        im = ax.imshow(np.rot90(ma_hist), interpolation='nearest',
                       norm=colors.LogNorm(vmin=ma_hist.min(), vmax=ma_hist.max()),
                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')
        '''else:
            im = ax.imshow(np.rot90(ma_hist), interpolation='nearest', vmin=0., vmax=ma_hist.max(),
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto')'''
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)

        # cbar.formatter.set_powerlimits((0, 0))
        # cbar.update_ticks()

        # ax.imshow(np.rot90(h_norm), interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    def subplot_tofhist(self, ax, tof, tof_bin, rec, tof_range):

        hist, bins = np.histogram(tof, bins=tof_bin)

        # Normalize with number of records
        rec_hist = hist / float(rec)

        # Plot the resulting histogram
        ax.plot(bins[:-1], rec_hist, '-', color='k')
        ax.set_yscale('log')
        # ax.set_xlim([tof_range[0], tof_range[-1]])
        ax.set_ylabel('Cnts/records', multialignment='center', fontsize=9)

    def subplot_phhist(self, ax, ph, ph_bin, rec, energ):

        ph_length = abs(energ[1] - energ[0])  # nsec
        hist, bins = np.histogram(ph, bins=ph_bin)

        # Normalize with number of records
        rec_hist = hist / float(rec)

        # Plot the resulting histogram
        ax.plot(bins[:-1], rec_hist, '-', color='k')
        ax.set_yscale('log')
        ax.set_ylabel('Cnts/records', multialignment='center', fontsize=9)

    def plt_tofspec(self, x, r, toflist, phlist, rec, bins, sensor, lisdate, dir, figcount):

        energ = self.get_npd2_OkeV

        # ================ Plot Setup ==============
        fig, axarr = plt.subplots(nrows=6, ncols=3, figsize=(17, 8))
        fig.subplots_adjust(left=0.06, right=0.95, top=0.9, bottom=0.05, wspace=0.3, hspace=0.4)
        fig.suptitle(
            'O Energy Spectra (coin 0: 1 Start, 1 Stop), x: ' + str(x[0]) + '-' + str(x[1]) + ', r: ' + str(r[0]) +
            '-' + str(r[1]) + '\n' + sensor.upper() + ', Ch: ' + dir + ', Date:' + lisdate[0], fontsize=11, y=0.97)

        '''for j in range(6):
            for i in range(3):
                axarr[j, i].axes.axis.set_tick_params(labelsize=9)'''

        # ============Slice Energy Spectrum ======
        tof_5kev, ph_5kev = self.slice_tofspec(toflist, phlist, energ['5keV'])
        tof_3kev, ph_3kev = self.slice_tofspec(toflist, phlist, energ['3keV'])
        tof_1kev, ph_1kev = self.slice_tofspec(toflist, phlist, energ['1.3keV'])
        tof_dot7, ph_dot7 = self.slice_tofspec(toflist, phlist, energ['.7keV'])
        tof_dot5, ph_dot5 = self.slice_tofspec(toflist, phlist, energ['.5keV'])
        tof_dot3, ph_dot3 = self.slice_tofspec(toflist, phlist, energ['.3keV'])

        # =========== Plotting TOF-PH==========
        self.subplot_tofph(axarr[0, 0], tof_5kev, ph_5kev, bins)
        self.subplot_tofph(axarr[1, 0], tof_3kev, ph_3kev, bins)
        self.subplot_tofph(axarr[2, 0], tof_1kev, ph_1kev, bins)
        self.subplot_tofph(axarr[3, 0], tof_dot7, ph_dot7, bins)
        self.subplot_tofph(axarr[4, 0], tof_dot5, ph_dot5, bins)
        self.subplot_tofph(axarr[5, 0], tof_dot3, ph_dot3, bins)

        # =========== Plot TOF hist ==========
        tof_bin = 100
        self.subplot_tofhist(axarr[0, 1], tof_5kev, tof_bin, rec, energ['5keV'])
        self.subplot_tofhist(axarr[1, 1], tof_3kev, tof_bin, rec, energ['3keV'])
        self.subplot_tofhist(axarr[2, 1], tof_1kev, tof_bin, rec, energ['1.3keV'])
        self.subplot_tofhist(axarr[3, 1], tof_dot7, tof_bin, rec, energ['.7keV'])
        self.subplot_tofhist(axarr[4, 1], tof_dot5, tof_bin, rec, energ['.5keV'])
        self.subplot_tofhist(axarr[5, 1], tof_dot3, tof_bin, rec, energ['.3keV'])

        # =========== Plot PH hist ==========================
        ph_bin = 100
        self.subplot_phhist(axarr[0, 2], ph_5kev, ph_bin, rec, energ['5keV'])
        self.subplot_phhist(axarr[1, 2], ph_3kev, ph_bin, rec, energ['3keV'])
        self.subplot_phhist(axarr[2, 2], ph_1kev, ph_bin, rec, energ['1.3keV'])
        self.subplot_phhist(axarr[3, 2], ph_dot7, ph_bin, rec, energ['.7keV'])
        self.subplot_phhist(axarr[4, 2], ph_dot5, ph_bin, rec, energ['.5keV'])
        self.subplot_phhist(axarr[5, 2], ph_dot3, ph_bin, rec, energ['.3keV'])

        # ========= Subplot setup ===============
        counts = [len(tof_dot5), len(tof_dot3), len(tof_1kev), len(tof_dot7), len(tof_dot5), len(tof_dot3)]
        keys = ['5keV', '3keV', '1.3keV', '.7keV', '.5keV', '.3keV']
        for j in range(6):
            axarr[j, 0].set_ylim([0, 255])
            axarr[j, 0].set_ylabel('PH', multialignment='center', fontsize=9)
            axarr[j, 0].set_title(str(keys[j]) + ' TOF-PH Spectra Normalized (Total Cnts: ' + str(counts[j]) + ')',
                                  fontsize=9)
            axarr[j, 1].set_title(str(keys[j]) + ' TOF Counts per Records, bin: ' + str(tof_bin), fontsize=9)
            axarr[j, 2].set_title(str(keys[j]) + ' PH Counts per Records, bin: ' + str(ph_bin), fontsize=9)
            axarr[j, 2].set_xlim([0, 255])
        axarr[5, 1].set_xlabel('TOF [ns]')
        axarr[5, 2].set_xlabel('PH')
        plt.savefig('/Users/eanunez/spacemaster/Kiruna/Master_Thesis/scripts/NPD_raw/figs/' + sensor + '_' +
                    lisdate[0] + 'espec_' + dir + '_' + str(figcount) + '.png')

        # self.tick_locator(axarr[j, 0], 50, 250)
        plt.show()

    def maxwell_plt(self, ax, vel, vel_bin, rec):

        hist, bins = np.histogram(vel, bins=vel_bin)

        # Normalize with number of records
        rec_hist = hist / float(rec)

        '''Checking values (confirmed correct)
        print(bins[0:2], bins[-2:])
        print(hist[0], hist[-1])'''

        # Plot the resulting histogram
        # ax.hist(vel, bins = vel_bin)
        # ax.plot(bins[:-1], rec_hist, '-', color='k', label= 'data')

        # scaling factor (kT/m^1/2)
        E1 = 7842.87  # [eV/amu^1/2] 80 ns
        E2 = 991.45  # 225 ns
        E3 = 200.78  # 500 ns

        # ====== keV to Joules Conversion =====
        J = 1.60218 * 10 ** -19  # [J/eV]
        kg = 1.66054 * 10 ** -27  # [kg/amu]
        E1J_kg = np.sqrt(E1 * J / float(kg))
        E2J_kg = np.sqrt(E2 * J / float(kg))

        mean, var, skew, kurt = maxwell.stats(vel, moments='mvsk')
        params = maxwell.fit(vel, floc=0.0, scale=E1J_kg)  # scale equiv to a = sqrt(kT/m)
        params2 = maxwell.fit(vel, floc=0.0, scale=E2J_kg)
        params3 = maxwell.fit(vel, floc=0)
        print(params)
        print(params2)
        print(params3)

        x = np.linspace(0, 1400., 500)

        # ax.plot(x, maxwell.pdf(x, *params), 'r', label = '%.1g eV/amu' % E1)
        # ax.plot(x, maxwell.pdf(x, *params2), 'b', label = '%.1g eV/amu' % E2)
        ax.plot(x, maxwell.pdf(x, *params3), 'g', label='maxwell (standardized)')
        # ax.plot(x, maxwell.pdf(x, *params4), 'c', label = 'normal')



        # ax.axvline(var, ymin = 0., facecolor='g', alpha=0.5, label = 'variance')

        # ax.set_yscale('log')
        ax.set_ylim([0, 0.0025])
        ax.set_title('Normalized to number of Records: ' + str(rec), fontsize=9)
        plt.legend(prop={'size': 8})

    @property
    def tofbin(self):
        return self._tofbin

    def set_tofbin(self, tofbin):
        self._tofbin = tofbin

    @property
    def __subplots(self):
        return self._subplots

    def set_subplots(self, nsubplots):
        self._subplots = nsubplots

    @property
    def get_figcount(self):
        return self._figcount

    def set_figcount(self, count):
        self._figcount += count

    @property
    def get_npd2_OkeV(self):
        return self.npd2_Okev

    @property
    def get_npd2_Htof(self):
        return self.npd2_Htof


def spice_npdraw(args):
    # Parsing Arguments
    parser = argparse.ArgumentParser(prog='python3 spice_npdraw.py')

    parser.add_argument("id", type=argparse.FileType('r'), help="input filename")
    parser.add_argument("-store", "--store", action="store_true", help="save data set with position info to .json file")
    parser.add_argument("-velplt", "--velplt", action="store_true", help="velocity plot as function of Venus Radius")
    parser.add_argument("-vso", "--vsoplot", action="store_true",
                        help="vso frame gen, required additional inputs: -date and -dir")
    parser.add_argument("-savefig", "--savefig", action="store_true", help="save figure")
    parser.add_argument("-date", "--date", type=str, action="store", help="input date (yyyy-mm-dd) or year")
    parser.add_argument('-dl', '--dlist', nargs='+', action="store",
                        help="inter list of dates (yyyy-mm-dd yyyy-mm-dd) or year. 'all' for the entire records")
    parser.add_argument("-dir", "--dir", type=str, action="store",
                        help="input which direction, options: dir0, dir1, dir2")
    parser.add_argument("-c", "--cover", action="store_true", help="coverage plot")
    parser.add_argument("-rec", "--rec", action="store_true", help="Records plot")
    parser.add_argument("-xrange", "--xr", nargs='+', action="store", help="x range to plot")
    parser.add_argument("-avgtof", "--avgtof", action="store_true", help="Average TOF plot")
    parser.add_argument("-event", "--event", action = "store_true", help="Number of Events Distribution Plot")

    args = parser.parse_args()

    # Setting path and filename
    dirpath = os.path.dirname(os.path.abspath(args.id.name))  # file path
    f_name = os.path.basename(args.id.name)  # filename
    full_path = os.path.abspath(args.id.name)
    vspice.init()

    # Pre-defined Var
    R_v = 6.051850 * 10 ** 3
    binning = [0.25, 0.25]

    # ================= Store Data - Spice Param ===================
    # File Checking
    if f_name.lower().endswith('.txt'):  # If text file containing list
        with open(full_path, 'r') as f:
            contents = [line.rstrip('\n') for line in f]
            for jsonline in contents:
                try:
                    full_path = dirpath + '/' + jsonline
                    jsonfile = Spice_npdraw(full_path, jsonline, f_name)
                    if args.store:
                        jsonfile.io_ephem()
                except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
                    print('Error: ', err)

    elif f_name.lower().endswith('.json'):
        try:
            file = Spice_npdraw(full_path, f_name, f_name)
            if args.store:
                file.io_ephem()

            plt_json = Spicenpd_plt(full_path, f_name, f_name)
            if args.velplt:
                if args.savefig:
                    plt_json.vel_plt(args.xr, savefig=True)
                else:
                    plt_json.vel_plt(args.xr)

            # ========== Events Plot Option =================
            elif args.event and args.dir and args.dlist:
                if args.savefig:
                    plt_json.event_plot(args.dir, R_v, binning, args.dlist, savefig= True)
                else:
                    plt_json.event_plot(args.dir, R_v, binning, args.dlist)

            # ========= Records Plot Option =================
            elif args.rec and args.dir and args.dlist:
                if args.savefig:
                    plt_json.rec_plot(args.dir, R_v, binning, args.dlist, savefig=True)
                else:
                    plt_json.rec_plot(args.dir, R_v, binning, args.dlist)

            # ======= avg TOF plot option ===============
            elif args.avgtof and args.dir:
                if args.savefig:
                    plt_json.tof_cover(args.dir, savefig = True)
                else:
                    plt_json.tof_cover(args.dir)
            else:
                print('missing additional arguments -cr/-c/-vso, -dir, -dl')
                parser.print_help()

        except (ValueError, RuntimeError, IndexError, ZeroDivisionError) as err:
            print('Error: ', err)

    else:

        print('Provide .txt or .json file containing data of the right format.')


if __name__ == "__main__":  # Main function
    import sys

    spice_npdraw(sys.argv[1:])
