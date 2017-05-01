Required packages: numpy, netCDF4

Installation:

1) copy files in the working directory

2) For python2.7 (NPD_raw_read.py), run directly from the terminal

    $ python NPD_raw_read.py filename

To show information of the file, run with optional argument -v

    $ python NPD_raw_read.py filename -v

3) For python3 (NPD_raw_readv1.py), use as a module and import inside python3

    >>> import NPD_raw_readv1 as raw

    >>> file = 'filename'

    >>> v = '-v'

    >>> raw_data = raw.NPD_raw_read(file, v)

"filename" is a netCDF name of file

"-v" is an optional argument to show data attributes inside the netCDF file

for more details, see descriptions inside the module/script

Since python3 has given the priority over python2.7, upcoming modules are designed to run for python3.

4) 'plt_filter.py' has the following functionalities: 
  - plot time series PH, 
  - stores filtered data to 'json' file for further data manipulation(new created file is based on type of NPD sensor and year)
  - accepts '.nc' file or '.txt' file containing list of '.nc' files as input

   $ python3 plt_filter.py filename -tph

See script for further details and optional arguments

5) 'timeSeries.py' has the following functionalities:
  - plots time series Pulse height distribution in each sensor and direction 
    (both 1D and 2D histogram) for long period, i.e. whole mission
  - plots time series Time-of-Flight(TOF) - Pulse height (PH)
    (See script for further details)
 
