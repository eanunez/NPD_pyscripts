'''
NAME:	NPD (Raw mode) Reading Script
PURPOSE: Read NPD (Raw mode) data, NetCDF format in python2.7 directly from bash shell using 'argparse' module
PROGRAMMER: Emmanuel A. Nunez (emmnun-4@student.ltu.se)
PARAMETERS:
                   id: data filename
                   -v: optional argument, print information contained in the file
          Sample Usage:
	           python NPD_raw_read.py npd1raw20080522152536.nc -v

REFERENCES:
    netcdf4-python -- http://code.google.com/p/netcdf4-python/
    Chris Slocum -- http://schubert.atmos.colostate.edu/~cslocum/netcdf_example.html
    Xiao-Dong Wang
    Yoshifumi Futaana
'''
#!/usr/local/bin/python
from netCDF4 import Dataset
import numpy as np
import argparse, os


def ncdump(nc_fid, verb):
	'''
	ncdump outputs dimensions, variables and their attribute information.
	The information is similar to that of NCAR's ncdump utility.
   	ncdump requires a valid instance of Dataset.
	
	Parameters
	 ----------
    	nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
   	 verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

	Returns
   	 -------
    	nc_attrs : list
        A Python list of the NetCDF file global attributes
    	nc_dims : list
        A Python list of the NetCDF file dimensions
    	nc_vars : list
        A Python list of the NetCDF file variables
	'''

	def print_ncattr(key):
		"""
		Prints the NetCDF file attributes for a given key

		Parameters
		 ----------
            	key : unicode
            	a valid netCDF4.Dataset.variables key
            	"""
		try:
			print "\t\ttype:", repr(nc_fid.variables[key].dtype)
           		for ncattr in nc_fid.variables[key].ncattrs():
               			 print '\t\t%s:' % ncattr,\
                      			repr(nc_fid.variables[key].getncattr(ncattr))
            	except KeyError:
            		print "\t\tWARNING: %s does not contain variable attributes" % key

	# NetCDF global attributes
    	nc_attrs = nc_fid.ncattrs()
    	if verb:
		print "NetCDF Global Attributes:"
        	for nc_attr in nc_attrs:
            		print '\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr))
	nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    	# Dimension shape information.
    	if verb:
        	print "NetCDF dimension information:"
       		for dim in nc_dims:
           		print "\tName:", dim 
            		print "\t\tsize:", len(nc_fid.dimensions[dim])
			print_ncattr(dim)
    	# Variable information.
    	nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    	if verb:
        	print "NetCDF variable information:"
       		for var in nc_vars:
            		if var not in nc_dims:
                		print '\tName:', var
               			print "\t\tdimensions:", nc_fid.variables[var].dimensions
              			print "\t\tshape:", nc_fid.variables[var].shape
	  			print "\t\tsize:", nc_fid.variables[var].size
                		print_ncattr(var)
    	return nc_attrs, nc_dims, nc_vars

def NPD_raw_read(args):

	#Parsing Arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("id", type=file, help="input filename")
	parser.add_argument("-v", "--verbose", action ="store_true")
	args = parser.parse_args()
	
	path = os.path.abspath(args.id.name)	# file path
	#print path
	nc_f = args.id.name  # filename
	nc_fid = Dataset(nc_f, 'r')  # Dataset is the class behavior to open the file
       
	# and create an instance of the ncCDF4 class
	if args.verbose:
		nc_attrs, nc_dims, nc_vars = ncdump(nc_fid, verb=True)
	else:
		nc_attrs, nc_dims, nc_vars = ncdump(nc_fid, verb=False)

	# Extract data from NetCDF file
	ST = nc_fid.variables['StartTime'][:]  # extract/copy the data(numpy.ndarray type)
	ET = nc_fid.variables['StopTime'][:]
	Time = nc_fid.variables['Time'][:]       	
	DeltaT = nc_fid.variables['DeltaT'][:]
	Cnts = nc_fid.variables['Cnts'][:]	
	Intens = nc_fid.variables['Intens'][:] 
	Calib = nc_fid.variables['Calib'][:]
        Regs = nc_fid.variables['Regs'][:]

	NOR = Time.size
	discard = np.chararray((NOR, 3)) # string array of NOR rows, 3 colums		
	
	#Bit manipulation to assign variables Raw: 32bit-7bit = 25bit, 7bit=spare
	# Check [Grigoriev,2007] for data format

	TOFs = np.bitwise_and(Intens, int('0xfff',0))
	Phs = np.bitwise_and(np.right_shift(Intens,12), int('0xff',0))
	Dirs = np.bitwise_and(np.right_shift(Intens,20), int('0x3',0))
	Coins = np.bitwise_and(np.right_shift(Intens,22), int('0x7',0))
	

	#Create Dictionary with (keys, value) pairs
	raw_data = {'ST':ST, 'ET':ET, 'Time':Time, 'DT':DeltaT, 'Cnts':Cnts, 'Calib':Calib, 'Regs':Regs, 'TOFs':TOFs, 'dirs':Dirs, 'coins':Coins, 'PHs':Phs, 'discard':discard}
	# Close original NetCDF file.
	nc_fid.close()
	return raw_data

if __name__ == "__main__":
	import sys
	NPD_raw_read(sys.argv[1:])
