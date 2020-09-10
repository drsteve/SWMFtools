import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import linalg

class supermag_parser(object): #TODO: inherit from spacepy.datamodel.SpaceData to mirror construction of SWMF MagFile
    """
    This class can be used to read data from supermag ascii files and to process the data
    """

    def __init__(self,fname=None,ncol_time=6,ncol_data=7,deriv_method='difference', legacy=False):
        if not fname is None:
            self.read_supermag_data(fname, ncol_time=ncol_time, ncol_data=ncol_data,
                                    deriv_method=deriv_method, legacy=legacy)
            
        
    def read_supermag_data(self, filename, ncol_time=6, ncol_data=7,
                           legacy=False, critfrac=0.1, badflag=999999.0,
                           deriv_method='spectral'):
        """
        Read and parse a SuperMAG magnetometer file.
        
        Optional Parameters
        -------------------
        legacy : bool
            For legacy SuperMag files, use True.
        """
        data = {}
        time = []
        if legacy:
            # Columns (after IAGA ID) for each param
            bncol = 0
            becol = 1
            bzcol = 2
            mltcol = 3
            mlatcol = 4
            szacol = 5
            declcol = 6
        with open(filename,'r') as f:
            look_for_header = True
            lines=[]
            for i, line in enumerate(f):
                datum = line.strip()
                if look_for_header and legacy:
                    istat = 0
                    lines.append(datum)
                    if len(lines) > 0:
                        if len(lines[i]) > 0:
                            if lines[i][0] == '=':
                                opts=lines[i-1].split()
                                for station in opts[-1].split(','):
                                    data[station] = {'time':[],'data':[]}
                                look_for_header=False
                elif look_for_header and not legacy:
                    istat = 0
                    lines.append(datum)
                    print('Checking header in {}, line {}\n{}'.format(filename, i, line))
                    if datum.startswith('Stations'):
                        statlist = datum.split(': ')[1].split(',')
                        for station in statlist:
                            data[station] = {'time':[],'data':[]}
                    if datum.startswith('Param'):
                        paramlist = datum.split(': ')[1].split('|')
                        paramlist = [prm.strip() for prm in paramlist]
                        bncol = paramlist.index('Mag. Field NEZ') - 1
                        becol = bncol + 1
                        bzcol = bncol + 2
                        szacol = paramlist.index('Solar Zenith Angle') - 1
                        mlatcol = paramlist.index('Mag. Lat.') - 1
                        mltcol = paramlist.index('MLT') - 1
                        declcol = paramlist.index('Mag. Declination') - 1
                        look_for_header = False
                else:
                    line_data = datum.split()
                    if istat==0:
                        # Line with time data and num. stations
                        tstamp = pd.Timestamp(*list(map(int,line_data[:-1])))
                        time.append(tstamp)
                        nstat = int(line_data[-1])
                    else:
                        # Line with data for a given station at one time
                        stat = line_data[0]
                        data[stat]['time'].append(time[-1])
                        data[stat]['data'].append(np.array([list(map(float,line_data[1:]))]))

                    if istat == nstat:
                        istat = 0
                    else:
                        istat+=1
            
            self.time = np.array(time)
            
            # Store data in object unless there is a significant portion missing.
            self.station={}
            for key in data.keys():
                if len(data[key]['time']) > critfrac*len(self.time):
                    # Make sure that there is some good data in the time series, 
                    # otherwise skip this station
                    if not all(np.array(data[key]['data']).squeeze()[:,0] == badflag): 
                        self.station[key] = {} #TODO: do as SpaceData
                        # Filter out bad data and interpolate over gaps
                        squeeze = lambda x: np.asarray(x).squeeze()
                        squeezedB = squeeze(data[key]['data'])
                        self.station[key]['B'] = np.c_[self.bad_data_filter(squeezedB[:,bncol], data[key]['time']),
                                                       self.bad_data_filter(squeezedB[:,becol], data[key]['time']),
                                                       self.bad_data_filter(squeezedB[:,bzcol], data[key]['time'])]
                        
                        B = self.station[key]['B']
                        # Calculate the time derivatives of the magnetic field
                        Bdot = np.c_[self.get_derivative(B[:,0], deriv_method=deriv_method, dt=60.0),
                                     self.get_derivative(B[:,1], deriv_method=deriv_method, dt=60.0),
                                     self.get_derivative(B[:,2], deriv_method=deriv_method, dt=60.0)]
                        
                        idBmax = linalg.norm(B[:,:2],axis=1).argmax()
                        idTmax = linalg.norm(Bdot[:,:2],axis=1).argmax()
                        
                        # Store the data
                        self.station[key]['time'] = self.time
                        self.station[key]['Bdot'] = Bdot
                        #TODO: interpolate the following onto the correct timebase
                        self.station[key]['mlt'] = squeeze(data[key]['data'])[:, mltcol]
                        self.station[key]['decl'] = squeeze(data[key]['data'])[:, declcol]
                        self.station[key]['mlat'] = squeeze(data[key]['data'])[0, mlatcol]
                        self.station[key]['sza'] = squeeze(data[key]['data'])[0, szacol]
                        self.station[key]['max_indices'] = [idBmax,idTmax]
                        self.station[key]['maxB'] = linalg.norm(B[idBmax,:2])
                        self.station[key]['maxBdot'] = linalg.norm(Bdot[idTmax,:2])
                                    
        return
    
    def filter_window(self,npts,frac=0.05,wid=0.025):
        # A rounded-rectangle function used for FFT windowing
        x=np.linspace(0,1,npts)
        y=0.5*(np.tanh((x-frac)/wid)-np.tanh((x-(1-frac))/wid))
        return y

    def get_derivative(self,data,deriv_method,dt=1.0):
        
        if deriv_method == 'spectral':
            # Calculate the derivative of a quantity using a spectral method
            filt = self.filter_window(data.size)
            freq = np.fft.fftfreq(data.size,d=dt)
            dfft = np.fft.fft(filt*data)
            ddot = np.real(np.fft.ifft(2j*np.pi*freq*dfft))
        else: # method = 'difference'
            ddot = np.zeros_like(data)
            ddot[1:-1] = (data[2:]-data[:-2])/(2*dt)
            ddot[0] = (3*data[0]-4*data[1]+data[2])/(2*dt)
            ddot[-1] = -(3*data[-1]-4*data[-2]+data[-3])/(2*dt)
            
        return ddot
  
    def bad_data_filter(self, data, ticks, baddata=999999.0):
        numericticks = np.asarray([tt.timestamp() for tt in ticks])
        # Replace bad data with linear interpolation,
        # applying persistence if bad data is at edge
        if np.all(data==baddata):
            interp_data = np.zeros_like(data)
        elif np.any(data==baddata) or len(ticks)!=len(self.time):
            new_data = np.empty((len(self.time), 3))
            first_good=np.where(data!=baddata)[0][0]
            last_good=np.where(data!=baddata)[0][-1]

            #build interpolating func. with good data only
            good_filter = data != baddata
            xvals = np.linspace(self.time[0].timestamp(), self.time[-1].timestamp(),
                                len(self.time))
            dinterp = interpolate.interp1d(numericticks[good_filter],
                                           data[good_filter],
                                           kind='linear', bounds_error=False,
                                           fill_value=(data[good_filter][0],data[good_filter][-1]))
            interp_data = dinterp(xvals)
        else:
            interp_data = data.copy()

        return interp_data
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import glob
    
    # Grab a data file
    data_file = glob.glob('../test_data/2*.txt')[0]
    data_file = '../test_data/supermag-20000810-20000812.txt'
    smp = supermag_parser(data_file)
    
    # Get the GMD peaks for each station
    dbmax=[]
    dtmax=[]
    mlat=[]
    for key,stat in smp.station.items():
        mlat.append(stat['mlat'])
        dbmax.append(stat['maxB'])
        dtmax.append(stat['maxBdot'])
       
    # Plot the latitudinal distribution of peak GMDs
    fig=plt.figure(1,figsize=(12,8))
    ax1,ax2=fig.subplots(2,1,sharex=True)
    ax1.scatter(mlat,dtmax,s=200,alpha=0.5,color='royalblue')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-1,3e1])
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel('Peak $dB/dt$ (nT/s)',fontsize=20)
    ax2.scatter(mlat,dbmax,s=200,alpha=0.5,color='royalblue')
    ax2.set_yscale('log')
    ax2.set_ylim([2e2,5e3])
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel('Magnetic Latitude (Degrees)',fontsize=20)
    ax2.set_ylabel('Peak $\Delta B$ (nT)',fontsize=20)
    plt.tight_layout()
    plt.show()
