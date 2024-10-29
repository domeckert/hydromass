from scipy import constants as con
from astropy import constants as const

ne2np=0.8337

cgsMpc=con.parsec*100.*1e6

kev2erg=1000.*con.eV/con.erg

cgsG=con.G*1e3

cgsamu=con.m_u*1e3

Msun=const.M_sun.value * 1e3

cgskpc = cgsMpc / 1e3

year = 60. * 60. * 24. * 365.

y_prefactor = 1.3018537e-27  # cm2/keV
