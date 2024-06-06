#Optical Depth calculation along a line of sight (LOS)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy 
import sympy
import multiprocessing

#Import data from text files--------------------------------------------------------------------------------------------------------------------------

			
			




spectral_data=np.loadtxt(fname="data/spectral_data.txt", delimiter=",", dtype="str") #import spectral line data
rec_data=np.loadtxt(fname="data/rec_data.txt", delimiter=",", dtype="str") #import recombination fit data



#Parameters

CosmoParam={
	"h" : 0.678,
	"Omega_M" : 0.308,
	"Omega_R" : 0,
	"Omega_DE" : 0.692,
	"c" : scipy.constants.c *10**2 #cm/s
}
SpectralParam={
	"specie" : "HI",
	"line_number" : 0, #number of the line in the data table HI 0 corresponds to Lyalpha
	"q_e" : -4.803*10**(-10), #in esu
	"m_e" : 9.109*10**(-28),  #in g
	}
SpectralParam["Lambda"] = float(spectral_data[spectral_data[:,0]==SpectralParam["specie"]][SpectralParam["line_number"]][3]) #in A
SpectralParam["f"] = float(spectral_data[spectral_data[:,0]==SpectralParam["specie"]][SpectralParam["line_number"]][6]) #oscilatorry strength
SpectralParam["mass"] = float(spectral_data[spectral_data[:,0]==SpectralParam["specie"]][SpectralParam["line_number"]][2])*scipy.constants.proton_mass #mass of the absorber
SpectralParam["nu"] = 10**8 * CosmoParam["c"]/SpectralParam["Lambda"] #transition frecuency, the prefactor changes from A to cm
SpectralParam["I_alpha"] = (1/(SpectralParam["nu"]) * np.pi *(SpectralParam["f"])* ((SpectralParam["q_e"]))**2 / ((SpectralParam["m_e"]) * CosmoParam["c"])) #total cross section parameter the units are esu,g,s,cm









#Voigt function--------------------------------------------------------------------------------------------------------------------------

def Voigt_Faddeeva(x,y):
	
	
	return scipy.special.wofz(x+y*1j).real


def Voigt_Tepper_Garcia(x,y):
	
	return  np.exp(-x**2)-2*y/(np.sqrt(np.pi)) * ( 1/(2*x**2) *(4*x**2+3)*(x**2+1)*np.exp(-2*x**2)-1/(4*x**4) *(2*x**2+3)*(1-np.exp(-2*x**2))          )








#Hubble factor function calculation--------------------------------------------------------------------------------------------------------------------------

def Hubble(z,Omega_DE=CosmoParam["Omega_DE"],Omega_R=CosmoParam["Omega_R"],Omega_M=CosmoParam["Omega_M"],h=CosmoParam["h"]): 
	
	H0=h*100 #in km/s/Mpc
	
	return H0*np.sqrt( Omega_R*(1+z)**4 +Omega_DE+Omega_M*(1+z)**3  )*1/(3.086*10**19) #returns in Km/s/Mpc, but I want it in s^-1

def Hubble_kmsmpc(z,Omega_DE=CosmoParam["Omega_DE"],Omega_R=CosmoParam["Omega_R"],Omega_M=CosmoParam["Omega_M"],h=CosmoParam["h"]): 
	
	H0=h*100 #in km/s/Mpc
	
	return H0*np.sqrt( Omega_R*(1+z)**4 +Omega_DE+Omega_M*(1+z)**3  )

def dztodx(dz,z):
	#transforms a pixel's redshift width dz to proper length dx. z is the pixel's redshift
	
	
	return CosmoParam["c"] * dz / (Hubble(z))


def dxtodz(dx,z):
	#transforms a pixel's proper width dx to redshfit width
	
	
	return 1/CosmoParam["c"] * dx *(Hubble(z))




#b parameter function--------------------------------------------------------------------------------------------------------------------------

def bParam(T):
	
	return np.sqrt(2*T*scipy.constants.Boltzmann/ (SpectralParam["mass"])) *100 #Convert from m/s to cm/s

	


#Recombination rate(alpha parameter) in cm^3 s^-1, T in Kelvin, updated according to https://arxiv.org/pdf/1406.6361.pdf and https://arxiv.org/pdf/astro-ph/9509083.pdf
def alpha(T):
	specie=SpectralParam["specie"]
	
	def fittingRec(a,b,T,T0,T1):
		
		return a* 1/( np.sqrt(T/T0) * (1+np.sqrt(T/T0))**(1-b) * (1+np.sqrt(T/T1))**(1+b)    )
	
	if specie=="He":
		if T<10**6:
			a=float(rec_data[rec_data[:,0]==specie][0,3])
			b=float(rec_data[rec_data[:,0]==specie][0,4])
			T0=float(rec_data[rec_data[:,0]==specie][0,5])
			T1=float(rec_data[rec_data[:,0]==specie][0,6])
		else:
			a=float(rec_data[rec_data[:,0]==specie][1,3])
			b=float(rec_data[rec_data[:,0]==specie][1,4])
			T0=float(rec_data[rec_data[:,0]==specie][1,5])
			T1=float(rec_data[rec_data[:,0]==specie][1,6])
		
	else:
		a=float(rec_data[rec_data[:,0]==specie][0,3])
		b=float(rec_data[rec_data[:,0]==specie][0,4])
		T0=float(rec_data[rec_data[:,0]==specie][0,5])
		T1=float(rec_data[rec_data[:,0]==specie][0,6])
		
	return fittingRec(a,b,T,T0,T1)
		
		
		
#Compute the optical depth for each redshift bin for given input fields and return the optical depth tau--------------------------------------------------------------------------------------------------------------------------
def ComputeTau(z,n_all,T_all,V_all,D_all):
	
	tau=np.zeros((z.size,n_all.shape[1]))
	
	ODweighted_T=np.zeros((z.size,n_all.shape[1]))
	ODweighted_n=np.zeros((z.size,n_all.shape[1]))
	ODweighted_D=np.zeros((z.size,n_all.shape[1]))		
	bin_redshift=np.append(np.diff(z),np.diff(z)[-1])#redshift bins
	Lz=abs(z[0]-z[-1]) #redshift box length
	
	for i in range(z.size):
		sg=1
		if i>z.size/2:
			sg=-1
			
			
			
		for j in range(z.size-1):
			
			#implement BC by evaluating whether the distance between pixels exceeds half the box size
			if abs(z[j]-z[i])>Lz/2 :
				 
				OD_cont=CosmoParam["c"] *SpectralParam["I_alpha"]/np.sqrt(np.pi) * dztodx(bin_redshift[j],z[j]-Lz) * n_all[j,:] / ( bParam(T_all[j,:])*(1+z[j]-Lz)) *  Voigt_Faddeeva( 	CosmoParam["c"]* (z[j]-z[i]-sg*Lz) /( bParam(T_all[j,:])*(1+z[i]) ) + (100000*V_all[j,:]) / (bParam(T_all[j,:]))         ,alpha(T_all[j,:]))
				
			else:
				OD_cont=CosmoParam["c"] *SpectralParam["I_alpha"]/np.sqrt(np.pi) * dztodx(bin_redshift[j],z[j]) * n_all[j,:] / ( bParam(T_all[j,:])*(1+z[j])) *  Voigt_Faddeeva( 	CosmoParam["c"]* (z[j]-z[i]) /( bParam(T_all[j,:])*(1+z[i]) ) + (100000*V_all[j,:]) / (bParam(T_all[j,:]))         ,alpha(T_all[j,:]))
			
			tau[i,:]=tau[i,:]+OD_cont
			ODweighted_T[i,:]=ODweighted_T[i,:]+T_all[j,:]*OD_cont
			ODweighted_n[i,:]=ODweighted_n[i,:]+n_all[j,:]*OD_cont
			ODweighted_D[i,:]=ODweighted_D[i,:]+D_all[j,:]*OD_cont
		
		#last iteration
		if abs(z[j]-z[i])> Lz/2:
			OD_cont=CosmoParam["c"] *SpectralParam["I_alpha"]/np.sqrt(np.pi) * dztodx(bin_redshift[z.size-1],z[z.size-1]) * n_all[z.size-1,:] / ( bParam(T_all[z.size-1,:])*(1+z[z.size-1])) *  Voigt_Faddeeva( CosmoParam["c"]* ((z[z.size-1]-z[i])-sg*Lz) /( bParam(T_all[z.size-1,:])*(1+z[i]) ) + (100000*V_all[z.size-1,:]) / (bParam(T_all[z.size-1,:]))         ,alpha(T_all[z.size-1,:])) 		
			
			
		else:
			OD_cont=CosmoParam["c"] *SpectralParam["I_alpha"]/np.sqrt(np.pi) * dztodx(bin_redshift[z.size-1],z[z.size-1]) * n_all[z.size-1,:] / ( bParam(T_all[z.size-1,:])*(1+z[z.size-1])) *  Voigt_Faddeeva( CosmoParam["c"]* (z[z.size-1]-z[i]) /( bParam(T_all[z.size-1,:])*(1+z[i]) ) + (100000*V_all[z.size-1,:]) / (bParam(T_all[z.size-1,:]))         ,alpha(T_all[z.size-1,:])) 		
			
			
		
		
		tau[i]=tau[i]+OD_cont
		ODweighted_T[i,:]=ODweighted_T[i,:]+ T_all[z.size-1,:]*OD_cont
		ODweighted_n[i,:]=ODweighted_n[i,:]+ n_all[z.size-1,:]*OD_cont
		ODweighted_D[i,:]=ODweighted_D[i,:]+ D_all[z.size-1,:]*OD_cont
  
		ODweighted_T[i,:]=ODweighted_T[i,:]/tau[i,:]
		ODweighted_n[i,:]=ODweighted_n[i,:]/tau[i,:]
		ODweighted_D[i,:]=ODweighted_D[i,:]/tau[i,:]

		
		
		
		
		
	
		
	return ((tau,ODweighted_n,ODweighted_T), ODweighted_D)
	

	
	

#-------------------------------------------------------------------------------------------------------------------------
import multiprocessing

n_cpu=multiprocessing.cpu_count()

def Compute_D(arg):
	model_z,model,thermal=arg
	z = np.load(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/z_all.npy")
	n_all = np.load(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/n_all.npy") #in cm^-3
	T_all = np.load(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/T_all.npy")#in K
	V_all = np.load(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/V_all.npy") #in km/s
	D_all=	np.load(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/Delta_all.npy")
	
	result=ComputeTau(z,n_all,T_all,V_all,D_all)	
 
	tau=result[0]
	ODweighted_D=result[1]
 
	np.save(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/tau.npy",tau)
	np.save(f"sherwood/planck1_20_1024_{model}{thermal}/los_{model_z}/Deltaw.npy",ODweighted_D)
	print(f"Finished {model} {thermal} at z={model_z}")
    

        
list_1 = ("4.1","4.2","4.3","4.4","4.5","4.6","4.7","4.8","4.9","5.0")
list_2 = ("cdm","wdm2","wdm3","wdm4","wdm8","wdm12")
list_3=("_hot","_cold")
 
unique_combinations = ( )
 
for i in range(len(list_1)):
    for j in range(len(list_2)):
        for k in range(len(list_3)):
        	unique_combinations.append((list_1[i], list_2[j],list_3[k]))	

if __name__ == '__main__':        
	with Pool(n_cpu) as pool:
		pool.map(Compute_D,unique_combinations)    