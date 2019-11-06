exo4 is an artificial dataset created using the exoplanet random generator from the online game Battalia.fr .
This data set describes 1000 randomly generated exoplanets with 11 numerical attributes and their associated class.
The file exo4_atm_X contains only a restricted number of attributes from the original data set.

Attributes:
- Pxxx : Atmosphere partial pressure between 0 and 100 [FLOAT] : H2O, He, CH4, H2, N2, O2, Ar, CO2, SO2, K 
- Type [CHAR]: class of the observed planet
	- "r" : rocky world
	- "g" : gas planet
	- "l" : lava world
	- "d" : desert planet
	- "i" : ice world

	
Similarity tree between the different types:	
	
  r,g,l,d,i
 /           \
g	      r,l,d,i
         /         \ 
        r        l,d,i
				/	    \
               l       d,i
		              /  | 
		             d   i   
				   
