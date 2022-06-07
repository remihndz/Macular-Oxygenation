/*****************************************************
Tissue uptake rates of solutes as a function of levels
Must be provided for each application.  TWS November 07.
Version 1.0, May 1, 2008.
With 9/08 updates
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
******************************************************/
#include <math.h>
#include "nrutil.h"
#include <stdio.h>

void tissrate(int nsp, float *p, float *mtiss, float *mptiss)
{
	extern float **tissparam;//added April 2010
	float pcr,m0,gf0,gf1;//not external, April 2010
	int isp;
	for(isp=1; isp<=nsp; isp++){
		switch (isp)
		{
		case 1: //oxygen
			m0 = tissparam[1][isp];
			pcr = tissparam[2][isp];
			if(p[isp] >= 0.){
				mtiss[isp] = -m0*p[isp]/(p[isp] + pcr);
				mptiss[isp] = -m0*pcr/SQR(p[isp] + pcr);
			}
			else{
				mtiss[isp] = 0.;
				mptiss[isp] = 0.;
			}
			break;
		case 2: //VEGF: non-permeable diffusible solute, based on Mac Gabhann and Popel - 2010
			if(p[1] <= 1.) mtiss[2] = 6.*tissparam[1][2];
			else if(p[1] <= 20.) mtiss[2] = (1. + 5.*pow((20. - p[1])/19.,3.))*tissparam[1][2];
			else mtiss[2] = tissparam[1][2];
			mtiss[2] -= tissparam[2][2]*p[2];
			mptiss[2] = -tissparam[2][2];
/*
		case 2: //non-permeable diffusible solute produced in hypoxic regions - old version 
			gf0 = tissparam[1][isp];
			gf1 = tissparam[2][isp];
			if(p[1] >= 0.) mtiss[isp] = gf0*pcr/(p[1] + pcr) - gf1*p[isp];
			else mtiss[isp] = gf0 - gf1*p[isp];
			mptiss[isp] = -gf1;
*/
			break;
		case 3: //permeable solute delivered in blood with linear consumption
			gf0 = tissparam[1][isp];
			gf1 = tissparam[2][isp];
			mtiss[isp] = -gf1*p[isp];
			mptiss[isp] = -gf1;
			break;
		case 4: //non-diffusible solute produced in hypoxic regions
			gf0 = tissparam[1][isp];
			gf1 = tissparam[2][isp];
			if(p[1] >= 0.) mtiss[isp] = gf0*pcr/(p[1] + pcr) - gf1*p[isp];
			else mtiss[isp] = gf0 - gf1*p[isp];
			mptiss[isp] = -gf1;
			break;
		default:
			printf("Error: nsp is too large in tissrate\n");
		}
	}
}