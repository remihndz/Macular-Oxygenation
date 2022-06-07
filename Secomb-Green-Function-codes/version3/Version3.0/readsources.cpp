/*****************************************************
Read source strengths from file.  TWS December 07.
Enables program to be restarted.
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
UNDER DEVELOPMENT!!!!!!!!!!
******************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <math.h>
#include "nrutil.h"
#include <stdio.h>

void readsources(void)
{
	extern int mxx,myy,mzz,nnt,nseg,nnv,nsp,max;
	extern int *mainseg,*lowflow,**tisspoints;
	extern float *axt,*ayt,*azt,*ds,*g0,*pmean,*pmin,*pmax;
	extern float **qt,**start,**scos,**qv,**ax,**pt,**pvseg,**pevseg;

	int i,iseg,itp,isp;
	FILE *ifp;
	char bb[100];

	ifp = fopen("TissueSources.out", "r");
	fscanf(ifp,"%i %i %i %i", &mxx,&myy,&mzz,&nnt);

	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	for(i=1; i<=mxx; i++) fscanf(ifp,"%g", &axt[i]);
	for(i=1; i<=myy; i++) fscanf(ifp,"%g", &ayt[i]);
	for(i=1; i<=mzz; i++) fscanf(ifp,"%g", &azt[i]);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	for(itp=1; itp<=nnt; itp++)
		fscanf(ifp,"%i %i %i ", &tisspoints[1][itp],&tisspoints[2][itp],&tisspoints[3][itp]);
	for(isp=1; isp<=nsp; isp++){
		fscanf(ifp,"%g",&g0[isp]);
		fgets(bb,max,ifp);
		for(i=1; i<=nnt; i++) fscanf(ifp,"%g ", &qt[i][isp]);
	}
	fclose(ifp);

	ifp = fopen("VesselSources.out", "r");//needs work for isp>1
	fscanf(ifp,"%i %i", &nseg,&nnv);

	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp,"%g %g %g %g %g %g %g", 
		&start[1][iseg],&start[2][iseg],&start[3][iseg],&scos[1][iseg],&scos[2][iseg],&scos[3][iseg],&ds[iseg]);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	for(i=1; i<=nnv; i++) fscanf(ifp,"%g %g %g", &ax[1][i],&ax[2][i],&ax[3][i]);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	for(i=1; i<=nnv; i++) fscanf(ifp,"%i", &mainseg[i]);
	for(isp=1; isp<=nsp; isp++){
		fgets(bb,max,ifp);
		fgets(bb,max,ifp);
		for(i=1; i<=nnv; i++) fscanf(ifp,"%g", &qv[i][isp]);
	}
	fclose(ifp);

	ifp = fopen("VesselLevels.out", "r");//needs work for isp>1


	fgets(bb,max,ifp);
	for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp,"%g", &lowflow[iseg]);
	for(isp=1; isp<=nsp; isp++){
		fgets(bb,max,ifp);
		for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp,"%g", &pvseg[iseg][isp]);
		fgets(bb,max,ifp);
		for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp,"%g", &pevseg[iseg][isp]);
	}
	fclose(ifp);

	ifp = fopen("TissueLevels.out", "r");

	for(isp=1; isp<=nsp; isp++){
		fgets(bb,max,ifp);
		for(itp=1; itp<=nnt; itp++)	fscanf(ifp,"%g", &pt[itp][isp]);
		fscanf(ifp,"%f %f %f", &pmean[isp],&pmin[isp],&pmax[isp]);
		fgets(bb,max,ifp);
	}
	fclose(ifp);
}