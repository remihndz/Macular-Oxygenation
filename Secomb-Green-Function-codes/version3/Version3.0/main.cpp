/************************************************************************
Main program to call greens
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
See greens.cpp for description of changes.
***********************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include "nrutil.h"

void input(void);
void analyzenet(void);
void picturenetwork(float *nodvar, float *segvar, const char fname[]);
void greens(void);
void readsources(void);
void contour(const char fname[]);
void histogram(void);
void setuparrays0();
void setuparrays1(int nseg, int nnod);
void setuparrays2(int nnv, int nnt);

int max=100,nmaxvessel,nmaxtissue,nmax,rungreens,initgreens,g0method,linmethod;
int mxx,myy,mzz,nnt,nseg,nnod,nnodfl,nnv,nsp,nnodbc,nodsegm,nsegfl;
int slsegdiv,nsl1,nsl2;
int is2d; //needed for 2d version

int *mainseg,*permsolute,*nodrank,*nodtyp,*nodout,*bcnodname,*bcnod,*bctyp,*lowflow;
int *nodname,*segname,*segtyp,*nspoint,*istart,*nl,*nk,*indx,*ista,*iend;
int *errvesselcount,*errtissuecount;
int *imaxerrvessel,*imaxerrtissue,*nresis;  //added April 2010
int *oxygen,*diffsolute; //added April 2010
int **segnodname,**nodseg,**tisspoints,**nodnod;
int ***nbou;

int **tissfix;	//added September 2010
float **tisserr,**dmtissdp,*mptissref;//September 2010;

float gtt;	//added September 2010
float fn,c,alphab,p50,cs,cext,hext,req,q0,flowfac=1.e6/60.;
float pi1 = atan(1.)*4.,fac = 1./4./pi1;
float lb,maxl,v,vol,vdom,errfac,tlength,alx,aly,alz,lowflowcrit;
float tlengthq,tlengthqhd;//added 8/09
float xmax,ymax,scalefac;
float w2d,r2d; //needed for 2d version

float *axt,*ayt,*azt,*ds,*diff,*pmin,*pmax,*pmean,*pref,*g0,*g0fac,*g0facnew,*sumal;
float *diam,*rseg,*q,*qq,*hd,*oxflux,*segc,*bcprfl,*bchd,*nodvar,*segvar,*qvtemp,*qvfac;
float **start,**scos,**ax,**cnode,**resisdiam,**resis,**bcp; //added April 2010
float **qv,**qt,**pv,**pev,**pt;
float **qvseg,**pvseg,**pevseg;

float *x,*y,*lseg,*ss,*cbar,*mtiss,*mptiss,*dqvsumdg0,*dqtsumdg0;
float *epsvessel,*epstissue,*eps,*errvessel,*errtissue,*pinit,*p;
float *rhs,*rhstest,*g0old,*ptt,*ptpt,*qtsum,*qvsum;
float **pvt,**pvprev,**qvprev,**cv,**dcdp,**tissparam;
float **ptprev,**ptv,**gamma1,**qcoeff1,**cv0,**conv0;
float **gvv,**end,**al,**alhd;
float ***rsta,***rend,***dtt;
float *xsl0,*xsl1,*xsl2,*clmin,*clint,*cl,**zv,***psl;
double **mat,**rhsg,*rhsl,*matx;

int main(int argc, char *argv[])
{
	int iseg,inod;

	input();

	is2d = 1; //set to 1 for 2d version, 0 otherwise
	// if(mzz == 1) is2d = 1; //assumes 2d version if all tissue points lie in one z-plane

	setuparrays0();

	setuparrays1(nseg,nnod);

	analyzenet();

	setuparrays2(nnv,nnt);

	std::cout << "Setuparray 2 finished." << std::endl;
	
	for(iseg=1; iseg<=nseg; iseg++) segvar[iseg] = segname[iseg];
	for(inod=1; inod<=nnod; inod++) nodvar[inod] = nodname[inod];
	picturenetwork(nodvar,segvar,"NetNodesSegs.ps");

	greens();

	for(iseg=1; iseg<=nseg; iseg++) segvar[iseg] = pvseg[iseg][1];
	for(inod=1; inod<=nnod; inod++) nodvar[inod] = nodname[inod];
	picturenetwork(nodvar,segvar,"NetNodesOxygen.ps");

	contour("Contour.ps");

	histogram();

	return 0;
}
