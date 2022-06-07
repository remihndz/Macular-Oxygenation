/*****************************************************
eval - Evaluate solute field from source strengths.  TWS November 07.
Version 3.0, May 17, 2011.
Modified to include non-diffusible solutes.  May 2010
******************************************************/
#include <math.h>
#include "nrutil.h"
#include <stdio.h>

void tissrate(int nsp, float *p, float *mtiss, float *mptiss);

float *eval(int slsegdiv, float req, float *x)
{
	extern int mxx,myy,mzz,nnt,nnv,nseg,nnod,nsp;
	extern int *mainseg,*segtyp,**tisspoints,*permsolute,*diffsolute,***nbou;
	extern int is2d; //needed for 2d version
	extern float pi1,fac,errfac;
	extern float *axt,*ayt,*azt,*ds,*diff,*g0,*y,*p,*mtiss,*mptiss,*pref;
	extern float **qt,**start,**scos,**qv,**ax,**pt;
	extern float w2d,r2d; //needed for 2d version

	float dist,gtt,gtv,lamx,lamy,lamz,err,dif;
	int i,j,k,ii,jj,kk,is,itp,isp,nmaxtissue=100,ktissue;
//initialize to g0
	for(isp=1; isp<=nsp; isp++) p[isp] = g0[isp];
//add contributions from tissue sources
	for(itp=1; itp<=nnt; itp++){
		dist = sqrt(SQR(x[1] - axt[tisspoints[1][itp]])
				  + SQR(x[2] - ayt[tisspoints[2][itp]])
				  + SQR(x[3] - azt[tisspoints[3][itp]]));
		if(dist <= req){
			if(is2d == 1) gtt = fac/w2d*(2.*log(r2d/req) + 1. - SQR(dist/req));
			else gtt = fac*(1.5 - 0.5*SQR(dist/req))/req;
		}
		else{
			if(is2d == 1) gtt = fac*2./w2d*log(r2d/dist);
			else gtt = fac/dist;
		}
		for(isp=1; isp<=nsp; isp++)	if(diffsolute[isp] == 1) p[isp] += gtt/diff[isp]*qt[itp][isp];
	}
//add contributions from vessel sources.  Subdivide subsegments.
//Note that vessel point is at midpoint of subsegment
	for(i=1; i<=nnv; i++){
		is = mainseg[i];
		for(k=1; k<=slsegdiv; k++){
			for(j=1; j<=3; j++)	y[j] = ax[j][i] + scos[j][is]*ds[is]*(-0.5 + (k-0.5)/slsegdiv);
			dist = sqrt(SQR(x[1] - y[1]) + SQR(x[2] - y[2]) + SQR(x[3] - y[3]));
			if(dist <= req){
				if(is2d == 1) gtv = fac/w2d*(2.*log(r2d/req) + 1. - SQR(dist/req));
				else gtv = fac*(1.5 - 0.5*SQR(dist/req))/req;
			}
			else{
				if(is2d == 1) gtv = fac*2./w2d*log(r2d/dist);
				else gtv = fac/dist;
			}
			for(isp=1; isp<=nsp; isp++)	if(permsolute[isp] == 1) p[isp] += gtv/diff[isp]*qv[i][isp]/slsegdiv;
		}
	}
//for non-diffusible solute, calculate by interpolating values at tissue points.  May 2010.
	for(isp=1; isp<=nsp; isp++)	if(diffsolute[isp] == 0){
		i = 0;
		while(x[1] > axt[i+1] && i < mxx) i++;
		if(i == 0) lamx = 1.;
		else if(i == mxx) lamx = 0.;
		else lamx = (x[1] - axt[i])/(axt[i+1] - axt[i]);
		j = 0;
		while(x[2] > ayt[j+1] && j < myy) j++;
		if(j == 0) lamy = 1.;
		else if(j == myy) lamy = 0.;
		else lamy = (x[2] - ayt[j])/(ayt[j+1] - ayt[j]);
		k = 0;
		while(x[3] > azt[k+1] && k < mzz) k++;
		if(k == 0) lamz = 1.;
		else if(k == mzz) lamz = 0.;
		else lamz = (x[3] - azt[k])/(azt[k+1] - azt[k]);
		p[isp] = 0.;
		for(ii=0; ii<=1; ii++) for(jj=0; jj<=1; jj++) for(kk=0; kk<=1; kk++)
			if(i+ii>=1 && i+ii<=mxx && j+jj>=1 && j+jj<=myy && k+kk>=1 && k+kk<=mzz){
				itp = nbou[i+ii][j+jj][k+kk];
				if(itp != 0) p[isp] += (1. - lamx + ii*(2.*lamx - 1))*(1. - lamy + jj*(2.*lamy - 1))
					*(1. - lamz + kk*(2.*lamz - 1))*pt[itp][isp];
			}
	}
//if possible, refine value using condition for rate = 0
	ktissue = 0;
	do{
		ktissue++;
		err = 0.;
		tissrate(nsp,p,mtiss,mptiss);
		for(isp=1; isp<=nsp; isp++)	if(diffsolute[isp] == 0){
			if(mptiss[isp] == 0.) printf("Error: mptiss[%i] = 0 at tissue point %i\n",isp,itp); 
			else dif = -mtiss[isp]/mptiss[isp];
			p[isp] += dif;
			err = FMAX(fabs(dif)/pref[isp],err);
		}
	}
	while(ktissue<=nmaxtissue && err > errfac);
	return p;
}