/************************************************************************
Green's function approach for multiple reacting species.
T.W. Secomb July 2007 - based on Greens.f by R. Hsu.
See http://www.physiology.arizona.edu/people/secomb/greens.html

Variable array sizes and bounds, using Numerical Recipes utilities.
Tissue-vessel and tissue-tissue matrices computed on the fly to save memory.
No nondimensionalization.  Lengths, diameters in microns, times in s.
Flows in nanoliters/min
Oxygen concentrations in cm^3/cm^3
Consumption rates in cm^3/cm^3/s - changed in V2
Partial pressures in mmHg
Diffusivities in cm2/s, converted to micron^2/s

Special parameters for oxygen
p50, fn --- parameters in the Hill equation
cs --- red blood cell oxygen binding capacity in cm^3 O2/cm^3
alphab --- average solubility in blood in cm^3 O2/cm^3/mmHg
gamma1 --- intravascular resistance, varies with vessel diameter, in mmHg.cm.s/cm^3 O2

Main variables:
  gvv --- Green's function matrix for vessels
  mat --- matrix for vessel strengths
  al --- matrix giving dependence of vessel convective fluxes on source strengths
  lseg --- segment length
  ds --- subsegment length
  qv --- oxygen efflux from subsegment
  pv --- PO2 in the subsegment
  cv --- oxygen concentration in the subsegment
  qt --- tissue source strength
  pvt --- PO2 on vessels due to source terms in tissue
  ptv --- PO2 in tissue due to source terms on vessels
  q --- flow rate, qq = abs(q)

Version 2.0, May 2010.
With 9/08 updates.  New vessel-vesel interaction coefficient. January 2009
With alternate terms for 2D version.  May 2009
With g0 computed as part of linear system, for permeable solutes.  March 2010
  g0method = 1:  include g0 in linear system to be solved - fastest method *****
  g0method = 2:  theoretical estimates of dqsum/dg0 - was used in Version 1
For impermeable solutes, method 2 is always used 
With choice of Gauss-Jordan, LU or biconjugate gradient linear solvers.  March 2010
  linmethod = 1:  Gaussian elimination - was used in Version 1
  linmethod = 2:  LU decomposition
  linmethod = 3:  biconjugate gradient (iterative) - fastest method *****
Does not require that species 1 is oxygen. Allows for non-diffusing solutes.  April 2010
Creates log file. April 2010.
During tissue loop, scales vessel sources so that qvsum = qtsum, for faster convergence.  April 2010.
Modified for compatibility with Mac XCode compiler.  April 2010.
Includes intravascular resistance for all solutes.  May 2010.
Includes non-diffusible solutes.  May 2010.

Version 3.0, May 17, 2011.
Uses convect instead of genalpha and genalphahd.
This gives improved convergence if hematocrit is non-uniform in network
**************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"

void putrank(void);
void initgreens();
void blood(float c, float hem, float *p, float *pp);
float bloodconc(float p,float h);
void tissrate(int nsp, float *p, float *mtiss, float *mptiss);
void convect(int isp);		//new subroutine, August 2010, replaces genalpha and genalphahd
void testconvect(int isp);
float *eval(int slsegdiv, float req, float *x);

void gaussj(double **a, int n, double **b, int m);
void ludcmp(double **a, int n, int *indx, double *d);
void lubksb(double **a, int n, int *indx, double *b);
double bicgstab(double **a, double *b, double *x, int n, double eps, int itmax);

void greens(void)
{
	extern int nmaxvessel,nmaxtissue,nmax,g0method,linmethod;
	extern int mxx,myy,mzz,nnt,nnv,nseg,nsp,nnodbc;
	extern int is2d; //needed for 2d version
	extern int *mainseg,**tisspoints,*permsolute,*segtyp;
	extern int *segname,*nspoint,*istart,*nodout,*bcnod,*lowflow;//not bcnodname.  April 2010
	extern int *errvesselcount,*errtissuecount;
	extern int *imaxerrvessel,*imaxerrtissue,*nresis;  //added April 2010
	extern int *oxygen,*diffsolute; //added April 2010
	extern int **nodseg;
	extern int ***nbou;
	extern int *indx;//added March 2010

	extern float p50,cs,req,q0,fac,flowfac,lowflowcrit,errfac/*,pcr,m0*/;
	extern float v,vol,vdom,tlength,pi1;
	extern float alx,aly,alz,w2d,r2d; //needed for 2d version
	extern float *axt,*ayt,*azt,*ds,*diff,*pmin,*pmax,*pmean,*g0,*g0fac,*g0facnew;
	extern float *pref;//added March 2010
	extern float *diam,*rseg,*q,*qq,*hd,*bchd,*qvtemp,*qvfac;
	extern float *x,*lseg,*mtiss,*mptiss,*dqvsumdg0,*dqtsumdg0;
	extern float *epsvessel,*epstissue,*eps,*errvessel,*errtissue,*p;
	extern float *rhs,*rhstest,*g0old,*ptt,*ptpt,*qtsum,*qvsum;
	
	extern float **tissparam;
	extern float **start,**scos,**ax,**bcp;
	extern float **qv,**qt,**pv,**pev,**pt,**resisdiam,**resis;
	extern float **qvseg,**pvseg,**pevseg;
	extern float **pvt,**pvprev,**qvprev,**cv,**dcdp;
	extern float **ptprev,**ptv,**gamma1,**cv0,**conv0;
	extern float **gvv,**end,**al,**alhd;
	extern double **mat,**rhsg,*rhsl,*matx;//rhsl,matx added March 2010
	extern float ***dtt;

	int i,j,k,ix,iy,iz,jx,jy,jz,iseg,nt,ineg,ihigh,isp,imaxerr;
	int ixdiff,iydiff,izdiff,isn,jseg,kmain,ktissue,kvessel,itp,jtp,convflag,convflagt,convflagv;

	float dtmin,x11,x22,x33,duration,rsegmax,dsmax,gvarfac;
	float gtt,gtv,gvt,disp2,ds2,dist,d,de2,dtave,den,dqsumdg0;
	float dif,err,qhdcrit;
	float req1,lam,lam3d,lam2d;//modified 9/09

//May increase req to req1 to broaden Green's functions Gvt and Gtv.  October 2008
//This supresses oscillations that can otherwise occur in highly hypoxic regions.
	req1 = req;//*4.0;
	w2d = alz;  //needed for 2d
	r2d = sqrt(SQR(alx) + SQR(aly) + SQR(alz));
	lam3d = 0.5;	//Underrelax iteration of tissue levels
	lam2d = 0.1;	//Use this one for 2D permeable solutes only.  May 2010
	int bicgstabit = 2000; //parameter for biconjugate gradient method.  March 2010
	double dd,bicgstaberr = 0.0001; //parameter for biconjugate gradient method.  March 2010

	FILE *ofp,*ofp1;

	clock_t tstart, tfinish;

//setup mainseg (must be done after setuparrays2)
	for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5)		
		for(i=0; i<nspoint[iseg]; i++) mainseg[istart[iseg] + i] = iseg;
//identify vessel points
	for(i=1; i<=nnv; i++){
		iseg = mainseg[i];
		isn = i - istart[iseg];
		for(j=1; j<=3; j++)	ax[j][i] = start[j][iseg] + scos[j][iseg]*ds[iseg]*(isn+0.5);
	}
//index tissue points
	for(i=1; i<=mxx; i++) for(j=1; j<=myy; j++) for(k=1; k<=mzz; k++){
		nt = nbou[i][j][k];
		if(nt > 0){
			tisspoints[1][nt] = i;
			tisspoints[2][nt] = j;
			tisspoints[3][nt] = k;
		}
	}
//calculate the distance of tissue points to the nearest vessel
	dtave = 0.;
	for(itp=1; itp<=nnt; itp++){
		i = tisspoints[1][itp];
		j = tisspoints[2][itp];
		k = tisspoints[3][itp];
		dtmin = 1.e6;
		for(jseg=1; jseg<=nseg; jseg++) if(segtyp[jseg] == 4 || segtyp[jseg] == 5){
			x11 = (axt[i]-start[1][jseg])*scos[2][jseg]-(ayt[j]-start[2][jseg])*scos[1][jseg];
			x22 = (ayt[j]-start[2][jseg])*scos[3][jseg]-(azt[k]-start[3][jseg])*scos[2][jseg];
			x33 = (azt[k]-start[3][jseg])*scos[1][jseg]-(axt[i]-start[1][jseg])*scos[3][jseg];
			disp2 = SQR(x11) + SQR(x22) + SQR(x33);
			ds2 = SQR(axt[i]-start[1][jseg]) + SQR(ayt[j]-start[2][jseg]) + SQR(azt[k]-start[3][jseg]);
			de2 = SQR(axt[i]-end[1][jseg]) + SQR(ayt[j]-end[2][jseg]) + SQR(azt[k]-end[3][jseg]);
			if(FMAX(ds2,de2)-disp2 > SQR(lseg[jseg])) d = sqrt(FMIN(ds2,de2)) - rseg[jseg];
			else d = sqrt(disp2) - rseg[jseg];
			if(d < dtmin) dtmin = d;
		}
		dtave = dtave + dtmin;
	}
	dtave = dtave/nnt;
	vdom = nnt*vol;
	den = sqrt(vdom/tlength);
	printf("Average distance from tissue node to the nearest vessel = %f\n",dtave);
	printf("Sqrt(Tissue Volume/vessel length) = %f\n", den);
//Calculate intravascular or wall transport resistance.  Zero unless specified in IntravascRes.dat.
//If not oxygen, assume value from data is 1/(wall permeability in um/s)
	for(isp=1; isp<=nsp; isp++)	for(iseg=1; iseg<=nseg; iseg++)	gamma1[iseg][isp] = 0.;
	for(isp=1; isp<=nsp; isp++)	if(nresis[isp] != 0) for(iseg=1; iseg<=nseg; iseg++){
		gamma1[iseg][isp] = resis[1][isp];
		for(j=2; j<=nresis[isp]; j++) if(diam[iseg] <= resisdiam[j][isp] && diam[iseg] > resisdiam[j-1][isp])
			gamma1[iseg][isp] = resis[j-1][isp] + (resis[j][isp]-resis[j-1][isp])
				*(diam[iseg]-resisdiam[j-1][isp])/(resisdiam[j][isp]-resisdiam[j-1][isp]);
		if(diam[iseg] > resisdiam[nresis[isp]][isp]) gamma1[iseg][isp] = resis[nresis[isp]][isp];
		if(oxygen[isp] != 1) gamma1[iseg][isp] = gamma1[iseg][isp]/2./pi1/diam[isp];
	}
//vessel ~ vessel matrix elements gvv
//Uses empirical fit to results from elliptical integral form for diagonal elements, updated 2009
//if center of one segment lies within the other segment, calculate Gvv as for self-interaction term
//based on larger radius and larger length. (Jan. 08)
	for(i=1; i<=nnv; i++){
		iseg = mainseg[i];
		for(j=1; j<=nnv; j++){
			jseg = mainseg[j];
			rsegmax = FMAX(rseg[iseg],rseg[jseg]);
			dist = sqrt(SQR(ax[1][j] - ax[1][i]) + SQR(ax[2][j] - ax[2][i]) + SQR(ax[3][j] - ax[3][i]));
			if(dist < rsegmax){
				dsmax = FMAX(ds[iseg],ds[jseg]);
//Version 1.0 of 3-D self-interaction coefficients.  To use, declare float te;
//				te = log10(dsmax/2/rsegmax);
//				gvv[i][j] = (0.076836+te*(-0.0262899+te*(0.000354634+
//					te*(0.000362085+te*0.0000448842))))/rsegmax;
//Version 2.0 of 3-D interaction coefficients for close or coincident segments.  See Sep. 2009 notes.
//Interaction falls off faster with dist for short segments (small dsmax), slower for long segments
//Results differ noticeably from those obtained with version 1.0
				gvarfac = 0.6*exp(-0.45*dsmax/rsegmax);
//for distinct vessels close together, make distance rsegmax in following calculation, to improve convergence. TWS2011
				if(iseg != jseg) dist = rsegmax;
				gvv[i][j] = (1.298/(1. + 0.297*powf(dsmax/rsegmax,0.838)) - gvarfac*SQR(dist/rsegmax))*fac/rsegmax;
//for 2D version, additional terms give effect of boundaries (reflection method)
				if(is2d == 1){
					gvv[i][j] -= fac*2./w2d*0.926*SQR(1.-1./(1.+0.36*dsmax/w2d))*powf(1.+dsmax/w2d,0.27);
					gvv[i][j] += fac*2./w2d*(log(r2d/w2d + 0.5 + 0.27/r2d*w2d) - 0.117);
				}
			}
			else{
				if(is2d == 1) gvv[i][j] = fac*2./w2d*log(r2d/dist);
				else gvv[i][j] = fac/dist;
			}
		}
	}
// tissue ~ vessel, vessel ~ tissue: compute matrix elements gvt, gtv on fly as needed
// tissue ~ tissue: construct matrix of distances from a corner node
	dtt[1][1][1] = 1.0;
	for(jx=1; jx<=mxx; jx++) for(jy=1; jy<=myy; jy++) for(jz=1; jz<=mzz; jz++){
		dist = sqrt(SQR(axt[1]-axt[jx]) + SQR(ayt[1]-ayt[jy]) + SQR(azt[1]-azt[jz]));
		if (jx*jy*jz != 1){ 
			if(is2d == 1) dtt[jx][jy][jz] = fac*2./w2d*log(r2d/dist);
			else dtt[jx][jy][jz] = fac/dist;
		}
	}
//diagonal elements of tissue ~ tissue matrix gtt
	if(is2d == 1) gtt = fac/w2d*(2.*log(r2d/req)+0.5);
	else gtt = 1.2*fac/req;
//detect and label vessels with very low q*hd - updated April 2010 - test if oxygen is one of the solutes
	qhdcrit = 0.;
	for(isp=1; isp<=nsp; isp++) if(oxygen[isp] == 1) qhdcrit = lowflowcrit*tissparam[1][isp];
	for(iseg=1; iseg<=nseg; iseg++){
		lowflow[iseg] = 0;
		if(qq[iseg]*hd[iseg] < qhdcrit) lowflow[iseg] = 1;
	}
	initgreens();
	putrank();
//	for(isp=1; isp<=nsp; isp++){//for testing purposes only
//		convect(isp);
//		testconvect(isp);	
//	}

//create log file
	ofp1 = fopen("GreensLog.txt", "w");
	fprintf(ofp1,"GreensLog.txt\n");
	fclose(ofp1);
	tstart = clock();
//********************** start of main loop *****************************
	for(kmain=1; kmain<=nmax; kmain++){
		printf("\n----- kmain = %i -----\n",kmain);
		for(isp=1; isp<=nsp; isp++){
			if(diffsolute[isp] == 1){
				for(itp=1; itp<=nnt; itp++)	ptprev[itp][isp] = pt[itp][isp];
				for(i=1; i<=nnv; i++) pvprev[i][isp] = pv[i][isp];
			}
			g0old[isp] = g0[isp];
		}
//********************** start of vessel loop *****************************
//compute contribution pvt from tissue source strengths qt
		for(i=1; i<=nnv; i++){
			for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1) {
				if(g0method == 1 && permsolute[isp] == 1) pvt[i][isp] = 0.;
				else pvt[i][isp] = g0[isp];
			}
			for(itp=1; itp<=nnt; itp++){
				dist = sqrt(SQR(ax[1][i] - axt[tisspoints[1][itp]])
					+ SQR(ax[2][i] - ayt[tisspoints[2][itp]])
					+ SQR(ax[3][i] - azt[tisspoints[3][itp]]));
				if(dist <= req1){
					if(is2d == 1) gvt = fac/w2d*(2.*log(r2d/req1) + 1. - SQR(dist/req1));
					else gvt = fac*(1.5 - 0.5*SQR(dist/req1))/req1;
				}
				else{
					if(is2d == 1) gvt = fac*2./w2d*log(r2d/dist);
					else gvt = fac/dist;
				}
				for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1) pvt[i][isp] += gvt/diff[isp]*qt[itp][isp];//May 2010 permsolute
			}
		}
//compute blood solute levels and PO2
		for(kvessel=1; kvessel<=nmaxvessel; kvessel++){
			convflagv = 1;
			for(isp=1; isp<=nsp; isp++){
				qvsum[isp] = 0.;
				dqvsumdg0[isp] = 0.;
				if(permsolute[isp] == 1){
					ineg = 0;
					ihigh = 0;
					convect(isp);		//new subroutine, August 2010
					for(i=1; i<=nnv; i++){
						iseg = mainseg[i];
						qvprev[i][isp] = qv[i][isp];
						if(oxygen[isp] == 1){
							if(lowflow[iseg] != 1){//only do this if not a lowflow segment.  June 2009.
								if(cv[i][isp] < 0.){
									ineg += 1;
									if(ineg == 1) printf("*** Warning: cblood is negative -%i",segname[iseg]);
									if(ineg > 1) printf("-%i",segname[iseg]);
								}
								if(cv[i][isp] > bloodconc(150.,hd[iseg])){	//August 17, 2010
									ihigh += 1;
									if(ihigh == 1) printf("*** Warning: cblood is high +%i",segname[iseg]);
									if(ihigh > 1) printf("+%i",segname[iseg]);
								}
								blood(cv[i][isp],hd[iseg],&pv[i][isp],&dcdp[i][isp]);
							}
						}
						else{
							pv[i][isp] = cv[i][isp];
							dcdp[i][isp] = 1.;
						}
					}
					if(ineg > 0 || ihigh > 0) printf("\n");
//generate linear system to be solved
					for(i=1; i<=nnv; i++){
						iseg = mainseg[i];
						rhs[i] = pv[i][isp] - pvt[i][isp];
						for(j=1; j<=nnv; j++){
							jseg = mainseg[j];
							mat[i][j] = gvv[i][j]/diff[isp] + al[i][j]/dcdp[i][isp]/qq[iseg]/flowfac;
							if(i == j) mat[i][j] += gamma1[iseg][isp]/ds[iseg];
							rhs[i] += al[i][j]*qv[j][isp]/dcdp[i][isp]/qq[iseg]/flowfac;
							if(oxygen[isp] == 1 && lowflow[mainseg[i]] == 1){  //low q*hd
								if(i == j) mat[i][j] = 1.;
								else mat[i][j] = 0.;
							}
						}
						if(oxygen[isp] == 1 && lowflow[iseg] == 1) rhs[i] = qvprev[i][isp];  //low q*hd
					}
//solve system of linear algebraic equations: Sum mat[i][j]*qv[j]= rhs[i]
					if(g0method == 1){
						for(i=1; i<=nnv; i++){
							if(oxygen[isp] == 1 && lowflow[mainseg[i]] == 1) mat[i][nnv+1] = 0.;  //low q*hd
							else mat[i][nnv+1] = 1.;
							mat[nnv+1][i] = 1.;
							mat[nnv+1][nnv+1] = 0.;
						}
						if(linmethod == 1){
							for(i=1; i<=nnv; i++) rhsg[i][1] = rhs[i];
							rhsg[nnv+1][1] = -qtsum[isp];
							gaussj(mat, nnv+1, rhsg, 1);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = rhsg[i][1];
								qvsum[isp] += qv[i][isp];
							}
							g0[isp] = rhsg[nnv+1][1];
						}
						if(linmethod == 2){
							ludcmp(mat, nnv+1, indx, &dd);
							for(i=1; i<=nnv; i++) rhsl[i] = rhs[i];
							rhsl[nnv+1] = -qtsum[isp];
							lubksb(mat, nnv+1, indx, rhsl);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = rhsl[i];
								qvsum[isp] += qv[i][isp];
							}
							g0[isp] = rhsl[nnv+1];
						}
						if(linmethod == 3){
							for(i=1; i<=nnv; i++) rhsl[i] = rhs[i];
							rhsl[nnv+1] = -qtsum[isp];
							for(i=1; i<=nnv; i++) matx[i] = qv[i][isp];
							matx[nnv+1] = g0[isp];
							bicgstab(mat, rhsl, matx, nnv+1, bicgstaberr, bicgstabit);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = matx[i];
								qvsum[isp] += qv[i][isp];
							}
							g0[isp] = matx[nnv+1];
						}
					}
					if(g0method == 2){
						if(linmethod == 1){
							for(i=1; i<=nnv; i++){
								rhsg[i][1] = rhs[i];
								rhsg[i][2] = -1.;
							}
							gaussj(mat, nnv, rhsg, 2);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = rhsg[i][1];
								qvsum[isp] += qv[i][isp];
								if(oxygen[isp] != 1 || lowflow[mainseg[i]] != 1) dqvsumdg0[isp] += rhsg[i][2];
							}
						}
						if(linmethod == 2){
							ludcmp(mat, nnv, indx, &dd);
							for(i=1; i<=nnv; i++) rhsl[i] = rhs[i];
							lubksb(mat, nnv, indx, rhsl);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = rhsl[i];
								qvsum[isp] += qv[i][isp];
							}
							for(i=1; i<=nnv; i++) rhsl[i] = -1.;
							lubksb(mat, nnv, indx, rhsl);
							for(i=1; i<=nnv; i++)
								if(oxygen[isp] != 1 || lowflow[mainseg[i]] != 1) dqvsumdg0[isp] += rhsl[i];
						}
						if(linmethod == 3){
							for(i=1; i<=nnv; i++){
								rhsl[i] = rhs[i];
								matx[i] = qv[i][isp];
							}
							bicgstab(mat, rhsl, matx, nnv, bicgstaberr, bicgstabit);
							for(i=1; i<=nnv; i++){
								qv[i][isp] = matx[i];
								qvsum[isp] += qv[i][isp];
							}
							for(i=1; i<=nnv; i++){
								rhsl[i] = -1.;
								matx[i] = 0.;
							}
							bicgstab(mat, rhsl, matx, nnv, bicgstaberr, bicgstabit);
							for(i=1; i<=nnv; i++)
								if(oxygen[isp] != 1 || lowflow[mainseg[i]] != 1) dqvsumdg0[isp] += matx[i];
						}
					}
//for low q*hd segments, calculate efflux based on change in extravascular oxygen level
//save values in qvtemp to avoid influence on eval, update qv, underrelax - July 2008
					for(i=1; i<=nnv; i++){
						iseg = mainseg[i];
						if(oxygen[isp] == 1 && lowflow[iseg] == 1){
							for(j=1; j<=3; j++) x[j] = ax[j][i] - 0.5*scos[j][iseg]*ds[iseg];
							p = eval(1,req,x);
							p[isp] = FMAX(p[isp],0.);
							pv[i][isp] = p[isp]/2.;   //added August 2009
							qvtemp[i] = q[iseg]*flowfac*bloodconc(p[isp],hd[iseg]); //q here (not qq) April 2008
							for(j=1; j<=3; j++) x[j] = ax[j][i] + 0.5*scos[j][iseg]*ds[iseg];
							p = eval(1,req,x);
							p[isp] = FMAX(p[isp],0.);
							pv[i][isp] += p[isp]/2.;  //added August 2009
							qvtemp[i] -= q[iseg]*flowfac*bloodconc(p[isp],hd[iseg]);
						}
					}
					for(i=1; i<=nnv; i++) if(oxygen[isp] == 1 && lowflow[mainseg[i]] == 1)
						qv[i][isp] = 0.5*qvtemp[i] + 0.5*qvprev[i][isp];
					errvessel[isp] = 0.;
					imaxerr = 0;
					errvesselcount[isp] = 0; //added June 2009
					for(i=1; i<=nnv; i++){
						dif = qv[i][isp] - qvprev[i][isp];
//If qv is large, use relative rather than absolute error  - May 2008
						if(qv[i][isp] != 0.) dif = dif*FMIN(1.,epsvessel[isp]/errfac/fabs(qv[i][isp]));
						if(fabs(dif) > errvessel[isp]){
							imaxerrvessel[isp] = mainseg[i];
							errvessel[isp] = fabs(dif);
						}
						if(fabs(dif) > epsvessel[isp]) errvesselcount[isp] += 1;
					}
					printf("Solute %i: qtsum = %f, qvsum = %f\n",isp,qtsum[isp],qvsum[isp]);
					printf("Solute %i: kvessel = %i, errvessel_q = %f, imaxerr = %i, g0 = %f\n",
						isp,kvessel,errvessel[isp],imaxerrvessel[isp],g0[isp]);
					if(errvesselcount[isp] > 0) convflagv = 0;
				}
			}
			if(convflagv == 1) goto vesselconv;
		}
		for(isp=1; isp<=nsp; isp++) if(errvesselcount[isp] > 0) 
			printf("*** Warning: solute %i, %i vessel source strengths not converged\n",
			isp,errvesselcount[isp]);
		vesselconv:;
//********************** end of vessel loop *****************************	
//********************** start of tissue loop *****************************
//Compute tissue source strengths iteratively by successive relaxation: updated qt values are immediately used.
//Continually scales up qv values so that their sum equals updated sum of qt values.  Added April 2010.
//contribution ptv from vessel source strengths qv
		for(itp=1; itp<=nnt; itp++){
			for(isp=1; isp<=nsp; isp++) ptv[itp][isp] = 0.;
			for(i=1; i<=nnv; i++){
				dist = sqrt(SQR(ax[1][i] - axt[tisspoints[1][itp]])
					+ SQR(ax[2][i] - ayt[tisspoints[2][itp]]) + SQR(ax[3][i] - azt[tisspoints[3][itp]]));
				if(dist <= req1){
					if(is2d == 1) gtv = fac/w2d*(2.*log(r2d/req1) + 1. - SQR(dist/req1));
					else gtv = fac*(1.5 - 0.5*SQR(dist/req1))/req1;
				}
				else{
					if(is2d == 1) gtv = fac*2./w2d*log(r2d/dist);
					else gtv = fac/dist;
				}
				for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1) ptv[itp][isp] += gtv/diff[isp]*qv[i][isp];
			}
		}
		for(isp=1; isp<=nsp; isp++) qvfac[isp] = 1.;
		for(ktissue=1; ktissue<=nmaxtissue; ktissue++){
//Scale all av, qvsum and ptv values so that qvsum = qtsum.  April 2010.
			for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1 && g0method == 1){
				qvfac[isp] = -qtsum[isp]/qvsum[isp];
				if(fabs(qvfac[isp]) > 2.) qvfac[isp] = 1.;  //avoid extreme values
				if(fabs(qvfac[isp]) < 0.5) qvfac[isp] = 1.;  //avoid extreme values
			}
			convflagt = 1;
			for(isp=1; isp<=nsp; isp++){
				qtsum[isp] = 0;
				errtissue[isp] = 0.;
				dqtsumdg0[isp] = 0.;
				errtissuecount[isp] = 0; //added June 2009
			}
//contribution ptt from tissue source strengths qt
			for(itp=1; itp<=nnt; itp++){
				ix = tisspoints[1][itp];
				iy = tisspoints[2][itp];
				iz = tisspoints[3][itp];
				for(isp=1; isp<=nsp; isp++)	ptt[isp] = 0.;//all solutes
				for(jtp=1; jtp<=nnt; jtp++){
					jx = tisspoints[1][jtp];
					jy = tisspoints[2][jtp];
					jz = tisspoints[3][jtp];
					ixdiff = abs(ix - jx) + 1;
					iydiff = abs(iy - jy) + 1;
					izdiff = abs(iz - jz) + 1;
					for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1){
						if(ix == jx && iy == jy && iz == jz) ptt[isp] += gtt/diff[isp]*qt[jtp][isp];
						else ptt[isp] += dtt[ixdiff][iydiff][izdiff]/diff[isp]*qt[jtp][isp];
					}
				}
				for(isp=1; isp<=nsp; isp++){
					if(is2d == 1 && permsolute[isp] == 1) lam = lam2d;
					else lam = lam3d;
					if(diffsolute[isp] == 1) pt[itp][isp] = (1.-lam)*pt[itp][isp]
						+ lam*(ptv[itp][isp]*qvfac[isp] + g0[isp] + ptt[isp]);//underrelaxation
					ptpt[isp] = pt[itp][isp];
				}
				tissrate(nsp,ptpt,mtiss,mptiss);
				for(isp=1; isp<=nsp; isp++){  //replace qt with value based on updated pt - all solutes
					dif = mtiss[isp]*vol - qt[itp][isp];
					qt[itp][isp] += dif;
					qtsum[isp] += qt[itp][isp];
					dqtsumdg0[isp] += mptiss[isp]*vol;
					if(diffsolute[isp] == 0){	//non-diffusible - use Newton method to solve for pt.  May 2010.
						if(mptiss[isp] == 0.) printf("Error: mptiss[%i] = 0 at tissue point %i\n",isp,itp); 
						else pt[itp][isp] -= mtiss[isp]/mptiss[isp];
					}
					if(fabs(dif) > errtissue[isp]){
						errtissue[isp] = fabs(dif);
						imaxerrtissue[isp] = itp;
					}
					if(fabs(dif) > epstissue[isp]) errtissuecount[isp] += 1;
				}
			}
			for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1){
				printf("Solute %i: qtsum = %f, qvsum = %f\n",isp,qtsum[isp],qvsum[isp]*qvfac[isp]);//May 2010
				printf("Solute %i: ktissue = %i, errtissue_q = %f, imaxerr = %i, g0 = %f\n",
					isp,ktissue,errtissue[isp],imaxerrtissue[isp],g0[isp]);
				if(errtissuecount[isp] > 0) convflagt = 0;
			}
			if(kmain > 1 && convflagt == 1) goto tissueconv;  //force full number of iterations when kmain = 1.  May 2010
		}
		for(isp=1; isp<=nsp; isp++) if(errtissuecount[isp] > 0) 
			printf("*** Warning: solute %i, %i tissue source strengths not converged\n",isp,errtissuecount[isp]);
		tissueconv:;
//Print log file.  April 2010
		ofp1 = fopen("GreensLog.txt", "a");
		kvessel = IMIN(kvessel,nmaxvessel);
		ktissue = IMIN(ktissue,nmaxtissue);
		fprintf(ofp1,"\n----- kmain = %i, kvessel = %i, ktissue = %i -----\n",kmain,kvessel,ktissue);
		for(isp=1; isp<=nsp; isp++){
			if(diffsolute[isp] == 1) fprintf(ofp1,"Solute %i: qtsum = %f, qvsum = %f, g0 = %f\n",
				isp,qtsum[isp],qvsum[isp]*qvfac[isp],g0[isp]);
			if(permsolute[isp] == 1) fprintf(ofp1,"Solute %i: errvessel_q = %f, imaxerr = %i\n",
				isp,errvessel[isp],segname[imaxerrvessel[isp]]);
			if(diffsolute[isp] == 1) fprintf(ofp1,"Solute %i: errtissue_q = %f, imaxerr = %i\n",
				isp,errtissue[isp],imaxerrtissue[isp]);
		}
		fclose(ofp1);
//********************** end of tissue loop *****************************
//Update g0.  If permsolute[isp] != 1, always use method 2.
//Method 2 is based on derivative wrt g0 - new version September 2009 - automatic estimation of g0fac
		for(isp=1; isp<=nsp; isp++) g0facnew[isp] = 0.;
		for(itp=1; itp<=nnt; itp++){
			for(isp=1; isp<=nsp; isp++) ptpt[isp] = pt[itp][isp];
			tissrate(nsp,ptpt,mtiss,mptiss);
			ix = tisspoints[1][itp];
			iy = tisspoints[2][itp];
			iz = tisspoints[3][itp];
			for(jtp=1; jtp<=nnt; jtp++){
				jx = tisspoints[1][jtp];
				jy = tisspoints[2][jtp];
				jz = tisspoints[3][jtp];
				ixdiff = abs(ix - jx) + 1;
				iydiff = abs(iy - jy) + 1;
				izdiff = abs(iz - jz) + 1;
				for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1){
					if(ix == jx && iy == jy && iz == jz) g0facnew[isp] += gtt/diff[isp]*mptiss[isp]*vol;
					else g0facnew[isp] += dtt[ixdiff][iydiff][izdiff]/diff[isp]*mptiss[isp]*vol;
				}
			}
		}
		for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1 && (g0method == 2 || permsolute[isp] == 0)){
			g0facnew[isp] = 1./(1. - g0facnew[isp]/nnt);
			dqsumdg0 = FMIN(dqvsumdg0[isp],0.) + FMIN(dqtsumdg0[isp],0.)*g0facnew[isp];
			if(fabs(dqsumdg0) > 1.e-6){
				dif = (qvsum[isp] + qtsum[isp])/dqsumdg0*g0fac[isp];//This g0fac should normally be 1.0.  September 2009
				g0[isp] -= dif;
			}
		}
//Convergence based on changes in pv, pt and g0
		convflag = 1;
		printf("\n");
		for(isp=1; isp<=nsp; isp++){
			err = 0.;
			imaxerr = 0;
			if(permsolute[isp] == 1) for(i=1; i<=nnv; i++){
				dif = fabs(pv[i][isp] - pvprev[i][isp]);
				if(dif > err){
					imaxerr = mainseg[i];
					err = dif;
				}
			}
			errvessel[isp] = err;
			imaxerrvessel[isp] = imaxerr;
			err = 0.;
			imaxerr = 0;
			if(diffsolute[isp] == 1) for(itp=1; itp<=nnt; itp++){
				dif = fabs(pt[itp][isp] - ptprev[itp][isp]);
				if(dif > err){
					imaxerr = itp;
					err = dif;
				}
			}
			errtissue[isp] = err;
			imaxerrtissue[isp] = imaxerr;
			if(errvessel[isp] > err){
				imaxerr = imaxerrvessel[isp];
				err = errvessel[isp];
			}
			else imaxerr = -imaxerr;
			dif = fabs(g0[isp] - g0old[isp]);
			if(dif > err){
				imaxerr = 0;
				err = dif;
			}
			printf("Solute %i: err = %f, imaxerr = %i (- for tissue point)\n",isp,err,imaxerr);
			if(err > eps[isp]) convflag = 0;
		}
//Print log file - April 2010
		ofp1 = fopen("GreensLog.txt", "a");
		for(isp=1; isp<=nsp; isp++){
			if(permsolute[isp] == 1) fprintf(ofp1,"Solute %i: errvessel_p = %f, imaxerr = %i\n",
				isp,errvessel[isp],segname[imaxerrvessel[isp]]);
			fprintf(ofp1,"Solute %i: errtissue_p = %f, imaxerr = %i\n",
				isp,errtissue[isp],imaxerrtissue[isp]);
		}
		fclose(ofp1);
		if(convflag == 1 && convflagv == 1 && convflagt == 1) goto mainconv;
	}
	printf("*** Warning: tissue or vessel solute levels not converged\n");
	mainconv:;
//********************** end of main loop *****************************
	tfinish = clock();
	duration = (float)(tfinish - tstart)/CLOCKS_PER_SEC;
	printf("\n%i iterations, %2.1f seconds for main loop\n", kmain,duration);
	ofp1 = fopen("GreensLog.txt", "a");
	fprintf(ofp1,"\n%i iterations, %2.1f seconds for main loop\n", kmain,duration);
	fclose(ofp1);
//Scale all qv values so that qvsum = qtsum.  April 2010.
	for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1 && g0method == 1){
		qvsum[isp] *= qvfac[isp];
		for(i=1; i<=nnv; i++) qv[i][isp] *= qvfac[isp];
	}
//general output file, not read by readsources
	ofp = fopen("GreensRes.out", "w");
	fprintf(ofp,"%i %i %i %i %i %i\n", nnv, nseg, mxx, myy, mzz, nnt);
	fprintf(ofp,"Total flow rate into region q0 = %f nl/min\n", q0);
	for(isp=1; isp<=nsp; isp++) fprintf(ofp,"g0[%i] = %f\n", isp,g0[isp]);
	fprintf(ofp,"\n");
//extravascular solute levels
	for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1){
		fprintf(ofp,"\nSolute %i\n", isp);
		fprintf(ofp,"Segment");
		if(permsolute[isp] == 1) fprintf(ofp,"Efflux Pvessel Ptissue Cvessel");
		fprintf(ofp,"\n");
		for(i=1; i<=nnv; i++){
			pev[i][isp] = pvt[i][isp];
			if(g0method == 1 && permsolute[isp] == 1) pev[i][isp] += g0[isp];
			if(permsolute[isp] == 1) for(j=1; j<=nnv; j++) pev[i][isp] += gvv[i][j]*qv[j][isp]/diff[isp];
		}
		for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
			qvseg[iseg][isp] = 0.;
			pevseg[iseg][isp] = 0.;
			pvseg[iseg][isp] = 0.;
		}
		for(i=1; i<=nnv; i++){
			iseg = mainseg[i];
			if(permsolute[isp] == 1) fprintf(ofp,"%4i %4i %10.4f %10.4f %10.4f %10.4f\n",
				 i,iseg,qv[i][isp],pv[i][isp],pev[i][isp],cv[i][isp]);
			qvseg[iseg][isp] += qv[i][isp];
			pevseg[iseg][isp] += pev[i][isp]/nspoint[iseg];
			pvseg[iseg][isp] += pv[i][isp]/nspoint[iseg];
		}
		fprintf(ofp,"Solute %i: qtsum = %f, qvsum = %f\n",isp,qtsum[isp],qvsum[isp]);
	}
	for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1){
		fprintf(ofp,"Solute %i: segment length pvseg pevseg qvseg gamma\n", isp);
		for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5)
			fprintf(ofp,"%4i %10.4f %10.4f %10.4f %10.4f %10.4f\n",
			segname[iseg],lseg[iseg],pvseg[iseg][isp],pevseg[iseg][isp],qvseg[iseg][isp],gamma1[iseg][isp]);
	}
	fclose(ofp);

//Write output files that allow restart of program without running greens
//These files are read by readsources
	ofp = fopen("TissueSources.out", "w");
	fprintf(ofp,"%i %i %i %i %f\n", mxx,myy,mzz,nnt,req);
	fprintf(ofp,"X, Y, Z coords of source points");
	for(i=1; i<=mxx; i++){
		if(i%10 == 1) fprintf(ofp,"\n");
		fprintf(ofp," %g", axt[i]);
	}
	for(i=1; i<=myy; i++){
		if(i%10 == 1) fprintf(ofp,"\n");
		fprintf(ofp," %g", ayt[i]);
	}
	for(i=1; i<=mzz; i++){ 
		if(i%10 == 1) fprintf(ofp,"\n");
		fprintf(ofp," %g", azt[i]);
	}
	fprintf(ofp,"\nTissue point xyz indices");
	for(itp=1; itp<=nnt; itp++){
		if(itp%10 == 1) fprintf(ofp,"\n");
		fprintf(ofp,"%4i %4i %4i", tisspoints[1][itp],tisspoints[2][itp],tisspoints[3][itp]);
	}
	for(isp=1; isp<=nsp; isp++) if(diffsolute[isp] == 1) {
		fprintf(ofp,"\n%g = g0, source strengths for solute %i",g0[isp],isp);
		for(i=1; i<=nnt; i++){
			if(i%10 == 1) fprintf(ofp,"\n");
			fprintf(ofp," %10g", qt[i][isp]);
		}
	}
	fclose(ofp);

	ofp = fopen("VesselSources.out", "w");
	fprintf(ofp,"%i %i\n", nseg,nnv);
	fprintf(ofp,"Segment start coords, direction cosines, length\n");
	for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5)
		fprintf(ofp,"%10g %10g %10g %10g %10g %10g %10g\n", 
		start[1][iseg],start[2][iseg],start[3][iseg],scos[1][iseg],scos[2][iseg],scos[3][iseg],ds[iseg]);
	fprintf(ofp,"Source point coords");
	for(i=1; i<=nnv; i++){
		if(i%3 == 1) fprintf(ofp,"\n");
		fprintf(ofp,"  %10g %10g %10g", ax[1][i],ax[2][i],ax[3][i]);
	}
	fprintf(ofp,"\nMain segment numbers of subsegments");
	for(i=1; i<=nnv; i++){
		if(i%20 == 1) fprintf(ofp,"\n");
		fprintf(ofp," %i", mainseg[i]);
	}
	for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1){
		fprintf(ofp,"\nSource strengths for solute %i",isp);
		for(i=1; i<=nnv; i++){
			if(i%10 == 1) fprintf(ofp,"\n");
			fprintf(ofp," %10g", qv[i][isp]);
		}
	}
	fclose(ofp);

	ofp = fopen("TissueLevels.out", "w");
	for(isp=1; isp<=nsp; isp++){
		pmax[isp] = -1.e8;
		pmean[isp] = 0.;
		pmin[isp] = 1.e8;
		fprintf(ofp,"Solute %i",isp);
		for(itp=1; itp<=nnt; itp++){
			if(itp%10 == 1) fprintf(ofp,"\n");
			fprintf(ofp,"%12f ", pt[itp][isp]);
			pmean[isp] += pt[itp][isp];
			pmax[isp] = FMAX(pt[itp][isp],pmax[isp]);
			pmin[isp] = FMIN(pt[itp][isp],pmin[isp]);
		}
		pmean[isp] = pmean[isp]/nnt;
		fprintf(ofp,"\n%f %f %f Solute %i: pmean, pmin, pmax\n", pmean[isp],pmin[isp],pmax[isp],isp);
	}
	fclose(ofp);

	ofp = fopen("VesselLevels.out", "w");
	fprintf(ofp,"lowflow");
	i = 0;
	for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
		i++;
		if(i%50 == 1) fprintf(ofp,"\n");
		fprintf(ofp,"%2i",lowflow[iseg]);
	}
	fprintf(ofp,"\n");
	for(isp=1; isp<=nsp; isp++) {
		i = 0;
		fprintf(ofp,"Solute %i pvseg", isp);
		for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
			i++;
			if(i%10 == 1) fprintf(ofp,"\n");
			fprintf(ofp,"%10.4f ",pvseg[iseg][isp]);
		}
		fprintf(ofp,"\n");
		fprintf(ofp,"Solute %i pevseg", isp);
		i = 0;
		for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
			i++;
			if(i%10 == 1) fprintf(ofp,"\n");
			fprintf(ofp,"%10.4f ",pevseg[iseg][isp]);
		}
		fprintf(ofp,"\n");
	}
	fclose(ofp);
}