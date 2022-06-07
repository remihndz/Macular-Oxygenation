/******************************************************
input - reads input files.  TWS January 08
Note that input files refer to segname and nodname, but
all arrays use iseg and inod, which are assigned when
reading the file, as indices.
Note use of format "%*[^\n]" to read comment text of
unknown length.  From Tuhin Roy, Nov. 08.
Version 2.0, May 1, 2010.
Version 3.0, May 17, 2011.
*******************************************************/
#define _CRT_SECURE_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

void input(void)
{
	extern int max,nmaxvessel,nmaxtissue,nmax,nsl1,nsl2,slsegdiv;
	extern int mxx,myy,mzz,nnt,nnv,nseg,nnod,nsp,nnodbc,nodsegm;
	extern int rungreens,g0method,linmethod;
	extern int is2d; //needed for 2d version
	extern int **tisspoints,*permsolute,*bcnodname,*bcnod,*bctyp;
	extern int *segname,*segtyp,*nodname,*nspoint,*nl;
	extern int *oxygen,*diffsolute,*nresis; //added April 2010
	extern int **segnodname,**nodseg;

	extern float fn,alphab,p50,cs,q0,errfac,lowflowcrit/*,pcr,m0,gf0,gf1*/;
	extern float lb,maxl,v,vol,req,pi1,q0,alx,aly,alz;
	extern float xmax,ymax,scalefac;
	extern float *axt,*ayt,*azt,*ds,*g0,*diff,*pmin,*pmax,*pmean,*pref,*g0fac;
	extern float *diam,*q,*hd,*bcprfl,*bchd;
	extern float *x,*xsl0,*xsl1,*xsl2,*clmin,*clint,*p,*cl,**zv,***psl;
	extern float **start,**scos,**ax,**cnode,**bcp,**resisdiam,**resis,**tissparam;

	int i,iseg,isp,nlmax;
	float totalq;
	FILE *ifp;
	char bb[100];

	ifp = fopen("SoluteParams.dat", "r");
	fgets(bb,max,ifp);
	printf("%s\n",bb);
	fscanf(ifp,"%i %i %i%*[^\n]", &rungreens,&g0method,&linmethod);
	if(g0method != 1 && g0method != 2) printf("*** Error: soluteparams.dat, g0method must be 1 or 2\n");
	if(linmethod < 1 || linmethod > 3) printf("*** Error: soluteparams.dat, linmethod must be 1, 2 or 3\n");
	fscanf(ifp,"%i %i %i%*[^\n]", &nmaxvessel,&nmaxtissue,&nmax);
	fscanf(ifp,"%f%*[^\n]", &errfac);
	fscanf(ifp,"%f%*[^\n]", &lowflowcrit);
	fscanf(ifp,"%f%*[^\n]", &p50);
	fscanf(ifp,"%f%*[^\n]", &fn);
	fscanf(ifp,"%f%*[^\n]", &cs);
	fscanf(ifp,"%f%*[^\n]", &alphab);
	fscanf(ifp,"%f%*[^\n]", &q0);
	fscanf(ifp,"%i%*[^\n]", &nsp);
	permsolute = ivector(1,nsp);
	diffsolute = ivector(1,nsp);
	oxygen = ivector(1,nsp);
	pref = vector(1,nsp);
	diff = vector(1,nsp);
	g0 = vector(1,nsp);
	g0fac = vector(1,nsp);
	tissparam = matrix(1,3,1,nsp);
	for(isp=1; isp<=nsp; isp++){
		fgets(bb,max,ifp);
		fgets(bb,max,ifp);
		printf("%s\n",bb);
		fscanf(ifp,"%i %i %i%*[^\n]", &permsolute[isp],&diffsolute[isp],&oxygen[isp]);
		if(diffsolute[isp] != 0 && diffsolute[isp] != 1) printf("*** Error: soluteparams.dat, diffsolute[isp] must be 0 or 1\n");
		if(oxygen[isp] != 0 && oxygen[isp] != 1) printf("*** Error: soluteparams.dat, oxygen[isp] must be 0 or 1\n");
		if(oxygen[isp] == 1) permsolute[isp] = 1; //oxygen is permeable
		if(permsolute[isp] == 1) diffsolute[isp] = 1;  //permeable solutes must be diffusible
		fscanf(ifp,"%f%*[^\n]", &pref[isp]);
		fscanf(ifp,"%f%*[^\n]", &diff[isp]);
		diff[isp] = diff[isp]*1.e8;
		for(i=1; i<=3; i++)	fscanf(ifp,"%f%*[^\n]", &tissparam[i][isp]);
		fscanf(ifp,"%f%*[^\n]", &g0[isp]);
		fscanf(ifp,"%f%*[^\n]", &g0fac[isp]);
	}
	fclose(ifp);

//intravascular oxygen resistance data.  Assume zero unless specified in data file.
	resisdiam = matrix(1,20,1,nsp); //added April 2010
	resis = matrix(1,20,1,nsp); //added April 2010
	nresis = ivector(1,nsp);  //added April 2010
	ifp = fopen("IntravascRes.dat", "r");
	for(isp=1; isp<=nsp; isp++){
		fscanf(ifp, "%i", &nresis[isp]);
		if(nresis[isp] > 20) printf("Error: too many points in IntravascRes.dat, nresis = %i > 20\n",nresis[isp]);
		fgets(bb,max,ifp);
		if(nresis[isp] > 0){
			fgets(bb,max,ifp);
			for(i=1; i<=nresis[isp]; i++) fscanf(ifp,"%f %f", &resisdiam[i][isp],&resis[i][isp]);
		}
	}
	fclose(ifp);

//network data file
	ifp = fopen("Network.dat", "r");
	fgets(bb,max,ifp);
	printf("%s\n",bb);
//dimensions of box in microns; vertex must be at origin
	fscanf(ifp,"%f %f %f%*[^\n]", &alx,&aly,&alz);
	fscanf(ifp,"%i %i %i%*[^\n]", &mxx,&myy,&mzz);
	fscanf(ifp,"%f%*[^\n]", &lb);
	fscanf(ifp,"%f%*[^\n]", &maxl);
	fscanf(ifp,"%i%*[^\n]", &nodsegm);
//number of segments in vessel network
	fscanf(ifp,"%i%*[^\n]", &nseg);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
//segment properties: name type nodefrom nodeto diameter flow hematocrit
	segname = ivector(1,nseg);
	segtyp = ivector(1,nseg);
	segnodname = imatrix(1,2,1,nseg);
	diam = vector(1,nseg);
	q = vector(1,nseg);
	hd = vector(1,nseg);
	for(iseg=1; iseg<=nseg; iseg++) fscanf(ifp, "%i %i %i %i %f %f %f%*[^\n]",
		&segname[iseg],&segtyp[iseg],&segnodname[1][iseg],&segnodname[2][iseg],&diam[iseg],&q[iseg],&hd[iseg]);
//number of nodes in vessel network
	fscanf(ifp,"%i%*[^\n]", &nnod);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
//coordinates of nodes
	nodname = ivector(1,nnod);
	cnode = matrix(1,3,1,nnod);
	for(i=1; i<=nnod; i++) fscanf(ifp, "%i %f %f %f%*[^\n]", &nodname[i],&cnode[1][i],&cnode[2][i],&cnode[3][i]);
//boundary nodes
	fscanf(ifp,"%i%*[^\n]", &nnodbc);
	fgets(bb,max,ifp);
	fgets(bb,max,ifp);
	bcnodname = ivector(1,nnodbc);
	bcnod = ivector(1,nnodbc);
	bctyp = ivector(1,nnodbc);
	bcprfl = vector(1,nnodbc);
	bchd = vector(1,nnodbc);
	bcp = matrix(1,nnodbc,1,nsp);
	totalq = 0.;
	for(i=1; i<=nnodbc; i++){
		fscanf(ifp,"%i %i %f %f%", &bcnodname[i],&bctyp[i],&bcprfl[i],&bchd[i]);
		for(isp=1; isp<=nsp; isp++) if(permsolute[isp] == 1) fscanf(ifp,"%f",&bcp[i][isp]);
		fscanf(ifp,"%*[^\n]");	//ignore any 'extra' solutes in data file
		if(bctyp[i] == 2 && bcprfl[i] > 0.) totalq = totalq + bcprfl[i];
	}
	fclose(ifp);

//scale flows according to total inflow rate q0
	for(iseg=1; iseg<=nseg; iseg++) q[iseg] = q[iseg]*q0/totalq;
	for(i=1; i<=nnodbc; i++) if(bctyp[i] == 2) bcprfl[i] = bcprfl[i]*q0/totalq;
//v = total box volume, vol = volume represented by each tissue point; req = radius of equivalent sphere
	v = alx*aly*alz;
	vol = v/(mxx*myy*mzz);
	if(is2d == 1) req = pow(vol*1./alz/pi1,0.5);//2d version
	else req = pow(vol*0.75/pi1,0.333333);
//Read parameters for slice on which P is computed for contour plot
	nl = ivector(1,nsp);
	xsl0 = vector(1,3);
	xsl1 = vector(1,3);
	xsl2 = vector(1,3);
	clmin = vector(1,nsp);
	clint = vector(1,nsp);

	ifp = fopen("ContourParams.dat", "r");
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl0[1],&xsl0[2],&xsl0[3],&slsegdiv);
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl1[1],&xsl1[2],&xsl1[3],&nsl1);
	fscanf(ifp, "%f %f %f %i%*[^\n]", &xsl2[1],&xsl2[2],&xsl2[3],&nsl2);
	nlmax = 1;
	for(isp=1; isp<=nsp; isp++){
		fscanf(ifp, "%f %f %i%*[^\n]", &clmin[isp],&clint[isp],&nl[isp]);
		if(nl[isp] > nlmax) nlmax = nl[isp];
	}
	fclose(ifp);
	xmax = sqrt(SQR(xsl1[1]-xsl0[1]) + SQR(xsl1[2]-xsl0[2]) + SQR(xsl1[3]-xsl0[3]));
	ymax = sqrt(SQR(xsl2[1]-xsl0[1]) + SQR(xsl2[2]-xsl0[2]) + SQR(xsl2[3]-xsl0[3]));
	scalefac = FMIN(500./xmax,700./ymax);//updated April 2010
	cl = vector(1,nlmax);
	zv = matrix(1,nsl1,1,nsl2);
	psl = f3tensor(1,nsl1,1,nsl2,1,nsp);
}