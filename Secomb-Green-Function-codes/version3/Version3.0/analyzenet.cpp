/************************************************************************
analyzenet - for Greens07_3D.  TWS October 07
Set up nodtyp, nodseg, nodnod arrays based on flowing segments
Nodes with no flowing segments or with fixed pressure are assigned nodtyp = 0
These values are subsequently overridden by putrank.
Version 3.0, May 17, 2011.
************************************************************************/
#define _CRT_SECURE_NO_DEPRECATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "nrutil.h"

int outboun(int method);

void analyzenet()
{
	extern int nseg,nnod,nnodbc,nnv,nnt,mxx,myy,mzz,nodsegm;
	extern int *bcnodname,*bcnod,*bctyp,*nodtyp,*segtyp,*nspoint,*nodname,*ista,*iend,*istart;
	extern int **segnodname,**nodnod,**nodseg;
	extern int ***nbou;
	extern float pi1,maxl,tlength,lb,alx,aly,alz;
	extern float tlengthq,tlengthqhd;//added 8/09, 8/10
	extern float *diam,*lseg,*rseg,*ds,*ss,*q,*qq,*axt,*ayt,*azt,*hd;
	extern float **cnode,**start,**end,**scos;
	extern float ***rsta,***rend;

	int i,j,k,iseg,inod,inod1,inod2,inodbc,m;
	float sintheta,sinfi,cosfi,t,delx,dely,delz;

//Find node numbers corresponding to segment nodes - for all segments
	for(inod=1; inod<=nnod; inod++) nodtyp[inod] = 0;
	for(iseg=1; iseg<=nseg; iseg++)	{
		for(inod=1; inod<=nnod; inod++) if(nodname[inod] == segnodname[1][iseg]){
			ista[iseg] = inod;
			goto foundit1;
		}
		printf("*** Error: No matching node found for nodname %i\n", segnodname[1][iseg]);
		//printf("*** segnodname[1][iseg=%i]=%i and nodname[1]=%i\n", iseg, segnodname[1][iseg], nodname[1]);
		//for(inod=1; inod<=nnod; inod++) printf("nodname[%i]=%i\n", inod, nodname[inod]);
		foundit1:;
		for(inod=1; inod<=nnod; inod++) if(nodname[inod] == segnodname[2][iseg]){
			iend[iseg] = inod;
			goto foundit2;
		}
		printf("*** Error: No matching node found for nodname %i\n", segnodname[2][iseg]);
		printf("*** Total number of nodes is %i\n", nnod);
		foundit2:;
	}
//Setup nodtyp, nodseg and nodnod
	for(iseg=1; iseg<=nseg; iseg++)	if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
		inod1 = ista[iseg];
		inod2 = iend[iseg];
		nodtyp[inod1] = nodtyp[inod1] + 1;
		nodtyp[inod2] = nodtyp[inod2] + 1;
		if(nodtyp[inod1] > nodsegm) printf("*** Error: Too many segments connected to node %i\n", inod1);
		if(nodtyp[inod2] > nodsegm) printf("*** Error: Too many segments connected to node %i\n", inod2);
		nodseg[nodtyp[inod1]][inod1] = iseg;
		nodseg[nodtyp[inod2]][inod2] = iseg;
		nodnod[nodtyp[inod1]][inod1] = inod2;
		nodnod[nodtyp[inod2]][inod2] = inod1;
	}
//Find node numbers corresponding to boundary nodes
	for (inodbc=1; inodbc<=nnodbc; inodbc++){
		for(inod=1; inod<=nnod; inod++) if(nodname[inod] == bcnodname[inodbc]){
			bcnod[inodbc] = inod;
			if(nodtyp[inod] != 1) printf("*** Error: Boundary node %i is not a 1-segment node, rather its a nodtype[%i]=%i\n", inod, inod, nodtyp[inod]);
			goto foundit;
		}
		printf("*** Error: No matching node found for nodname %i\n", bcnodname[inodbc]);
		foundit:;
	}
//start[k][iseg] = coordinates of starting point of segment i
	tlength = 0.;
	tlengthq = 0.;//added 8/09
	tlengthqhd = 0.;//added 8/10
	for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
		rseg[iseg] = diam[iseg]/2.0;
		qq[iseg] = fabs(q[iseg]);
		for(k=1; k<=3; k++){
			start[k][iseg] = cnode[k][ista[iseg]];
			end[k][iseg] = cnode[k][iend[iseg]];
			ss[k] = end[k][iseg] - start[k][iseg];
		}
		lseg[iseg] = sqrt(SQR(ss[1]) + SQR(ss[2]) + SQR(ss[3]));
		if(lseg[iseg] == 0.) printf("*** Error: segment %i has zero length\n",iseg);//added May 2010
		for(j=1; j<=3; j++)	scos[j][iseg] = ss[j]/lseg[iseg];
		tlength += lseg[iseg];
		tlengthq += lseg[iseg]*qq[iseg];//added 8/09
		tlengthqhd += lseg[iseg]*qq[iseg]*hd[iseg];//added 8/10
//find points on vessel cylinders (used for tissue region bounds)
		for(k=0; k<=15; k++){
			t = k*2*pi1/16.0;
			sintheta = sqrt(1.0 - SQR(scos[3][iseg]));
			if(sintheta > 0.0001){
				cosfi = -scos[2][iseg]/sintheta;
				sinfi = scos[1][iseg]/sintheta;
			}
			else {
				cosfi = 1.;
				sinfi = 0.;
			}
			rsta[1][k+1][iseg] = rseg[iseg]*cosfi*cos(t)-rseg[iseg]*scos[3][iseg]*sinfi*sin(t) + start[1][iseg];
			rsta[2][k+1][iseg] = rseg[iseg]*sinfi*cos(t)+rseg[iseg]*scos[3][iseg]*cosfi*sin(t) + start[2][iseg];
			rsta[3][k+1][iseg] = rseg[iseg]*sintheta*sin(t) + start[3][iseg];
			rend[1][k+1][iseg] = rseg[iseg]*cosfi*cos(t)-rseg[iseg]*scos[3][iseg]*sinfi*sin(t) + end[1][iseg];
			rend[2][k+1][iseg] = rseg[iseg]*sinfi*cos(t)+rseg[iseg]*scos[3][iseg]*cosfi*sin(t) + end[2][iseg];
			rend[3][k+1][iseg] = rseg[iseg]*sintheta*sin(t) + end[3][iseg];
		}
	}
//subdivide segments into small elements as needed
//compute coordinates of source points
//nnv = total number of elements
	nnv = 0;
	for(iseg=1; iseg<=nseg; iseg++) if(segtyp[iseg] == 4 || segtyp[iseg] == 5){
		m = lseg[iseg]/maxl + 1;
		nspoint[iseg] = m;
		istart[iseg] = nnv + 1;
		ds[iseg] = lseg[iseg]/m;
		nnv += m;
	}
	printf("Total vessel points = %i\n", nnv);
//compute coordinates of tissue points
	delx = alx/mxx;
	for(i=1; i<=mxx; i++) axt[i] = (i-0.5f)*delx;
	dely = aly/myy;
	for(i=1; i<=myy; i++) ayt[i] = (i-0.5f)*dely;
	delz = alz/mzz;
	for(i=1; i<=mzz; i++) azt[i] = (i-0.5f)*delz;
//create array of tissue points inside tissue domain boundaries using method 1 or 2
	nnt = outboun(2);
}
