

#ifndef _GLOBALS
#define _GLOBALS

#include "user.h"

#define	X0 -XLEN / 2.0f
#define	Y0 -YLEN / 2.0f
#define	Z0 -MARGIN

#define	DX XLEN / NX
#define	DY YLEN / NY
#define	DZ ZLEN / NZ

#define DX_EQ XLEN / NX_EQ
#define	DY_EQ YLEN / NY_EQ
#define	DZ_EQ ZLEN / NZ_EQ

#define	ULEN NU * DU
#define	VLEN NV * DV
#define	U0 -ULEN / 2.0f
#define	V0 -MARGIN

#define IMAGE_LEN NX * NY * NZ
#define DET_LEN NU * NV
#define FRAME_LEN 12
#define IMAGE_BYTES IMAGE_LEN * sizeof(short)
#define DET_BYTES DET_LEN * sizeof(float)
#define FRAME_BYTES FRAME_LEN * sizeof(float)

#endif