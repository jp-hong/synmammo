

#ifndef _USER
#define _USER

//geometry
#define SOURCE_TO_DETECTOR 627.0f
#define	AXIS_TO_DETECTOR 33.0f
#define	MARGIN 1.5f

//input image volume
#define NX 300
#define	NY 2592
#define	NZ 1440

#define XLEN 60.0f
#define	YLEN 259.0f
#define	ZLEN 144.0f

//detector
#define NU 2592
#define NV 1440
#define DU 0.1122684f
#define DV 0.1122684f

//#define NU 3888
//#define NV 3072
//#define DU 0.0748456f
//#define DV 0.0748456f

//bandpass filter parameters (P : projection, MIP : maximum intensity projection)
#define FILTER_SMALL_DIAMETER 3.0f			//filter small structures up to this size
#define FILTER_LARGE_DIAMETER 20.0f			//filter large structures down to this size
#define SUPPRESS_STRIPES 0					//0 : no suppression, 1 : horizontal, 2 : vertical
#define TOLERANCE_DIAMETER 0				//number in percentage (%)

//file path & naming parameters
#define INPUT_FILE_DIRECTORY "D:\\Work\\DBTRecon\\Data\\0330_01\\20150330_105452_DBT_v1.0.038\\"
#define OUTPUT_FILE_DIRECTORY "D:\\Work\\DBTRecon\\Data\\0330_01\\20150330_105452_DBT_v1.0.038\\"
#define DATA_FILE_NAME "new_image.post_recon_data"
#define OUTPUT_FILE_NAME "300_"
#define OUTPUT_FILE_EXTENSION ".dat"

#define SAVE_PROJECTIONS false
#define SAVE_FILTERED_IMAGES false


#endif